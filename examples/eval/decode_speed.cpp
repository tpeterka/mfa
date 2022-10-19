//--------------------------------------------------------------
// Speed test for decoding points in different ways
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <typeinfo>

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include "opts.h"

#include "block.hpp"

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    // default command line arguments
    int     num_iters   = 10000;                // number of random points to decode
    int     gradval     = 0;                    // flag to decode gradient and value simultaneously in FastGrad()
    string  infile      = "approx.mfa";         // diy input file
    bool    help;                               // show help

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('n', "num_iters",   num_iters,  " number of random points to decode");
    ops >> opts::Option('g', "gradval",     gradval,    " flag to decode gradient and value simultaneously in FastGrad()");
    ops >> opts::Option('f', "infile",      infile,     " diy input file name");
    ops >> opts::Option('h', "help",        help,       " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX");               // used for blocks to be moved out of core
    diy::Master      master(world,
            -1,
            -1,
            &Block<real_t>::create,
            &Block<real_t>::destroy,
            &storage,
            &Block<real_t>::save,
            &Block<real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), -1);   // number of blocks set by read_blocks()

    // read MFA model
    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block<real_t>::load);
    int nblocks = master.size();
    std::cout << nblocks << " blocks read from file "<< infile << "\n";

    // Get a single block
    Block<real_t>* b    = static_cast<Block<real_t>*>(master.block(0));
    int dom_dim = b->dom_dim;  // dimensionality of input domain
    int pt_dim  = b->pt_dim;   // dimensionality of output point

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr << "real_type: " << typeid(b->bounds_mins[0]).name() << endl;
    cerr << "num_iters = " << num_iters << endl;
    cerr << "dom_dim: " << dom_dim << endl;
    cerr << "pt_dim: " << pt_dim << endl;
    cerr << "degree: " << b->mfa->var(0).p(0) << endl;
    cerr << "Decode gradient & value together: " << ((gradval == 1) ? "yes" : "no") << endl;
#ifdef MFA_TBB
    cerr << "threading: TBB" << endl;
#endif
#ifdef MFA_KOKKOS
    cerr << "threading: Kokkos" << endl;
#endif
#ifdef MFA_SYCL
    cerr << "threading: SYCL" << endl;
#endif
#ifdef MFA_SERIAL
    cerr << "threading: serial" << endl;
#endif
    fprintf(stderr, "-------------------------------------\n\n");

    // Set up data structures for test
    vector<real_t> value_sums;
    vector<real_t> deriv_sums;

    // Use C++ random number generation so we can fix the random seed
    unsigned seed = 5;     
    std::mt19937 gen(seed);
    std::uniform_real_distribution<real_t> dis(0.0, 1.0);
    MatrixX<real_t> params = MatrixX<real_t>::NullaryExpr(num_iters, dom_dim, [&](){return dis(gen);});


    // MatrixX<real_t> params = MatrixX<real_t>::Random(num_iters, dom_dim);
    params = params + MatrixX<real_t>::Ones(num_iters, dom_dim);
    params *= 0.5; // scale random numbers to [0,1]

    real_t          out_val = 0;
    VectorX<real_t> out_pt_full(pt_dim);        // out_pt (science + geom)
    VectorX<real_t> out_pt(pt_dim-dom_dim);     // out_pt (science variable only)
    VectorX<real_t> out_grad(dom_dim);
    MatrixX<real_t> outpts_full = MatrixX<real_t>::Zero(num_iters, pt_dim);
    MatrixX<real_t> outpts = MatrixX<real_t>::Zero(num_iters, pt_dim-dom_dim);
    MatrixX<real_t> outgrads = MatrixX<real_t>::Zero(num_iters, dom_dim);


    // Naive implementation of decoding points
    // Decodes all mfa models (geometry + vars)
    cerr << "Starting original decode..." << endl;
    double decode_time_og = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { 
                for (int l = 0; l < num_iters; l++)
                {
                    b->decode_point(cp, params.row(l), out_pt_full); 
                    outpts_full.row(l) = out_pt_full;  
                } 
            });
    decode_time_og = MPI_Wtime() - decode_time_og;
    cerr << "   done." << endl;
    value_sums.push_back(outpts_full.rightCols(pt_dim-dom_dim).sum());


    // Define one decoder ahead of time
    // Decodes on science variable mfa models
    cerr << "Starting decoder-level decode..." << endl;     // construct decoder only once & use decode_info
    VectorXi no_derivs;
    const mfa::MFA_Data<real_t>&  mfa_data = b->mfa->var(0);
    const TensorProduct<real_t>&  t = mfa_data.tmesh.tensor_prods[0];
    mfa::Decoder<real_t>    decoder(mfa_data, 0);
    mfa::DecodeInfo<real_t> decode_info(mfa_data, no_derivs);

    double decode_time_dl = MPI_Wtime();
    for (int l = 0; l < num_iters; l++)
    {
        decoder.VolPt(params.row(l), out_pt, decode_info, t);
        outpts.row(l) = out_pt;
    }
    decode_time_dl = MPI_Wtime() - decode_time_dl;
    cerr << "   done." << endl;
    value_sums.push_back(outpts.sum());


    // Use FastVolPt instead of VolPt
    // Decodes on science variable mfa models
    cerr << "Starting FastVolPt decode..." << endl;     // construct decoder only once & use decode_info
    mfa::FastDecodeInfo<real_t> fast_decode_info(decoder);
    double decode_time_fv = MPI_Wtime();
    for (int l = 0; l < num_iters; l++)
    {
        decoder.FastVolPt(params.row(l), out_pt, fast_decode_info, t);
        outpts.row(l) = out_pt;
    }
    decode_time_fv = MPI_Wtime() - decode_time_fv;
    cerr << "   done." << endl;
    value_sums.push_back(outpts.sum());


    // Use differentiate_point() 3 times to compute gradient
    cerr << "Starting DifferentiatePoint decode..." << endl;     // construct decoder only once & use decode_info
    double decode_time_diffpoint = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { 
                for (int l = 0; l < num_iters; l++)
                {
                    for (int i = 0; i < dom_dim; i++)
                    {
                        b->differentiate_point(cp, params.row(l), 1, i, -1, out_pt_full); 
                        outgrads(l,i) = out_pt_full(dom_dim);
                    }
                }
            });
    decode_time_diffpoint = MPI_Wtime() - decode_time_diffpoint;
    cerr << "   done." << endl;
    deriv_sums.push_back(outgrads.sum());


    // Use FastGrad instead of differentiate_point()
    // Decodes on science variable mfa models
    cerr << "Starting FastGrad decode..." << endl;     // construct decoder only once & use decode_info
    fast_decode_info.ResizeDers(1);
    real_t* valueptr = nullptr;
    if (gradval == 1) valueptr = &out_val;
    double decode_time_fastgrad = MPI_Wtime();
    for (int l = 0; l < num_iters; l++)
    {
        decoder.FastGrad(params.row(l), fast_decode_info, t, out_grad, valueptr);
        outgrads.row(l) = out_grad;
        outpts(l, 0) = (valueptr == nullptr ? 0 : *valueptr);
    }
    decode_time_fastgrad = MPI_Wtime() - decode_time_fastgrad;
    cerr << "   done." << endl;
    value_sums.push_back(outpts.sum());
    deriv_sums.push_back(outgrads.sum());

    // Report sum of gradient components for each decoded point to check for consistency
    cout << "\nSanity Check! The following sums should match" << endl;
    cout << "  Value Sums: " << endl;
    cout << "    DecodePoint: " << value_sums[0] << endl;
    cout << "    VolPt:       " << value_sums[1] << endl;
    cout << "    FastVolPt:   " << value_sums[2] << endl;
    if (gradval) cout << "    FastGrad:    " << value_sums[3] << endl;
    cout << endl;

    cout << "  Deriv Sums: " << endl;
    cout << "    DifferentiatePoint: " << deriv_sums[0] << endl;
    cout << "    FastGrad:           " << deriv_sums[1] << endl;

    // Print results
    cout << "\n\nOriginal Decode:" << endl;
    cout << "---------------------" << endl;
    cout << "Total iterations: " << num_iters << endl;
    cout << "Total decode time: " << decode_time_og << "s" << endl;
    cout << "Time per iter: " << decode_time_og / num_iters * 1000000 << "us" << endl;

    cout << "\n\nDecoder-Level Decode:" << endl;
    cout << "---------------------" << endl;
    cout << "Total iterations: " << num_iters << endl;
    cout << "Total decode time: " << decode_time_dl << "s" << endl;
    cout << "Time per iter: " << decode_time_dl / num_iters * 1000000 << "us" << endl;

    cout << "\n\nFastVolPt Decode:" << endl;
    cout << "---------------------" << endl;
    cout << "Total iterations: " << num_iters << endl;
    cout << "Total decode time: " << decode_time_fv << "s" << endl;
    cout << "Time per iter: " << decode_time_fv / num_iters * 1000000 << "us" << endl;

    cout << "\n\nDifferentiatePoint Decode:" << endl;
    cout << "---------------------" << endl;
    cout << "Total iterations: " << num_iters << endl;
    cout << "Total decode time: " << decode_time_diffpoint << "s" << endl;
    cout << "Time per iter: " << decode_time_diffpoint / num_iters * 1000000 << "us" << endl;

    cout << "\n\nFastGrad Decode:" << endl;
    cout << "---------------------" << endl;
    cout << "Total iterations: " << num_iters << endl;
    cout << "Total decode time: " << decode_time_fastgrad << "s" << endl;
    cout << "Time per iter: " << decode_time_fastgrad / num_iters * 1000000 << "us" << endl;


    cout << "\n\n==========================" << endl;
    cout << "Approx Speedup (Decode): " << setprecision(3) << (decode_time_og/decode_time_fv) << "x" << endl;
    cout << "============================" << endl;

    if (gradval)    // compare times to compute both value and gradient
    {
        cout << "\n\n==========================" << endl;
        cout << "Approx Speedup (Value+Grad): " << setprecision(3) << ((decode_time_og+decode_time_diffpoint)/decode_time_fastgrad) << "x" << endl;
        cout << "============================" << endl;
    }
    else            // compare times to compute gradient       
    {
        cout << "\n\n==========================" << endl;
        cout << "Approx Speedup (Derivs): " << setprecision(3) << (decode_time_diffpoint/decode_time_fastgrad) << "x" << endl;
        cout << "============================" << endl;
    }
}
