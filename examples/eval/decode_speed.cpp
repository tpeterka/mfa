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
    string  infile      = "approx.out";         // diy input file
    bool    help;                               // show help

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('n', "num_iters", num_iters, "number of random points to decode");
    ops >> opts::Option('f', "infile",  infile,  " diy input file name");
    ops >> opts::Option('h', "help",    help,    " show help");

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
    cerr << "degree: " << b->vars[0].mfa_data->p(0) << endl;
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
    vector<real_t> sums;

    MatrixX<real_t> params = MatrixX<real_t>::Random(num_iters, dom_dim);
    params = params + MatrixX<real_t>::Ones(num_iters, dom_dim);
    params *= 0.5; // scale random numbers to [0,1]

    VectorX<real_t> out_pt_full(pt_dim);        // out_pt (science + geom)
    MatrixX<real_t> outpts_full = MatrixX<real_t>::Zero(num_iters, pt_dim);
    VectorX<real_t> out_pt(pt_dim-dom_dim);     // out_pt (science variable only)
    MatrixX<real_t> outpts = MatrixX<real_t>::Zero(num_iters, pt_dim-dom_dim);


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
    sums.push_back(outpts_full.sum());


    // Define one decoder ahead of time
    // Decodes on science variable mfa models
    cerr << "Starting decoder-level decode..." << endl;     // construct decoder only once & use decode_info
    VectorXi no_derivs;
    mfa::MFA_Data<real_t>&  mfa_data = *(b->vars[0].mfa_data);
    TensorProduct<real_t>&  t = mfa_data.tmesh.tensor_prods[0];
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
    sums.push_back(outpts.sum());


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
    sums.push_back(outpts.sum());


    // hack to force compiler to carry out all decodes
    // also a good sanity check that all decode methods are equivalent
    for (auto ii = 0; ii < sums.size(); ii++)   
        cout << sums[ii] << " ";
    cout << endl;

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


    cout << "\n\n==========================" << endl;
    cout << "Approx Speedup: " << setprecision(3) << (decode_time_og/decode_time_fv) << "x" << endl;
    cout << "============================" << endl;
}
