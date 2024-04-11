//--------------------------------------------------------------
// Speed test for encoding MFAs with various methods
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
#include "example-setup.hpp"

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    // default command line arguments
    int         pt_dim          = 3;        // dimension of input points
    int         dom_dim         = 2;        // dimension of domain (<= pt_dim)
    int         scalar          = 1;        // flag for scalar or vector-valued science variables (0 == multiple scalar vars)
    int         vars_degree     = 4;        // degree for science variables (same for all dims)
    int         ndomp           = 100;      // input number of domain points (same for all dims)
    vector<int> vars_nctrl      = {11};     // initial # control points for all science variables (default same for all dims)
    string      input           = "sinc";   // input dataset
    int         rand_seed       = -1;       // seed to use for random data generation (-1 == no randomization)
    real_t      regularization  = 0;        // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int         reg1and2        = 0;        // flag for regularizer: 0 = regularize only 2nd derivs. 1 = regularize 1st and 2nd
    int         verbose         = 0;        // MFA verbosity (0 = no extra output)
    bool        help            = false;    // show help

    // test-specific variables
    int         structured      = 1;
    int         adaptive        = 0;
    int         rounds          = 0;
    int         e_threshold     = 1e-1;

    // constant parameters
    const int       seed        = 5;
    const int       geom_degree = 1;
    const int       geom_nctrl  = -1;
    const real_t    rot         = 0.0;      // rotation angle in degrees
    const real_t    twist       = 0.0;      // twist (waviness) of domain (0.0-1.0)
    const real_t    noise       = 0.0;      // fraction of noise
    const int       weighted    = 1;
    const int       tot_blocks  = 1;

    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('l', "scalar",      scalar,     " flag for scalar or vector-valued science variables");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");
    ops >> opts::Option('e', "errorbound",  e_threshold," error threshold for adaptive encoding");
    ops >> opts::Option('z', "verbose",     verbose,    " output verbosity (0/1)");
    ops >> opts::Option('h', "help",        help,       " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // Domain bounds
    Bounds<real_t> dom_bounds(dom_dim);
    set_dom_bounds(dom_bounds, input);

    // initialize DIY for in-core use only
    diy::Master             master(world, 1, -1, &Block<real_t>::create, &Block<real_t>::destroy);
    diy::ContiguousAssigner assigner(world.size(), tot_blocks);
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);

    // Perform DIY decomposition
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    // Different examples can reset this below
    vector<int> model_dims = {dom_dim, pt_dim - dom_dim};
    if (scalar) // Set up (pt_dim - dom_dim) separate scalar variables
    {
        model_dims.assign(pt_dim - dom_dim + 1, 1);
        model_dims[0] = dom_dim;                        // index 0 == geometry
    }

    // set up parameters for examples
    mfa::MFAInfo mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                input, "", ndomp, structured, rand_seed, rot, twist, noise,
                reg1and2, regularization, adaptive, verbose, mfa_info, d_args);

    // Get a single block
    Block<real_t>* b    = static_cast<Block<real_t>*>(master.block(0));
    b->verbose = verbose;
    auto cp = master.proxy(0); 

    fmt::print("Starting Encode Tests\n");

    //----------------------------------------------------  
    // Test 1:
    // Structured data
    // Fixed encode
    fmt::print("Test 1:\n");
    fmt::print("  * Generating data...\n");
    d_args.structured = true;
    b->generate_analytical_data(cp, input, mfa_info, d_args);

    fmt::print("  * Encoding...\n");
    double etime_1 = MPI_Wtime();
    b->fixed_encode_block(cp, mfa_info);
    etime_1 = MPI_Wtime() - etime_1;
    fmt::print("Test 1 Done.\n");

    delete b->mfa;      b->mfa = nullptr;
    delete b->input;    b->input = nullptr;
    delete b->approx;   b->approx = nullptr;
    delete b->errs;     b->errs = nullptr;
    //----------------------------------------------------  


    //----------------------------------------------------  
    // Test 2:
    // Unstructured data
    // Fixed encode
    fmt::print("Test 2:\n");
    fmt::print("  * Generating data...\n");
    d_args.structured = false;
    d_args.rand_seed = seed;
    b->generate_analytical_data(cp, input, mfa_info, d_args);

    fmt::print("  * Encoding...\n");
    double etime_2 = MPI_Wtime();
    b->fixed_encode_block(cp, mfa_info);
    etime_2 = MPI_Wtime() - etime_2;
    fmt::print("Test 2 Done.\n");
    delete b->mfa;      b->mfa = nullptr;
    delete b->input;    b->input = nullptr;
    delete b->approx;   b->approx = nullptr;
    delete b->errs;     b->errs = nullptr;
    //----------------------------------------------------  


    //----------------------------------------------------  
    // Test 3:
    // Structured data
    // Adaptive encode
    // Error threshold 1e-1
    fmt::print("Test 3:\n");
    fmt::print("  * Generating data...\n");
    e_threshold = 1e-1;
    d_args.structured = true;
    d_args.rand_seed = -1;
    for (int i = 0; i < mfa_info.nvars(); i++) // reset nctrl to a small number for adaptive
    {
        mfa_info.var_model_infos[i].nctrl_pts = VectorXi::Constant(dom_dim, 10); 
    }
    b->generate_analytical_data(cp, input, mfa_info, d_args);

    fmt::print("  * Encoding...\n");
    double etime_3 = MPI_Wtime();
    b->adaptive_encode_block(cp, e_threshold, rounds, mfa_info);
    etime_3 = MPI_Wtime() - etime_3;
    fmt::print("Test 3 Done.\n");
    delete b->mfa;      b->mfa = nullptr;
    delete b->input;    b->input = nullptr;
    delete b->approx;   b->approx = nullptr;
    delete b->errs;     b->errs = nullptr;
    //---------------------------------------------------- 

    //----------------------------------------------------  
    // Test 4:
    // Structured data
    // Adaptive encode
    // Error threshold 1e-3
    fmt::print("Test 4:\n");
    fmt::print("  * Generating data...\n");
    e_threshold = 1e-3;
    d_args.structured = true;
    d_args.rand_seed = -1;
    for (int i = 0; i < mfa_info.nvars(); i++) // reset nctrl to a small number for adaptive
    {
        mfa_info.var_model_infos[i].nctrl_pts = VectorXi::Constant(dom_dim, 10); 
    }
    b->generate_analytical_data(cp, input, mfa_info, d_args);

    fmt::print("  * Encoding...\n");
    double etime_4 = MPI_Wtime();
    b->adaptive_encode_block(cp, e_threshold, rounds, mfa_info);
    etime_4 = MPI_Wtime() - etime_4;
    fmt::print("Test 4 Done.\n");
    delete b->mfa;      b->mfa = nullptr;
    delete b->input;    b->input = nullptr;
    delete b->approx;   b->approx = nullptr;
    delete b->errs;     b->errs = nullptr;
    //---------------------------------------------------- 

    fmt::print("Test 1 Timing: {:.2g}s\n", etime_1);
    fmt::print("Test 2 Timing: {:.2g}s\n", etime_2);
    fmt::print("Test 3 Timing: {:.2g}s\n", etime_3);
    fmt::print("Test 4 Timing: {:.2g}s\n", etime_4);


    // // Report sum of gradient components for each decoded point to check for consistency
    // cout << "\nSanity Check! The following sums should match" << endl;
    // cout << "  Value Sums: " << endl;
    // cout << "    DecodePoint: " << value_sums[0] << endl;
    // cout << "    VolPt:       " << value_sums[1] << endl;
    // cout << "    FastVolPt:   " << value_sums[2] << endl;
    // if (gradval) cout << "    FastGrad:    " << value_sums[3] << endl;
    // cout << endl;

    // cout << "  Deriv Sums: " << endl;
    // cout << "    DifferentiatePoint: " << deriv_sums[0] << endl;
    // cout << "    FastGrad:           " << deriv_sums[1] << endl;

    // // Print results
    // cout << "\n\nOriginal Decode:" << endl;
    // cout << "---------------------" << endl;
    // cout << "Total iterations: " << num_iters << endl;
    // cout << "Total decode time: " << decode_time_og << "s" << endl;
    // cout << "Time per iter: " << decode_time_og / num_iters * 1000000 << "us" << endl;

    // cout << "\n\nDecoder-Level Decode:" << endl;
    // cout << "---------------------" << endl;
    // cout << "Total iterations: " << num_iters << endl;
    // cout << "Total decode time: " << decode_time_dl << "s" << endl;
    // cout << "Time per iter: " << decode_time_dl / num_iters * 1000000 << "us" << endl;

    // cout << "\n\nFastVolPt Decode:" << endl;
    // cout << "---------------------" << endl;
    // cout << "Total iterations: " << num_iters << endl;
    // cout << "Total decode time: " << decode_time_fv << "s" << endl;
    // cout << "Time per iter: " << decode_time_fv / num_iters * 1000000 << "us" << endl;

    // cout << "\n\nDifferentiatePoint Decode:" << endl;
    // cout << "---------------------" << endl;
    // cout << "Total iterations: " << num_iters << endl;
    // cout << "Total decode time: " << decode_time_diffpoint << "s" << endl;
    // cout << "Time per iter: " << decode_time_diffpoint / num_iters * 1000000 << "us" << endl;

    // cout << "\n\nFastGrad Decode:" << endl;
    // cout << "---------------------" << endl;
    // cout << "Total iterations: " << num_iters << endl;
    // cout << "Total decode time: " << decode_time_fastgrad << "s" << endl;
    // cout << "Time per iter: " << decode_time_fastgrad / num_iters * 1000000 << "us" << endl;


    // cout << "\n\n==========================" << endl;
    // cout << "Approx Speedup (Decode): " << setprecision(3) << (decode_time_og/decode_time_fv) << "x" << endl;
    // cout << "============================" << endl;

    // if (gradval)    // compare times to compute both value and gradient
    // {
    //     cout << "\n\n==========================" << endl;
    //     cout << "Approx Speedup (Value+Grad): " << setprecision(3) << ((decode_time_og+decode_time_diffpoint)/decode_time_fastgrad) << "x" << endl;
    //     cout << "============================" << endl;
    // }
    // else            // compare times to compute gradient       
    // {
    //     cout << "\n\n==========================" << endl;
    //     cout << "Approx Speedup (Derivs): " << setprecision(3) << (decode_time_diffpoint/decode_time_fastgrad) << "x" << endl;
    //     cout << "============================" << endl;
    // }
}
