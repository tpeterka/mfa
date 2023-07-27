//--------------------------------------------------------------
// example of encoding / decoding multidimensional data sets 
// sample from an analytical formula
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
#include <set>

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
    int         geom_degree     = 1;        // degree for geometry (same for all dims)
    int         vars_degree     = 4;        // degree for science variables (same for all dims)
    int         ndomp           = 100;      // input number of domain points (same for all dims)
    int         ntest           = 0;        // number of input test points in each dim for analytical error tests
    int         geom_nctrl      = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl      = {11};     // initial # control points for all science variables (default same for all dims)
    string      input           = "sinc";   // input dataset
    real_t      rot             = 0.0;      // rotation angle in degrees
    real_t      twist           = 0.0;      // twist (waviness) of domain (0.0-1.0)
    real_t      noise           = 0.0;      // fraction of noise
    int         structured      = 1;        // input data format (bool 0/1)
    int         rand_seed       = -1;       // seed to use for random data generation (-1 == no randomization)
    real_t      regularization  = 0;        // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int         reg1and2        = 0;        // flag for regularizer: 0 = regularize only 2nd derivs. 1 = regularize 1st and 2nd
    int         verbose         = 1;        // MFA verbosity (0 = no extra output)
    int         strong_sc       = 1;        // strong scaling (bool 0 or 1, 0 = weak scaling)
    int         weighted        = 0;        // Use NURBS weights (0/1)
    real_t      ghost           = 0.1;      // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    int         tot_blocks      = 1;        // 
    int         adaptive        = 0;        // do analytical encode (0/1)
    real_t      e_threshold     = 1e-1;     // error threshold for adaptive fitting
    int         rounds          = 0;
    bool        help            = false;    // show help

    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('l', "scalar",      scalar,     " flag for scalar or vector-valued science variables");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('a', "ntest",       ntest,      " number of test points in each dimension of domain (for analytical error calculation)");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('r', "rotate",      rot,        " rotation angle of domain in degrees");
    ops >> opts::Option('t', "twist",       twist,      " twist (waviness) of domain (0.0-1.0)");
    ops >> opts::Option('s', "noise",       noise,      " fraction of noise (0.0 - 1.0)");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");
    ops >> opts::Option('o', "overlap",     ghost,      " relative ghost zone overlap (0.0 - 1.0)");
    ops >> opts::Option('z', "strong_sc",   strong_sc,  " strong scaling (1 = strong, 0 = weak)");
    ops >> opts::Option('z', "tot_blocks",  tot_blocks, " total number of blocks");
    ops >> opts::Option('z', "adaptive",    adaptive,   " do adaptive encode (0/1)");
    ops >> opts::Option('e', "errorbound",  e_threshold," error threshold for adaptive encoding");
    ops >> opts::Option('z', "rounds",      rounds,     " max number of rounds for adaptive encoding");
    ops >> opts::Option('z', "verbose",     verbose,    " output verbosity (0/1)");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // print input arguments
    echo_mfa_settings("analytical example", dom_dim, pt_dim, scalar, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        regularization, reg1and2, adaptive, e_threshold, rounds);
    echo_data_settings(input, "", ndomp, ntest);
    echo_data_mod_settings(structured, rand_seed, rot, twist, noise);

    // initialize DIY
    diy::FileStorage          storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &Block<real_t>::create,
                                     &Block<real_t>::destroy,
                                     &storage,
                                     &Block<real_t>::save,
                                     &Block<real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

    // set global domain bounds and decompose
    Bounds<real_t> dom_bounds(dom_dim);
    set_dom_bounds(dom_bounds, input);
    
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
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
    MFAInfo     mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                input, "", ndomp, structured, rand_seed, rot, twist, noise,
                reg1and2, regularization, adaptive, verbose, mfa_info, d_args);

    // Create data set for modeling. Input keywords are defined in example_signals.hpp
    if (analytical_signals.count(input) == 1)
    {
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->generate_analytical_data(cp, input, mfa_info, d_args); 
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
    }

    // compute the MFA
    fprintf(stderr, "\nStarting encoding...\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
    { 
        if (!adaptive)
            b->fixed_encode_block(cp, mfa_info);
        else
            b->adaptive_encode_block(cp, e_threshold, rounds, mfa_info);
    });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "Encoding done.\n");

    // decode at custom grid and compute ground-truth errors
    double decode_time = MPI_Wtime();
    if (ntest > 0)
    {
        cerr << "Computing analytical error" << endl;
        int nvars = model_dims.size() - 1;
        vector<real_t> L1(nvars), L2(nvars), Linf(nvars);
        vector<int> grid_size(dom_dim, ntest);
        mfa::PointSet<real_t>* temp_in = nullptr;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->analytical_error_field(cp, grid_size, input, L1, L2, Linf, d_args, temp_in, b->approx, b->errs); });

        // print analytical errors
        for (int i = 0; i < nvars; i++)
        {
            fprintf(stderr, "\n------ Analytical error: Var %i -------\n", i);
            fprintf(stderr, "L-1        norm = %e\n", L1[i]);
            fprintf(stderr, "L-2        norm = %e\n", L2[i]);
            fprintf(stderr, "L-infinity norm = %e\n", Linf[i]);
            fprintf(stderr, "-------------------------------------\n\n");
        }
    }
    // decode at original point locations and compute pointwise error
    else
    {
        fprintf(stderr, "\nFinal decoding and computing max. error...\n");
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->range_error(cp, true, false);
        });
    }
    decode_time = MPI_Wtime() - decode_time;

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, true); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.mfa", world, master);
}
