//--------------------------------------------------------------
// example of computing line integrals from an encoded MFA
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

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include "opts.h"
#include "block.hpp"
#include "rayblock.hpp"
#include "example-setup.hpp"


using namespace std;



int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                        // number of local blocks
    int tot_blocks  = nblocks * world.size();   // number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int    pt_dim       = 3;                    // dimension of input points
    int    dom_dim      = 2;                    // dimension of domain (<= pt_dim)
    int    geom_degree  = 1;                    // degree for geometry (same for all dims)
    int    vars_degree  = 4;                    // degree for science variables (same for all dims)
    int    ndomp        = 100;                  // input number of domain points (same for all dims)
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    vector<int>    vars_nctrl   = {11};                   // input number of control points for all science variables (same for all dims)
    string input        = "sinc";               // input dataset
    int    weighted     = 0;                    // solve for and use weights (bool 0/1)
    string infile;                              // input file name
    int    structured   = 1;                    // input data format (bool 0/1)
    int    rand_seed    = -1;                   // seed to use for random data generation (-1 == no randomization)
    float  regularization = 0;                  // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int    reg1and2     = 0;                       // flag for regularizer: 0 --> regularize only 2nd derivs. 1 --> regularize 1st and 2nd
    bool   help         = false;                // show help

    const int verbose = 1;
    const int scalar = 1;

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");

    int n_alpha = 120;
    int n_rho = 120;
    int n_samples = 120;
    int v_alpha = 100;
    int v_rho = 100;
    int v_samples = 100;
    int num_ints = 10000;
    ops >> opts::Option('z', "n_alpha", n_alpha, " number of rotational samples for line integration");
    ops >> opts::Option('z', "n_rho", n_rho, " number of samples in offset direction for line integration");
    ops >> opts::Option('z', "n_samples", n_samples, " number of samples along ray for line integration");
    ops >> opts::Option('z', "v_alpha", v_alpha, " number of rotational control points for line integration");
    ops >> opts::Option('z', "v_rho", v_rho, " number of control points in offset direction for line integration");
    ops >> opts::Option('z', "v_samples", v_samples, " number of control points along ray for line integration");
    ops >> opts::Option('z', "num_ints", num_ints, " number of random line integrals to compute");


    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // print input arguments
    echo_mfa_settings("line int example", pt_dim, dom_dim, 1, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        regularization, reg1and2, false, 0, 0);
    echo_data_settings(input, infile, ndomp, 0);

    // initialize DIY
    diy::FileStorage          storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &RayBlock<real_t>::create,
                                     &RayBlock<real_t>::destroy,
                                     &storage,
                                     &RayBlock<real_t>::save,
                                     &RayBlock<real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

    // set global domain bounds and decompose
    Bounds<real_t> dom_bounds(dom_dim);
    set_dom_bounds(dom_bounds, input);

    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { RayBlock<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    // Different examples can reset this below
    vector<int> model_dims = {dom_dim, pt_dim - dom_dim};
    if (scalar) // Set up (pt_dim - dom_dim) separate scalar variables
    {
        model_dims.assign(pt_dim - dom_dim + 1, 1);
        model_dims[0] = dom_dim;                        // index 0 == geometry
    }

    // Create empty info classes
    mfa::MFAInfo    mfa_info(dom_dim, verbose);
    DomainArgs      d_args(dom_dim, model_dims);

    // set up parameters for examples
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                input, infile, ndomp, structured, rand_seed, 0, 0, 0,
                reg1and2, regularization, false, verbose, mfa_info, d_args);

    // Create data set for modeling. Input keywords are defined in example-setup.hpp
    if (analytical_signals.count(input) == 1)
    {
        master.foreach([&](RayBlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
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
    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](RayBlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, mfa_info); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    double decode_time = MPI_Wtime();
    master.foreach([&](RayBlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
    { 
        // Compute errors for original MFA
        // b->range_error(cp, true, false);
        // b->print_block(cp, true);

        b->create_ray_model(cp, mfa_info, d_args, n_samples, n_rho, n_alpha, v_samples, v_rho, v_alpha);
        b->compute_random_ints(cp, d_args, num_ints);
        b->compute_sinogram(cp);
    });
    decode_time = MPI_Wtime() - decode_time;

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](RayBlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_ray_model(cp); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.mfa", world, master);
}
