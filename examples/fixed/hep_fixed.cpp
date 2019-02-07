//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points and a
// single block in a split model w/ one model containing geometry and other model science variables
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
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

#include "block.hpp"
#include "opts.h"

using namespace std;

typedef  diy::RegularDecomposer<Bounds> Decomposer;

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
    int    ndomp        = 25;                   // input number of domain points (same for all dims)
    int    ntest        = 0;                    // number of input test points in each dim for analytical error tests
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    int    vars_nctrl   = 11;                   // input number of control points for all science variables (same for all dims)
    string input        = "f16";                // input dataset
    bool   weighted     = true;                 // solve for and use weights
    real_t noise        = 0.0;                  // fraction of noise

    // get command line arguments
    opts::Options ops(argc, argv);
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('a', "ntest",       ntest,      " number of test points in each dimension of domain (for analytical error calculation)");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('n', "noise",       noise,      " fraction of noise (0.0 - 1.0)");

    if (ops >> opts::Present('h', "help", " show help"))
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // minimal number of geometry control points if not specified
    if (geom_nctrl == -1)
        geom_nctrl = geom_degree + 1;

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "pt_dim = "         << pt_dim       << " dom_dim = "        << dom_dim      <<
        "\ngeom_degree = "  << geom_degree  << " vars_degree = "    << vars_degree  <<
        "\ninput pts = "    << ndomp        << " geom_ctrl pts = "  << geom_nctrl   <<
        "\nvars_ctrl_pts = "<< vars_nctrl   << " test_points = "    << ntest        <<
        "\ninput = "        << input        <<  endl;
#ifdef CURVE_PARAMS
    cerr << "parameterization method = curve" << endl;
#else
    cerr << "parameterization method = domain" << endl;
#endif
#ifdef MFA_NO_TBB
    cerr << "TBB: off" << endl;
#else
    cerr << "TBB: on" << endl;
#endif
#ifdef MFA_NO_WEIGHTS
    cerr << "weighted = 0" << endl;
#else
    cerr << "weighted = " << weighted << endl;
#endif
    fprintf(stderr, "-------------------------------------\n\n");

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

    // even though this is a single-block example, we want diy to do a proper decomposition with a link
    // so that everything works downstream (reading file with links, e.g.)
    // therefore, set some dummy global domain bounds and decompose the domain
    Bounds dom_bounds(dom_dim);
    for (int i = 0; i < dom_dim; ++i)
    {
        dom_bounds.min[i] = 0.0;
        dom_bounds.max[i] = 1.0;
    }
    Decomposer decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain, const RCLink& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // set default args for diy foreach callback functions
    DomainArgs d_args;
    d_args.pt_dim       = pt_dim;
    d_args.dom_dim      = dom_dim;
    d_args.weighted     = weighted;
    d_args.n            = noise;
    d_args.t            = 0.0;
    d_args.multiblock   = false;
    d_args.verbose      = 1;
    for (int i = 0; i < pt_dim - dom_dim; i++)
        d_args.f[i] = 1.0;
    for (int i = 0; i < MAX_DIM; i++)
    {
        d_args.geom_p[i]    = geom_degree;
        d_args.vars_p[i]    = vars_degree;
        d_args.ndom_pts[i]  = ndomp;
    }

    // initilize input data

    // f16 function
    if (input == "f16")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]               = -1.0;
            d_args.max[i]               = 1.0;
            d_args.geom_nctrl_pts[i]    = geom_nctrl;
            d_args.vars_nctrl_pts[i]    = vars_nctrl;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // f17 function
    if (input == "f17")
    {
        d_args.min[0] = 80.0;   d_args.max[0] = 100.0;
        d_args.min[1] = 5.0;    d_args.max[1] = 10.0;
        d_args.min[2] = 90.0;   d_args.max[2] = 93.0;
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.geom_nctrl_pts[i]    = geom_nctrl;
            d_args.vars_nctrl_pts[i]    = vars_nctrl;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // f18 function
    if (input == "f18")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]               = -0.95;
            d_args.max[i]               = 0.95;
            d_args.geom_nctrl_pts[i]    = geom_nctrl;
            d_args.vars_nctrl_pts[i]    = vars_nctrl;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // compute the MFA

    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, d_args); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    double decode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, 1, true); });
    decode_time = MPI_Wtime() - decode_time;

    // compute the norms of analytical errors synthetic function w/o noise at different domain points than the input
    if (ntest > 0)
    {
        real_t L1, L2, Linf;                                // L-1, 2, infinity norms

        for (int i = 0; i < MAX_DIM; i++)
            d_args.ndom_pts[i] = ntest;

        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->analytical_error(cp, input, L1, L2, Linf, d_args); });

        // print analytical errors
        fprintf(stderr, "\n------ Analytical error norms -------\n");
        fprintf(stderr, "L-1        norm = %e\n", L1);
        fprintf(stderr, "L-2        norm = %e\n", L2);
        fprintf(stderr, "L-infinity norm = %e\n", Linf);
        fprintf(stderr, "-------------------------------------\n\n");
    }

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach(&Block<real_t>::print_block);
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
