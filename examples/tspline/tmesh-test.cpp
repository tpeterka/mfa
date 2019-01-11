//--------------------------------------------------------------
// test driver for tmesh class
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
    diy::create_logger("trace");

    // initialize MPI
    diy::mpi::environment  env(argc, argv); // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;           // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                     // number of local blocks
    int tot_blocks  = nblocks * world.size();
    int mem_blocks  = -1;                    // everything in core for now
    int num_threads = 1;                     // needed in order to do timing

    // default command line arguments
    real_t norm_err_limit = 1.0;                      // maximum normalized error limit
    int    pt_dim         = 3;                        // dimension of input points
    int    dom_dim        = 2;                        // dimension of domain (<= pt_dim)
    int    geom_degree    = 1;                        // degree for geometry (same for all dims)
    int    vars_degree    = 4;                        // degree for science variables (same for all dims)
    int    ndomp          = 100;                      // input number of domain points (same for all dims)
    string input          = "sinc";                   // input dataset
    int    max_rounds     = 0;                        // max. number of rounds (0 = no maximum)
    bool   weighted       = true;                     // solve for and use weights
    real_t rot            = 0.0;                      // rotation angle in degrees
    real_t twist          = 0.0;                      // twist (waviness) of domain (0.0-1.0)

    // get command line arguments
    opts::Options ops(argc, argv);
    ops >> opts::Option('e', "error",       norm_err_limit, " maximum normalized error limit");
    ops >> opts::Option('d', "pt_dim",      pt_dim,         " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,        " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree,    " degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree,    " degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,          " number of input points in each dimension of domain");
    ops >> opts::Option('i', "input",       input,          " input dataset");
    ops >> opts::Option('r', "rounds",      max_rounds,     " maximum number of iterations");
    ops >> opts::Option('w', "weights",     weighted,       " solve for and use weights");
    ops >> opts::Option('r', "rotate",      rot,            " rotation angle of domain in degrees");
    ops >> opts::Option('t', "twist",       twist,          " twist (waviness) of domain (0.0-1.0)");

    if (ops >> opts::Present('h', "help", " show help"))
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "error = "          << norm_err_limit   << " pt_dim = "         << pt_dim       << " dom_dim = "        << dom_dim      <<
        "\ngeom_degree = "  << geom_degree      << " vars_degree = "    << vars_degree  <<
        "\ninput pts = "    << ndomp            << " input = "          << input        << " max. rounds = "    << max_rounds   << endl;
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
    d_args.multiblock   = false;
    d_args.verbose      = 1;
    d_args.r            = 0.0;
    d_args.t            = 0.0;
    for (int i = 0; i < pt_dim - dom_dim; i++)
        d_args.f[i] = 1.0;
    for (int i = 0; i < MAX_DIM; i++)
    {
        d_args.geom_p[i]    = geom_degree;
        d_args.vars_p[i]    = vars_degree;
        d_args.ndom_pts[i]  = ndomp;
    }

    // start with some data so that all the initialization works
    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]       = -4.0 * M_PI;
            d_args.max[i]       = 4.0  * M_PI;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range
        d_args.r            = rot * M_PI / 180.0;   // domain rotation angle in rads
        d_args.t            = twist;                // twist (waviness) of domain
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_sinc_data(cp, d_args); });
    }

    // initialize tmesh with a tensor product
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->init_tmesh(cp, d_args); });

    // print tmesh
    fmt::print(stderr, "\n----- Initial T-mesh -----\n\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_tmesh(cp); });
    fmt::print(stderr, "--------------------------\n\n");


    // test finding a local knot vector
    vector<size_t> anchor(dom_dim);
    anchor[0] = 1;
    anchor[1] = 4;
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->local_knot_vector(cp, anchor); });

    // refine tmesh
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->refine1_tmesh(cp); });

    // print tmesh
    fmt::print(stderr, "\n----- T-mesh after first refinement -----\n\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_tmesh(cp); });
    fmt::print(stderr, "--------------------------\n\n");

    // test finding a local knot vector
    anchor[0] = 5;
    anchor[1] = 5;
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->local_knot_vector(cp, anchor); });

    // refine tmesh again
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->refine2_tmesh(cp); });

    // print tmesh
    fmt::print(stderr, "\n----- T-mesh after second refinement -----\n\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_tmesh(cp); });
    fmt::print(stderr, "--------------------------\n\n");

    // test finding a local knot vector
    anchor[0] = 5;
    anchor[1] = 1;
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->local_knot_vector(cp, anchor); });

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
