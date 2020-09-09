//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points and
// multiple blocks
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

#include "opts.h"

#include "block.hpp"

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int tot_blocks  = world.size();             // default number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int    pt_dim       = 3;                    // dimension of input points
    int    dom_dim      = 2;                    // dimension of domain (<= pt_dim)
    int    geom_degree  = 1;                    // degree for geometry (same for all dims)
    int    vars_degree  = 4;                    // degree for science variables (same for all dims)
    int    ndomp        = 100;                  // input number of domain points (same for all dims)
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    int    vars_nctrl   = 11;                   // input number of control points for all science variables (same for all dims)
    string input        = "sine";               // input dataset
    int    weighted     = 1;                    // solve for and use weights (bool 0 or 1)
    int    strong_sc    = 1;                    // strong scaling (bool 0 or 1, 0 = weak scaling)
    real_t ghost        = 0.1;                  // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    int    error        = 1;                    // decode all input points and check error (bool 0 or 1)
    string infile;                              // input file name
    bool   help;                                // show help

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('b', "tot_blocks",  tot_blocks, " total number of blocks");
    ops >> opts::Option('t', "strong_sc",   strong_sc,  " strong scaling (1 = strong, 0 = weak)");
    ops >> opts::Option('o', "overlap",     ghost,      " relative ghost zone overlap (0.0 - 1.0)");
    ops >> opts::Option('f', "infile",      infile,     " input file name");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // minimal number of geometry control points if not specified
    if (geom_nctrl == -1)
        geom_nctrl = geom_degree + 1;

    // echo args
    if (world.rank() == 0)
    {
        fprintf(stderr, "\n--------- Input arguments ----------\n");
        cerr <<
            "pt_dim = "         << pt_dim       << " dom_dim = "        << dom_dim      <<
            "\ngeom_degree = "  << geom_degree  << " vars_degree = "    << vars_degree  <<
            "\ninput pts = "    << ndomp        << " geom_ctrl pts = "  << geom_nctrl   <<
            "\nvars_ctrl_pts = "<< vars_nctrl   << " input = "          << input        <<
            " tot_blocks = "    << tot_blocks   << " strong scaling = " << strong_sc    << endl;
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
    }

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

    // set global domain bounds
    Bounds<real_t> dom_bounds(dom_dim);
    for (int i = 0; i < dom_dim; ++i)
    {
        dom_bounds.min[i] = -4.0 * M_PI;
        dom_bounds.max[i] =  4.0 * M_PI;
    }

    // decompose the domain into blocks
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost); });
    vector<int> divs(dom_dim);                          // number of blocks in each dimension
    decomposer.fill_divisions(divs);

    DomainArgs d_args(dom_dim, pt_dim);

    // set default args for diy foreach callback functions
    d_args.weighted     = weighted;
    d_args.multiblock   = true;
    d_args.verbose      = 0;
    d_args.r            = 0.0;
    d_args.t            = 0.0;
    for (int i = 0; i < dom_dim; i++)
    {
        d_args.geom_p[i]    = geom_degree;
        d_args.vars_p[0][i] = vars_degree;      // assuming one science variable, vars_p[0]
        if (strong_sc)                          // strong scaling, reduced number of points per block
        {
            d_args.ndom_pts[i]          = ndomp      / divs[i];
            d_args.geom_nctrl_pts[i]    = geom_nctrl / divs[i] > geom_degree ? geom_nctrl / divs[i] : geom_degree + 1;
            d_args.vars_nctrl_pts[0][i] = vars_nctrl / divs[i];     // assuming one science variable, vars_nctrl_pts[0]
        } else                                  // weak scaling, same number of points per block
        {
            d_args.ndom_pts[i]          = ndomp;
            d_args.geom_nctrl_pts[i]    = geom_nctrl;
            d_args.vars_nctrl_pts[0][i] = vars_nctrl;               // assuming one science variable, vars_nctrl_pts[0]
        }
    }
    for (int i = 0; i < pt_dim - dom_dim; i++)
        d_args.f[i] = 1.0;

    // initilize input data

    // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
    if (input == "sine")
    {
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = i + 1;                        // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 10.0 * (i + 1);               // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // compute the MFA
    double encode_time = MPI_Wtime();
    if (world.rank() == 0)
        fprintf(stderr, "\nStarting fixed encoding...\n\n");
    world.barrier();                     // to synchronize timing
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, d_args); });
    world.barrier();                     // to synchronize timing
    encode_time = MPI_Wtime() - encode_time;
    if (world.rank() == 0)
        fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    double decode_time = MPI_Wtime();
    if (error)
    {
        if (world.rank() == 0)
            fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#ifdef CURVE_PARAMS     // normal distance
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->error(cp, 0, true); });
#else                   // range coordinate difference
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->range_error(cp, 0, true, true); });
#endif
        decode_time = MPI_Wtime() - decode_time;
    }

    // print block results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, error); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    if (error)
        fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // print overall timing results
    if (world.rank() == 0)
        fprintf(stderr, "\noverall encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);

    // check the results of the last (only) science variable
    // only checking error for one rank and one configuration of total number of blocks
    if (world.rank() == 0 && tot_blocks == 4)
    {
        Block<real_t>* b        = static_cast<Block<real_t>*>(master.block(0));
        int     ndom_dims       = b->mfa->ndom_pts().size();        // domain dimensionality
        real_t  range_extent    = b->domain.col(ndom_dims).maxCoeff() - b->domain.col(ndom_dims).minCoeff();
        real_t  err_factor      = 1.0e-3;
        real_t  our_err         = b->max_errs[0] / range_extent;    // actual normalized max_err
        real_t  expect_err;                                         // expected (normalized max) error

        // for ./fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 500 -v 50 -b 4 -t 1 -w 0
        if (strong_sc && !ghost)
            expect_err   = 4.021332e-07;
        if (strong_sc && ghost)
            expect_err   = 9.842469e-07;
        // for ./fixed-multiblock-test> -i sinc -d 3 -m 2 -p 1 -q 5 -n 100 -v 10 -b 4 -t 0 -w 0
        if (!strong_sc && !ghost)
            expect_err   = 7.246910e-03;
        if (!strong_sc && ghost)
            expect_err   = 7.655586e-03;

        if (fabs(expect_err - our_err) / expect_err > err_factor)
        {
            fprintf(stderr, "our error (%e) and expected error (%e) differ by more than a factor of %e\n", our_err, expect_err, err_factor);
            abort();
        }
    }
}
