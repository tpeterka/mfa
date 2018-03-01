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

#include "../block.hpp"
#include "../opts.h"

using namespace std;

typedef  diy::RegularDecomposer<Bounds> Decomposer;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int tot_blocks  = world.size();             // default number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int    pt_dim    = 3;                        // dimension of input points
    int    dom_dim   = 2;                        // dimension of domain (<= pt_dim)
    int    degree    = 4;                        // degree (same for all dims)
    int    ndomp     = 100;                      // input number of domain points (same for all dims)
    int    nctrl     = 11;                       // input number of control points (same for all dims)
    string input     = "sine";                   // input dataset
    bool   weighted  = true;                     // solve for and use weights

    // get command line arguments
    using namespace opts;
    Options ops(argc, argv);
    ops >> Option('d', "pt_dim",     pt_dim,   " dimension of points");
    ops >> Option('m', "dom_dim",    dom_dim,  " dimension of domain");
    ops >> Option('p', "degree",     degree,   " degree in each dimension of domain");
    ops >> Option('n', "ndomp",      ndomp,    " number of input points in each dimension of domain");
    ops >> Option('c', "nctrl",      nctrl,    " number of control points in each dimension");
    ops >> Option('i', "input",      input,    " input dataset");
    ops >> Option('w', "weights",    weighted, " solve for and use weights");
    ops >> Option('b', "tot_blocks", tot_blocks, "total number of blocks");

    if (ops >> Present('h', "help", "show help"))
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "pt_dim = "    << pt_dim << " dom_dim = "    << dom_dim  <<
        "\ndegree = " << degree  << " input pts = "  << ndomp    << " ctrl pts = " << nctrl   <<
        "\ninput = "  << input   << " weighted = "   << weighted << " tot_blocks = " << tot_blocks << endl;
#ifdef CURVE_PARAMS
    cerr << "parameterization method = curve" << endl;
#else
    cerr << "parameterization method = domain" << endl;
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

    // set global domain bounds
    Bounds dom_bounds;
    for (int i = 0; i < dom_dim; ++i)
    {
        dom_bounds.min[i] = -4.0 * M_PI;
        dom_bounds.max[i] =  4.0 * M_PI;
    }

    // decompose the domain into blocks
    diy::RegularDecomposer<Bounds> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain, const diy::Link& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim); });

    DomainArgs d_args;

    // set default args for diy foreach callback functions
    d_args.pt_dim       = pt_dim;
    d_args.dom_dim      = dom_dim;
    d_args.weighted     = weighted;
    d_args.multiblock   = true;
    d_args.f            = 1.0;
    for (int i = 0; i < MAX_DIM; i++)
    {
        d_args.p[i]         = degree;
        d_args.ndom_pts[i]  = ndomp;
        d_args.nctrl_pts[i] = nctrl;
    }

    // initilize input data

    // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
    if (input == "sine")
    {
        d_args.s = 1.0;              // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_sine_data(cp, d_args); });
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        d_args.s = 20.0;              // scaling factor on range
        d_args.f = 1.0;               // frequency multiplier
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_sinc_data(cp, d_args); });
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
#ifdef CURVE_PARAMS     // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });
#else                   // range coordinate difference
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, true); });
#endif
    decode_time = MPI_Wtime() - decode_time;

    // debug: save knot span domains for comparing error with location in knot span
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->knot_span_domains(cp); });

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach(&Block<real_t>::print_block);
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
