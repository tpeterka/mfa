//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ adaptive number of control points
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <vector>
#include <iostream>
#include <cmath>

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
    diy::mpi::environment  env(argc, argv); // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;           // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                     // number of local blocks
    int tot_blocks  = nblocks * world.size();
    int mem_blocks  = -1;                    // everything in core for now
    int num_threads = 1;                     // needed in order to do timing

    // default command line arguments
    real_t norm_err_limit = 1.0;                 // maximum normalized error limit
    int    pt_dim         = 3;                   // dimension of input points
    int    dom_dim        = 2;                   // dimension of domain (<= pt_dim)
    int    degree         = 4;                   // degree (same for all dims)
    int    ndomp          = 100;                 // input number of domain points (same for all dims)
    string input          = "sinc";              // input dataset

    // get command line arguments
    opts::Options ops(argc, argv);
    ops >> opts::Option('e', "error",   norm_err_limit, " maximum normalized error limit");
    ops >> opts::Option('d', "pt_dim",  pt_dim,         " dimension of points");
    ops >> opts::Option('m', "dom_dim", dom_dim,        " dimension of domain");
    ops >> opts::Option('p', "degree",  degree,         " degree in each dimension of domain");
    ops >> opts::Option('n', "ndomp",   ndomp,          " number of input points in each dimension of domain");
    ops >> opts::Option('i', "input",   input,          " input dataset");

    if (ops >> opts::Present('h', "help", "show help"))
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "error = "    << norm_err_limit << " pt_dim = "     << pt_dim << " dom_dim = " << dom_dim <<
        "\ndegree = " << degree         << " input pts = "  << ndomp  <<
        "\ninput = "  << input          << endl;
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
    diy::decompose(world.rank(), assigner, master);

    DomainArgs d_args;

    // set default args for diy foreach callback functions
    d_args.pt_dim       = pt_dim;
    d_args.dom_dim      = dom_dim;
    for (int i = 0; i < MAX_DIM; i++)
    {
        d_args.p[i]         = degree;
        d_args.ndom_pts[i]  = ndomp;
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]       = -4.0 * M_PI;
            d_args.max[i]       = 4.0  * M_PI;
        }
        d_args.s            = 10.0;              // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_sinc_data(cp, d_args); });
    }

    // compute the MFA

    fprintf(stderr, "\nStarting nonlinear encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->nonlinear_encode_block(cp, norm_err_limit); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nNonlinear encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#ifdef CURVE_PARAMS     // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, 1, true); });
#else                   // range coordinate difference
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, 1, true); });
#endif

    // debug: save knot span domains for comparing error with location in knot span
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->knot_span_domains(cp); });

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach(&Block<real_t>::print_block);
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
