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
    float norm_err_limit = 1.0;                 // maximum normalized error limit
    int   pt_dim         = 3;                   // dimension of input points
    int   dom_dim        = 2;                   // dimension of domain (<= pt_dim)
    int   degree         = 4;                   // degree (same for all dims)
    int   ndomp          = 100;                 // input number of domain points (same for all dims)

    // get command line arguments
    using namespace opts;
    Options ops(argc, argv);
    ops >> Option('e', "error",  norm_err_limit, "maximum normalized error limit");
    ops >> Option('d', "pt_dim", pt_dim, " dimension of points");
    ops >> Option('m', "dom_dim", dom_dim, " dimension of domain");
    ops >> Option('p', "degree", degree, "degree in each dimension of domain");
    ops >> Option('n', "ndomp", ndomp, "  number of input points in each dimension of domain");

    if (ops >> Present('h', "help", "show help"))
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // initialize DIY
    diy::FileStorage          storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &Block<precision>::create,
                                     &Block<precision>::destroy,
                                     &storage,
                                     &Block<precision>::save,
                                     &Block<precision>::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);
    diy::decompose(world.rank(), assigner, master);

    DomainArgs d_args;

    // 1d sinc function f(x) = sin(x)/x
    d_args.pt_dim       = pt_dim;
    d_args.dom_dim      = dom_dim;
    d_args.p[0]         = degree;
    d_args.ndom_pts[0]  = ndomp;
    d_args.min[0]       = -4.0 * M_PI;
    d_args.max[0]       = 4.0 * M_PI;
    d_args.s            = 10.0;              // scaling factor on range
    master.foreach([&](Block<precision>* b, const diy::Master::ProxyWithLink& cp)
                   { b->generate_sinc_data(cp, d_args); });

    // encode data
    fprintf(stderr, "\nStarting nonlinear encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<precision>* b, const diy::Master::ProxyWithLink& cp)
            { b->nonlinear_encode_block(cp, norm_err_limit); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nNonlinear encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#if 0       // normal distance
    master.foreach([&](Block<precision>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });
#else       // range coordinate difference
    master.foreach([&](Block<precision>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, true); });
#endif

    // debug: save knot span domains for comparing error with location in knot span
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    master.foreach([&](Block<precision>* b, const diy::Master::ProxyWithLink& cp)
            { b->knot_span_domains(cp); });

    // print results
    master.foreach(&Block<precision>::print_block);
    fprintf(stderr, "encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
