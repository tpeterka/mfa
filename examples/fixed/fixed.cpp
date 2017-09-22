//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points
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
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                        // number of local blocks
    int tot_blocks  = nblocks * world.size();   // number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int   pt_dim    = 3;                        // dimension of input points
    int   dom_dim   = 2;                        // dimension of domain (<= pt_dim)
    int   degree    = 4;                        // degree (same for all dims)
    int   ndomp     = 100;                      // input number of domain points (same for all dims)
    int   nctrl     = 11;                       // input number of control points (same for all dims)

    // get command line arguments
    using namespace opts;
    Options ops(argc, argv);
    ops >> Option('d', "pt_dim",  pt_dim,   " dimension of points");
    ops >> Option('m', "dom_dim", dom_dim,  " dimension of domain");
    ops >> Option('p', "degree",  degree,   " degree in each dimension of domain");
    ops >> Option('n', "ndomp",   ndomp,    " number of input points in each dimension of domain");
    ops >> Option('c', "nctrl",   nctrl,    " number of control points in each dimension");

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
                                     &Block::create,
                                     &Block::destroy,
                                     &storage,
                                     &Block::save,
                                     &Block::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);
    diy::decompose(world.rank(), assigner, master);

    DomainArgs d_args;

    // 1d sine function f(x) = sin(x)
//     d_args.pt_dim       = pt_dim;
//     d_args.dom_dim      = dom_dim;
//     d_args.p[0]         = degree;
//     d_args.ndom_pts[0]  = ndomp;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.s            = 1.0;              // scaling factor on range
//     d_args.nctrl_pts[0] = nctrl;            // set numbers of control points here, matching dimensionality of data
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sine_data(cp, d_args); });

    // 2d sine function f(x,y) = sine(x)sine(y)
    d_args.pt_dim       = pt_dim;
    d_args.dom_dim      = dom_dim;
    d_args.p[0]         = degree;
    d_args.p[1]         = degree;
    d_args.ndom_pts[0]  = ndomp;
    d_args.ndom_pts[1]  = ndomp;
    d_args.min[0]       = -4.0 * M_PI;
    d_args.min[1]       = -4.0 * M_PI;
    d_args.max[0]       = 4.0 * M_PI;
    d_args.max[1]       = 4.0 * M_PI;
    d_args.s            = 1.0;              // scaling factor on range
    d_args.nctrl_pts[0] = nctrl;             // set numbers of control points here, matching dimensionality of data
    d_args.nctrl_pts[1] = nctrl;
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->generate_sine_data(cp, d_args); });

    // 1d sinc function f(x) = sin(x)/x
//     d_args.pt_dim       = pt_dim;
//     d_args.dom_dim      = dom_dim;
//     d_args.p[0]         = degree;
//     d_args.ndom_pts[0]  = ndomp;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     d_args.nctrl_pts[0] = nctrl;            // set numbers of control points here, matching dimensionality of data
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 2d sinc function f(x,y) = sinc(x)sinc(y)
//      d_args.pt_dim       = pt_dim;
//      d_args.dom_dim      = dom_dim;
//      d_args.p[0]         = degree;
//      d_args.p[1]         = degree;
//      d_args.ndom_pts[0]  = ndomp;
//      d_args.ndom_pts[1]  = ndomp;
//      d_args.min[0]       = -4.0 * M_PI;
//      d_args.min[1]       = -4.0 * M_PI;
//      d_args.max[0]       = 4.0 * M_PI;
//      d_args.max[1]       = 4.0 * M_PI;
//      d_args.s            = 10.0;              // scaling factor on range
//     d_args.nctrl_pts[0] = nctrl;             // set numbers of control points here, matching dimensionality of data
//     d_args.nctrl_pts[1] = nctrl;
//      master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                     { b->generate_sinc_data(cp, d_args); });

   // 3d sinc function
//     d_args.pt_dim       = pt_dim;
//     d_args.dom_dim      = dom_dim;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.p[2]         = degree;
//     d_args.ndom_pts[0]  = ndomp;
//     d_args.ndom_pts[1]  = ndomp;
//     d_args.ndom_pts[2]  = ndomp;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.min[2]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.max[2]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     d_args.nctrl_pts[0] = nctrl;             // set numbers of control points here, matching dimensionality of data
//     d_args.nctrl_pts[1] = nctrl;
//     d_args.nctrl_pts[2] = nctrl;
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

   // 4d sinc function
//     d_args.pt_dim       = pt_dim;
//     d_args.dom_dim      = dom_dim;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.p[2]         = degree;
//     d_args.p[3]         = degree;
//     d_args.ndom_pts[0]  = ndomp;
//     d_args.ndom_pts[1]  = ndomp;
//     d_args.ndom_pts[2]  = ndomp;
//     d_args.ndom_pts[3]  = ndomp;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.min[2]       = -4.0 * M_PI;
//     d_args.min[3]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.max[2]       = 4.0 * M_PI;
//     d_args.max[3]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     d_args.nctrl_pts[0] = nctrl;             // set numbers of control points here, matching dimensionality of data
//     d_args.nctrl_pts[1] = nctrl;
//     d_args.nctrl_pts[2] = nctrl;
//     d_args.nctrl_pts[3] = nctrl;
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 1d S3D
//     d_args.pt_dim       = 2;
//     d_args.dom_dim      = 1;
//     d_args.p[0]         = degree;
//     d_args.ndom_pts[0]  = 704;
//     d_args.ndom_pts[1]  = 540;
//     d_args.ndom_pts[2]  = 550;
//     d_args.nctrl_pts[0] = 140;                // set numbers of control points here, matching dimensionality of data
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/flame/6_small.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_1d_slice_3d_vector_data(cp, d_args); });

    // 2d S3D
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.ndom_pts[0]  = 704;
//     d_args.ndom_pts[1]  = 540;
//     d_args.ndom_pts[2]  = 550;
//     d_args.nctrl_pts[0] = 140;                // set numbers of control points here, matching dimensionality of data
//     d_args.nctrl_pts[1] = 108;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/flame/6_small.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_2d_slice_3d_vector_data(cp, d_args); });

    // 3d S3D
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.p[2]         = degree;
//     d_args.ndom_pts[0]  = 704;
//     d_args.ndom_pts[1]  = 540;
//     d_args.ndom_pts[2]  = 550;
//     d_args.nctrl_pts[0] = 140;                // set numbers of control points here, matching dimensionality of data
//     d_args.nctrl_pts[1] = 108;
//     d_args.nctrl_pts[2] = 110;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/flame/6_small.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_3d_vector_data(cp, d_args); });

    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, d_args); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#if 0       // normal distance
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });
#else       // range coordinate difference
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, true); });
#endif

    // debug: save knot span domains for comparing error with location in knot span
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->knot_span_domains(cp); });

    // print results
    master.foreach(&Block::print_block);
    fprintf(stderr, "encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
