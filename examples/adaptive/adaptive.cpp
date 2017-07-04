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

#include "block.hpp"
#include "../opts.h"

#include "../include/opts.h"

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

    float norm_err_limit = 1.0;             // maximum normalized errro limit

    // get command line arguments
    using namespace opts;
    Options ops(argc, argv);
    ops
        >> Option('e', "error",  norm_err_limit,        "maximum normalized error limit")
        ;
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

    // small 2d sinc function f(x,y) = sinc(x)sinc(y)
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.ndom_pts[1]  = 100;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 2d sinc function f(x,y) = sinc(x)sinc(y)
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.ndom_pts[0]  = 500;
//     d_args.ndom_pts[1]  = 500;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

   // small 3d sinc function
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.p[2]         = 4;
//     d_args.ndom_pts[0]  = 20;
//     d_args.ndom_pts[1]  = 20;
//     d_args.ndom_pts[2]  = 20;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.min[2]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.max[2]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

   // 3d sinc function
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.p[2]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.ndom_pts[1]  = 100;
//     d_args.ndom_pts[2]  = 100;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.min[2]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.max[2]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

   // 4d sinc function
    d_args.pt_dim       = 5;
    d_args.dom_dim      = 4;
    d_args.p[0]         = 4;
    d_args.p[1]         = 4;
    d_args.p[2]         = 4;
    d_args.p[3]         = 4;
    d_args.ndom_pts[0]  = 100;
    d_args.ndom_pts[1]  = 100;
    d_args.ndom_pts[2]  = 100;
    d_args.ndom_pts[3]  = 100;
    d_args.min[0]       = -4.0 * M_PI;
    d_args.min[1]       = -4.0 * M_PI;
    d_args.min[2]       = -4.0 * M_PI;
    d_args.min[3]       = -4.0 * M_PI;
    d_args.max[0]       = 4.0 * M_PI;
    d_args.max[1]       = 4.0 * M_PI;
    d_args.max[2]       = 4.0 * M_PI;
    d_args.max[3]       = 4.0 * M_PI;
    d_args.s            = 10.0;              // scaling factor on range
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                   { b->generate_sinc_data(cp, d_args); });

    // 3d S3D
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = 3;
//     d_args.p[1]         = 3;
//     d_args.p[2]         = 3;
//     d_args.ndom_pts[0]  = 704;
//     d_args.ndom_pts[1]  = 540;
//     d_args.ndom_pts[2]  = 550;
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_3d_file_data(cp, d_args); });

    // encode data
    fprintf(stderr, "Starting adaptive encoding...\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->adaptive_encode_block(cp, norm_err_limit); });
    encode_time = MPI_Wtime() - encode_time;

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });

    // print results
    master.foreach(&Block::print_block);
    fprintf(stderr, "encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
