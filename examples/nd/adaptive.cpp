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

    int nctrl_pts;

    // generate

    DomainArgs d_args;

    // 1d sinc function f(x) = sinc(x)
//     float norm_err_limit =9.2e-3;           // normalized maximum allowable error
//     d_args.pt_dim       = 2;
//     d_args.dom_dim      = 1;
//     d_args.p[0]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.nctrl_pts[0] = 20;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.s            = 10.0;           // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });
//
    // 1d very small sinc function
//     float norm_err_limit = 1.5e-1;
//     d_args.pt_dim       = 2;
//     d_args.dom_dim      = 1;
//     d_args.p[0]         = 2;
//     d_args.ndom_pts[0]  = 20;
//     d_args.nctrl_pts[0] = 6;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // small 2d sinc function f(x,y) = sinc(x)sinc(y)
//     float norm_err_limit = 3.0e-1;
//     float norm_err_limit = 3.0e-2;
//     float norm_err_limit = 3.0e-3;
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.ndom_pts[1]  = 100;
//     d_args.nctrl_pts[0] = 8;
//     d_args.nctrl_pts[1] = 8;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 2d sinc function f(x,y) = sinc(x)sinc(y)
//     float norm_err_limit = 1.0e-2;           // normalized maximum allowable error
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.ndom_pts[1]  = 100;
//     d_args.nctrl_pts[0] = 20;
//     d_args.nctrl_pts[1] = 20;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // --- experiments for the paper ---

    // 2d sinc function
//     float norm_err_limit = 3.0e-1;
//     float norm_err_limit = 3.0e-2;
//     float norm_err_limit = 3.0e-3;
//     float norm_err_limit = 3.0e-4;
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.ndom_pts[1]  = 100;
//     d_args.nctrl_pts[0] = 20;
//     d_args.nctrl_pts[1] = 20;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

   // 3d sinc function
//     float norm_err_limit = 1.0e-2;
//     float norm_err_limit = 1.0e-3;
    float norm_err_limit = 1.0e-4;
//     float norm_err_limit = 1.0e-5;
    d_args.pt_dim       = 4;
    d_args.dom_dim      = 3;
    d_args.p[0]         = 4;
    d_args.p[1]         = 4;
    d_args.p[2]         = 4;
    d_args.ndom_pts[0]  = 100;
    d_args.ndom_pts[1]  = 100;
    d_args.ndom_pts[2]  = 100;
    d_args.nctrl_pts[0] = 20;
    d_args.nctrl_pts[1] = 20;
    d_args.nctrl_pts[2] = 20;
    d_args.min[0]       = -4.0 * M_PI;
    d_args.min[1]       = -4.0 * M_PI;
    d_args.min[2]       = -4.0 * M_PI;
    d_args.max[0]       = 4.0 * M_PI;
    d_args.max[1]       = 4.0 * M_PI;
    d_args.max[2]       = 4.0 * M_PI;
    d_args.s            = 10.0;              // scaling factor on range
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                   { b->generate_sinc_data(cp, d_args); });

    // 3d S3D
//     float norm_err_limit = 1.0e0;
//     float norm_err_limit = 1.0e-1;
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.p[2]         = 4;
//     d_args.ndom_pts[0]  = 704;
//     d_args.ndom_pts[1]  = 540;
//     d_args.ndom_pts[2]  = 550;
//     d_args.nctrl_pts[0] = 140;
//     d_args.nctrl_pts[1] = 108;
//     d_args.nctrl_pts[2] = 110;
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_3d_file_data(cp, d_args); });

    double encode_time = MPI_Wtime();

    // loop until the error is low enough
    //
    // TODO: need algorithm how to decide correct starting number of control points and knots
    fprintf(stderr, "Starting adaptive encoding...\n");
    int iter;
    bool done;
    for (iter = 0; ; iter++)
    {
        fprintf(stderr, "-----\n\nEncoding iteration %d...\n", iter);
        if (iter == 0)
            master.foreach(&Block::encode_block);
        else
            master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                           { b->reencode_block(cp, norm_err_limit, done); });

        // debug: compute max error to see that it is decreasing
//         fprintf(stderr, "\n iter=%d computing max. error...\n", iter);
//         master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                 { b->error(cp, false); });

        if (done)
            break;
    }
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "-----\n\nAdaptive adaptive encoding done in %d iteration(s)\n", iter);

    // debug: compute max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });

    // print results
    master.foreach(&Block::print_block);
    fprintf(stderr, "encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
