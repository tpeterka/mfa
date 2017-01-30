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

    // 1d sinc function f(x) - sinc(x)
//     float norm_err_limit = 1.5e-2;           // normalized maximum allowable error
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
    // 2d sinc function f(x,y) = sinc(x)sinc(y)
    float norm_err_limit = 1.0e-2;           // normalized maximum allowable error
    d_args.pt_dim       = 3;
    d_args.dom_dim      = 2;
    d_args.p[0]         = 4;
    d_args.p[1]         = 4;
    d_args.ndom_pts[0]  = 100;
    d_args.ndom_pts[1]  = 100;
    d_args.nctrl_pts[0] = 20;
    d_args.nctrl_pts[1] = 20;
    d_args.min[0]       = -4.0 * M_PI;
    d_args.min[1]       = -4.0 * M_PI;
    d_args.max[0]       = 4.0 * M_PI;
    d_args.max[1]       = 4.0 * M_PI;
    d_args.s            = 10.0;              // scaling factor on range
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                   { b->generate_sinc_data(cp, d_args); });

    double encode_time = MPI_Wtime();

    // loop until the error is low enough
    //
    // TODO: need algorithm how to decide correct starting number of control points and
    // knots, and how to converge on the desired error limit (don't need to add control points
    // at every point where the error is too big, one strategically placed control point may
    // suffice). Start with a larger error and cut in half each time?
    // TODO, for now just 2 iterations after some fixed number of control points, but
    // much more research to do here
    fprintf(stderr, "Starting adaptive encoding...\n");
    int iter;
    for (iter = 0; ; iter++)
    {
        fprintf(stderr, "Encoding iteration %d...\n", iter);
        if (iter == 0)
            master.foreach(&Block::encode_block);
        else
            master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                           { b->reencode_block(cp, norm_err_limit); });

        // TODO: remove eventually; should not need complete decoding to adapt knots
        fprintf(stderr, "Encoding iteration %d done. Decoding and computing max. error...\n", iter);
        master.foreach(&Block::decode_block);
        master.foreach(&Block::max_error);

        // TODO: need to get max error of all blocks once there are more than one
        if (fabs(((Block*)master.block(0))->max_err) < norm_err_limit)
            break;
    }
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "Adaptive adaptive encoding done in %d iteration(s)\n", iter + 1);

    // print results
    master.foreach(&Block::print_block);
    fprintf(stderr, "encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
