//--------------------------------------------------------------
// a simple example of encoding / decoding some higher dimensional data
// using a diy block model to manage the data
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.h>

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

    // 3d constant function f(x,y,z) = 1
    // d_args.pt_dim       = 4;
    // d_args.dom_dim      = 3;
    // d_args.p[0]         = 2;
    // d_args.p[1]         = 2;
    // d_args.p[2]         = 2;
    // d_args.ndom_pts[0]  = 4;
    // d_args.ndom_pts[1]  = 4;
    // d_args.ndom_pts[2]  = 4;
    // d_args.nctrl_pts[0] = 3;
    // d_args.nctrl_pts[1] = 3;
    // d_args.nctrl_pts[2] = 3;
    // d_args.min[0]       = 0.0;
    // d_args.min[1]       = 0.0;
    // d_args.min[2]       = 0.0;
    // d_args.min[3]       = 0.0;
    // d_args.max[0]       = d_args.ndom_pts[0] - 1;
    // d_args.max[1]       = d_args.ndom_pts[1] - 1;
    // d_args.max[2]       = d_args.ndom_pts[2] - 1;
    // d_args.max[3]       = 1.0;
    // d_args.s            = 1.0;
    // master.foreach(&Block::generate_constant_data, &d_args);

    // 1d magnitude function f(x) = ||x||
    // d_args.pt_dim       = 2;
    // d_args.dom_dim      = 1;
    // d_args.p[0]         = 1;
    // d_args.ndom_pts[0]  = 40;
    // d_args.nctrl_pts[0] = 20;
    // d_args.min[0]       = 0.0;
    // d_args.max[0]       = d_args.ndom_pts[0] - 1;
    // master.foreach(&Block::generate_magnitude_data, &d_args);

    // 2d magnitude function f(x,y) = ||(x,y)||
    d_args.pt_dim       = 3;
    d_args.dom_dim      = 2;
    d_args.p[0]         = 1;
    d_args.p[1]         = 1;

    // full size
    // d_args.ndom_pts[0]  = 40;
    // d_args.ndom_pts[1]  = 40;
    // d_args.nctrl_pts[0] = 20;
    // d_args.nctrl_pts[1] = 20;

    // small size for debugging
    d_args.ndom_pts[0]  = 10;
    d_args.ndom_pts[1]  = 10;
    d_args.nctrl_pts[0] = 3;
    d_args.nctrl_pts[1] = 3;

    d_args.min[0]       = 0.0;
    d_args.min[1]       = 0.0;
    d_args.max[0]       = d_args.ndom_pts[0] - 1;
    d_args.max[1]       = d_args.ndom_pts[1] - 1;
    master.foreach(&Block::generate_magnitude_data, &d_args);

    // 3d magnitude function f(x,y,z) = ||(x,y,z)||
    // d_args.pt_dim       = 4;
    // d_args.dom_dim      = 3;
    // d_args.p[0]         = 1;
    // d_args.p[1]         = 1;
    // d_args.p[2]         = 1;
    // d_args.ndom_pts[0]  = 40;
    // d_args.ndom_pts[1]  = 40;
    // d_args.ndom_pts[2]  = 40;
    // d_args.nctrl_pts[0] = 20;
    // d_args.nctrl_pts[1] = 20;
    // d_args.nctrl_pts[2] = 20;
    // d_args.min[0]       = 0.0;
    // d_args.min[1]       = 0.0;
    // d_args.min[2]       = 0.0;
    // d_args.max[0]       = d_args.ndom_pts[0] - 1;
    // d_args.max[1]       = d_args.ndom_pts[1] - 1;
    // d_args.max[2]       = d_args.ndom_pts[2] - 1;
    // master.foreach(&Block::generate_magnitude_data, &d_args);

    // 4d magnitude function f(x,y,z,t) = ||(x,y,z,t)||
    // d_args.pt_dim       = 5;
    // d_args.dom_dim      = 4;
    // d_args.p[0]         = 1;
    // d_args.p[1]         = 1;
    // d_args.p[2]         = 1;
    // d_args.p[3]         = 1;
    // d_args.ndom_pts[0]  = 40;
    // d_args.ndom_pts[1]  = 40;
    // d_args.ndom_pts[2]  = 40;
    // d_args.ndom_pts[3]  = 40;
    // d_args.nctrl_pts[0] = 20;
    // d_args.nctrl_pts[1] = 20;
    // d_args.nctrl_pts[2] = 20;
    // d_args.nctrl_pts[3] = 20;
    // d_args.min[0]       = 0.0;
    // d_args.min[1]       = 0.0;
    // d_args.min[2]       = 0.0;
    // d_args.min[3]       = 0.0;
    // d_args.max[0]       = d_args.ndom_pts[0] - 1;
    // d_args.max[1]       = d_args.ndom_pts[1] - 1;
    // d_args.max[2]       = d_args.ndom_pts[2] - 1;
    // d_args.max[3]       = d_args.ndom_pts[3] - 1;
    // master.foreach(&Block::generate_magnitude_data, &d_args);

    fprintf(stderr, "Encoding...\n");

    // encode
    master.foreach(&Block::approx_block, &nctrl_pts);

    fprintf(stderr, "Encoding done. Decoding and computing max. error...\n");

    // compute max error
    master.foreach(&Block::mag_max_error);

    // print results
    master.foreach(&Block::print_block);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);

    // debug: read the file back
    // int read_nblocks;                                       // number of blocks read
    // diy::Master               read_master(world,
    //                                       -1,
    //                                       -1,
    //                                       &Block::create,
    //                                       &Block::destroy);
    // diy::ContiguousAssigner   read_assigner(world.size(), -1);
    // diy::io::read_blocks("approx.out", world, read_assigner, read_master, &Block::load);
    // read_nblocks = read_master.size();
    // fprintf(stderr, "%d blocks read\n", read_nblocks);
    // read_master.foreach(&Block::print_block);
}
