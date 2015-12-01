//--------------------------------------------------------------
// a simple example of encoding / decoding some curve data
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

    // constant function
    // d_args.p       = 3;
    // d_args.npts    = 20;
    // d_args.min_x   = 0.0;
    // d_args.max_x   = d_args.npts - 1.0;
    // d_args.y_scale = 1.0;
    // nctrl_pts      = 10;
    // master.foreach(&Block::generate_constant_data, &d_args);

    // circle function
    // d_args.p       = 3;
    // d_args.npts    = 100;
    // d_args.min_x   = 0.0;
    // d_args.max_x   = M_PI / 2.0;
    // d_args.y_scale = 1.0;
    // nctrl_pts      = 10;
    // master.foreach(&Block::generate_circle_data, &d_args);

    // sine function
    // d_args.p       = 3;
    // d_args.npts    = 100;
    // d_args.min_x   = 0.0;
    // d_args.max_x   = 2 * M_PI;
    // d_args.y_scale = 1.0;
    // nctrl_pts      = 10;
    // master.foreach(&Block::generate_sine_data, &d_args);

    // sinc function
    d_args.p       = 3;
    d_args.npts    = 1000;
    d_args.min_x   = -4.0 * M_PI;
    d_args.max_x   = 4.0 * M_PI;
    d_args.y_scale = 10.0;
    nctrl_pts      = 70;
    master.foreach(&Block::generate_sinc_data, &d_args);

    // read file
    // d_args.p     = 3;
    // d_args.npts  = 704;
    // nctrl_pts    = 140;
    // master.foreach(&Block::read_file_data, &d_args);

    // encode
    master.foreach(&Block::approx_block, &nctrl_pts);

    // compute error
    ErrArgs e_args;
    e_args.max_niter  = 10;                  // max number of search iterations
    e_args.err_bound  = 0.001;               // desrired error bound
    e_args.search_rad = 4;                   // search range is +/- this many input parameters
    master.foreach(&Block::max_error, &e_args);

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
    // read_nblocks = master.size();
    // fprintf(stderr, "%d blocks read\n", read_nblocks);
}
