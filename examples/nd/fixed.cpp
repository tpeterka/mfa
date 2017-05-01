//--------------------------------------------------------------
// examples of encoding / decoding higher dimensional data w/ fixed number of control points
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

    // 1d sinc function
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

//     // 1d very small sinc function
//     d_args.pt_dim       = 2;
//     d_args.dom_dim      = 1;
//     d_args.p[0]         = 3;
//     d_args.ndom_pts[0]  = 20;
//     d_args.nctrl_pts[0] = 6;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // test of 2 overlapping sinc functions
    // 101 total domain points (100 spans) split into 2 parts with p+1 overlapping points
//     int ghost = 5;
// #if 1
//     d_args.pt_dim       = 2;
//     d_args.dom_dim      = 1;
//     d_args.p[0]         = 3;
//     d_args.ndom_pts[0]  = 51 + ghost;
//     d_args.nctrl_pts[0] = 15;
//     d_args.min[0]       = -2.0 * M_PI;
//     d_args.max[0]       = (float)(ghost - 1) * 4.0/100 * M_PI;
//     d_args.s            = 5.0;             // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                { b->generate_sinc_data(cp, d_args); });
// #else
//     d_args.pt_dim       = 2;
//     d_args.dom_dim      = 1;
//     d_args.p[0]         = 3;
//     d_args.ndom_pts[0]  = 51 + ghost;
//     d_args.nctrl_pts[0] = 15;
//     d_args.min[0]       = -(float)(ghost - 1) * 4.0/100 * M_PI;
//     d_args.max[0]       = 2.0 * M_PI;
//     d_args.s            = 5.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });
// #endif

//     // 2d sinc function f(x,y) = sinc(x)sinc(y)
//     float norm_err_limit = 2.5e-1;
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

    // 2d very small sinc function f(x,y) = sinc(x)sinc(y)
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = 2;
//     d_args.p[1]         = 2;
//     d_args.ndom_pts[0]  = 20;
//     d_args.ndom_pts[1]  = 20;
//     d_args.nctrl_pts[0] = 6;
//     d_args.nctrl_pts[1] = 6;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

   // 3d sinc function f(x,y,z) = sinc(x)sinc(y)sinc(z)
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.p[2]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.ndom_pts[1]  = 100;
//     d_args.ndom_pts[2]  = 100;
//     d_args.nctrl_pts[0] = 8;
//     d_args.nctrl_pts[1] = 8;
//     d_args.nctrl_pts[2] = 8;
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
//     d_args.pt_dim       = 5;
//     d_args.dom_dim      = 4;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.p[2]         = 4;
//     d_args.p[3]         = 4;
//     d_args.ndom_pts[0]  = 20;
//     d_args.ndom_pts[1]  = 20;
//     d_args.ndom_pts[2]  = 20;
//     d_args.ndom_pts[3]  = 20;
//     d_args.nctrl_pts[0] = 10;
//     d_args.nctrl_pts[1] = 10;
//     d_args.nctrl_pts[2] = 10;
//     d_args.nctrl_pts[3] = 10;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.min[2]       = -4.0 * M_PI;
//     d_args.min[3]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.max[2]       = 4.0 * M_PI;
//     d_args.max[3]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 1d read file
    // d_args.pt_dim       = 2;
    // d_args.dom_dim      = 1;
    // d_args.p[0]         = 3;
    // d_args.ndom_pts[0]  = 704;
    // d_args.nctrl_pts[0] = 140;
    // master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
    //                { b->read_1d_file_data(cp, d_args); });

    // 2d read file
    // d_args.pt_dim       = 3;
    // d_args.dom_dim      = 2;
    // d_args.p[0]         = 4;
    // d_args.p[1]         = 4;
    // d_args.ndom_pts[0]  = 704;
    // d_args.ndom_pts[1]  = 540;
    // d_args.nctrl_pts[0] = 140;
    // d_args.nctrl_pts[1] = 108;
    // master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
    //                { b->read_2d_file_data(cp, d_args); });

    // --- experiments for paper ---

    // 1d sinc function
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

    // 2d sinc function
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

   // 4d sinc function
//     d_args.pt_dim       = 5;
//     d_args.dom_dim      = 4;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.p[2]         = 4;
//     d_args.p[3]         = 4;
//     d_args.ndom_pts[0]  = 100;
//     d_args.ndom_pts[1]  = 100;
//     d_args.ndom_pts[2]  = 100;
//     d_args.ndom_pts[3]  = 100;
//     d_args.nctrl_pts[0] = 20;
//     d_args.nctrl_pts[1] = 20;
//     d_args.nctrl_pts[2] = 20;
//     d_args.nctrl_pts[3] = 20;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.min[1]       = -4.0 * M_PI;
//     d_args.min[2]       = -4.0 * M_PI;
//     d_args.min[3]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.max[1]       = 4.0 * M_PI;
//     d_args.max[2]       = 4.0 * M_PI;
//     d_args.max[3]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 1d S3D
//     d_args.pt_dim       = 2;
//     d_args.dom_dim      = 1;
//     d_args.p[0]         = 4;
//     d_args.ndom_pts[0]  = 704;
//     d_args.nctrl_pts[0] = 140;
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_1d_file_data(cp, d_args); });

    // 2d S3D
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = 4;
//     d_args.p[1]         = 4;
//     d_args.ndom_pts[0]  = 704;
//     d_args.ndom_pts[1]  = 540;
//     d_args.nctrl_pts[0] = 140;
//     d_args.nctrl_pts[1] = 108;
//     master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_2d_file_data(cp, d_args); });

    // 3d S3D
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

    fprintf(stderr, "Encoding...\n");
    double encode_time = MPI_Wtime();
    master.foreach(&Block::encode_block);
    encode_time = MPI_Wtime() - encode_time;

    fprintf(stderr, "Encoding done. Decoding and computing max. error...\n");
    double decode_time = MPI_Wtime();
    master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });
    decode_time = MPI_Wtime() - decode_time;

    // print results
    master.foreach(&Block::print_block);
    fprintf(stderr, "encoding time = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time = %.3lf s.\n", decode_time);

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
