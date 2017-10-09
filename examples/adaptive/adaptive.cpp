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
    diy::create_logger("trace");

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
                                     &Block<real_t>::create,
                                     &Block<real_t>::destroy,
                                     &storage,
                                     &Block<real_t>::save,
                                     &Block<real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);
    diy::decompose(world.rank(), assigner, master);

    DomainArgs d_args;

    // 1d sinc function f(x) = sin(x)/x
//     d_args.pt_dim       = pt_dim;
//     d_args.dom_dim      = dom_dim;
//     d_args.p[0]         = degree;
//     d_args.ndom_pts[0]  = ndomp;
//     d_args.min[0]       = -4.0 * M_PI;
//     d_args.max[0]       = 4.0 * M_PI;
//     d_args.s            = 10.0;              // scaling factor on range
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 2d sinc function f(x,y) = sinc(x)sinc(y)
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
     d_args.s            = 10.0;              // scaling factor on range
     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->generate_sinc_data(cp, d_args); });

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
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
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
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->generate_sinc_data(cp, d_args); });

    // 2d S3D
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.ndom_pts[0]  = 704;
//     d_args.ndom_pts[1]  = 540;
//     d_args.ndom_pts[2]  = 550;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/flame/6_small.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
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
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/flame/6_small.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_3d_vector_data(cp, d_args); });

    // 3d S3D subset
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.p[2]         = degree;
//     d_args.starts[0]    = 100;                    // NB starts are in full 3D even if example is in 2D
//     d_args.starts[1]    = 0;
//     d_args.starts[2]    = 140;
//     d_args.ndom_pts[0]  = 250;
//     d_args.ndom_pts[1]  = 300;
//     d_args.ndom_pts[2]  = 275;
//     d_args.full_dom_pts[0] = 704;               // full domain size (not just the desired subset)
//     d_args.full_dom_pts[1] = 540;
//     d_args.full_dom_pts[2] = 550;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/flame/6_small.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_3d_subset_3d_vector_data(cp, d_args); });

    // 2d nek5000
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.ndom_pts[0]  = 200;
//     d_args.ndom_pts[1]  = 200;
//     d_args.ndom_pts[2]  = 200;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/nek5000/200x200x200/0.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_2d_slice_3d_vector_data(cp, d_args); });

    // 3d nek5000
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.p[2]         = degree;
//     d_args.ndom_pts[0]  = 200;
//     d_args.ndom_pts[1]  = 200;
//     d_args.ndom_pts[2]  = 200;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/nek5000/200x200x200/0.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_3d_vector_data(cp, d_args); });

    // 2d rti
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.ndom_pts[0]  = 288;
//     d_args.ndom_pts[1]  = 512;
//     d_args.ndom_pts[2]  = 512;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/rti/dd07g_xxsmall_le.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_2d_slice_3d_vector_data(cp, d_args); });

    // 3d rti
//     d_args.pt_dim       = 4;
//     d_args.dom_dim      = 3;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.p[2]         = degree;
//     d_args.ndom_pts[0]  = 288;
//     d_args.ndom_pts[1]  = 512;
//     d_args.ndom_pts[2]  = 512;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/rti/dd07g_xxsmall_le.xyz", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_3d_vector_data(cp, d_args); });

    // 2d cesm (this is the full dimensionality of this dataset)
//     d_args.pt_dim       = 3;
//     d_args.dom_dim      = 2;
//     d_args.p[0]         = degree;
//     d_args.p[1]         = degree;
//     d_args.ndom_pts[0]  = 1800;
//     d_args.ndom_pts[1]  = 3600;
//     strncpy(d_args.infile, "/Users/tpeterka/datasets/CESM-ATM-tylor/1800x3600/FLDSC_1_1800_3600.dat", sizeof(d_args.infile));
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                    { b->read_2d_scalar_data(cp, d_args); });

    fprintf(stderr, "\nStarting adaptive encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->adaptive_encode_block(cp, norm_err_limit); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nAdaptive encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#if 0       // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });
#else       // range coordinate difference
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, true); });
#endif

    // debug: save knot span domains for comparing error with location in knot span
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->knot_span_domains(cp); });

    // print results
    master.foreach(&Block<real_t>::print_block);
    fprintf(stderr, "encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
