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
#include <string>

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
    real_t norm_err_limit = 1.0;                      // maximum normalized error limit
    int    pt_dim         = 3;                        // dimension of input points
    int    dom_dim        = 2;                        // dimension of domain (<= pt_dim)
    int    degree         = 4;                        // degree (same for all dims)
    int    ndomp          = 100;                      // input number of domain points (same for all dims)
    string input          = "sinc";                   // input dataset
    int    max_rounds     = 0;                        // max. number of rounds (0 = no maximum)
    bool   weighted       = true;                     // solve for and use weights

    // get command line arguments
    using namespace opts;
    Options ops(argc, argv);
    ops >> Option('e', "error",   norm_err_limit, " maximum normalized error limit");
    ops >> Option('d', "pt_dim",  pt_dim,         " dimension of points");
    ops >> Option('m', "dom_dim", dom_dim,        " dimension of domain");
    ops >> Option('p', "degree",  degree,         " degree in each dimension of domain");
    ops >> Option('n', "ndomp",   ndomp,          " number of input points in each dimension of domain");
    ops >> Option('i', "input",   input,          " input dataset");
    ops >> Option('r', "rounds",  max_rounds,     " maximum number of iterations");
    ops >> Option('w', "weights", weighted,       " solve for and use weights");

    if (ops >> Present('h', "help", "show help"))
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "error = "      << norm_err_limit << " pt_dim = "      << pt_dim << " dom_dim = " << dom_dim <<
        "\ndegree = "   << degree         << " input pts = "   << ndomp  <<
        "\ninput = "    << input          << " max. rounds = " << max_rounds <<
        "\nweighted = " << weighted       << endl;
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
    d_args.weighted     = weighted;
    for (int i = 0; i < MAX_DIM; i++)
    {
        d_args.p[i]         = degree;
        d_args.ndom_pts[i]  = ndomp;
    }

    // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
    if (input == "sine")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]       = -4.0 * M_PI;
            d_args.max[i]       = 4.0  * M_PI;
        }
        d_args.s            = 1.0;              // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_sine_data(cp, d_args); });
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

    // S3D dataset
    if (input == "s3d")
    {
        d_args.ndom_pts[0]  = 704;
        d_args.ndom_pts[1]  = 540;
        d_args.ndom_pts[2]  = 550;
        strncpy(d_args.infile, "/Users/tpeterka/datasets/flame/6_small.xyz", sizeof(d_args.infile));
        if (dom_dim == 1)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_1d_slice_3d_vector_data(cp, d_args); });
        else if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, d_args); });
        else
        {
            fprintf(stderr, "S3D data only available in 1, 2, or 3d domain\n");
            exit(0);
        }
    }

    // nek5000 dataset
    if (input == "nek")
    {
        d_args.ndom_pts[0]  = 200;
        d_args.ndom_pts[1]  = 200;
        d_args.ndom_pts[2]  = 200;
        strncpy(d_args.infile, "/Users/tpeterka/datasets/nek5000/200x200x200/0.xyz", sizeof(d_args.infile));
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, d_args); });
        else
        {
            fprintf(stderr, "nek5000 data only available in 2 or 3d domain\n");
            exit(0);
        }
    }

    // rti dataset
    if (input == "rti")
    {
        d_args.ndom_pts[0]  = 288;
        d_args.ndom_pts[1]  = 512;
        d_args.ndom_pts[2]  = 512;
        strncpy(d_args.infile, "/Users/tpeterka/datasets/rti/dd07g_xxsmall_le.xyz", sizeof(d_args.infile));
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, d_args); });
        else
        {
            fprintf(stderr, "rti data only available in 2 or 3d domain\n");
            exit(0);
        }
    }

    // cesm dataset
    if (input == "cesm")
    {
        d_args.ndom_pts[0]  = 1800;
        d_args.ndom_pts[1]  = 3600;
        strncpy(d_args.infile, "/Users/tpeterka/datasets/CESM-ATM-tylor/1800x3600/FLDSC_1_1800_3600.dat", sizeof(d_args.infile));
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_scalar_data(cp, d_args); });
        else
        {
            fprintf(stderr, "cesm data only available in 2 or 3d domain\n");
            exit(0);
        }
    }

    // compute the MFA

    fprintf(stderr, "\nStarting adaptive encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->adaptive_encode_block(cp, norm_err_limit, max_rounds); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\nAdaptive encoding done.\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    double decode_time = MPI_Wtime();
#ifdef CURVE_PARAMS     // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, true); });
#else                   // range coordinate difference
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, true); });
#endif
    decode_time = MPI_Wtime() - decode_time;

    // debug: save knot span domains for comparing error with location in knot span
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->knot_span_domains(cp); });

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach(&Block<real_t>::print_block);
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
