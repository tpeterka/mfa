//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points and a
// single block in a split model w/ one model containing geometry and other model science variables
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

#include "opts.h"

#include "block.hpp"

using namespace std;

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
    int    pt_dim       = 3;                    // dimension of input points
    int    dom_dim      = 2;                    // dimension of domain (<= pt_dim)
    int    geom_degree  = 1;                    // degree for geometry (same for all dims)
    int    vars_degree  = 4;                    // degree for science variables (same for all dims)
    int    ndomp        = 100;                  // input number of domain points (same for all dims)
    int    ntest        = 0;                    // number of input test points in each dim for analytical error tests
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    int    vars_nctrl   = 11;                   // input number of control points for all science variables (same for all dims)
    string input        = "sinc";               // input dataset
    int    weighted     = 1;                    // solve for and use weights (bool 0/1)
    real_t rot          = 0.0;                  // rotation angle in degrees
    real_t twist        = 0.0;                  // twist (waviness) of domain (0.0-1.0)
    real_t noise        = 0.0;                  // fraction of noise
    int    error        = 1;                    // decode all input points and check error (bool 0/1)
    string infile;                              // input file name
    int    structured   = 1;                    // input data format (bool 0/1)
    int    rand_seed    = -1;                   // seed to use for random data generation (-1 == no randomization)
    bool   help         = false;                // show help


    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('a', "ntest",       ntest,      " number of test points in each dimension of domain (for analytical error calculation)");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('r', "rotate",      rot,        " rotation angle of domain in degrees");
    ops >> opts::Option('t', "twist",       twist,      " twist (waviness) of domain (0.0-1.0)");
    ops >> opts::Option('s', "noise",       noise,      " fraction of noise (0.0 - 1.0)");
    ops >> opts::Option('c', "error",       error,      " decode entire error field (default=true)");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // minimal number of geometry control points if not specified
    if (geom_nctrl == -1)
        geom_nctrl = geom_degree + 1;
    if (vars_nctrl == -1)
        vars_nctrl = vars_degree + 1;

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "pt_dim = "         << pt_dim       << " dom_dim = "        << dom_dim      <<
        "\ngeom_degree = "  << geom_degree  << " vars_degree = "    << vars_degree  <<
        "\ninput pts = "    << ndomp        << " geom_ctrl pts = "  << geom_nctrl   <<
        "\nvars_ctrl_pts = "<< vars_nctrl   << " test_points = "    << ntest        <<
        "\ninput = "        << input        << " noise = "          << noise        << 
        "\nstructured = "   << structured   << endl;

#ifdef CURVE_PARAMS
    cerr << "parameterization method = curve" << endl;
#else
    cerr << "parameterization method = domain" << endl;
#endif
#ifdef MFA_TBB
    cerr << "threading: TBB" << endl;
#endif
#ifdef MFA_KOKKOS
    cerr << "threading: Kokkos" << endl;
#endif
#ifdef MFA_SYCL
    cerr << "threading: SYCL" << endl;
#endif
#ifdef MFA_SERIAL
    cerr << "threading: serial" << endl;
#endif
#ifdef MFA_NO_WEIGHTS
    cerr << "weighted = 0" << endl;
#else
    cerr << "weighted = " << weighted << endl;
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

    // even though this is a single-block example, we want diy to do a proper decomposition with a link
    // so that everything works downstream (reading file with links, e.g.)
    // therefore, set some dummy global domain bounds and decompose the domain
    Bounds<real_t> dom_bounds(dom_dim);
    for (int i = 0; i < dom_dim; ++i)
    {
        dom_bounds.min[i] = 0.0;
        dom_bounds.max[i] = 1.0;
    }
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // set default args for diy foreach callback functions
    DomainArgs d_args(dom_dim, pt_dim);
    d_args.weighted     = weighted;
    d_args.n            = noise;
    d_args.multiblock   = false;
    d_args.verbose      = 1;
    d_args.structured   = structured;
    d_args.rand_seed    = rand_seed;
    for (int i = 0; i < pt_dim - dom_dim; i++)
        d_args.f[i] = 1.0;
    for (int i = 0; i < dom_dim; i++)
    {
        d_args.geom_p[i]            = geom_degree;
        d_args.vars_p[0][i]         = vars_degree;      // assuming one science variable, vars_p[0]
        d_args.ndom_pts[i]          = ndomp;
        d_args.geom_nctrl_pts[i]    = geom_nctrl;
        d_args.vars_nctrl_pts[0][i] = vars_nctrl;       // assuming one science variable, vars_nctrl_pts[0]
    }

    // initialize input data

    // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
    if (input == "sine")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = i + 1;                        // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range
        d_args.r = rot * M_PI / 180.0;   // domain rotation angle in rads
        d_args.t = twist;                // twist (waviness) of domain
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // polysinc functions
    if (input == "psinc1" || input == "psinc2")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // Marschner-Lobb function [M&L]: Marschner & Lobb, IEEE Vis 1994
    // only defined for 3d domain
    if (input == "ml")
    {
        if (dom_dim != 3)
        {
            fprintf(stderr, "Error: Marschner-Lobb function is only defined for a 3d domain.\n");
            exit(0);
        }
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -1.0;
            d_args.max[i]               = 1.0;
        }
        d_args.f[0] = 6.0;                  // f_M in the M&L paper
        d_args.s[0] = 0.25;                 // alpha in the M&L paper
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // f16 function
    if (input == "f16")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -1.0;
            d_args.max[i]               = 1.0;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // f17 function
    if (input == "f17")
    {
        d_args.min[0] = 80.0;   d_args.max[0] = 100.0;
        d_args.min[1] = 5.0;    d_args.max[1] = 10.0;
        d_args.min[2] = 90.0;   d_args.max[2] = 93.0;
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // f18 function
    if (input == "f18")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -0.95;
            d_args.max[i]               = 0.95;
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // S3D dataset
    if (input == "s3d")
    {
        d_args.ndom_pts.resize(3);
        d_args.vars_nctrl_pts[0].resize(3);
        d_args.ndom_pts[0]          = 704;
        d_args.ndom_pts[1]          = 540;
        d_args.ndom_pts[2]          = 550;
        d_args.vars_nctrl_pts[0][0] = 140;
        d_args.vars_nctrl_pts[0][1] = 108;
        d_args.vars_nctrl_pts[0][2] = 110;
        d_args.infile               = infile;
//         d_args.infile               = "/Users/tpeterka/datasets/flame/6_small.xyz";
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
            fprintf(stderr, "S3D data only available in 2 or 3d domain\n");
            exit(0);
        }

        // for testing, hard-code a subset of a 3d domain, 1/2 the size in each dim and centered
        // in this case, actual size is just under 1/2 size in each dim to satisfy DIY's (MPI's)
        // limitation on the size of a file write
        // (MPI uses int for the size, and DIY as yet does not chunk writes into smaller sizes)
//         d_args.starts[0]            = 125;
//         d_args.starts[1]            = 50;
//         d_args.starts[2]            = 125;
//         d_args.ndom_pts[0]          = 350;
//         d_args.ndom_pts[1]          = 250;
//         d_args.ndom_pts[2]          = 250;
//         d_args.full_dom_pts[0]      = 704;
//         d_args.full_dom_pts[1]      = 540;
//         d_args.full_dom_pts[2]      = 550;
//         master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                 { b->read_3d_subset_3d_vector_data(cp, d_args); });
    }

    // nek5000 dataset
    if (input == "nek")
    {   
        d_args.ndom_pts.resize(3);
        for (int i = 0; i < 3; i++)
            d_args.ndom_pts[i]          = 200;
        for (int i = 0; i < dom_dim; i++)
            d_args.vars_nctrl_pts[0][i] = 100;

        d_args.infile = infile;
//         d_args.infile = "/Users/tpeterka/datasets/nek5000/200x200x200/0.xyz";
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
        d_args.ndom_pts.resize(3);
        d_args.vars_nctrl_pts[0].resize(3);
        d_args.ndom_pts[0]  = 288;
        d_args.ndom_pts[1]  = 512;
        d_args.ndom_pts[2]  = 512;
        d_args.vars_nctrl_pts[0][0] = 72;
        d_args.vars_nctrl_pts[0][1] = 128;
        d_args.vars_nctrl_pts[0][2] = 128;
        d_args.infile = infile;
//         d_args.infile = "/Users/tpeterka/datasets/rti/dd07g_xxsmall_le.xyz";
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

//         // for testing, hard-code a subset of a 3d domain, 1/2 the size in each dim and centered
//         d_args.starts[0]        = 72;
//         d_args.starts[1]        = 128;
//         d_args.starts[2]        = 128;
//         d_args.ndom_pts[0]      = 144;
//         d_args.ndom_pts[1]      = 256;
//         d_args.ndom_pts[2]      = 256;
//         d_args.full_dom_pts[0]  = 288;
//         d_args.full_dom_pts[1]  = 512;
//         d_args.full_dom_pts[2]  = 512;
//         master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//                 { b->read_3d_subset_3d_vector_data(cp, d_args); });
    }

    // cesm dataset
    if (input == "cesm")
    {
        d_args.ndom_pts.resize(2);
        d_args.vars_nctrl_pts[0].resize(2);
        d_args.ndom_pts[0]          = 1800;
        d_args.ndom_pts[1]          = 3600;
        d_args.vars_nctrl_pts[0][0] = 180;
        d_args.vars_nctrl_pts[0][1] = 360;
        d_args.infile = infile;
//      d_args.infile = "/Users/tpeterka/datasets/CESM-ATM-tylor/1800x3600/FLDSC_1_1800_3600.dat";
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_scalar_data(cp, d_args); });
        else
        {
            fprintf(stderr, "cesm data only available in 2d domain\n");
            exit(0);
        }
    }

    // write initial points
    diy::io::write_blocks("initial.out", world, master);

    // compute the MFA

    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, d_args); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    double decode_time = MPI_Wtime();
    if (error)
    {
        fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#ifdef CURVE_PARAMS     // normal distance
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->error(cp, 1, true); });
#else                   // range coordinate difference
        bool saved_basis = structured; // TODO: basis functions are currently only saved during encoding of structured data
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { 
                    b->range_error(cp, 1, true, saved_basis);
                    // b->create_ray_model(cp, d_args);
                     });
#endif
        decode_time = MPI_Wtime() - decode_time;
    }

    // debug: write original and approximated data for reading into z-checker
    // only for one block (one file name used, ie, last block will overwrite earlier ones)
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//             { b->write_raw(cp); });

    // debug: save knot span domains for comparing error with location in knot span
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//             { b->knot_span_domains(cp); });

    // compute the norms of analytical errors synthetic function w/o noise at different domain points than the input
    if (ntest > 0)
    {
        real_t L1, L2, Linf;                                // L-1, 2, infinity norms

        for (int i = 0; i < dom_dim; i++)
            d_args.ndom_pts[i] = ntest;

        vector<vec3d> unused;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->analytical_error(cp, input, L1, L2, Linf, d_args, false, unused, NULL, unused, NULL); });

        // print analytical errors
        fprintf(stderr, "\n------ Analytical error norms -------\n");
        fprintf(stderr, "L-1        norm = %e\n", L1);
        fprintf(stderr, "L-2        norm = %e\n", L2);
        fprintf(stderr, "L-infinity norm = %e\n", Linf);
        fprintf(stderr, "-------------------------------------\n\n");
    }

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, error); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    if (error)
        fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
