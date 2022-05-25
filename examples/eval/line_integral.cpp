//--------------------------------------------------------------
// example of computing line integrals from an encoded MFA
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
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
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    int    vars_nctrl   = 11;                   // input number of control points for all science variables (same for all dims)
    string input        = "sinc";               // input dataset
    int    weighted     = 0;                    // solve for and use weights (bool 0/1)
    string infile;                              // input file name
    int    structured   = 1;                    // input data format (bool 0/1)
    int    rand_seed    = -1;                   // seed to use for random data generation (-1 == no randomization)
    float  regularization = 0;                  // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int    reg1and2     = 0;                       // flag for regularizer: 0 --> regularize only 2nd derivs. 1 --> regularize 1st and 2nd
    bool   help         = false;                // show help


    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");

    int n_alpha = 0;
    int n_rho = 0;
    int n_samples = 0;
    int v_alpha = 0;
    int v_rho = 0;
    int v_samples = 0;
    int num_ints = 0;
    ops >> opts::Option('z', "n_alpha", n_alpha, " number of rotational samples for line integration");
    ops >> opts::Option('z', "n_rho", n_rho, " number of samples in offset direction for line integration");
    ops >> opts::Option('z', "n_samples", n_samples, " number of samples along ray for line integration");
    ops >> opts::Option('z', "v_alpha", v_alpha, " number of rotational control points for line integration");
    ops >> opts::Option('z', "v_rho", v_rho, " number of control points in offset direction for line integration");
    ops >> opts::Option('z', "v_samples", v_samples, " number of control points along ray for line integration");
    ops >> opts::Option('z', "num_ints", num_ints, " number of random line integrals to compute");


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
        "\nvars_ctrl_pts = "<< vars_nctrl   << 
        "\ninput = "        << input        << 
        "\nstructured = "   << structured   << endl;

#ifdef CURVE_PARAMS
    cerr << "Curve parametrization not supported. Exiting." << endl;
    exit(0);
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
    weighted = 0;
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
    d_args.n            = 0;
    d_args.t            = 0;
    d_args.multiblock   = false;
    d_args.verbose      = 1;
    d_args.structured   = structured;
    d_args.rand_seed    = rand_seed;
    d_args.regularization   = regularization;
    d_args.reg1and2     = reg1and2;
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

    // sinc and polysinc functions
    if (input == "sinc" || input == "psinc1" || input == "psinc2")
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
        d_args.ndom_pts.resize(2);
        d_args.vars_nctrl_pts[0].resize(2);
        d_args.ndom_pts[0]          = 1800;
        d_args.ndom_pts[1]          = 3600;
        d_args.vars_nctrl_pts[0][0] = 300;
        d_args.vars_nctrl_pts[0][1] = 600;
        d_args.infile = infile;
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_scalar_data(cp, d_args); });
        else
        {
            fprintf(stderr, "cesm data only available in 2d domain\n");
            exit(0);
        }
    }

    // miranda dataset
    if (input == "miranda")
    {
        d_args.ndom_pts.resize(3);
        d_args.vars_nctrl_pts[0].resize(3);
        d_args.ndom_pts[0]          = 256;
        d_args.ndom_pts[1]          = 384;
        d_args.ndom_pts[2]          = 384;
        d_args.vars_nctrl_pts[0][0] = 256;  // 192;
        d_args.vars_nctrl_pts[0][1] = 384;  // 288;
        d_args.vars_nctrl_pts[0][2] = 384;  // 288;
        d_args.infile = infile;
//      d_args.infile = "/Users/tpeterka/datasets/miranda/SDRBENCH-Miranda-256x384x384/density.d64";
        if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_scalar_data<double>(cp, d_args); });
        else
        {
            fprintf(stderr, "miranda data only available in 3d domain\n");
            exit(0);
        }
    }

     // tornado dataset
    if (input == "tornado")
    {
        d_args.ndom_pts.resize(3);
        d_args.vars_nctrl_pts[0].resize(3);
        d_args.ndom_pts[0]          = 128;
        d_args.ndom_pts[1]          = 128;
        d_args.ndom_pts[2]          = 128;
        d_args.vars_nctrl_pts[0][0] = 100;
        d_args.vars_nctrl_pts[0][1] = 100;
        d_args.vars_nctrl_pts[0][2] = 100;
        d_args.infile               = infile;
//         d_args.infile               = "/Users/tpeterka/datasets/tornado/bov/1.vec.bov";

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
            fprintf(stderr, "tornado data only available in 1, 2, or 3d domain\n");
            exit(0);
        }
    }

    if (input == "edelta")
    {
        int varid = 0;              // scalar quantity to read from file
        int all_vars = 4;
        int geom_dim = 3;
        d_args.vars_p[0][0] = 2;
        d_args.vars_p[0][1] = 2;
        // d_args.vars_nctrl_pts[0][0] = 57;
        // d_args.vars_nctrl_pts[0][1] = 40;
        d_args.vars_nctrl_pts[0][0] = 130;
        d_args.vars_nctrl_pts[0][1] = 92;
        d_args.tot_ndom_pts = 108822;
        d_args.min[0] = -1.10315;
        d_args.max[0] = 4.97625;
        d_args.min[1] = -2.19155;
        d_args.max[1] = 2.27595;
        d_args.infile = infile;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_unstructured_data(cp, d_args, varid, all_vars, geom_dim); });
    }

    if (input == "climate")
    {
        int varid = 0;
        int all_vars = 1;
        int geom_dim = 3;
        d_args.tot_ndom_pts = 585765;
        d_args.min[0] = -2.55;
        d_args.max[0] = -1.449;
        d_args.min[1] = -2.55;
        d_args.max[1] = -1.449;
        d_args.infile = infile;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_unstructured_data(cp, d_args, varid, all_vars, geom_dim); });
    }

    if (input == "nuclear")
    {
        if (dom_dim != 3)
        {
            cerr << "dom_dim must be 3 to run nuclear example" << endl;
            exit(1);
        }
        int varid = 1;
        int all_vars = 7;
        int geom_dim = 3;
        d_args.tot_ndom_pts = 63048;
        d_args.min[0] = 1.5662432;
        d_args.max[0] = 30.433756;
        d_args.min[1] = -7.5980764;
        d_args.max[1] = 22.4019242;
        d_args.min[2] = 10;
        d_args.max[2] = 35;
        d_args.vars_nctrl_pts[0][2] = 15; // reduce number of control points in z-direction
        d_args.infile = infile;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_unstructured_data(cp, d_args, varid, all_vars, geom_dim); });
    }

    // compute the MFA
    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, d_args); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    double decode_time = MPI_Wtime();
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    bool saved_basis = structured; // TODO: basis functions are currently only saved during encoding of structured data
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { 
                b->range_error(cp, 1, true, saved_basis);
                b->print_block(cp, true);

                // Assumes one scalar science variable. Used for relative error metric
                real_t extent = b->input->domain.col(dom_dim).maxCoeff() - b->input->domain.col(dom_dim).minCoeff();

                b->create_ray_model(cp, d_args, 1, n_samples, n_rho, n_alpha, v_samples, v_rho, v_alpha);

                real_t result = 0;
                VectorX<real_t> start_pt(2), end_pt(2);

                // horizontal line where function is identically 0
                // = 0.0
                start_pt(0) = -3*M_PI; start_pt(1) = 0;
                end_pt(0) = 3*M_PI; end_pt(1) = 0;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(-3pi, 0)---(3pi, 0):       " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "error:                      " << result << endl << endl;

                // vertical line
                // = 0.0
                start_pt(0) = M_PI/2; start_pt(1) = -2*M_PI;
                end_pt(0) = M_PI/2; end_pt(1) = 2*M_PI;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(pi/2, -2pi)---(pi/2, 2pi): " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "error:                      " << result << endl << endl;
                
                // horizontal line
                // = 2.0
                start_pt(0) = 0; start_pt(1) = M_PI/2;
                end_pt(0) = M_PI; end_pt(1) = M_PI/2;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(0, pi/2)---(pi, pi/2):     " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "relative error:             " << abs((result-2)/2) << endl << endl;

                // line y=x
                // = 5.75864344326
                start_pt(0) = 0, start_pt(1) = 0;
                end_pt(0) = 8, end_pt(1) = 8;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(0, 0)---(8, 8):            " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "relative error:             " << abs((result-5.75864344326)/5.75864344326) << endl << endl;

                // "arbitrary" line
                // = 1.2198958397433
                start_pt(0) = -2; start_pt(1) = -4;
                end_pt(0) = 3; end_pt(1) = 11;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(-2, -4)---(3, 11):         " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "relative error:             " << abs((result-1.2198958397433)/1.2198958397433) << endl << endl;

                std::vector<real_t> ierrs_abs;
                std::vector<real_t> ierrs_rel;
                std::random_device dev;
                std::mt19937 rng(dev());
                real_t x0, x1, y0, y1;
                real_t ierror_abs=0, ierror_rel=0, actual=0, rms_abs=0, rms_rel=0, avg_abs=0, avg_rel=0, len=0;
                std::uniform_real_distribution<double> dist(0,1); 
                for (int i = 0; i < num_ints; i++)
                {
                    x0 = dist(rng)* 8*M_PI - 4*M_PI;
                    y0 = dist(rng)* 8*M_PI - 4*M_PI;
                    x1 = dist(rng)* 8*M_PI - 4*M_PI;
                    y1 = dist(rng)* 8*M_PI - 4*M_PI;
                    start_pt(0) = x0;
                    start_pt(1) = y0;
                    end_pt(0) = x1;
                    end_pt(1) = y1;
                    len = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
                    result = b->integrate_ray(cp, start_pt, end_pt, 1) / len;   // normalize by segment length
                    actual = sintest(start_pt, end_pt) / len;                        // normalize by segment length
                    ierror_abs = abs(result - actual);
                    ierror_rel = ierror_abs/extent;
                    ierrs_abs.push_back(ierror_abs);
                    ierrs_rel.push_back(ierror_rel);
                    // cerr << "(" << x0 << ", " << y0 << ")---(" << x1 << ", " << y1 << "):  " << endl;
                    // cerr << "  Result: " << setprecision(6) << result << endl;
                    // cerr << "  Actual: " << setprecision(6) << actual << endl;
                    // cerr << "  Length:    " << setprecision(3) << len << endl;
                    // cerr << "  Error (abs):  " << setprecision(6) << ierror_abs << endl;
                    // cerr << "  Error (rel):  " << setprecision(6) << ierror_rel << endl;
                }

                cerr << "\nComputed " << num_ints << " random line integrals." << endl;
                cerr << "  Max error (abs): " << setprecision(6) << *max_element(ierrs_abs.begin(), ierrs_abs.end()) << 
                            "\t" << "Max error (rel): " << *max_element(ierrs_rel.begin(), ierrs_rel.end()) << endl;
                cerr << "  Min error (abs): " << setprecision(6) << *min_element(ierrs_abs.begin(), ierrs_abs.end()) << 
                            "\t" << "Min error (rel): " << *min_element(ierrs_rel.begin(), ierrs_rel.end()) << endl;
                for (int j = 0; j < ierrs_abs.size(); j++)
                {
                    rms_abs += ierrs_abs[j] * ierrs_abs[j];
                    rms_rel += ierrs_rel[j] * ierrs_rel[j];
                    avg_abs += ierrs_abs[j];
                    avg_rel += ierrs_rel[j];
                }
                rms_abs = rms_abs/ierrs_abs.size();
                rms_abs = sqrt(rms_abs);
                rms_rel = rms_rel/ierrs_rel.size();
                rms_rel = sqrt(rms_rel);
                avg_abs = avg_abs/ierrs_abs.size();
                avg_rel = avg_rel/ierrs_rel.size();
                cerr << "  Avg error (abs): " << setprecision(6) << avg_abs << "\t" << "Avg error (rel): " << avg_rel << endl;
                cerr << "  RMS error (abs): " << setprecision(6) << rms_abs << "\t" << "RMS error (rel): " << rms_rel << endl;

                ofstream errfile_abs, errfile_rel;
                errfile_abs.open("li_errors_abs.txt");
                errfile_rel.open("li_errors_rel.txt");
                for (int i = 0; i < ierrs_abs.size(); i++)
                {
                    errfile_abs << ierrs_abs[i] << endl;
                    errfile_rel << ierrs_rel[i] << endl;
                }
                errfile_abs.close();
                errfile_rel.close();

                ofstream segmenterrfile;
                segmenterrfile.open("seg_errors.txt");
                int test_n_alpha = 150;
                int test_n_rho = 150;
                real_t r_lim = b->bounds_maxs(1);   // WARNING TODO: make r_lim query-able in RayMFA class
                for (int i = 0; i < test_n_alpha; i++)
                {
                    for (int j = 0; j < test_n_rho; j++)
                    {
                        real_t alpha = 3.14159265 / (test_n_alpha-1) * i;
                        real_t rho = r_lim*2 / (test_n_rho-1) * j - r_lim;
                        real_t x0, x1, y0, y1;   // end points of full line

                        b->get_box_intersections(alpha, rho, x0, y0, x1, y1, b->box_mins, b->box_maxs);
                        if (x0==0 && y0==0 && x1==0 && y1==0)
                        {
                            segmenterrfile << alpha << " " << rho << " " << 0 << endl;
                        }
                        else
                        {
                            real_t length = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
                            start_pt(0) = x0; 
                            start_pt(1) = y0;
                            end_pt(0) = x1;
                            end_pt(1) = y1;

                            real_t test_result = b->integrate_ray(cp, start_pt, end_pt, 1) / length;   // normalize by segment length
                            real_t test_actual = sintest(start_pt, end_pt) / length;

                            real_t e_abs = abs(test_result - test_actual);
                            real_t e_rel = e_abs/extent;

                            segmenterrfile << alpha << " " << rho << " " << e_rel << endl;
                        }
                        
                    }
                }
                segmenterrfile.close();
            });
    decode_time = MPI_Wtime() - decode_time;

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, true); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);
}
