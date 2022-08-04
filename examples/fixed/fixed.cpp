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
#include "example-setup.hpp"

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
    bool   scalar       = true;                 // dimension of each science variable (true == multiple scalar vars)
    int    geom_degree  = 1;                    // degree for geometry (same for all dims)
    int    vars_degree  = 4;                    // degree for science variables (same for all dims)
    int    ndomp        = 100;                  // input number of domain points (same for all dims)
    int    ntest        = 0;                    // number of input test points in each dim for analytical error tests
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl   = {11};                   // input number of control points for all science variables (default same for all dims)
    string input        = "sinc";               // input dataset
    int    weighted     = 1;                    // solve for and use weights (bool 0/1)
    real_t rot          = 0.0;                  // rotation angle in degrees
    real_t twist        = 0.0;                  // twist (waviness) of domain (0.0-1.0)
    real_t noise        = 0.0;                  // fraction of noise
    int    error        = 1;                    // decode all input points and check error (bool 0/1)
    string infile;                              // input file name
    int    structured   = 1;                    // input data format (bool 0/1)
    int    rand_seed    = -1;                   // seed to use for random data generation (-1 == no randomization)
    real_t regularization = 0;                  // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int     reg1and2 = 0;                       // flag for regularizer: 0 --> regularize only 2nd derivs. 1 --> regularize 1st and 2nd
    int    verbose      = 1;                    // MFA verbosity (0 = no extra output)
    bool   help         = false;                // show help


    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('l', "scalar",      scalar,     " flag for scalar or vector-valued science variables");
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
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // // minimal number of geometry control points if not specified
    // if (geom_nctrl == -1)
    //     geom_nctrl = geom_degree + 1;
    // if (vars_nctrl == -1)
    //     vars_nctrl = vars_degree + 1;

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "pt_dim = "         << pt_dim       << " dom_dim = "        << dom_dim      <<
        "\ngeom_degree = "  << geom_degree  << " vars_degree = "    << vars_degree  <<
        "\ninput pts = "    << ndomp        << " geom_ctrl pts = "  << geom_nctrl   <<
        "\nvars_ctrl_pts = ";
        for (int i = 0; i < vars_nctrl.size(); i++)
        {
            cerr << vars_nctrl[i] << " ";
        }
    cerr << "\ntest_points = "    << ntest        <<
        "\ninput = "        << input        << " noise = "          << noise        << 
        "\nstructured = "   << structured   << " scalar = " << boolalpha << scalar << endl;

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

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    // Different examples can reset this below
    vector<int> model_dims;
    if (scalar) // Set up (pt_dim - dom_dim) separate scalar variables
    {
        model_dims.assign(pt_dim - dom_dim + 1, 1);
        model_dims[0] = dom_dim;                        // index 0 == geometry
    }
    else    // Set up a single vector-valued variable
    {   
        model_dims = {dom_dim, pt_dim - dom_dim};
    }

    // Create empty info classes
    MFAInfo     mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);

    // // set MFA arguments
    // ModelInfo   geom_info(dom_dim, dom_dim);
    // ModelInfo   var_info(dom_dim, 1, vars_degree, vars_nctrl);
    // vector<ModelInfo> all_vars_info(pt_dim - dom_dim, var_info);
    // MFAInfo     mfa_info(dom_dim, 1, geom_info, all_vars_info);
    // mfa_info.weighted       = weighted;
    // mfa_info.regularization = regularization;
    // mfa_info.reg1and2       = reg1and2;

    // // set default args for diy foreach callback functions
    // DomainArgs d_args(dom_dim, model_dims);
    // d_args.n            = noise;
    // d_args.t            = twist;
    // d_args.multiblock   = false;
    // d_args.structured   = structured;
    // d_args.rand_seed    = rand_seed;
    // d_args.ndom_pts     = vector<int>(dom_dim, ndomp);
    
    // initialize input data
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                input, infile, ndomp, structured, rand_seed, rot, twist, noise,
                weighted, reg1and2, regularization, verbose, mfa_info, d_args);

    // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
    if (input == "sine")
    {
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     d_args.min[i]               = -4.0 * M_PI;
        //     d_args.max[i]               = 4.0  * M_PI;
        // }
        // for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
        //     d_args.s[i] = i + 1;                        // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, mfa_info, d_args); });
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     d_args.min[i]               = -4.0 * M_PI;
        //     d_args.max[i]               = 4.0  * M_PI;
        // }
        // for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
        //     d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range
        // d_args.r = rot * M_PI / 180.0;   // domain rotation angle in rads
        // d_args.t = twist;                // twist (waviness) of domain
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, mfa_info, d_args); });
    }

    // polysinc functions
    if (input == "psinc1" || input == "psinc2")
    {
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     d_args.min[i]               = -4.0 * M_PI;
        //     d_args.max[i]               = 4.0  * M_PI;
        // }
        // for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
        //     d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, mfa_info, d_args); });
    }

    // Marschner-Lobb function [M&L]: Marschner & Lobb, IEEE Vis 1994
    // only defined for 3d domain
    if (input == "ml")
    {
        // if (dom_dim != 3)
        // {
        //     fprintf(stderr, "Error: Marschner-Lobb function is only defined for a 3d domain.\n");
        //     exit(0);
        // }
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     d_args.min[i]               = -1.0;
        //     d_args.max[i]               = 1.0;
        // }
        // d_args.f[0] = 6.0;                  // f_M in the M&L paper
        // d_args.s[0] = 0.25;                 // alpha in the M&L paper
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, mfa_info, d_args); });
    }

    // f16 function
    if (input == "f16")
    {
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     d_args.min[i]               = -1.0;
        //     d_args.max[i]               = 1.0;
        // }
        // for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
        //     d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, mfa_info, d_args); });
    }

    // f17 function
    if (input == "f17")
    {
        // d_args.min[0] = 80.0;   d_args.max[0] = 100.0;
        // d_args.min[1] = 5.0;    d_args.max[1] = 10.0;
        // d_args.min[2] = 90.0;   d_args.max[2] = 93.0;
        // for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
        //     d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, mfa_info, d_args); });
    }

    // f18 function
    if (input == "f18")
    {
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     d_args.min[i]               = -0.95;
        //     d_args.max[i]               = 0.95;
        // }
        // for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
        //     d_args.s[i] = 1.0;                          // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, mfa_info, d_args); });
    }

    // S3D dataset
    if (input == "s3d")
    {
        // d_args.ndom_pts.resize(3);  // Hard-coded to full data set size
        // d_args.ndom_pts[0] = 704;
        // d_args.ndom_pts[1] = 540;
        // d_args.ndom_pts[2] = 550;
        // if (dom_dim >= 1) mfa_info.var_model_infos[0].nctrl_pts[0] = 140;
        // if (dom_dim >= 2) mfa_info.var_model_infos[0].nctrl_pts[1] = 108;
        // if (dom_dim >= 3) mfa_info.var_model_infos[0].nctrl_pts[2] = 110;
        // d_args.infile               = infile;
//         d_args.infile               = "/Users/tpeterka/datasets/flame/6_small.xyz";
        if (dom_dim == 1)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_1d_slice_3d_vector_data(cp, mfa_info, d_args); });
        else if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, mfa_info, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, mfa_info, d_args); });
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
//         d_args.ndom_pts.resize(3);  // Hard-coded to full data set size
//         d_args.ndom_pts[0] = 200;
//         d_args.ndom_pts[1] = 200;
//         d_args.ndom_pts[2] = 200;
//         for (int i = 0; i < dom_dim; i++)
//         {
//             mfa_info.var_model_infos[0].nctrl_pts[i] = 100;
//         }
//         d_args.infile = infile;
// //         d_args.infile = "/Users/tpeterka/datasets/nek5000/200x200x200/0.xyz";
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, mfa_info, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, mfa_info, d_args); });
        else
        {
            fprintf(stderr, "nek5000 data only available in 2 or 3d domain\n");
            exit(0);
        }
    }

    // rti dataset
    if (input == "rti")
    {
//         d_args.ndom_pts.resize(3);  // Hard-coded to full data set size
//         d_args.ndom_pts[0] = 288;
//         d_args.ndom_pts[1] = 512;
//         d_args.ndom_pts[2] = 512;
//         if (dom_dim >= 1) mfa_info.var_model_infos[0].nctrl_pts[0] = 72;
//         if (dom_dim >= 2) mfa_info.var_model_infos[0].nctrl_pts[1] = 128;
//         if (dom_dim >= 3) mfa_info.var_model_infos[0].nctrl_pts[2] = 128;
//         d_args.infile = infile;
// //         d_args.infile = "/Users/tpeterka/datasets/rti/dd07g_xxsmall_le.xyz";
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, mfa_info, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, mfa_info, d_args); });
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
        if (dom_dim == 2)
        {
    //         d_args.ndom_pts[0]          = 1800;
    //         d_args.ndom_pts[1]          = 3600;
    //         mfa_info.var_model_infos[0].nctrl_pts[0] = 300;
    //         mfa_info.var_model_infos[0].nctrl_pts[1] = 600;
    //         d_args.infile = infile;
    // //      d_args.infile = "/Users/tpeterka/datasets/CESM-ATM-tylor/1800x3600/FLDSC_1_1800_3600.dat";

            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_scalar_data(cp, mfa_info, d_args); });
        }
        else
        {
            fprintf(stderr, "cesm data only available in 2d domain\n");
            exit(0);
        }
    }

    // miranda dataset
    if (input == "miranda")
    {
        if (dom_dim == 3)
        {
    //         d_args.ndom_pts[0]          = 256;
    //         d_args.ndom_pts[1]          = 384;
    //         d_args.ndom_pts[2]          = 384;
    //         mfa_info.var_model_infos[0].nctrl_pts[0] = 256; // 192
    //         mfa_info.var_model_infos[0].nctrl_pts[1] = 384; // 288
    //         mfa_info.var_model_infos[0].nctrl_pts[2] = 384; // 288
    // //      d_args.infile = "/Users/tpeterka/datasets/miranda/SDRBENCH-Miranda-256x384x384/density.d64";
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_scalar_data<double>(cp, mfa_info, d_args); });
        }
        else
        {
            fprintf(stderr, "miranda data only available in 3d domain\n");
            exit(0);
        }
    }

     // tornado dataset
    if (input == "tornado")
    {
//         d_args.ndom_pts.resize(3);  // Hard-coded to full data set size
//         d_args.ndom_pts[0] = 128;
//         d_args.ndom_pts[1] = 128;
//         d_args.ndom_pts[2] = 128;
//         for (int i = 0; i < dom_dim; i++)
//         {
//             mfa_info.var_model_infos[0].nctrl_pts[i] = 100;
//         }
//         d_args.infile               = infile;
// //         d_args.infile               = "/Users/tpeterka/datasets/tornado/bov/1.vec.bov";

        if (dom_dim == 1)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_1d_slice_3d_vector_data(cp, mfa_info, d_args); });
        else if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, mfa_info, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, mfa_info, d_args); });
        else
        {
            fprintf(stderr, "tornado data only available in 1, 2, or 3d domain\n");
            exit(0);
        }
    }

    if (input == "edelta")
    {
        // int varid = 0;              // scalar quantity to read from file
        // int all_vars = 4;
        // int geom_dim = 3;
        // ModelInfo new_geometry(dom_dim, geom_dim);
        // mfa_info.addGeomInfo(new_geometry);

        // mfa_info.var_model_infos[0].p[0] = 2;
        // mfa_info.var_model_infos[0].p[0] = 2;
        // mfa_info.var_model_infos[0].nctrl_pts[0] = 130; // 57
        // mfa_info.var_model_infos[0].nctrl_pts[1] = 92;  // 40

        // d_args.tot_ndom_pts = 108822;
        // d_args.min[0] = -1.10315;
        // d_args.max[0] = 4.97625;
        // d_args.min[1] = -2.19155;
        // d_args.max[1] = 2.27595;
        // d_args.infile = infile;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_unstructured_data(cp, mfa_info, d_args); });
    }

    if (input == "climate")
    {
        // int varid = 0;
        // int all_vars = 1;
        // int geom_dim = 3;
        // ModelInfo new_geometry(dom_dim, geom_dim);
        // mfa_info.addGeomInfo(new_geometry);

        // d_args.tot_ndom_pts = 585765;
        // d_args.min[0] = -2.55;
        // d_args.max[0] = -1.449;
        // d_args.min[1] = -2.55;
        // d_args.max[1] = -1.449;
        // d_args.infile = infile;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_unstructured_data(cp, mfa_info, d_args); });
    }

    if (input == "nuclear")
    {
        // if (dom_dim != 3)
        // {
        //     cerr << "dom_dim must be 3 to run nuclear example" << endl;
        //     exit(1);
        // }
        // int varid = 1;
        // int all_vars = 7;
        // int geom_dim = 3;
        // ModelInfo new_geometry(dom_dim, geom_dim);
        // mfa_info.addGeomInfo(new_geometry);

        // d_args.tot_ndom_pts = 63048;
        // d_args.min[0] = 1.5662432;
        // d_args.max[0] = 30.433756;
        // d_args.min[1] = -7.5980764;
        // d_args.max[1] = 22.4019242;
        // d_args.min[2] = 10;
        // d_args.max[2] = 35;
        // mfa_info.var_model_infos[0].nctrl_pts[2] = 15; // reduce number of control points in z-direction
        // d_args.infile = infile;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_unstructured_data(cp, mfa_info, d_args); });
    }

    // compute the MFA
    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, mfa_info); });
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
    diy::io::write_blocks("approx.mfa", world, master);
}
