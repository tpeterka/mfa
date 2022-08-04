//--------------------------------------------------------------
// Helper functions to set up pre-defined examples
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef MFA_EX_SETUP_HPP
#define MFA_EX_SETUP_HPP

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include "block.hpp"

using namespace std;

// void set_args(  int&            geom_degree,
//                 int&            geom_nctrl,
//                 int&            vars_degree,
//                 int&            vars_nctrl,
//                 vector<int>&    ndom_pts,
//                 vector<real_t>& dom_mins,
//                 vector<real_t>& dom_maxs,
//                 vector<real_t>& s_vec,
//                 vector<real_t>& f_vec,
//                 vector<int>&    starts,
//                 vector<int>&    full_dom_pts
//                 )


// TEMP Code snippet for future use
//
// for (int i = 0; i < dom_dim; i++)
//     {
//         d_args.geom_p[i]            = geom_p[i];
//         d_args.geom_nctrl_pts[i]    = geom_nctrl_pts[i];

//         for (int k = 0; k < d_args.nvars; k++)
//         {
//             d_args.vars_p[k][i]         = vars_p[k][i];      // assuming one science variable, vars_p[0]
//             d_args.vars_nctrl_pts[k][i] = vars_nctrl_pts[k][i];       // assuming one science variable, vars_nctrl_pts[0]
//         }
//     }
//     d_args.weighted     = weighted;
//     d_args.reg1and2     = reg1and2;
//     d_args.regularization   = regularization;
//     d_args.verbose      = 1;

//     // Data set parameters
//     for (int i = 0; i < dom_dim; i++)
//     {
//         d_args.min[i]               = dom_mins[i];
//         d_args.max[i]               = dom_maxs[i];
//         d_args.ndom_pts[i]          = ndom_pts[i];
//         d_args.starts[i]            = starts[i];
//         d_args.full_dom_pts[i]      = full_dom_pts[i];
//     }
//     d_args.tot_ndom_pts = tot_ndom_pts;
//     d_args.r            = rot * M_PI / 180;          // TODO change to degrees
//     d_args.t            = twist;
//     d_args.n            = noise;
//     for (int i = 0; i < pt_dim - dom_dim; i++)
//     {
//         d_args.s[i]     = scaling[i];
//         d_args.f        = freq[i];
//     }
//     d_args.infile       = infile;
//     d_args.multiblock   = false;
//     d_args.structured   = structured;
//     d_args.rand_seed    = rand_seed;
/////////////




    // Sets the degree and number of control points 
    // For geom and each var, degree is same in each domain dimension
    // For geom, # ctrl points is the same in each domain dimension
    // For each var, # ctrl points varies per domain dimension, but is the same for all of the vars
    void set_mfa_info(int dom_dim, vector<int> model_dims, 
                        int geom_degree, int geom_nctrl,
                        int vars_degree, vector<int> vars_nctrl,
                        MFAInfo& mfa_info)
    {
        // Clear any existing data in mfa_info
        mfa_info.reset();

        int nvars       = model_dims.size() - 1;
        int geom_dim    = model_dims[0];

        // If only one value for vars_nctrl was parsed, assume it applies to all dims
        if (vars_nctrl.size() == 1 & dom_dim > 1)
        {
            vars_nctrl = vector<int>(dom_dim, vars_nctrl[0]);
        }

        // Minimal necessary control points
        if (geom_nctrl == -1) geom_nctrl = geom_degree + 1;
        for (int i = 0; i < vars_nctrl.size(); i++)
        {
            if (vars_nctrl[i] == -1) vars_nctrl[i] = vars_degree + 1;
        }

        ModelInfo geom_info(dom_dim, geom_dim, geom_degree, geom_nctrl);
        mfa_info.addGeomInfo(geom_info);

        for (int k = 0; k < nvars; k++)
        {
            ModelInfo var_info(dom_dim, model_dims[k], vars_degree, vars_nctrl);
            mfa_info.addVarInfo(var_info);
        }


        // geom_p.resize(dom_dim);
        // geom_nctrl_pts.resize(dom_dim);
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     geom_p[i] = geom_degree;
        //     geom_nctrl_pts[i] = geom_nctrl;
        // }

        // vars_p.resize(nvars);
        // vars_nctrl_pts.resize(nvars);
        // for (int j = 0; j < nvars; j++)
        // {
        //     vars_p[j].resize(dom_dim);
        //     vars_nctrl_pts[j].resize(dom_dim);
        //     for (int i = 0; i < dom_dim; i++)
        //     {
        //         vars_p[j][i] = vars_degree;
        //         vars_nctrl_pts[j][i] = vars_nctrl;
        //     }
        // }
    }

    // Currently for single block examples only
    void setup_args( int dom_dim, int pt_dim, vector<int> model_dims,
                        int geom_degree, int geom_nctrl, int vars_degree, vector<int> vars_nctrl,
                        string input, string infile, int ndomp,
                        int structured, int rand_seed, real_t rot, real_t twist, real_t noise,
                        int weighted, int reg1and2, real_t regularization, int verbose,
                        MFAInfo& mfa_info, DomainArgs& d_args)
    {


        // pt_dim
        // dom_dim
        // --ndomp
        // input
        // --weighted
        // --rot
        // --twist
        // --noise
        // --infile
        // --structured
        // --rand_seed
        // --regularization
        // --reg1and2

        assert(vars_nctrl.size() == dom_dim);

        // Set basic info for DomainArgs
        d_args.updateModelDims(model_dims);
        d_args.multiblock   = false;
        d_args.r            = rot * M_PI / 180;
        d_args.t            = twist;
        d_args.n            = noise;
        d_args.infile       = infile;
        d_args.structured   = structured;
        d_args.rand_seed    = rand_seed;

        // Specify size, location of domain 
        d_args.ndom_pts     = vector<int>(dom_dim, ndomp);
        d_args.full_dom_pts = vector<int>(dom_dim, ndomp);
        d_args.starts       = vector<int>(dom_dim, 0);
        d_args.tot_ndom_pts = 1;
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.tot_ndom_pts *= ndomp;
        }

        // Set default extents of physical domain
        d_args.min.assign(dom_dim, 0.0);
        d_args.max.assign(dom_dim, 1.0);

        // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
        if (input == "sine")
        {
            d_args.min.assign(dom_dim, -4.0 * M_PI);
            d_args.max.assign(dom_dim,  4.0 * M_PI);
            for (int i = 0; i < d_args.model_dims.size()-1; i++)      // for all science variables
                d_args.s[i] = i + 1;                        // scaling factor on range

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //         { b->generate_analytical_data(cp, input, d_args); });
        }

        // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
        if (input == "sinc")
        {
            d_args.min.assign(dom_dim, -4.0 * M_PI);
            d_args.max.assign(dom_dim,  4.0 * M_PI);
            for (int i = 0; i < d_args.model_dims.size()-1; i++)      // for all science variables
                d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //         { b->generate_analytical_data(cp, input, d_args); });
        }

        // polysinc functions
        if (input == "psinc1" || input == "psinc2" || input == "psinc3")
        {
            d_args.min.assign(dom_dim, -4.0 * M_PI);
            d_args.max.assign(dom_dim,  4.0 * M_PI);
            for (int i = 0; i < d_args.model_dims.size()-1; i++)      // for all science variables
                d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //         { b->generate_analytical_data(cp, input, d_args); });
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

            d_args.min.assign(dom_dim, -1.0);
            d_args.max.assign(dom_dim,  1.0);
            d_args.f[0] = 6.0;                  // f_M in the M&L paper
            d_args.s[0] = 0.25;                 // alpha in the M&L paper

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //         { b->generate_analytical_data(cp, input, d_args); });
        }

        // f16 function
        if (input == "f16")
        {
            if (dom_dim != 2)
            {
                fprintf(stderr, "Error: f16 function is only defined for a 2d domain.\n");
                exit(0);
            }

            d_args.min = {-1, -1};
            d_args.max = { 1,  1};

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //         { b->generate_analytical_data(cp, input, d_args); });
        }

        // f17 function
        if (input == "f17")
        {
            if (dom_dim != 3)
            {
                fprintf(stderr, "Error: f17 function is only defined for a 3d domain.\n");
                exit(0);
            }

            d_args.min = {80,   5, 90};
            d_args.max = {100, 10, 93};

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //         { b->generate_analytical_data(cp, input, d_args); });
        }

        // f18 function
        if (input == "f18")
        {
            if (dom_dim != 4)
            {
                fprintf(stderr, "Error: f18 function is only defined for a 4d domain.\n");
                exit(0);
            }

            d_args.min = {-0.95, -0.95, -0.95, -0.95};
            d_args.max = { 0.95,  0.95,  0.95,  0.95};

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //         { b->generate_analytical_data(cp, input, d_args); });
        }

        // S3D dataset:  flame/6_small.xyz
        if (input == "s3d")
        {
            d_args.full_dom_pts = {704, 540, 550};  // Hard-coded to full data set size
            d_args.ndom_pts = d_args.full_dom_pts;

            if (dom_dim >= 1) vars_nctrl[0] = 140;
            if (dom_dim >= 2) vars_nctrl[1] = 108;
            if (dom_dim >= 3) vars_nctrl[2] = 110;

            // for testing, hard-code a subset of a 3d domain, 1/2 the size in each dim and centered
            // in this case, actual size is just under 1/2 size in each dim to satisfy DIY's (MPI's)
            // limitation on the size of a file write
            // (MPI uses int for the size, and DIY as yet does not chunk writes into smaller sizes)
            // d_args.ndom_pts = {350, 2250, 250};
            // d_args.starts   = {125, 50, 125};

            // if (dom_dim == 1)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_1d_slice_3d_vector_data(cp, d_args); });
            // else if (dom_dim == 2)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_2d_slice_3d_vector_data(cp, d_args); });
            // else if (dom_dim == 3)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_3d_vector_data(cp, d_args); });
            // else
            // {
            //     fprintf(stderr, "S3D data only available in 2 or 3d domain\n");
            //     exit(0);
            // }


    //         master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
    //                 { b->read_3d_subset_3d_vector_data(cp, d_args); });
        }

        // nek5000 dataset:  nek5000/200x200x200/0.xyz
        if (input == "nek")
        {
            d_args.full_dom_pts = {200, 200, 200};      // Hard-coded to full data set size
            d_args.ndom_pts = d_args.full_dom_pts;
            vars_nctrl.assign(dom_dim, 100);


            // if (dom_dim == 2)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_2d_slice_3d_vector_data(cp, d_args); });
            // else if (dom_dim == 3)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_3d_vector_data(cp, d_args); });
            // else
            // {
            //     fprintf(stderr, "nek5000 data only available in 2 or 3d domain\n");
            //     exit(0);
            // }
        }

        // rti dataset:  rti/dd07g_xxsmall_le.xyz
        if (input == "rti")
        {
            d_args.full_dom_pts = {288, 512, 512};      // Hard-coded to full data set size
            d_args.ndom_pts = d_args.full_dom_pts;

            if (dom_dim >= 1) vars_nctrl[0] = 72;
            if (dom_dim >= 2) vars_nctrl[1] = 128;
            if (dom_dim >= 3) vars_nctrl[2] = 128;

            // for testing, hard-code a subset of a 3d domain, 1/2 the size in each dim and centered
            // d_args.ndom_pts = {144, 256, 256};
            // d_args.starts   = {72, 128, 128};

            // if (dom_dim == 2)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_2d_slice_3d_vector_data(cp, d_args); });
            // else if (dom_dim == 3)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_3d_vector_data(cp, d_args); });
            // else
            // {
            //     fprintf(stderr, "rti data only available in 2 or 3d domain\n");
            //     exit(0);
            // }




    //         master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
    //                 { b->read_3d_subset_3d_vector_data(cp, d_args); });
        }

        // cesm dataset:  CESM-ATM-tylor/1800x3600/FLDSC_1_1800_3600.dat
        if (input == "cesm")
        {
            d_args.full_dom_pts = {1800, 3600};
            d_args.ndom_pts = d_args.full_dom_pts;
            vars_nctrl      = {300, 600};

            // if (dom_dim == 2)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_2d_scalar_data(cp, d_args); });
            // else
            // {
            //     fprintf(stderr, "cesm data only available in 2d domain\n");
            //     exit(0);
            // }
        }

        // miranda dataset:  miranda/SDRBENCH-Miranda-256x384x384/density.d64
        if (input == "miranda")
        {
            d_args.full_dom_pts = {256, 384, 384};
            d_args.ndom_pts = d_args.full_dom_pts;

            vars_nctrl = {256, 384, 384};   // 192, 288, 288

            // if (dom_dim == 3)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_3d_scalar_data<double>(cp, d_args); });
            // else
            // {
            //     fprintf(stderr, "miranda data only available in 3d domain\n");
            //     exit(0);
            // }
        }

        // tornado dataset:  tornado/bov/1.vec.bov
        if (input == "tornado")
        {
            d_args.full_dom_pts = {128, 128, 128};
            d_args.ndom_pts = d_args.full_dom_pts;

            if (dom_dim >= 1) vars_nctrl[0] = 100;
            if (dom_dim >= 2) vars_nctrl[1] = 100;
            if (dom_dim >= 3) vars_nctrl[2] = 100;

            // if (dom_dim == 1)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_1d_slice_3d_vector_data(cp, d_args); });
            // else if (dom_dim == 2)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_2d_slice_3d_vector_data(cp, d_args); });
            // else if (dom_dim == 3)
            //     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_3d_vector_data(cp, d_args); });
            // else
            // {
            //     fprintf(stderr, "tornado data only available in 1, 2, or 3d domain\n");
            //     exit(0);
            // }
        }

        // EDelta Wing dataset (unstructured):  edelta/edelta.txt
        if (input == "edelta")
        {
            if (dom_dim != 2 || pt_dim != 7)
            {
                fprintf(stderr, "Error: Incorrect dimensionality for edelta example.\n");
                exit(0);
            }

            model_dims = {3, 1, 1, 1, 1};
            d_args.updateModelDims(model_dims);
            // model_dims.resize(5);
            // model_dims[0] = geom_dim;
            // model_dims[1] = 1;
            // model_dims[2] = 1;
            // model_dims[3] = 1;
            // model_dims[4] = 1;

            d_args.min.resize(3);
            d_args.max.resize(3);
            d_args.min[0] = -1.1032;    d_args.max[0] = 4.9763;
            d_args.min[1] = -2.1916;    d_args.max[1] = 2.2760;
            d_args.min[2] = 0.019779;   d_args.max[2] = 0.019779;
            
            d_args.tot_ndom_pts = 108822;

            vars_degree = 2;
            vars_nctrl[0] = 130;
            vars_nctrl[1] = 92;

            // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            //             { b->read_3d_unstructured_data(cp, d_args); });
        }

        // Climate dataset:  climate/climate-small.txt
        if (input == "climate")
        {
            if (dom_dim != 2 || pt_dim != 4)
            {
                fprintf(stderr, "Error: Incorrect dimensionality for climate example.\n");
                exit(0);
            }

            model_dims = {3, 1};
            d_args.updateModelDims(model_dims);

            d_args.min.resize(3);
            d_args.max.resize(3);
            d_args.min[0] = -2.55;  d_args.max[0] = -1.449;
            d_args.min[1] = -2.55;  d_args.max[1] = -1.449;
            d_args.min[2] =  0;     d_args.max[2] =  0;

            d_args.tot_ndom_pts = 585765;
        }

        // Nuclear dataset:  nuclear/sahex_core.txt
        if (input == "nuclear")
        {
            if (dom_dim != 3)
            {
                cerr << "dom_dim must be 3 to run nuclear example" << endl;
                exit(0);
            }

            model_dims = {3, 1, 1, 1, 1};
            d_args.updateModelDims(model_dims);

            d_args.min[0] =  1.5662432; d_args.max[0] = 30.433756;
            d_args.min[1] = -7.5980764; d_args.max[1] = 22.4019242;
            d_args.min[2] =  10;        d_args.max[2] = 35;

            vars_nctrl[2] = 15;  // fix (reduce) number of control points in z-direction
            
            d_args.tot_ndom_pts = 63048;
        }

        set_mfa_info(dom_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl, mfa_info);
        mfa_info.verbose          = verbose;
        mfa_info.weighted         = weighted;
        mfa_info.regularization   = regularization;
        mfa_info.reg1and2         = reg1and2;

    } // setup_args()



#endif // MFA_EX_SETUP_HPP