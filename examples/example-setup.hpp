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
#include <set>

#include "block.hpp"

using namespace std;
    void echo_mfa_settings(string run_name, int dom_dim, int pt_dim, int scalar,
                            int geom_degree, int geom_nctrl, int vars_degree, vector<int>& vars_nctrl,
                            real_t regularization, int reg1and2,
                            int adaptive, real_t e_threshold, int rounds,
                            ostream& os = std::cerr)
    {
        os << ">>> Running \'" << run_name << "\'" << endl;
        os << endl;
        os << "--------- MFA Settings ----------" << endl;
        os << "pt_dim   = " << pt_dim      << '\t' << "dom_dim    = " << dom_dim 
                << '\t' << "scalar: " << boolalpha << (bool)scalar << endl;
        os << "geom_deg = " << geom_degree << '\t' << "vars_deg = " << vars_degree << endl;
        os << "encoding type = " << (adaptive ? "adaptive" : "fixed") << endl;
        if (adaptive)
        {
        os << "error    = " << e_threshold << '\t' << "max rounds = " << (rounds == 0 ? "unlimited" : to_string(rounds)) << endl;
        }
        else
        {
        os << "geom_nctrl = " << geom_nctrl << '\t' << "vars_nctrl = " << mfa::print_vec(vars_nctrl) << endl;
        }
        os << "regularization = " << regularization << ", type: " << 
            (regularization == 0 ? "N/A" : (reg1and2 > 0 ? "1st and 2nd derivs" : "2nd derivs only")) << endl;

#ifdef MFA_NO_WEIGHTS
        os << "weighted: false" << endl;
#else
        os << "weighted: " << boolalpha <<  (bool)weighted << endl;
#endif
#ifdef CURVE_PARAMS
        os << "parameterization method: curve" << endl;
        os << "ERROR: curve parametrization not currently supported in examples. Exiting." << endl;
        exit(1);
#else
        os << "parameterization method: domain" << endl;
#endif
#ifdef MFA_TBB
        os << "threading: TBB" << endl;
#endif
#ifdef MFA_KOKKOS
        os << "threading: Kokkos" << endl;
        os << "KOKKOS execution space: " << Kokkos::DefaultExecutionSpace::name() << "\n";
#endif
#ifdef MFA_SYCL
        os << "threading: SYCL" << endl;
#endif
#ifdef MFA_SERIAL
        os << "threading: serial" << endl;
#endif

        return;
    }

    // Print basic info about data set
    void echo_data_settings(string input, string infile, int ndomp, int ntest, ostream& os = std::cerr)
    {
        bool is_analytical = (analytical_signals.count(input) == 1);

        os << "--------- Data Settings ----------" << endl;
        os << "input: "       << input       << ", " << "infile: " << (is_analytical ? "N/A" : infile) << endl;
        os << "num pts    = " << ndomp       << '\t' << "test pts    = " << (ntest > 0 ? to_string(ntest) : "N/A") << endl;

        return;
    }

    // Print basic info about data set
    void echo_data_settings(string input, string infile, int ndomp, vector<int> ntest, ostream& os = std::cerr)
    {
        bool is_analytical = (analytical_signals.count(input) == 1);

        os << "--------- Data Settings ----------" << endl;
        os << "input: "       << input       << ", " << "infile: " << (is_analytical ? "N/A" : infile) << endl;
        os << "num pts    = " << ndomp       << '\t' << "test pts    = " << mfa::print_vec(ntest) << endl;

        return;
    }  

    // Print all info about data set
    void echo_data_mod_settings(int structured, int rand_seed, real_t rot, real_t twist, real_t noise, ostream& os = std::cerr)
    {
        os << "structured = " << boolalpha << (bool)structured << '\t' << "random seed = " << rand_seed << endl;
        os << "rotation   = " << setw(7) << left << rot << '\t' << "twist       = " << twist << endl;
        os << "noise      = " << setw(7) << left << noise << endl;

        return;
    }

    // Set DIY Bounds for decomposition based on example
    // If the input is not an analytical signal, then dummy bounds are used
    // because we assume the example will be run on a single block
    void set_dom_bounds(Bounds<real_t>& dom_bounds, string input)
    {
        if (input == "sine" || input == "cosine" || input == "sinc" ||
            input == "psinc1" || input == "psinc2" || input == "psinc3")
        {
            for (int i = 0; i < dom_bounds.min.dimension(); i++)
            {
                dom_bounds.min[i] = -4.0 * M_PI;
                dom_bounds.max[i] =  4.0 * M_PI;
            }
        }
        else if (input == "ml")
        {
            for (int i = 0; i < dom_bounds.min.dimension(); i++)
            {
                dom_bounds.min[i] = -1.0;
                dom_bounds.max[i] =  1.0;
            }
        }
        else if (input == "f16")
        {
            dom_bounds.min = {-1, -1};
            dom_bounds.max = { 1,  1};
        }
        else if (input == "f17")
        {
            dom_bounds.min = {80,   5, 90};
            dom_bounds.max = {100, 10, 93}; 
        }
        else if (input == "f18")
        {
            dom_bounds.min = {-0.95, -0.95, -0.95, -0.95};
            dom_bounds.max = { 0.95,  0.95,  0.95,  0.95};
        }
        else if (datasets_4d.count(input) || datasets_3d.count(input) || datasets_2d.count(input) || datasets_unstructured.count(input))
        {
            for (int i = 0; i < dom_bounds.min.dimension(); i++)
            {
                dom_bounds.min[i] = 0.0;
                dom_bounds.max[i] = 1.0;
            }
        }
        else
        {
            cerr << "Unrecognized input in set_dom_bounds(). Exiting." << endl;
            exit(1);
        }

        return;
    }

    // Sets the degree and number of control points 
    // For geom and each var, degree is same in each domain dimension
    // For geom, # ctrl points is the same in each domain dimension
    // For each var, # ctrl points varies per domain dimension, but is the same for all of the vars
    // void set_mfa_info(vector<int> model_dims, MFAInfo& mfa_info)
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
            ModelInfo var_info(dom_dim, model_dims[k+1], vars_degree, vars_nctrl);
            mfa_info.addVarInfo(var_info);
        }
    }

    void echo_multiblock_settings(MFAInfo& mfa_info, DomainArgs& d_args, int nproc, int tot_blocks, vector<int>& divs, int strong_sc, real_t ghost, ostream& os = std::cerr)
    {
        os << "------- Multiblock Settings ---------" << endl;
        os << "Total MPI processes  =  " << nproc << "\t" << "Total blocks = " << tot_blocks << endl;
        os << "Blocks per dimension = " << mfa::print_vec(divs) << endl;
        os << "Ghost overlap  = " << ghost << endl;
        os << "Strong scaling = " << boolalpha << (bool)strong_sc << endl;
        os << "Per-block settings:" << endl;
        os << "    Input pts (each dim):      " << mfa::print_vec(d_args.ndom_pts) << endl;
        os << "    Geom ctrl pts (each dim):  " << mfa::print_vec(mfa_info.geom_model_info.nctrl_pts) << endl;
        for (int k = 0; k < mfa_info.nvars(); k++)
        {
            os << "    Var " << k << " ctrl pts (each dim): " << mfa::print_vec(mfa_info.var_model_infos[k].nctrl_pts) << endl;
        }

        return;
    }

    // Currently for single block examples only
    // void setup_args(vector<int> model_dims, MFAInfo& mfa_info, DomainArgs& d_args)
    void setup_args( int dom_dim, int pt_dim, vector<int> model_dims,
                        int geom_degree, int geom_nctrl, int vars_degree, vector<int> vars_nctrl,
                        string input, string infile, int ndomp,
                        int structured, int rand_seed, real_t rot, real_t twist, real_t noise,
                        int reg1and2, real_t regularization, bool adaptive, int verbose,
                        MFAInfo& mfa_info, DomainArgs& d_args)
    {
        int weighted = 0;

        // If only one value for vars_nctrl was parsed, assume it applies to all dims
        if (vars_nctrl.size() == 1 & dom_dim > 1)
        {
            vars_nctrl = vector<int>(dom_dim, vars_nctrl[0]);
        }

        // Set basic info for DomainArgs
        d_args.updateModelDims(model_dims);
        d_args.multiblock   = false;
        d_args.r            = rot * M_PI / 180;
        d_args.t            = twist;
        d_args.n            = noise;
        d_args.infile       = infile;
        d_args.infile2      = "";
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

        // sine, cosine functions
        if (input == "sine" || input == "cosine")
        {
            d_args.min.assign(dom_dim, -4.0 * M_PI);
            d_args.max.assign(dom_dim,  4.0 * M_PI);
            for (int i = 0; i < d_args.model_dims.size()-1; i++)
                d_args.s[i] = i + 1;
        }

        // sinc, polysinc functions
        if (input == "sinc" || input == "psinc1" || input == "psinc2" || input == "psinc3")
        {
            d_args.min.assign(dom_dim, -4.0 * M_PI);
            d_args.max.assign(dom_dim,  4.0 * M_PI);
            for (int i = 0; i < d_args.model_dims.size()-1; i++)
                d_args.s[i] = 10.0 * (i + 1);
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
        }

        // S3D dataset:  flame/6_small.xyz
        if (input == "s3d")
        {
            d_args.full_dom_pts = {704, 540, 550};  // Hard-coded to full data set size
            d_args.ndom_pts = d_args.full_dom_pts;

            if (!adaptive)
            {
                if (dom_dim >= 1) vars_nctrl[0] = 140;
                if (dom_dim >= 2) vars_nctrl[1] = 108;
                if (dom_dim >= 3) vars_nctrl[2] = 110;
            }

            // for testing, hard-code a subset of a 3d domain, 1/2 the size in each dim and centered
            // in this case, actual size is just under 1/2 size in each dim to satisfy DIY's (MPI's)
            // limitation on the size of a file write
            // (MPI uses int for the size, and DIY as yet does not chunk writes into smaller sizes)
            // d_args.ndom_pts = {350, 2250, 250};
            // d_args.starts   = {125, 50, 125};
        }

        // nek5000 dataset:  nek5000/200x200x200/0.xyz
        if (input == "nek")
        {
            d_args.full_dom_pts = {200, 200, 200};      // Hard-coded to full data set size
            d_args.ndom_pts = d_args.full_dom_pts;
            
            if (!adaptive)
            {
                if (dom_dim >= 1) vars_nctrl[0] = 100;
                if (dom_dim >= 2) vars_nctrl[1] = 100;
                if (dom_dim >= 3) vars_nctrl[2] = 100;
            }
        }

        // rti dataset:  rti/dd07g_xxsmall_le.xyz
        if (input == "rti")
        {
            d_args.full_dom_pts = {288, 512, 512};      // Hard-coded to full data set size
            d_args.ndom_pts = d_args.full_dom_pts;

            if (!adaptive)
            {
                if (dom_dim >= 1) vars_nctrl[0] = 72;
                if (dom_dim >= 2) vars_nctrl[1] = 128;
                if (dom_dim >= 3) vars_nctrl[2] = 128;
            }
        }

        // tornado dataset:  tornado/bov/1.vec.bov
        if (input == "tornado")
        {
            d_args.full_dom_pts = {128, 128, 128};
            d_args.ndom_pts = d_args.full_dom_pts;

            if (!adaptive)
            {
                if (dom_dim >= 1) vars_nctrl[0] = 100;
                if (dom_dim >= 2) vars_nctrl[1] = 100;
                if (dom_dim >= 3) vars_nctrl[2] = 100;
            }
        }

        // time series tornado dataset
        if (input == "tornado4d")
        {
            d_args.full_dom_pts = {128, 128, 128, 49};
            d_args.ndom_pts = d_args.full_dom_pts;

            // number of control points in space dimensions
            if (!adaptive)
            {
                vars_nctrl = {100, 100, 100, 35};
            }
        }

        // miranda dataset:  miranda/SDRBENCH-Miranda-256x384x384/density.d64
        if (input == "miranda")
        {
            d_args.full_dom_pts = {256, 384, 384};
            d_args.ndom_pts = d_args.full_dom_pts;

            if (!adaptive)
            {
                if (dom_dim >= 1) vars_nctrl[0] = 256; // 192
                if (dom_dim >= 2) vars_nctrl[1] = 384; // 288
                if (dom_dim >= 3) vars_nctrl[2] = 384; // 288
            }
        }

        // cesm dataset:  CESM-ATM-tylor/1800x3600/FLDSC_1_1800_3600.dat
        if (input == "cesm")
        {
            d_args.full_dom_pts = {1800, 3600};
            d_args.ndom_pts = d_args.full_dom_pts;

            if (!adaptive)
            {
                vars_nctrl = {300, 600};
            }
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
            d_args.updateModelDims(model_dims);     // Need to update d_args with new model_dims

            d_args.min.resize(3);
            d_args.max.resize(3);
            d_args.min[0] = -1.1032;    d_args.max[0] = 4.9763;
            d_args.min[1] = -2.1916;    d_args.max[1] = 2.2760;
            d_args.min[2] = 0.019779;   d_args.max[2] = 0.019779;
            
            d_args.tot_ndom_pts = 108822;

            vars_degree = 2;
            vars_nctrl[0] = 130;
            vars_nctrl[1] = 92;
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
            d_args.updateModelDims(model_dims);     // Need to update d_args with new model_dims

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
            d_args.updateModelDims(model_dims);     // Need to update d_args with new model_dims

            d_args.min[0] =  1.5662432; d_args.max[0] = 30.433756;
            d_args.min[1] = -7.5980764; d_args.max[1] = 22.4019242;
            d_args.min[2] =  10;        d_args.max[2] = 35;

            vars_nctrl[2] = 15;  // set (reduce) number of control points in z-direction
            
            d_args.tot_ndom_pts = 63048;
        }

        // NASA Fun3d retropropulsion dataset
        if (input == "nasa")
        {
            if (dom_dim != 3)
            {
                cerr << "dom_dim must be 3 to run nasa example" << endl;
                exit(1);
            }

            model_dims = {3, 1};
            d_args.updateModelDims(model_dims);

            d_args.min[0] = -1; d_args.max[0] = 0;
            d_args.min[1] = 0;  d_args.max[1] = 2;
            d_args.min[2] = -5; d_args.max[2] = -4;

            d_args.tot_ndom_pts = 55365;
        }

        set_mfa_info(dom_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl, mfa_info);
        mfa_info.verbose          = verbose;
        mfa_info.weighted         = weighted;
        mfa_info.regularization   = regularization;
        mfa_info.reg1and2         = reg1and2;
    } // setup_args()

#endif // MFA_EX_SETUP_HPP