//--------------------------------------------------------------
// writes all vtk files for initial, approximated, and control points
//
// optionally generates test data for analytical functions and writes to vtk
//
// output precision is float irrespective whether input is float or double
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include    "mfa/mfa.hpp"
#include    <iostream>
#include    <stdio.h>

#include    <diy/master.hpp>
#include    <diy/io/block.hpp>

#include    "opts.h"

#include    "writer.hpp"
#include    "block.hpp"

// TODO: Only scalar-valued and 3D vector-valued variables are supported (because of the VTK writer)
// If a variable has a different output dimension, the writer will skip that variable and continue.
template<typename T>
void write_pointset_vtk(mfa::PointSet<T>* ps, char* filename, int sci_var = -1)
{
    if (ps == nullptr)
    {
        cout << "Did not write " << filename << " due to uninitialized pointset" << endl;
        return;
    }
    if (ps->npts == 0)
    {
        cout << "Did not write " << filename << " due to empty pointset" << endl;
        return;
    }

    int dom_dim = ps->dom_dim;
    int pt_dim  = ps->pt_dim;
    int geom_dim = ps->geom_dim();
    int nvars = ps->nvars();
    bool include_var = true;        // Include the specified science variable in the geometry coordinates
    int var_col = ps->model_dims().head(sci_var + 1).sum(); // column of the variable to be visualized

    if (geom_dim < 1 || geom_dim > 3)
    {
        cerr << "Did not write " << filename << " due to improper dimension in pointset" << endl;
        return;
    }
    if (ps->var_dim(sci_var) != 1 && geom_dim < 3)
    {
        cerr << "For " << filename << ", specified science variable (#" << sci_var << ") is not a scalar. Output will be planar." << endl;
        include_var = false;
    }
    if (sci_var < 0)
    {
        include_var = false;
    }

    vector<int> npts_dim;  // only used if data is structured
    if (ps->is_structured())
    {
        for (size_t k = 0; k < 3; k++)
        {
            if (k < dom_dim) 
                npts_dim.push_back(ps->ndom_pts(k));
            else
                npts_dim.push_back(1);
        }
    }
    
    float** pt_data = new float*[nvars];
    for (size_t k = 0; k < nvars; k++)
    {
        pt_data[k]  = new float[ps->npts * ps->var_dim(k)];
    }

    vec3d           pt;
    vector<vec3d>   pt_coords;
    for (int j = 0; j < ps->npts; j++)
    {
        // Add geometric coordinates
        if (geom_dim == 1)
        {
            pt.x = ps->domain(j, 0);
            pt.y = include_var ? ps->domain(j, var_col) : 0.0;
            pt.z = 0.0;
        }
        else if (geom_dim == 2)
        {
            pt.x = ps->domain(j, 0);
            pt.y = ps->domain(j, 1);
            pt.z = include_var ? ps->domain(j, var_col) : 0.0;
        }
        else
        {
            pt.x = ps->domain(j, 0);
            pt.y = ps->domain(j, 1);
            pt.z = ps->domain(j, 2);
        }
        pt_coords.push_back(pt);

        // Add science variable data
        int offset_idx = 0;
        int all_vars_dim = pt_dim - geom_dim;
        for (int k = 0; k < nvars; k++)
        {
            int vd = ps->var_dim(k);
            for (int l = 0; l < vd; l++)
            {
                pt_data[k][j*vd + l] = ps->domain(j, geom_dim + offset_idx);
                offset_idx++;
            }
        }    
    }

    // science variable settings
    int* vardims        = new int[nvars];
    char** varnames     = new char*[nvars];
    int* centerings     = new int[nvars];
    for (int k = 0; k < nvars; k++)
    {
        vardims[k]      = ps->var_dim(k);
        varnames[k]     = new char[256];
        centerings[k]   = 1;
        sprintf(varnames[k], "var%d", k);
    }

    // write raw original points
    if (ps->is_structured())
    {
        write_curvilinear_mesh(
            /* const char *filename */                  filename,
            /* int useBinary */                         0,
            /* int *dims */                             &npts_dim[0],
            /* float *pts */                            &(pt_coords[0].x),
            /* int nvars */                             nvars,
            /* int *vardim */                           vardims,
            /* int *centering */                        centerings,
            /* const char * const *varnames */          varnames,
            /* float **vars */                          pt_data);
    }
    else
    {
        write_point_mesh(
        /* const char *filename */                      filename,
        /* int useBinary */                             0,
        /* int npts */                                  pt_coords.size(),
        /* float *pts */                                &(pt_coords[0].x),
        /* int nvars */                                 nvars,
        /* int *vardim */                               vardims,
        /* const char * const *varnames */              varnames,
        /* float **vars */                              pt_data);
    }  

    delete[] vardims;
    for (int i = 0; i < nvars; i++)
        delete[] varnames[i];
    delete[] varnames;
    delete[] centerings;
    for (int j = 0; j < nvars; j++)
    {
        delete[] pt_data[j];
    }
    delete[] pt_data;
}

// package rendering data
void PrepRenderingData(
        int&                        nvars,
        vector<vec3d>&              geom_ctrl_pts,
        vector< vector <vec3d> >&   vars_ctrl_pts,
        float**&                    vars_ctrl_data,
        Block<real_t>*              block,
        int&                        pt_dim)                 // (output) dimensionality of point
{
    vec3d p;

    // number of geometry dimensions and science variables
    int ndom_dims   = block->mfa->geom_dim();               // number of geometry dims
    nvars           = block->mfa->nvars();                  // number of science variables
    pt_dim          = block->mfa->pt_dim;                   // dimensionality of point


    // geometry control points

    // compute vectors of individual control point coordinates for the tensor product
    const mfa::MFA_Data<real_t>& geom = block->mfa->geom();
    vector<vector<float>> ctrl_pts_coords(ndom_dims);
    for (int k = 0; k < ndom_dims; k++)
    {
        // TODO: hard-coded for one tensor product
        ctrl_pts_coords[k].resize(geom.tmesh.tensor_prods[0].nctrl_pts(k));
        for (size_t j = 0; j < (size_t)(geom.tmesh.tensor_prods[0].nctrl_pts(k)); j++)
        {
            float tsum = 0.0;
            for (int l = 1; l < geom.p(k) + 1; l++)
                tsum += geom.tmesh.all_knots[k][j + l];
            tsum /= float(geom.p(k));
            ctrl_pts_coords[k][j] = block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k));
        }
    }

    // form the tensor product of control points from the vectors of individual coordinates
    // TODO: hard-coded for one tensor product
    vector<size_t> ijk(ndom_dims);                              // indices of control point
    for (size_t j = 0; j < (size_t)(geom.tmesh.tensor_prods[0].ctrl_pts.rows()); j++)
    {
        // first 3 dims stored as mesh geometry
        p.x = ctrl_pts_coords[0][ijk[0]];
        if (ndom_dims < 2)
            p.y = 0.0;
        else
            p.y = ctrl_pts_coords[1][ijk[1]];
        if (ndom_dims < 3)
            p.z = 0.0;
        else
            p.z = ctrl_pts_coords[2][ijk[2]];
        geom_ctrl_pts.push_back(p);

        // update ijk of next point
        for (int k = 0; k < ndom_dims; k++)
        {
            if (ijk[k] < geom.tmesh.tensor_prods[0].nctrl_pts(k) - 1)
            {
                ijk[k]++;
                break;
            }
            else
                ijk[k] = 0;
        }
    }

    // science variable control points
    // TODO: hard-coded for one tensor product
    vars_ctrl_pts.resize(nvars);
    vars_ctrl_data = new float*[nvars];
    for (size_t i = 0; i < nvars; i++)
    {
        const mfa::MFA_Data<real_t> var = block->mfa->var(i);
        vars_ctrl_data[i] = new float[var.tmesh.tensor_prods[0].ctrl_pts.rows()];

        // compute vectors of individual control point coordinates for the tensor product
        vector<vector<float>> ctrl_pts_coords(ndom_dims);
        for (int k = 0; k < ndom_dims; k++)
        {
            ctrl_pts_coords[k].resize(var.tmesh.tensor_prods[0].nctrl_pts(k));
            for (size_t j = 0; j < (size_t)(var.tmesh.tensor_prods[0].nctrl_pts(k)); j++)
            {
                float tsum = 0.0;
                for (int l = 1; l < var.p(k) + 1; l++)
                    tsum += var.tmesh.all_knots[k][j + l];
                tsum /= float(var.p(k));
                ctrl_pts_coords[k][j] = block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k));
            }
        }

        // form the tensor product of control points from the vectors of individual coordinates
        // TODO: hard-coded for one tensor product
        vector<size_t> ijk(ndom_dims);                              // indices of control point
        for (size_t j = 0; j < (size_t)(var.tmesh.tensor_prods[0].ctrl_pts.rows()); j++)
        {
            // first 3 dims stored as mesh geometry
            // control point position and optionally science variable, if the total fits in 3d
            p.x = ctrl_pts_coords[0][ijk[0]];
            if (ndom_dims < 2)
            {
                p.y = var.tmesh.tensor_prods[0].ctrl_pts(j, 0);
                p.z = 0.0;
            }
            else
            {
                p.y = ctrl_pts_coords[1][ijk[1]];
                if (ndom_dims < 3)
                    p.z = var.tmesh.tensor_prods[0].ctrl_pts(j, 0);
                else
                    p.z = ctrl_pts_coords[2][ijk[2]];
            }
            vars_ctrl_pts[i].push_back(p);

            // science variable also stored as data
            vars_ctrl_data[i][j] = var.tmesh.tensor_prods[0].ctrl_pts(j, 0);

            // update ijk of next point
            for (int k = 0; k < ndom_dims; k++)
            {
                if (ijk[k] < var.tmesh.tensor_prods[0].nctrl_pts(k) - 1)
                {
                    ijk[k]++;
                    break;
                }
                else
                    ijk[k] = 0;
            }
        }
    }
}

// write vtk files for initial, approximated, control points
void write_vtk_files(
        Block<real_t>* b,
        const          diy::Master::ProxyWithLink& cp,
        int&           dom_dim,                     // (output) domain dimensionality
        int&           pt_dim)                      // (output) point dimensionality
{
    int                         nvars;              // number of science variables (excluding geometry)
    vector<vec3d>               geom_ctrl_pts;      // control points (<= 3d) in geometry
    vector < vector <vec3d> >   vars_ctrl_pts;      // control points (<= 3d) in science variables
    float**                     vars_ctrl_data;     // control point data values (4d)

    // package rendering data
    PrepRenderingData(nvars,
                      geom_ctrl_pts,
                      vars_ctrl_pts,
                      vars_ctrl_data,
                      b,
                      pt_dim);


    // science variable settings
    int vardim          = 1;
    int centering       = 1;
    int* vardims        = new int[nvars];
    char** varnames     = new char*[nvars];
    int* centerings     = new int[nvars];
    float* vars;
    for (int i = 0; i < nvars; i++)
    {
        vardims[i]      = 1;                                // TODO; treating each variable as a scalar (for now)
        varnames[i]     = new char[256];
        centerings[i]   = 1;
        sprintf(varnames[i], "var%d", i);
    }

    // write geometry control points
    char filename[256];
    sprintf(filename, "geom_control_points_gid_%d.vtk", cp.gid());
    if (geom_ctrl_pts.size())
        write_point_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int npts */                                  geom_ctrl_pts.size(),
            /* float *pts */                                &(geom_ctrl_pts[0].x),
            /* int nvars */                                 0,
            /* int *vardim */                               NULL,
            /* const char * const *varnames */              NULL,
            /* float **vars */                              NULL);

    // write science variables control points
    for (auto i = 0; i < nvars; i++)
    {
        sprintf(filename, "var%d_control_points_gid_%d.vtk", i, cp.gid());
        if (vars_ctrl_pts[i].size())
            write_point_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int npts */                                  vars_ctrl_pts[i].size(),
            /* float *pts */                                &(vars_ctrl_pts[i][0].x),
            /* int nvars */                                 nvars,
            /* int *vardim */                               vardims,
            /* const char * const *varnames */              varnames,
            /* float **vars */                              vars_ctrl_data);
    }

    char input_filename[256];
    char approx_filename[256];
    char errs_filename[256];
    sprintf(input_filename, "initial_points_gid_%d.vtk", cp.gid());
    sprintf(approx_filename, "approx_points_gid_%d.vtk", cp.gid());
    sprintf(errs_filename, "error_gid_%d.vtk", cp.gid());
    write_pointset_vtk(b->input, input_filename); cerr << "A" << endl;
    write_pointset_vtk(b->approx, approx_filename); cerr << "B" << endl;
    write_pointset_vtk(b->errs, errs_filename); cerr << "C" << endl;

    delete[] vardims;
    for (int i = 0; i < nvars; i++)
        delete[] varnames[i];
    delete[] varnames;
    delete[] centerings;
    for (int j = 0; j < nvars; j++)
    {
        delete[] vars_ctrl_data[j];
    }
    delete[] vars_ctrl_data;
}

// generate analytical test data and write to vtk
void test_and_write(Block<real_t>*                      b,
                    const diy::Master::ProxyWithLink&   cp,
                    string                              input,
                    int                                 sci_var,
                    DomainArgs&                         args)
{
    if (!b->dom_dim)
        b->dom_dim =  b->mfa->dom_dim;

    // default args for evaluating analytical functions
    for (auto i = 0; i < b->mfa->nvars(); i++)
    {
        args.f[i] = 1.0;
        if (input == "sine")
            args.s[i] = i + 1;
        if (input == "sinc")
            args.s[i] = 10.0 * (i + 1);
    }

    // compute the norms of analytical errors synthetic function w/o noise at different domain points than the input
    int nvars = b->mfa->nvars();
    vector<real_t> L1(nvars), L2(nvars), Linf(nvars);                                // L-1, 2, infinity norms
    mfa::PointSet<real_t> *exact_pts = nullptr, *approx_pts = nullptr, *error_pts = nullptr;
    b->analytical_error_field(cp, input, L1, L2, Linf, args, exact_pts, approx_pts, error_pts);

    // print analytical errors
    fprintf(stderr, "\n------ Analytical error norms -------\n");
    fprintf(stderr, "L-1        norm = %e\n", L1);
    fprintf(stderr, "L-2        norm = %e\n", L2);
    fprintf(stderr, "L-infinity norm = %e\n", Linf);
    fprintf(stderr, "-------------------------------------\n\n");

    char exact_filename[256], approx_filename[256], error_filename[256];
    sprintf(exact_filename, "test_exact_pts_gid_%d.vtk", cp.gid());
    sprintf(approx_filename, "test_approx_pts_gid_%d.vtk", cp.gid());
    sprintf(error_filename, "test_error_pts_gid_%d.vtk", cp.gid());
    write_pointset_vtk(exact_pts, exact_filename, sci_var);
    write_pointset_vtk(approx_pts, approx_filename, sci_var);
    write_pointset_vtk(error_pts, error_filename, sci_var);

    delete exact_pts;
    delete approx_pts;
    delete error_pts;
}

int main(int argc, char ** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);       // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;                 // equivalent of MPI_COMM_WORLD

    string                      input  = "sine";        // input dataset
    int                         ntest  = 0;             // number of input test points in each dim for analytical error tests
    string                      infile = "approx.mfa";  // diy input file
    bool                        help;                   // show help
    int                         dom_dim, pt_dim;        // domain and point dimensionality, respectively
    int                         sci_var = 0;            // science variable to render geometrically for 1d and 2d domains

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('f', "infile",      infile,     " diy input file name");
    ops >> opts::Option('a', "ntest",       ntest,      " number of test points in each dimension of domain (for analytical error calculation)");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('h', "help",        help,       " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr << "infile = " << infile << " test_points = "    << ntest <<        endl;
    if (ntest)
        cerr << "input = "          << input     << endl;
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
    fprintf(stderr, "-------------------------------------\n\n");

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX");     // used for blocks to be moved out of core
    diy::Master      master(world,
            1,
            -1,
            &Block<real_t>::create,
            &Block<real_t>::destroy);
    diy::ContiguousAssigner   assigner(world.size(), -1); // number of blocks set by read_blocks()

    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block<real_t>::load);
    std::cout << master.size() << " blocks read from file "<< infile << "\n\n";

    // write vtk files for initial and approximated points
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { write_vtk_files(b, cp, dom_dim, pt_dim); });

    // rest of the code tests analytical functions and writes those files

    if (ntest <= 0)
        exit(0);

    // Get dimensions of each science variable from the loaded MFA
    VectorXi model_dims = master.block<Block<real_t>>(0)->mfa->model_dims();
    vector<int> mdims(model_dims.size());
    for (int i = 0; i < model_dims.size(); i++)
    {
        mdims[i] = model_dims(i);
    }

    // arguments for analytical functions
    DomainArgs d_args(dom_dim, mdims);
    d_args.ndom_pts = vector<int>(dom_dim, ntest);

    if (input == "sine")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
        }
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
        }
    }

    // f16 function
    if (input == "f16")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -1.0;
            d_args.max[i]               = 1.0;
        }
    }

    // f17 function
    if (input == "f17")
    {
        d_args.min[0] = 80.0;   d_args.max[0] = 100.0;
        d_args.min[1] = 5.0;    d_args.max[1] = 10.0;
        d_args.min[2] = 90.0;   d_args.max[2] = 93.0;
    }

    // f18 function
    if (input == "f18")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -0.95;
            d_args.max[i]               = 0.95;
        }
    }

    // compute the norms of analytical errors of synthetic function w/o noise at test points
    // and write true points and test points to vtk
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { test_and_write(b, cp, input, sci_var, d_args); });
}
