//--------------------------------------------------------------
// converts output files from DIY to VTK
//
// output precision is float irrespective whether input is float or double
//
// Iulian Grindeanu and Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include    "mfa/mfa.hpp"
#include    "block.hpp"
#include    <iostream>
#include    <stdio.h>

#include    <diy/master.hpp>
#include    <diy/io/block.hpp>

#include    "writer.hpp"

// package rendering data
void PrepRenderingData(
        vector<int>&                nraw_pts,
        vector<vec3d>&              raw_pts,
        float**&                    raw_data,
        int&                        nvars,
        vector<int>&                geom_nctrl_pts,
        vector< vector <int> >&     vars_nctrl_pts,
        vector<vec3d>&              geom_ctrl_pts,
        vector< vector <vec3d> >&   vars_ctrl_pts,
        float**&                    vars_ctrl_data,
        vector<vec3d>&              approx_pts,
        float**&                    approx_data,
        vector<vec3d>&              err_pts,
        diy::Master&                master)
{
    for (int i = 0; i < master.size(); i++)          // blocks
    {
        vec3d p;
        Block<real_t>* block = master.block< Block<real_t> >(i);

        // number of geometry dimensions and science variables
        int ndom_dims   = block->geometry.ctrl_pts.cols();          // number of geometry dims
        nvars           = block->vars.size();                       // number of science variables

        // number of raw points
        for (size_t j = 0; j < (size_t)(block->ndom_pts.size()); j++)
            nraw_pts.push_back(block->ndom_pts(j));

        // raw geometry and raw science variables
        raw_data = new float*[nvars];
        for (size_t j = 0; j < nvars; j++)
            raw_data[j] = new float[block->domain.rows()];

        for (size_t j = 0; j < (size_t)(block->domain.rows()); j++)
        {
            p.x = block->domain(j, 0);                              // mesh geometry stored in 3d
            p.y = block->domain(j, 1);
            p.z = block->domain.cols() > 2 ? block->domain(j, 2) : 0.0;
            raw_pts.push_back(p);

            for (int k = 0; k < nvars; k++)                         // science variables
                raw_data[k][j] = block->domain(j, ndom_dims + k);
        }

        // number of geometry control points
        for (size_t j = 0; j < (size_t)(block->geometry.nctrl_pts.size()); j++)
            geom_nctrl_pts.push_back(block->geometry.nctrl_pts(j));

        // number of science variable control points
        vars_nctrl_pts.resize(nvars);
        for (size_t i = 0; i < nvars; i++)
            for (size_t j = 0; j < (size_t)(block->vars[i].nctrl_pts.size()); j++)
                vars_nctrl_pts[i].push_back(block->vars[i].nctrl_pts(j));

        // geometry control points

        // compute vectors of individual control point coordinates for the tensor product
        vector<vector<float>> ctrl_pts_coords(ndom_dims);
        size_t k0               = 0;            // starting offset of knots in current dim
        size_t nknots_prev_dim  = 0;            // number of knots in previous dim
        for (int k = 0; k < ndom_dims; k++)
        {
            ctrl_pts_coords[k].resize(block->geometry.nctrl_pts(k));
            k0 += nknots_prev_dim;
            for (size_t j = 0; j < (size_t)(block->geometry.nctrl_pts(k)); j++)
            {
                float tsum = 0.0;
                for (int l = 1; l < block->geometry.p(k) + 1; l++)
                    tsum += block->geometry.knots(k0 + j + l);
                tsum /= float(block->geometry.p(k));
                ctrl_pts_coords[k][j] = block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k));
            }
            nknots_prev_dim = block->geometry.nctrl_pts(k) + block->geometry.p(k) + 1;    // nknots = n + p + 1 by defn.
        }

        // form the tensor product of control points from the vectors of individual coordinates
        vector<size_t> ijk(ndom_dims);                              // indices of control point
        for (size_t j = 0; j < (size_t)(block->geometry.ctrl_pts.rows()); j++)
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
                if (ijk[k] < block->geometry.nctrl_pts(k) - 1)
                {
                    ijk[k]++;
                    break;
                }
                else
                    ijk[k] = 0;
            }
        }

        // science variable control points
        vars_ctrl_pts.resize(nvars);
        vars_ctrl_data = new float*[nvars];
        for (size_t i = 0; i < nvars; i++)
        {
            vars_ctrl_data[i] = new float[block->vars[i].ctrl_pts.rows()];

            // compute vectors of individual control point coordinates for the tensor product
            vector<vector<float>> ctrl_pts_coords(ndom_dims);
            size_t k0               = 0;            // starting offset of knots in current dim
            size_t nknots_prev_dim  = 0;            // number of knots in previous dim
            for (int k = 0; k < ndom_dims; k++)
            {
                ctrl_pts_coords[k].resize(block->vars[i].nctrl_pts(k));
                k0 += nknots_prev_dim;
                for (size_t j = 0; j < (size_t)(block->vars[i].nctrl_pts(k)); j++)
                {
                    float tsum = 0.0;
                    for (int l = 1; l < block->vars[i].p(k) + 1; l++)
                        tsum += block->vars[i].knots(k0 + j + l);
                    tsum /= float(block->vars[i].p(k));
                    ctrl_pts_coords[k][j] = block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k));
                }
                nknots_prev_dim = block->vars[i].nctrl_pts(k) + block->vars[i].p(k) + 1;    // nknots = n + p + 1 by defn.
            }

            // form the tensor product of control points from the vectors of individual coordinates
            vector<size_t> ijk(ndom_dims);                              // indices of control point
            for (size_t j = 0; j < (size_t)(block->vars[i].ctrl_pts.rows()); j++)
            {
                // first 3 dims stored as mesh geometry
                // control point position and optionally science variable, if the total fits in 3d
                p.x = ctrl_pts_coords[0][ijk[0]];
                if (ndom_dims < 2)
                {
                    p.y = block->vars[i].ctrl_pts(j, 0);
                    p.z = 0.0;
                }
                else
                {
                    p.y = ctrl_pts_coords[1][ijk[1]];
                    if (ndom_dims < 3)
                        p.z = block->vars[i].ctrl_pts(j, 0);
                    else
                        p.z = ctrl_pts_coords[2][ijk[2]];
                }
                vars_ctrl_pts[i].push_back(p);

                // science variable also stored as data
                vars_ctrl_data[i][j] = block->vars[i].ctrl_pts(j, 0);

                // update ijk of next point
                for (int k = 0; k < ndom_dims; k++)
                {
                    if (ijk[k] < block->vars[i].nctrl_pts(k) - 1)
                    {
                        ijk[k]++;
                        break;
                    }
                    else
                        ijk[k] = 0;
                }
            }
        }

        // approximated points
        approx_data = new float*[nvars];
        for (size_t j = 0; j < nvars; j++)
            approx_data[j] = new float[block->domain.rows()];

        for (size_t j = 0; j < (size_t)(block->approx.rows()); j++)
        {
            p.x = block->approx(j, 0);                      // first 3 dims stored as mesh geometry
            p.y = block->approx(j, 1);
            p.z = block->approx.cols() > 2 ? block->approx(j, 2) : 0.0;
            approx_pts.push_back(p);

            for (int k = 0; k < nvars; k++)                         // science variables
                approx_data[k][j] = block->approx(j, ndom_dims + k);
        }

        // error points
        // error values and max_err are not normalized by data range
        float max_err = 0.0;
        for (size_t j = 0; j < (size_t)(block->errs.rows()); j++)
        {
            p.x = block->errs(j, 0);
            p.y = block->errs(j, 1);
            p.z = block->errs.cols() > 2 ?
                block->errs(j, 2) : 0.0;
            err_pts.push_back(p);
            if (block->errs.cols() > 2 && p.z > max_err)
                max_err = p.z;
            if (block->errs.cols() == 2 && p.y > max_err)
                max_err = p.y;
        }
    }
}


int main(int argc, char ** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);       // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;                 // equivalent of MPI_COMM_WORLD

    int                         nvars;              // number of science variables (excluding geometry)
    vector<int>                 nraw_pts;           // number of input points in each dim.
    vector<vec3d>               raw_pts;            // input raw data points (<= 3d)
    float**                     raw_data;           // input raw data values (4d)
    vector <int>                geom_nctrl_pts;     // number of control pts in each dim of geometry
    vector < vector <int> >     vars_nctrl_pts;     // number of control pts in each dim. of each science variable
    vector<vec3d>               geom_ctrl_pts;      // control points (<= 3d) in geometry
    vector < vector <vec3d> >   vars_ctrl_pts;      // control points (<= 3d) in science variables
    float**                     vars_ctrl_data;     // control point data values (4d)
    vector<vec3d>               approx_pts;         // aproximated data points (<= 3d)
    float**                     approx_data;        // approximated data values (4d)
    vector<vec3d>               err_pts;            // abs value error field
    string infile(argv[1]);

    diy::FileStorage storage("./DIY.XXXXXX");     // used for blocks to be moved out of core
    diy::Master      master(world,
            1,
            -1,
            &Block<real_t>::create,
            &Block<real_t>::destroy);
    diy::ContiguousAssigner   assigner(world.size(), -1); // number of blocks set by read_blocks()

    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block<real_t>::load);
    std::cout << master.size() << " blocks read from file "<< infile << "\n";

    // package rendering data
    PrepRenderingData(nraw_pts,
                      raw_pts,
                      raw_data,
                      nvars,
                      geom_nctrl_pts,
                      vars_nctrl_pts,
                      geom_ctrl_pts,
                      vars_ctrl_pts,
                      vars_ctrl_data,
                      approx_pts,
                      approx_data,
                      err_pts,
                      master);

    // pad dimensions up to 3
    size_t dom_dims = geom_nctrl_pts.size();
    for (auto i = 0; i < 3 - dom_dims; i++)
    {
        geom_nctrl_pts.push_back(1);
        nraw_pts.push_back(1);
    }

    // number of control points for science variables
    for (size_t j = 0; j < nvars; j++)
    {
        dom_dims = vars_nctrl_pts[j].size();
        for (auto i = 0; i < 3 - dom_dims; i++)
            vars_nctrl_pts[j].push_back(1);
    }

    // copy error as a new variable (z dimension, or maybe magnitude?)
    vector<float> errm(err_pts.size());
    for (size_t i=0; i<err_pts.size(); i++)
        errm[i] = err_pts[i].z;

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
    sprintf(filename, "geom_control_points.vtk");
    write_curvilinear_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int *dims */                                 &geom_nctrl_pts[0],
            /* float *pts */                                &(geom_ctrl_pts[0].x),
            /* int nvars */                                 0,
            /* int *vardim */                               NULL,
            /* int *centering */                            NULL,
            /* const char * const *varnames */              NULL,
            /* float **vars */                              NULL);

    // write science variables control points
    for (auto i = 0; i < nvars; i++)
    {
        sprintf(filename, "var%d_control_points.vtk", i);
        write_curvilinear_mesh(
                /* const char *filename */                  filename,
                /* int useBinary */                         0,
                /* int *dims */                             &vars_nctrl_pts[i][0],
                /* float *pts */                            &(vars_ctrl_pts[i][0].x),
                /* int nvars */                             nvars,
                /* int *vardim */                           vardims,
                /* int *centering */                        centerings,
                /* const char * const *varnames */          varnames,
                /* float **vars */                          vars_ctrl_data);
    }

    // write raw original points
    if (raw_pts.size())
        write_curvilinear_mesh(
                /* const char *filename */                  "initial_points.vtk",
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(raw_pts[0].x),
                /* int nvars */                             nvars,
                /* int *vardim */                           vardims,
                /* int *centering */                        centerings,
                /* const char * const *varnames */          varnames,
                /* float **vars */                          raw_data);
    else
    {
        vars = &errm[0];
        const char* name_err ="error";
        write_curvilinear_mesh(
                /* const char *filename */                  "initial_points.vtk",
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(raw_pts[0].x),
                /* int nvars */                             1,
                /* int *vardim */                           &vardim,
                /* int *centering */                        &centering,
                /* const char * const *varnames */          &name_err,
                /* float **vars */                          &vars);
    }

    // write approx points
    if (approx_pts.size())
        write_curvilinear_mesh(
                /* const char *filename */                  "approx_points.vtk",
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(approx_pts[0].x),
                /* int nvars */                             nvars,
                /* int *vardim */                           vardims,
                /* int *centering */                        centerings,
                /* const char * const *varnames */          varnames,
                /* float **vars */                          approx_data);

    else
        write_curvilinear_mesh(
                /* const char *filename */                  "approx_points.vtk",
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(approx_pts[0].x),
                /* int nvars */                             0,
                /* int *vardim */                           NULL,
                /* int *centering */                        NULL,
                /* const char * const *varnames */          NULL,
                /* float **vars */                          NULL);

    // write error
    vars = &errm[0];
    const char* name_err ="error";
    write_curvilinear_mesh(
            /* const char *filename */                      "error.vtk",
            /* int useBinary */                             0,
            /* int *dims */                                 &nraw_pts[0],
            /* float *pts */                                &(err_pts[0].x),
            /* int nvars */                                 1,
            /* int *vardim */                               &vardim,
            /* int *centering */                            &centering,
            /* const char * const *varnames */              &name_err,
            /* float **vars */                              &vars);

    delete[] vardims;
    for (int i = 0; i < nvars; i++)
        delete[] varnames[i];
    delete[] varnames;
    delete[] centerings;
    for (int j = 0; j < nvars; j++)
    {
        delete[] raw_data[j];
        delete[] vars_ctrl_data[j];
        delete[] approx_data[j];
    }
    delete[] raw_data;
    delete[] vars_ctrl_data;
    delete[] approx_data;
}
