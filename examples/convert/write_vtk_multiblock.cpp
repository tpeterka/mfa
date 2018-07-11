//--------------------------------------------------------------
// converts output files from DIY to VTK
//
// output precision is float irrespective whether input is float or double
//
// Iulian Grindeanu and Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include "mfa/mfa.hpp"
#include "../split_block.hpp"
#include <iostream>

#include <diy/master.hpp>
#include <diy/io/block.hpp>

#include "writer.hpp"

// 3d point or vector
struct vec3d
{
    float x, y, z;
    float mag() { return sqrt(x*x + y*y + z*z); }
};

// package rendering data
void PrepRenderingData(
        vector<int>&                nraw_pts,
        vector<vec3d>&              raw_pts,
        vector<float>&              raw_data,
        int&                        nvars,
        vector<int>&                geom_nctrl_pts,
        vector< vector <int> >&     vars_nctrl_pts,
        vector<vec3d>&              geom_ctrl_pts,
        vector< vector <vec3d> >&   vars_ctrl_pts,
        vector<vec3d>&              approx_pts,
        vector<float>&              approx_data,
        vector<vec3d>&              err_pts,
        vector<vec3d>&              block_mins,
        vector<vec3d>&              block_maxs,
        Block<real_t>*              block)
{
    vec3d p;

    // number of raw points
    for (size_t j = 0; j < (size_t)(block->ndom_pts.size()); j++)
        nraw_pts.push_back(block->ndom_pts(j));

    // raw points
    for (size_t j = 0; j < (size_t)(block->domain.rows()); j++)
    {
        p.x = block->domain(j, 0);                      // first 3 dims stored as mesh geometry
        p.y = block->domain(j, 1);
        p.z = block->domain.cols() > 2 ?
            block->domain(j, 2) : 0.0;
        raw_pts.push_back(p);

        if (block->domain.cols() > 3)                   // 4th dim stored as mesh data
            raw_data.push_back(block->domain(j, 3));
    }

    // number of science variables
    nvars = block->vars.size();

    // number of geometry control points
    for (size_t j = 0; j < (size_t)(block->geometry.nctrl_pts.size()); j++)
        geom_nctrl_pts.push_back(block->geometry.nctrl_pts(j));

    // number of science variable control points
    vars_nctrl_pts.resize(nvars);
    for (size_t i = 0; i < nvars; i++)
        for (size_t j = 0; j < (size_t)(block->vars[i].nctrl_pts.size()); j++)
            vars_nctrl_pts[i].push_back(block->vars[i].nctrl_pts(j));

    // geometry control points
    for (size_t j = 0; j < (size_t)(block->geometry.ctrl_pts.rows()); j++)
    {
        p.x = block->geometry.ctrl_pts(j, 0);
        p.y = block->geometry.ctrl_pts.cols() > 2 ?
            block->geometry.ctrl_pts(j, 1) : 0.0;
        p.z = block->geometry.ctrl_pts.cols() > 2 ?
            block->geometry.ctrl_pts(j, 2) : 0.0;
        geom_ctrl_pts.push_back(p);
    }

    // science variable control points
    vars_ctrl_pts.resize(nvars);
    for (size_t i = 0; i < nvars; i++)
        for (size_t j = 0; j < (size_t)(block->vars[i].ctrl_pts.rows()); j++)
        {
            p.x = block->vars[i].ctrl_pts(j, 0);
            p.y = block->vars[i].ctrl_pts.cols() > 2 ?
                block->vars[i].ctrl_pts(j, 1) : 0.0;
            p.z = block->vars[i].ctrl_pts.cols() > 2 ?
                block->vars[i].ctrl_pts(j, 2) : 0.0;
            vars_ctrl_pts[i].push_back(p);
        }

    // approximated points
    for (size_t j = 0; j < (size_t)(block->approx.rows()); j++)
    {
        p.x = block->approx(j, 0);                      // first 3 dims stored as mesh geometry
        p.y = block->approx(j, 1);
        p.z = block->approx.cols() > 2 ?
            block->approx(j, 2) : 0.0;
        approx_pts.push_back(p);

        if (block->approx.cols() > 3)                   // 4th dim stored as mesh data
            approx_data.push_back(block->approx(j, 3));
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

    // block mins
    p.x = block->domain_mins(0);
    p.y = block->domain_mins(1);
    p.z = block->domain_mins.size() > 2 ?
        block->domain_mins(2) : 0.0;
    block_mins.push_back(p);

    // block maxs
    p.x = block->domain_maxs(0);
    p.y = block->domain_maxs(1);
    p.z = block->domain_maxs.size() > 2 ?
        block->domain_maxs(2) : 0.0;
    block_maxs.push_back(p);
}

void write_vtk_files(
        Block<real_t>* b,
        const          diy::Master::ProxyWithLink& cp)
{
    int                         nvars;              // number of science variables (excluding geometry)
    vector<int>                 nraw_pts;           // number of input points in each dim.
    vector<vec3d>               raw_pts;            // input raw data points (<= 3d)
    vector<float>               raw_data;           // input raw data values (4d)
    vector <int>                geom_nctrl_pts;     // number of control pts in each dim of geometry
    vector < vector <int> >     vars_nctrl_pts;     // number of control pts in each dim. of each science variable
    vector<vec3d>               geom_ctrl_pts;      // control points (<= 3d) in geometry
    vector < vector <vec3d> >   vars_ctrl_pts;      // control points (<= 3d) in science variables
    vector<vec3d>               approx_pts;         // aproximated data points (<= 3d)
    vector<float>               approx_data;        // approximated data values (4d)
    vector<vec3d>               err_pts;            // abs value error field
    vector<vec3d>               block_mins;         // block mins
    vector<vec3d>               block_maxs;         // block maxs

    // package rendering data
    PrepRenderingData(nraw_pts,
            raw_pts,
            raw_data,
            nvars,
            geom_nctrl_pts,
            vars_nctrl_pts,
            geom_ctrl_pts,
            vars_ctrl_pts,
            approx_pts,
            approx_data,
            err_pts,
            block_mins,
            block_maxs,
            b);

    // pad dimensions up to 3
    size_t dom_dims = geom_nctrl_pts.size();
    for (auto i = 0; i < 3 - dom_dims; i++)
    {
        geom_nctrl_pts.push_back(1);
        nraw_pts.push_back(1);
    }

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

    char filename[256];
    float *vars;
    int vardim     = 1;
    int centering  = 1;

    // write geometry control points
    sprintf(filename, "geom_control_points_gid_%d.vtk", cp.gid());

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
        sprintf(filename, "var%d_control_points_gid_%d.vtk", i, cp.gid());
        write_curvilinear_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int *dims */                                 &vars_nctrl_pts[i][0],
            /* float *pts */                                &(vars_ctrl_pts[i][0].x),
            /* int nvars */                                 0,
            /* int *vardim */                               NULL,
            /* int *centering */                            NULL,
            /* const char * const *varnames */              NULL,
            /* float **vars */                              NULL);
    }

    // write raw original points
    sprintf(filename, "initial_points_gid_%d.vtk", cp.gid());
    if (raw_data.size())
    {
        vars = &raw_data[0];
        const char* name_raw_data = "raw_data";
        write_curvilinear_mesh(
                /* const char *filename */                  filename,
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(raw_pts[0].x),
                /* int nvars */                             1,
                /* int *vardim */                           &vardim,
                /* int *centering */                        &centering,
                /* const char * const *varnames */          &name_raw_data,
                /* float **vars */                          &vars);
    }
    else
    {
        vars = &errm[0];
        const char* name_err ="error";
        write_curvilinear_mesh(
                /* const char *filename */                  filename,
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
    sprintf(filename, "approx_points_gid_%d.vtk", cp.gid());
    if (approx_data.size())
    {
        vars = &approx_data[0];
        const char* name_approx_data ="approx_data";
        write_curvilinear_mesh(
                /* const char *filename */                  filename,
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(approx_pts[0].x),
                /* int nvars */                             1,
                /* int *vardim */                           &vardim,
                /* int *centering */                        &centering,
                /* const char * const *varnames */          &name_approx_data,
                /* float **vars */                          &vars);

    }
    else
    {
        write_curvilinear_mesh(
                /* const char *filename */                  filename,
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(approx_pts[0].x),
                /* int nvars */                             0,
                /* int *vardim */                           NULL,
                /* int *centering */                        NULL,
                /* const char * const *varnames */          NULL,
                /* float **vars */                          NULL);
    }

    // write error
    sprintf(filename, "error_gid_%d.vtk", cp.gid());
    vars = &errm[0];
    const char* name_err ="error";
    write_curvilinear_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int *dims */                                 &nraw_pts[0],
            /* float *pts */                                &(err_pts[0].x),
            /* int nvars */                                 1,
            /* int *vardim */                               &vardim,
            /* int *centering */                            &centering,
            /* const char * const *varnames */              &name_err,
            /* float **vars */                              &vars);
}

int main(int argc, char ** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);       // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;                 // equivalent of MPI_COMM_WORLD

    string infile(argv[1]);                       // diy input file

    diy::Master master(world,
            1,
            -1,
            &Block<real_t>::create,
            &Block<real_t>::destroy);
    diy::ContiguousAssigner   assigner(world.size(), -1);   // number of blocks set by read_blocks()

    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block<real_t>::load);
    int nblocks = master.size();                            // number of local blocks
    std::cout << nblocks << " blocks read from file "<< infile << "\n";

    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { write_vtk_files(b, cp); });
}
