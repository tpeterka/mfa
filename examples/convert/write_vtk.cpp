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

// make combinations of min, max corner vertices in index and real space
void CellVertices(
        int             ndom_dims,                      // number of domain dimensions
        vec3d&          min,                            // min corner
        vec3d&          max,                            // max corner
        vector<vec3d>&  tensor_pts)                     // (output) vertices
{
    vec3d p;

    p.x = min.x;
    p.y = min.y;
    p.z = min.z;
    tensor_pts.push_back(p);

    p.x = max.x;
    tensor_pts.push_back(p);

    if (ndom_dims > 1)
    {
        p.y = max.y;
        tensor_pts.push_back(p);

        p.x = min.x;
        tensor_pts.push_back(p);

        if (ndom_dims > 2)
        {
            p.x = min.x;
            p.y = min.y;
            p.z = max.z;
            tensor_pts.push_back(p);

            p.x = max.x;
            tensor_pts.push_back(p);

            p.y = max.y;
            tensor_pts.push_back(p);

            p.x = min.x;
            tensor_pts.push_back(p);
        }
    }
}

// prep tmesh tensor extents
void PrepTmeshTensorExtents(
        int             nvars,                              // number of variables
        int             ndom_dims,                          // number of domain dimensions
        vector<vec3d>&  tensor_pts_real,                    // (output) points in real space
        vector<vec3d>&  tensor_pts_index,                   // (output) points in index space
        Block<real_t>*  block)                              // curent block
{
    for (auto j = 0; j < nvars; j++)
    {
        vec3d min_real, max_real, min_index, max_index;         // extents in real and index space

        mfa::Tmesh<real_t>& tmesh = block->vars[j].mfa_data->tmesh;

        // form extents in index and real space
        for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
        {
            min_index.x = tmesh.tensor_prods[k].knot_mins[0];
            min_real.x  = block->core_mins[0] + tmesh.all_knots[0][tmesh.tensor_prods[k].knot_mins[0]] *
                (block->core_maxs[0] - block->core_mins[0]);
            if (ndom_dims > 1)
            {
                min_index.y = tmesh.tensor_prods[k].knot_mins[1];
                min_real.y  = block->core_mins[1] + tmesh.all_knots[1][tmesh.tensor_prods[k].knot_mins[1]] *
                    (block->core_maxs[1] - block->core_mins[1]);
            }
            else
            {
                min_index.y = 0.0;
                min_real.y  = 0.0;
            }
            if (ndom_dims > 2)
            {
                min_index.z = tmesh.tensor_prods[k].knot_mins[2];
                min_real.z  = block->core_mins[2] + tmesh.all_knots[2][tmesh.tensor_prods[k].knot_mins[2]] *
                    (block->core_maxs[2] - block->core_mins[2]);
            }
            else
            {
                min_index.z = 0.0;
                min_real.z  = 0.0;
            }

            max_index.x = tmesh.tensor_prods[k].knot_maxs[0];
            max_real.x  = block->core_mins[0] + tmesh.all_knots[0][tmesh.tensor_prods[k].knot_maxs[0]] *
                (block->core_maxs[0] - block->core_mins[0]);
            if (ndom_dims > 1)
            {
                max_index.y = tmesh.tensor_prods[k].knot_maxs[1];
                max_real.y  = block->core_mins[1] + tmesh.all_knots[1][tmesh.tensor_prods[k].knot_maxs[1]] *
                    (block->core_maxs[1] - block->core_mins[1]);
            }
            else
            {
                max_index.y = 0.0;
                max_real.y  = 0.0;
            }
            if (ndom_dims > 2)
            {
                max_index.z = tmesh.tensor_prods[k].knot_maxs[2];
                max_real.z  = block->core_mins[2] + tmesh.all_knots[2][tmesh.tensor_prods[k].knot_maxs[2]] *
                    (block->core_maxs[2] - block->core_mins[2]);
            }
            else
            {
                max_index.z = 0.0;
                max_real.z  = 0.0;
            }

            // make vertex points for cells
            CellVertices(ndom_dims, min_index, max_index, tensor_pts_index);
            CellVertices(ndom_dims, min_real, max_real, tensor_pts_real);
        }   // tensor products
    }   // nvars
}

// package rendering data
void PrepRenderingData(
        vector<int>&                nraw_pts,
        vector<vec3d>&              raw_pts,
        float**&                    raw_data,
        int&                        nvars,
        vector<vec3d>&              geom_ctrl_pts,
        vector< vector <vec3d> >&   vars_ctrl_pts,
        float**&                    vars_ctrl_data,
        vector<vec3d>&              approx_pts,
        float**&                    approx_data,
        vector<vec3d>&              err_pts,
        float**&                    err_data,
        vector<int> &               nblend_pts,
        vector<vec3d>&              blend_pts,
        float**&                    blend_data,
        vector<vec3d>&              tensor_pts_real,
        vector<vec3d>&              tensor_pts_index,
        vector<int>&                ntensor_pts,
        Block<real_t>*              block,
        int&                        pt_dim)                 // (output) dimensionality of point
{
    vec3d p;

    // number of geometry dimensions and science variables
    int ndom_dims   = block->geometry.mfa_data->tmesh.tensor_prods[0].ctrl_pts.cols();          // number of geometry dims
    nvars           = block->vars.size();                       // number of science variables
    pt_dim          = block->input->pt_dim;                     // dimensionality of point

    // number of raw points
    for (size_t j = 0; j < (size_t)(block->input->ndom_pts.size()); j++)
        nraw_pts.push_back(block->input->ndom_pts(j));


    // number of output points for blend
    for (size_t j = 0; j < (size_t)(block->ndom_outpts.size()); j++)
        nblend_pts.push_back(block->ndom_outpts(j));

    // raw geometry and science variables
    raw_data = new float*[nvars];
    for (size_t j = 0; j < nvars; j++)
        raw_data[j] = new float[block->input->npts];

    for (size_t j = 0; j < (size_t)(block->input->npts); j++)
    {
        p.x = block->input->domain(j, 0);                      // first 3 dims stored as mesh geometry
        p.y = block->input->domain(j, 1);
        p.z = block->input->domain.cols() > 2 ? block->input->domain(j, 2) : 0.0;
        raw_pts.push_back(p);

        for (int k = 0; k < nvars; k++)                         // science variables
            raw_data[k][j] = block->input->domain(j, ndom_dims + k);
    }

    // --- geometry control points ---

    // compute vectors of individual control point coordinates for the tensor product
    vector<vector<float>> ctrl_pts_coords(ndom_dims);
    for (auto t = 0; t < block->geometry.mfa_data->tmesh.tensor_prods.size(); t++)                      // tensor products
    {
        TensorProduct<real_t>& tc = block->geometry.mfa_data->tmesh.tensor_prods[t];
        mfa::VolIterator vol_iter(tc.nctrl_pts);
        for (auto k = 0; k < ndom_dims; k++)                                                            // domain dimensions
        {
            int skip = 0;
            // starting knot in sequence for computing control point coordinate
            KnotIdx knot_min = tc.knot_mins[k];
            if (knot_min)
            {
                // skip knots at a deeper level than the tensor
                for (auto l = 0; l < block->geometry.mfa_data->p(k); l++)
                {
                    while (block->geometry.mfa_data->tmesh.all_knot_levels[k][knot_min - l - skip] > tc.level)
                        skip++;
                }
                knot_min -= (block->geometry.mfa_data->p(k) - 1 + skip);
            }

            for (auto j = 0; j < tc.nctrl_pts(k); j++)                      // control points
            {
                float tsum  = 0.0;
                int skip1   = skip;                                         // number of knots at a deeper level that should be skipped
                // skip knots at a deeper level than the tensor
                for (int l = 1; l < block->geometry.mfa_data->p(k) + 1; l++)
                {
                    // skip knots at a deeper level than the tensor
                    while (block->geometry.mfa_data->tmesh.all_knot_levels[k][knot_min + j + l + skip1] > tc.level)
                        skip1++;
                    tsum += block->geometry.mfa_data->tmesh.all_knots[k][knot_min + j + l + skip1];
                }
                tsum /= float(block->geometry.mfa_data->p(k));
                ctrl_pts_coords[k].push_back(block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k)));

                // debug
//                 fprintf(stderr, "t=%d k=%d j=%d tsum=%.3lf ctrl_pts_coord=%.3lf\n", t, k, j, tsum, ctrl_pts_coords[k].back());
            }   // control points
        }   // domain dimensions
    }   // tensor products

    // form the tensor product of control points from the vectors of individual coordinates
    VectorXi ofst = VectorXi::Zero(3);                              // offset of indices for current tensor
    for (auto t = 0; t < block->geometry.mfa_data->tmesh.tensor_prods.size(); t++)                      // tensor products
    {
        TensorProduct<real_t>& tc   = block->geometry.mfa_data->tmesh.tensor_prods[t];
        mfa::VolIterator vol_iter(tc.nctrl_pts);
        VectorXi ijk(ndom_dims);
        while (!vol_iter.done())                                    // control points
        {
            vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);

            if (tc.weights(vol_iter.cur_iter()) == MFA_NAW)
            {
                vol_iter.incr_iter();
                continue;
            }

            // first 3 dims stored as mesh geometry
            p.x = ctrl_pts_coords[0][ofst(0) + ijk(0)];
            if (ndom_dims < 2)
                p.y = 0.0;
            else
                p.y = ctrl_pts_coords[1][ofst(1) + ijk(1)];
            if (ndom_dims < 3)
                p.z = 0.0;
            else
                p.z = ctrl_pts_coords[2][ofst(2) + ijk(2)];
            geom_ctrl_pts.push_back(p);

            // debug
//             fprintf(stderr, "t = %d geom_ctrl_pt = [%.3lf %.3lf]\n", t, geom_ctrl_pts.back().x, geom_ctrl_pts.back().y);

            vol_iter.incr_iter();
        }       // control points
        ofst.head(ndom_dims) += tc.nctrl_pts;
    }       // tensor products

    // --- science variable control points ---

    vars_ctrl_pts.resize(nvars);
    vars_ctrl_data = new float*[nvars];
    for (size_t i = 0; i < nvars; i++)                              // science variables
    {
        size_t nctrl_pts = 0;
        for (auto t = 0; t < block->vars[i].mfa_data->tmesh.tensor_prods.size(); t++)                   // tensor products
        {
            size_t prod = 1;
            for (auto k = 0; k < ndom_dims; k++)                                                        // domain dimensions
                prod *= block->vars[i].mfa_data->tmesh.tensor_prods[t].nctrl_pts(k);
            nctrl_pts += prod;
        }
        vars_ctrl_data[i] = new float[nctrl_pts];

        // compute vectors of individual control point coordinates for the tensor product
        vector<vector<float>> ctrl_pts_coords(ndom_dims);
        for (auto t = 0; t < block->vars[i].mfa_data->tmesh.tensor_prods.size(); t++)                   // tensor products
        {
            TensorProduct<real_t>& tc = block->vars[i].mfa_data->tmesh.tensor_prods[t];
            mfa::VolIterator vol_iter(tc.nctrl_pts);
            for (auto k = 0; k < ndom_dims; k++)                                                        // domain dimensions
            {
                int skip = 0;
                // starting knot in sequence for computing control point coordinate
                KnotIdx knot_min = tc.knot_mins[k];
                if (knot_min)
                {
                    // skip knots at a deeper level than the tensor
                    for (auto l = 0; l < block->vars[i].mfa_data->p(k); l++)
                    {
                        while (block->vars[i].mfa_data->tmesh.all_knot_levels[k][knot_min - l - skip] > tc.level)
                            skip++;
                    }
                    knot_min -= (block->vars[i].mfa_data->p(k) - 1 + skip);
                }

                for (auto j = 0; j < tc.nctrl_pts(k); j++)              // control points
                {
                    float tsum  = 0.0;
                    int skip1   = skip;                                 // number of knots at a deeper level that should be skipped
                    for (auto l = 1; l < block->vars[i].mfa_data->p(k) + 1; l++)
                    {
                        // skip knots at a deeper level than the tensor
                        while (block->vars[i].mfa_data->tmesh.all_knot_levels[k][knot_min + j + l + skip1] > tc.level)
                            skip1++;
                        tsum += block->vars[i].mfa_data->tmesh.all_knots[k][knot_min + j + l + skip1];
                    }
                    tsum /= float(block->vars[i].mfa_data->p(k));
                    ctrl_pts_coords[k].push_back(block->core_mins(k) + tsum * (block->core_maxs(k) - block->core_mins(k)));

                    // debug
//                     fprintf(stderr, "t=%d k=%d j=%d tsum=%.3lf ctrl_pts_coord=%.3lf\n", t, k, j, tsum, ctrl_pts_coords[k].back());
                }   // control points
            }   // domain dimensions
        }   // tensor products

        // form the tensor product of control points from the vectors of individual coordinates
        VectorXi ofst = VectorXi::Zero(3);                              // offset of indices for current tensor
        for (auto t = 0; t < block->vars[i].mfa_data->tmesh.tensor_prods.size(); t++)                  // tensor products
        {
            TensorProduct<real_t>& tc   = block->vars[i].mfa_data->tmesh.tensor_prods[t];
            mfa::VolIterator vol_iter(tc.nctrl_pts);
            VectorXi ijk(ndom_dims);
            while (!vol_iter.done())                                        // control points
            {
                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);

                if (tc.weights(vol_iter.cur_iter()) == MFA_NAW)
                {
                    vol_iter.incr_iter();
                    continue;
                }

                // first 3 dims stored as mesh geometry
                // control point position and optionally science variable, if the total fits in 3d
                p.x = ctrl_pts_coords[0][ofst(0) + ijk(0)];
                if (ndom_dims < 2)
                {
                    p.y = tc.ctrl_pts(vol_iter.cur_iter(), 0);
                    p.z = 0.0;
                }
                else
                {
                    p.y = ctrl_pts_coords[1][ofst(1) + ijk(1)];
                    if (ndom_dims < 3)
                        p.z = tc.ctrl_pts(vol_iter.cur_iter(), 0);
                    else
                        p.z = ctrl_pts_coords[2][ofst(2) + ijk(2)];
                }
                vars_ctrl_pts[i].push_back(p);

                // science variable also stored as data
                vars_ctrl_data[i][vars_ctrl_pts[i].size() - 1] = tc.ctrl_pts(vol_iter.cur_iter(), 0);

                // debug
//                 fprintf(stderr, "t=%d ctrl_pt= [%.3lf %.3lf %.3lf]\n", t, vars_ctrl_pts[i].back().x, vars_ctrl_pts[i].back().y, vars_ctrl_data[i][vars_ctrl_pts[i].size() - 1]);

                vol_iter.incr_iter();
            }   // control points
            ofst.head(ndom_dims) += tc.nctrl_pts;
        }   // tensor products
    }   // science variables

    // approximated points
    approx_data = new float*[nvars];
    blend_data  = new float*[nvars];
    for (size_t j = 0; j < nvars; j++)
    {
        approx_data[j]  = new float[block->input->npts];
        blend_data[j]   = new float[block->blend.rows()];
    }

    for (size_t j = 0; j < (size_t)(block->approx.rows()); j++)
    {
        p.x = block->approx(j, 0);                      // first 3 dims stored as mesh geometry
        p.y = block->approx(j, 1);
        p.z = block->approx.cols() > 2 ? block->approx(j, 2) : 0.0;
        approx_pts.push_back(p);

        for (int k = 0; k < nvars; k++)                         // science variables
            approx_data[k][j] = block->approx(j, ndom_dims + k);
    }

    for (size_t j = 0; j < (size_t)(block->blend.rows()); j++)
    {
        p.x = block->blend(j, 0);                      // first 3 dims stored as mesh geometry
        p.y = block->blend(j, 1);
        p.z = block->blend.cols() > 2 ? block->blend(j, 2) : 0.0;
        blend_pts.push_back(p);

        for (int k = 0; k < nvars; k++)                         // science variables
            blend_data[k][j] = block->blend(j, ndom_dims + k);
    }

    // error points
    // error values are not normalized by data range
    err_data = new float*[nvars];
    for (size_t j = 0; j < nvars; j++)
        err_data[j]  = new float[block->input->domain.rows()];
    for (size_t j = 0; j < (size_t)(block->errs.rows()); j++)
    {
        p.x = block->errs(j, 0);
        p.y = block->errs(j, 1);
        p.z = block->errs.cols() > 2 ? block->errs(j, 2) : 0.0;
        err_pts.push_back(p);
        
        for (int k = 0; k < nvars; k++)                         // science variables
            err_data[k][j] = block->errs(j, ndom_dims + k);
    }

    // tmesh tensor extents
    ntensor_pts.resize(3);
    for (auto i = 0; i < 3; i++)
        ntensor_pts[i] = (i >= ndom_dims ? 1 : 2);
    PrepTmeshTensorExtents(nvars, ndom_dims, tensor_pts_real, tensor_pts_index, block);
}

// write vtk files for initial, approximated, control points
void write_vtk_files(
        Block<real_t>* b,
        const          diy::Master::ProxyWithLink& cp,
        int&           dom_dim,                     // (output) domain dimensionality
        int&           pt_dim)                      // (output) point dimensionality
{
    int                         nvars;              // number of science variables (excluding geometry)
    vector<int>                 nraw_pts;           // number of input points in each dim.
    vector<vec3d>               raw_pts;            // input raw data points (<= 3d)
    float**                     raw_data;           // input raw data values (4d)
    vector<vec3d>               geom_ctrl_pts;      // control points (<= 3d) in geometry
    vector < vector <vec3d> >   vars_ctrl_pts;      // control points (<= 3d) in science variables
    float**                     vars_ctrl_data;     // control point data values (4d)
    vector<vec3d>               approx_pts;         // approximated data points (<= 3d)
    float**                     approx_data;        // approximated data values (4d)
    vector<vec3d>               err_pts;            // abs value error field
    float**                     err_data;           // abs value error values (for each science variable)
    vector<int>                 nblend_pts;         // number of out points in each dim.
    vector<vec3d>               blend_pts;          // blended data points (<= 3d)
    vector<vec3d>               tensor_pts_real;    // tmesh tensor product extents in real space
    vector<vec3d>               tensor_pts_index;   // tmesh tensor product extents in index space
    vector<int>                 ntensor_pts;        // number of tensor extent points in each dim.
    float**                     blend_data;

    // package rendering data
    PrepRenderingData(nraw_pts,
                      raw_pts,
                      raw_data,
                      nvars,
                      geom_ctrl_pts,
                      vars_ctrl_pts,
                      vars_ctrl_data,
                      approx_pts,
                      approx_data,
                      err_pts,
                      err_data,
                      nblend_pts,
                      blend_pts,
                      blend_data,
                      tensor_pts_real,
                      tensor_pts_index,
                      ntensor_pts,
                      b,
                      pt_dim);

    bool structured = b->input->structured;

    // pad dimensions up to 3
    dom_dim = b->dom_dim;
    for (auto i = 0; i < 3 - dom_dim; i++)
    {
        nraw_pts.push_back(1);
        nblend_pts.push_back(1); // used for blending only in 2d?
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

    // write raw original points
    sprintf(filename, "initial_points_gid_%d.vtk", cp.gid());
    if (raw_pts.size())
    {
        if (structured)
        {
            write_curvilinear_mesh(
                /* const char *filename */                  filename,
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(raw_pts[0].x),
                /* int nvars */                             nvars,
                /* int *vardim */                           vardims,
                /* int *centering */                        centerings,
                /* const char * const *varnames */          varnames,
                /* float **vars */                          raw_data);
        }
        else
        {
            write_point_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int npts */                                  raw_pts.size(),
            /* float *pts */                                &(raw_pts[0].x),
            /* int nvars */                                 nvars,
            /* int *vardim */                               vardims,
            /* const char * const *varnames */              varnames,
            /* float **vars */                              raw_data);
        }  
    }

    // write approx points
    sprintf(filename, "approx_points_gid_%d.vtk", cp.gid());
    if (approx_pts.size())
    {
        if (structured)
        {
            write_curvilinear_mesh(
                /* const char *filename */                  filename,
                /* int useBinary */                         0,
                /* int *dims */                             &nraw_pts[0],
                /* float *pts */                            &(approx_pts[0].x),
                /* int nvars */                             nvars,
                /* int *vardim */                           vardims,
                /* int *centering */                        centerings,
                /* const char * const *varnames */          varnames,
                /* float **vars */                          approx_data);
        }
        else
        {
            write_point_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int npts */                                  approx_pts.size(),
            /* float *pts */                                &(approx_pts[0].x),
            /* int nvars */                                 nvars,
            /* int *vardim */                               vardims,
            /* const char * const *varnames */              varnames,
            /* float **vars */                              approx_data);
        }   
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
        

    if (blend_pts.size())
    {
        sprintf(filename, "blend_gid_%d.vtk", cp.gid());
        write_curvilinear_mesh(
                /* const char *filename */                  filename,
                /* int useBinary */                         0,
                /* int *dims */                             &nblend_pts[0],
                /* float *pts */                            &(blend_pts[0].x),
                /* int nvars */                             nvars,
                /* int *vardim */                           vardims,
                /* int *centering */                        centerings,
                /* const char * const *varnames */          varnames,
                /* float **vars */                          blend_data);
    }

    // write error
    sprintf(filename, "error_gid_%d.vtk", cp.gid());
    if (err_pts.size())
    {
        if (structured)
        {
            write_curvilinear_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int *dims */                                 &nraw_pts[0],
            /* float *pts */                                &(err_pts[0].x),
            /* int nvars */                                 nvars,
            /* int *vardim */                               vardims,
            /* int *centering */                            centerings,
            /* const char * const *varnames */              varnames,
            /* float **vars */                              err_data);
        }
        else
        {
            write_point_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int npts */                                  err_pts.size(),
            /* float *pts */                                &(err_pts[0].x),
            /* int nvars */                                 nvars,
            /* int *vardim */                               vardims,
            /* const char * const *varnames */              varnames,
            /* float **vars */                              err_data);
        }  
    }
        

    // write tensor product extents
    int pts_per_cell = pow(2, dom_dim);
    int ncells = tensor_pts_real.size() / pts_per_cell;
    vector<int> cell_types(ncells);
    for (auto i = 0; i < cell_types.size(); i++)
    {
        if (dom_dim == 1)
            cell_types[i] = VISIT_LINE;
        else if (dom_dim == 2)
            cell_types[i] = VISIT_QUAD;
        else
            cell_types[i] = VISIT_HEXAHEDRON;
    }
    vector<float> tensor_data(tensor_pts_real.size(), 1.0); // tensor data set to fake value
    vector<int> conn(tensor_pts_real.size());               // connectivity
    for (auto i = 0; i < conn.size(); i++)
        conn[i] = i;
    vars = &tensor_data[0];
    sprintf(filename, "tensor_real_gid_%d.vtk", cp.gid());
    const char* name_tensor ="tensor0";

    // in real space
    write_unstructured_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int npts */                                  tensor_pts_real.size(),
            /* float *pts */                                &(tensor_pts_real[0].x),
            /* int ncells */                                ncells,
            /* int *celltypes */                            &cell_types[0],
            /* int *conn */                                 &conn[0],
            /* int nvars */                                 1,
            /* int *vardim */                               &vardim,
            /* int *centering */                            &centering,
            /* const char * const *varnames */              &name_tensor,
            /* float **vars */                              &vars);

    // in index space
    sprintf(filename, "tensor_index_gid_%d.vtk", cp.gid());
    write_unstructured_mesh(
            /* const char *filename */                      filename,
            /* int useBinary */                             0,
            /* int npts */                                  tensor_pts_index.size(),
            /* float *pts */                                &(tensor_pts_index[0].x),
            /* int ncells */                                ncells,
            /* int *celltypes */                            &cell_types[0],
            /* int *conn */                                 &conn[0],
            /* int nvars */                                 1,
            /* int *vardim */                               &vardim,
            /* int *centering */                            &centering,
            /* const char * const *varnames */              &name_tensor,
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
        delete[] blend_data[j];
        delete[] err_data[j];
    }
    delete[] raw_data;
    delete[] vars_ctrl_data;
    delete[] approx_data;
    delete[] blend_data;
    delete[] err_data;
}

// generate analytical test data and write to vtk
void test_and_write(Block<real_t>*                      b,
                    const diy::Master::ProxyWithLink&   cp,
                    string                              input,
                    DomainArgs&                         args)
{
    int                         nvars;              // number of science variables
    vector<int>                 ntest_pts;          // number of test points in each dim.
    vector<vec3d>               true_pts;           // locations of true points (<= 3d) (may include data value in 2nd or 3rd coord)
    vector<vec3d>               test_pts;           // locations of test points (<= 3d) (may include data value in 2nd or 3rd coord)
    float**                     true_data;          // true data values (4d)
    float**                     test_data;          // test data values (4d)

    DomainArgs* a   = &args;

    nvars = b->vars.size();
    if (!b->dom_dim)
        b->dom_dim =  b->input->ndom_pts.size();

    // default args for evaluating analytical functions
    for (auto i = 0; i < nvars; i++)
    {
        a->f[i] = 1.0;
        if (input == "sine")
            a->s[i] = i + 1;
        if (input == "sinc")
            a->s[i] = 10.0 * (i + 1);
    }

    // number of test points
    size_t tot_ntest = 1;
    for (auto j = 0; j < b->dom_dim; j++)
    {
        ntest_pts.push_back(a->ndom_pts[j]);
        tot_ntest *= a->ndom_pts[j];
    }

    true_pts.resize(tot_ntest);
    test_pts.resize(tot_ntest);

    // allocate variable data
    true_data = new float*[nvars];
    test_data = new float*[nvars];
    for (size_t j = 0; j < nvars; j++)
    {
        true_data[j] = new float[tot_ntest];
        test_data[j] = new float[tot_ntest];
    }

    // compute the norms of analytical errors synthetic function w/o noise at different domain points than the input
    real_t L1, L2, Linf;                                // L-1, 2, infinity norms
    b->analytical_error(cp, input, L1, L2, Linf, args, true, true_pts, true_data, test_pts, test_data);

    // print analytical errors
    fprintf(stderr, "\n------ Analytical error norms -------\n");
    fprintf(stderr, "L-1        norm = %e\n", L1);
    fprintf(stderr, "L-2        norm = %e\n", L2);
    fprintf(stderr, "L-infinity norm = %e\n", Linf);
    fprintf(stderr, "-------------------------------------\n\n");

    // pad dimensions up to 3
    for (auto i = 0; i < 3 - b->dom_dim; i++)
        ntest_pts.push_back(1);

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

    // write true points
    char filename[256];
    sprintf(filename, "true_points_gid_%d.vtk", cp.gid());
    write_curvilinear_mesh(
            /* const char *filename */                  filename,
            /* int useBinary */                         0,
            /* int *dims */                             &ntest_pts[0],
            /* float *pts */                            &(true_pts[0].x),
            /* int nvars */                             nvars,
            /* int *vardim */                           vardims,
            /* int *centering */                        centerings,
            /* const char * const *varnames */          varnames,
            /* float **vars */                          true_data);

    // write test points
    sprintf(filename, "test_points_gid_%d.vtk", cp.gid());
    write_curvilinear_mesh(
            /* const char *filename */                  filename,
            /* int useBinary */                         0,
            /* int *dims */                             &ntest_pts[0],
            /* float *pts */                            &(test_pts[0].x),
            /* int nvars */                             nvars,
            /* int *vardim */                           vardims,
            /* int *centering */                        centerings,
            /* const char * const *varnames */          varnames,
            /* float **vars */                          test_data);


    delete[] vardims;
    for (int i = 0; i < nvars; i++)
        delete[] varnames[i];
    delete[] varnames;
    delete[] centerings;
    for (int j = 0; j < nvars; j++)
    {
        delete[] true_data[j];
        delete[] test_data[j];
    }
    delete[] true_data;
    delete[] test_data;
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
    vector<vec3d>               geom_ctrl_pts;      // control points (<= 3d) in geometry
    vector < vector <vec3d> >   vars_ctrl_pts;      // control points (<= 3d) in science variables
    float**                     vars_ctrl_data;     // control point data values (4d)
    vector<vec3d>               approx_pts;         // aproximated data points (<= 3d)
    float**                     approx_data;        // approximated data values (4d)
    vector<vec3d>               err_pts;            // abs value error field
    string                      input  = "sine";        // input dataset
    int                         ntest  = 0;             // number of input test points in each dim for analytical error tests
    string                      infile = "approx.out";  // diy input file
    bool                        help;                   // show help
    int                         dom_dim, pt_dim;        // domain and point dimensionality, respectively

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

    // arguments for analytical functions
    DomainArgs d_args(dom_dim, pt_dim);

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
    for (int i = 0; i < dom_dim; i++)
        d_args.ndom_pts[i] = ntest;

    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { test_and_write(b, cp, input, d_args); });
}
