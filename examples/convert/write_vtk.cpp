#include "mfa/mfa.hpp"
#include "../block.hpp"
#include <iostream>

#include <diy/master.hpp>
#include <diy/io/block.hpp>

#include "writer.hpp"

// 3d point or vector
struct vec3d
{
    float x, y, z;
    float mag() { return sqrt(x*x+y*y+z*z) ;}
};

// package rendering data
void PrepRenderingData(
        vector<int>&   nraw_pts,
        vector<vec3d>& raw_pts,
        vector<int>&   nctrl_pts,
        vector<vec3d>& ctrl_pts,
        vector<vec3d>& approx_pts,
        vector<vec3d>& err_pts,
        vector<int>&   nknot_pts,
        vector<vec3d>& knot_pts,
        vector<vec3d>& block_mins,
        vector<vec3d>& block_maxs,
        int            nblocks,
        diy::Master&   master)
{
    for (int i = 0; i < nblocks; i++)          // blocks
    {
        vec3d p;
        Block* block = master.block<Block>(i);

        // number of raw points
       for (size_t j = 0; j < (size_t)(block->ndom_pts.size()); j++)
            nraw_pts.push_back(block->ndom_pts(j));

        // raw points
        for (size_t j = 0; j < (size_t)(block->domain.rows()); j++)
        {
            p.x = block->domain(j, 0);
            p.y = block->domain(j, 1);
            p.z = block->domain.cols() > 2 ?
                block->domain(j, 2) : 0.0;
            raw_pts.push_back(p);
        }

        // number of control points
        for (size_t j = 0; j < (size_t)(block->nctrl_pts.size()); j++)
            nctrl_pts.push_back(block->nctrl_pts(j));

        // control points
        for (size_t j = 0; j < (size_t)(block->ctrl_pts.rows()); j++)
        {
            p.x = block->ctrl_pts(j, 0);
            p.y = block->ctrl_pts(j, 1);
            p.z = block->ctrl_pts.cols() > 2 ?
                block->ctrl_pts(j, 2) : 0.0;
            ctrl_pts.push_back(p);
        }

        // approximated points
        for (size_t j = 0; j < (size_t)(block->approx.rows()); j++)
        {
            p.x = block->approx(j, 0);
            p.y = block->approx(j, 1);
            p.z = block->approx.cols() > 2 ?
                block->approx(j, 2) : 0.0;
            approx_pts.push_back(p);
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

        // number of knot points
        for (size_t j = 0; j < (size_t)(block->p.size()); j++)
            nknot_pts.push_back(block->nctrl_pts(j) - block->p(j) + 1);
        nknot_pts.push_back(2);                         // 2 layers of points

        // knot points
        // range values go from just below 0 to just above the max error
        // NB: only for 1d and 2d domains
        for (size_t j = 0; j < (size_t)(block->span_mins.rows()); j++)
        {
            p.x = block->domain(block->span_mins(j), 0);
            if (block->domain.cols() > 2)
            {
                p.y = block->domain(block->span_mins(j), 1);
                p.z = -max_err * 0.1;
            }
            else
            {
                p.y = -max_err * 0.1;
                p.z = 0.0;
            }
            knot_pts.push_back(p);

            // if end of row is reached, add one knot point for max
            if ((j + 1) % (block->nctrl_pts(0) - block->p(0)) == 0)
            {
                p.x = block->domain(block->span_maxs(j), 0);
                if (block->domain.cols() > 2)
                {
                    p.y = block->domain(block->span_mins(j), 1);
                    p.z = -max_err * 0.1;
                }
                else
                {
                    p.y = -max_err * 0.1;
                    p.z = 0.0;
                }
                knot_pts.push_back(p);
            }
        }
        // add one more row of knot points for max
        if (block->domain.cols() > 2)
        {
            // so = starting span of last row of spans
            size_t so = (block->nctrl_pts(0) - block->p(0)) * (block->nctrl_pts(1) - block->p(1) - 1);
            for (size_t j = 0; j < (size_t)(block->nctrl_pts(0) - block->p(0) + 1); j++)
            {
                if (j < block->nctrl_pts(0) - block->p(0))
                        p.x = block->domain(block->span_mins(so + j), 0);
                else
                        p.x = block->domain(block->span_maxs(so + j - 1), 0);
                if (block->domain.cols() > 2)
                {
                    p.y = block->domain(block->span_maxs(so), 1);
                    p.z = -max_err * 0.1;
                }
                else
                {
                    p.y = -max_err * 0.1;
                    p.z = 0.0;
                }
                knot_pts.push_back(p);
            }
        }
        // add an upper layer of points just above the max error
        size_t nknots = knot_pts.size();
        for (size_t j = 0; j < nknots; j++)
        {
            p.x = knot_pts[j].x;
            if (block->domain.cols() > 2)
            {
                p.y = knot_pts[j].y;
                p.z = max_err * 1.1;
            }
            else
            {
                p.y = max_err * 1.1;
                p.z = 0;
            }
            knot_pts.push_back(p);
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
}


int main(int argc, char ** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);       // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;                 // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                          // number of local blocks
    int tot_blocks  = nblocks * world.size();
    int mem_blocks  = -1;                         // everything in core for now
    int num_threads = 1;                          // needed in order to do timing

    float norm_err_limit = 1.0;                   // maximum normalized errro limit


    vector<int>   nraw_pts;                       // number of input points in each dim.
    vector<vec3d> raw_pts;                        // input raw data points
    vector<int>   nctrl_pts;                      // number of control pts in each dim.
    vector<vec3d> ctrl_pts;                       // control points
    vector<vec3d> approx_pts;                     // aproximated data points
    vector<vec3d> err_pts;                        // abs value error field
    vector<int>   nknot_pts;                      // number of knot span points in each dim.
    vector<vec3d> knot_pts;                       // knot span points
    vector<vec3d> block_mins;                     // block mins
    vector<vec3d> block_maxs;                     // block maxs
    string infile(argv[1]);

    diy::Master               master(world,
            -1,
            -1,
            &Block::create,
            &Block::destroy);
    diy::ContiguousAssigner   assigner(world.size(),
            -1);                                  // number of blocks set by read_blocks()
    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block::load);
    nblocks = master.size();
    std::cout << nblocks << " blocks read from file "<< infile << "\n";

    // package rendering data
    PrepRenderingData(nraw_pts,
            raw_pts,
            nctrl_pts,
            ctrl_pts,
            approx_pts,
            err_pts,
            nknot_pts,
            knot_pts,
            block_mins,
            block_maxs,
            nblocks,
            master);

    // pad dimensions up to 3
    size_t dom_dims = nctrl_pts.size();
    for (auto i = 0; i < 3 - dom_dims; i++)
    {
        nctrl_pts.push_back(1);
        nraw_pts.push_back(1);
        nknot_pts.push_back(1);
    }

    // write first control points
    write_curvilinear_mesh(
            /* const char *filename */ "control_points.vtk",
            /* int useBinary */ 0,
            /* int *dims */ &nctrl_pts[0],
            /* float *pts */ &(ctrl_pts[0].x),
            /* int nvars */ 0,
            /* int *vardim */ NULL,
            /* int *centering */ NULL,
            /* const char * const *varnames */NULL,
            /* float **vars */ NULL);

    // write error as a new variable (z dimension, or maybe magnitude?)
    std::vector<float> errm(err_pts.size());
    for (size_t i=0; i<err_pts.size(); i++)
    {
        errm[i] = err_pts[i].z;
    }
    const char * name_err ="error";


    int centering[1] = {1}; // so it is point data

    // write then raw original points
    int vardim[1] = {1};
    float * pval[1] =  { &errm[0] };

    write_curvilinear_mesh(/* const char *filename */ "initial_points.vtk",
            /* int useBinary */ 0,
            /* int *dims */ &nraw_pts[0],
            /* float *pts */ &(raw_pts[0].x),
            /* int nvars */ 1,
            /* int *vardim */ vardim,
            /* int *centering */ centering,
            /* const char * const *varnames */ &name_err,
            /* float **vars */ pval);

    // write then approx points
    write_curvilinear_mesh(/* const char *filename */ "approx_points.vtk",
            /* int useBinary */ 0,
            /* int *dims */ &nraw_pts[0],
            /* float *pts */ &(approx_pts[0].x),
            /* int nvars */ 0,
            /* int *vardim */ NULL,
            /* int *centering */ NULL,
            /* const char * const *varnames */NULL,
            /* float **vars */ NULL);

    // write then error
    write_curvilinear_mesh(/* const char *filename */ "error.vtk",
            /* int useBinary */ 0,
            /* int *dims */ &nraw_pts[0],
            /* float *pts */ &(err_pts[0].x),
            /* int nvars */ 1,
            /* int *vardim */ vardim,
            /* int *centering */ centering,
            /* const char * const *varnames */ &name_err,
            /* float **vars */ pval);

    // write knot points
    write_curvilinear_mesh(
            "knots.vtk",                            // filename
            0,                                      // binary
            &nknot_pts[0],                          // dims
            &(knot_pts[0].x),                       // points
            0,                                      // nvars
            NULL,                                   // vardim
            NULL,                                   // centering
            NULL,                                   // varnames
            NULL);                                  // vars
}
