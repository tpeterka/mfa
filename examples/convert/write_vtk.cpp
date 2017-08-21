#include "mfa/mfa.hpp"
#include "../block.hpp"
#include <iostream>

#include <diy/master.hpp>
#include <diy/io/block.hpp>

#include "writer.hpp"
// 3d point or vector
struct vec3d {
  float x, y, z;
  float mag() { return sqrt(x*x+y*y+z*z) ;}
};

// package rendering data
void PrepRenderingData(vector<int>&   nraw_pts,
                       vector<vec3d>& raw_pts,
                       vector<int>&   nctrl_pts,
                       vector<vec3d>& ctrl_pts,
                       vector<vec3d>& approx_pts,
                       vector<vec3d>& err_pts,
                       vector<vec3d>& mins,
                       vector<vec3d>& maxs,
                       int            nblocks,
                       diy::Master&   master)
{
    for (int i = 0; i < nblocks; i++)          // blocks
    {
        vec3d p;

        // number of raw points
        for (size_t j = 0; j < (size_t)(master.block<Block>(i)->ndom_pts.size()); j++)
            nraw_pts.push_back(master.block<Block>(i)->ndom_pts(j));
        // raw points
        for (size_t j = 0; j < (size_t)(master.block<Block>(i)->domain.rows()); j++)
        {
            p.x = master.block<Block>(i)->domain(j, 0);
            p.y = master.block<Block>(i)->domain(j, 1);
            p.z = master.block<Block>(i)->domain.cols() > 2 ?
                master.block<Block>(i)->domain(j, 2) : 0.0;
            raw_pts.push_back(p);
        }
        // number of control points
        for (size_t j = 0; j < (size_t)(master.block<Block>(i)->nctrl_pts.size()); j++)
            nctrl_pts.push_back(master.block<Block>(i)->nctrl_pts(j));
        // control points
        for (size_t j = 0; j < (size_t)(master.block<Block>(i)->ctrl_pts.rows()); j++)
        {
            p.x = master.block<Block>(i)->ctrl_pts(j, 0);
            p.y = master.block<Block>(i)->ctrl_pts(j, 1);
            p.z = master.block<Block>(i)->ctrl_pts.cols() > 2 ?
                master.block<Block>(i)->ctrl_pts(j, 2) : 0.0;
            ctrl_pts.push_back(p);
        }
        // approximated points
        for (size_t j = 0; j < (size_t)(master.block<Block>(i)->approx.rows()); j++)
        {
            p.x = master.block<Block>(i)->approx(j, 0);
            p.y = master.block<Block>(i)->approx(j, 1);
            p.z = master.block<Block>(i)->approx.cols() > 2 ?
                master.block<Block>(i)->approx(j, 2) : 0.0;
            approx_pts.push_back(p);
        }
        // error points
        for (size_t j = 0; j < (size_t)(master.block<Block>(i)->errs.rows()); j++)
        {
            p.x = master.block<Block>(i)->errs(j, 0);
            p.y = master.block<Block>(i)->errs(j, 1);
            p.z = master.block<Block>(i)->errs.cols() > 2 ?
                master.block<Block>(i)->errs(j, 2) : 0.0;
            err_pts.push_back(p);
        }
        // block mins
        p.x = master.block<Block>(i)->domain_mins(0);
        p.y = master.block<Block>(i)->domain_mins(1);
        p.z = master.block<Block>(i)->domain_mins.size() > 2 ?
            master.block<Block>(i)->domain_mins(2) : 0.0;
        mins.push_back(p);
        // block maxs
        p.x = master.block<Block>(i)->domain_maxs(0);
        p.y = master.block<Block>(i)->domain_maxs(1);
        p.z = master.block<Block>(i)->domain_maxs.size() > 2 ?
            master.block<Block>(i)->domain_maxs(2) : 0.0;
        maxs.push_back(p);
    }
}


int main(int argc, char ** argv)
{
  // initialize MPI
  diy::mpi::environment  env(argc, argv); // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
  diy::mpi::communicator world;           // equivalent of MPI_COMM_WORLD

  int nblocks     = 1;                     // number of local blocks
  int tot_blocks  = nblocks * world.size();
  int mem_blocks  = -1;                    // everything in core for now
  int num_threads = 1;                     // needed in order to do timing

  float norm_err_limit = 1.0;             // maximum normalized errro limit


  vector<int>   nraw_pts;                                 // number of input points in each dim.
  vector<vec3d> raw_pts;                                  // input raw data points
  vector<int>   nctrl_pts;                                // number of control pts in each dim.
  vector<vec3d> ctrl_pts;                                 // control points
  vector<vec3d> approx_pts;                               // aproximated data points
  vector<vec3d> err_pts;                                  // abs value error field
  vector<vec3d> mins;                                     // block mins
  vector<vec3d> maxs;                                     // block maxs
  string infile(argv[1]);

  diy::Master               master(world,
                                   -1,
                                   -1,
                                   &Block::create,
                                   &Block::destroy);
  diy::ContiguousAssigner   assigner(world.size(), -1);   // number of blocks set by read_blocks()
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
                    mins,
                    maxs,
                    nblocks,
                    master);

  if (nctrl_pts.size()==2)
    nctrl_pts.push_back(1); // add a third dim

  if (nraw_pts.size()==2)
    nraw_pts.push_back(1); // add a third dim

  // write first control points
  write_curvilinear_mesh(/* const char *filename */ "control_points.vtk",
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


  int centering[1]={1}; // so it is point data
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
                         /* int nvars */ 0,
                         /* int *vardim */ NULL,
                         /* int *centering */ NULL,
                         /* const char * const *varnames */NULL,
                         /* float **vars */ NULL);

}
