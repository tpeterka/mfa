//---------------------------------------------------------------------------
//
// the C++ part of mfa rendering program
// consists of reading the diy file, preparing the data structures
// of geometry to be rendered, and printing the mfa statistics
//
// essentially everything except the rendering, which is web-based
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
//--------------------------------------------------------------------------
#include <stdio.h>

#include <nan.h>

#include "mfa/mfa.hpp"
#include "../examples/block.hpp"

#include <diy/master.hpp>
#include <diy/io/block.hpp>

using namespace v8;

// 3d point or vector
struct vec3d {
  float x, y, z;
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

// main
NAN_METHOD(Main)
{
    static bool first_time = true;

    vector<int>   nraw_pts;                                 // number of input points in each dim.
    vector<vec3d> raw_pts;                                  // input raw data points
    vector<int>   nctrl_pts;                                // number of control pts in each dim.
    vector<vec3d> ctrl_pts;                                 // control points
    vector<vec3d> approx_pts;                               // aproximated data points
    vector<vec3d> err_pts;                                  // abs value error field
    vector<vec3d> mins;                                     // block mins
    vector<vec3d> maxs;                                     // block maxs

    Nan::HandleScope scope;

    // parse info
    if (info.Length() < 7)                                  // do not count the command name
    {
        fprintf(stderr, "Usage: draw(<filename>, <nraw_pts> <raw_pts>, <nctrl_pts>, "
                "<ctrl_pts>, <approx_pts>, <bbs>\n");
        return;
    }

    // init diy and read the file
    int nblocks;                                            // total number of blocks
    string infile(*Nan::Utf8String(info[0]));                // input file name
    diy::mpi::environment* env;
    if (first_time)
        env = new diy::mpi::environment(0, 0);
    diy::mpi::communicator    world;
    diy::Master               master(world,
                                     -1,
                                     -1,
                                     &Block::create,
                                     &Block::destroy);
    diy::ContiguousAssigner   assigner(world.size(), -1);   // number of blocks set by read_blocks()
    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block::load);
    nblocks = master.size();
    fprintf(stderr, "%d blocks read from file %s\n", nblocks, infile.c_str());

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

    // copy rendering data to output javascript arrays
    Handle<Array> js_nraw_pts       = Handle<Array>::Cast(info[1]);
    Handle<Array> js_raw_pts        = Handle<Array>::Cast(info[2]);
    Handle<Array> js_nctrl_pts      = Handle<Array>::Cast(info[3]);
    Handle<Array> js_ctrl_pts       = Handle<Array>::Cast(info[4]);
    Handle<Array> js_approx_pts     = Handle<Array>::Cast(info[5]);
    Handle<Array> js_err_pts        = Handle<Array>::Cast(info[6]);
    Handle<Array> js_bbs            = Handle<Array>::Cast(info[7]);

  for (size_t i = 0; i < nraw_pts.size(); i++)
        js_nraw_pts->Set(i, Nan::New(nraw_pts[i]));
    for (size_t i = 0; i < raw_pts.size(); i++)
    {
        js_raw_pts->Set(i * 3    , Nan::New(raw_pts[i].x));
        js_raw_pts->Set(i * 3 + 1, Nan::New(raw_pts[i].y));
        js_raw_pts->Set(i * 3 + 2, Nan::New(raw_pts[i].z));
    }
    for (size_t i = 0; i < nctrl_pts.size(); i++)
        js_nctrl_pts->Set(i, Nan::New(nctrl_pts[i]));
    for (size_t i = 0; i < ctrl_pts.size(); i++)
    {
        js_ctrl_pts->Set(i * 3    , Nan::New(ctrl_pts[i].x));
        js_ctrl_pts->Set(i * 3 + 1, Nan::New(ctrl_pts[i].y));
        js_ctrl_pts->Set(i * 3 + 2, Nan::New(ctrl_pts[i].z));
    }
    for (size_t i = 0; i < approx_pts.size(); i++)
    {
        js_approx_pts->Set(i * 3    , Nan::New(approx_pts[i].x));
        js_approx_pts->Set(i * 3 + 1, Nan::New(approx_pts[i].y));
        js_approx_pts->Set(i * 3 + 2, Nan::New(approx_pts[i].z));
    }
    for (size_t i = 0; i < err_pts.size(); i++)
    {
        js_err_pts->Set(i * 3    , Nan::New(err_pts[i].x));
        js_err_pts->Set(i * 3 + 1, Nan::New(err_pts[i].y));
        js_err_pts->Set(i * 3 + 2, Nan::New(err_pts[i].z));
    }
    for (size_t i = 0; i < mins.size(); i++)
    {
        js_bbs->Set(i * 6    , Nan::New(mins[i].x));
        js_bbs->Set(i * 6 + 1, Nan::New(mins[i].y));
        js_bbs->Set(i * 6 + 2, Nan::New(mins[i].z));
        js_bbs->Set(i * 6 + 3, Nan::New(maxs[i].x));
        js_bbs->Set(i * 6 + 4, Nan::New(maxs[i].y));
        js_bbs->Set(i * 6 + 5, Nan::New(maxs[i].z));
    }

    first_time = false;
    info.GetReturnValue().Set(0);
}

void Init(Handle<Object> exports)
{
    exports->Set(Nan::New("draw").ToLocalChecked(),
                 Nan::New<FunctionTemplate>(Main)->GetFunction());
}

NODE_MODULE(draw, Init)
