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

#include "mfa/types.hpp"
#include "../examples/simple/block.hpp"

#include <diy/master.hpp>
#include <diy/io/block.hpp>

using namespace v8;

// 3d point or vector
struct vec3d {
  float x, y, z;
};

// package rendering data
void PrepRenderingData(vector<vec3d>& raw_pts,
                       vector<vec3d>& ctrl_pts,
                       vector<vec3d>& approx_pts,
                       vector<vec3d>& mins,
                       vector<vec3d>& maxs,
                       int            nblocks,
                       diy::Master&   master)
{
    int n;

    for (int i = 0; i < nblocks; i++)          // blocks
    {
        vec3d p;
        // raw points
        for (size_t j = 0; j < master.block<Block>(i)->domain.size(); j++)
        {
            p.x = master.block<Block>(i)->domain[j][0];
            p.y = master.block<Block>(i)->domain_dim >= 2 ?
                master.block<Block>(i)->domain[j][1] : master.block<Block>(i)->range[j];
            p.z = master.block<Block>(i)->domain_dim >= 2 ?
                master.block<Block>(i)->range[j] : 0.0;
            raw_pts.push_back(p);
        }
        // control points
        for (size_t j = 0; j < master.block<Block>(i)->ctrl_pts.size(); j++)
        {
            p.x = master.block<Block>(i)->ctrl_pts[j][0];
            p.y = master.block<Block>(i)->ctrl_pts[j][1];
            p.z = master.block<Block>(i)->domain_dim >= 2 ?
                master.block<Block>(i)->ctrl_pts[j][2] : 0.0;
            ctrl_pts.push_back(p);
        }
        // approximated points
        for (size_t j = 0; j < master.block<Block>(i)->approx.size(); j++)
        {
            p.x = master.block<Block>(i)->approx[j][0];
            p.y = master.block<Block>(i)->approx[j][1];
            p.z = master.block<Block>(i)->domain_dim >= 2 ?
                master.block<Block>(i)->approx[j][2] : 0.0;
            approx_pts.push_back(p);
        }
        // block mins
        p.x = master.block<Block>(i)->domain_mins[0];
        p.y = master.block<Block>(i)->domain_dim >= 2 ?
            master.block<Block>(i)->domain_mins[1] : master.block<Block>(i)->range_min;
        p.z = master.block<Block>(i)->domain_dim >= 2 ?
            master.block<Block>(i)->range_min : 0.0;
        mins.push_back(p);
        // block maxs
        p.x = master.block<Block>(i)->domain_maxs[0];
        p.y = master.block<Block>(i)->domain_dim >= 2 ?
            master.block<Block>(i)->domain_maxs[1] : master.block<Block>(i)->range_max;
        p.z = master.block<Block>(i)->domain_dim >= 2 ?
            master.block<Block>(i)->range_max : 0.0;
        maxs.push_back(p);
    }
}

// main
NAN_METHOD(Main)
{
    static bool first_time = true;

    float domain_min[3], domain_max[3];                     // global domain extents
    float range_min[3], range_max[3];                       // global range extents
    vector<vec3d> raw_pts;                                  // input raw data points
    vector<vec3d> ctrl_pts;                                 // control points
    vector<vec3d> approx_pts;                               // aproximated data points
    vector<vec3d> mins;                                     // block mins
    vector<vec3d> maxs;                                     // block maxs

    NanScope();

    // parse args
    if (args.Length() < 5)                                  // do not count the command name
    {
        fprintf(stderr, "Usage: draw(<filename>, <raw_pts>, <ctrl_pts>, <approx_pts>, <bbs>\n");
        NanReturnUndefined();
    }

    // init diy and read the file
    int nblocks;                                            // total number of blocks
    string infile(*NanAsciiString(args[0]));                // input file name
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
    PrepRenderingData(raw_pts,
                      ctrl_pts,
                      approx_pts,
                      mins,
                      maxs,
                      nblocks,
                      master);

    // copy rendering data to output javascript arrays
    Handle<Array> js_raw_pts        = Handle<Array>::Cast(args[1]);
    Handle<Array> js_ctrl_pts       = Handle<Array>::Cast(args[2]);
    Handle<Array> js_approx_pts     = Handle<Array>::Cast(args[3]);
    Handle<Array> js_bbs            = Handle<Array>::Cast(args[4]);

    for (size_t i = 0; i < raw_pts.size(); i++)
    {
        js_raw_pts->Set(i * 3    , NanNew(raw_pts[i].x));
        js_raw_pts->Set(i * 3 + 1, NanNew(raw_pts[i].y));
        js_raw_pts->Set(i * 3 + 2, NanNew(raw_pts[i].z));
    }
    for (size_t i = 0; i < ctrl_pts.size(); i++)
    {
        js_ctrl_pts->Set(i * 3    , NanNew(ctrl_pts[i].x));
        js_ctrl_pts->Set(i * 3 + 1, NanNew(ctrl_pts[i].y));
        js_ctrl_pts->Set(i * 3 + 2, NanNew(ctrl_pts[i].z));
    }
    for (size_t i = 0; i < approx_pts.size(); i++)
    {
        js_approx_pts->Set(i * 3    , NanNew(approx_pts[i].x));
        js_approx_pts->Set(i * 3 + 1, NanNew(approx_pts[i].y));
        js_approx_pts->Set(i * 3 + 2, NanNew(approx_pts[i].z));
    }
    for (size_t i = 0; i < mins.size(); i++)
    {
        js_bbs->Set(i * 6    , NanNew(mins[i].x));
        js_bbs->Set(i * 6 + 1, NanNew(mins[i].y));
        js_bbs->Set(i * 6 + 2, NanNew(mins[i].z));
        js_bbs->Set(i * 6 + 3, NanNew(maxs[i].x));
        js_bbs->Set(i * 6 + 4, NanNew(maxs[i].y));
        js_bbs->Set(i * 6 + 5, NanNew(maxs[i].z));
    }

    first_time = false;
    NanReturnValue(0);
}

void Init(Handle<Object> exports)
{
    exports->Set(NanNew("draw"), NanNew<FunctionTemplate>(Main)->GetFunction());
}

NODE_MODULE(draw, Init)
