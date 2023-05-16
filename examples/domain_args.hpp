#ifndef _MFA_DOMAIN_ARGS
#define _MFA_DOMAIN_ARGS

#include <vector>
#include <string>
#include <mfa/types.hpp>

// arguments to block foreach functions
struct DomainArgs
{
    DomainArgs(int dom_dim, vector<int> mdims) 
    {
        // set up per-science variable data: model_dims, s, and f
        updateModelDims(mdims);

        tot_ndom_pts = 0;
        starts.resize(dom_dim);
        ndom_pts.resize(dom_dim);
        full_dom_pts.resize(dom_dim);
        min.resize(dom_dim);
        max.resize(dom_dim);
        r = 0;
        t = 0;
        n = 0;
        multiblock = false;
        structured = true;   // Assume structured input by default
        rand_seed  = -1;
    }
    vector<int>         model_dims;                 // dimension of each model (including geometry)
    size_t              tot_ndom_pts;
    vector<int>         starts;                     // starting offsets of ndom_pts (optional, usually assumed 0)
    vector<int>         ndom_pts;                   // number of points in domain (possibly a subset of full domain)
    vector<int>         full_dom_pts;               // number of points in full domain in case a subset is taken
    vector<real_t>      min;                        // minimum corner of domain
    vector<real_t>      max;                        // maximum corner of domain
    vector<real_t>      s;                          // scaling factor for each variable or any other usage
    real_t              r;                          // x-y rotation of domain or any other usage
    vector<real_t>      f;                          // frequency multiplier for each variable or any other usage
    real_t              t;                          // waviness of domain edges or any other usage
    real_t              n;                          // noise factor [0.0 - 1.0]
    string              infile;                     // input filename
    string              infile2;
    bool                multiblock;                 // multiblock domain, get bounds from block
    bool                structured;                 // input data lies on unstructured grid
    int                 rand_seed;                  // seed for generating random data. -1: no randomization, 0: choose seed at random

    void updateModelDims(vector<int> mdims)
    {
        int nvars = mdims.size() - 1;

        model_dims = mdims;
        s.assign(nvars, 1);
        f.assign(nvars, 1);

        return;
    }
};

#endif // _MFA_DOMAIN_ARGS