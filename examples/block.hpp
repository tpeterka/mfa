//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include <stdio.h>

#include <Eigen/Dense>

#define MAX_DIM 8                           // a user limit, not mfa's

using namespace std;

// set input and ouptut precision here, float or double
#if 0
typedef float                          real_t;
#else
typedef double                         real_t;
#endif

typedef Eigen::MatrixXf                MatrixXf;
typedef Eigen::VectorXf                VectorXf;
typedef MatrixXf::Index                Index;

template <typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using VectorX  = Eigen::Matrix<T, Eigen::Dynamic, 1>;

typedef diy::ContinuousBounds          Bounds;
typedef diy::RegularContinuousLink     RCLink;

// arguments to block foreach functions
struct DomainArgs
{
    int       pt_dim;                            // dimension of points
    int       dom_dim;                           // dimension of domain (<= pt_dim)
    int       geom_p[MAX_DIM];                   // degree in each dimension of geometry
    int       vars_p[MAX_DIM];                   // degree in each dimension of science variables
    int       starts[MAX_DIM];                   // starting offsets of ndom_pts (optional, usually assumed 0)
    int       ndom_pts[MAX_DIM];                 // number of input points in each dimension of domain
    int       full_dom_pts[MAX_DIM];             // number of points in full domain in case a subset is taken
    int       geom_nctrl_pts[MAX_DIM];           // number of input points in each dimension of geometry
    int       vars_nctrl_pts[MAX_DIM];           // number of input points in each dimension of all science variables
    real_t    min[MAX_DIM];                      // minimum corner of domain
    real_t    max[MAX_DIM];                      // maximum corner of domain
    real_t    s[MAX_DIM];                        // scaling factor for each variable or any other usage
    real_t    r;                                 // x-y rotation of domain or any other usage
    real_t    f[MAX_DIM];                        // frequency multiplier for each variable or any other usage
    real_t    t;                                 // waviness of domain edges or any other usage
    char      infile[256];                       // input filename
    bool      weighted;                          // solve for and use weights (default = true)
    bool      multiblock;                        // multiblock domain, get bounds from block
    int       verbose;                           // output level
};

struct ErrArgs
{
    int    max_niter;                           // max num iterations to search for nearest curve pt
    real_t err_bound;                           // desired error bound (stop searching if less)
    int    search_rad;                          // number of parameter steps to search path on either
                                                // side of parameter value of input point
};

// a solved and stored MFA model (geometry or science variable or both)
template <typename T>
struct Model
{
    VectorXi    p;                              // degree in each dimension
    VectorXi    nctrl_pts;                      // number of control points in each dimension
    MatrixX<T>  ctrl_pts;                       // NURBS control points (1st dim changes fastest)
    VectorX<T>  weights;                        // weights associated with control points
    VectorX<T>  knots;                          // NURBS knots (1st dim changes fastest)
    mfa::MFA<T> *mfa;                           // MFA object

};

// block
template <typename T>
struct Block
{
    static
        void* create()          { return new Block; }

    static
        void destroy(void* b)   { delete static_cast<Block*>(b); }

    static
        void add(                               // add the block to the decomposition
            int              gid,               // block global id
            const Bounds&    core,              // block bounds without any ghost added
            const Bounds&    bounds,            // block bounds including any ghost region added
            const Bounds&    domain,            // global data bounds
            const diy::Link& link,              // neighborhood
            diy::Master&     master,            // diy master
            int              dom_dim,           // domain dimensionality
            int              pt_dim)            // point dimensionality
    {
        Block*          b   = new Block;
        diy::Link*      l   = new diy::Link(link);
        diy::Master&    m   = const_cast<diy::Master&>(master);
        m.add(gid, b, l);

        b->domain_mins.resize(pt_dim);
        b->domain_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            b->domain_mins(i) = core.min[i];
            b->domain_maxs(i) = core.max[i];
        }
    }

    static
        void save(
                const void*        b_,
                diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;

            diy::save(bb, b->ndom_pts);
            diy::save(bb, b->domain);
            diy::save(bb, b->domain_mins);
            diy::save(bb, b->domain_maxs);

            // geometry
            diy::save(bb, b->geometry.p);
            diy::save(bb, b->geometry.nctrl_pts);
            diy::save(bb, b->geometry.ctrl_pts);
            diy::save(bb, b->geometry.weights);
            diy::save(bb, b->geometry.knots);

            // science variables
            diy::save(bb, b->vars.size());
            for (auto i = 0; i < b->vars.size(); i++)
            {
                diy::save(bb, b->vars[i].p);
                diy::save(bb, b->vars[i].nctrl_pts);
                diy::save(bb, b->vars[i].ctrl_pts);
                diy::save(bb, b->vars[i].weights);
                diy::save(bb, b->vars[i].knots);
            }

            diy::save(bb, b->approx);
            diy::save(bb, b->errs);
            diy::save(bb, b->span_mins);
            diy::save(bb, b->span_maxs);
        }
    static
        void load(
                void*              b_,
                diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;

            diy::load(bb, b->ndom_pts);
            diy::load(bb, b->domain);
            diy::load(bb, b->domain_mins);
            diy::load(bb, b->domain_maxs);

            // geometry
            diy::load(bb, b->geometry.p);
            diy::load(bb, b->geometry.nctrl_pts);
            diy::load(bb, b->geometry.ctrl_pts);
            diy::load(bb, b->geometry.weights);
            diy::load(bb, b->geometry.knots);

            // science variables
            size_t nvars;
            diy::load(bb, nvars);
            b->vars.resize(nvars);
            for (auto i = 0; i < b->vars.size(); i++)
            {
                diy::load(bb, b->vars[i].p);
                diy::load(bb, b->vars[i].nctrl_pts);
                diy::load(bb, b->vars[i].ctrl_pts);
                diy::load(bb, b->vars[i].weights);
                diy::load(bb, b->vars[i].knots);
            }

            diy::load(bb, b->approx);
            diy::load(bb, b->errs);
            diy::load(bb, b->span_mins);
            diy::load(bb, b->span_maxs);
        }

    // f(x,y,...) = sine(x)/x * sine(y)/y * ...
    void generate_sinc_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a   = &args;
        int nvars       = a->pt_dim - a->dom_dim;             // number of science variables
        vars.resize(nvars);
        int tot_ndom_pts    = 1;
        geometry.p.resize(a->dom_dim);
        for (int j = 0; j < nvars; j++)
            vars[j].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            for (int j = 0; j < nvars; j++)
                vars[j].p(i) =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);
//         s = a->s[0];

        // get local block bounds
        // if single block, they are passed in args
        // if multiblock, they were decomposed by diy and are already in the block's domain_mins, maxs
        if (!a->multiblock)
        {
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
        }

        // assign values to the domain (geometry)
        int cs = 1;                           // stride of a coordinate in this dim
        real_t eps = 1.0e-5;                   // roundoff error
        for (int i = 0; i < a->dom_dim; i++)  // all dimensions in the domain
        {
            real_t d = (domain_maxs(i) - domain_mins(i)) / (ndom_pts(i) - 1);
            int k = 0;
            int co = 0;                       // j index of start of a new coordinate value
            for (int j = 0; j < tot_ndom_pts; j++)
            {
                if (domain_mins(i) + k * d > domain_maxs(i) + eps)
                    k = 0;
                domain(j, i) = domain_mins(i) + k * d;
                if (j + 1 - co >= cs)
                {
                    k++;
                    co = j + 1;
                }
            }
            cs *= ndom_pts(i);
        }

        real_t min, max;                       // extents of range

        // assign values to the range (science variables)
        // f(x,y,z,...) = sine(x)/x * sine(y)/y * sine(z)/z * ...
        for (int j = 0; j < tot_ndom_pts; j++)
        {
            for (int k = 0; k < nvars; k++)        // for all science variables
            {
                real_t res = 1.0;                  // product of the sinc functions
                for (int i = 0; i < a->dom_dim; i++)
                {
                    if (domain(j, i) != 0.0)
                        res *= (sin(domain(j, i) * a->f[k] ) / domain(j, i));
                }
                res *= a->s[k];
                domain(j, a->dom_dim + k) = res;

                if (j == 0 || res > max)
                    max = res;
                if (j == 0 || res < min)
                    min = res;
            }
        }

        // optional wavy domain
        if (a->t && a->pt_dim >= 3)
        {
            for (auto j = 0; j < tot_ndom_pts; j++)
            {
                real_t x = domain(j, 0);
                real_t y = domain(j, 1);
                domain(j, 0) += a->t * sin(y);
                domain(j, 1) += a->t * sin(x);
                if (j == 0 || domain(j, 0) < domain_mins(0))
                    domain_mins(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) < domain_mins(1))
                    domain_mins(1) = domain(j, 1);
                if (j == 0 || domain(j, 0) > domain_maxs(0))
                    domain_maxs(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) > domain_maxs(1))
                    domain_maxs(1) = domain(j, 1);
            }
        }

        // optional rotation of the domain
        if (a->r && a->pt_dim >= 3)
        {
            for (auto j = 0; j < tot_ndom_pts; j++)
            {
                real_t x = domain(j, 0);
                real_t y = domain(j, 1);
                domain(j, 0) = x * cos(a->r) - y * sin(a->r);
                domain(j, 1) = x * sin(a->r) + y * cos(a->r);
                if (j == 0 || domain(j, 0) < domain_mins(0))
                    domain_mins(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) < domain_mins(1))
                    domain_mins(1) = domain(j, 1);
                if (j == 0 || domain(j, 0) > domain_maxs(0))
                    domain_maxs(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) > domain_maxs(1))
                    domain_maxs(1) = domain(j, 1);
            }
        }

        // extents
        domain_mins(a->pt_dim - 1) = min;
        domain_maxs(a->pt_dim - 1) = max;
//         fprintf(stderr, "gid = %d\n", cp.gid());
        cerr << "domain_mins:\n" << domain_mins << endl;
        cerr << "domain_maxs:\n" << domain_maxs << "\n" << endl;

        cerr << "domain:\n" << domain << endl;
    }


    // y = sine(x)
    void generate_sine_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a   = &args;
        int nvars       = a->pt_dim - a->dom_dim;             // number of science variables
        vars.resize(nvars);
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        for (int j = 0; j < nvars; j++)
            vars[j].p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            for (int j = 0; j < nvars; j++)
                vars[j].p(i) =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);
//         s = a->s[0];

        // get local block bounds
        // if single block, they are passed in args
        // if multiblock, they were decomposed by diy and are already in the block's domain_mins, maxs
        if (!a->multiblock)
        {
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
        }

        // assign values to the domain (geometry)
        int cs = 1;                           // stride of a coordinate in this dim
        real_t eps = 1.0e-5;                  // floating point roundoff error
        for (int i = 0; i < a->dom_dim; i++)  // all dimensions in the domain
        {
            real_t d = (domain_maxs(i) - domain_mins(i)) / (ndom_pts(i) - 1);
            int k = 0;
            int co = 0;                       // j index of start of a new coordinate value
            for (int j = 0; j < tot_ndom_pts; j++)
            {
                if (domain_mins(i) + k * d > domain_maxs(i) + eps)
                    k = 0;
                domain(j, i) = domain_mins(i) + k * d;
                if (j + 1 - co >= cs)
                {
                    k++;
                    co = j + 1;
                }
            }
            cs *= ndom_pts(i);
        }

        real_t min, max;                       // extents of range

        // assign values to the range (science variables)
        // f(x,y,z,...) = sine(x) * sine(y) * sine(z) * ...
        for (int j = 0; j < tot_ndom_pts; j++)
        {
            for (int k = 0; k < nvars; k++)        // for all science variables
            {
                real_t res = 1.0;                  // product of the sine functions
                for (int i = 0; i < a->dom_dim; i++)
                    res *= sin(domain(j, i));
                res *= a->s[k];
                domain(j, a->pt_dim - 1) = res;

                if (j == 0 || res > max)
                    max = res;
                if (j == 0 || res < min)
                    min = res;
            }
        }

        // extents
        domain_mins(a->pt_dim - 1) = min;
        domain_maxs(a->pt_dim - 1) = max;
        fprintf(stderr, "gid = %d\n", cp.gid());
        cerr << "domain_mins:\n" << domain_mins << endl;
        cerr << "domain_maxs:\n" << domain_maxs << "\n" << endl;

        //             cerr << "domain:\n" << domain << endl;
    }


    // read a floating point 3d vector dataset and take one 1-d curve out of the middle of it
    // f = (x, velocity magnitude)
    void read_1d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            vars[0].p(i)    =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);
        vector<float> vel(3 * tot_ndom_pts);

        // rest is hard-coded for 1d

        // open file and seek to a slice in the center
        FILE *fd = fopen(a->infile, "r");
        assert(fd);
        fseek(fd, (a->ndom_pts[0] * a->ndom_pts[1] * a->ndom_pts[2] / 2 + a->ndom_pts[0] * a->ndom_pts[1] / 2) * 12, SEEK_SET);

        // read all three components of velocity and compute magnitude
        if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            domain(i, 1) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
            //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 1) < domain_mins(1))
                domain_mins(1) = domain(i, 1);
            if (i == 0 || domain(i, 1) > domain_maxs(1))
                domain_maxs(1) = domain(i, 1);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
        {
            domain(n, 0) = i;
            n++;
        }

        // extents
        domain_mins(0) = 0.0;
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2-d surface out of the middle of it
    // f = (x, y, velocity magnitude)
    void read_2d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            vars[0].p(i)    =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);
        vector<float> vel(3 * tot_ndom_pts);

        // rest is hard-coded for 2d

        // open file and seek to a slice in the center
        FILE *fd = fopen(a->infile, "r");
        assert(fd);
        // middle plane in z, offset = full x,y range * 1/2 z range
        fseek(fd, (a->ndom_pts[0] * a->ndom_pts[1] * a->ndom_pts[2] / 2) * 12, SEEK_SET);

        // read all three components of velocity and compute magnitude
        if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            domain(i, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
            //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 2) < domain_mins(2))
                domain_mins(2) = domain(i, 2);
            if (i == 0 || domain(i, 2) > domain_maxs(2))
                domain_maxs(2) = domain(i, 2);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
            for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
            {
                domain(n, 0) = i;
                domain(n, 1) = j;
                n++;
            }

        // extents
        domain_mins(0) = 0.0;
        domain_mins(1) = 0.0;
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2d (parallel to x-y plane) subset
    // f = (x, y, velocity magnitude)
    void read_2d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            vars[0].p(i)    =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);
        vector<float> vel(a->full_dom_pts[0] * a->full_dom_pts[1] * 3);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // rest is hard-coded for 2d

        // seek to start of desired full x-y plane
        size_t ofst = 0;                                                    // offset to seek to (in bytes)
        ofst += a->starts[2] * a->full_dom_pts[0] * a->full_dom_pts[1];     // z direction
        ofst *= 12;                                                         // 3 components * 4 bytes
        fseek(fd, ofst, SEEK_SET);

        // read all three components of velocity for the entire plane (not just the subset)
        if (!fread(&vel[0], sizeof(float), a->full_dom_pts[0] * a->full_dom_pts[1] * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }

        // compute velocity magnitude only for the points in the subset, dropping the rest
        size_t ijk[2] = {0, 0};                          // i,j,k indices of current point
        size_t n = 0;
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            // is the point in the subset?
            bool keep = true;
            if (ijk[0] < a->starts[0] || ijk[0] >= a->starts[0] + a->ndom_pts[0] ||
                    ijk[1] < a->starts[1] || ijk[1] >= a->starts[1] + a->ndom_pts[1])
                keep = false;

            // debug
            //                 fprintf(stderr, "i=%ld ijk=[%ld %ld] keep=%d\n", i, ijk[0], ijk[1], keep);

            if (keep)
            {
                domain(n, 0) = ijk[0];                  // domain is just i,j
                domain(n, 1) = ijk[1];
                // range (function value) is magnitude of velocity
                domain(n, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                        vel[3 * i + 1] * vel[3 * i + 1] +
                        vel[3 * i + 2] * vel[3 * i + 2]);
                n++;
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }

            // increment ijk
            if (ijk[0] == a->full_dom_pts[0] - 1)
            {
                ijk[0] = 0;
                ijk[1]++;
            }
            else
                ijk[0]++;
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 2) < domain_mins(2))
                domain_mins(2) = domain(i, 2);
            if (i == 0 || domain(i, 2) > domain_maxs(2))
                domain_maxs(2) = domain(i, 2);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        domain_mins(0) = domain(0, 0);
        domain_mins(1) = domain(0, 1);
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 3d vector dataset
    // f = (x, y, z, velocity magnitude)
    void read_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            vars[0].p(i)    =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);

        vector<float> vel(3 * tot_ndom_pts);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // read all three components of velocity and compute magnitude
        if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            domain(i, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
            //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
        }

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 3) < domain_mins(3))
                domain_mins(3) = domain(i, 3);
            if (i == 0 || domain(i, 3) > domain_maxs(3))
                domain_maxs(3) = domain(i, 3);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                {
                    domain(n, 0) = i;
                    domain(n, 1) = j;
                    domain(n, 2) = k;
                    n++;
                }

        // extents
        domain_mins(0) = 0.0;
        domain_mins(1) = 0.0;
        domain_mins(2) = 0.0;
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);
        domain_maxs(2) = domain(tot_ndom_pts - 1, 2);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 3d vector dataset and take a 3d subset out of it
    // f = (x, y, z, velocity magnitude)
    void read_3d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            vars[0].p(i)    =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);
        vector<float> vel(a->full_dom_pts[0] * a->full_dom_pts[1] * a->full_dom_pts[2] * 3);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // rest is hard-coded for 3d

        // read all three components of velocity (not just the subset)
        if (!fread(&vel[0], sizeof(float), a->full_dom_pts[0] * a->full_dom_pts[1] * a->full_dom_pts[2] * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }

        // compute velocity magnitude only for the points in the subset, dropping the rest
        size_t ijk[3] = {0, 0, 0};                          // i,j,k indices of current point
        size_t n = 0;
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            // is the point in the subset?
            bool keep = true;
            if (ijk[0] < a->starts[0] || ijk[0] >= a->starts[0] + a->ndom_pts[0] ||
                    ijk[1] < a->starts[1] || ijk[1] >= a->starts[1] + a->ndom_pts[1] ||
                    ijk[2] < a->starts[2] || ijk[2] >= a->starts[2] + a->ndom_pts[2])
                keep = false;

            // debug
            //                 fprintf(stderr, "i=%ld ijk=[%ld %ld %ld] keep=%d\n", i, ijk[0], ijk[1], ijk[2], keep);

            if (keep)
            {
                domain(n, 0) = ijk[0];                  // domain is just i,j
                domain(n, 1) = ijk[1];
                domain(n, 2) = ijk[2];
                domain(n, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                        vel[3 * i + 1] * vel[3 * i + 1] +
                        vel[3 * i + 2] * vel[3 * i + 2]);
                n++;
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }

            // increment ijk
            if (ijk[0] == a->full_dom_pts[0] - 1)
            {
                ijk[0] = 0;
                if (ijk[1] == a->full_dom_pts[1] - 1)
                {
                    ijk[1] = 0;
                    ijk[2]++;
                }
                else
                    ijk[1]++;
            }
            else
                ijk[0]++;
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 3) < domain_mins(3))
                domain_mins(3) = domain(i, 3);
            if (i == 0 || domain(i, 3) > domain_maxs(3))
                domain_maxs(3) = domain(i, 3);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        domain_mins(0) = domain(0, 0);
        domain_mins(1) = domain(0, 1);
        domain_mins(2) = domain(0, 2);
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);
        domain_maxs(2) = domain(tot_ndom_pts - 1, 2);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 2d scalar dataset
    // f = (x, y, value)
    void read_2d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            vars[0].p(i)    =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);

        vector<float> val(tot_ndom_pts);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // read data values
        if (!fread(&val[0], sizeof(float), tot_ndom_pts, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < val.size(); i++)
            domain(i, 2) = val[i];

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 2) < domain_mins(2))
                domain_mins(2) = domain(i, 2);
            if (i == 0 || domain(i, 2) > domain_maxs(2))
                domain_maxs(2) = domain(i, 2);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
            for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
            {
                domain(n, 0) = i;
                domain(n, 1) = j;
                n++;
            }

        // extents
        domain_mins(0) = 0.0;
        domain_mins(1) = 0.0;
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 3d scalar dataset
    // f = (x, y, z, value)
    void read_3d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.p.resize(a->dom_dim);
        vars[0].p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            geometry.p(i)   =  a->geom_p[i];
            vars[0].p(i)    =  a->vars_p[i];
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, a->pt_dim);

        vector<float> val(tot_ndom_pts);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // read data values
        if (!fread(&val[0], sizeof(float), tot_ndom_pts, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < val.size(); i++)
            domain(i, 3) = val[i];

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 3) < domain_mins(3))
                domain_mins(3) = domain(i, 3);
            if (i == 0 || domain(i, 3) > domain_maxs(3))
                domain_maxs(3) = domain(i, 3);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                {
                    domain(n, 0) = i;
                    domain(n, 1) = j;
                    domain(n, 2) = k;
                    n++;
                }

        // extents
        domain_mins(0) = 0.0;
        domain_mins(1) = 0.0;
        domain_mins(2) = 0.0;
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);
        domain_maxs(2) = domain(tot_ndom_pts - 1, 2);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // fixed number of control points encode block
    void fixed_encode_block(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;

        // initialize control points
        // TODO: for now geometry and science variables same number of control points; vary later
        geometry.nctrl_pts.resize(a->dom_dim);
        for (auto j = 0; j < a->dom_dim; j++)
            geometry.nctrl_pts(j) =  a->geom_nctrl_pts[j];
        for (auto i = 0; i < vars.size(); i++)
        {
            vars[i].nctrl_pts.resize(a->dom_dim);
            for (auto j = 0; j < a->dom_dim; j++)
                vars[i].nctrl_pts(j) =  a->vars_nctrl_pts[j];
        }

        // encode geometry
        int ndom_dims = ndom_pts.size();                // domain dimensionality
        geometry.mfa = new mfa::MFA<T>(geometry.p,
                                       ndom_pts,
                                       domain,
                                       geometry.ctrl_pts,
                                       geometry.nctrl_pts,
                                       geometry.weights,
                                       geometry.knots,
                                       0,
                                       ndom_dims - 1);
        geometry.mfa->FixedEncode(geometry.nctrl_pts, a->verbose, a->weighted);

        // encode science variables
        for (auto i = 0; i< vars.size(); i++)
        {
            vars[i].mfa = new mfa::MFA<T>(vars[i].p,
                                          ndom_pts,
                                          domain,
                                          vars[i].ctrl_pts,
                                          vars[i].nctrl_pts,
                                          vars[i].weights,
                                          vars[i].knots,
                                          ndom_dims,
                                          ndom_dims + i);
            vars[i].mfa->FixedEncode(vars[i].nctrl_pts, a->verbose, a->weighted);
        }
    }

    // TODO: convert the following to split models

//     // adaptively encode block to desired error limit
//     void adaptive_encode_block(
//             const diy::Master::ProxyWithLink& cp,
//             real_t                            err_limit,
//             int                               max_rounds,
//             DomainArgs&                       args)
//     {
//         DomainArgs* a = &args;
//         nctrl_pts.resize(0);            // 0 size means MFA will initialize to minimum p+1
//         mfa = new mfa::MFA<T>(p, ndom_pts, domain, ctrl_pts, nctrl_pts, weights, knots);
//         mfa->AdaptiveEncode(err_limit, nctrl_pts, a->verbose, a->weighted, max_rounds);
//     }
// 
//     // nonlinear encoding of block to desired error limit
//     // only for 1D so far
//     void nonlinear_encode_block(
//             const   diy::Master::ProxyWithLink& cp,
//             real_t   err_limit)
//     {
//         // set initial control points here
//         // TODO: what if there aren'e enough control points (p + 1 is the minimum needed)?
//         real_t grad;                             // current gradient (finite difference)
//         real_t prev_grad = 0.0;                  // previous gradient (finite difference)
//         nctrl_pts.resize(1);
//         for (auto i = 0; i < domain.rows(); i++)
//         {
//             if (i == 0 || i == domain.rows() - 1)
//             {
//                 // first and last control points coincide with domain
//                 ctrl_pts.conservativeResize(ctrl_pts.rows() + 1, domain.cols());
//                 ctrl_pts.row(ctrl_pts.rows() - 1) = domain.row(i);
//             }
//             else
//             {
//                 grad = (domain(i, 1) - domain(i - 1, 1)) / (domain(i, 0) - domain(i - 1, 0));
//                 // set control point at local min/max (gradient sign change)
//                 // TODO: checking exactly for 0.0 is not robust
//                 if ((grad == 0.0) || (grad > 0.0 && prev_grad < 0.0) || (grad < 0.0 && prev_grad > 0.0))
//                 {
//                     ctrl_pts.conservativeResize(ctrl_pts.rows() + 1, domain.cols());
//                     ctrl_pts.row(ctrl_pts.rows() - 1) = domain.row(i);
//                 }
//                 prev_grad = grad;
//             }
//         }
//         nctrl_pts(0) = ctrl_pts.rows();
// 
//         // set initial weights to 1.0
//         weights = VectorX<T>::Ones(ctrl_pts.rows());
// 
//         // debug
//         cerr << ctrl_pts.rows() << " initial control points:\n" << ctrl_pts << "\n" << endl;
// 
//         mfa = new mfa::MFA<T>(p, ndom_pts, domain, ctrl_pts, nctrl_pts, weights, knots);
//         mfa->NonlinearEncode(err_limit, nctrl_pts);
//     }

    // decode entire block
    void decode_block(const diy::Master::ProxyWithLink& cp)
    {
        approx.resize(domain.rows(), domain.cols());

            int ndom_dims = ndom_pts.size();                // domain dimensionality
            // geometry
            geometry.mfa->Decode(approx, 0, ndom_dims - 1);

            // science variables
            for (auto i = 0; i < vars.size(); i++)
                vars[i].mfa->Decode(approx, ndom_dims, ndom_dims + i);
    }

    // TODO: convert the following to split models

//     // differentiate entire block
//     void differentiate_block(
//             const diy::Master::ProxyWithLink& cp,
//             int                               verbose,  // output level
//             int                               deriv,    // which derivative to take (1 = 1st, 2 = 2nd, ...) in each domain dim.
//             int                               partial)  // limit to partial derivative in just this dimension (-1 = no limit)
//     {
//         approx.resize(domain.rows(), domain.cols());
//         mfa = new mfa::MFA<T>(p, ndom_pts, domain, ctrl_pts, nctrl_pts, weights, knots);
//         VectorXi derivs(p.size());
//         for (auto i = 0; i < derivs.size(); i++)
//             derivs(i) = deriv;
// 
//         // optional limit to one partial derivative
//         if (deriv && p.size() > 1 && partial >= 0)
//         {
//             for (auto i = 0; i < p.size(); i++)
//             {
//                 if (i != partial)
//                     derivs(i) = 0;
//             }
//         }
// 
//         mfa->Decode(approx, verbose, derivs);
// 
//         // the derivative is a vector of same dimensionality as domain
//         // derivative needs to be scaled by domain extent because u,v,... are in [0.0, 1.0]
//         if (deriv)
//         {
//             if (p.size() == 1 || partial >= 0) // TODO: not for mixed partials
//             {
//                 if (p.size() == 1)
//                     partial = 0;
//                 for (auto j = 0; j < approx.cols(); j++)
//                     // scale once for each derivative
//                     for (auto i = 0; i < deriv; i++)
//                         approx.col(j) /= (domain_maxs(partial) - domain_mins(partial));
//             }
//         }
// 
//         // for plotting, set all but the last dimension to be the same as the input domain
//         if (deriv)
//             for (auto i = 0; i < domain.cols() - 1; i++)
//                 approx.col(i) = domain.col(i);
//     }

//     // compute error field and maximum error in the block
//     // uses normal distance to the curve, surface, etc.
//     void error(
//             const   diy::Master::ProxyWithLink& cp,
//             int     verbose,                                 // output level
//             bool    decode_block)                            // decode entire block first
//     {
//         errs.resize(domain.rows(), domain.cols());
//         errs = domain;
// 
//         if (decode_block)
//         {
//             approx.resize(domain.rows(), domain.cols());
// 
//             int ndom_dims = ndom_pts.size();                // domain dimensionality
//             // geometry
//             geometry.mfa->Decode(approx, 0, ndom_dims - 1);
// 
//             // science variables
//             for (auto i = 0; i < vars.size(); i++)
//                 vars[i].mfa->Decode(approx, ndom_dims, ndom_dims + i);
//         }
// 
// #ifndef MFA_NO_TBB                                          // TBB version
// 
//         // distance computation
//         size_t max_idx;
//         if (decode_block)
//         {
//             parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
//                     {
//                     VectorX<T> cpt = approx.row(i);
//                     errs(i, errs.cols() - 1) = fabs(mfa->NormalDistance(cpt, i));
//                     });
//         }
//         else
//         {
//             parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
//                     {
//                     errs(i, errs.cols() - 1) = mfa->Error(i, verbose);
//                     });
//         }
//         sum_sq_err = 0.0;
//         for (size_t i = 0; i < domain.rows(); i++)
//         {
//             sum_sq_err += (errs(i, errs.cols() - 1) * errs(i, errs.cols() - 1));
//             if (i == 0 || errs(i, errs.cols() - 1) > max_err)
//             {
//                 max_err = errs(i, errs.cols() - 1);
//                 max_idx = i;
//             }
//         }
// 
// #else                                               // single thread version
// 
//         // distance computation
//         // TODO: only last variable
//         size_t max_idx;
//         for (size_t i = 0; i < (size_t)domain.rows(); i++)
//         {
//             if (decode_block)
//             {
//                 VectorX<T> cpt = approx.row(i);
//                 errs(i, errs.cols() - 1) = fabs(mfa->NormalDistance(cpt, i));
//             }
//             else
//                 errs(i, errs.cols() - 1) = mfa->Error(i, verbose);
//             if (i == 0 || fabs(errs(i, errs.cols() - 1)) > fabs(max_err))
//             {
//                 max_err = errs(i, errs.cols() - 1);
//                 max_idx = i;
//             }
//         }
// 
// #endif
// 
// //         mfa->max_err = max_err;
//     }

    // compute error field and maximum error in the block
    // uses difference between range values
    void range_error(
            const   diy::Master::ProxyWithLink& cp,
            int     verbose,                                 // output level
            bool    decode_block)                            // decode entire block first
    {
        errs.resize(domain.rows(), domain.cols());
        errs = domain;

        if (decode_block)
        {
            approx.resize(domain.rows(), domain.cols());

            int ndom_dims = ndom_pts.size();                // domain dimensionality
            // geometry
            geometry.mfa->Decode(verbose, approx, 0, ndom_dims - 1);

            // science variables
            for (auto i = 0; i < vars.size(); i++)
                vars[i].mfa->Decode(verbose, approx, ndom_dims, ndom_dims + i);
        }

#ifndef MFA_NO_TBB                                          // TBB version

        // distance computation
        // TODO: only distance between last science variables
        size_t max_idx;
        int last = errs.cols() - 1;                 // range coordinate
        if (decode_block)
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    VectorX<T> cpt = approx.row(i);
                    errs(i, last) = fabs(cpt(last) - domain(i, last));
                    });
        }
        else
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    errs(i, last) = vars[vars.size() - 1].mfa->RangeError(i, verbose);
                    });
        }
        sum_sq_err = 0.0;
        for (size_t i = 0; i < domain.rows(); i++)
        {
            sum_sq_err += (errs(i, last) * errs(i, last));
            if (i == 0 || errs(i, last) > max_err)
            {
                max_err = errs(i, last);
                max_idx = i;
            }
        }

#else                                               // single thread version

        // distance computation
        // TODO: only distance between last science variables
        size_t max_idx;
        int last   = errs.cols() - 1;               // range coordinate
        sum_sq_err = 0.0;
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (decode_block)
            {
                VectorX<T> cpt = approx.row(i);
                errs(i, last) = fabs(cpt(last) - domain(i, last));
            }
            else
                errs(i, last) = vars[vars.size() - 1].mfa->RangeError(i, verbose);
            sum_sq_err += (errs(i, last) * errs(i, last));
            if (i == 0 || errs(i, last) > max_err)
            {
                max_err = errs(i, last);
                max_idx = i;
            }
        }

#endif

//         mfa->max_err = max_err;
    }

    // save knot span domains for later comparison with error field
    void knot_span_domains(const diy::Master::ProxyWithLink& cp)
    {
        // geometry
        geometry.mfa->KnotSpanDomains(span_mins, span_maxs);

        // science variables
        for (auto i = 0; i < vars.size(); i++)
            vars[i].mfa->KnotSpanDomains(span_mins, span_maxs);
    }

    void print_block(const diy::Master::ProxyWithLink& cp)
    {
        // max extent of input data points
        int last            = domain.cols() - 1;
        real_t range_extent = domain.col(last).maxCoeff() - domain.col(last).minCoeff();

//         fprintf(stderr, "gid = %d\n", cp.gid());
//         cerr << "domain\n" << domain << endl;

        // geometry
        cerr << "----- geometry model -----" << endl;
        cerr << "nctrl_pts:\n" << geometry.nctrl_pts << endl;
//         cerr << geometry.ctrl_pts.rows() << " final control points\n" << geometry.ctrl_pts << endl;
//         cerr << geometry.weights.size()  << " final weights\n" << geometry.weights << endl;
//         cerr << geometry.knots.size() << " knots\n" << geometry.knots << endl;
        fprintf(stderr, "# output ctrl pts     = %ld\n", geometry.ctrl_pts.rows());
        fprintf(stderr, "# output knots        = %ld\n", geometry.knots.size());
        cerr << "--------------------------" << endl;


        // science variables
        cerr << "----- science variable models -----" << endl;
        for (auto i = 0; i < vars.size(); i++)
        {
            cerr << "----- var " << i << " -----" << endl;
            cerr << "nctrl_pts:\n" << vars[i].nctrl_pts << endl;
//             cerr << vars[i].ctrl_pts.rows() << " final control points\n" << vars[i].ctrl_pts << endl;
//             cerr << vars[i].weights.size()  << " final weights\n" << vars[i].weights << endl;
//             cerr << vars[i].knots.size() << " knots\n" << vars[i].knots << endl;
            cerr << "--------------------------" << endl;
            fprintf(stderr, "# output ctrl pts     = %ld\n", vars[i].ctrl_pts.rows());
            fprintf(stderr, "# output knots        = %ld\n", vars[i].knots.size());
        }
        cerr << "-----------------------------------" << endl;

        cerr << approx.rows() << " approximated points\n" << approx << endl;
        fprintf(stderr, "range extent          = %e\n",  range_extent);
        fprintf(stderr, "max_err               = %e\n",  max_err);
        fprintf(stderr, "normalized max_err    = %e\n",  max_err / range_extent);
        fprintf(stderr, "sum of squared errors = %e\n",  sum_sq_err);
        //         DEPRECATE
//         fprintf(stderr, "L2 error              = %e\n",  sqrt(sum_sq_err / nctrl_pts.rows()));
        fprintf(stderr, "RMS error             = %e\n",  sqrt(sum_sq_err / domain.rows()));
        fprintf(stderr, "normalized RMS error  = %e\n",  sqrt(sum_sq_err / domain.rows()) / range_extent);
        fprintf(stderr, "# input points        = %ld\n", domain.rows());

        // compute compression ratio
        float in_coords = domain.rows() * domain.cols();
        float out_coords = geometry.ctrl_pts.rows() * geometry.ctrl_pts.cols();
        out_coords += geometry.knots.size();
        for (auto i = 0; i < vars.size(); i++)
        {
            out_coords += (vars[i].ctrl_pts.rows() * vars[i].ctrl_pts.cols());
            out_coords += vars[i].knots.size();
        }
        fprintf(stderr, "compression ratio     = %.2f\n", in_coords / out_coords);

//         fprintf(stderr, "compression ratio     = %.2f\n",
//                 (real_t)(domain.rows()) / (ctrl_pts.rows() + knots.size() / ctrl_pts.cols()));
        fprintf(stderr, "\n");
    }

    void print_deriv(const diy::Master::ProxyWithLink& cp)
    {
        fprintf(stderr, "gid = %d\n", cp.gid());
        cerr << "domain\n" << domain << endl;
        cerr << approx.rows() << " derivatives\n" << approx << endl;
        fprintf(stderr, "\n");
    }

    // write original and approximated data in raw format
    // only for one block (one file name used, ie, last block will overwrite earlier ones)
    void write_raw(const diy::Master::ProxyWithLink& cp)
    {
        int last = domain.cols() - 1;           // last column in domain points

        // write original points
        ofstream domain_outfile;
        domain_outfile.open("orig.raw", ios::binary);
        vector<T> out_domain(domain.rows());
        for (auto i = 0; i < domain.rows(); i++)
            out_domain[i] = domain(i, last);
        domain_outfile.write((char*)(&out_domain[0]), domain.rows() * sizeof(T));
        domain_outfile.close();

#if 0
        // debug: read back original points
        ifstream domain_infile;
        vector<T> in_domain(domain.rows());
        domain_infile.open("orig.raw", ios::binary);
        domain_infile.read((char*)(&in_domain[0]), domain.rows() * sizeof(T));
        domain_infile.close();
        for (auto i = 0; i < domain.rows(); i++)
            if (in_domain[i] != domain(i, last))
                fprintf(stderr, "Error writing raw data: original data does match writen/read back data\n");
#endif

        // write approximated points
        ofstream approx_outfile;
        approx_outfile.open("approx.raw", ios::binary);
        vector<T> out_approx(approx.rows());
        for (auto i = 0; i < approx.rows(); i++)
            out_approx[i] = approx(i, last);
        approx_outfile.write((char*)(&out_approx[0]), approx.rows() * sizeof(T));
        approx_outfile.close();

#if 0
        // debug: read back original points
        ifstream approx_infile;
        vector<T> in_approx(approx.rows());
        approx_infile.open("approx.raw", ios::binary);
        approx_infile.read((char*)(&in_approx[0]), approx.rows() * sizeof(T));
        approx_infile.close();
        for (auto i = 0; i < approx.rows(); i++)
            if (in_approx[i] != approx(i, last))
                fprintf(stderr, "Error writing raw data: approximated data does match writen/read back data\n");
#endif
    }

    // input data
    VectorXi            ndom_pts;              // number of domain points in each dimension
    MatrixX<T>          domain;                // input data (1st dim changes fastest)
    VectorX<T>          domain_mins;           // local domain minimum corner
    VectorX<T>          domain_maxs;           // local domain maximum corner

    // MFA models
    Model<T>            geometry;               // geometry MFA
    vector< Model<T> >  vars;                   // science variable MFAs

    // output data
    MatrixX<T>          approx;                 // points in approximated volume

    // errors
    real_t              max_err;                // maximum (abs value) distance from input points to curve
    real_t              sum_sq_err;             // sum of squared errors
    MatrixX<T>          errs;                   // error field (abs. value, not normalized by data range)
    real_t              s;                      // scaling factor on range values (for error checking)

    // knot spans (for debugging)
    VectorXi            span_mins;              // idx of minimum domain points of all knot spans
    VectorXi            span_maxs;              // idx of maximum domain points of all knot spans
};

namespace diy
{
        template <typename T>
        struct Serialization<MatrixX<T>>
        {
            static
                void save(diy::BinaryBuffer& bb, const MatrixX<T>& m)
                {
                    diy::save(bb, m.rows());
                    diy::save(bb, m.cols());
                    diy::save(bb, m.data(), m.rows() * m.cols());
                }
            static
                void load(diy::BinaryBuffer& bb, MatrixX<T>& m)
                {
                    Index rows, cols;
                    diy::load(bb, rows);
                    diy::load(bb, cols);
                    m.resize(rows, cols);
                    diy::load(bb, m.data(), rows * cols);
                }
        };
        template <typename T>
        struct Serialization<VectorX<T>>
        {
            static
                void save(diy::BinaryBuffer& bb, const VectorX<T>& v)
                {
                    diy::save(bb, v.size());
                    diy::save(bb, v.data(), v.size());
                }
            static
                void load(diy::BinaryBuffer& bb, VectorX<T>& v)
                {
                    Index size;
                    diy::load(bb, size);
                    v.resize(size);
                    diy::load(bb, v.data(), size);
                }
        };
        template<>
        struct Serialization<VectorXi>
        {
            static
                void save(diy::BinaryBuffer& bb, const VectorXi& v)
                {
                    diy::save(bb, v.size());
                    diy::save(bb, v.data(), v.size());
                }
            static
                void load(diy::BinaryBuffer& bb, VectorXi& v)
                {
                    Index size;
                    diy::load(bb, size);
                    v.resize(size);
                    diy::load(bb, v.data(), size);
                }
        };
}
