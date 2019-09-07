//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include    <mfa/mfa.hpp>

#include    <diy/master.hpp>
#include    <diy/reduce-operations.hpp>
#include    <diy/decomposition.hpp>
#include    <diy/assigner.hpp>
#include    <diy/io/block.hpp>
#include    <diy/pick.hpp>
#include    <diy/fmt/format.h>

#include    <stdio.h>

#include    <Eigen/Dense>

#include    <random>

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

typedef diy::Bounds<real_t>            Bounds;
typedef diy::RegularLink<Bounds>       RCLink;
typedef diy::RegularDecomposer<Bounds> Decomposer;

// 3d point or vector
struct vec3d
{
    float x, y, z;
    float mag() { return sqrt(x*x + y*y + z*z); }
};

// arguments to block foreach functions
struct DomainArgs
{
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
    real_t    n;                                 // noise factor [0.0 - 1.0]
    char      infile[256];                       // input filename
    bool      weighted;                          // solve for and use weights (default = true)
    bool      multiblock;                        // multiblock domain, get bounds from block
    int       verbose;                           // output level
};

struct ErrArgs
{ int    max_niter;                           // max num iterations to search for nearest curve pt
    real_t err_bound;                           // desired error bound (stop searching if less)
    int    search_rad;                          // number of parameter steps to search path on either
                                                // side of parameter value of input point
};

// a solved and stored MFA model (geometry or science variable or both)
template <typename T>
struct Model
{
    int                 min_dim;                // starting coordinate of this model in full-dimensional data
    int                 max_dim;                // ending coordinate of this model in full-dimensional data
    mfa::MFA_Data<T>    *mfa_data;             // MFA model data
};

// block
template <typename T>
struct Block
{
    // dimensionality
    int                 dom_dim;                // dimensionality of domain (geometry)
    int                 pt_dim;                 // dimensionality of full point (geometry + science vars)

    // input data
    MatrixX<T>          domain;                 // input data (1st dim changes fastest)
    VectorX<T>          bounds_mins;            // local domain minimum corner
    VectorX<T>          bounds_maxs;            // local domain maximum corner
    VectorX<T>          core_mins;              // local domain minimum corner w/o ghost
    VectorX<T>          core_maxs;              // local domain maximum corner w/o ghost

    // MFA object
    mfa::MFA<T>         *mfa;

    // MFA models
    Model<T>            geometry;               // geometry MFA
    vector<Model<T>>    vars;                   // science variable MFAs

    // output data
    MatrixX<T>          approx;                 // points in approximated volume

    // errors for each science variable
    vector<T>           max_errs;               // maximum (abs value) distance from input points to curve
    vector<T>           sum_sq_errs;            // sum of squared errors

    // error field for last science variable only
    MatrixX<T>          errs;                   // error field (abs. value, not normalized by data range)

    static
        void* create()          { return new Block; }

    static
        void destroy(void* b)
        {
            Block* block = static_cast<Block*>(b);
            if (block->mfa)
                delete block->mfa;
            delete block;
        }

    static
        void add(                               // add the block to the decomposition
            int              gid,               // block global id
            const Bounds&    core,              // block bounds without any ghost added
            const Bounds&    bounds,            // block bounds including any ghost region added
            const Bounds&    domain,            // global data bounds
            const RCLink&    link,              // neighborhood
            diy::Master&     master,            // diy master
            int              dom_dim,           // domain dimensionality
            int              pt_dim,            // point dimensionality
            T                ghost_factor = 0.0)// amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    {
        Block*          b   = new Block;
        RCLink*         l   = new RCLink(link);
        diy::Master&    m   = const_cast<diy::Master&>(master);
        m.add(gid, b, l);

        b->dom_dim = dom_dim;
        b->pt_dim  = pt_dim;

        // NB: using bounds to hold full point dimensionality, but using core to hold only domain dimensionality
        b->bounds_mins.resize(pt_dim);
        b->bounds_maxs.resize(pt_dim);
        b->core_mins.resize(dom_dim);
        b->core_maxs.resize(dom_dim);

        // manually set ghosted block bounds as a factor increase of original core bounds
        for (int i = 0; i < dom_dim; i++)
        {
            T ghost_amount = ghost_factor * (core.max[i] - core.min[i]);
            if (core.min[i] > domain.min[i])
                b->bounds_mins(i) = core.min[i] - ghost_amount;
            else
                b->bounds_mins(i)= core.min[i];

            if (core.max[i] < domain.max[i])
                b->bounds_maxs(i) = core.max[i] + ghost_amount;
            else
                b->bounds_maxs(i) = core.max[i];
            b->core_mins(i) = core.min[i];
            b->core_maxs(i) = core.max[i];
        }

        b->mfa = NULL;
    }

    static
        void save(
                const void*        b_,
                diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;

            // TODO: don't save domain in practice
            diy::save(bb, b->domain);

            // top-level mfa data
            diy::save(bb, b->dom_dim);
            diy::save(bb, b->mfa->ndom_pts());

            diy::save(bb, b->bounds_mins);
            diy::save(bb, b->bounds_maxs);
            diy::save(bb, b->core_mins);
            diy::save(bb, b->core_maxs);

            // geometry
            diy::save(bb, b->geometry.mfa_data->p);
            diy::save(bb, b->geometry.mfa_data->tmesh.tensor_prods.size());
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.nctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.ctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.weights);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.knot_mins);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.knot_maxs);
            diy::save(bb, b->geometry.mfa_data->tmesh.all_knots);
            diy::save(bb, b->geometry.mfa_data->tmesh.all_knot_levels);

            // science variables
            diy::save(bb, b->vars.size());
            for (auto i = 0; i < b->vars.size(); i++)
            {
                diy::save(bb, b->vars[i].mfa_data->p);
                diy::save(bb, b->vars[i].mfa_data->tmesh.tensor_prods.size());
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.nctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.ctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.weights);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.knot_mins);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.knot_maxs);
                diy::save(bb, b->vars[i].mfa_data->tmesh.all_knots);
                diy::save(bb, b->vars[i].mfa_data->tmesh.all_knot_levels);
            }

            diy::save(bb, b->approx);
            diy::save(bb, b->errs);
        }
    static
        void load(
                void*              b_,
                diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;

            // TODO: don't load domain in practice
            diy::load(bb, b->domain);

            // top-level mfa data
            diy::load(bb, b->dom_dim);
            VectorXi ndom_pts(b->dom_dim);
            diy::load(bb, ndom_pts);
            b->mfa = new mfa::MFA<T>(b->dom_dim, ndom_pts, b->domain);

            diy::load(bb, b->bounds_mins);
            diy::load(bb, b->bounds_maxs);
            diy::load(bb, b->core_mins);
            diy::load(bb, b->core_maxs);

            VectorXi    p;                  // degree of the mfa
            size_t      ntensor_prods;      // number of tensor products in the tmesh

            // geometry
            diy::load(bb, p);
            diy::load(bb, ntensor_prods);
            b->geometry.mfa_data = new mfa::MFA_Data<T>(p, ntensor_prods);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.nctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.ctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.weights);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.knot_mins);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.knot_maxs);
            diy::load(bb, b->geometry.mfa_data->tmesh.all_knots);
            diy::load(bb, b->geometry.mfa_data->tmesh.all_knot_levels);

            // science variables
            size_t nvars;
            diy::load(bb, nvars);
            b->vars.resize(nvars);
            for (auto i = 0; i < b->vars.size(); i++)
            {
                diy::load(bb, p);
                diy::load(bb, ntensor_prods);
                b->vars[i].mfa_data = new mfa::MFA_Data<T>(p, ntensor_prods);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.nctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.ctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.weights);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.knot_mins);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.knot_maxs);
                diy::load(bb, b->vars[i].mfa_data->tmesh.all_knots);
                diy::load(bb, b->vars[i].mfa_data->tmesh.all_knot_levels);
            }

            diy::load(bb, b->approx);
            diy::load(bb, b->errs);
        }

    // evaluate sine function
    T sine(VectorX<T>&  domain_pt,
           DomainArgs&  args,
           int          k)                  // current science variable
    {
        DomainArgs* a = &args;
        T retval = 1.0;
        for (auto i = 0; i < dom_dim; i++)
            retval *= sin(domain_pt(i) * a->f[k]);
        retval *= a->s[k];

        return retval;
    }

    // evaluate sinc function
    T sinc(VectorX<T>&  domain_pt,
           DomainArgs&  args,
           int          k)                  // current science variable
    {
        DomainArgs* a = &args;
        T retval = 1.0;
        for (auto i = 0; i < dom_dim; i++)
        {
            if (domain_pt(i) != 0.0)
                retval *= (sin(domain_pt(i) * a->f[k] ) / domain_pt(i));
        }
        retval *= a->s[k];

        return retval;
    }

    // evaluate f16 function
    T f16(VectorX<T>&   domain_pt)
    {
        T retval =
            (pow(domain_pt(0), 4)                        +
             pow(domain_pt(1), 4)                        +
             pow(domain_pt(0), 2) * pow(domain_pt(1), 2) +
             domain_pt(0) * domain_pt(1)                 ) /
            (pow(domain_pt(0), 3)                        +
             pow(domain_pt(1), 3)                        +
             4                                           );

        return retval;
    }

    // evaluate f17 function
    T f17(VectorX<T>&   domain_pt)
    {
        T E         = domain_pt(0);
        T G         = domain_pt(1);
        T M         = domain_pt(2);
        T gamma     = sqrt(M * M * (M * M + G * G));
        T kprop     = (2.0 * sqrt(2.0) * M * G * gamma ) / (M_PI * sqrt(M * M + gamma));
        T retval    = kprop / ((E * E - M * M) * (E * E - M * M) + M * M * G * G);

        return retval;
    }

    // evaluate f18 function
    T f18(VectorX<T>&   domain_pt)
    {
        T x1        = domain_pt(0);
        T x2        = domain_pt(1);
        T x3        = domain_pt(2);
        T x4        = domain_pt(3);
        T retval    = (atanh(x1) + atanh(x2) + atanh(x3) + atanh(x4)) / ((pow(x1, 2) - 1) * pow(x2, -1));

        return retval;
    }

    // synthetic analytical data
    void generate_analytical_data(
            const diy::Master::ProxyWithLink&   cp,
            string&                             fun,        // function to evaluate
            DomainArgs&                         args)
    {
        DomainArgs* a   = &args;
        int nvars       = pt_dim - dom_dim;             // number of science variables
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        int tot_ndom_pts    = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        for (int j = 0; j < nvars; j++)
        {
            vars[j].min_dim = dom_dim + j;
            vars[j].max_dim = vars[j].min_dim + 1;
        }
        VectorXi ndom_pts(dom_dim);
        for (int i = 0; i < dom_dim; i++)
            ndom_pts(i) = a->ndom_pts[i];

        // get local block bounds
        // if single block, they are passed in args
        // if multiblock, they were decomposed by diy and are already in the block's bounds_mins, maxs
        if (!a->multiblock)
        {
            bounds_mins.resize(pt_dim);
            bounds_maxs.resize(pt_dim);
            core_mins.resize(dom_dim);
            core_maxs.resize(dom_dim);
            for (int i = 0; i < dom_dim; i++)
            {
                bounds_mins(i)  = a->min[i];
                bounds_maxs(i)  = a->max[i];
                core_mins(i)    = a->min[i];
                core_maxs(i)    = a->max[i];
            }
        }

        // adjust number of domain points and starting domain point for ghost
        VectorX<T> d(dom_dim);               // step in domain points in each dimension
        VectorX<T> p0(dom_dim);              // starting point in each dimension
        int nghost_pts;                         // number of ghost points in current dimension
        for (int i = 0; i < dom_dim; i++)
        {
            d(i) = (core_maxs(i) - core_mins(i)) / (ndom_pts(i) - 1);
            // min direction
            nghost_pts = floor((core_mins(i) - bounds_mins(i)) / d(i));
            ndom_pts(i) += nghost_pts;
            p0(i) = core_mins(i) - nghost_pts * d(i);
            // max direction
            nghost_pts = floor((bounds_maxs(i) - core_maxs(i)) / d(i));
            ndom_pts(i) += nghost_pts;
            tot_ndom_pts *= ndom_pts(i);
        }

        domain.resize(tot_ndom_pts, pt_dim);

        // assign values to the domain (geometry)
        vector<int> dom_idx(dom_dim);                   // current index of domain point in each dim, initialized to 0s
        for (auto j = 0; j < tot_ndom_pts; j++)         // flattened loop over all the points in a domain in dimension dom_dim
        {
            // compute geometry coordinates of domain point
            for (auto i = 0; i < dom_dim; i++)
                domain(j, i) = p0(i) + dom_idx[i] * d(i);

            dom_idx[0]++;

            // for all dimensions except last, check for end of the line, part of flattened loop logic
            for (auto k = 0; k < dom_dim - 1; k++)
            {
                if (dom_idx[k] == ndom_pts(k))
                {
                    dom_idx[k] = 0;
                    dom_idx[k + 1]++;
                }
            }
        }

        // normal distribution for generating noise
        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 1.0);

        // assign values to the range (science variables)
        // hard-coded for 2 domain dimensions and 1 science variable
        VectorX<T> dom_pt(dom_dim);
        for (int j = 0; j < tot_ndom_pts; j++)
        {
            dom_pt = domain.block(j, 0, 1, dom_dim).transpose();
            T retval;
            for (auto k = 0; k < nvars; k++)        // for all science variables
            {
                if (fun == "sine")
                    retval = sine(dom_pt, args, k);
                if (fun == "sinc")
                    retval = sinc(dom_pt, args, k);
                if (fun == "f16")
                    retval = f16(dom_pt);
                if (fun == "f17")
                    retval = f17(dom_pt);
                if (fun == "f18")
                    retval = f18(dom_pt);
                domain(j, dom_dim + k) = retval;
            }

            // add some noise
            double noise = distribution(generator);
            domain(j, dom_dim) *= (1.0 + a->n * noise);

            if (j == 0 || domain(j, dom_dim) > bounds_maxs(dom_dim))
                bounds_maxs(dom_dim) = domain(j, dom_dim);
            if (j == 0 || domain(j, dom_dim) < bounds_mins(dom_dim))
                bounds_mins(dom_dim) = domain(j, dom_dim);
        }

        // optional wavy domain
        if (a->t && pt_dim >= 3)
        {
            for (auto j = 0; j < tot_ndom_pts; j++)
            {
                real_t x = domain(j, 0);
                real_t y = domain(j, 1);
                domain(j, 0) += a->t * sin(y);
                domain(j, 1) += a->t * sin(x);
                if (j == 0 || domain(j, 0) < bounds_mins(0))
                    bounds_mins(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) < bounds_mins(1))
                    bounds_mins(1) = domain(j, 1);
                if (j == 0 || domain(j, 0) > bounds_maxs(0))
                    bounds_maxs(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) > bounds_maxs(1))
                    bounds_maxs(1) = domain(j, 1);
            }
        }

        // optional rotation of the domain
        if (a->r && pt_dim >= 3)
        {
            for (auto j = 0; j < tot_ndom_pts; j++)
            {
                real_t x = domain(j, 0);
                real_t y = domain(j, 1);
                domain(j, 0) = x * cos(a->r) - y * sin(a->r);
                domain(j, 1) = x * sin(a->r) + y * cos(a->r);
                if (j == 0 || domain(j, 0) < bounds_mins(0))
                    bounds_mins(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) < bounds_mins(1))
                    bounds_mins(1) = domain(j, 1);
                if (j == 0 || domain(j, 0) > bounds_maxs(0))
                    bounds_maxs(0) = domain(j, 0);
                if (j == 0 || domain(j, 1) > bounds_maxs(1))
                    bounds_maxs(1) = domain(j, 1);
            }
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // extents
        fprintf(stderr, "gid = %d\n", cp.gid());
        cerr << "core_mins:\n" << core_mins << endl;
        cerr << "core_maxs:\n" << core_maxs << endl;
        cerr << "bounds_mins:\n" << bounds_mins << endl;
        cerr << "bounds_maxs:\n" << bounds_maxs << endl;

//         cerr << "ndom_pts:\n" << ndom_pts << "\n" << endl;
//         cerr << "domain:\n" << domain << endl;
    }

    // read a floating point 3d vector dataset and take one 1-d curve out of the middle of it
    // f = (x, velocity magnitude)
    void read_1d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        int nvars = 1;
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        vars[0].min_dim = dom_dim;
        vars[0].max_dim = vars[0].min_dim + 1;
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, pt_dim);
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
            if (i == 0 || domain(i, 1) < bounds_mins(1))
                bounds_mins(1) = domain(i, 1);
            if (i == 0 || domain(i, 1) > bounds_maxs(1))
                bounds_maxs(1) = domain(i, 1);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
        {
            domain(n, 0) = i;
            n++;
        }

        // extents
        bounds_mins(0) = 0.0;
        bounds_maxs(0) = domain(tot_ndom_pts - 1, 0);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2-d surface out of the middle of it
    // f = (x, y, velocity magnitude)
    void read_2d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        int nvars = 1;
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        vars[0].min_dim = dom_dim;
        vars[0].max_dim = vars[0].min_dim + 1;
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, pt_dim);
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
            if (i == 0 || domain(i, 2) < bounds_mins(2))
                bounds_mins(2) = domain(i, 2);
            if (i == 0 || domain(i, 2) > bounds_maxs(2))
                bounds_maxs(2) = domain(i, 2);
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
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_maxs(0) = domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = domain(tot_ndom_pts - 1, 1);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2d (parallel to x-y plane) subset
    // f = (x, y, velocity magnitude)
    void read_2d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        int nvars = 1;
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        vars[0].min_dim = dom_dim;
        vars[0].max_dim = vars[0].min_dim + 1;
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, pt_dim);
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
            if (i == 0 || domain(i, 2) < bounds_mins(2))
                bounds_mins(2) = domain(i, 2);
            if (i == 0 || domain(i, 2) > bounds_maxs(2))
                bounds_maxs(2) = domain(i, 2);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        bounds_mins(0) = domain(0, 0);
        bounds_mins(1) = domain(0, 1);
        bounds_maxs(0) = domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = domain(tot_ndom_pts - 1, 1);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset
    // f = (x, y, z, velocity magnitude)
    void read_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        int nvars = 1;
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        vars[0].min_dim = dom_dim;
        vars[0].max_dim = vars[0].min_dim + 1;
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, pt_dim);

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
            if (i == 0 || domain(i, 3) < bounds_mins(3))
                bounds_mins(3) = domain(i, 3);
            if (i == 0 || domain(i, 3) > bounds_maxs(3))
                bounds_maxs(3) = domain(i, 3);
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
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_mins(2) = 0.0;
        bounds_maxs(0) = domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = domain(tot_ndom_pts - 1, 1);
        bounds_maxs(2) = domain(tot_ndom_pts - 1, 2);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take a 3d subset out of it
    // f = (x, y, z, velocity magnitude)
    void read_3d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        int nvars = 1;
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        vars[0].min_dim = dom_dim;
        vars[0].max_dim = vars[0].min_dim + 1;
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, pt_dim);
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
            if (i == 0 || domain(i, 3) < bounds_mins(3))
                bounds_mins(3) = domain(i, 3);
            if (i == 0 || domain(i, 3) > bounds_maxs(3))
                bounds_maxs(3) = domain(i, 3);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        bounds_mins(0) = domain(0, 0);
        bounds_mins(1) = domain(0, 1);
        bounds_mins(2) = domain(0, 2);
        bounds_maxs(0) = domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = domain(tot_ndom_pts - 1, 1);
        bounds_maxs(2) = domain(tot_ndom_pts - 1, 2);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 2d scalar dataset
    // f = (x, y, value)
    void read_2d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        int nvars = 1;
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        vars[0].min_dim = dom_dim;
        vars[0].max_dim = vars[0].min_dim + 1;
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, pt_dim);

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
            if (i == 0 || domain(i, 2) < bounds_mins(2))
                bounds_mins(2) = domain(i, 2);
            if (i == 0 || domain(i, 2) > bounds_maxs(2))
                bounds_maxs(2) = domain(i, 2);
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
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_maxs(0) = domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = domain(tot_ndom_pts - 1, 1);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d scalar dataset
    // f = (x, y, z, value)
    void read_3d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        geometry.min_dim = 0;
        geometry.max_dim = dom_dim - 1;
        int nvars = 1;
        vars.resize(nvars);
        max_errs.resize(nvars);
        sum_sq_errs.resize(nvars);
        vars[0].min_dim = dom_dim;
        vars[0].max_dim = vars[0].min_dim + 1;
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        domain.resize(tot_ndom_pts, pt_dim);

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
            if (i == 0 || domain(i, 3) < bounds_mins(3))
                bounds_mins(3) = domain(i, 3);
            if (i == 0 || domain(i, 3) > bounds_maxs(3))
                bounds_maxs(3) = domain(i, 3);
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
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_mins(2) = 0.0;
        bounds_maxs(0) = domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = domain(tot_ndom_pts - 1, 1);
        bounds_maxs(2) = domain(tot_ndom_pts - 1, 2);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        mfa = new mfa::MFA<T>(dom_dim, ndom_pts, domain);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // fixed number of control points encode block
    void fixed_encode_block(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;

        VectorXi nctrl_pts(dom_dim);
        VectorXi p(dom_dim);
        for (auto j = 0; j < dom_dim; j++)
        {
            nctrl_pts(j)    = a->geom_nctrl_pts[j];
            p(j)            = a->geom_p[j];
        }

        // encode geometry
        if (a->verbose && cp.master()->communicator().rank() == 0)
            fprintf(stderr, "\nEncoding geometry\n\n");
        geometry.mfa_data = new mfa::MFA_Data<T>(p,
                                       mfa->ndom_pts(),
                                       domain,
                                       mfa->params(),
                                       nctrl_pts,
                                       0,
                                       dom_dim - 1);
        // TODO: consider not weighting the geometry (only science variables), depends on geometry complexity
        mfa->FixedEncode(*geometry.mfa_data, domain, nctrl_pts, a->verbose, a->weighted);

        // encode science variables
        // same number of control points and same degree for all variables in this example
        for (auto j = 0; j < dom_dim; j++)
        {
            nctrl_pts(j)    = a->vars_nctrl_pts[j];
            p(j)            = a->vars_p[j];
        }

        for (auto i = 0; i< vars.size(); i++)
        {
            if (a->verbose && cp.master()->communicator().rank() == 0)
                fprintf(stderr, "\nEncoding science variable %d\n\n", i);
            vars[i].mfa_data = new mfa::MFA_Data<T>(p,
                                          mfa->ndom_pts(),
                                          domain,
                                          mfa->params(),
                                          nctrl_pts,
                                          dom_dim + i,        // assumes each variable is scalar
                                          dom_dim + i);
            mfa->FixedEncode(*(vars[i].mfa_data), domain, nctrl_pts, a->verbose, a->weighted);
        }
    }

    // adaptively encode block to desired error limit
    void adaptive_encode_block(
            const diy::Master::ProxyWithLink& cp,
            real_t                            err_limit,
            int                               max_rounds,
            DomainArgs&                       args)
    {
        DomainArgs* a = &args;
        VectorXi nctrl_pts(dom_dim);
        VectorXi p(dom_dim);
        VectorXi ndom_pts(dom_dim);
        VectorX<T> extents = bounds_maxs - bounds_mins;
        for (auto j = 0; j < dom_dim; j++)
        {
            nctrl_pts(j)    = a->geom_nctrl_pts[j];
            ndom_pts(j)     = a->ndom_pts[j];
            p(j)            = a->geom_p[j];
        }

        // encode geometry
        if (a->verbose && cp.master()->communicator().rank() == 0)
            fprintf(stderr, "\nEncoding geometry\n\n");
        geometry.mfa_data = new mfa::MFA_Data<T>(p,
                                       mfa->ndom_pts(),
                                       domain,
                                       mfa->params(),
                                       nctrl_pts,
                                       0,
                                       dom_dim - 1);
        // TODO: consider not weighting the geometry (only science variables), depends on geometry complexity
        mfa->AdaptiveEncode(*geometry.mfa_data, domain, err_limit, a->verbose, a->weighted, extents, max_rounds);

        // encode science variables
        for (auto j = 0; j < dom_dim; j++)
        {
            nctrl_pts(j)    = a->vars_nctrl_pts[j];
            p(j)            = a->vars_p[j];
        }

        for (auto i = 0; i< vars.size(); i++)
        {
            if (a->verbose && cp.master()->communicator().rank() == 0)
                fprintf(stderr, "\nEncoding science variable %d\n\n", i);
            vars[i].mfa_data = new mfa::MFA_Data<T>(p,
                                          mfa->ndom_pts(),
                                          domain,
                                          mfa->params(),
                                          nctrl_pts,
                                          dom_dim + i,        // assumes each variable is scalar
                                          dom_dim + i);
            mfa->AdaptiveEncode(*(vars[i].mfa_data), domain, err_limit, a->verbose, a->weighted, extents, max_rounds);
        }
    }

    // decode entire block
    void decode_block(
            const   diy::Master::ProxyWithLink& cp,
            int                                 verbose,        // debug level
            bool                                saved_basis)    // whether basis functions were saved and can be reused
    {
        approx.resize(domain.rows(), domain.cols());

        // geometry
        mfa->DecodeDomain(*geometry.mfa_data, verbose, approx, 0, dom_dim - 1, saved_basis);

        // science variables
        for (auto i = 0; i < vars.size(); i++)
            mfa->DecodeDomain(*(vars[i].mfa_data), verbose, approx, dom_dim + i, dom_dim + i, saved_basis);  // assumes each variable is scalar
    }

    // differentiate entire block
    void differentiate_block(
            const diy::Master::ProxyWithLink& cp,
            int                               verbose,  // debug level
            int                               deriv,    // which derivative to take (1 = 1st, 2 = 2nd, ...) in each domain dim.
            int                               partial,  // limit to partial derivative in just this dimension (-1 = no limit)
            int                               var)      // differentiate only this one science variable (0 to nvars -1, -1 = all vars)
    {
        approx.resize(domain.rows(), domain.cols());
        VectorXi derivs(dom_dim);

        for (auto i = 0; i < derivs.size(); i++)
            derivs(i) = deriv;

        // optional limit to one partial derivative
        if (deriv && dom_dim > 1 && partial >= 0)
        {
            for (auto i = 0; i < dom_dim; i++)
            {
                if (i != partial)
                    derivs(i) = 0;
            }
        }

        // science variables
        for (auto i = 0; i < vars.size(); i++)
            if (var < 0 || var == i)
            {
                // TODO: hard-coded for one tensor product
                vars[i].mfa_data = new mfa::MFA_Data<T>(vars[i].mfa_data->p,
                                              mfa->ndom_pts(),
                                              vars[i].mfa_data->tmesh,
                                              dom_dim + i,        // assumes each variable is scalar
                                              dom_dim + i);
                mfa->DecodeDomain(*(vars[i].mfa_data), verbose, approx, dom_dim + i, dom_dim + i, false, derivs);  // assumes each variable is scalar
            }

        // the derivative is a vector of same dimensionality as domain
        // derivative needs to be scaled by domain extent because u,v,... are in [0.0, 1.0]
        if (deriv)
        {
            if (dom_dim == 1 || partial >= 0) // TODO: not for mixed partials
            {
                if (dom_dim == 1)
                    partial = 0;
                for (auto j = 0; j < approx.cols(); j++)
                    // scale once for each derivative
                    for (auto i = 0; i < deriv; i++)
                        approx.col(j) /= (bounds_maxs(partial) - bounds_mins(partial));
            }
        }

        // for plotting, set the geometry coordinates to be the same as the input
        if (deriv)
            for (auto i = 0; i < dom_dim; i++)
                approx.col(i) = domain.col(i);
    }

// TODO: convert the following to split models

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
//             int dom_dim = ndom_pts.size();                // domain dimensionality
//             // geometry
//             geometry.mfa->Decode(approx, 0, dom_dim - 1);
// 
//             // science variables
//             for (auto i = 0; i < vars.size(); i++)
//                 vars[i].mfa->Decode(approx, dom_dim, dom_dim + i);
//         }
// 
// #ifndef MFA_NO_TBB                                          // TBB version
// 
//         // distance computation
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
//                 max_err = errs(i, errs.cols() - 1);
//         }
// 
// #else                                               // single thread version
// 
//         // distance computation
//         // TODO: only last variable
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
//                 max_err = errs(i, errs.cols() - 1);
//         }
// 
// #endif
// 
// //         mfa->max_err = max_err;
//     }

    // compute error to synthethic, non-noisy function (for HEP applications)
    // outputs L1, L2, Linfinity error
    // optionally outputs true and test point locations and true and test values there
    // if optional test output wanted, caller has to allocate true_pts, test_pts, true_data, and test_data
    void analytical_error(
            const diy::Master::ProxyWithLink&     cp,
            string&                               fun,                // function to evaluate
            T&                                    L1,                 // (output) L-1 norm
            T&                                    L2,                 // (output) L-2 norm
            T&                                    Linf,               // (output) L-infinity norm
            DomainArgs&                           args,               // input args
            bool                                  output,             // whether to output test_pts, true_data, test_data or only compute error norms
            vector<vec3d>&                        true_pts,           // (output) true points locations
            float**                               true_data,          // (output) true data values (4d)
            vector<vec3d>&                        test_pts,           // (output) test points locations
            float**                               test_data)          // (output) test data values (4d)
    {
        DomainArgs* a   = &args;

        T sum_errs      = 0.0;                                  // sum of absolute values of errors (L-1 norm)
        T sum_sq_errs   = 0.0;                                  // sum of squares of errors (square of L-2 norm)
        T max_err       = -1.0;                                 // maximum absolute value of error (L-infinity norm)

        if (!dom_dim)
            dom_dim = mfa->ndom_pts().size();

        size_t tot_ndom_pts = 1;
        for (auto i = 0; i < dom_dim; i++)
            tot_ndom_pts *= a->ndom_pts[i];

        vector<int> dom_idx(dom_dim);                           // current index of domain point in each dim, initialized to 0s

        // steps in each dimension in paramater space and real space
        vector<T> dom_step_real(dom_dim);                       // spacing between domain points in real space
        vector<T> dom_step_param(dom_dim);                      // spacing between domain points in parameter space
        for (auto i = 0; i < dom_dim; i++)
        {
            dom_step_param[i] = 1.0 / (double)(a->ndom_pts[i] - 1);
            dom_step_real[i] = dom_step_param[i] * (a->max[i] - a->min[i]);
        }

        // flattened loop over all the points in a domain in dimension dom_dim
        fmt::print(stderr, "Testing analytical error norms over a total of {} points\n", tot_ndom_pts);
        for (auto j = 0; j < tot_ndom_pts; j++)
        {
            // compute current point in real and parameter space
            VectorX<T> dom_pt_real(dom_dim);                // one domain point in real space
            VectorX<T> dom_pt_param(dom_dim);               // one domain point in parameter space
            for (auto i = 0; i < dom_dim; i++)
            {
                dom_pt_real(i) = a->min[i] + dom_idx[i] * dom_step_real[i];
                dom_pt_param(i) = dom_idx[i] * dom_step_param[i];
            }

            // evaluate function at dom_pt_real
            T true_val;
            if (fun == "sinc")
                true_val = sinc(dom_pt_real, args, 0);      // hard-coded for one science variable
            if (fun == "sine")
                true_val = sine(dom_pt_real, args, 0);      // hard-codded for one science variable
            if (fun == "f16")
                true_val = f16(dom_pt_real);
            if (fun == "f17")
                true_val = f17(dom_pt_real);
            if (fun == "f18")
                true_val = f18(dom_pt_real);

            // evaluate MFA at dom_pt_param
            VectorX<T> cpt(1);                              // hard-coded for one science variable
            mfa->DecodePt(*(vars[0].mfa_data), dom_pt_param, cpt);       // hard-coded for one science variable
            T test_val = cpt(0);

            // compute and accrue error
            T err = fabs(true_val - test_val);
            sum_errs += err;                                // L1
            sum_sq_errs += err * err;                       // L2
            if (err > max_err)                              // Linf
                max_err = err;

            if (output)
            {
                vec3d true_pt, test_pt;
                true_pt.x = test_pt.x = dom_pt_real(0);
                if (dom_dim > 2)        // 3d or greater domain
                {
                    true_pt.y = test_pt.y = dom_pt_real(1);
                    true_pt.z = test_pt.z = dom_pt_real(2);
                }
                else if (dom_dim > 1)   // 2d domain
                {
                    true_pt.y = test_pt.y = dom_pt_real(1);
                    true_pt.z = true_val;
                    test_pt.z = test_val;
                }
                else                    // 1d domain
                {
                    true_pt.y = true_val;
                    test_pt.y = test_val;
                    true_pt.z = test_pt.z = 0.0;
                }

                true_pts[j] = true_pt;
                test_pts[j] = test_pt;

                true_data[0][j] = true_val;
                test_data[0][j] = test_val;

//                 fmt::print(stderr, "x={} y={} true_val={} test_val={}\n", test_pt.x, test_pt.y, true_val, test_val);
            }

            dom_idx[0]++;

            // for all dimensions except last, check if pt_idx is at the end, part of flattened loop logic
            for (auto k = 0; k < dom_dim - 1; k++)
            {
                if (dom_idx[k] == mfa->ndom_pts()(k))
                {
                    dom_idx[k] = 0;
                    dom_idx[k + 1]++;
                }
            }
        }                                                   // for all points in flattened loop

        L1    = sum_errs;
        L2    = sqrt(sum_sq_errs);
        Linf  = max_err;
    }

    // compute error field and maximum error in the block
    // uses coordinate-wise difference between values
    void range_error(
            const   diy::Master::ProxyWithLink& cp,
            int     verbose,                                // output level
            bool    decode_block_,                          // decode entire block first
            bool    saved_basis)                            // whether basis functions were saved and can be reused
    {
        errs.resize(domain.rows(), domain.cols());
        errs            = domain;

        if (decode_block_)
            decode_block(cp, verbose, saved_basis);

#ifndef MFA_NO_TBB                                          // TBB version

        // distance computation
        if (decode_block_)
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                        VectorX<T> cpt = approx.row(i);
                        for (auto j = 0; j < domain.cols(); j++)
                        {
                            T err = fabs(cpt(j) - domain(i, j));
                            if (j >= dom_dim)
                                errs(i, j) = err;           // error for each science variable
                        }
                    });
        }
        else
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                        VectorX<T> err;                                 // errors for all coordinates in current model
                        for (auto k = 0; k < vars.size() + 1; k++)      // for all models, geometry + science
                        {
                            if (k == 0)                                 // geometry
                            {
                                err.resize(geometry.max_dim - geometry.min_dim);
                                mfa->AbsCoordError(*geometry.mfa_data, domain, i, err, verbose);
                            }
                            else
                            {
                                err.resize(vars[k - 1].max_dim - vars[k - 1].min_dim);
                                mfa->AbsCoordError(*(vars[k - 1].mfa_data), domain, i, err, verbose);
                            }

                            for (auto j = 0; j < err.size(); j++)
                                if (k)                                              // science variables
                                    errs(i, vars[k - 1].min_dim + j) = err(j); // error for each science variable
                        }
                    });
        }

#else                                               // single thread version

        for (auto i = 0; i < (size_t)domain.rows(); i++)
        {
            if (decode_block_)
            {
                VectorX<T> cpt = approx.row(i);
                for (auto j = 0; j < domain.cols(); j++)
                {
                    T err = fabs(cpt(j) - domain(i, j));
                    if (j >= dom_dim)
                        errs(i, j) = err;           // error for each science variable
                }
            }
            else
            {
                VectorX<T> err;                                 // errors for all coordinates in current model
                for (auto k = 0; k < vars.size() + 1; k++)      // for all models, geometry + science
                {
                    if (k == 0)                                 // geometry
                    {
                        err.resize(geometry.max_dim - geometry.min_dim);
                        mfa->AbsCoordError(*geometry.mfa_data, domain, i, err, verbose);
                    }
                    else
                    {
                        err.resize(vars[k - 1].max_dim - vars[k - 1].min_dim);
                        mfa->AbsCoordError(*(vars[k - 1].mfa_data),domain, i, err, verbose);
                    }

                    for (auto j = 0; j < err.size(); j++)
                    {
                        if (k)                                              // science variables
                            errs(i, vars[k - 1].min_dim + j) = err(j);      // error for each science variable
                    }
                }
            }
        }

#endif

        for (auto j = dom_dim; j < domain.cols(); j++)
            sum_sq_errs[j - dom_dim] = 0.0;
        for (auto i = 0; i < domain.rows(); i++)
        {
            for (auto j = dom_dim; j < domain.cols(); j++)
            {
                sum_sq_errs[j - dom_dim] += (errs(i, j) * errs(i, j));
                if ((i == 0 && j == dom_dim) || errs(i, j) > max_errs[j - dom_dim])
                    max_errs[j - dom_dim] = errs(i, j);
            }
        }
    }

    void print_block(const diy::Master::ProxyWithLink& cp,
                     bool                              error)       // error was computed
    {
        fprintf(stderr, "gid = %d\n", cp.gid());
//         cerr << "domain\n" << domain << endl;

        // geometry
        // TODO: hard-coded for one tensor product
        cerr << "\n------- geometry model -------" << endl;
        cerr << "# output ctrl pts     = [ " << geometry.mfa_data->tmesh.tensor_prods[0].nctrl_pts.transpose() << " ]" << endl;

        //  debug: print control points and weights
//         print_ctrl_weights(geometry.mfa_data->tmesh.;
        // debug: print knots
//         print_knots(geometry.mfa_data->tmesh.;

        fprintf(stderr, "# output knots        = [ ");
        for (auto j = 0 ; j < geometry.mfa_data->tmesh.all_knots.size(); j++)
        {
            fprintf(stderr, "%ld ", geometry.mfa_data->tmesh.all_knots[j].size());
        }
        fprintf(stderr, "]\n");

        cerr << "-----------------------------" << endl;

        // science variables
        cerr << "\n----- science variable models -----" << endl;
        for (auto i = 0; i < vars.size(); i++)
        {
            real_t range_extent = domain.col(dom_dim + i).maxCoeff() - domain.col(dom_dim + i).minCoeff();
            cerr << "\n---------- var " << i << " ----------" << endl;
            cerr << "# ouput ctrl pts      = [ " << vars[i].mfa_data->tmesh.tensor_prods[0].nctrl_pts.transpose() << " ]" << endl;

            //  debug: print control points and weights
//             print_ctrl_weights(vars[i].mfa_data->tmesh.;
            // debug: print knots
//             print_knots(vars[i].mfa_data->tmesh.;


            fprintf(stderr, "# output knots        = [ ");
            for (auto j = 0 ; j < vars[i].mfa_data->tmesh.all_knots.size(); j++)
            {
                fprintf(stderr, "%ld ", vars[i].mfa_data->tmesh.all_knots[j].size());
            }
            fprintf(stderr, "]\n");

            cerr << "-----------------------------" << endl;

            T rms_err = sqrt(sum_sq_errs[i] / (domain.rows()));
            fprintf(stderr, "range extent          = %e\n",  range_extent);
            if (error)
            {
                fprintf(stderr, "max_err               = %e\n",  max_errs[i]);
                fprintf(stderr, "normalized max_err    = %e\n",  max_errs[i] / range_extent);
                fprintf(stderr, "sum of squared errors = %e\n",  sum_sq_errs[i]);
                fprintf(stderr, "RMS error             = %e\n",  rms_err);
                fprintf(stderr, "normalized RMS error  = %e\n",  rms_err / range_extent);
            }
            cerr << "-----------------------------" << endl;
        }
        cerr << "\n-----------------------------------" << endl;

        //  debug: print approximated points
//         cerr << approx.rows() << " approximated points\n" << approx << endl;
//         fprintf(stderr, "# input points        = %ld\n", domain.rows());

        fprintf(stderr, "compression ratio     = %.2f\n", compute_compression());
    }

    // compute compression ratio
    float compute_compression()
    {
        // TODO: hard-coded for one tensor product
        float in_coords = domain.rows() * domain.cols();
        float out_coords = geometry.mfa_data->tmesh.tensor_prods[0].ctrl_pts.rows() *
            geometry.mfa_data->tmesh.tensor_prods[0].ctrl_pts.cols();
        for (auto j = 0; j < geometry.mfa_data->tmesh.all_knots.size(); j++)
            out_coords += geometry.mfa_data->tmesh.all_knots[j].size();
        for (auto i = 0; i < vars.size(); i++)
        {
            out_coords += (vars[i].mfa_data->tmesh.tensor_prods[0].ctrl_pts.rows() *
                    vars[i].mfa_data->tmesh.tensor_prods[0].ctrl_pts.cols());
            for (auto j = 0; j < vars[i].mfa_data->tmesh.all_knots.size(); j++)
                out_coords += vars[i].mfa_data->tmesh.all_knots[j].size();
        }
        return(in_coords / out_coords);
    }

    //  debug: print control points and weights in all tensor products of a tmesh
    void print_ctrl_weights(mfa::Tmesh<T>& tmesh)
    {
        for (auto i = 0; i < tmesh.tensor_prods.size(); i++)
        {
            cerr << "tensor_prods[" << i << "]:\n" << endl;
            cerr << tmesh.tensor_prods[i].ctrl_pts.rows() <<
                " final control points\n" << tmesh.tensor_prods[i].ctrl_pts << endl;
            cerr << tmesh.tensor_prods[i].weights.size()  <<
                " final weights\n" << tmesh.tensor_prods[i].weights << endl;
        }
    }

    // debug: print knots in a tmesh
    void print_knots(mfa::Tmesh<T>& tmesh)
    {
        for (auto j = 0 ; j < tmesh.all_knots.size(); j++)
        {
            fprintf(stderr, "%ld knots[%d]: [ ", tmesh.all_knots[j].size(), j);
            for (auto k = 0; k < tmesh.all_knots[j].size(); k++)
                fprintf(stderr, "%.3lf ", tmesh.all_knots[j][k]);
            fprintf(stderr, " ]\n");
        }
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

    // send decoded ghost points
    // assumes entire block was already decoded
    void send_ghost_pts(const diy::Master::ProxyWithLink&   cp,
                        const Decomposer&                   decomposer)
    {
        RCLink *l = static_cast<RCLink *>(cp.link());
        map<diy::BlockID, vector<VectorX<T> > > outgoing_pts;
        vector<T>   dom_pt(dom_dim);                    // only domain coords of point, for checking neighbor bounds
        VectorX<T>  full_pt(approx.cols());             // full coordinates of point
        T eps = 1.0e-6;

        // check decoded points whether they fall into neighboring block bounds (including ghost)
        for (auto i = 0; i < (size_t)approx.rows(); i++)
        {
            vector<int> dests;                      // link neighbor targets (not gids)
            auto it = dests.begin();
            insert_iterator<vector<int> > insert_it(dests, it);
            for (auto j = 0; j < dom_dim; j++)
                dom_pt[j] = approx(i, j);
            diy::near(*l, dom_pt, eps, insert_it, decomposer.domain);
            if (dests.size())
                full_pt = approx.row(i);

            // prepare map of pts going to each neighbor
            for (auto j = 0; j < dests.size(); j++)
            {
                diy::BlockID bid = l->target(dests[j]);
                outgoing_pts[bid].push_back(full_pt);
                // debug: print the point
                cerr << "gid " << cp.gid() << " sent " << full_pt.transpose() << " to gid " << bid.gid << endl;
            }
        }

        // enqueue the vectors of points to send to each neighbor block
        for (auto it = outgoing_pts.begin(); it != outgoing_pts.end(); it++)
            for (auto i = 0; i < it->second.size(); i++)
                cp.enqueue(it->first, it->second[i]);
    }

    void recv_ghost_pts(const diy::Master::ProxyWithLink& cp)
    {
        VectorX<T> pt(approx.cols());                   // incoming point

        // gids of incoming neighbors in the link
        std::vector<int> in;
        cp.incoming(in);

        // for all neighbor blocks
        // dequeue data received from this neighbor block in the last exchange
        for (unsigned i = 0; i < in.size(); ++i)
        {
            while (cp.incoming(in[i]))
            {
                cp.dequeue(in[i], pt);
                // debug: print the point
                cerr << "gid " << cp.gid() << " received " << pt.transpose() << endl;
            }
        }
    }

    // ----- t-mesh methods -----

    // initialize t-mesh with some test data
    void init_tmesh(const diy::Master::ProxyWithLink&   cp,
                    DomainArgs&                         args)
    {
        DomainArgs* a = &args;

        // geometry mfa
        geometry.mfa = new mfa::MFA<T>(geometry.p,
                                       mfa->ndom_pts(),
                                       domain,
                                       geometry.ctrl_pts,
                                       geometry.nctrl_pts,
                                       geometry.weights,
                                       geometry.knots,
                                       0,
                                       dom_dim - 1);

        // science variable mfas
        for (auto i = 0; i< vars.size(); i++)
        {
            vars[i].mfa = new mfa::MFA<T>(vars[i].p,
                                          mfa->ndom_pts(),
                                          domain,
                                          vars[i].ctrl_pts,
                                          vars[i].nctrl_pts,
                                          vars[i].weights,
                                          vars[i].knots,
                                          dom_dim + i,        // assumes each variable is scalar
                                          dom_dim + i);
        }

        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // initialize all_knots
        tmesh.all_knots.resize(dom_dim);
        tmesh.all_knot_levels.resize(dom_dim);

        for (auto i = 0; i < dom_dim; i++)
        {
            tmesh.all_knots[i].resize(6);       // hard-coded to match diagram
            tmesh.all_knot_levels[i].resize(6);
            for (auto j = 0; j < tmesh.all_knots[i].size(); j++)
            {
                tmesh.all_knots[i][j] = j / static_cast<T>(tmesh.all_knots[i].size() - 1);
                tmesh.all_knot_levels[i][j] = 0;
            }
        }

        // initialize first tensor product
        vector<KnotIdx> knot_mins(dom_dim);
        vector<KnotIdx> knot_maxs(dom_dim);
        for (auto i = 0; i < dom_dim; i++)
        {
            knot_mins[i] = 0;
            knot_maxs[i] = 5;
        }

        tmesh.append_tensor(knot_mins, knot_maxs);
    }

    // refine the t-mesh the first time
    void refine1_tmesh(const diy::Master::ProxyWithLink&   cp)
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // insert new knots into all_knots
        tmesh.insert_knot(0, 2, 1, 0.3);
        tmesh.insert_knot(1, 3, 1, 0.5);

        // insert tensor product
        vector<KnotIdx> knot_mins(dom_dim);
        vector<KnotIdx> knot_maxs(dom_dim);
        assert(dom_dim == 2);           // testing 2d for now
        knot_mins[0] = 0;
        knot_mins[1] = 1;
        knot_maxs[0] = 4;
        knot_maxs[1] = 5;
        tmesh.append_tensor(knot_mins, knot_maxs);
    }

    // refine the t-mesh the second time
    void refine2_tmesh(const diy::Master::ProxyWithLink&   cp)
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // insert new knots into all_knots
        tmesh.insert_knot(0, 4, 2, 0.5);
        tmesh.insert_knot(1, 4, 2, 0.55);

        // insert tensor product
        vector<size_t> knot_mins(dom_dim);
        vector<size_t> knot_maxs(dom_dim);
        assert(dom_dim == 2);           // testing 2d for now
        knot_mins[0] = 2;
        knot_mins[1] = 2;
        knot_maxs[0] = 6;
        knot_maxs[1] = 6;
        tmesh.append_tensor(knot_mins, knot_maxs);
    }

    // print the t-mesh
    void print_tmesh(const diy::Master::ProxyWithLink&      cp)
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        tmesh.print();
    }

    // decode a point in the t-mesh
    void decode_tmesh(const diy::Master::ProxyWithLink&     cp,
                      const VectorX<T>&                     param)      // parameters of point to decode
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // compute range of anchor points for a given point to decode
        vector<vector<size_t>> anchors(dom_dim);                        // anchors affecting the decoding point
        tmesh.anchors(param, anchors);

        // print anchors
        fmt::print(stderr, "for decoding point = [ ");
        for (auto i = 0; i < dom_dim; i++)
            fmt::print(stderr, "{} ", param[i]);
        fmt::print(stderr, "],\n");

        for (auto i = 0; i < dom_dim; i++)
        {
            fmt::print(stderr, "dim {} local anchors = [", i);
            for (auto j = 0; j < anchors[i].size(); j++)
                fmt::print(stderr, "{} ", anchors[i][j]);
            fmt::print(stderr, "]\n");
        }
        fmt::print(stderr, "\n--------------------------\n\n");

        // compute local knot vectors for each anchor in Cartesian product of anchors

        int tot_nanchors = 1;                                               // total number of anchors in flattened space
        for (auto i = 0; i < dom_dim; i++)
            tot_nanchors *= anchors[i].size();
        vector<int> anchor_idx(dom_dim);                                    // current index of anchor in each dim, initialized to 0s

        for (auto j = 0; j < tot_nanchors; j++)
        {
            vector<size_t> anchor(dom_dim);                                 // one anchor from anchors
            for (auto i = 0; i < dom_dim; i++)
                anchor[i] = anchors[i][anchor_idx[i]];
            anchor_idx[0]++;

            // for all dimensions except last, check if anchor_idx is at the end
            for (auto k = 0; k < dom_dim - 1; k++)
            {
                if (anchor_idx[k] == anchors[k].size())
                {
                    anchor_idx[k] = 0;
                    anchor_idx[k + 1]++;
                }
            }

            vector<vector<size_t>> loc_knot_vec(dom_dim);                   // local knot vector
            tmesh.local_knot_vector(anchor, loc_knot_vec);

            // print local knot vectors
            fmt::print(stderr, "for anchor = [ ");
            for (auto i = 0; i < dom_dim; i++)
                fmt::print(stderr, "{} ", anchor[i]);
            fmt::print(stderr, "],\n");

            for (auto i = 0; i < dom_dim; i++)
            {
                fmt::print(stderr, "dim {} local knot vector = [", i);
                for (auto j = 0; j < loc_knot_vec[i].size(); j++)
                    fmt::print(stderr, "{} ", loc_knot_vec[i][j]);
                fmt::print(stderr, "]\n");
            }
            fmt::print(stderr, "\n--------------------------\n\n");

            // TODO: insert any missing knots and control points (port Youssef's knot insertion algorithm)
            // TODO: insert missing knots into tmesh so that knot lines will be intersected

            // TODO: compute basis function in each dimension

            // TODO: locate corresponding control point

            // TODO: multiply basis function by control point and add to current sum
        }

        // TODO: normalize sum of basis functions * control points
    }
};

namespace diy
{
    //         DEPRECATE: not necessary to specialize vector<vector>>; diy handles correctly
//         template <typename T>
//         struct Serialization< vector<vector<T>> >
//         {
//             static
//                 void save(diy::BinaryBuffer& bb, const vector<vector<T>>& v)
//                 {
//                     diy::save(bb, v.size());
//                     for (auto i = 0; i < v.size(); i++)
//                         diy::save(bb, v[i]);
//                 }
//             static
//                 void load(diy::BinaryBuffer& bb, vector<vector<T>>& v)
//                 {
//                     size_t size;
//                     diy::load(bb, size);
//                     v.resize(size);
//                     for (size_t i = 0; i < v.size(); i++)
//                             diy::load(bb, v[i]);
//                 }
//         };

        template <typename T>
        struct Serialization<MatrixX<T>>
        {
            static
                void save(diy::BinaryBuffer& bb, const MatrixX<T>& m)
                {
                    diy::save(bb, m.rows());
                    diy::save(bb, m.cols());
                    for (size_t i = 0; i < m.rows(); ++i)
                        for (size_t j = 0; j < m.cols(); ++j)
                            diy::save(bb, m(i, j));
                }
            static
                void load(diy::BinaryBuffer& bb, MatrixX<T>& m)
                {
                    Index rows, cols;
                    diy::load(bb, rows);
                    diy::load(bb, cols);
                    m.resize(rows, cols);
                    for (size_t i = 0; i < m.rows(); ++i)
                        for (size_t j = 0; j < m.cols(); ++j)
                            diy::load(bb, m(i, j));
                }
        };

        template <typename T>
        struct Serialization<VectorX<T>>
        {
            static
                void save(diy::BinaryBuffer& bb, const VectorX<T>& v)
                {
                    diy::save(bb, v.size());
                    for (size_t i = 0; i < v.size(); ++i)
                        diy::save(bb, v(i));
                }
            static
                void load(diy::BinaryBuffer& bb, VectorX<T>& v)
                {
                    Index size;
                    diy::load(bb, size);
                    v.resize(size);
                    for (size_t i = 0; i < size; ++i)
                        diy::load(bb, v(i));
                }
        };

        template<>
        struct Serialization<VectorXi>
        {
            static
                void save(diy::BinaryBuffer& bb, const VectorXi& v)
                {
                    diy::save(bb, v.size());
                    for (size_t i = 0; i < v.size(); ++i)
                        diy::save(bb, v(i));
                }
            static
                void load(diy::BinaryBuffer& bb, VectorXi& v)
                {
                    Index size;
                    diy::load(bb, size);
                    v.resize(size);
                    for (size_t i = 0; i < size; ++i)
                        diy::load(bb, v.data()[i]);
                 }
        };
}
