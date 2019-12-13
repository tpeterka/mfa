//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include    <mfa/mfa.hpp>
#include    <mfa/block_base.hpp>

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

using namespace std;

// set input and ouptut precision here, float or double
#if 0
typedef float                          real_t;
#else
typedef double                         real_t;
#endif

// 3d point or vector
struct vec3d
{
    float x, y, z;
    float mag() { return sqrt(x*x + y*y + z*z); }
};

// arguments to block foreach functions
struct DomainArgs : public ModelInfo
{
    DomainArgs(int dom_dim, int pt_dim) :
        ModelInfo(dom_dim, pt_dim)
    {
        starts.resize(dom_dim);
        full_dom_pts.resize(dom_dim);
        min.resize(dom_dim);
        max.resize(dom_dim);
        s.resize(pt_dim);
        f.resize(pt_dim);
    }
    vector<int>         starts;                     // starting offsets of ndom_pts (optional, usually assumed 0)
    vector<int>         full_dom_pts;               // number of points in full domain in case a subset is taken
    vector<real_t>      min;                        // minimum corner of domain
    vector<real_t>      max;                        // maximum corner of domain
    vector<real_t>      s;                          // scaling factor for each variable or any other usage
    real_t              r;                          // x-y rotation of domain or any other usage
    vector<real_t>      f;                          // frequency multiplier for each variable or any other usage
    real_t              t;                          // waviness of domain edges or any other usage
    real_t              n;                          // noise factor [0.0 - 1.0]
    char                infile[256];                // input filename
    bool                multiblock;                 // multiblock domain, get bounds from block
};

// block
template <typename T>
struct Block : public BlockBase<T>
{
    static
        void* create()              { return mfa::create<Block>(); }

    static
        void destroy(void* b)       { mfa::destroy<Block>(b); }

    static
        void add(                                   // add the block to the decomposition
            int                 gid,                // block global id
            const Bounds<T>&    core,               // block bounds without any ghost added
            const Bounds<T>&    bounds,             // block bounds including any ghost region added
            const Bounds<T>&    domain,             // global data bounds
            const RCLink<T>&    link,               // neighborhood
            diy::Master&        master,             // diy master
            int                 dom_dim,            // domain dimensionality
            int                 pt_dim,             // point dimensionality
            T                   ghost_factor = 0.0) // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    {
        mfa::add<Block, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);
    }

    static
        void save(const void* b_, diy::BinaryBuffer& bb)    { mfa::save<Block, T>(b_, bb); }
    static
        void load(void* b_, diy::BinaryBuffer& bb)          { mfa::load<Block, T>(b_, bb); }

    // NB: Because BlockBase, the parent of Block, is templated, the C++ compiler requires
    // access to members in BlockBase to be preceded by "this->".
    // Otherwise, the compiler can't be sure that the member exists. [Myers Effective C++, item 43]
    // This is annoying but unavoidable.

    // evaluate sine function
    T sine(VectorX<T>&  domain_pt,
           DomainArgs&  args,
           int          k)                  // current science variable
    {
        DomainArgs* a = &args;
        T retval = 1.0;
        for (auto i = 0; i < this->dom_dim; i++)
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
        for (auto i = 0; i < this->dom_dim; i++)
        {
            if (domain_pt(i) != 0.0)
                retval *= (sin(domain_pt(i) * a->f[k] ) / domain_pt(i));
        }
        retval *= a->s[k];

        return retval;
    }

    // evaluate Marschner-Lobb function [Marschner and Lobb, IEEE VIS, 1994]
    // only for a 3d domain
    // using args f[0] and s[0] for f_M and alpha, respectively, in the paper
    T ml(VectorX<T>&  domain_pt,
           DomainArgs&  args)
    {
        DomainArgs* a   = &args;
        T& fm           = a->f[0];
        T& alpha        = a->s[0];
//         T fm = 6.0;
//         T alpha = 0.25;
        T& x            = domain_pt(0);
        T& y            = domain_pt(1);
        T& z            = domain_pt(2);

        T rad       = sqrt(x * x + y * y + z * z);
        T rho       = cos(2 * M_PI * fm * cos(M_PI * rad / 2.0));
        T retval    = (1.0 - sin(M_PI * z / 2.0) + alpha * (1.0 + rho * sqrt(x * x + y * y))) / (2 * (1.0 + alpha));

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
        int nvars       = this->pt_dim - this->dom_dim;             // number of science variables
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        int tot_ndom_pts    = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        for (int j = 0; j < nvars; j++)
        {
            this->vars[j].min_dim = this->dom_dim + j;
            this->vars[j].max_dim = this->vars[j].min_dim + 1;
        }
        VectorXi ndom_pts(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
            ndom_pts(i) = a->ndom_pts[i];

        // get local block bounds
        // if single block, they are passed in args
        // if multiblock, they were decomposed by diy and are already in the block's bounds_mins, maxs
        if (!a->multiblock)
        {
            this->bounds_mins.resize(this->pt_dim);
            this->bounds_maxs.resize(this->pt_dim);
            this->core_mins.resize(this->dom_dim);
            this->core_maxs.resize(this->dom_dim);
            for (int i = 0; i < this->dom_dim; i++)
            {
                this->bounds_mins(i)  = a->min[i];
                this->bounds_maxs(i)  = a->max[i];
                this->core_mins(i)    = a->min[i];
                this->core_maxs(i)    = a->max[i];
            }
        }

        // adjust number of domain points and starting domain point for ghost
        VectorX<T> d(this->dom_dim);               // step in domain points in each dimension
        VectorX<T> p0(this->dom_dim);              // starting point in each dimension
        int nghost_pts;                         // number of ghost points in current dimension
        for (int i = 0; i < this->dom_dim; i++)
        {
            d(i) = (this->core_maxs(i) - this->core_mins(i)) / (ndom_pts(i) - 1);
            // min direction
            nghost_pts = floor((this->core_mins(i) - this->bounds_mins(i)) / d(i));
            ndom_pts(i) += nghost_pts;
            p0(i) = this->core_mins(i) - nghost_pts * d(i);
            // max direction
            nghost_pts = floor((this->bounds_maxs(i) - this->core_maxs(i)) / d(i));
            ndom_pts(i) += nghost_pts;
            tot_ndom_pts *= ndom_pts(i);
        }

        this->domain.resize(tot_ndom_pts, this->pt_dim);

        // assign values to the domain (geometry)
        vector<int> dom_idx(this->dom_dim);                   // current index of domain point in each dim, initialized to 0s
        for (auto j = 0; j < tot_ndom_pts; j++)         // flattened loop over all the points in a domain in dimension dom_dim
        {
            // compute geometry coordinates of domain point
            for (auto i = 0; i < this->dom_dim; i++)
                this->domain(j, i) = p0(i) + dom_idx[i] * d(i);

            dom_idx[0]++;

            // for all dimensions except last, check for end of the line, part of flattened loop logic
            for (auto k = 0; k < this->dom_dim - 1; k++)
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
        VectorX<T> dom_pt(this->dom_dim);
        for (int j = 0; j < tot_ndom_pts; j++)
        {
            dom_pt = this->domain.block(j, 0, 1, this->dom_dim).transpose();
            T retval;
            for (auto k = 0; k < nvars; k++)        // for all science variables
            {
                if (fun == "sine")
                    retval = sine(dom_pt, args, k);
                if (fun == "sinc")
                    retval = sinc(dom_pt, args, k);
                if (fun == "ml")
                {
                    if (this->dom_dim != 3)
                    {
                        fprintf(stderr, "Error: Marschner-Lobb function is only defined for a 3d domain.\n");
                        exit(0);
                    }
                    retval = ml(dom_pt, args);
                }
                if (fun == "f16")
                    retval = f16(dom_pt);
                if (fun == "f17")
                    retval = f17(dom_pt);
                if (fun == "f18")
                    retval = f18(dom_pt);
                this->domain(j, this->dom_dim + k) = retval;
            }

            // add some noise
            double noise = distribution(generator);
            this->domain(j, this->dom_dim) *= (1.0 + a->n * noise);

            if (j == 0 || this->domain(j, this->dom_dim) > this->bounds_maxs(this->dom_dim))
                this->bounds_maxs(this->dom_dim) = this->domain(j, this->dom_dim);
            if (j == 0 || this->domain(j, this->dom_dim) < this->bounds_mins(this->dom_dim))
                this->bounds_mins(this->dom_dim) = this->domain(j, this->dom_dim);
        }

        // optional wavy domain
        if (a->t && this->pt_dim >= 3)
        {
            for (auto j = 0; j < tot_ndom_pts; j++)
            {
                real_t x = this->domain(j, 0);
                real_t y = this->domain(j, 1);
                this->domain(j, 0) += a->t * sin(y);
                this->domain(j, 1) += a->t * sin(x);
                if (j == 0 || this->domain(j, 0) < this->bounds_mins(0))
                    this->bounds_mins(0) = this->domain(j, 0);
                if (j == 0 || this->domain(j, 1) < this->bounds_mins(1))
                    this->bounds_mins(1) = this->domain(j, 1);
                if (j == 0 || this->domain(j, 0) > this->bounds_maxs(0))
                    this->bounds_maxs(0) = this->domain(j, 0);
                if (j == 0 || this->domain(j, 1) > this->bounds_maxs(1))
                    this->bounds_maxs(1) = this->domain(j, 1);
            }
        }

        // optional rotation of the domain
        if (a->r && this->pt_dim >= 3)
        {
            for (auto j = 0; j < tot_ndom_pts; j++)
            {
                real_t x = this->domain(j, 0);
                real_t y = this->domain(j, 1);
                this->domain(j, 0) = x * cos(a->r) - y * sin(a->r);
                this->domain(j, 1) = x * sin(a->r) + y * cos(a->r);
                if (j == 0 || this->domain(j, 0) < this->bounds_mins(0))
                    this->bounds_mins(0) = this->domain(j, 0);
                if (j == 0 || this->domain(j, 1) < this->bounds_mins(1))
                    this->bounds_mins(1) = this->domain(j, 1);
                if (j == 0 || this->domain(j, 0) > this->bounds_maxs(0))
                    this->bounds_maxs(0) = this->domain(j, 0);
                if (j == 0 || this->domain(j, 1) > this->bounds_maxs(1))
                    this->bounds_maxs(1) = this->domain(j, 1);
            }
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // extents
        fprintf(stderr, "gid = %d\n", cp.gid());
        cerr << "core_mins:\n" << this->core_mins << endl;
        cerr << "core_maxs:\n" << this->core_maxs << endl;
        cerr << "bounds_mins:\n" << this->bounds_mins << endl;
        cerr << "bounds_maxs:\n" << this->bounds_maxs << endl;

//         cerr << "ndom_pts:\n" << ndom_pts << "\n" << endl;
//         cerr << "domain:\n" << this->domain << endl;
    }

    // read a floating point 3d vector dataset and take one 1-d curve out of the middle of it
    // f = (x, velocity magnitude)
    void read_1d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        int nvars = 1;
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        this->vars[0].min_dim = this->dom_dim;
        this->vars[0].max_dim = this->vars[0].min_dim + 1;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        this->domain.resize(tot_ndom_pts, this->pt_dim);
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
            this->domain(i, 1) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
            //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)this->domain.rows(); i++)
        {
            if (i == 0 || this->domain(i, 1) < this->bounds_mins(1))
                this->bounds_mins(1) = this->domain(i, 1);
            if (i == 0 || this->domain(i, 1) > this->bounds_maxs(1))
                this->bounds_maxs(1) = this->domain(i, 1);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
        {
            this->domain(n, 0) = i;
            n++;
        }

        // extents
        this->bounds_mins(0) = 0.0;
        this->bounds_maxs(0) = this->domain(tot_ndom_pts - 1, 0);
        this->core_mins.resize(this->dom_dim);
        this->core_maxs.resize(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            this->core_mins(i) = this->bounds_mins(i);
            this->core_maxs(i) = this->bounds_maxs(i);
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // debug
        cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2-d surface out of the middle of it
    // f = (x, y, velocity magnitude)
    void read_2d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        int nvars = 1;
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        this->vars[0].min_dim = this->dom_dim;
        this->vars[0].max_dim = this->vars[0].min_dim + 1;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        this->domain.resize(tot_ndom_pts, this->pt_dim);
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
            this->domain(i, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
            //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)this->domain.rows(); i++)
        {
            if (i == 0 || this->domain(i, 2) < this->bounds_mins(2))
                this->bounds_mins(2) = this->domain(i, 2);
            if (i == 0 || this->domain(i, 2) > this->bounds_maxs(2))
                this->bounds_maxs(2) = this->domain(i, 2);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
            for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
            {
                this->domain(n, 0) = i;
                this->domain(n, 1) = j;
                n++;
            }

        // extents
        this->bounds_mins(0) = 0.0;
        this->bounds_mins(1) = 0.0;
        this->bounds_maxs(0) = this->domain(tot_ndom_pts - 1, 0);
        this->bounds_maxs(1) = this->domain(tot_ndom_pts - 1, 1);
        this->core_mins.resize(this->dom_dim);
        this->core_maxs.resize(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            this->core_mins(i) = this->bounds_mins(i);
            this->core_maxs(i) = this->bounds_maxs(i);
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // debug
        cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2d (parallel to x-y plane) subset
    // f = (x, y, velocity magnitude)
    void read_2d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        int nvars = 1;
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        this->vars[0].min_dim = this->dom_dim;
        this->vars[0].max_dim = this->vars[0].min_dim + 1;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        this->domain.resize(tot_ndom_pts, this->pt_dim);
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
                this->domain(n, 0) = ijk[0];                  // domain is just i,j
                this->domain(n, 1) = ijk[1];
                // range (function value) is magnitude of velocity
                this->domain(n, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
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
        for (size_t i = 0; i < (size_t)this->domain.rows(); i++)
        {
            if (i == 0 || this->domain(i, 2) < this->bounds_mins(2))
                this->bounds_mins(2) = this->domain(i, 2);
            if (i == 0 || this->domain(i, 2) > this->bounds_maxs(2))
                this->bounds_maxs(2) = this->domain(i, 2);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        this->bounds_mins(0) = this->domain(0, 0);
        this->bounds_mins(1) = this->domain(0, 1);
        this->bounds_maxs(0) = this->domain(tot_ndom_pts - 1, 0);
        this->bounds_maxs(1) = this->domain(tot_ndom_pts - 1, 1);
        this->core_mins.resize(this->dom_dim);
        this->core_maxs.resize(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            this->core_mins(i) = this->bounds_mins(i);
            this->core_maxs(i) = this->bounds_maxs(i);
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // debug
        cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset
    // f = (x, y, z, velocity magnitude)
    void read_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        int nvars = 1;
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        this->vars[0].min_dim = this->dom_dim;
        this->vars[0].max_dim = this->vars[0].min_dim + 1;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        this->domain.resize(tot_ndom_pts, this->pt_dim);

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
            this->domain(i, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
            //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
        }

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)this->domain.rows(); i++)
        {
            if (i == 0 || this->domain(i, 3) < this->bounds_mins(3))
                this->bounds_mins(3) = this->domain(i, 3);
            if (i == 0 || this->domain(i, 3) > this->bounds_maxs(3))
                this->bounds_maxs(3) = this->domain(i, 3);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                {
                    this->domain(n, 0) = i;
                    this->domain(n, 1) = j;
                    this->domain(n, 2) = k;
                    n++;
                }

        // extents
        this->bounds_mins(0) = 0.0;
        this->bounds_mins(1) = 0.0;
        this->bounds_mins(2) = 0.0;
        this->bounds_maxs(0) = this->domain(tot_ndom_pts - 1, 0);
        this->bounds_maxs(1) = this->domain(tot_ndom_pts - 1, 1);
        this->bounds_maxs(2) = this->domain(tot_ndom_pts - 1, 2);
        this->core_mins.resize(this->dom_dim);
        this->core_maxs.resize(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            this->core_mins(i) = this->bounds_mins(i);
            this->core_maxs(i) = this->bounds_maxs(i);
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // debug
        cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take a 3d subset out of it
    // f = (x, y, z, velocity magnitude)
    void read_3d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        int nvars = 1;
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        this->vars[0].min_dim = this->dom_dim;
        this->vars[0].max_dim = this->vars[0].min_dim + 1;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        this->domain.resize(tot_ndom_pts, this->pt_dim);
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
                this->domain(n, 0) = ijk[0];                  // domain is just i,j
                this->domain(n, 1) = ijk[1];
                this->domain(n, 2) = ijk[2];
                this->domain(n, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
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
        for (size_t i = 0; i < (size_t)this->domain.rows(); i++)
        {
            if (i == 0 || this->domain(i, 3) < this->bounds_mins(3))
                this->bounds_mins(3) = this->domain(i, 3);
            if (i == 0 || this->domain(i, 3) > this->bounds_maxs(3))
                this->bounds_maxs(3) = this->domain(i, 3);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        this->bounds_mins(0) = this->domain(0, 0);
        this->bounds_mins(1) = this->domain(0, 1);
        this->bounds_mins(2) = this->domain(0, 2);
        this->bounds_maxs(0) = this->domain(tot_ndom_pts - 1, 0);
        this->bounds_maxs(1) = this->domain(tot_ndom_pts - 1, 1);
        this->bounds_maxs(2) = this->domain(tot_ndom_pts - 1, 2);
        this->core_mins.resize(this->dom_dim);
        this->core_maxs.resize(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            this->core_mins(i) = this->bounds_mins(i);
            this->core_maxs(i) = this->bounds_maxs(i);
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // debug
        cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

    // read a floating point 2d scalar dataset
    // f = (x, y, value)
    void read_2d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        int nvars = 1;
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        this->vars[0].min_dim = this->dom_dim;
        this->vars[0].max_dim = this->vars[0].min_dim + 1;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        this->domain.resize(tot_ndom_pts, this->pt_dim);

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
            this->domain(i, 2) = val[i];

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)this->domain.rows(); i++)
        {
            if (i == 0 || this->domain(i, 2) < this->bounds_mins(2))
                this->bounds_mins(2) = this->domain(i, 2);
            if (i == 0 || this->domain(i, 2) > this->bounds_maxs(2))
                this->bounds_maxs(2) = this->domain(i, 2);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
            for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
            {
                this->domain(n, 0) = i;
                this->domain(n, 1) = j;
                n++;
            }

        // extents
        this->bounds_mins(0) = 0.0;
        this->bounds_mins(1) = 0.0;
        this->bounds_maxs(0) = this->domain(tot_ndom_pts - 1, 0);
        this->bounds_maxs(1) = this->domain(tot_ndom_pts - 1, 1);
        this->core_mins.resize(this->dom_dim);
        this->core_maxs.resize(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            this->core_mins(i) = this->bounds_mins(i);
            this->core_maxs(i) = this->bounds_maxs(i);
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // debug
        cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

    // read a floating point 3d scalar dataset
    // f = (x, y, z, value)
    void read_3d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->geometry.min_dim = 0;
        this->geometry.max_dim = this->dom_dim - 1;
        int nvars = 1;
        this->vars.resize(nvars);
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        this->vars[0].min_dim = this->dom_dim;
        this->vars[0].max_dim = this->vars[0].min_dim + 1;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        this->domain.resize(tot_ndom_pts, this->pt_dim);

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
            this->domain(i, 3) = val[i];

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)this->domain.rows(); i++)
        {
            if (i == 0 || this->domain(i, 3) < this->bounds_mins(3))
                this->bounds_mins(3) = this->domain(i, 3);
            if (i == 0 || this->domain(i, 3) > this->bounds_maxs(3))
                this->bounds_maxs(3) = this->domain(i, 3);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                {
                    this->domain(n, 0) = i;
                    this->domain(n, 1) = j;
                    this->domain(n, 2) = k;
                    n++;
                }

        // extents
        this->bounds_mins(0) = 0.0;
        this->bounds_mins(1) = 0.0;
        this->bounds_mins(2) = 0.0;
        this->bounds_maxs(0) = this->domain(tot_ndom_pts - 1, 0);
        this->bounds_maxs(1) = this->domain(tot_ndom_pts - 1, 1);
        this->bounds_maxs(2) = this->domain(tot_ndom_pts - 1, 2);
        this->core_mins.resize(this->dom_dim);
        this->core_maxs.resize(this->dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            this->core_mins(i) = this->bounds_mins(i);
            this->core_maxs(i) = this->bounds_maxs(i);
        }

        this->mfa = new mfa::MFA<T>(this->dom_dim, ndom_pts, this->domain);

        // debug
        cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

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

        if (!this->dom_dim)
            this->dom_dim = this->mfa->ndom_pts().size();

        size_t tot_ndom_pts = 1;
        for (auto i = 0; i < this->dom_dim; i++)
            tot_ndom_pts *= a->ndom_pts[i];

        vector<int> dom_idx(this->dom_dim);                           // current index of domain point in each dim, initialized to 0s

        // steps in each dimension in paramater space and real space
        vector<T> dom_step_real(this->dom_dim);                       // spacing between domain points in real space
        vector<T> dom_step_param(this->dom_dim);                      // spacing between domain points in parameter space
        for (auto i = 0; i < this->dom_dim; i++)
        {
            dom_step_param[i] = 1.0 / (double)(a->ndom_pts[i] - 1);
            dom_step_real[i] = dom_step_param[i] * (a->max[i] - a->min[i]);
        }

        // flattened loop over all the points in a domain in dimension dom_dim
        fmt::print(stderr, "Testing analytical error norms over a total of {} points\n", tot_ndom_pts);
        for (auto j = 0; j < tot_ndom_pts; j++)
        {
            // compute current point in real and parameter space
            VectorX<T> dom_pt_real(this->dom_dim);                // one domain point in real space
            VectorX<T> dom_pt_param(this->dom_dim);               // one domain point in parameter space
            for (auto i = 0; i < this->dom_dim; i++)
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
            this->mfa->DecodePt(*(this->vars[0].mfa_data), dom_pt_param, cpt);       // hard-coded for one science variable
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
                if (this->dom_dim > 2)        // 3d or greater domain
                {
                    true_pt.y = test_pt.y = dom_pt_real(1);
                    true_pt.z = test_pt.z = dom_pt_real(2);
                }
                else if (this->dom_dim > 1)   // 2d domain
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
            for (auto k = 0; k < this->dom_dim - 1; k++)
            {
                if (dom_idx[k] == a->ndom_pts[k])
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
};
