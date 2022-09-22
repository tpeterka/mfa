//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_BLOCK
#define _MFA_BLOCK

#include    <mfa/mfa.hpp>
#include    <mfa/block_base.hpp>

#include    <diy/master.hpp>
#include    <diy/reduce-operations.hpp>
#include    <diy/decomposition.hpp>
#include    <diy/assigner.hpp>
#include    <diy/io/block.hpp>
#include    <diy/io/bov.hpp>
#include    <diy/pick.hpp>

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
    vec3d(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    vec3d() {}
};

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

// block
template <typename T>
struct Block : public BlockBase<T>
{
    using Base = BlockBase<T>;
    using Base::dom_dim;
    using Base::pt_dim;
    using Base::core_mins;
    using Base::core_maxs;
    using Base::bounds_mins;
    using Base::bounds_maxs;
    using Base::overlaps;
    using Base::input;
    using Base::approx;
    using Base::errs;
    using Base::mfa;

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

    void generate_analytical_data(
            const diy::Master::ProxyWithLink&   cp,
            string&                             fun,
            MFAInfo&                            mfa_info,
            DomainArgs&                         args)
    {
        if (args.rand_seed >= 0)  // random point cloud
        {
            if (cp.gid() == 0)
                cout << "Generating data on random point cloud for function: " << fun << endl;

            if (args.structured)
            {
                cerr << "ERROR: Cannot perform structured encoding of random point cloud" << endl;
                exit(1);
            }

            // Used by generate_random_analytical_data
            args.tot_ndom_pts = 1;
            for (size_t k = 0; k < dom_dim; k++)
            {
                args.tot_ndom_pts *= args.ndom_pts[k];
            }

            // create unsigned conversion of seed
            // note: seed is always >= 0 in this code block
            unsigned useed = (unsigned)args.rand_seed;
            generate_random_analytical_data(cp, fun, mfa_info, args, useed);
        }
        else    // structured grid of points
        {
            if (cp.gid() == 0)
                cout << "Generating data on structured grid for function: " << fun << endl;
            
            generate_rectilinear_analytical_data(cp, fun, mfa_info, args);
        }
    }

    // synthetic analytic (scalar) data, sampled on unstructured point cloud
    // when seed = 0, we choose a time-dependent seed for the random number generator
    void generate_random_analytical_data(
            const diy::Master::ProxyWithLink&   cp,
            string&                             fun,
            MFAInfo&                            mfa_info,
            DomainArgs&                         args,
            unsigned int                        seed)
    {
        assert(!args.structured);

        DomainArgs* a = &args;

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        // Prepare containers
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);

        // Set block bounds (if not already done by DIY)
        if (!a->multiblock)
        {
            bounds_mins.resize(pt_dim);
            bounds_maxs.resize(pt_dim);
            core_mins.resize(gdim);
            core_maxs.resize(gdim);
            for (int i = 0; i < gdim; i++)
            {
                bounds_mins(i)  = a->min[i];
                bounds_maxs(i)  = a->max[i];
                core_mins(i)    = a->min[i];
                core_maxs(i)    = a->max[i];
            }
        }

        // decide overlap in each direction; they should be symmetric for neighbors
        // so if block a overlaps block b, block b overlaps a the same area
        for (size_t k = 0; k < gdim; k++)
        {
            overlaps(k) = fabs(core_mins(k) - bounds_mins(k));
            T m2 = fabs(bounds_maxs(k) - core_maxs(k));
            if (m2 > overlaps(k))
                overlaps(k) = m2;
        }


        // Create input data set and add to block
        input = new mfa::PointSet<T>(dom_dim, mdims, a->tot_ndom_pts);

        // Choose a system-dependent seed if seed==0
        if (seed == 0)
            seed = chrono::system_clock::now().time_since_epoch().count();

        std::default_random_engine df_gen(seed);
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);

        // Fill domain with randomly distributed points
        if (cp.gid() == 0)
            cout << "Void Sparsity: " << (1 - a->t)*100 << "%" << endl;

        double keep_frac = 1 - a->t;
        if (keep_frac < 0 || keep_frac > 1)
        {
            cerr << "Invalid value of void density" << endl;
            exit(1);
        }
        size_t nvoids = 4;
        double radii_frac = 1.0/8.0;   // fraction of domain width to set as void radius
        VectorX<T> radii(nvoids);
        MatrixX<T> centers(gdim, nvoids);
        for (size_t nv = 0; nv < nvoids; nv++) // Randomly generate the centers of each void
        {
            for (size_t k = 0; k < gdim; k++)
            {
                centers(k,nv) = core_mins(k) + u_dist(df_gen) * (core_maxs(k) - core_mins(k));
            }

            radii(nv) = radii_frac * (core_maxs - core_mins).minCoeff();
        }

        VectorX<T> dom_pt(gdim);
        for (size_t j = 0; j < input->domain.rows(); j++)
        {
            VectorX<T> candidate_pt(gdim);

            bool keep = true;
            do
            {
                // Generate a random point
                for (size_t k = 0; k < gdim; k++)
                {
                    candidate_pt(k) = core_mins(k) + u_dist(df_gen) * (core_maxs(k) - core_mins(k));
                }

                // Consider discarding point if within a certain radius of a void
                for (size_t nv = 0; nv < nvoids; nv++)
                {
                    if ((candidate_pt - centers.col(nv)).norm() < radii(nv))
                    {
                        keep = false;
                        break;
                    }
                }

                // Keep this point anyway a certain fraction of the time
                if (keep == false)
                {
                    if (u_dist(df_gen) <= keep_frac)
                        keep = true;
                }
            } while (!keep);

            // Add point to Input
            for (size_t k = 0; k < gdim; k++)
            {
                input->domain(j,k) = candidate_pt(k);
            }

            input->geom_coords(j, dom_pt);          // fill dom_pt
            for (auto k = 0; k < nvars; k++)        // for all science variables
            {
                int dmin = input->var_min(k);
                int vardim = input->var_dim(k);
                VectorX<T> out_pt(input->var_dim(k));

                evaluate_function(fun, dom_pt, out_pt, args, k);
                input->domain.row(j).segment(dmin, vardim) = out_pt;
            }  
        }

        bounds_mins = input->domain.colwise().minCoeff();
        bounds_maxs = input->domain.colwise().maxCoeff();

        input->set_domain_params(core_mins, core_maxs);     // Set explicit bounding box for parameter space

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // extents
        cerr << "gid = " << cp.gid() << endl;
        cerr << "core_mins:\n" << core_mins.transpose() << endl;
        cerr << "core_maxs:\n" << core_maxs.transpose() << endl;
        cerr << "bounds_mins:\n" << bounds_mins.transpose() << endl;
        cerr << "bounds_maxs:\n" << bounds_maxs.transpose() << endl;
    }

    // Creates a synthetic dataset on a rectilinear grid of points
    // This grid can be treated as EITHER a "structured" or "unstructured"
    // PointSet by setting the args.structured field appropriately
    void generate_rectilinear_analytical_data(
            const diy::Master::ProxyWithLink&   cp,
            string&                             fun,        // function to evaluate
            MFAInfo&                            mfa_info,
            DomainArgs&                         args)
    {
        DomainArgs* a   = &args;

        // TODO: This assumes that dom_dim = dimension of ambient geometry.
        //       Not always true, can model a 2d surface in 3d, e.g.
        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        for (int i = 0; i < dom_dim; i++)
            ndom_pts(i) = a->ndom_pts[i];

        // get local block bounds
        // if single block, they are passed in args
        // if multiblock, they were decomposed by diy and are already in the block's bounds_mins, maxs
        // TODO: Need a strategy to partition domain when dom_dim != geom_dim
        //       For instance, want to be able to partition a 2D manifold in 3D space with a 2D DIY 
        //       decomposition.
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
            for (int i = dom_dim; i < pt_dim; i++)
            {
                bounds_mins(i) = numeric_limits<T>::min();
                bounds_maxs(i) = numeric_limits<T>::max();
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
            // tot_ndom_pts *= ndom_pts(i);

            // decide overlap in each direction; they should be symmetric for neighbors
            // so if block a overlaps block b, block b overlaps a the same area
            this->overlaps(i) = fabs(core_mins(i) - bounds_mins(i));
            T m2 = fabs(bounds_maxs(i) - core_maxs(i));
            if (m2 > overlaps(i))
                overlaps(i) = m2;
        }

        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, ndom_pts.prod(), ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, ndom_pts.prod());

        // assign values to the domain (geometry)
        mfa::VolIterator vol_it(ndom_pts);
        // current index of domain point in each dim, initialized to 0s
        // flattened loop over all the points in a domain
        while (!vol_it.done())
        {
            int j = (int)vol_it.cur_iter();
            // compute geometry coordinates of domain point
            for (auto i = 0; i < dom_dim; i++)
                input->domain(j, i) = p0(i) + vol_it.idx_dim(i) * d(i);

            vol_it.incr_iter();
        }

        // normal distribution for generating noise
        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 1.0);

        // assign values to the range (science variables)
        VectorX<T> dom_pt(dom_dim);
        for (int j = 0; j < input->domain.rows(); j++)
        {
            input->geom_coords(j, dom_pt);          // fill dom_pt
            for (auto k = 0; k < nvars; k++)        // for all science variables
            {
                int dmin = input->var_min(k);
                int vardim = input->var_dim(k);
                VectorX<T> out_pt(input->var_dim(k));

                evaluate_function(fun, dom_pt, out_pt, args, k);    // fill out_pt
                input->domain.row(j).segment(dmin, vardim) = out_pt;
            }  

            // add some noise (optional)
            if (a->n != 0)
            {
                for (int k = gdim; k < input->pt_dim; k++)
                {
                    double noise = distribution(generator);
                    input->domain(j, k) *= (1.0 + a->n * noise);
                }
            }
        }

        // optional wavy domain
        if (a->t && gdim >= 2)
        {
            for (auto j = 0; j < input->domain.rows(); j++)
            {
                real_t x = input->domain(j, 0);
                real_t y = input->domain(j, 1);
                input->domain(j, 0) += a->t * sin(y);
                input->domain(j, 1) += a->t * sin(x);
            }
        }

        // optional rotation of the domain
        if (a->r && gdim >= 2)
        {
            for (auto j = 0; j < input->domain.rows(); j++)
            {
                real_t x = input->domain(j, 0);
                real_t y = input->domain(j, 1);
                input->domain(j, 0) = x * cos(a->r) - y * sin(a->r);
                input->domain(j, 1) = x * sin(a->r) + y * cos(a->r);
            }
        }

        this->bounds_maxs = input->domain.colwise().maxCoeff();
        this->bounds_mins = input->domain.colwise().minCoeff();

        // map_dir is used in blending discrete, but because we need to aggregate the discrete logic, we have to use
        // it even for continuous bounds, so in analytical data
        // this is used in s3d data because the actual domain dim is derived
        for (int k = 0; k < dom_dim; k++)
            this->map_dir.push_back(k);

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // extents
        cerr << "gid = " << cp.gid() << endl;
        cerr << "core_mins: " << core_mins.transpose() << endl;
        cerr << "core_maxs: " << core_maxs.transpose() << endl;
        cerr << "bounds_mins: " << bounds_mins.transpose() << endl;
        cerr << "bounds_maxs: " << bounds_maxs.transpose() << endl;
    }

    // read a floating point 3d vector dataset and take one 1-d curve out of the middle of it
    // f = (x, velocity magnitude)
    void read_1d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 1);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 1);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);

        VectorXi ndom_pts(dom_dim);
        this->bounds_mins.resize(pt_dim);
        this->bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        vector<float> vel(3 * tot_ndom_pts);

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        // rest is hard-coded for 1d

        // open file and seek to a slice in the center
        FILE *fd = fopen(a->infile.c_str(), "r");
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
            input->domain(i, 1) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            input->domain(i, 1) *= a->s[0];
            // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
            //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 1) < bounds_mins(1))
                bounds_mins(1) = input->domain(i, 1);
            if (i == 0 || input->domain(i, 1) > bounds_maxs(1))
                bounds_maxs(1) = input->domain(i, 1);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
        {
            input->domain(n, 0) = i;
            n++;
        }

        // extents
        bounds_mins(0) = 0.0;
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();
        
        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2-d surface out of the middle of it
    // f = (x, y, velocity magnitude)
    void read_2d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 2);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 2);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        this->bounds_mins.resize(pt_dim);
        this->bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        vector<float> vel(3 * tot_ndom_pts);

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        // rest is hard-coded for 2d

        // open file and seek to a slice in the center
        FILE *fd = fopen(a->infile.c_str(), "r");
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
            input->domain(i, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            input->domain(i, 2) *= a->s[0];
//              fprintf(stderr, "vel [%.3f %.3f %.3f]\n",
//                      vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]);
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 2) < bounds_mins(2))
                bounds_mins(2) = input->domain(i, 2);
            if (i == 0 || input->domain(i, 2) > bounds_maxs(2))
                bounds_maxs(2) = input->domain(i, 2);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
            for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
            {
                input->domain(n, 0) = i;
                input->domain(n, 1) = j;
                n++;
            }

        // extents
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = input->domain(tot_ndom_pts - 1, 1);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take one 2d (parallel to x-y plane) subset
    // f = (x, y, velocity magnitude)
    void read_2d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 2);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 2);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }
        vector<float> vel(a->full_dom_pts[0] * a->full_dom_pts[1] * 3);

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        FILE *fd = fopen(a->infile.c_str(), "r");
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
                input->domain(n, 0) = ijk[0];                  // domain is just i,j
                input->domain(n, 1) = ijk[1];
                // range (function value) is magnitude of velocity
                input->domain(n, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
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
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 2) < bounds_mins(2))
                bounds_mins(2) = input->domain(i, 2);
            if (i == 0 || input->domain(i, 2) > bounds_maxs(2))
                bounds_maxs(2) = input->domain(i, 2);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        bounds_mins(0) = input->domain(0, 0);
        bounds_mins(1) = input->domain(0, 1);
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = input->domain(tot_ndom_pts - 1, 1);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset
    // f = (x, y, z, velocity magnitude)
    void read_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 3);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 3);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        this->bounds_mins.resize(pt_dim);
        this->bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        vector<float> vel(3 * tot_ndom_pts);

        FILE *fd = fopen(a->infile.c_str(), "r");
        assert(fd);

        // read all three components of velocity and compute magnitude
        if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            input->domain(i, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                    vel[3 * i + 1] * vel[3 * i + 1] +
                    vel[3 * i + 2] * vel[3 * i + 2]);
            input->domain(i, 3) *= a->s[0];
//             if (i < 1000)
//              fprintf(stderr, "vel [%.3f %.3f %.3f]\n",
//                      vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]);
        }

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 3) < bounds_mins(3))
                bounds_mins(3) = input->domain(i, 3);
            if (i == 0 || input->domain(i, 3) > bounds_maxs(3))
                bounds_maxs(3) = input->domain(i, 3);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                {
                    input->domain(n, 0) = i;
                    input->domain(n, 1) = j;
                    input->domain(n, 2) = k;
                    n++;
                }

        // extents
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_mins(2) = 0.0;
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = input->domain(tot_ndom_pts - 1, 1);
        bounds_maxs(2) = input->domain(tot_ndom_pts - 1, 2);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector time-varying dataset, ie, 4d
    // f = (x, y, z, t, velocity magnitude)
    void read_4d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 4);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 4);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        size_t tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        this->bounds_mins.resize(pt_dim);
        this->bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }
        int num_skip = a->r;

        // construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        size_t space_ndom_pts = tot_ndom_pts / ndom_pts(3);    // total number of domain points in one time step
        vector<float> vel(3 * space_ndom_pts);

        ifstream fd;
        fd.open(a->infile);
        if (!fd.is_open())
        {
            fmt::print(stderr, "Error: read_4d_vector_data(): Unable to open file {}\n", a->infile);
            abort();
        }

        string line;

        for (auto j = 0; j < ndom_pts(3); j++)           // for all files
        {
            getline(fd, line);

            // debug
            fmt::print(stderr, "read_4d_vector_data(): opening file {}\n", line);

            ifstream cur_fd;
            cur_fd.open(line);
            if (!cur_fd.is_open())
            {
                fmt::print(stderr, "Error: read_4d_vector_data(): Unable to open file {}\n", line);
                abort();
            }

            size_t ofst = j * space_ndom_pts;           // starting offset in input->domain for this time step

            // read all three components of velocity and compute magnitude
            cur_fd.read((char*)(&vel[0]), space_ndom_pts * 3 * sizeof(float));
            if (!cur_fd)
            {
                fmt::print(stderr, "read_4d_vector_data(): unable to read file {} only {} bytes read\n", line, cur_fd.gcount());
                abort();
            }
            for (size_t i = 0; i < space_ndom_pts; i++)
            {
                input->domain(ofst + i, 4) =
                    sqrt(   vel[3 * i    ] * vel[3 * i    ] +
                            vel[3 * i + 1] * vel[3 * i + 1] +
                            vel[3 * i + 2] * vel[3 * i + 2] );
                input->domain(i, 4) *= a->s[0];
                // debug: print the first few velocities
//                 if (i < 5)
//                     fprintf(stderr, "vel [%.3f %.3f %.3f]\n", vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]);
            }

            // rest is hard-coded for 4d

            // find extent of range
            for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
            {
                if (i == 0 || input->domain(i, 4) < bounds_mins(4))
                    bounds_mins(4) = input->domain(i, 4);
                if (i == 0 || input->domain(i, 4) > bounds_maxs(4))
                    bounds_maxs(4) = input->domain(i, 4);
            }

            // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
            size_t n = 0;
            for (size_t l = 0; l < (size_t)(ndom_pts(3)); l++)
                for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
                    for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                        for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                        {
                            input->domain(n, 0) = i;
                            input->domain(n, 1) = j;
                            input->domain(n, 2) = k;
                            input->domain(n, 3) = l;
                            n++;
                        }

            cur_fd.close();
        }   // for all files
        fd.close();

        // extents
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_mins(2) = 0.0;
        bounds_mins(3) = 0.0;
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = input->domain(tot_ndom_pts - 1, 1);
        bounds_maxs(2) = input->domain(tot_ndom_pts - 1, 2);
        bounds_maxs(3) = input->domain(tot_ndom_pts - 1, 3);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d vector dataset and take a 3d subset out of it
    // f = (x, y, z, velocity magnitude)
    void read_3d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 3);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 3);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        vector<float> vel(a->full_dom_pts[0] * a->full_dom_pts[1] * a->full_dom_pts[2] * 3);

        FILE *fd = fopen(a->infile.c_str(), "r");
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
                input->domain(n, 0) = ijk[0];                  // domain is just i,j
                input->domain(n, 1) = ijk[1];
                input->domain(n, 2) = ijk[2];
                input->domain(n, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
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
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 3) < bounds_mins(3))
                bounds_mins(3) = input->domain(i, 3);
            if (i == 0 || input->domain(i, 3) > bounds_maxs(3))
                bounds_maxs(3) = input->domain(i, 3);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        bounds_mins(0) = input->domain(0, 0);
        bounds_mins(1) = input->domain(0, 1);
        bounds_mins(2) = input->domain(0, 2);
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = input->domain(tot_ndom_pts - 1, 1);
        bounds_maxs(2) = input->domain(tot_ndom_pts - 1, 2);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 2d scalar dataset
    // f = (x, y, value)
    void read_2d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 2);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 2);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        vector<float> val(tot_ndom_pts);

        FILE *fd = fopen(a->infile.c_str(), "r");
        assert(fd);

        // read data values
        if (!fread(&val[0], sizeof(float), tot_ndom_pts, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < val.size(); i++)
            input->domain(i, 2) = val[i];

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 2) < bounds_mins(2))
                bounds_mins(2) = input->domain(i, 2);
            if (i == 0 || input->domain(i, 2) > bounds_maxs(2))
                bounds_maxs(2) = input->domain(i, 2);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
            for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
            {
                input->domain(n, 0) = i;
                input->domain(n, 1) = j;
                n++;
            }

        // extents
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = input->domain(tot_ndom_pts - 1, 1);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    // read a floating point 3d scalar dataset
    // f = (x, y, z, value)
    template <typename P>                   // input file precision (e.g., float or double)
    void read_3d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 3);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 3);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)     =  a->ndom_pts[i];
            tot_ndom_pts    *= ndom_pts(i);
        }

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        vector<P> val(tot_ndom_pts);

        FILE *fd = fopen(a->infile.c_str(), "r");
        assert(fd);

        // read data values
        if (!fread(&val[0], sizeof(P), tot_ndom_pts, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < val.size(); i++)
            input->domain(i, 3) = val[i];

        // rest is hard-coded for 3d

        // find extent of range
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 3) < bounds_mins(3))
                bounds_mins(3) = input->domain(i, 3);
            if (i == 0 || input->domain(i, 3) > bounds_maxs(3))
                bounds_maxs(3) = input->domain(i, 3);
        }

        // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
        int n = 0;
        for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                {
                    input->domain(n, 0) = i;
                    input->domain(n, 1) = j;
                    input->domain(n, 2) = k;
                    n++;
                }

        // extents
        bounds_mins(0) = 0.0;
        bounds_mins(1) = 0.0;
        bounds_mins(2) = 0.0;
        bounds_maxs(0) = input->domain(tot_ndom_pts - 1, 0);
        bounds_maxs(1) = input->domain(tot_ndom_pts - 1, 1);
        bounds_maxs(2) = input->domain(tot_ndom_pts - 1, 2);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }


    // TODO: Is this restricted to 3D data only at this point? It has been revised multiple times
    // since it was first named. It could also be extended to multiple science variables easily.
    // There is some support for geom_dim, but it is a bit fragile --David
    void read_3d_unstructured_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.pt_dim() == pt_dim);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);

        input = new mfa::PointSet<T>(dom_dim, mdims, args.tot_ndom_pts);

        // vector<float> vel(pt_dim * args.tot_ndom_pts);

        FILE *fd = fopen(args.infile.c_str(), "r");
        assert(fd);

        // // read file into buffer
        // if (!fread(&vel[0], sizeof(float), args.tot_ndom_pts * pt_dim, fd))
        // {
        //     fprintf(stderr, "Error: unable to read file\n");
        //     exit(0);
        // }


        // build PointSet
        float val = 0;
        for (int i = 0; i < input->npts; i++)
        {
            for (int j = 0; j < input->pt_dim; j++)
            {
                fscanf(fd, "%f", &val);
                input->domain(i, j) = val;
            }
            // for (int k = 0; k < gdim; k++)
            // {
            //     fscanf(fd, "%f", &val);
            //     input->domain(i, k) = val;
            // }
            // for (int k = 0; k < nvars; k++)  
            // {

            //     fscanf(fd, "%f", &val);
            //     input->domain(i, gdim + k) = val;
            //     // if (k == varid)
            //     // {
            //     //     input->domain(i, gdim) = val;
            //     // }
            // }

            // for (int k = 0; k < pt_dim; k++)
            // {
            //     fscanf(fd, "%f", &val);
            //     input->domain(i, k) = val;
            //     // input->domain(i, k) = vel[i * pt_dim + k];
            // }
        }

        // compute bounds in each dimension
        // TODO WARNING use of bounds when dom_dim != geom_dim is not well-defined!
        for (int i = 0; i < pt_dim; i++)
        {
            bounds_mins = input->domain.colwise().minCoeff();
            bounds_maxs = input->domain.colwise().maxCoeff();
        }
        core_mins = bounds_mins.head(dom_dim);
        core_maxs = bounds_maxs.head(dom_dim);

        VectorX<T> dom_mins(dom_dim), dom_maxs(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            dom_mins(i) = args.min[i];
            dom_maxs(i) = args.max[i];
        }
        input->set_domain_params(dom_mins, dom_maxs);

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

    void get_box_intersections(
        T alpha,
        T rho,
        T& x0,
        T& y0,
        T& x1,
        T& y1,
        const VectorX<T>& mins,
        const VectorX<T>& maxs)
    {
        T xl = mins(0);
        T xh = maxs(0);
        T yl = mins(1);
        T yh = maxs(1);

        T yh_int = (rho - yh * sin(alpha)) / cos(alpha);
        T yl_int = (rho - yl * sin(alpha)) / cos(alpha);
        T xh_int = (rho - xh * cos(alpha)) / sin(alpha);
        T xl_int = (rho - xl * cos(alpha)) / sin(alpha);
        // T x0, x1, y0, y1;
 
        // cerr << "ia=" << ia << ", ir=" << ir << endl;
        // cerr << "rho=" << rho << ", alpha=" << alpha << endl;
        // cerr << xl_int << " " << xh_int << " " << yl_int << " " << yh_int << endl;

        // "box intersection" setup
        // start/end coordinates of the ray formed by intersecting 
        // the line with bounding box of the data
        if (alpha == 0)    // vertical lines (top to bottom)
        {
            x0 = rho;
            y0 = yh;
            x1 = rho;
            y1 = yl;
        }
        else if (sin(alpha) == 0 && alpha > 0) // vertical lines (bottom to top)
        {
            x0 = rho;
            y0 = yl;
            x1 = rho;
            y1 = yh;
        }
        else if (cos(alpha)==0) // horizontal lines
        {
            x0 = xl;
            y0 = rho;
            x1 = xh;
            y1 = rho;
        }
        else if (xl_int >= yl && xl_int <= yh)  // enter left
        {
            x0 = xl;
            y0 = xl_int;

            if (yl_int >= xl && yl_int <= xh)   // enter left, exit bottom
            {
                y1 = yl;
                x1 = yl_int;
            }
            else if (yh_int >= xl && yh_int <= xh)  // enter left, exit top
            {
                y1 = yh;
                x1 = yh_int;
            }
            else if (xh_int >= yl && xh_int <= yh)  // enter left, exit right
            {
                x1 = xh;
                y1 = xh_int;
            }
            else
            {
                cerr << "ERROR: invalid state 1" << endl;
                // cerr << "ia = " << ia << ", ir = " << ir << endl;
                exit(1);
            }
        }
        else if (yl_int >= xl && yl_int <= xh)  // enter or exit bottom
        {
            if (yh_int >= xl && yh_int <= xh)   // enter/exit top & bottom
            {
                if (sin(alpha) == 0)    // vertical line case (should have been handled above)
                {
                    cerr << "ERROR: invalid state 6" << endl;
                    x0 = yl_int;
                    y0 = yl;
                    x1 = yh_int;
                    y1 = yh;
                }
                else if (sin(alpha) == 0 && alpha > 0)     // opposite vertical line case (should have been handled above)
                {
                    cerr << "ERROR: invalid state 7" << endl;
                    x0 = yh_int;
                    y0 = yh;
                    x1 = yl_int;
                    y1 = yl;
                }
                // else if (yl_int < yh_int)   // enter bottom, exit top
                else if (alpha > 3.14159265358979/2)
                { 
                    x0 = yl_int;
                    y0 = yl;
                    x1 = yh_int;
                    y1 = yh;
                }
                // else if (yl_int > yh_int)   // enter top, exit bottom
                else if (alpha < 3.14159265358979/2)
                {
                    x0 = yh_int;
                    y0 = yh;
                    x1 = yl_int;
                    y1 = yl;
                }
                else
                {
                    cerr << "ERROR: invalid state 2" << endl;
                    // cerr << "ia = " << ia << ", ir = " << ir << endl;
                    exit(1);
                }
            }
            else if (xh_int >= yl && xh_int <= yh)  // enter bottom, exit right
            {
                x0 = yl_int;
                y0 = yl;
                x1 = xh;
                y1 = xh_int;
            }
            else
            {
                cerr << "ERROR: invalid state 3" << endl;
                // cerr << "ia = " << ia << ", ir = " << ir << endl;
                exit(1);
            }
        }
        else if (yh_int >= xl && yh_int <= xh)  // enter top (cannot be exit top b/c of cases handled previously)
        {
            if (xh_int >= yl && xh_int <= yh)   // enter top, exit right
            {
                x0 = yh_int;
                y0 = yh;
                x1 = xh;
                y1 = xh_int;
            }
            else
            {
                cerr << "ERROR: invalid state 4" << endl;
                // cerr << "ia = " << ia << ", ir = " << ir << endl;
                exit(1);
            }
        }
        else
        {
            x0 = 0;
            y0 = 0;
            x1 = 0;
            y1 = 0;
            // cerr << "ERROR: invalid state 5" << endl;
            // cerr << "ia = " << ia << ", ir = " << ir << endl;
            // exit(1);
        }
    }

    // ONLY 2d AT THE MOMENT
    void create_ray_model(
        const       diy::Master::ProxyWithLink& cp,
        MFAInfo& mfa_info,
        DomainArgs& args,
        bool fixed_length,
        int n_samples,
        int n_rho,
        int n_alpha,
        int v_samples,
        int v_rho,
        int v_alpha)
    {
        // precondition: Block already contains a fully encoded MFA
        DomainArgs* a = &args;

        const double pi = 3.14159265358979;
        assert (dom_dim == 2); // TODO: extended to any dimensionality
        if (n_samples == 0 || n_rho == 0 || n_alpha == 0)
        {
            cerr << "ERROR: Did not set n_samples, n_rho, or n_alpha before creating a ray model. See command line help" << endl;
            exit(1);
        }
        if (v_samples == 0 || v_rho == 0 || v_alpha == 0)
        {
            cerr << "ERROR: Did not set v_samples, v_rho, or v_alpha before creating a ray model. See command line help" << endl;
            exit(1);
        }

        int new_dd = dom_dim + 1;   // new dom_dim
        int new_pd = pt_dim + 1;    // new pt_dim

        // Create a model_dims vector for the auxilliary model, and increase the dimension of geometry
        VectorXi new_mdims = mfa->model_dims();
        new_mdims[0] += 1;  

        VectorXi ndom_pts{{n_samples, n_rho, n_alpha}};
        int npts = ndom_pts.prod();

        mfa::PointSet<T>* ray_input = nullptr;

cout << "\n\nATTENTION: Using experimental separable constrained encode\n" << endl;
        // if (fixed_length)
        //     ray_input = new mfa::PointSet<T>(new_dd, new_mdims, npts);
        // else
            ray_input = new mfa::PointSet<T>(new_dd, new_mdims, npts, ndom_pts);
       

        // extents of domain in physical space
        VectorX<T> param(dom_dim);
        VectorX<T> outpt(pt_dim);
        const T xl = bounds_mins(0);
        const T xh = bounds_maxs(0);
        const T yl = bounds_mins(1);
        const T yh = bounds_maxs(1);

        this->box_mins = bounds_mins.head(dom_dim);
        this->box_maxs = bounds_maxs.head(dom_dim);

        // TODO: make this generic
        double r_lim = 0;
        if (fixed_length)
        {
            double max_radius = max(max(abs(xl),abs(xh)), max(abs(yl),abs(yh)));
            r_lim = max_radius * 1.5;
        } 
        else
        {
            r_lim = 0.99 * xh; // HACK this only works for square domains centered at origin, and for the "box intersection" setup
        }
        double dr = r_lim * 2 / (n_rho-1);
        double da = pi / (n_alpha-1); // d_alpha; amount to rotate on each slice

        // fill ray data set
        double alpha    = 0;   // angle of rotation
        double rho      = -r_lim;
        for (int ia = 0; ia < n_alpha; ia++)
        {
            alpha = ia * da;

            for (int ir = 0; ir < n_rho; ir++)
            {
                rho = -r_lim + ir * dr;

                T x0, y0, x1, y1, span_x, span_y;
                if (fixed_length)
                {
                    // "parallel-plate setup"
                    // start/end coordinates of the ray (alpha, rho)
                    // In this setup the length of every segment (x0,y0)--(x1,y1) is constant
                    span_x = 2 * r_lim * sin(alpha);
                    span_y = 2 * r_lim * cos(alpha);
                    x0 = rho * cos(alpha) - r_lim * sin(alpha);
                    x1 = rho * cos(alpha) + r_lim * sin(alpha);
                    y0 = rho * sin(alpha) + r_lim * cos(alpha);
                    y1 = rho * sin(alpha) - r_lim * cos(alpha);
                }
                else
                {
                    get_box_intersections(alpha, rho, x0, y0, x1, y1, this->box_mins, this->box_maxs);
                    span_x = x1 - x0;
                    span_y = y1 - y0;
                }

                
                // "rotating chords" setup
                // start/end coordinates of the chord formed by intersecting 
                // the line with the circle of radius r_lim
                // T delta_x = 2*sqrt(r_lim*r_lim - rho*rho) * sin(alpha);
                // T delta_y = 2*sqrt(r_lim*r_lim - rho*rho) * cos(alpha);
                // T x0 = rho * cos(alpha) - delta_x/2;
                // T x1 = rho * cos(alpha) + delta_x/2;
                // T y0 = rho * sin(alpha) + delta_y/2;
                // T y1 = rho * sin(alpha) - delta_y/2;

                T dx = span_x / (n_samples-1);
                T dy = span_y / (n_samples-1);

                // cerr << "x0: " << x0 << ", x1: " << x1 << endl;
                // cerr << "y0: " << y0 << ", y1: " << y1 << endl;
                // cerr << "dx: " << dx << endl;
                // cerr << "dy: " << dy << endl;
                // cerr << "span_x: " << span_x << endl;
                // cerr << "span_y: " << span_y << endl;

                for (int is = 0; is < n_samples; is++)
                {
                    int idx = ia*n_rho*n_samples + ir*n_samples + is;
                    ray_input->domain(idx, 0) = (double)is / (n_samples-1);
                    ray_input->domain(idx, 1) = rho;
                    ray_input->domain(idx, 2) = alpha;

                    T x = x0 + is * dx;
                    T y = 0;

                    if (fixed_length)
                        y = y0 - is * dy;
                    else
                        y = y0 + is * dy;

                    // If this point is not in the original domain
                    if (x < xl - 1e-8 || x > xh + 1e-8 || y < yl - 1e-8 || y > yh + 1e-8)
                    {
                        if (fixed_length)  // do nothing in fixed_length setting
                        {
                            ray_input->domain(idx, new_dd) = 0;
                            continue;
                        }
                        else                // else complain and zero-pad (this should not happen)
                        {
                            cerr << "NOT IN DOMAIN" << endl;
                            cerr << "  " << x << "\t" << y << endl;
                            ray_input->domain(idx,3) = 0;
                        }
                    }
                    else    // point is in domain, decode value from existing MFA
                    {
                        param(0) = (x - xl) / (xh - xl);
                        param(1) = (y - yl) / (yh - yl);

                        // Truncate to [0,1] in the presence of small round-off errors
                        param(0) = param(0) < 0 ? 0 : param(0);
                        param(1) = param(1) < 0 ? 0 : param(1);
                        param(0) = param(0) > 1 ? 1 : param(0);
                        param(1) = param(1) > 1 ? 1 : param(1);

                        outpt.resize(pt_dim);
                        this->mfa->Decode(param, outpt);
                        ray_input->domain.block(idx, new_dd, 1, pt_dim - dom_dim) = outpt.tail(pt_dim - dom_dim).transpose();
                    }
                }
            }
        }
        
        // Set parameters for new input
        VectorX<T> input_mins(new_dd), input_maxs(new_dd);
        if (fixed_length)
        {
            input_mins(0) = 0; input_maxs(0) = 1;
            input_mins(1) = -r_lim; input_maxs(1) = r_lim;
            input_mins(2) = 0; input_maxs(2) = pi;
        }
        cerr << "input_mins: " << input_mins << endl;
        cerr << "input_maxs: " << input_maxs << endl;

        if (fixed_length)
            ray_input->set_bounds(input_mins, input_maxs);

        ray_input->set_domain_params();

        // ------------ Creation of new MFA ------------- //
        //
        // Create a new top-level MFA
        int verbose = mfa_info.verbose && cp.master()->communicator().rank() == 0; 
        mfa::MFA<T>* ray_mfa = new mfa::MFA<T>(new_dd, verbose);

        // Set up new geometry
        ray_mfa->AddGeometry(new_dd);

        // Set nctrl_pts, degree for variables
        VectorXi nctrl_pts(new_dd);
        VectorXi p(new_dd);
        for (auto i = 0; i< this->mfa->nvars(); i++)
        {
            int min_p = 20;
            int max_nctrl_pts = 0;
            for (int j = 0; j < dom_dim; j++)
            {
                if (this->mfa->var(i).p(j) < min_p)
                    min_p = this->mfa->var(i).p(j);

                if (this->mfa->var(i).tmesh.tensor_prods[0].nctrl_pts(j) > max_nctrl_pts)
                    max_nctrl_pts = this->mfa->var(i).tmesh.tensor_prods[0].nctrl_pts(j);
            }

            // TODO: this needs to be fixed
            // its possible that max_nctrl_pts is too many if one dimension is much smaller than the others.
            for (auto j = 0; j < new_dd; j++)
            {
                // p(j)  = 3;
                p(j)            = min_p;
                // nctrl_pts(j)    = max_nctrl_pts * floor(sqrt(2));
            }

            nctrl_pts(0) = v_samples;
            nctrl_pts(1) = v_rho;
            nctrl_pts(2) = v_alpha;
            // p(2) = 2;

            ray_mfa->AddVariable(p, nctrl_pts, 1);
        }

        // // Encode ray model. TODO: regularized encode
        // bool force_unified = fixed_length;  // force a unified encoding to use the regularizer
        // ray_mfa->FixedEncode(*ray_input, mfa_info.regularization, mfa_info.reg1and2, false, force_unified);
        
        ray_mfa->FixedEncodeGeom(*ray_input, 0, false);
        ray_mfa->FixedEncodeSeparableCons(0, *ray_input);


        // ----------- Replace old block members with new ---------- //
        // reset block members as needed
        dom_dim = new_dd;
        pt_dim = new_pd;

        // replace original mfa with ray-mfa
        delete this->mfa;
        this->mfa = ray_mfa;
        ray_mfa = nullptr;

        // replace original input with ray-model input
        delete input;
        input = ray_input;
        ray_input = nullptr;

        if (new_pd != this->mfa->pt_dim)    // sanity check
        {
            cerr << "ERROR: pt_dim does not match in create_ray_model()" << endl;
            exit(1);
        }

        VectorX<T> old_bounds_mins = bounds_mins;
        VectorX<T> old_bounds_maxs = bounds_maxs;

        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        bounds_mins(0) = 0;
        bounds_maxs(0) = 1;
        bounds_mins(1) = -r_lim;
        bounds_maxs(1) = r_lim;
        bounds_mins(2) = 0;
        bounds_maxs(2) = pi;
        for (int i = 0; i < this->mfa->pt_dim - this->mfa->geom_dim(); i++)
        {
            bounds_mins(3+i) = old_bounds_mins(2+i);
            bounds_maxs(3+i) = old_bounds_maxs(2+i);
        }
        core_mins = bounds_mins.head(dom_dim);
        core_maxs = bounds_maxs.head(dom_dim);

        this->max_errs.resize(this->mfa->nvars());
        this->sum_sq_errs.resize(this->mfa->nvars());

        this->is_ray_model = true;


        // --------- Decode (for visualization) --------- //
        // this->decode_block(cp, 0);
        // this->range_error(cp, true, true);

        vector<int> grid_size = {100, 100, 100};
        VectorXi gridpoints(3);
        gridpoints(0) = grid_size[0];
        gridpoints(1) = grid_size[1];
        gridpoints(2) = grid_size[2];
        this->decode_block(cp, true);
        

cerr << "\n===========" << endl;
cerr << "f(x) = sin(x) hardcoded in create_ray_model()" << endl;
cerr << "===========\n" << endl;
        delete this->errs;
        this->errs = new mfa::PointSet<T>(dom_dim, mfa->model_dims(), gridpoints.prod(), gridpoints);
        outpt = VectorX<T>::Zero(1);
        param.resize(dom_dim); // n.b. this is now the NEW dom_dim
        for (int k = 0; k < grid_size[2]; k++)
        {
            for (int j = 0; j < grid_size[1]; j++)
            {
                T rh_param = (T) j / (grid_size[1]-1);
                T al_param = (T) k / (grid_size[2]-1);
                T rh = input->mins(1) + (input->maxs(1) - input->mins(1)) * rh_param;
                T al = input->mins(2) + (input->maxs(2) - input->mins(2)) * al_param;

                T x0, y0, x1, y1, span_x, span_y;

                // "parallel-plate setup"
                // start/end coordinates of the ray (alpha, rho)
                // In this setup the length of every segment (x0,y0)--(x1,y1) is constant
                span_x = 2 * r_lim * sin(al);
                span_y = 2 * r_lim * cos(al);
                x0 = rh * cos(al) - r_lim * sin(al);
                x1 = rh * cos(al) + r_lim * sin(al);
                y0 = rh * sin(al) + r_lim * cos(al);
                y1 = rh * sin(al) - r_lim * cos(al);

                T dx = span_x / (grid_size[0]-1);
                T dy = span_y / (grid_size[0]-1);

                for (int i = 0; i < grid_size[0]; i++)
                {
                    T t_param = (T)i / (grid_size[0]-1);
                    int idx = k*grid_size[0]*grid_size[1] + j*grid_size[0] + i;
                    T a = input->mins(0) + (input->maxs(0) - input->mins(0)) / (grid_size[0]-1) * i;
                    
                    T x = x0 + i * dx;
                    T y = 0;

                    if (fixed_length)
                        y = y0 - i * dy;
                    else
                        y = y0 + i * dy;
                    
                    param(0) = t_param;
                    param(1) = rh_param;
                    param(2) = al_param;

                    // Truncate to [0,1] in the presence of small round-off errors
                    param(0) = param(0) < 0 ? 0 : param(0);
                    param(1) = param(1) < 0 ? 0 : param(1);
                    param(2) = param(2) < 0 ? 0 : param(2);
                    param(0) = param(0) > 1 ? 1 : param(0);
                    param(1) = param(1) > 1 ? 1 : param(1);
                    param(2) = param(2) > 1 ? 1 : param(2);

                    this->mfa->DecodeVar(0, param, outpt);

                    T trueval = sin(x) * sin(y);

                    this->errs->domain(idx, 0) = t_param;
                    this->errs->domain(idx, 1) = rh;
                    this->errs->domain(idx, 2) = al;
                    this->errs->domain(idx, 3) = abs(trueval - outpt(0));

                    // ignore "errors" when querying outside the domain
                    if (x < xl || x > xh || y < yl || y > yh)
                        this->errs->domain(idx, 3) = 0;
                }
            }
        }

        // Compute error metrics
        MatrixX<T>& errpts = this->errs->domain;

        for (auto j = dom_dim; j < this->errs->pt_dim; j++)
            this->sum_sq_errs[j - dom_dim] = 0.0;
        for (auto i = 0; i < this->errs->npts; i++)
        {
            for (auto j = dom_dim; j < this->errs->pt_dim; j++)
            {
                this->sum_sq_errs[j - dom_dim] += (errpts(i, j) * errpts(i, j));
                if ((i == 0 && j == dom_dim) || errpts(i, j) > this->max_errs[j - dom_dim])
                    this->max_errs[j - dom_dim] = errpts(i, j);
            }
        }
    }

    pair<T,T> dualCoords(const VectorX<T>& a, const VectorX<T>& b)
    {
        const double pi = 3.14159265358979;

        T a_x = a(0);
        T a_y = a(1);
        T b_x = b(0);
        T b_y = b(1);

         // distance in x and y between the endpoints of the segment
        T delta_x = b_x - a_x;
        T delta_y = b_y - a_y;


        T alpha = -1;
        T rho = 0;

        if (a_x == b_x)
        {
            alpha = 0;
            rho = a_x;
        }
        else
        {
            T m = (b_y-a_y)/(b_x-a_x);
            alpha = pi/2 - atan(-m);            // acot(x) = pi/2 - atan(x)
            rho = (a_y - m*a_x)/(sqrt(1+m*m));  // cos(atan(x)) = 1/sqrt(1+m*m), sin(pi/2-x) = cos(x)
        }

        return make_pair(alpha, rho);
    }

    T integrate_ray(
        const   diy::Master::ProxyWithLink& cp,
        const   VectorX<T>& a,
        const   VectorX<T>& b,
                bool fixed_length)
    {
        const double pi = 3.14159265358979;
        const bool verbose = false;

        // TODO: This is for 2d only right now
        if (a.size() != 2 && b.size() != 2)
        {
            cerr << "ERROR: Incorrect dimension in integrate ray. Exiting." << endl;
            exit(1);
        }

        auto ar_coords = dualCoords(a, b);
        T alpha = ar_coords.first;
        T rho   = ar_coords.second;

        T a_x = a(0);
        T a_y = a(1);
        T b_x = b(0);
        T b_y = b(1);

        // // distance in x and y between the endpoints of the segment
        // T delta_x = b_x - a_x;
        // T delta_y = b_y - a_y;

        // T m = (b_y-a_y)/(b_x-a_x);
        // T alpha = -1;
        // T rho = 0;

        // if (a_x == b_x)
        // {
        //     alpha = 0;
        //     rho = a_x;
        // }
        // else
        // {
        //     alpha = pi/2 - atan(-m);          // acot(x) = pi/2 - atan(x)
        //     rho = (a_y - m*a_x)/(sqrt(1+m*m));     // cos(atan(x)) = 1/sqrt(1+m*m), sin(pi/2-x) = cos(x)
        // }

        T x0, x1, y0, y1;   // end points of full line
        T u0 = 0, u1 = 0;
        T length = 0;
        T r_lim = bounds_maxs(1);   // WARNING TODO: make r_lim query-able in RayMFA class
        if (fixed_length)
        {
            x0 = rho * cos(alpha) - r_lim * sin(alpha);
            x1 = rho * cos(alpha) + r_lim * sin(alpha);
            y0 = rho * sin(alpha) + r_lim * cos(alpha);
            y1 = rho * sin(alpha) - r_lim * cos(alpha);
        }
        else
        {
            get_box_intersections(alpha, rho, x0, y0, x1, y1, this->box_mins, this->box_maxs);
        }

        // parameter values along ray for 'start' and 'end'
        // compute in terms of Euclidean distance to avoid weird cases
        //   when line is nearly horizontal or vertical
        T x_sep = abs(x1 - x0);
        T y_sep = abs(y1 - y0);
        if (fixed_length)
            length = 2 * r_lim;
        else
            length = sqrt(x_sep*x_sep + y_sep*y_sep);
        
        if (x_sep > y_sep)  // want to avoid dividing by near-epsilon numbers
        {
            u0 = abs(a_x - x0) / x_sep;
            u1 = abs(b_x - x0) / x_sep;
        }
        else
        {
            u0 = abs(a_y - y0) / y_sep;
            u1 = abs(b_y - y0) / y_sep;
        }

        // Scalar valued path integrals do not have an orientation, so we always
        // want the limits of integration to go from smaller to larger.
        if (u0 > u1)
        {
            T temp  = u1;
            u1 = u0;
            u0 = temp;
        }

        if (verbose)
        {
            cerr << "RAY: (" << a(0) << ", " << a(1) << ") ---- (" << b(0) << ", " << b(1) << ")" << endl;
            cerr << "|  m: " << ((a_x==b_x) ? "inf" : to_string((b_y-a_y)/(b_x-a_x)).c_str()) << endl;
            cerr << "|  alpha:  " << alpha << ",   rho: " << rho << endl;
            cerr << "|  length: " << length << endl;
            cerr << "|  u0: " << u0 << ",  u1: " << u1 << endl;
            cerr << "+---------------------------------------\n" << endl;
        }

        VectorX<T> output(1); // todo: this is hardcoded for the first (scalar) variable only
        this->integrate_axis_ray(cp, alpha, rho, u0, u1, length, output);

        return output(0);
    }

    // Compute error metrics between a pointset and an analytical function
    // evaluated at the points in the pointset
    void analytical_error_pointset(
        const diy::Master::ProxyWithLink&   cp,
        mfa::PointSet<T>*                   ps,
        string                              fun,
        vector<T>&                          L1, 
        vector<T>&                          L2,
        vector<T>&                          Linf,
        DomainArgs&                         args,
        const std::function<void(const VectorX<T>&, VectorX<T>&, DomainArgs&, int)>& f = {}) const
    {
        int nvars = ps->nvars();
        if (L1.size() != nvars || L2.size() != nvars || Linf.size() != nvars)
        {
            cerr << "ERROR: Error metric vector sizes do not match in analytical_error_pointset().\nAborting" << endl;
            exit(1);
        }

        // Compute the analytical error at each point
        T l1err = 0, l2err = 0, linferr = 0;
        VectorX<T> dom_pt(ps->geom_dim());

        for (int k = 0; k < ps->nvars(); k++)
        {
            VectorX<T> true_pt(ps->var_dim(k));
            VectorX<T> test_pt(ps->var_dim(k));
            VectorX<T> residual(ps->var_dim(k));

            for (auto pt_it = ps->begin(), pt_end = ps->end(); pt_it != pt_end; ++pt_it)
            {
                pt_it.geom_coords(dom_pt); // extract the geometry coordinates

                // Get exact value. If 'f' is non-NULL, ignore 'fun'
                if (f)
                    f(dom_pt, true_pt, args, k);
                else
                    evaluate_function(fun, dom_pt, true_pt, args, k);

                // Get approximate value
                pt_it.var_coords(k, test_pt);

                // NOTE: For now, we are using the norm of the residual for all error statistics.
                //       Is this the most appropriate way to measure errors norms of a vector field?
                //       May want to consider revisiting this.
                //
                // Compute errors for this point. When the science variable is vector-valued, we 
                // distinguish between the L1, L2, and Linfty distances. L1 distance is 
                // used for 'sum_errs', L2 for 'sum_sq_errs,' and Linfty for 'max_err.'
                // Thus, 'max_err' reports the maximum difference in any vector
                // component, taken over all of the points in the Pointset.
                //
                // n.b. When the science variable is scalar valued, L2 error and Linfty error are the same. 
                residual = (true_pt - test_pt).cwiseAbs();
                // l1err   = residual.sum();
                l2err   = residual.norm();          // L2 difference between vectors 
                // linferr = residual.maxCoeff();      // Maximum difference in components

                // Update error statistics
                L1[k]   += l2err;
                L2[k]   += l2err * l2err;
                if (l2err > Linf[k]) Linf[k] = l2err;
            }

            L1[k] = L1[k] / ps->npts;
            L2[k] = sqrt(L2[k] / ps->npts);
        }
    }


    // Simplified function signature when we don't need to keep the PointSets
    void analytical_error_field(
        const diy::Master::ProxyWithLink&   cp,
        string                              fun,                // analytical function name
        vector<T>&                          L1,                 // (output) L-1 norm
        vector<T>&                          L2,                 // (output) L-2 norm
        vector<T>&                          Linf,               // (output) L-infinity norm
        DomainArgs&                         args,               // input args
        vector<T>                           subset_mins = vector<T>(),
        vector<T>                           subset_maxs = vector<T>() ) // (optional) subset of the domain to consider for errors
    {
        mfa::PointSet<T>* unused = nullptr;
        analytical_error_field(cp, fun, L1, L2, Linf, args, unused, unused, unused, subset_mins, subset_maxs);
    }

    // Compute error field on a regularly spaced grid of points. The size of the grid
    // is given by args.ndom_pts. Error metrics are saved in L1, L2, Linf. The fields 
    // of the exact, approximate, and residual data are save to PointSets.
    void analytical_error_field(
        const diy::Master::ProxyWithLink&   cp,
        string                              fun,                // analytical function name
        vector<T>&                          L1,                 // (output) L-1 norm
        vector<T>&                          L2,                 // (output) L-2 norm
        vector<T>&                          Linf,               // (output) L-infinity norm
        DomainArgs&                         args,               // input args
        mfa::PointSet<T>*&                  exact_pts,          // PointSet to contain analytical signal
        mfa::PointSet<T>*&                  approx_pts,         // PointSet to contain approximation
        mfa::PointSet<T>*&                  error_pts,          // PointSet to contain errors
        vector<T>                           subset_mins = vector<T>(),
        vector<T>                           subset_maxs = vector<T>() ) // (optional) subset of the domain to consider for errors
    {
        int nvars = this->mfa->nvars();
        if (L1.size() != nvars || L2.size() != nvars || Linf.size() != nvars)
        {
            cerr << "ERROR: Error metric vector sizes do not match in analytical_error_field().\nAborting" << endl;
            exit(1);
        }

        // Check if we accumulated errors over subset of domain only and report
        bool do_subset = false;
        bool in_box = true;
        if (subset_mins.size() != 0)
        {
            do_subset = true;

            if (cp.gid() == 0)
            {
                cout << "Accumulating errors over subset of domain" << endl;
                cout << "  subset mins: " << mfa::print_vec(subset_mins) << endl;
                cout << "  subset maxs: " << mfa::print_vec(subset_maxs) << endl;
            }

            if (subset_mins.size() != subset_maxs.size())
            {
                cerr << "ERROR: Dimensions of subset_mins and subset_maxs do not match" << endl;
                exit(1);
            }
            if (subset_mins.size() != dom_dim)
            {
                cerr << "ERROR: subset dimension does not match dom_dim" << endl;
                exit(1);
            }
        }
        
        // Size of grid on which to test error
        VectorXi test_pts(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            test_pts(i) = args.ndom_pts[i];
        }

        // Free any existing memory at PointSet pointers
        if (exact_pts) cerr << "Warning: Overwriting \'exact_pts\' pointset in analytical_error_field()" << endl;
        if (approx_pts) cerr << "Warning: Overwriting \'approx_pts\' pointset in analytical_error_field()" << endl;
        if (error_pts) cerr << "Warning: Overwriting \'error_pts\' pointset in analytical_error_field()" << endl;
        delete exact_pts;
        delete approx_pts;
        delete error_pts;

        // Set up PointSets with grid parametrizations
        exact_pts = new mfa::PointSet<T>(mfa->dom_dim, mfa->model_dims(), test_pts.prod(), test_pts);
        approx_pts= new mfa::PointSet<T>(mfa->dom_dim, mfa->model_dims(), test_pts.prod(), test_pts);
        error_pts = new mfa::PointSet<T>(mfa->dom_dim, mfa->model_dims(), test_pts.prod(), test_pts);
        approx_pts->set_grid_params();

        // Decode on above-specified grid
        mfa->Decode(*approx_pts, false);

        // Copy geometric point coordinates into error and exact PointSets
        exact_pts->domain.leftCols(exact_pts->geom_dim()) = approx_pts->domain.leftCols(approx_pts->geom_dim());
        error_pts->domain.leftCols(error_pts->geom_dim()) = approx_pts->domain.leftCols(approx_pts->geom_dim());

        // Compute the analytical error at each point and accrue errors
        T l1err = 0, l2err = 0, linferr = 0;
        VectorX<T> dom_pt(approx_pts->geom_dim());

        for (int k = 0; k < nvars; k++)
        {
            VectorX<T> true_pt(approx_pts->var_dim(k));
            VectorX<T> test_pt(approx_pts->var_dim(k));
            VectorX<T> residual(approx_pts->var_dim(k));
            int num_pts_in_box = 0;

            for (auto pt_it = approx_pts->begin(), pt_end = approx_pts->end(); pt_it != pt_end; ++pt_it)
            {
                pt_it.geom_coords(dom_pt); // extract the geometry coordinates

                // Get exact and approximate values
                evaluate_function(fun, dom_pt, true_pt, args, k);
                pt_it.var_coords(k, test_pt);

                // Update error field
                residual = (true_pt - test_pt).cwiseAbs();
                for (int j = 0; j <= error_pts->var_dim(k); j++)
                {
                    error_pts->domain(pt_it.idx(), error_pts->var_min(k) + j) = residual(j);
                    exact_pts->domain(pt_it.idx(), exact_pts->var_min(k) + j) = true_pt(j);
                }

                // Accrue error only in subset
                in_box = true;
                if (do_subset) 
                {
                    for (int i = 0; i < dom_dim; i++)
                        in_box = in_box && (dom_pt(i) >= subset_mins[i]) && (dom_pt(i) <= subset_maxs[i]);
                }

                if (in_box)
                {
                    // NOTE: For now, we are using the norm of the residual for all error statistics.
                    //       Is this the most appropriate way to measure errors norms of a vector field?
                    //       May want to consider revisiting this.
                    //
                    // l1err   = residual.sum();           // L1 difference between vectors
                    l2err   = residual.norm();          // L2 difference between vectors 
                    // linferr = residual.maxCoeff();      // Maximum difference in components

                    L1[k]   += l2err;
                    L2[k]   += l2err * l2err;
                    if (l2err > Linf[k]) Linf[k] = l2err;

                    num_pts_in_box++;
                }
            }

            L1[k] = L1[k] / num_pts_in_box;
            L2[k] = sqrt(L2[k] / num_pts_in_box);
        }
    }

    // static
    // void readfile_unstructured(
    //         int gid,                        // block global id
    //         const Bounds<int> &core,        // block bounds without any ghost added
    //         const Bounds<int> &bounds,      // block bounds including any ghost region added
    //         const RCLink<int> &link,        // neighborhood
    //         diy::Master &master,            // diy master
    //         std::vector<int> &mapDimension, // domain dimensionality map;
    //         std::string &s3dfile,           // input file with data
    //         std::vector<unsigned> &shape,   // important, shape of the block
    //         int chunk,                      // vector dimension for data input (usually 2 or 3)
    //         int transpose,                  // diy is MPI_C_ORDER always; offer option to transpose
    //         MFAInfo& mfa_info)              // info class describing the MFA
    // {
    //     Block<T> *b = new Block<T>;
    //     RCLink<int> *l = new RCLink<int>(link);
    //     diy::Master & m = const_cast<diy::Master&>(master);
    //     // write core and bounds only for first block
    //     if (0 == gid) {
    //         std::cout << "block:" << gid << "\n  core \t\t  bounds \n";
    //         for (int j = 0; j < 3; j++)
    //             std::cout << " " << core.min[j] << ":" << core.max[j] << "\t\t"
    //                 << " " << bounds.min[j] << ":" << bounds.max[j] << "\n";
    //     }
    //     m.add(gid, b, l);

    //     b->dom_dim = (int) mapDimension.size();
    //     diy::mpi::io::file in(master.communicator(), s3dfile, diy::mpi::io::file::rdonly);
    //     diy::io::BOV reader(in, shape);

    //     int size_data_read = 1;
    //     for (int j = 0; j < 3; j++)  // we know how the s3d data is organized
    //         size_data_read *= (bounds.max[j] - bounds.min[j] + 1);
    //     std::vector<float> data;
    //     data.resize(size_data_read * chunk);
    //     // read bounds will be multiplied by 3 in first direction
    //     Bounds<int> extBounds = bounds;
    //     extBounds.min[2] *= chunk; // multiply by 3
    //     extBounds.max[2] *= chunk; // multiply by 3
    //     extBounds.max[2] += chunk - 1; // the last coordinate is larger!!
    //     bool collective = true; //
    //     reader.read(extBounds, &data[0], collective);
    // }

    static
    void readfile(                          // add the block to the decomposition
            int gid,                        // block global id
            const Bounds<int> &core,        // block bounds without any ghost added
            const Bounds<int> &bounds,      // block bounds including any ghost region added
            const RCLink<int> &link,        // neighborhood
            diy::Master &master,            // diy master
            std::vector<int> &mapDimension, // domain dimensionality map;
            std::string &s3dfile,           // input file with data
            std::vector<unsigned> &shape,   // important, shape of the block
            int chunk,                      // vector dimension for data input (usually 2 or 3)
            int transpose,                  // diy is MPI_C_ORDER always; offer option to transpose
            MFAInfo& mfa_info)              // info class describing the MFA
    {
        Block<T> *b = new Block<T>;
        RCLink<int> *l = new RCLink<int>(link);
        diy::Master & m = const_cast<diy::Master&>(master);
        // write core and bounds only for first block
        if (0 == gid) {
            std::cout << "block:" << gid << "\n  core \t\t  bounds \n";
            for (int j = 0; j < 3; j++)
                std::cout << " " << core.min[j] << ":" << core.max[j] << "\t\t"
                    << " " << bounds.min[j] << ":" << bounds.max[j] << "\n";
        }
        m.add(gid, b, l);

        b->dom_dim = (int) mapDimension.size();
        diy::mpi::io::file in(master.communicator(), s3dfile, diy::mpi::io::file::rdonly);
        diy::io::BOV reader(in, shape);

        int size_data_read = 1;
        for (int j = 0; j < 3; j++)  // we know how the s3d data is organized
            size_data_read *= (bounds.max[j] - bounds.min[j] + 1);
        std::vector<float> data;
        data.resize(size_data_read * chunk);
        // read bounds will be multiplied by 3 in first direction
        Bounds<int> extBounds = bounds;
        extBounds.min[2] *= chunk; // multiply by 3
        extBounds.max[2] *= chunk; // multiply by 3
        extBounds.max[2] += chunk - 1; // the last coordinate is larger!!
        bool collective = true; //
        reader.read(extBounds, &data[0], collective);

        // assumes one scalar science variable
        b->pt_dim = b->dom_dim + 1;
        int nvars = 1;
        b->max_errs.resize(nvars);
        b->sum_sq_errs.resize(nvars);
        b->bounds_mins.resize(b->pt_dim);
        b->bounds_maxs.resize(b->pt_dim);
        VectorXi ndom_pts;  // this will be local now, and used in def of mfa
        ndom_pts.resize(b->dom_dim);
        int tot_ndom_pts = 1;
        for (size_t j = 0; j < b->dom_dim; j++) {
            int dir = mapDimension[j];
            int size_in_dir = -bounds.min[dir] + bounds.max[dir] + 1;
            tot_ndom_pts *= size_in_dir;
            ndom_pts(j) = size_in_dir;
            if (0 == gid)
                cerr << "  dimension " << j << " " << size_in_dir << endl;
        }

        if (!transpose && b->dom_dim > 1)
        {
            if (b->dom_dim == 2)
            {
                int tmp = ndom_pts(0);
                ndom_pts(0) = ndom_pts(1);
                ndom_pts(1) = tmp;
            }
            else if (b->dom_dim == 3)
            {
                int tmp = ndom_pts(2);
                ndom_pts(2) = ndom_pts(0);
                ndom_pts(0) = tmp; // reverse counting
            }
        }

        // Construct point set to contain input
        VectorXi model_dims(2);
        model_dims(0) = b->dom_dim;
        model_dims(1) = b->pt_dim - b->dom_dim;
        b->input = new mfa::PointSet<T>(b->dom_dim, model_dims, tot_ndom_pts, ndom_pts);

        if (0 == gid)
            cerr << " total local size : " << tot_ndom_pts << endl;
        if (b->dom_dim == 1) // 1d problem, the dimension would be x direction
        {
            int dir0 = mapDimension[0];
            b->map_dir.push_back(dir0); // only one dimension, rest are not varying
            for (int i = 0; i < tot_ndom_pts; i++) {
                b->input->domain(i, 0) = bounds.min[dir0] + i;
                int idx = 3 * i;
                float val = 0;
                for (int k = 0; k < chunk; k++)
                    val += data[idx + k] * data[idx + k];
                val = sqrt(val);
                b->input->domain(i, 1) = val;
            }
            mfa_info.var_model_infos[0].nctrl_pts[0] = mfa_info.var_model_infos[0].nctrl_pts[dir0]; // only one direction that matters
        } else if (b->dom_dim == 2) // 2d problem, second direction would be x, first would be y
        {
            if (transpose) {
                int n = 0;
                int idx = 0;
                int dir0 = mapDimension[0]; // so now y would vary to 704 in 2d 1 block similar case for s3d (transpose)
                int dir1 = mapDimension[1];
                // we do not transpose anymore
                b->map_dir.push_back(dir0);
                b->map_dir.push_back(dir1);
                for (int i = 0; i < ndom_pts(0); i++) {
                    for (int j = 0; j < ndom_pts(1); j++) {
                        n = j * ndom_pts(0) + i;
                        b->input->domain(n, 0) = bounds.min[dir0] + i; //
                        b->input->domain(n, 1) = bounds.min[dir1] + j;
                        float val = 0;
                        for (int k = 0; k < chunk; k++)
                            val += data[idx + k] * data[idx + k];
                        val = sqrt(val);
                        b->input->domain(n, 2) = val;
                        idx += 3;
                    }
                }
            } else {
                // keep the order as Paraview, x would be the first that varies
                // corresponds to original implementation, which needs to transpose dimensions
                int n = 0;
                int idx = 0;
                int dir0 = mapDimension[1]; // so x would vary to 704 in 2d 1 block similar case
                int dir1 = mapDimension[0];
                b->map_dir.push_back(dir0);
                b->map_dir.push_back(dir1);
                for (int j = 0; j < ndom_pts(1); j++) {
                    for (int i = 0; i < ndom_pts(0); i++) {
                        b->input->domain(n, 1) = bounds.min[dir1] + j;
                        b->input->domain(n, 0) = bounds.min[dir0] + i;
                        float val = 0;
                        for (int k = 0; k < chunk; k++)
                            val += data[idx + k] * data[idx + k];
                        b->input->domain(n, 2) = sqrt(val);
                        n++;
                        idx += 3;
                    }
                }
            }
        }

        else if (b->dom_dim == 3) {
            if (transpose) {
                int n = 0;
                int idx = 0;
                b->map_dir.push_back(mapDimension[0]);
                b->map_dir.push_back(mapDimension[1]);
                b->map_dir.push_back(mapDimension[2]);
                // last dimension would correspond to x, as in the 2d example
                for (int i = 0; i < ndom_pts(0); i++)
                    for (int j = 0; j < ndom_pts(1); j++)
                        for (int k = 0; k < ndom_pts(2); k++) {
                            // max is ndom_pts(0)*ndom_pts(1)*(ndom_pts(2)-1)+ ndom_pts(0)*(ndom_pts(1)-1)+(ndom_pts(0)-1)
                            //  = ndom_pts(0)*ndom_pts(1)*ndom_pts(2) -ndom_pts(0)*ndom_pts(1) + ndom_pts(0)*ndom_pts(1)
                            //             -ndom_pts(0) + ndom_pts(0)-1 =   ndom_pts(0)*ndom_pts(1)*ndom_pts(2) - 1;
                            n = k * ndom_pts(0) * ndom_pts(1) + j * ndom_pts(0)
                                + i;
                            b->input->domain(n, 0) = bounds.min[0] + i;
                            b->input->domain(n, 1) = bounds.min[1] + j;
                            b->input->domain(n, 2) = bounds.min[2] + k;
                            float val = 0;
                            for (int k = 0; k < chunk; k++)
                                val += data[idx + k] * data[idx + k];
                            val = sqrt(val);
                            b->input->domain(n, 3) = val;
                            idx += 3;
                        }
            } else // visualization order
            {
                int n = 0;
                int idx = 0;
                b->map_dir.push_back(mapDimension[2]); // reverse
                b->map_dir.push_back(mapDimension[1]);
                b->map_dir.push_back(mapDimension[0]);
                // last dimension would correspond to x, as in the 2d example
                for (int k = 0; k < ndom_pts(2); k++)
                    for (int j = 0; j < ndom_pts(1); j++)
                        for (int i = 0; i < ndom_pts(0); i++) {
                            b->input->domain(n, 2) = bounds.min[0] + k;
                            b->input->domain(n, 1) = bounds.min[1] + j;
                            b->input->domain(n, 0) = bounds.min[2] + i; // this now corresponds to x
                            float val = 0;
                            for (int k = 0; k < chunk; k++)
                                val += data[idx + k] * data[idx + k];
                            b->input->domain(n, 3) = sqrt(val);
                            n++;
                            idx += 3;
                        }
            }
        }
        b->core_mins.resize(b->dom_dim);
        b->core_maxs.resize(b->dom_dim);
        b->overlaps.resize(b->dom_dim);
        for (int i = 0; i < b->dom_dim; i++) {
            //int index = b->dom_dim-1-i;
            int index = i;
            if (!transpose)
                index = b->dom_dim - 1 - i;
            b->bounds_mins(i) = bounds.min[mapDimension[index]];
            b->bounds_maxs(i) = bounds.max[mapDimension[index]];
            b->core_mins(i) = core.min[mapDimension[index]];
            b->core_maxs(i) = core.max[mapDimension[index]];
            // decide overlap in each direction; they should be symmetric for neighbors
            // so if block a overlaps block b, block b overlaps a the same area
            b->overlaps(i) = fabs(b->core_mins(i) - b->bounds_mins(i));
            T m2 = fabs(b->bounds_maxs(i) - b->core_maxs(i));
            if (m2 > b->overlaps(i))
                b->overlaps(i) = m2;
        }

        // set bounds_min/max for science variable (last coordinate)
        b->bounds_mins(b->dom_dim) = b->input->domain.col(b->dom_dim).minCoeff();
        b->bounds_maxs(b->dom_dim) = b->input->domain.col(b->dom_dim).maxCoeff();

        b->input->set_domain_params();

        // TODO: check that this construction of ProxyWithLink is valid
        // b->setup_models(diy::Master::ProxyWithLink(diy::Master::Proxy(&m, gid), b, l), nvars, args);     // adds models to MFA
        b->setup_MFA(m.proxy(m.lid(gid)), mfa_info);
    }

};


void max_err_cb(Block<real_t> *b,                  // local block
        const diy::ReduceProxy &rp,                // communication proxy
        const diy::RegularMergePartners &partners) // partners of the current block
{
    unsigned round = rp.round();    // current round number

    // step 1: dequeue and merge
    for (int i = 0; i < rp.in_link().size(); ++i) {
        int nbr_gid = rp.in_link().target(i).gid;
        if (nbr_gid == rp.gid()) {

            continue;
        }

        std::vector<real_t> in_vals;
        rp.dequeue(nbr_gid, in_vals);

        for (size_t j = 0; j < in_vals.size() / 2; ++j) {
            if (b->max_errs_reduce[2 * j] < in_vals[2 * j]) {
                b->max_errs_reduce[2 * j] = in_vals[2 * j];
                b->max_errs_reduce[2 * j + 1] = in_vals[2 * j + 1]; // received from this block
            }
        }
    }

    // step 2: enqueue
    for (int i = 0; i < rp.out_link().size(); ++i) // redundant since size should equal to 1
    {
        // only send to root of group, but not self
        if (rp.out_link().target(i).gid != rp.gid()) {
            rp.enqueue(rp.out_link().target(i), b->max_errs_reduce);
        } //else
        //fmt::print(stderr, "[{}:{}] Skipping sending to self\n", rp.gid(), round);
    }
}


// REMOVE:
// Computes the analytical line integral from p1 to p2 of sin(x)sin(y). Use for testing
template<typename T>
T sintest( const VectorX<T>& p1,
           const VectorX<T>& p2)
{
    T x1 = p1(0);
    T y1 = p1(1);
    T x2 = p2(0);
    T y2 = p2(1);

    T mx = x2 - x1;
    T my = y2 - y1;
    T fctr = sqrt(mx*mx + my*my);

    // a1 = mx, b1 = x1, a2 = my, b2 = y1
    T int0 = 0, int1 = -7;
    if (mx != my)
    {
        int0 = 0.5*(sin(x1-y1+(mx-my)*0)/(mx-my) - sin(x1+y1+(mx+my)*0)/(mx+my));
        int1 = 0.5*(sin(x1-y1+(mx-my)*1)/(mx-my) - sin(x1+y1+(mx+my)*1)/(mx+my));
    }
    else
    {
        int0 = 0.5 * (0*cos(x1-y1) + sin(x1+y1+(mx+my)*0)/(mx+my));
        int1 = 0.5 * (1*cos(x1-y1) + sin(x1+y1+(mx+my)*1)/(mx+my));
    }
    
    return (int1 - int0) * fctr;
}

// evaluate sine function
template<typename T>
void sine(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    T retval = 1.0;
    for (auto i = 0; i < domain_pt.size(); i++)
        retval *= sin(domain_pt(i) * args.f[k]);
    retval *= args.s[k];

    for (int l = 0; l < output_pt.size(); l++)
    {
        output_pt(l) = retval * (1+l);
    }

    return;
}

template<typename T>
void cosine(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    T retval = 1.0;
    for (auto i = 0; i < domain_pt.size(); i++)
        retval *= cos(domain_pt(i) * args.f[k]);
    retval *= args.s[k];

    for (int l = 0; l < output_pt.size(); l++)
    {
        output_pt(l) = retval * (1+l);
    }

    return;        
}

// evaluate the "negative cosine plus one" (f(x) = -cos(x)+1) function
// used primarily to test integration of sine
template<typename T>
void ncosp1(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs& args, int k)
{
    if (output_pt.size() != 1)
    {
        fprintf(stderr, "Error: ncosp1 only defined for scalar output.\n");
        exit(0);
    }

    T retval = 1.0;
    for (auto i = 0; i < domain_pt.size(); i++)
        retval *= 1 - cos(domain_pt(i) * args.f[k]);
    retval *= args.s[k];

    output_pt(0) = retval;
    return;        
}

// evaluate sinc function
template<typename T>
void sinc(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs& args, int k)
{
    T retval = 1.0;
    for (auto i = 0; i < domain_pt.size(); i++)
    {
        if (domain_pt(i) != 0.0)
            retval *= (sin(domain_pt(i) * args.f[k] ) / domain_pt(i));
    }
    retval *= args.s[k];

    for (int l = 0; l < output_pt.size(); l++)
    {
        output_pt(l) = retval * (1+l);
    }

    return;
}

// evaluate n-d poly-sinc function version 1
template<typename T>
void polysinc1(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    int dim = output_pt.size();

    // a = (x + 1)^2 + (y - 1)^2 + (z + 1)^2 + ...
    // b = (x - 1)^2 + (y + 1)^2 + (z - 1)^2 + ...
    T a = 0.0;
    T b = 0.0;
    for (auto i = 0; i < domain_pt.size(); i++)
    {
        T s, r;
        if (i % 2 == 0)
        {
            s = domain_pt(i) + 1.0;
            r = domain_pt(i) - 1.0;
        }
        else
        {
            s = domain_pt(i) - 1.0;
            r = domain_pt(i) + 1.0;
        }
        a += (s * s);
        b += (r * r);
    }

    // Modulate sinc shape for each output coordinate if science variable is vector-valued
    for (int l = 0; l < dim; l++)
    {
        // a1 = sinc(a*(1+l)); b1 = sinc(b*(1+l))
        T a1 = (a == 0.0 ? 1.0*(1+l) : sin(a*(1+l)) / a);
        T b1 = (b == 0.0 ? 1.0*(1+l) : sin(b*(1+l)) / b);
        output_pt(l) = args.s[k] * (a1 + b1);               // scale entire science variable by s[k]
    }

    return;
}

// evaluate n-d poly-sinc function version 2
template<typename T>
void polysinc2(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    int dim = output_pt.size();

    // a = x^2 + y^2 + z^2 + ...
    // b = 2(x - 2)^2 + (y + 2)^2 + (z - 2)^2 + ...
    T a = 0.0;
    T b = 0.0;
    for (auto i = 0; i < domain_pt.size(); i++)
    {
        T s, r;
        s = domain_pt(i);
        if (i % 2 == 0)
        {
            r = domain_pt(i) - 2.0;
        }
        else
        {
            r = domain_pt(i) + 2.0;
        }
        a += (s * s);
        if (i == 0)
            b += (2.0 * r * r);
        else
            b += (r * r);
    }

    // Modulate sinc shape for each output coordinate if science variable is vector-valued
    for (int l = 0; l < dim; l++)
    {
        // a1 = sinc(a*(1+l)); b1 = sinc(b*(1+l))
        T a1 = (a == 0.0 ? 1.0*(1+l) : sin(a*(1+l)) / a);
        T b1 = (b == 0.0 ? 1.0*(1+l) : sin(b*(1+l)) / b);
        output_pt(l) = args.s[k] * (a1 + b1);               // scale entire science variable by s[k]
    }

    return;
}

// evaluate n-d poly-sinc function version 3 (differs from version 2 in definition of a)
template<typename T>
void polysinc3(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    int dim = output_pt.size();

    // a = sqrt(x^2 + y^2 + z^2 + ...)
    // b = 2(x - 2)^2 + (y + 2)^2 + (z - 2)^2 + ...
    T a = 0.0;
    T b = 0.0;
    for (auto i = 0; i < domain_pt.size(); i++)
    {
        T s, r;
        s = domain_pt(i);
        if (i % 2 == 0)
        {
            r = domain_pt(i) - 2.0;
        }
        else
        {
            r = domain_pt(i) + 2.0;
        }
        a += (s * s);
        if (i == 0)
            b += (2.0 * r * r);
        else
            b += (r * r);
    }
    a = sqrt(a);

    // Modulate sinc shape for each output coordinate if science variable is vector-valued
    for (int l = 0; l < dim; l++)
    {
        // a1 = sinc(a*(1+l)); b1 = sinc(b*(1+l))
        T a1 = (a == 0.0 ? 1.0*(1+l) : sin(a*(1+l)) / a);
        T b1 = (b == 0.0 ? 1.0*(1+l) : sin(b*(1+l)) / b);
        output_pt(l) = args.s[k] * (a1 + b1);               // scale entire science variable by s[k]
    }

    return;
}

// evaluate Marschner-Lobb function [Marschner and Lobb, IEEE VIS, 1994]
// only for a 3d domain
// using args f[0] and s[0] for f_M and alpha, respectively, in the paper
template<typename T>
void ml(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    if (domain_pt.size() != 3 || output_pt.size() != 1)
    {
        fprintf(stderr, "Error: Marschner-Lobb function is only defined for a 3d domain and scalar output.\n");
        exit(0);
    }
    T fm           = args.f[0];
    T alpha        = args.s[0];
//         T fm = 6.0;
//         T alpha = 0.25;
    T x            = domain_pt(0);
    T y            = domain_pt(1);
    T z            = domain_pt(2);

    T rad       = sqrt(x * x + y * y + z * z);
    T rho       = cos(2 * M_PI * fm * cos(M_PI * rad / 2.0));
    T retval    = (1.0 - sin(M_PI * z / 2.0) + alpha * (1.0 + rho * sqrt(x * x + y * y))) / (2 * (1.0 + alpha));

    output_pt(0) = retval;
    return;
}

// evaluate f16 function
template<typename T>
void f16(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    if (domain_pt.size() != 2 || output_pt.size() != 1)
    {
        fprintf(stderr, "Error: f16 function is only defined for a 2d domain and scalar output.\n");
        exit(0);
    }

    T retval =
        (pow(domain_pt(0), 4)                        +
        pow(domain_pt(1), 4)                        +
        pow(domain_pt(0), 2) * pow(domain_pt(1), 2) +
        domain_pt(0) * domain_pt(1)                 ) /
        (pow(domain_pt(0), 3)                        +
        pow(domain_pt(1), 3)                        +
        4                                           );

    output_pt(0) = retval;
    return;
}

// evaluate f17 function
template<typename T>
void f17(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    if (domain_pt.size() != 3 || output_pt.size() != 1)
    {
        fprintf(stderr, "Error: f17 function is only defined for a 3d domain and scalar output.\n");
        exit(0);
    }

    T E         = domain_pt(0);
    T G         = domain_pt(1);
    T M         = domain_pt(2);
    T gamma     = sqrt(M * M * (M * M + G * G));
    T kprop     = (2.0 * sqrt(2.0) * M * G * gamma ) / (M_PI * sqrt(M * M + gamma));
    T retval    = kprop / ((E * E - M * M) * (E * E - M * M) + M * M * G * G);

    output_pt(0) = retval;
    return;
}

// evaluate f18 function
template<typename T>
void f18(const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs&  args, int k)
{
    if (domain_pt.size() != 4 || output_pt.size() != 1)
    {
        fprintf(stderr, "Error: f18 function is only defined for a 4d domain and scalar output.\n");
        exit(0);
    }

    T x1        = domain_pt(0);
    T x2        = domain_pt(1);
    T x3        = domain_pt(2);
    T x4        = domain_pt(3);
    T retval    = (atanh(x1) + atanh(x2) + atanh(x3) + atanh(x4)) / ((pow(x1, 2) - 1) * pow(x2, -1));

    output_pt(0) = retval;
    return;
}

template<typename T>
void evaluate_function(string fun, const VectorX<T>& domain_pt, VectorX<T>& output_pt, DomainArgs& args, int k)
{
    if (fun == "sine")          return sine(        domain_pt, output_pt, args, k);
    else if (fun == "cosine")   return cosine(      domain_pt, output_pt, args, k);
    else if (fun == "ncosp1")   return ncosp1(      domain_pt, output_pt, args, k);
    else if (fun == "sinc")     return sinc(        domain_pt, output_pt, args, k);
    else if (fun == "psinc1")   return polysinc1(   domain_pt, output_pt, args, k);
    else if (fun == "psinc2")   return polysinc2(   domain_pt, output_pt, args, k);
    else if (fun == "psinc3")   return polysinc3(   domain_pt, output_pt, args, k);
    else if (fun == "ml")       return ml(          domain_pt, output_pt, args, k);
    else if (fun == "f16")      return f16(         domain_pt, output_pt, args, k);
    else if (fun == "f17")      return f17(         domain_pt, output_pt, args, k);
    else if (fun == "f18")      return f18(         domain_pt, output_pt, args, k);
    else
    {
        cerr << "Invalid function name in evaluate_function. Aborting." << endl;
        exit(0);
    }

    return;
}

#endif // _MFA_BLOCK