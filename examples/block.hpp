//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_BLOCK
#define _MFA_BLOCK

#include    <random>
#include    <stdio.h>
#include    <mfa/types.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/block_base.hpp>
#include    <diy/master.hpp>
#include    <diy/reduce-operations.hpp>
#include    <diy/decomposition.hpp>
#include    <diy/assigner.hpp>
#include    <diy/io/block.hpp>
#include    <diy/io/bov.hpp>
#include    <diy/pick.hpp>
#include    <Eigen/Dense>

#include    "domain_args.hpp"
#include    "example_signals.hpp"

using namespace std;



// 3d point or vector
struct vec3d
{
    float x, y, z;
    float mag() { return sqrt(x*x + y*y + z*z); }
    vec3d(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    vec3d() {}
};



// block
template <typename T, typename U=T>
struct Block : public BlockBase<T, U>
{
    using Base = BlockBase<T, U>;
    using Base::dom_dim;
    using Base::pt_dim;
    using Base::core;
    using Base::bounds;
    using Base::domain;
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
        void add_int(                                   // add the block to the decomposition
            int                 gid,                // block global id
            const Bounds<int>&  core,               // block bounds without any ghost added
            const Bounds<int>&  bounds,             // block bounds including any ghost region added
            const Bounds<int>&  domain,             // global data bounds
            const RCLink<int>&  link,               // neighborhood
            diy::Master&        master,             // diy master
            int                 dom_dim,            // domain dimensionality
            int                 pt_dim)             // point dimensionality
    {
        mfa::add_int<Block, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim);
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
        auto unitrange = [&](){return u_dist(df_gen);};
        auto randpoint = [&](){return (VectorX<T>::NullaryExpr(gdim, unitrange).array() * (core_maxs - core_mins).array()).matrix() + core_mins;};

        // Fill domain with randomly distributed points
        double sparsity = a->t;
        double keep_frac = 1;
        if (sparsity < 0 || sparsity > 1)
        {
            cerr << "Invalid value of void density" << endl;
            exit(1);
        }
        if (cp.gid() == 0)
            cout << "Void Sparsity: " << (sparsity)*100 << "%" << endl;

        const size_t nvoids = 4;
        const double radii_frac = 1.0/8.0;   // fraction of domain width to set as void radius
        T radius = radii_frac * (core_maxs - core_mins).minCoeff();
        vector<VectorX<T>> centers(nvoids);
        for (size_t nv = 0; nv < nvoids; nv++) // Randomly generate the centers of each void
        {
            centers[nv] = randpoint();
        }

        VectorX<T> dom_pt(gdim);
        for (size_t j = 0; j < input->npts; j++)
        {
            // Generate a random point, randomly discard if within a certain radius of a void
            while (true)
            {
                keep_frac = 1;
                dom_pt = randpoint();

                for (size_t nv = 0; nv < nvoids; nv++)
                {
                    if ((dom_pt - centers[nv]).norm() < radius)
                    {
                        keep_frac = sparsity;
                        break;
                    }
                }

                if (u_dist(df_gen) <= keep_frac)
                    break;
            }

            // Add point to Input
            for (size_t k = 0; k < gdim; k++)
            {
                input->domain(j,k) = dom_pt(k);
            }

            // Evaluate function at point and add to Input
            for (auto k = 0; k < nvars; k++)
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
        mfa::print_bbox(core_mins, core_maxs, "Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "Bounds");
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
            p0(i) = this->core_mins(i) - nghost_pts * d(i);
            // decide overlap in each direction; they should be symmetric for neighbors
            // so if block a overlaps block b, block b overlaps a the same area
            this->overlaps(i) = nghost_pts * d(i);
            // max direction
            nghost_pts = floor((bounds_maxs(i) - core_maxs(i)) / d(i));
            ndom_pts(i) += nghost_pts;
            // tot_ndom_pts *= ndom_pts(i);
            T m2 = nghost_pts * d(i);
            if (m2 > this->overlaps(i))
                this->overlaps(i) = m2;
#ifdef BLEND_VERBOSE
            std::cout <<" dir: i " << i << " core:" << this->core_mins(i)  << " " <<
                     this->core_maxs(i) << " d(i) =" << d(i) << " overlap: "<< this->overlaps(i) << "\n";
#endif
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
                T x = input->domain(j, 0);
                T y = input->domain(j, 1);
                input->domain(j, 0) += a->t * sin(y);
                input->domain(j, 1) += a->t * sin(x);
            }
        }

        // optional rotation of the domain
        if (a->r && gdim >= 2)
        {
            for (auto j = 0; j < input->domain.rows(); j++)
            {
                T x = input->domain(j, 0);
                T y = input->domain(j, 1);
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
        mfa::print_bbox(core_mins, core_maxs, "Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "Bounds");
    }

    // input a given data buffer into the correct format for encoding
    void input_1d_data(
            T*                  data,
            DomainArgs&         args)
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
        this->vars[0].max_dim = this->vars[0].min_dim;
        VectorXi ndom_pts(this->dom_dim);
        this->bounds_mins.resize(this->pt_dim);
        this->bounds_maxs.resize(this->pt_dim);
        for (int i = 0; i < this->dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }

        // construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, pt_dim, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, pt_dim, tot_ndom_pts);

        // rest is hard-coded for 1d

        size_t n = 0;
        for (size_t i = 0; i < tot_ndom_pts; i++)
        {
            input->domain(i, 0) = data[n++];
            input->domain(i, 1) = data[n++];
        }

        // find extents
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            if (i == 0 || input->domain(i, 0) < bounds_mins(0))
                bounds_mins(0) = input->domain(i, 0);
            if (i == 0 || input->domain(i, 1) < bounds_mins(1))
                bounds_mins(1) = input->domain(i, 1);
            if (i == 0 || input->domain(i, 0) > bounds_maxs(0))
                bounds_maxs(0) = input->domain(i, 0);
            if (i == 0 || input->domain(i, 1) > bounds_maxs(1))
                bounds_maxs(1) = input->domain(i, 1);
        }
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        // create the model
        input->init_params();
        this->mfa = new mfa::MFA<T>(dom_dim);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
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
                input->domain(ofst + i, 4) *= a->s[0];
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

        FILE *fd = fopen(args.infile.c_str(), "r");
        if (!fd)
        {
            cerr << "ERROR: Could not read file '" << args.infile << "'. Exiting." << endl;
            exit(1);
        }

        // build PointSet
        float val = 0;
        for (int i = 0; i < input->npts; i++)
        {
            for (int j = 0; j < input->pt_dim; j++)
            {
                fscanf(fd, "%f", &val);
                input->domain(i, j) = val;
            }
        }

        // compute bounds in each dimension
        // TODO WARNING use of bounds when dom_dim != geom_dim is not well-defined!
        if (args.min.size() != dom_dim || args.max.size() != dom_dim)
        {
            cerr << "ERROR: Invalid size of DomainArgs::min or DomainArgs::max. Exiting." << endl;
            exit(1);
        }
        for (int i = 0; i < dom_dim; i++)
        {
            // TODO If not multiblock? We need to disambiguate core_mins, args.min, and domain_mins
            core_mins(i) = args.min[i];
            core_maxs(i) = args.max[i];
        }
        bounds_mins.head(dom_dim) = core_mins;
        bounds_maxs.head(dom_dim) = core_maxs;
        bounds_mins.tail(pt_dim-dom_dim) = input->domain.rightCols(pt_dim-dom_dim).colwise().minCoeff();
        bounds_maxs.tail(pt_dim-dom_dim) = input->domain.rightCols(pt_dim-dom_dim).colwise().maxCoeff();

        input->set_domain_params(core_mins, core_maxs);

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        mfa::print_bbox(core_mins, core_maxs, "Core");
        mfa::print_bbox(bounds_mins, bounds_maxs, "Bounds");
        // cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
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
        vector<int>&                        grid,               // size of regular grid
        string                              fun,                // analytical function name
        vector<T>&                          L1,                 // (output) L-1 norm
        vector<T>&                          L2,                 // (output) L-2 norm
        vector<T>&                          Linf,               // (output) L-infinity norm
        DomainArgs&                         args,               // input args
        const std::function<void(const VectorX<T>&, VectorX<T>&, DomainArgs&, int)>& f = {},
        vector<T>                           subset_mins = vector<T>(),
        vector<T>                           subset_maxs = vector<T>() ) // (optional) subset of the domain to consider for errors
    {
        mfa::PointSet<T>* unused = nullptr;
        analytical_error_field(cp, grid, fun, L1, L2, Linf, args, unused, unused, unused, f, subset_mins, subset_maxs);
    }

    // Compute error field on a regularly spaced grid of points. The size of the grid
    // is given by args.ndom_pts. Error metrics are saved in L1, L2, Linf. The fields 
    // of the exact, approximate, and residual data are save to PointSets.
    void analytical_error_field(
        const diy::Master::ProxyWithLink&   cp,
        vector<int>&                        grid,               // size of regular grid
        string                              fun,                // analytical function name
        vector<T>&                          L1,                 // (output) L-1 norm
        vector<T>&                          L2,                 // (output) L-2 norm
        vector<T>&                          Linf,               // (output) L-infinity norm
        DomainArgs&                         args,               // input args
        mfa::PointSet<T>*&                  exact_pts,          // PointSet to contain analytical signal
        mfa::PointSet<T>*&                  approx_pts,         // PointSet to contain approximation
        mfa::PointSet<T>*&                  error_pts,          // PointSet to contain errors
        const std::function<void(const VectorX<T>&, VectorX<T>&, DomainArgs&, int)>& f = {},
        vector<T>                           subset_mins = vector<T>(),
        vector<T>                           subset_maxs = vector<T>() ) // (optional) subset of the domain to consider for errors
    {
        int nvars = mfa->nvars();
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
            test_pts(i) = grid[i];
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

                // Get exact value. If 'f' is non-NULL, ignore 'fun'
                if (f)
                    f(dom_pt, true_pt, args, k);
                else
                    evaluate_function(fun, dom_pt, true_pt, args, k);
                    
                // Get approximate value
                pt_it.var_coords(k, test_pt);

                // Update error field
                residual = (true_pt - test_pt).cwiseAbs();
                for (int j = 0; j < error_pts->var_dim(k); j++)
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

    void readBOV(
        const diy::Master::ProxyWithLink& cp,
              string         filename,
        const vector<int>&   shape,
              int            vecSize,
              vector<float>& data,
              bool           fileOrderC = false)
    {
        vector<int> readShape(3);
        Bounds<int> readBounds(3);

        // Switch domain shape to C-ordering for parallel read
        if (fileOrderC)
        {
            readShape = shape;
            readBounds = bounds;
        }
        else 
        {
            for (int i = 0; i < 3; i++)
            {
                readShape[i] = shape[2-i];
                readBounds.min[i] = bounds.min[2-i];
                readBounds.max[i] = bounds.max[2-i];
            }
        }
        
        readBounds.min[2] *= vecSize;
        readBounds.max[2] *= vecSize;
        readBounds.max[2] += vecSize - 1;
        readShape[2] *= vecSize;

        int dataSize = 1;
        for (int i = 0; i < 3; i++)
        {
            dataSize *= (readBounds.max[i] - readBounds.min[i] + 1);
        }

        // int gid = cp.gid();
        diy::mpi::io::file in(cp.master()->communicator(), filename, diy::mpi::io::file::rdonly);
        diy::io::BOV reader(in, readShape);

        data.clear();
        data.resize(dataSize);
        reader.read(readBounds, &data[0], true);
    }

    void read_box_data_3d(   
        const diy::Master::ProxyWithLink&   cp,
              string                        infile,
        const vector<int>&                  shape,
              bool                          fileOrderC,
              int                           vecSize,
              MFAInfo&                      mfa_info)
    {
        // assumes one scalar science variable
        int nvars = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);

        // decide the actual dimension of the problem, looking at the starts and ends
        std::vector<int> mapDim;
        for (int i = 0; i < 3; i++) {
            if (core.min[i] < core.max[i]) {
                mapDim.push_back(i);
            }
        }
        if (mapDim.size() != dom_dim)
        {
            if (cp.gid() == 0)
            {
                cerr << "ERROR: Number of nontrivial dimensions does not match dom_dim. Exiting." << endl;
            }
            exit(1);
        }

        VectorXi ndom_pts(dom_dim);  // this will be local now, and used in def of mfa
        int tot_ndom_pts = 1;
        for (size_t j = 0; j < dom_dim; j++) {
            int dir = mapDim[j];
            int size_in_dir = -bounds.min[dir] + bounds.max[dir] + 1;
            tot_ndom_pts *= size_in_dir;
            ndom_pts(j) = size_in_dir;
        }

        // Construct point set to contain input
        VectorXi model_dims(2);
        model_dims(0) = dom_dim;
        model_dims(1) = pt_dim - dom_dim;
        input = new mfa::PointSet<T>(dom_dim, model_dims, tot_ndom_pts, ndom_pts);

        vector<float> data;
        readBOV(cp, infile, shape, vecSize, data, fileOrderC);

        if (dom_dim == 1) // 1d problem, the dimension would be x direction
        {
            int dir0 = mapDim[0];
            this->map_dir.push_back(dir0); // only one dimension, rest are not varying
            for (int i = 0; i < tot_ndom_pts; i++) {
                input->domain(i, 0) = bounds.min[dir0] + i;
                int idx = vecSize * i;
                float val = 0;
                for (int k = 0; k < vecSize; k++)
                    val += data[idx + k] * data[idx + k];
                val = sqrt(val);
                input->domain(i, 1) = val;
            }
        } 
        else if (dom_dim == 2) // 2d problem, second direction would be x, first would be y
        {
            if (!fileOrderC) {
                int n = 0;
                int idx = 0;
                int dir0 = mapDim[0]; // so now y would vary to 704 in 2d 1 block similar case for s3d (transpose)
                int dir1 = mapDim[1];
                // we do not transpose anymore
                this->map_dir.push_back(dir0);
                this->map_dir.push_back(dir1);
                for (int i = 0; i < ndom_pts(0); i++) {
                    for (int j = 0; j < ndom_pts(1); j++) {
                        n = j * ndom_pts(0) + i;
                        input->domain(n, 0) = bounds.min[dir0] + i; //
                        input->domain(n, 1) = bounds.min[dir1] + j;
                        float val = 0;
                        for (int k = 0; k < vecSize; k++)
                            val += data[idx + k] * data[idx + k];
                        val = sqrt(val);
                        input->domain(n, 2) = val;
                        idx += vecSize;
                    }
                }
            } 
            else 
            {
                // keep the order as Paraview, x would be the first that varies
                // corresponds to original implementation, which needs to transpose dimensions
                int n = 0;
                int idx = 0;
                int dir0 = mapDim[1]; // so x would vary to 704 in 2d 1 block similar case
                int dir1 = mapDim[0];
                this->map_dir.push_back(dir0);
                this->map_dir.push_back(dir1);
                for (int j = 0; j < ndom_pts(1); j++) {
                    for (int i = 0; i < ndom_pts(0); i++) {
                        input->domain(n, 1) = bounds.min[dir1] + j;
                        input->domain(n, 0) = bounds.min[dir0] + i;
                        float val = 0;
                        for (int l = 0; l < vecSize; l++)
                            val += data[idx + l] * data[idx + l];
                        input->domain(n, 2) = sqrt(val);
                        n++;
                        idx += vecSize;
                    }
                }
            }
        }
        else if (dom_dim == 3) 
        {
            if (!fileOrderC) {
                int n = 0;
                int idx = 0;
                this->map_dir.push_back(mapDim[0]);
                this->map_dir.push_back(mapDim[1]);
                this->map_dir.push_back(mapDim[2]);
                // last dimension would correspond to x, as in the 2d example
                for (int k = 0; k < ndom_pts(2); k++) {
                    for (int j = 0; j < ndom_pts(1); j++) {
                        for (int i = 0; i < ndom_pts(0); i++) {
                            input->domain(n, 0) = bounds.min[0] + i;
                            input->domain(n, 1) = bounds.min[1] + j;
                            input->domain(n, 2) = bounds.min[2] + k;
                            float val = 0;
                            for (int l = 0; l < vecSize; l++)
                                val += data[idx + l] * data[idx + l];
                            val = sqrt(val);
                            input->domain(n, 3) = val;
                            n++;
                            idx += vecSize;
                        }
                    }
                }
            } 
            else
            {
                int n = 0;
                int idx = 0;
                this->map_dir.push_back(mapDim[0]);
                this->map_dir.push_back(mapDim[1]);
                this->map_dir.push_back(mapDim[2]);
                // last dimension would correspond to x, as in the 2d example
                for (int i = 0; i < ndom_pts(0); i++) {
                    for (int j = 0; j < ndom_pts(1); j++) {
                        for (int k = 0; k < ndom_pts(2); k++) {
                            n = k * ndom_pts(0) * ndom_pts(1) + j * ndom_pts(0) + i;
                            input->domain(n, 0) = bounds.min[0] + i; // this now corresponds to x
                            input->domain(n, 1) = bounds.min[1] + j;
                            input->domain(n, 2) = bounds.min[2] + k;
                            float val = 0;
                            for (int l = 0; l < vecSize; l++)
                                val += data[idx + l] * data[idx + l];
                            input->domain(n, 3) = sqrt(val);
                            idx += vecSize;
                        }
                    }
                }
            }
        }
        input->set_domain_params();

        // Finalize block for encoding
        for (int i = 0; i < dom_dim; i++) {
            bounds_mins(i) = bounds.min[mapDim[i]];
            bounds_maxs(i) = bounds.max[mapDim[i]];
            core_mins(i) = core.min[mapDim[i]];
            core_maxs(i) = core.max[mapDim[i]];
            // decide overlap in each direction; they should be symmetric for neighbors
            // so if block a overlaps block b, block b overlaps a the same area
            overlaps(i) = fabs(core_mins(i) - bounds_mins(i));
            T m2 = fabs(bounds_maxs(i) - core_maxs(i));
            if (m2 > overlaps(i))
                overlaps(i) = m2;
        }

        // set bounds_min/max for science variable (last coordinate)
        bounds_mins(dom_dim) = input->domain.col(dom_dim).minCoeff();
        bounds_maxs(dom_dim) = input->domain.col(dom_dim).maxCoeff();

        this->setup_MFA(cp, mfa_info);
    }
};

template<typename U>
void max_err_cb(Block<real_t, U> *b,                  // local block
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


#endif // _MFA_BLOCK