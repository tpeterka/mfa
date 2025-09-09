//--------------------------------------------------------------
// One diy block that can handle ray integrals
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_RAY_BLOCK
#define _MFA_RAY_BLOCK

#include "block.hpp"

template <typename T>
struct RayBlock : public Block<T>
{
    using Base = typename Block<T>::template BlockBase<T>;
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
        void* create()              { return mfa::create<RayBlock>(); }

    static
        void destroy(void* b)       { mfa::destroy<RayBlock>(b); }

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
        mfa::add<RayBlock, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);
    }

    static
        void save(const void* b_, diy::BinaryBuffer& bb)    { mfa::save<RayBlock, T>(b_, bb); }
    static
        void load(void* b_, diy::BinaryBuffer& bb)          { mfa::load<RayBlock, T>(b_, bb); }

    RayBlock() :
        ray_mfa(nullptr),
        ray_input(nullptr),
        ray_approx(nullptr),
        ray_errs(nullptr)
    { }

    ~RayBlock()
    {
        delete ray_mfa;
        delete ray_input;
        delete ray_approx;
        delete ray_errs;
    }

    int                 ray_dom_dim;                 // dom_dim of the extended model (i.e. dom_dim+1)
    mfa::MFA<T>         *ray_mfa;
    mfa::PointSet<T>    *ray_input;                 // input data
    mfa::PointSet<T>    *ray_approx;                // output data
    mfa::PointSet<T>    *ray_errs;                  // error field

    VectorX<T>          ray_bounds_mins;            // local domain minimum corner
    VectorX<T>          ray_bounds_maxs;            // local domain maximum corner
    VectorX<T>          ray_core_mins;              // local domain minimum corner w/o ghost
    VectorX<T>          ray_core_maxs;              // local domain maximum corner w/o ghost

    mfa::Stats<T>       stats;
    mfa::Stats<T>       ray_stats;

    const double pi = 3.14159265358979;
    T r_lim{0};

    int trap_samples{0};        // number of samples used in trapezoid rule (used for testing)

    void get_box_intersections(
        T alpha,
        T rho,
        T& x0,
        T& y0,
        T& x1,
        T& y1,
        const VectorX<T>& mins,
        const VectorX<T>& maxs) const
    {
        T xl = mins(0);
        T xh = maxs(0);
        T yl = mins(1);
        T yh = maxs(1);

        T yh_int = (rho - yh * sin(alpha)) / cos(alpha);
        T yl_int = (rho - yl * sin(alpha)) / cos(alpha);
        T xh_int = (rho - xh * cos(alpha)) / sin(alpha);
        T xl_int = (rho - xl * cos(alpha)) / sin(alpha);

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
                throw mfa::MFAError("invalid state 1 in get_box_intersections");
            }
        }
        else if (yl_int >= xl && yl_int <= xh)  // enter or exit bottom
        {
            if (yh_int >= xl && yh_int <= xh)   // enter/exit top & bottom
            {
                if (sin(alpha) == 0)    // vertical line case (should have been handled above)
                {
                    fmt::print(stderr, "WARNING: invalid state 6, this should not happen\n");
                    x0 = yl_int;
                    y0 = yl;
                    x1 = yh_int;
                    y1 = yh;
                }
                else if (sin(alpha) == 0 && alpha > 0)     // opposite vertical line case (should have been handled above)
                {
                    fmt::print(stderr, "WARNING: invalid state 7, this should not happen\n");
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
                    throw mfa::MFAError("invalid state 2 in get_box_intersections");
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
                throw mfa::MFAError("invalid state 3 in get_box_intersections");
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
                throw mfa::MFAError("invalid state 4 in get_box_intersections");
            }
        }
        else
        {
            x0 = 0;
            y0 = 0;
            x1 = 0;
            y1 = 0;
        }
    }

    // Check if a point p is in the original domain of the data
    bool in_domain2d(const VectorX<T>& p)
    {
        return (p(0) >= bounds_mins(0)) && (p(0) <= bounds_maxs(0)) && (p(1) >= bounds_mins(1)) && (p(1) <= bounds_maxs(1));
    }

    bool in_domain2d(T x, T y)
    {
        return (x >= bounds_mins(0)) && (x <= bounds_maxs(0)) && (y >= bounds_mins(1)) && (y <= bounds_maxs(1));
    }

    bool in_domain3d(const VectorX<T>& p)
    {
          return (p(0) >= bounds_mins(0) && p(0) <= bounds_maxs(0) &&
                p(1) >= bounds_mins(1) && p(1) <= bounds_maxs(1) &&
                p(2) >= bounds_mins(2) && p(2) <= bounds_maxs(2));
    }

    void compute_bounds()
    {
        if (dom_dim == 2) compute_bounds2d();
        else if (dom_dim == 3) compute_bounds3d();
        else throw mfa::MFAError("Unsupported dimension in RayBlock::compute_bounds()");
    }

    // extents of domain in physical space
    void compute_bounds2d()
    {
        // Get maximal distance of domain edge from origin
        // WARNING: we are assuming the domain is (rougly) centered at the origin!
        double max_radius = max(bounds_mins.cwiseAbs().maxCoeff(), bounds_maxs.cwiseAbs().maxCoeff());
        r_lim = max_radius * 1.5;

        // Set extents of rotated model
        ray_bounds_mins.resize(pt_dim + 1);
        ray_bounds_maxs.resize(pt_dim + 1);
        ray_bounds_mins(0) = 0;
        ray_bounds_maxs(0) = 1;
        ray_bounds_mins(1) = -r_lim;
        ray_bounds_maxs(1) = r_lim;
        ray_bounds_mins(2) = 0;
        ray_bounds_maxs(2) = pi;
        for (int i = dom_dim; i < pt_dim; i++)
        {
            ray_bounds_mins(i+ray_dom_dim-dom_dim) = bounds_mins(i);
            ray_bounds_maxs(i+ray_dom_dim-dom_dim) = bounds_maxs(i);
        }
        ray_core_mins = ray_bounds_mins.head(dom_dim+1);
        ray_core_maxs = ray_bounds_maxs.head(dom_dim+1);
    }

    void compute_bounds3d()
    {
        // Get maximal distance of domain edge from origin
        // WARNING: we are assuming the domain is (rougly) centered at the origin!
        double max_radius = max(bounds_mins.cwiseAbs().maxCoeff(), bounds_maxs.cwiseAbs().maxCoeff());
        r_lim = max_radius * 1.75; 

        // Set extents of rotated model
        ray_bounds_mins.resize(pt_dim + ray_dom_dim-dom_dim);
        ray_bounds_maxs.resize(pt_dim + ray_dom_dim-dom_dim);
        ray_bounds_mins(0) = 0;
        ray_bounds_maxs(0) = 1;
        ray_bounds_mins(1) = -r_lim;
        ray_bounds_maxs(1) = r_lim;
        ray_bounds_mins(2) = -r_lim;
        ray_bounds_maxs(2) = r_lim;
        ray_bounds_mins(3) = 0;
        ray_bounds_maxs(3) = pi;
        ray_bounds_mins(4) = 0;
        ray_bounds_maxs(4) = pi;
        for (int i = dom_dim; i < pt_dim; i++)
        {
            ray_bounds_mins(i+ray_dom_dim-dom_dim) = bounds_mins(i);
            ray_bounds_maxs(i+ray_dom_dim-dom_dim) = bounds_maxs(i);
        }
        ray_core_mins = ray_bounds_mins.head(ray_dom_dim);
        ray_core_maxs = ray_bounds_maxs.head(ray_dom_dim);
    }

    void sample_rotations(mfa::PointSet<T>& rotation_space, const vector<int>& rotation_samples)
    {
        if (dom_dim == 2) sample_rotations2d(rotation_space, rotation_samples);
        else if (dom_dim == 3) sample_rotations3d(rotation_space, rotation_samples);
    }

    // rotation samples = {n_samples, n_rho, n_alpha}
    void sample_rotations2d(mfa::PointSet<T>& rotation_space, const vector<int>& rotation_samples)
    {
        int n_samples = rotation_samples[0];
        int n_rho = rotation_samples[1];
        int n_alpha = rotation_samples[2];
        VectorX<T> param(dom_dim);
        VectorX<T> outpt(pt_dim);

        // Increments of rho and alpha
        double dr = r_lim * 2 / (n_rho-1);
        double da = pi / (n_alpha-1);
        double dt = 1.0 / (n_samples-1);

        // fill ray data set
        double alpha    = 0;   // angle of rotation
        double rho      = -r_lim;
        double t        = 0;
        VectorX<T> s0(2);
        VectorX<T> s1(2);
        VectorX<T> p(2);
        for (int ia = 0; ia < n_alpha; ia++)
        {
            alpha = ia * da;
            s0 << cos(alpha), sin(alpha);
            s1 << -1*sin(alpha), cos(alpha);

            for (int ir = 0; ir < n_rho; ir++)
            {
                rho = -r_lim + ir * dr;

                for (int is = 0; is < n_samples; is++)
                {
                    t = is * dt;

                    int idx = ia*n_rho*n_samples + ir*n_samples + is;
                    ray_input->domain(idx, 0) = t;
                    ray_input->domain(idx, 1) = rho;
                    ray_input->domain(idx, 2) = alpha;

                    p = rho*s0 + 2*r_lim*(t-0.5)*s1;

                    // If this point is not in the original domain
                    if (!in_domain2d(p))
                    {
                        // add dummy value, which will never be queried
                        ray_input->domain(idx, ray_dom_dim) = 0;
                    }
                    else    // point is in domain, decode value from existing MFA
                    {
                        param(0) = (p(0) - bounds_mins(0)) / (bounds_maxs(0) - bounds_mins(0));
                        param(1) = (p(1) - bounds_mins(1)) / (bounds_maxs(1) - bounds_mins(1));

                        // Truncate to [0,1] in the presence of small round-off errors
                        param(0) = param(0) < 0 ? 0 : param(0);
                        param(1) = param(1) < 0 ? 0 : param(1);
                        param(0) = param(0) > 1 ? 1 : param(0);
                        param(1) = param(1) > 1 ? 1 : param(1);

                        // Todo assemble Param object and then Decode once per angular setup
                        mfa->Decode(param, outpt); // TODO change this fast decode and change the size of outpt!
                        ray_input->domain.block(idx, ray_dom_dim, 1, pt_dim - dom_dim) = outpt.tail(pt_dim - dom_dim).transpose();
                    }
                }
            }
        }
    }

    // rotation samples = {n_samples, n_rho, n_nu, n_theta, n_phi}
    void sample_rotations3d(mfa::PointSet<T>& rotation_space, const vector<int>& rotation_samples)
    {
        int n_samples = rotation_samples[0];
        int n_rho = rotation_samples[1];
        int n_nu = rotation_samples[2];
        int n_theta = rotation_samples[3];
        int n_phi = rotation_samples[4];
        VectorX<T> param(dom_dim);
        VectorX<T> outpt(pt_dim-dom_dim);

        // Increments of rho and alpha
        double dt = 1.0 / (n_samples-1);
        double drho = r_lim * 2 / (n_rho-1);
        double dnu = r_lim * 2 / (n_nu-1);
        double dtheta = pi / (n_theta-1);
        double dphi = pi / (n_phi-1);

        // spherical unit vectors
        VectorX<T> s0(3);
        VectorX<T> s1(3);
        VectorX<T> s2(3);

        // fill ray data set
        double t        = 0;
        double rho      = -r_lim;
        double nu       = -r_lim;
        double theta    = 0;    // azimuthal angle
        double phi      = 0;    // polar angle
        VectorX<T> extentsRecip = (core_maxs - core_mins).cwiseInverse(); // performance optimization
        VectorX<T> p(3);

        mfa::Decoder<T> decoder(mfa->var(0), 0);
        mfa::FastDecodeInfo<T> di(decoder);
        for (int iphi = 0; iphi < n_phi; iphi++)
        {
            for (int itheta = 0; itheta < n_theta; itheta++)
            {
                phi = iphi * dphi;
                theta = itheta * dtheta;

                // Compute spherical basis for this orientation
                s0 << cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi);
                s1 << -1*sin(theta), cos(theta), 0;
                s2 << cos(theta)*cos(phi), sin(theta)*cos(phi), -1*sin(phi);

                for (int inu = 0; inu < n_nu; inu++)
                {
                    for (int irho = 0; irho < n_rho; irho++)
                    {
                        for (int it = 0; it < n_samples; it++)
                        {
                            nu = -r_lim + inu * dnu;
                            rho = -r_lim + irho * drho;
                            t = it * dt;

                            p = rho*s0 + nu*s1 + 2*r_lim*(t-0.5)*s2;

                            int idx = it + n_samples*(irho + n_rho*(inu + n_nu*(itheta + n_theta*iphi)));

                            ray_input->domain(idx, 0) = t;
                            ray_input->domain(idx, 1) = rho;
                            ray_input->domain(idx, 2) = nu;
                            ray_input->domain(idx, 3) = theta;
                            ray_input->domain(idx, 4) = phi;

                            if (!in_domain3d(p))
                            {
                                ray_input->domain(idx, 5) = 0;
                            }
                            else
                            {
                                param = (p-core_mins).cwiseProduct(extentsRecip);

                                // Truncate to [0,1] in the presence of small round-off errors
                                param(0) = param(0) < 0 ? 0 : param(0);
                                param(1) = param(1) < 0 ? 0 : param(1);
                                param(2) = param(2) < 0 ? 0 : param(2);
                                param(0) = param(0) > 1 ? 1 : param(0);
                                param(1) = param(1) > 1 ? 1 : param(1);
                                param(2) = param(2) > 1 ? 1 : param(2);

                                // Todo assemble Param object and then Decode once per angular setup
                                decoder.FastVolPt(param, outpt, di, mfa->var(0).tmesh.tensor_prods[0]);

                                ray_input->domain.block(idx, ray_dom_dim, 1, pt_dim - dom_dim) = outpt.tail(pt_dim - dom_dim).transpose();
                            }
                        }
                    }
                }
            }
        }
    }

    // ONLY 2d AT THE MOMENT
    // precondition: Block already contains a fully encoded MFA
    void create_ray_model(
        const       diy::Master::ProxyWithLink& cp,
        mfa::MFAInfo& mfa_info,
        DomainArgs& args,
        const vector<int>& ray_samples,
        const vector<int>& ray_nctrl)
    {
        // Update dimensionality
        ray_dom_dim = 2*dom_dim - 1;        // 2d --> 3d, 3d --> 5d
        VectorXi new_mdims = mfa->model_dims();
        new_mdims[0] = ray_dom_dim;  

        // Sanity checks for input
        if (ray_samples.size() != ray_dom_dim)
        {
            throw mfa::MFAError(fmt::format("Incorrect dimension for ray_samples, ray_samples.size()={}, ray_dom_dim={}", ray_samples.size(), ray_dom_dim));
        }
        if (ray_nctrl.size() != ray_dom_dim)
        {
            throw mfa::MFAError(fmt::format("Incorrect dimension for ray_nctrl, ray_nctrl.size()={}, ray_dom_dim={}", ray_nctrl.size(), ray_dom_dim));
        }
        for (int i = 0; i < ray_dom_dim; i++)
        {
            if (ray_samples[i] == 0)
            {
                throw mfa::MFAError(fmt::format("Did not set ray_samples[{}].", i));
            }
            if (ray_nctrl[i] == 0)
            {
                throw mfa::MFAError(fmt::format("Did not set ray_nctrl[{}].", i));
            }
        }

        // Set up domain to sample in rotation space
        VectorXi ndom_pts(ray_dom_dim);
        for (int i = 0; i < ray_dom_dim; i++)
        {
            ndom_pts(i) = ray_samples[i];
        }
        ray_input = new mfa::PointSet<T>(ray_dom_dim, new_mdims, ndom_pts.prod(), ndom_pts);

        // Sample the rotation space
        compute_bounds();                               // set r_lim, ray_core, and ray_bounds
        sample_rotations(*ray_input, ray_samples);     // fill ray_input
        ray_input->set_bounds(ray_core_mins, ray_core_maxs);
        ray_input->set_domain_params();

        // Set nctrl_pts, degree for variables
        VectorXi p(ray_dom_dim);
        VectorXi nctrl_pts(ray_dom_dim);
        for (int i = 0; i < ray_dom_dim; i++)
        {
            nctrl_pts(i) = ray_nctrl[i];
        }

        // Create empty Ray MFA
        int verbose = mfa_info.verbose && cp.master()->communicator().rank() == 0; 
        ray_mfa = new mfa::MFA<T>(ray_dom_dim, verbose);

        // Set up geometry and variable models
        ray_mfa->AddGeometry(ray_dom_dim);
        for (auto i = 0; i< mfa->nvars(); i++)
        {
            // set ray model degree to minimum degree of original model
            int min_p = mfa->var(i).p.minCoeff();
            p = min_p * VectorXi::Ones(ray_dom_dim);

            ray_mfa->AddVariable(p, nctrl_pts, 1);
        }

        // Encode ray model. 
        ray_mfa->FixedEncodeGeom(*ray_input, false);
        ray_mfa->RayEncode(0, *ray_input);

        // // // --------- Decode and compute errors --------- //
        // fmt::print(stderr, "Computing errors on uniform grid...\n");
        // mfa::PointSet<T>* unused = nullptr;
        // analytical_ray_error_field(cp, ray_mfa, ndom_pts, "sine", args, unused, ray_approx, ray_errs);
        // delete unused;
        // fmt::print(stderr, "done.\n");
    }

    // Convert (t, rho, theta) to (x, y) and return true if the x,y coords are in the original domain
    bool radon2cart(const VectorX<T>& radon_coords, VectorX<T>& cart_coords)
    {
        if (r_lim == 0)
        {
            throw mfa::MFAError("r_lim=0 in RayBlock::radon2cart()");
        }

        if (radon_coords.size() == 3) return radon2cart2d(radon_coords, cart_coords);
        else if (radon_coords.size() == 5) return radon2cart3d(radon_coords, cart_coords);
        else throw mfa::MFAError("Incompatible vector dimension in radon2cart");

        return false;
    }

    bool radon2cart2d(const VectorX<T>& radon_coords, VectorX<T>& cart_coords)
    {
        T t = radon_coords(0);
        T rho = radon_coords(1);
        T alpha = radon_coords(2);

        // TODO: remove allocation for basis vectors
        VectorX<T> s0(2);
        VectorX<T> s1(2);
        s0 << cos(alpha), sin(alpha);
        s1 << -1*sin(alpha), cos(alpha);
        cart_coords = rho*s0 + 2*r_lim*(t-0.5)*s1;

        return in_domain2d(cart_coords);
    }

    bool radon2cart3d(const VectorX<T>& radon_coords, VectorX<T>& cart_coords)
    {
        T t = radon_coords(0);
        T rho = radon_coords(1);
        T nu = radon_coords(2);
        T theta = radon_coords(3);
        T phi = radon_coords(4);

        VectorX<T> s0(3);
        VectorX<T> s1(3);
        VectorX<T> s2(3);
        s0 << cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi);
        s1 << -1*sin(theta), cos(theta), 0;
        s2 << cos(theta)*cos(phi), sin(theta)*cos(phi), -1*sin(phi);
        cart_coords = rho*s0 + nu*s1 + 2*r_lim*(t-0.5)*s2;

        return in_domain3d(cart_coords);
    }

    // Compute error field on a regularly spaced grid of points. The size of the grid
    // is given by args.ndom_pts. Error metrics are saved in L1, L2, Linf. The fields 
    // of the exact, approximate, and residual data are save to PointSets.
    void analytical_ray_error_field(
        const diy::Master::ProxyWithLink&   cp,
        mfa::MFA<T>*                        models,
        VectorXi&                           grid,               // size of regular grid
        string                              fun,                // analytical function name
        DomainArgs&                         args,               // input args
        mfa::PointSet<T>*&                  exact_pts,          // PointSet to contain analytical signal
        mfa::PointSet<T>*&                  approx_pts,         // PointSet to contain approximation
        mfa::PointSet<T>*&                  error_pts,          // PointSet to contain errors
        const std::function<void(const VectorX<T>&, VectorX<T>&, DomainArgs&, int)>& f = {}) // (optional) lambda expression for signal
    {
        // Initialize container for error statistics
        ray_stats.init(ray_input);

        // Free any existing memory at PointSet pointers
        if (exact_pts) fmt::print(stderr, "WARNING: Overwriting exact_pts pointset in analytical_ray_error_field()\n");
        if (approx_pts) fmt::print(stderr, "WARNING: Overwriting approx_pts pointset in analytical_ray_error_field()\n");
        if (error_pts) fmt::print(stderr, "WARNING: Overwriting error_pts pointset in analytical_ray_error_field()\n");
        delete exact_pts;
        delete approx_pts;
        delete error_pts;

        // Set up PointSets with grid parametrizations
        exact_pts = new mfa::PointSet<T>(models->dom_dim, models->model_dims(), grid.prod(), grid);
        approx_pts= new mfa::PointSet<T>(models->dom_dim, models->model_dims(), grid.prod(), grid);
        error_pts = new mfa::PointSet<T>(models->dom_dim, models->model_dims(), grid.prod(), grid);
        approx_pts->set_grid_params();

        // Decode on above-specified grid
        models->Decode(*approx_pts);

        // Copy geometric point coordinates into error and exact PointSets
        exact_pts->domain.leftCols(exact_pts->geom_dim()) = approx_pts->domain.leftCols(approx_pts->geom_dim());
        error_pts->domain.leftCols(error_pts->geom_dim()) = approx_pts->domain.leftCols(approx_pts->geom_dim());

        // Compute the analytical error at each point and accrue errors
        VectorX<T> xy_pt(mfa->geom_dim());
        VectorX<T> dom_pt(approx_pts->geom_dim());
        for (int k = 0; k < models->nvars(); k++)
        {
            VectorX<T> true_pt(approx_pts->var_dim(k));
            VectorX<T> test_pt(approx_pts->var_dim(k));
            VectorX<T> residual(approx_pts->var_dim(k));
            int num_pts = 0;

            for (auto pt_it = approx_pts->begin(), pt_end = approx_pts->end(); pt_it != pt_end; ++pt_it)
            {
                pt_it.geom_coords(dom_pt); // extract the geometry coordinates

                if (radon2cart(dom_pt, xy_pt))
                {
                    // Get exact value. If 'f' is non-NULL, ignore 'fun'
                    if (f)
                        f(xy_pt, true_pt, args, k);
                    else
                        evaluate_function(fun, xy_pt, true_pt, args, k);
                        
                    // Get approximate value
                    pt_it.var_coords(k, test_pt);

                    // Update error field
                    residual = (true_pt - test_pt).cwiseAbs();
                    for (int j = 0; j < error_pts->var_dim(k); j++)
                    {
                        error_pts->domain(pt_it.idx(), error_pts->var_min(k) + j) = residual(j);
                        exact_pts->domain(pt_it.idx(), exact_pts->var_min(k) + j) = true_pt(j);
                    }

                    // NOTE: For now, we are using the norm of the residual for all error statistics.
                    //       Is this the most appropriate way to measure errors norms of a vector field?
                    //       May want to consider revisiting this.
                    ray_stats.update(k, residual.norm());
                }
                else    // does not correspond to real xy point
                {
                    for (int j = 0; j < error_pts->var_dim(k); j++)
                    {
                        error_pts->domain(pt_it.idx(), error_pts->var_min(k) + j) = 0;
                        exact_pts->domain(pt_it.idx(), exact_pts->var_min(k) + j) = 0;
                    }
                }
            }
        }
    }

    // decode entire rotation space  at the same parameter locations as 'ray_input'
    void decode_ray_block(const diy::Master::ProxyWithLink& cp)
    {
        if (ray_approx)
        {
            fmt::print(stderr, "WARNING: Overwriting ray_approx pointset in RayBlock::decode_ray_block()\n");
            delete ray_approx;
        }
        ray_approx = new mfa::PointSet<T>(ray_input->params, ray_input->model_dims());  // Set decode params from ray_input params

        ray_mfa->Decode(*ray_approx);
    }

    void dualCoords3d(const VectorX<T>& a, const VectorX<T>& b, 
                    T& rho, T& nu, T& theta, T& phi) const
    {
        const double pi = 3.14159265358979;

        T hx = b(0) - a(0);
        T hy = b(1) - a(1);
        T hz = b(2) - a(2);
        T recipNorm = 1 / (sqrt(hx*hx + hy*hy + hz*hz));

        hx *= recipNorm;
        hy *= recipNorm;
        hz *= recipNorm;

        // phi = stableArcTan(1.0, abs(hz), sqrt(1 - hz*hz));
        // fmt::print(stderr, "phi1: {}\n", phi);
        phi = atan2(-hz, sqrt(1 - hz*hz)); // TODO check this  
        // fmt::print(stderr, "phi2: {}\n", phi);

        if (phi < 0)
        {
            // fmt::print(stderr, "adjusting phi\n");
            phi += pi;
        }

        // todo: add edge case when phi=pi/2
        theta = atan2(hy, hx);
        if (theta < 0)
        {
            // fmt::print(stderr, "adjusting theta\n");
            theta += pi;
            phi = pi - phi;
        }

        // Compute the projection of a into the plane spanned by s1, s2
        // (this plane is perpendicular to s3, and s3 || h.)
        T hDotA = hx*a(0) + hy*a(1) + hz*a(2); // dot product of h with a
        T px = a(0) - hDotA*hx;
        T py = a(1) - hDotA*hy;
        T pz = a(2) - hDotA*hz;

        // fmt::print(stderr, "p: {} {} {}\n", px, py, pz);
        // fmt::print(stderr, "p dot h: {}\n", px*hx + py*hy + pz*hz);

        // The vector p = (px, py, pz) is defined such that
        // p = rho*s0 + nu*s1 + 0*s2
        // 
        // Thus,
        // (cos(theta)sin(phi)   -sin(theta)    cos(theta)cos(phi)  )     ( rho )     ( px )
        // (                                                        )     (     )     (    )
        // (sin(theta)sin(phi)   cos(theta)     sin(theta)cos(phi)  )  x  ( nu  )  =  ( py )   (I)
        // (                                                        )     (     )     (    )
        // (cos(phi)                0              -sin(phi)        )     (  0  )     ( pz )
        // 
        // which implies
        // (cos(theta)   -sin(theta)        )     ( rho*sin(phi) )     ( px )
        // (                                )  x  (              )  =  (    )      (II)
        // (sin(theta)   cos(theta)         )     (     nu       )     ( py )
        //
        // Note: 
        // if sin(phi) == 0, then the first system is:
        // 
        // (    0      -sin(theta)     +/- cos(theta)  )     ( rho )     ( px )
        // (                                           )     (     )     (    )
        // (    0       cos(theta)     +/- sin(theta)  )  x  ( nu  )  =  ( py )   (I*)
        // (                                           )     (     )     (    )
        // (   +/- 1        0                0         )     (  0  )     ( pz )
        // 
        // and thus rho = cos(phi)*pz
        // and      nu  = -px/sin(theta)  and/or  nu = py/cos(theta)
        // 
        // All of these matrices are orthogonal, so A^{-1} = A^T
        // Thus either system can easily be solved without full matrix inversion. 
        // To avoid numerical errors, we choose which system to solve based on 
        // whether sin(phi) or cos(phi) is near zero.

        T CT = cos(theta);
        T ST = sin(theta);
        T SP = sin(phi);
        T CP = cos(phi);

        if (abs(SP) > 1e-12)  // solve system (II) above
        {
            rho = (CT*px + ST*py) / SP;
            nu = (-ST*px + CT*py);
         
            // rho = px * CT / SP - ST * py;
            // nu  = px * ST / SP + CT * py;
        }
        else  // solve system (I*)
        {
            rho = CP*pz;
            if (ST > 1e-6)
            {
                nu = -px/ST;
            }
            else
            {
                nu = py/CT;
            }
        }
        // else            // solve system (II) above
        // {
        //     rho = -ST*py + CT*pz/CP;
        //     nu  =  CT*py + ST*pz/CP;
        // }   
    }

    pair<T,T> dualCoords2d(const VectorX<T>& a, const VectorX<T>& b) const
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

            // to compute rho, use the fact that rho = x*cos(alpha) + y*sin(alpha),
            // then plug in the point (0, y_int), where y_int is the y-intercept of the line.
            // Thus, rho = y_int*sin(alpha).
            // Finally, the line equation can be written as y = y_int + m*x, so
            // y_int = a_y - m*a_x.
            rho = (a_y - m*a_x)/(sqrt(1+m*m));  // cos(atan(m)) = 1/sqrt(1+m*m), sin(pi/2-x) = cos(x)
        }

        return make_pair(alpha, rho);
    }

    T trapezoid(
        const   diy::Master::ProxyWithLink& cp,
        const   DomainArgs& d_args,
        const   VectorX<T>& a,
        const   VectorX<T>& b,
                int         nSamples = -1) const
    {
        mfa::Decoder<T> decoder(mfa->var(0), 0);     // nb. turning off verbose output when decoding single points

        T result = 0;
        VectorX<T> au(dom_dim);
        VectorX<T> bu(dom_dim);
        VectorX<T> du(dom_dim);
        VectorX<T> param1(dom_dim);
        VectorX<T> param2(dom_dim);
        VectorX<T> outpt1(1);
        VectorX<T> outpt2(1);

        // todo: eventually remove trap_samples from RayBlock. More stable to 
        //       always specify resolution when calling trapezoid()
        if (nSamples == -1)
        {
            nSamples = trap_samples;
        }

        // Compute parametrization of start and end points
        for (int j = 0; j < dom_dim; j++)
        {
            au(j) = (a(j) - core_mins(j)) / (core_maxs(j) - core_mins(j));
            bu(j) = (b(j) - core_mins(j)) / (core_maxs(j) - core_mins(j));
            du(j) = (bu(j) - au(j)) / (nSamples-1);
        }

        T step = (b-a).norm() / (nSamples-1);

        // Sample base MFA and compute trapezoid rule approximation of integral
        for (int i = 0; i < nSamples - 1; i++)
        {
            param1 = au + i*du;
            param2 = au + (i+1)*du;

            decoder.VolPt(param1, outpt1, mfa->var(0).tmesh.tensor_prods[0]);
            decoder.VolPt(param2, outpt2, mfa->var(0).tmesh.tensor_prods[0]);
            
            result += (outpt1(0) + outpt2(0))/2 * step;
        }

        return result;
    }

    T stableArcTan(const VectorX<T>& A, const VectorX<T>& B, const VectorX<T>& C) const
    {
        return stableArcTan(A.norm(), B.norm(), C.norm());
    }

    // Computes the arctangent of the angle between vectors A and B in a numerically stable
    // way.
    // See: https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors
    // and: https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
    // 
    // This will always return a number in [0, pi]. The normal range of atan is [-pi/2, pi/2],
    // but this function operates only on positive values, so its range is positive. (and also
    // note the output of atan is doubled)
    T stableArcTan(T a, T b, T c) const
    {
        // T a = A.norm();
        // T b = B.norm();
        // T c = C.norm();
        T t = 0, u = 0;

        // sort a >= b, c
        // if (a < c)
        // {
        //     t = a;
        //     a = c;
        //     c = t;
        // }
        if (a < b)
        {
            t = a;
            a = b;
            b = t;
        }

        // start computing
        if (c > b)
        {
            u = b - (a-c);
        }
        else
        {
            u = c - (a-b);
        }

        // T temp1 = sqrt( (((a-b)+c)*u) / ((a+(b+c))*((a-c)+b)) );
        // T temp2 = atan(temp1);
        // fmt::print(stderr, "temp1: {}\n", temp1);
        // fmt::print(stderr, "temp2: {}\n", temp2);

        return 2*atan(sqrt( (((a-b)+c)*u) / ((a+(b+c))*((a-c)+b)) ));
    }

    T integrate_ray(
        const   diy::Master::ProxyWithLink& cp,
        // mfa::Decoder<T>& integralDecoder,
        const   VectorX<T>& a,
        const   VectorX<T>& b) const
    {
        if (dom_dim == 2) return integrate_ray_2d(cp, a, b);
        else if (dom_dim == 3) return integrate_ray_3d(cp, a, b);
        else throw mfa::MFAError("Incorrect dimension in integrate_ray");
    
        return -1;
    }

    T integrate_ray_3d(
        const   diy::Master::ProxyWithLink& cp,
        // mfa::Decoder<T>& integralDecoder,
        const   VectorX<T>& a,
        const   VectorX<T>& b) const
    {
        // const double pi = 3.14159265358979;
        const bool verbose = false;

        // TODO: This is for 2d only right now
        if (a.size() != 3 || b.size() != 3)
        {
            throw mfa::MFAError("Incorrect dimension in integrate ray");
        }

        T rho = 0;
        T nu = 0;
        T theta = 0;
        T phi = 0;
        dualCoords3d(a, b, rho, nu, theta, phi);

        // fmt::print(stderr, "phi in integrate_ray: {}\n", phi);

        // T a_x = a(0);
        // T a_y = a(1);
        // T b_x = b(0);
        // T b_y = b(1);
        T u0 = 0, u1 = 0;
        T length = 2*r_lim;

        if (phi > 0.001 && phi < 3.14)
        {
            u0 = (a(2) - rho*cos(phi)) / (-2*r_lim*sin(phi)) + 0.5;
            u1 = (b(2) - rho*cos(phi)) / (-2*r_lim*sin(phi)) + 0.5;
            // fmt::print(stderr, "TRACE: case 1\n");
        }
        else if (theta > 0.001 && theta < 3.14)
        {
            u0 = (a(1) - rho*sin(theta)*sin(phi) - nu*cos(theta)) / (2*r_lim*sin(theta)*cos(phi)) + 0.5;
            u1 = (b(1) - rho*sin(theta)*sin(phi) - nu*cos(theta)) / (2*r_lim*sin(theta)*cos(phi)) + 0.5;
            // fmt::print(stderr, "TRACE: case 2\n");
        }
        else
        {
            u0 = (a(0) - rho*cos(theta)*sin(phi) + nu*sin(theta)) / (2*r_lim*cos(theta)*cos(phi)) + 0.5;
            u1 = (b(0) - rho*cos(theta)*sin(phi) + nu*sin(theta)) / (2*r_lim*cos(theta)*cos(phi)) + 0.5;
            // fmt::print(stderr, "TRACE: case 3\n");
        }

        // fmt::print(stderr, "{}", mfa::print_vec(a));
        // fmt::print(stderr, "{}", mfa::print_vec(b));
        // fmt::print(stderr, "b-a norm: {}\n", mfa::print_vec((b-a).normalized()));
        // fmt::print(stderr, "s2:       {} {} {}\n", cos(theta)*cos(phi), sin(theta)*cos(phi), -1*sin(phi));
        // fmt::print(stderr, "{} {} {} {}\n", rho, nu, theta, phi);
        // fmt::print(stderr, "{} {}\n", u0, u1);   
        // fmt::print(stderr, "-----------------------\n");
        
        // Scalar valued path integrals do not have an orientation, so we always
        // want the limits of integration to go from smaller to larger.
        if (u0 > u1)
        {
            T temp  = u1;
            u1 = u0;
            u0 = temp;
        }

        VectorX<T> output(1); // todo: this is hardcoded for the first (scalar) variable only
        VectorX<T> params(ray_dom_dim);
        params(0) = 0;  // unused
        params(1) = (rho+r_lim) / (2*r_lim);
        params(2) = (nu+r_lim) / (2*r_lim);
        params(3) = theta / pi;
        params(4) = phi / pi;

        params(0) = params(0) < 0 ? 0 : params(0);
        params(1) = params(1) < 0 ? 0 : params(1);
        params(2) = params(2) < 0 ? 0 : params(2);
        params(3) = params(3) < 0 ? 0 : params(3);
        params(4) = params(4) < 0 ? 0 : params(4);
        params(0) = params(0) > 1 ? 1 : params(0);
        params(1) = params(1) > 1 ? 1 : params(1);
        params(2) = params(2) > 1 ? 1 : params(2);
        params(3) = params(3) > 1 ? 1 : params(3);
        params(4) = params(4) > 1 ? 1 : params(4); 

        // fmt::print(stderr, "params: {}\n", mfa::print_vec(params));

        ray_mfa->Integrate1D(0, 0, u0, u1, params, output);
         
        // integralDecoder.AxisIntegral(0, u0, u1, params, output);

        output *= length;

        return output(0);
    }

    void checkParams(const VectorX<T>& params)
    {
        for (int i = 0; i < params.size(); i++)
        {
            if (params(i) < 0)
            {
                if (params(i) < -1e-8)
                {
                    throw mfa::MFAError(fmt::format("Out of bounds parameter. Dimension {}, Value = {}", i, params(i)));
                }
                params(i) = 0;
            }
            if (params(i) > 1)
            {
                if (params(i) > 1 + 1e-8)
                {
                    throw mfa::MFAError(fmt::format("Out of bounds parameter. Dimension {}, Value = {}", i, params(i)));
                }
                params(i) = 1;
            }
        }
    }

    T integrate_ray_2d(
        const   diy::Master::ProxyWithLink& cp,
        // mfa::Decoder<T>& integralDecoder,
        const   VectorX<T>& a,
        const   VectorX<T>& b) const
    {
        const double pi = 3.14159265358979;
        const bool verbose = false;

        // TODO: This is for 2d only right now
        if (a.size() != 2 || b.size() != 2)
        {
            throw mfa::MFAError("Incorrect dimension in integrate ray");
        }

        auto [alpha, rho] = dualCoords2d(a, b);

        T a_x = a(0);
        T a_y = a(1);
        T b_x = b(0);
        T b_y = b(1);
        T u0 = 0, u1 = 0;
        
        // x = rho*cos(alpha) + 2R(u-0.5)(-sin(alpha))
        // y = rho*sin(alpha) + 2R(u-0.5)(cos(alpha))
        if (alpha > 0.1 && alpha < 3.0)
        {
            u0 = (a_x - rho*cos(alpha)) / (-2*r_lim*sin(alpha)) + 0.5;
            u1 = (b_x - rho*cos(alpha)) / (-2*r_lim*sin(alpha)) + 0.5;
        }
        else
        {
            u0 = (a_y - rho*sin(alpha)) / (2*r_lim*cos(alpha)) + 0.5;
            u1 = (b_y - rho*sin(alpha)) / (2*r_lim*cos(alpha)) + 0.5;
        }
        T length = 2*r_lim;
        

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
            fmt::print(stderr, "RAY: {} ---- {}\n", mfa::print_vec(a), mfa::print_vec(b));
            fmt::print(stderr, "|  m: {}\n", (a_x==b_x) ? "inf" : to_string((b_y-a_y)/(b_x-a_x)).c_str());
            fmt::print(stderr, "|  alpha:  {},   rho: {}\n", alpha, rho);
            fmt::print(stderr, "|  length: {}\n", length);
            fmt::print(stderr, "|  u0: {},  u1: {}\n", u0, u1);
            fmt::print(stderr, "+---------------------------------------\n\n");
        }

        VectorX<T> output(1); // todo: this is hardcoded for the first (scalar) variable only
        VectorX<T> params(ray_dom_dim);
        params(0) = 0;  // unused
        params(1) = (rho - ray_bounds_mins(1)) / (ray_bounds_maxs(1) - ray_bounds_mins(1));
        params(2) = (alpha - ray_bounds_mins(2)) / (ray_bounds_maxs(2) - ray_bounds_mins(2));

        ray_mfa->Integrate1D(0, 0, u0, u1, params, output);
         
        // integralDecoder.AxisIntegral(0, u0, u1, params, output);

        output *= length;

        return output(0);
    }

    // Compute segment errors in a RayMFA
    void compute_sinogram(
        const   diy::Master::ProxyWithLink& cp,
        const   DomainArgs& d_args, 
                bool discrete) const
    {
        fmt::print(stderr, "Computing sinogram\n");
        // // Initialize decoder
        // mfa::Decoder<T> integralDecoder(ray_mfa->var(0), 0);  // no verbose output for single points

        real_t extent = input->domain.col(dom_dim).maxCoeff() - input->domain.col(dom_dim).minCoeff();

        ofstream sinotruefile;
        ofstream sinoapproxfile;
        ofstream sinoerrorfile;
        string sino_true_filename = "sinogram_true_gid" + to_string(cp.gid()) + ".txt";
        string sino_approx_filename = "sinogram_approx_gid" + to_string(cp.gid()) + ".txt";
        string sino_error_filename = "sinogram_error_gid" + to_string(cp.gid()) + ".txt";
        sinotruefile.open(sino_true_filename);
        sinoapproxfile.open(sino_approx_filename);
        sinoerrorfile.open(sino_error_filename);
        int test_n_alpha = 450;
        int test_n_rho = 450;

        const double nanvalue = std::numeric_limits<double>::quiet_NaN();

        VectorX<T> start_pt(dom_dim), end_pt(dom_dim);
        for (int i = 0; i < test_n_alpha; i++)
        {
            for (int j = 0; j < test_n_rho; j++)
            {
                T alpha = 3.14159265 / (test_n_alpha-1) * i;
                T rho = r_lim*2 / (test_n_rho-1) * j - r_lim;
                T x0, x1, y0, y1;   // end points of full line

                get_box_intersections(alpha, rho, x0, y0, x1, y1, core_mins, core_maxs);
                if (x0==0 && y0==0 && x1==0 && y1==0)
                {   
                    sinotruefile << alpha << " " << rho << " " << "0 " << "nan" << endl;
                    sinoapproxfile << alpha << " " << rho << " " << "0 " << "nan" << endl;
                    sinoerrorfile << alpha << " " << rho << " " << "0 " << "nan" << endl;
                }
                else
                {
                    T length = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
                    start_pt(0) = x0; 
                    start_pt(1) = y0;
                    end_pt(0) = x1;
                    end_pt(1) = y1;

                    T test_result = 0;
                    if (!discrete)  // Use RayModel integration
                    {
                        test_result = integrate_ray(cp, start_pt, end_pt) / length;   // normalize by segment length
                    }
                    else    // Use trapezoid rule
                    {
                        test_result = trapezoid(cp, d_args, start_pt, end_pt) / length; 
                    }

                    // T test_result = integrate_ray(cp, start_pt, end_pt) / length;   // normalize by segment length
                    T test_actual = sintest(start_pt, end_pt) / length;

                    T e_abs = abs(test_result - test_actual);
                    T e_rel = e_abs/extent;

                    sinotruefile << alpha << " " << rho << " 0 " << test_actual << endl;
                    sinoapproxfile << alpha << " " << rho << " 0 " << test_result << endl;
                    sinoerrorfile << alpha << " " << rho << " 0 " << e_abs << endl;
                }
                
            }
        }
        sinotruefile.close();
        sinoapproxfile.close();
        sinoerrorfile.close();
        
        return;
    }

    int get_discrete_resolution(
        const   diy::Master::ProxyWithLink& cp,
        const   DomainArgs& d_args,
        int     num_ints,
        int     seed = 0)
    {
        // Error summary
        VectorX<T> oldStatsVec, newStatsVec, relStatsVec;
        mfa::Stats<T> oldStats(true), newStats(true);
        // newStats.init(input);

        // Randomness generation
        std::random_device dev;
        if (seed == 0)
        {
            seed = dev();
        }

        fmt::print(stderr, "Starting loop to converge discrete integration ({} integrals per iteration)\n", num_ints);

        bool converged = false;
        int itCount = 0;
        int nSamples = 50;
        int maxIterations = 100;
        double scaleFactor = 1.5;
        for (int k = 0; k < maxIterations; k++, itCount++)
        {
            // Update resolution of trapzoid rule
            nSamples *= scaleFactor;

            fmt::print(stderr, "Iteration {}: \n", itCount);
            fmt::print(stderr, "  nSamples: {}\n", nSamples);

            // Re-initizalize Stats
            newStats.init(input);

            // Restart random number generation. Important! We should start with the same
            // see in each iteration of the loop we the lines chosen are the same in 
            // each iteration
            // Might be better to compute num_ints endpoints ahead of time, save them, and
            // then always use the same ones, to avoid mucking around with prng
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> dist(0,1); 

            // // Initialize decoder
            // mfa::Decoder<T> integralDecoder(ray_mfa->var(0), 0);  // no verbose output for single points

            real_t result = 0, len = 0;
            VectorX<real_t> start_pt(dom_dim), end_pt(dom_dim);
            for (int i = 0; i < num_ints; i++)
            {
                for (int j = 0; j < dom_dim; j++)
                {
                    start_pt(j) = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
                    end_pt(j)   = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
                    // start_pt(j) = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
                    // end_pt(j)   = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
                }
                len = (end_pt - start_pt).norm();

                result = trapezoid(cp, d_args, start_pt, end_pt, nSamples) / len; 
                newStats.update(0, result);
            }

            if (k > 0)
            {
                oldStats.dump_log_Eigen(0, oldStatsVec);
                newStats.dump_log_Eigen(0, newStatsVec);
                relStatsVec = (newStatsVec - oldStatsVec).cwiseAbs().array() / oldStatsVec.cwiseAbs().array();

                double stddev = (relStatsVec.array()-relStatsVec.mean()).square().sum() / (num_ints-1);
                fmt::print(stderr, "  Min change: {}\n", relStatsVec.minCoeff());
                fmt::print(stderr, "  Max change: {}\n", relStatsVec.maxCoeff());
                fmt::print(stderr, "  Avg change: {}\n", relStatsVec.mean());
                fmt::print(stderr, "  Std dev:    {}\n", stddev);
                if ((relStatsVec.array() < 0.05).all() && relStatsVec.mean() < 1e-6)
                {
                    fmt::print(stderr, "****Breaking loop\n");
                    converged = true;
                }
            }
            
            if (converged) break;

            oldStats = newStats;
        }   // end resolution loop

        if (!converged) 
        {
            fmt::print(stderr, "WARNING: Discrete integration did not converge after {} iterations\n", maxIterations);
            return -1;
        }

        fmt::print(stderr, "Discrete integration converged with {} samples after {} iterations\n", nSamples, itCount);

        return nSamples;
    }

    void integral_speed_test(
        const   diy::Master::ProxyWithLink& cp,
        const   DomainArgs& d_args,
        int     num_ints,
        int     discreteRes,
        bool    discrete,
        int     seed)       // require seed for reproducibility
    {
        fmt::print(stderr, "Testing integration speed\n", num_ints);
        fmt::print(stderr, "  Number of integrals: {}\n", num_ints);
        fmt::print(stderr, "  Computation method: {}\n", discrete ? "discrete" : "spline");

        // Error summary
        mfa::Stats<T> stats(true);
        stats.init(input);

        // Randomness generation
        std::random_device dev;
        if (seed == 0)
        {
            seed = dev();
        }
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0,1); 

        real_t result = 0, groundTruth = 0, len = 0, err = 0;
        VectorX<real_t> start_pt(dom_dim), end_pt(dom_dim);

        auto startTime = std::chrono::steady_clock::now();
        for (int i = 0; i < num_ints; i++)
        {
            for (int j = 0; j < dom_dim; j++)
            {
                start_pt(j) = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
                end_pt(j)   = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
            }
            len = (end_pt - start_pt).norm();

            if (discrete)
            {
                result = trapezoid(cp, d_args, start_pt, end_pt) / len; 
            }
            else
            {
                result = integrate_ray(cp, start_pt, end_pt) / len;   // normalize by segment length
            }

            stats.update(0, result);
        }
        auto endTime = std::chrono::steady_clock::now();
        fmt::print(stderr, "  Computation Time: {} ms\n", chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
    }

    void integral_error(
        const   diy::Master::ProxyWithLink& cp,
        const   DomainArgs& d_args,
        int     num_ints,
        int     discreteRes,
        bool    discrete = false,
        int     seed = 0)
    {
        if (discreteRes == -1)
        {
            discreteRes = get_discrete_resolution(cp, d_args, num_ints, seed);
        }

        fmt::print(stderr, "Computing errors random line integrals\n", num_ints);
        fmt::print(stderr, "  Number of integrals: {}\n", num_ints);
        fmt::print(stderr, "  Computation method: {}\n", discrete ? "discrete" : "spline");
        fmt::print(stderr, "  Ground Truth Resolution: {}\n", discreteRes);

        // Error summary
        mfa::Stats<T> stats(true);
        stats.init(input);

        // Randomness generation
        std::random_device dev;
        if (seed == 0)
        {
            seed = dev();
        }
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0,1); 

        // // Initialize decoder
        // mfa::Decoder<T> integralDecoder(ray_mfa->var(0), 0);  // no verbose output for single points

        real_t result = 0, groundTruth = 0, len = 0, err = 0;
        VectorX<real_t> start_pt(dom_dim), end_pt(dom_dim);
        for (int i = 0; i < num_ints; i++)
        {
            for (int j = 0; j < dom_dim; j++)
            {
                start_pt(j) = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
                end_pt(j)   = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
                // start_pt(j) = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
                // end_pt(j)   = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
            }
            len = (end_pt - start_pt).norm();

            if (discrete)
            {
                result = trapezoid(cp, d_args, start_pt, end_pt) / len; 
            }
            else
            {
                result = integrate_ray(cp, start_pt, end_pt) / len;   // normalize by segment length
            }
            groundTruth =  trapezoid(cp, d_args, start_pt, end_pt, discreteRes) / len;
            
            // actual = sintest(start_pt, end_pt) / len;                        // normalize by segment length
            err = abs(result - groundTruth);
            stats.update(0, err);
        }

        fmt::print(stderr, "  Done.\n", num_ints);
        stats.set_style(mfa::PrintStyle::Side);
        stats.print_var(0);
        stats.write_all_vars("li_errors");
    }

    void compute_random_ints(
        const   diy::Master::ProxyWithLink& cp,
        const   DomainArgs& d_args,
        int     num_ints,
        bool    discrete = false,
        int     seed = 0)
    {
        // Error summary
        mfa::Stats<T> stats(true);
        stats.init(input);

        // Randomness generation
        std::random_device dev;
        if (seed == 0)
        {
            seed = dev();
        }
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0,1); 

        // // Initialize decoder
        // mfa::Decoder<T> integralDecoder(ray_mfa->var(0), 0);  // no verbose output for single points

        real_t result = 0, actual = 0, len = 0, err = 0;
        VectorX<real_t> start_pt(dom_dim), end_pt(dom_dim);
        for (int i = 0; i < num_ints; i++)
        {
            for (int j = 0; j < dom_dim; j++)
            {
                start_pt(j) = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
                end_pt(j)   = dist(rng) * (core_maxs(j) - core_mins(j)) + core_mins(j);
                // start_pt(j) = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
                // end_pt(j)   = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
            }
            len = (end_pt - start_pt).norm();

            if (!discrete)  // Use RayModel integration
            {
                result = integrate_ray(cp, start_pt, end_pt) / len;   // normalize by segment length
            }
            else    // Use trapezoid rule
            {
                result = trapezoid(cp, d_args, start_pt, end_pt) / len; 
            }
            actual = sintest(start_pt, end_pt) / len;                        // normalize by segment length
            err = abs(result - actual);
            stats.update(0, err);
        }

        fmt::print(stderr, "\nComputed {} random line integrals.\n", num_ints);
        stats.set_style(mfa::PrintStyle::Side);
        stats.print_var(0);
        stats.write_all_vars("li_errors");
    }

    void print_knots_ctrl(const mfa::MFA_Data<T>& model) const
    {
        VectorXi tot_nctrl_pts_dim = VectorXi::Zero(model.dom_dim);        // number contrl points per dim.
        size_t tot_nctrl_pts = 0;                                        // total number of control points

        for (auto j = 0; j < model.ntensors(); j++)
        {
            tot_nctrl_pts_dim += model.tmesh.tensor_prods[j].nctrl_pts;
            tot_nctrl_pts += model.tmesh.tensor_prods[j].nctrl_pts.prod();
        }
        // print number of control points per dimension only if there is one tensor
        if (model.ntensors() == 1)
            fmt::print(stderr, "# output ctrl pts     = [{}]\n", fmt::join(tot_nctrl_pts_dim, " "));
        fmt::print(stderr, "tot # output ctrl pts = {}\n", tot_nctrl_pts);

        fmt::print(stderr, "# output knots        = [ ");
        for (auto j = 0 ; j < model.tmesh.all_knots.size(); j++)
        {
            fmt::print(stderr, "{} ", model.tmesh.all_knots[j].size());
        }
        fmt::print(stderr, "]\n");
    }

    void print_ray_model(const diy::Master::ProxyWithLink& cp)    // error was computed
    {
        if (!ray_mfa)
        {
            fmt::print(stderr, "gid = {}: No Ray MFA found.\n", cp.gid());
            return;
        }

        fmt::print(stderr, "gid = {}\n", cp.gid());

        // geometry
        fmt::print(stderr, "---------------- geometry model ----------------\n");
        print_knots_ctrl(ray_mfa->geom());
        fmt::print(stderr, "------------------------------------------------\n");

        // science variables
        fmt::print(stderr, "\n----------- science variable models ------------\n");
        for (int i = 0; i < ray_mfa->nvars(); i++)
        {
            fmt::print(stderr, "-------------------- var {} --------------------\n", i);
            print_knots_ctrl(ray_mfa->var(i));
            fmt::print(stderr, "------------------------------------------------\n");
            if (ray_stats.initialized)
            {
                ray_stats.print_var(i);
                fmt::print(stderr, "------------------------------------------------\n");
            }
        }
        
        if (ray_stats.initialized)
        {
            ray_stats.print_max();
            fmt::print(stderr, "------------------------------------------------\n");
        }
        fmt::print(stderr, "# input points        = {}\n", ray_input->npts);
        fmt::print(stderr, "compression ratio     = {:.2f}\n", compute_ray_compression());
    }

    // compute compression ratio
    float compute_ray_compression() const
    {
        float in_coords = (ray_input->npts) * (ray_input->pt_dim);
        float out_coords = 0.0;
        for (auto j = 0; j < ray_mfa->geom().ntensors(); j++)
            out_coords += ray_mfa->geom().tmesh.tensor_prods[j].ctrl_pts.rows() *
                ray_mfa->geom().tmesh.tensor_prods[j].ctrl_pts.cols();
        for (auto j = 0; j < ray_mfa->geom().tmesh.all_knots.size(); j++)
            out_coords += ray_mfa->geom().tmesh.all_knots[j].size();
        for (auto i = 0; i < ray_mfa->nvars(); i++)
        {
            for (auto j = 0; j < ray_mfa->var(i).ntensors(); j++)
                out_coords += ray_mfa->var(i).tmesh.tensor_prods[j].ctrl_pts.rows() *
                    ray_mfa->var(i).tmesh.tensor_prods[j].ctrl_pts.cols();
            for (auto j = 0; j < ray_mfa->var(i).tmesh.all_knots.size(); j++)
                out_coords += ray_mfa->var(i).tmesh.all_knots[j].size();
        }
        return in_coords / out_coords;
    }
}; // RayBlock

#endif // _MFA_RAY_BLOCK_HPP
