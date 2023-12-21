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

    T r_lim{0};

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
    // precondition: Block already contains a fully encoded MFA
    void create_ray_model(
        const       diy::Master::ProxyWithLink& cp,
        MFAInfo& mfa_info,
        DomainArgs& args,
        int n_samples,
        int n_rho,
        int n_alpha,
        int v_samples,
        int v_rho,
        int v_alpha)
    {
        const double pi = 3.14159265358979;
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

        // Update dimensionality
        ray_dom_dim = dom_dim + 1;
        VectorXi new_mdims = mfa->model_dims();
        new_mdims[0] += 1;  

        VectorXi ndom_pts{{n_samples, n_rho, n_alpha}};
        ray_input = new mfa::PointSet<T>(ray_dom_dim, new_mdims, ndom_pts.prod(), ndom_pts);

        // extents of domain in physical space
        VectorX<T> param(dom_dim);
        VectorX<T> outpt(pt_dim);
        const T xl = bounds_mins(0);
        const T xh = bounds_maxs(0);
        const T yl = bounds_mins(1);
        const T yh = bounds_maxs(1);

        // TODO: make this generic
        double max_radius = max(max(abs(xl),abs(xh)), max(abs(yl),abs(yh)));
        r_lim = max_radius * 1.5;

        // Increments of rho and alpha
        double dr = r_lim * 2 / (n_rho-1);
        double da = pi / (n_alpha-1);

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
            ray_bounds_mins(i+1) = bounds_mins(i);
            ray_bounds_maxs(i+1) = bounds_maxs(i);
        }
        ray_core_mins = ray_bounds_mins.head(dom_dim+1);
        ray_core_maxs = ray_bounds_maxs.head(dom_dim+1);

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

                // "parallel-plate setup"
                // start/end coordinates of the ray (alpha, rho)
                // In this setup the length of every segment (x0,y0)--(x1,y1) is constant
                span_x = 2 * r_lim * sin(alpha);
                span_y = 2 * r_lim * cos(alpha);
                x0 = rho * cos(alpha) - r_lim * sin(alpha);
                x1 = rho * cos(alpha) + r_lim * sin(alpha);
                y0 = rho * sin(alpha) + r_lim * cos(alpha);
                y1 = rho * sin(alpha) - r_lim * cos(alpha);

                T dx = span_x / (n_samples-1);
                T dy = span_y / (n_samples-1);

                for (int is = 0; is < n_samples; is++)
                {
                    int idx = ia*n_rho*n_samples + ir*n_samples + is;
                    ray_input->domain(idx, 0) = (double)is / (n_samples-1);
                    ray_input->domain(idx, 1) = rho;
                    ray_input->domain(idx, 2) = alpha;

                    T x = x0 + is * dx;
                    T y = y0 - is * dy;

                    // If this point is not in the original domain
                    if (x < xl + 1e-8 || x > xh - 1e-8 || y < yl + 1e-8 || y > yh - 1e-8)
                    {
                        // add dummy value, which will never be queried
                        ray_input->domain(idx, ray_dom_dim) = 0;
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

                        mfa->Decode(param, outpt);
                        ray_input->domain.block(idx, ray_dom_dim, 1, pt_dim - dom_dim) = outpt.tail(pt_dim - dom_dim).transpose();
                    }
                }
            }
        }

        ray_input->set_bounds(ray_core_mins, ray_core_maxs);
        ray_input->set_domain_params();

        // ------------ Creation of new MFA ------------- //
        int verbose = mfa_info.verbose && cp.master()->communicator().rank() == 0; 
        ray_mfa = new mfa::MFA<T>(ray_dom_dim, verbose);

        // Set up new geometry
        ray_mfa->AddGeometry(ray_dom_dim);

        // Set nctrl_pts, degree for variables
        VectorXi nctrl_pts(ray_dom_dim);
        VectorXi p(ray_dom_dim);
        for (auto i = 0; i< mfa->nvars(); i++)
        {
            int min_p = mfa->var(i).p.minCoeff();
            int max_nctrl = mfa->var(i).tmesh.tensor_prods[0].nctrl_pts.maxCoeff();

            // set ray model degree to minimum degree of original model
            p = min_p * VectorXi::Ones(ray_dom_dim);
            nctrl_pts(0) = v_samples;
            nctrl_pts(1) = v_rho;
            nctrl_pts(2) = v_alpha;
            // p(0) = 2;
            // p(1) = 2;
            // p(2) = 2;

            ray_mfa->AddVariable(p, nctrl_pts, 1);
        }

        // Encode ray model. 
        ray_mfa->FixedEncodeGeom(*ray_input, false);
        ray_mfa->RayEncode(0, *ray_input);

        // --------- Decode and compute errors --------- //
        fmt::print("Computing errors on uniform grid...\n");
        mfa::PointSet<T>* unused = nullptr;
        VectorXi grid_size{{n_samples, n_rho, n_alpha}};
        analytical_ray_error_field(cp, ray_mfa, grid_size, "sine", args, unused, ray_approx, ray_errs);
        fmt::print("done.\n");

        // delete input;
        // delete approx;
        // delete errs;
        // input = ray_input;
        // approx = ray_approx;
        // errs = ray_errs;
        // ray_input = nullptr;
        // ray_approx = nullptr;
        // ray_errs = nullptr;
    }

    // Convert (t, rho, theta) to (x, y) and return true if the x,y coords are in the original domain
    bool radon2cart(const VectorX<T>& radon_coords, VectorX<T>& cart_coords)
    {
        if (r_lim == 0)
        {
            fmt::print("ERROR: r_lim=0 in RayBlock::radon2cart()\nExiting.\n");
            exit(1);
        }

        T t = radon_coords(0);
        T rho = radon_coords(1);
        T alpha = radon_coords(2);

        T x0, y0, span_x, span_y;
        T SA = sin(alpha);
        T CA = cos(alpha);
        span_x = 2 * r_lim * SA;
        span_y = 2 * r_lim * CA;
        x0 = rho * CA - r_lim * SA;
        y0 = rho * SA + r_lim * CA;

        T x = x0 + t*span_x;
        T y = y0 - t*span_y;

        cart_coords(0) = x;
        cart_coords(1) = y;

        T xl = core_mins(0);
        T xh = core_maxs(0);
        T yl = core_mins(1);
        T yh = core_maxs(1);

        return (x >= xl) && (x <= xh) && (y >= yl) && (y <= yh);
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
        if (exact_pts) cerr << "Warning: Overwriting \'exact_pts\' pointset in analytical_ray_error_field()" << endl;
        if (approx_pts) cerr << "Warning: Overwriting \'approx_pts\' pointset in analytical_ray_error_field()" << endl;
        if (error_pts) cerr << "Warning: Overwriting \'error_pts\' pointset in analytical_ray_error_field()" << endl;
        delete exact_pts;
        delete approx_pts;
        delete error_pts;

        // Set up PointSets with grid parametrizations
        exact_pts = new mfa::PointSet<T>(models->dom_dim, models->model_dims(), grid.prod(), grid);
        approx_pts= new mfa::PointSet<T>(models->dom_dim, models->model_dims(), grid.prod(), grid);
        error_pts = new mfa::PointSet<T>(models->dom_dim, models->model_dims(), grid.prod(), grid);
        approx_pts->set_grid_params();

        // Decode on above-specified grid
        models->Decode(*approx_pts, false);

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
            cerr << "WARNING: Overwriting \"ray_approx\" pointset in RayBlock::decode_ray_block" << endl;
            delete ray_approx;
        }
        ray_approx = new mfa::PointSet<T>(ray_input->params, ray_input->model_dims());  // Set decode params from ray_input params

        ray_mfa->Decode(*ray_approx, false);
    }

    pair<T,T> dualCoords(const VectorX<T>& a, const VectorX<T>& b) const
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
        const   VectorX<T>& b) const
    {
        const double pi = 3.14159265358979;
        const bool verbose = false;

        // TODO: This is for 2d only right now
        if (a.size() != 2 || b.size() != 2)
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

        T x0, x1, y0, y1;   // end points of full line
        T u0 = 0, u1 = 0;
        T length = 2*r_lim;
        x0 = rho * cos(alpha) - r_lim * sin(alpha);
        x1 = rho * cos(alpha) + r_lim * sin(alpha);
        y0 = rho * sin(alpha) + r_lim * cos(alpha);
        y1 = rho * sin(alpha) - r_lim * cos(alpha);

        // parameter values along ray for 'start' and 'end'
        // compute in terms of Euclidean distance to avoid weird cases
        //   when line is nearly horizontal or vertical
        T x_sep = abs(x1 - x0);
        T y_sep = abs(y1 - y0);
        
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
        VectorX<T> params(ray_dom_dim);
        params(0) = 0;  // unused
        params(1) = (rho - ray_bounds_mins(1)) / (ray_bounds_maxs(1) - ray_bounds_mins(1));
        params(2) = (alpha - ray_bounds_mins(2)) / (ray_bounds_maxs(2) - ray_bounds_mins(2));

        ray_mfa->Integrate1D(0, 0, u0, u1, params, output);
        output *= length;

        return output(0);
    }

    // Compute segment errors in a RayMFA
    void compute_sinogram(const   diy::Master::ProxyWithLink& cp) const
    {
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
        int test_n_alpha = 150;
        int test_n_rho = 150;

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
                    sinotruefile << alpha << " " << rho << " " << " 0 0" << endl;
                    sinoapproxfile << alpha << " " << rho << " " << " 0 0" << endl;
                    sinoerrorfile << alpha << " " << rho << " " << " 0 0" << endl;
                }
                else
                {
                    T length = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
                    start_pt(0) = x0; 
                    start_pt(1) = y0;
                    end_pt(0) = x1;
                    end_pt(1) = y1;

                    T test_result = integrate_ray(cp, start_pt, end_pt) / length;   // normalize by segment length
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

    void compute_random_ints(
        const   diy::Master::ProxyWithLink& cp,
        const   DomainArgs& d_args,
        int     num_ints)
    {
        // Error summary
        mfa::Stats<T> stats(true);
        stats.init(input);

        // Randomness generation
        std::uniform_real_distribution<double> dist(0,1); 
        std::random_device dev;
        std::mt19937 rng(dev());

        real_t result = 0, actual = 0, len = 0, err = 0;
        VectorX<real_t> start_pt(dom_dim), end_pt(dom_dim);
        for (int i = 0; i < num_ints; i++)
        {
            for (int j = 0; j < dom_dim; j++)
            {
                start_pt(j) = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
                end_pt(j)   = dist(rng) * (d_args.max[j]-d_args.min[j]) + d_args.min[j];
            }
            len = (end_pt - start_pt).norm();

            result = integrate_ray(cp, start_pt, end_pt) / len;   // normalize by segment length
            actual = sintest(start_pt, end_pt) / len;                        // normalize by segment length
            err = abs(result - actual);
            stats.update(0, err);
        }

        fmt::print("\nComputed {} random line integrals.\n", num_ints);
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
            cerr << "# output ctrl pts     = [ " << tot_nctrl_pts_dim.transpose() << " ]" << endl;
        cerr << "tot # output ctrl pts = " << tot_nctrl_pts << endl;

        cerr << "# output knots        = [ ";
        for (auto j = 0 ; j < model.tmesh.all_knots.size(); j++)
        {
            cerr << model.tmesh.all_knots[j].size() << " ";
        }
        cerr << "]" << endl;
    }

    void print_ray_model(const diy::Master::ProxyWithLink& cp)    // error was computed
    {
        if (!ray_mfa)
        {
            fmt::print("gid = {}: No Ray MFA found.\n", cp.gid());
            return;
        }

        fmt::print("gid = {}\n", cp.gid());

        // geometry
        fmt::print("---------------- geometry model ----------------\n");
        print_knots_ctrl(ray_mfa->geom());
        fmt::print("------------------------------------------------\n");

        // science variables
        fmt::print("\n----------- science variable models ------------\n");
        for (int i = 0; i < ray_mfa->nvars(); i++)
        {
            fmt::print("-------------------- var {} --------------------\n", i);
            print_knots_ctrl(ray_mfa->var(i));
            fmt::print("------------------------------------------------\n");
            ray_stats.print_var(i);
            fmt::print("------------------------------------------------\n");
        }
        
        ray_stats.print_max();
        fmt::print("------------------------------------------------\n");
        fmt::print("# input points        = {}\n", ray_input->npts);
        fmt::print("compression ratio     = {:.2f}\n", compute_ray_compression());
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
