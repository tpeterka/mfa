//--------------------------------------------------------------
// mfa data model
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _DATA_MODEL_HPP
#define _DATA_MODEL_HPP

// --- data model ---
//
// using Eigen dense MartrixX to represent vectors of n-dimensional points
// rows: points; columns: point coordinates
//
// There are two types of dimensionality:
// 1. The dimensionality of the NURBS tensor product (p.size())
// (1D = NURBS curve, 2D = surface, 3D = volume 4D = hypervolume, etc.)
// 2. The dimensionality of individual control points (ctrl_pts.cols())
// p.size() < ctrl_pts.cols()
//
// ------------------

namespace mfa
{
    // n.b. We don't store this as a member of MFA_Data because we want multiple threads to 
    //      interact with the same MFA_Data simultaneously. In a threaded environment,
    //      BasisFunInfo should be thread-local
    template<typename T>
    struct BasisFunInfo
    {
        vector<T>           right;  // right parameter differences (t_k - u)
        vector<T>           left;   // left parameter differences (u - t_l)
        vector<T>           np1;    // storage for integral calculation ("N for degree p+1")
        vector<vector<T>>   ndu;    // storage for derivative calculation
        array<vector<T>, 2> a;      // coefficients for high-order derivs
        int                 qmax;   // Largest spline order (p+1) among all dimensions

        BasisFunInfo(const VectorXi& q) : 
            qmax(0)
        {
            for (int i = 0; i < q.size(); i++)
            {
                if (q(i) > qmax)
                    qmax = q(i);
            }

            right.resize(qmax);
            left.resize(qmax);
            np1.resize(qmax);

            ndu.resize(qmax);
            for (int i = 0; i < ndu.size(); i++)
            {
                ndu[i].resize(qmax);
            }

            a[0].resize(qmax);
            a[1].resize(qmax);
        }

        BasisFunInfo(const vector<int>& q) :
            qmax(0)
        {
            for (int i = 0; i < q.size(); i++)
            {
                if (q[i] > qmax)
                    qmax = q[i];
            }

            right.resize(qmax);
            left.resize(qmax);
            np1.resize(qmax);

            ndu.resize(qmax);
            for (int i = 0; i < ndu.size(); i++)
            {
                ndu[i].resize(qmax);
            }

            a[0].resize(qmax);
            a[1].resize(qmax);
        }

        void reset(int dim)
        {
            // left/right is of size max_p + 1, but we will only ever access
            // the first q(i) entries when considering dimension 'i'
            for (int i = 0; i < qmax; i++)
            {
                left[i] = 0;
                right[i] = 0;
            }
        }
    };

    template <typename T>                       // float or double
    struct MFA_Data
    {
        int                       dom_dim;       // number of domain dimensions
        int                       min_dim;       // starting coordinate of this model in full-dimensional data
        int                       max_dim;       // ending coordinate of this model in full-dimensional data
        VectorXi                  p;             // polynomial degree in each domain dimension
        vector<MatrixX<T>>        N;             // vector of basis functions for each dimension
                                                 // for all input points (matrix rows) and control points (matrix cols)
        Tmesh<T>                  tmesh;         // t-mesh of knots, control points, weights
        T                         max_err;       // unnormalized absolute value of maximum error
        bool                      verbose{false};

        // Creates a new (unsolved) MFA_Data
        MFA_Data(
                const VectorXi&             p_,             // polynomial degree in each dimension
                VectorXi                    nctrl_pts_,     // optional number of control points in each dim (size 0 means minimum p+1)
                int                         min_dim_ = -1,  // starting coordinate for input data
                int                         max_dim_ = -1) :// ending coordinate for input data
            dom_dim(p_.size()),
            min_dim(min_dim_),
            max_dim(max_dim_),
            p(p_),
            tmesh(dom_dim, p_, min_dim_, max_dim_)
        {
            if (min_dim_ == -1)
                min_dim = 0;
            if (max_dim == -1)
                max_dim = 0;

            // set number of control points to the minimum, p + 1, if they have not been initialized
            if (!nctrl_pts_.size())
            {
                nctrl_pts_.resize(dom_dim);
                for (auto i = 0; i < dom_dim; i++)
                    nctrl_pts_(i) = p(i) + 1;
            }

            // initialize tmesh knots
            tmesh.init_knots(nctrl_pts_);
            set_knots();
        }

        // constructor for reading in a solved mfa
        MFA_Data(
                const VectorXi&     p_,             // polynomial degree in each dimension
                const Tmesh<T>&     tmesh_,         // solved tmesh
                int                 min_dim_ = -1,  // starting coordinate for input data
                int                 max_dim_ = -1) :// ending coordinate for input data
            dom_dim(p_.size()),
            min_dim(min_dim_),
            max_dim(max_dim_),
            p(p_),
            tmesh(tmesh_)
        {
            if (min_dim_ == -1)
                min_dim = 0;
            if (max_dim == -1)
                max_dim = tmesh_.tensor_prods[0].ctrl_pts.cols() - 1;
        }

        // constructor when reading mfa in and knowing nothing about it yet except its degree and dimensionality
        MFA_Data(
                const VectorXi&     p_,             // polynomial degree in each dimension
                size_t              ntensor_prods,  // number of tensor products to allocate in tmesh
                int                 min_dim_ = -1,  // starting coordinate for input data
                int                 max_dim_ = -1) :// ending coordinate for input data
            dom_dim(p_.size()),
            min_dim(min_dim_),
            max_dim(max_dim_),
            p(p_),
            tmesh(dom_dim, p_, min_dim_, max_dim_, ntensor_prods)
        {
            if (min_dim_ == -1)
                min_dim = 0;
            if (max_dim_ == -1)
                max_dim = 0;
        }

        ~MFA_Data() {}

        inline int dim() const
        {
            return max_dim - min_dim + 1;
        }

        int ntensors() const
        {
            return tmesh.tensor_prods.size();
        }

        void set_knots(const vector<vector<T>>& knots = vector<vector<T>>())
        {
            // Initialize knot data structures
            if (knots.size() != 0)
            {
                // Set from user-specified knots (with checks)
                customKnots(knots);
            }
            else 
            {
                // Set uniformly spaced knots
                uniformKnots();
            }
        }

        void set_param_idxs(const PointSet<T>& input)
        {
            // Set up all_knot_param_idxs for structured data
            if (input.is_structured())
            {
                for (int k = 0; k < dom_dim; k++)
                {
                    // all_knot_param_idxs[k][j] gives the index of the first data point param that 
                    // is greater than or equal to knot[k][j].
                    // NOTE: This data point may not be in the knot span [t_j, t_{j+1}]. All we know
                    //       is that it is the first point greater than t_j.
                    // 
                    // If consecutive entries of all_knot_param_idxs have the same value, then the span
                    // between those two knots must be missing an input point. In this case we issue a 
                    // warning and continue.
                    auto& params = input.params->param_grid;
                    int last = tmesh.all_knots[k].size() - 1;
                    int knot_idx = 0;
                    for (int i = 0; i < params[k].size(); i++)
                    {
                        while (tmesh.all_knots[k][knot_idx] <= params[k][i] && knot_idx <= last)
                        {
                            tmesh.all_knot_param_idxs[k][knot_idx] = i;

                            // Warn if it looks like an interior knot span is missing an input point.
                            if (knot_idx > p(k) + 1 && knot_idx < last - p(k))
                            {
                                if (tmesh.all_knot_param_idxs[k][knot_idx] == tmesh.all_knot_param_idxs[k][knot_idx-1])
                                {
                                    cerr << "WARNING: Missing input point between knots " << knot_idx-1 << " and " << knot_idx << " in dimension " << k << endl;
                                }
                            }
                            knot_idx++;
                        }
                    }
                    
                    // If the largest input parameter is not equal to 1.0, then all_knot_param_idxs will 
                    // not be set up properly. The Tmesh code assumes that every knot (including the last) must 
                    // have at least one input point >= that knot value. If this is not true, then we should 
                    // abort the code. Otherwise, the logic will be incorrect.
                    if (knot_idx <= last)
                    {
                        cerr << "ERROR: all_knot_param_idxs set incorrectly in MFAData::set_knots(). Exiting." << endl;
                        exit(1);
                    }
                }
            }
            else
            {
                cerr << "ERROR: Cannot set all_knot_params_idxs with unstructured data. Exiting." << endl;
                exit(1);
            }
        }

        // original version of basis functions from algorithm 2.2 of P&T, p. 70
        // computes one row of basis function values for a given parameter value
        // writes results in a row of N
        //
        // assumes N has been allocated by caller
        void OrigBasisFuns(
                int                     cur_dim,    // current dimension
                T                       u,          // parameter value
                int                     span,       // index of span in the knots vector containing u
                MatrixX<T>&             N,          // matrix of (output) basis function values
                int                     row) const  // row in N of result
        {
            // initialize row to 0
            N.row(row).setZero();

            // init
            vector<T> scratch(p(cur_dim) + 1);                  // scratchpad, same as N in P&T p. 70
            scratch[0] = 1.0;

            // temporary recurrence results
            // left(j)  = u - knots(span + 1 - j)
            // right(j) = knots(span + j) - u
            vector<T> left(p(cur_dim) + 1);
            vector<T> right(p(cur_dim) + 1);

            // fill N
            for (int j = 1; j <= p(cur_dim); j++)
            {
                // left[j] is u = the jth knot in the correct level to the left of span
                left[j]  = u - tmesh.all_knots[cur_dim][span + 1 - j];
                // right[j] = the jth knot in the correct level to the right of span - u
                right[j] = tmesh.all_knots[cur_dim][span + j] - u;

                T saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    T temp = scratch[r] / (right[r + 1] + left[j - r]);
                    scratch[r] = saved + right[r + 1] * temp;
                    saved = left[j - r] * temp;
                }
                scratch[j] = saved;
            }

            // copy scratch to N
            for (int j = 0; j < p(cur_dim) + 1; j++)
                N(row, span - p(cur_dim) + j) = scratch[j];

            // debug
//             cerr << N << endl;
        }

        // Helper function for computing integrals of basis functions
        // Computes sum of all <degree>-basis functions at or past b_idx, evaluated at u

        T IntBasisFunsHelper(
                int         cur_dim,
                T           u,
                int         span,
                int         basis_idx,      // index of basis function to integrate
                BasisFunInfo<T>& bfi) const
        {
            const int degree = this->p(cur_dim);

            // special cases when limit of integration is at edge of parameter space
            if (u == 0) return 0;
            if (u == 1) return 1;
            
            // special cases when u \in [t_s, t_{s+1}] is "far" from the basis fxn being integrated
            int last_covered = span - degree - 1;       // support: [t_{s-p-1}, t_s]
            int first_uncovered = span + 1;             // support: [t_{s+1}, t_{s+p+2}]
            if (basis_idx <= last_covered) return 1;
            if (basis_idx >= first_uncovered) return 0;

            // Let s:=span. p:=degree. 
            // Computes the degree-(p+1) basis functions:
            // T_{s-p-1}, T_{s-p}, ... T_{s}
            // These are the only degree-(p+1) basis functions that are nonzero in
            // span s, and therefore nonzero at u
            FastBasisFunsK(cur_dim, u, span, bfi.np1, bfi, degree+1);

            // Let r:=basis_idx. To compute the the integral of the degree-p basis function N_r,
            // need to sum all degree-(p+1) basis functions starting with index r. 
            // That is, integral = SUM_{j>=r} T_j(u)
            // However, T_j(u) is only nonzero for j \in [s-p-1, s] (see above)
            // Thus, we need to sum the degree-(p+1) basis functions N_j if:
            //   j >= s-p-1, j >= r, and j <= s
            // One way to do this is to sum all basis fxns stored in N above, 
            // which are the indices [s-p-1, s], except we start partway through
            // the array if r > s-p-1. This is what happens below.

            // idx of the first (p+1)-degree basis function which is nonzero at u
            // NOTE we are now considering functions with degree greater than the MFA
            int first_idx   = span - degree - 1;            

            // Where to start the summation of high-degree basis functions
            // Note that, because we handled special cases above, it must be that:
            //   span - degree <= basis_idx <= span
            // Thus, we always have:
            //   1 <= skip <= degree + 1
            // This makes sense, because if skip were 0, then the code would sum the entire
            // N vector, which has to be 1 by the properties of basis functions. This is why
            // we can treat it as a special case earlier and avoid extra computation.
            int skip        = basis_idx - first_idx;    

            T   sum         = 0;
            for (int j = skip; j < degree + 2; j++)
            {
                sum += bfi.np1[j];
            }

            return sum;
        }

        // Computes the definite integral on [a,b] of the basis function with index 'basis_idx'
        // Reference: "On Calculating with B-Splines. II: Integration," by de Boor, Lyche, and Schumaker (1976)
        //            Lemma 2.1
        T IntBasisFun(
                int cur_dim,            // dimension in parameter space
                int basis_idx,          // basis function to integrate
                T   a,                  // lower limit of integration
                T   b,                  // upper limit of integration
                int spana,              // span containing a
                int spanb,              // span containing b
                BasisFunInfo<T>& bfi) const
        {
            int deg         = this->p(cur_dim);
            int lower_span  = basis_idx;            // knot index of lower bound of basis support
            int upper_span  = basis_idx + deg + 1;  // knot index of upper bound of basis support
            T   suma        = 0;
            T   sumb        = 0;

            T k_start   = tmesh.all_knots[cur_dim][lower_span];
            T k_end     = tmesh.all_knots[cur_dim][upper_span];
            T scaling   = (k_end - k_start) / (deg+1);

            // Compute the sum of all (p+1)-degree basis functions, with index >= basis_idx
            suma = IntBasisFunsHelper(cur_dim, a, spana, basis_idx, bfi);
            sumb = IntBasisFunsHelper(cur_dim, b, spanb, basis_idx, bfi);

            return scaling * (sumb-suma);  
        }

        // Same as FastBasisFuns but 'degree' can be different from the degree of the MFA_Data
        // (useful for computing integrals of basis functions)
        // This requires a check if the knot index is past the extents of our knot vector, 
        // which requires IF statements not present in FastBasisFuns. FastBasisFuns is so performance
        // critical that we make a separate function with the if-logic.
        //
        // NOTE: 'degree' can be greater than the degree of the MFA, which means that the support of some
        //       of these basis functions might extend past our (finite) knot vector
        //       Whenever this is the case, we assume we have an infinite number of pinned knots at 0 and 1.
        //       Thus, any knot "before" the first knot is also at 0; any knot "after" the last knot is also at 1.
        void FastBasisFunsK( int                cur_dim,
                            T                   u,
                            int                 span,
                            vector<T>&          N,              // vector of (output) basis function values 
                            BasisFunInfo<T>&    bfi,
                            int                 degree) const
        {
            assert(N.size() == degree + 1);
            assert(bfi.qmax >= degree + 1);

            const T tid_max = tmesh.all_knots[cur_dim].size() - 1;  // index of last knot
            T tl = 0, tr = 0;                       //leftmost and rightmost knots in a given iteration

            // nb. we do not need to zero out the entirety of N, since the existing entries of N 
            //     are never accessed (they are always overwritten first)
            N[0] = 1;   

            for (int j = 1; j <= degree; j++)
            {
                tl = (span + 1 - j < 0) ? 0 : tmesh.all_knots[cur_dim][span + 1 - j];
                tr = (span + j > tid_max) ? 1 : tmesh.all_knots[cur_dim][span + j];

                // left[j] is u - the jth knot in the correct level to the left of span
                // right[j] is the jth knot in the correct level to the right of span - u
                bfi.left[j]  = u - tl;
                bfi.right[j] = tr - u;

                T saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    T temp = N[r] / (bfi.right[r + 1] + bfi.left[j - r]);
                    N[r] = saved + bfi.right[r + 1] * temp;
                    saved = bfi.left[j - r] * temp;
                }
                N[j] = saved;
            }
        }

        void BasisFunsK(
                int         degree,
                int         cur_dim,
                T           u,
                int         span,
                MatrixX<T>& N,
                int         row) const
        {
            vector<T> loc_knots(degree + 2);

            // initialize row to 0
            N.row(row).setZero();

            for (auto j = 0; j < degree + 1; j++)
            {
                bool ignore = false;
                int b_idx = span - degree + j;  // index of the first basis function with support in span

                for (auto i = 0; i < degree + 2; i++)
                {
                    // when degree > p, not all spans will have precisely
                    // degree+1 active basis funs; this is because only p+1
                    // knots are pinned.
                    if (b_idx + i < 0 || b_idx + i >= tmesh.all_knots[cur_dim].size())
                    {
                        // cerr << "ignored basis function, index " << b_idx + i << endl;
                        ignore = true;
                        continue;
                    }

                    loc_knots[i] = tmesh.all_knots[cur_dim][b_idx + i];
                }
    // cerr << N.rows() << "  " << N.cols() << "  " << row << "  " << b_idx << endl;
                if (!ignore)
                    N(row, b_idx) = OneBasisFunK(degree, cur_dim, u, loc_knots);
            }
        }

        T OneBasisFunK( int         degree,             // degree of the basis function to compute
                        int         cur_dim,            // current domain dimension
                        T           u,                  // parameter in current dimension
                        vector<T>   loc_knots)  const   // knot vector defining the support of the basis function
        {
            vector<T> N(degree + 1);                    // triangular table result
            const vector<T>& U = loc_knots;                 // alias for knot vector for current dimension

            // corner case: 1 at right edge of local knot vector
            if (u == 1.0)
            {
                bool edge = true;
                for (auto j = 0; j < degree + 1; j++)
                {
                    if (loc_knots[1 + j] != 1.0)
                    {
                        edge = false;
                        break;
                    }
                }
                if (edge)
                    return 1.0;
                
                /* else return 0; 
                TODO? if not edge but u==1.0, then we should always return 0? */
            }

           // initialize 0-th degree functions
            for (auto j = 0; j < degree + 1; j++)
            {
                if (u >= U[j] && u < U[j + 1])
                    N[j] = 1.0;
                else
                    N[j] = 0.0;
            }

            // compute triangular table
            T saved, uleft, uright, temp;
            for (auto k = 1; k < degree + 1; k++)
            {
                if (N[0] == 0.0)
                    saved = 0.0;
                else
                    saved = ((u - U[0]) * N[0]) / (U[k] - U[0]);
                for (auto j = 0; j < degree - k + 1; j++)
                {
                    uleft     = U[j + 1];
                    uright    = U[j + k + 1];
                    if (N[j + 1] == 0.0)
                    {
                        N[j]    = saved;
                        saved   = 0.0;
                    }
                    else
                    {
                        temp    = N[j + 1] / (uright - uleft);
                        N[j]    = saved + (uright - u) * temp;
                        saved   = (u - uleft) * temp;
                    }
                }
            }
            return N[0];
        }

        // same as OrigBasisFuns but allocate left/right scratch space ahead of time,
        // and compute N vector in place.
        // NOTE: In a threaded environment, a thread-local BasisFunInfo should be passed
        //       as an argument. Concurrent access to the same BFI will cause a data race.
        // 
        // NOTE: In this version, N is assumed to be size p+1, and we compute in place instead
        //       of using a "scratch" vector
        void FastBasisFuns(
            int                 cur_dim,        // current dimension
            T                   u,              // parameter value
            int                 span,           // index of span in the knots vector containing u
            vector<T>&          N,              // vector of (output) basis function values 
            BasisFunInfo<T>&    bfi) const      // scratch space
        {
            // nb. we do not need to zero out the entirety of N, since the existing entries of N 
            //     are never accessed (they are always overwritten first)
            N[0] = 1;   

            for (int j = 1; j <= p(cur_dim); j++)
            {
                // left[j] is u - the jth knot in the correct level to the left of span
                // right[j] is the jth knot in the correct level to the right of span - u
                bfi.left[j]  = u - tmesh.all_knots[cur_dim][span + 1 - j];
                bfi.right[j] = tmesh.all_knots[cur_dim][span + j] - u;

                T saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    T temp = N[r] / (bfi.right[r + 1] + bfi.left[j - r]);
                    N[r] = saved + bfi.right[r + 1] * temp;
                    saved = bfi.left[j - r] * temp;
                }
                N[j] = saved;
            }
        }

        // Faster version of DerBasisFuns.
        // * Utilizes BasisFunInfo to avoid allocating matrices on the fly.
        // * Stores reciprocal of knot differences to minimize divisions
        // * Matrix of derivs is size (nders+1)x(p+1), instead of (nders+1)x(nctrlpts)
        void FastBasisFunsDers(
            int                 cur_dim,        // current dimension
            T                   u,              // parameter value
            int                 span,           // index of span in the knots vector containing u
            int                 nders,          // number of derivatives
            vector<vector<T>>&  D,              // matrix of (output) basis function values/derivs
            BasisFunInfo<T>&    bfi) const      // scratch space
        {
            if (nders == 1)
                return FastBasisFunsDer1(cur_dim, u, span, D, bfi);

            assert(D.size() == nders+1); // PRECONDITION: D has been resized to fit all necessary derivs
            
            const int deg = p(cur_dim);

            // matrix from p. 70 of P&T
            // upper triangle is basis functions
            // lower triangle is reciprocal of knot differences
            bfi.ndu[0][0] = 1.0;

            // fill ndu / compute 0th derivatives
            for (int j = 1; j <= deg; j++)
            {
                bfi.left[j]  = u - tmesh.all_knots[cur_dim][span + 1 - j];
                bfi.right[j] = tmesh.all_knots[cur_dim][span + j] - u;

                T saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    // lower triangle
                    bfi.ndu[j][r] = 1 / (bfi.right[r + 1] + bfi.left[j - r]);
                    T temp = bfi.ndu[r][j - 1] * bfi.ndu[j][r];
                    // upper triangle
                    bfi.ndu[r][j] = saved + bfi.right[r + 1] * temp;
                    saved = bfi.left[j - r] * temp;
                }
                bfi.ndu[j][j] = saved;
            }

            // Copy 0th derivatives
            for (int j = 0; j <= deg; j++)
                D[0][j] = bfi.ndu[j][deg];            // TODO: compute these basis functions in-place in N above?

            // compute derivatives according to eq. 2.10
            // 1st row = first derivative, 2nd row = 2nd derivative, ...
            for (int r = 0; r <= deg; r++)
            {
                int s1, s2;                             // alternate rows in array a
                s1      = 0;
                s2      = 1;
                bfi.a[0][0] = 1.0;

                for (int k = 1; k <= nders; k++)        // over all the derivatives up to the d_th one
                {
                    T d    = 0.0;
                    int rk = r - k;
                    int pk = deg - k;

                    if (r >= k)
                    {
                        bfi.a[s2][0] = bfi.a[s1][0] * bfi.ndu[pk + 1][rk];
                        d            = bfi.a[s2][0] * bfi.ndu[rk][pk];
                    }

                    int j1, j2;
                    if (rk >= -1)
                        j1 = 1;
                    else
                        j1 = -rk;
                    if (r - 1 <= pk)
                        j2 = k - 1;
                    else
                        j2 = deg - r;

                    for (int j = j1; j <= j2; j++)
                    {
                        bfi.a[s2][j] = (bfi.a[s1][j] - bfi.a[s1][j - 1]) * bfi.ndu[pk + 1][rk + j];
                        d += bfi.a[s2][j] * bfi.ndu[rk + j][pk];
                    }

                    if (r <= pk)
                    {
                        bfi.a[s2][k] = -bfi.a[s1][k - 1] * bfi.ndu[pk + 1][r];
                        d += bfi.a[s2][k] * bfi.ndu[r][pk];
                    }

                    D[k][r] = d;
                    swap(s1, s2);
                }                                       // for k
            }                                           // for r

            // multiply through by the correct factors in eq. 2.10
            int r = deg;
            for (int k = 1; k <= nders; k++)
            {
                for (int i = 0; i <= deg; i++)
                {
                    D[k][i] *= r;
                }
                r *= deg - k;
            }
        }

        // Specialization of FastBasisFunsDers for 1st derivatives, which is much
        // simpler than the general case.  This method is called from 
        // FastBasisFunsDers when nders=1, so it should not need to be called by
        // the user.
        void FastBasisFunsDer1(
            int                 cur_dim,        // current dimension
            T                   u,              // parameter value
            int                 span,           // index of span in the knots vector containing u
            vector<vector<T>>&  D,              // matrix of (output) basis function values/derivs 
            BasisFunInfo<T>&    bfi) const      // scratch space
        {
            assert(D.size() == 2); // PRECONDITION: D has been resized to fit all necessary derivs
            
            const int deg = p(cur_dim);
            const int pk  = deg - 1;

            // matrix from p. 70 of P&T
            // upper triangle is basis functions
            // lower triangle is reciprocal of knot differences
            bfi.ndu[0][0] = 1.0;

            // fill ndu / compute 0th derivatives
            for (int j = 1; j <= deg; j++)
            {
                bfi.left[j]  = u - tmesh.all_knots[cur_dim][span + 1 - j];
                bfi.right[j] = tmesh.all_knots[cur_dim][span + j] - u;

                T saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    // lower triangle
                    bfi.ndu[j][r] = 1 / (bfi.right[r + 1] + bfi.left[j - r]);
                    T temp = bfi.ndu[r][j - 1] * bfi.ndu[j][r];
                    // upper triangle
                    bfi.ndu[r][j] = saved + bfi.right[r + 1] * temp;
                    saved = bfi.left[j - r] * temp;
                }
                bfi.ndu[j][j] = saved;
            }

            // Copy 0th derivatives
            for (int j = 0; j <= deg; j++)
                D[0][j] = bfi.ndu[j][deg];
                
            // Compute 1st derivatives
            T d = 0.0;
            D[1][0]     = -bfi.ndu[0][pk] * bfi.ndu[deg][0];
            D[1][deg]   = bfi.ndu[deg-1][pk] * bfi.ndu[deg][deg-1];
            for (int r = 1; r < deg; r++)
            {
                d = bfi.ndu[r-1][pk] * bfi.ndu[deg][r-1];
                d += -bfi.ndu[r][pk] * bfi.ndu[deg][r];

                D[1][r] = d;
            }

            // multiply through by the correct factors in eq. 2.10
            for (int i = 0; i <= deg; i++)
            {
                D[1][i] *= deg;
            }
        }

        // tmesh version of basis functions that computes one basis function at a time for each local knot vector
        // computes one row of basis function values for a given parameter value
        // writes results in a row of N
        // algorithm 2.2 of P&T, p. 70
        //
        // assumes N has been allocated by caller
        void BasisFuns(
                int                     cur_dim,    // current dimension
                T                       u,          // parameter value
                int                     span,       // index of span in the knots vector containing u
                MatrixX<T>&             N,          // matrix of (output) basis function values
                int                     row) const  // row in N of result
        {
            vector<T> loc_knots(p(cur_dim) + 2);

            // initialize row to 0
            N.row(row).setZero();

            for (auto j = 0; j < p(cur_dim) + 1; j++)
            {
                for (auto i = 0; i < p(cur_dim) + 2; i++)
                    loc_knots[i] = tmesh.all_knots[cur_dim][span - p(cur_dim) + j + i];

                N(row, span - p(cur_dim) + j) = OneBasisFun(cur_dim, u, loc_knots);
            }

            // debug
//             cerr << N << "\n---" << endl;
        }

        // computes and returns one basis function value for a given parameter value and local knot vector
        // based on algorithm 2.4 of P&T, p. 74
        //
        T OneBasisFun(
                int                     cur_dim,            // current dimension
                T                       u,                  // parameter value
                const vector<T>&        loc_knots) const    // local knot vector
        {
            vector<T> N(p(cur_dim) + 1);                    // triangular table result
            const vector<T>& U = loc_knots;                 // alias for knot vector for current dimension

            // corner case: 1 at right edge of local knot vector
            if (u == 1.0)
            {
                bool edge = true;
                for (auto j = 0; j < p(cur_dim) + 1; j++)
                {
                    if (loc_knots[1 + j] != 1.0)
                    {
                        edge = false;
                        break;
                    }
                }
                if (edge)
                    return 1.0;
                
                /* else return 0; 
                TODO? if not edge but u==1.0, then we should always return 0? */
            }

            // initialize 0-th degree functions
            for (auto j = 0; j <= p(cur_dim); j++)
            {
                if (u >= U[j] && u < U[j + 1])
                    N[j] = 1.0;
                else
                    N[j] = 0.0;
            }

            // compute triangular table
            T saved, uleft, uright, temp;
            for (auto k = 1; k <= p(cur_dim); k++)
            {
                if (N[0] == 0.0)
                    saved = 0.0;
                else
                    saved = ((u - U[0]) * N[0]) / (U[k] - U[0]);
                for (auto j = 0; j < p(cur_dim) - k + 1; j++)
                {
                    uleft     = U[j + 1];
                    uright    = U[j + k + 1];
                    if (N[j + 1] == 0.0)
                    {
                        N[j]    = saved;
                        saved   = 0.0;
                    }
                    else
                    {
                        temp    = N[j + 1] / (uright - uleft);
                        N[j]    = saved + (uright - u) * temp;
                        saved   = (u - uleft) * temp;
                    }
                }
            }
            return N[0];
        }

        // computes and returns one differentiated basis function value 
        // for a given parameter value and local knot vector
        // Algorithm 2.5 in P&T
        //
        // NOTE Algorithm 2.5 in P&T has an error: 
        //   In the loop "compute table of width k," Uright
        //   Should be U[i+j+p-k+jj+1], instead of U[i+j+p+jj+1]
        T OneDerBasisFun(
                int                     cur_dim,            // current dimension
                int                     der,                // derivative order
                T                       u,                  // parameter value
                const vector<T>&        loc_knots,          // local knot vector
                BasisFunInfo<T>&        bfi) const    
        {
            const vector<T>& U = loc_knots;                 // alias for knot vector for current dimension (size p+2)
            const int pc = p(cur_dim);

            // If not in local support
            if (u < U[0] || u >= U[pc + 1])
                return 0;

            T saved = 0, uleft = 0, uright = 0, temp = 0;

            // matrix from p. 70 of P&T
            // upper triangle is basis functions
            // lower triangle is knot differences
            // nn ~ bfi.ndu
            for (int j = 0; j <= pc; j++)
            {
                if (u >= U[j] && u < U[j+1])
                    bfi.ndu[j][0] = 1;
                else
                    bfi.ndu[j][0] = 0;
            }

            for (int k = 1; k <= pc; k++)
            {
                if (bfi.ndu[0][k-1] == 0) 
                    saved = 0;
                else 
                    saved = ((u - U[0]) * bfi.ndu[0][k-1]) / (U[k] - U[0]);
                
                for (int j = 0; j < pc - k + 1; j++)
                {
                    uleft = U[j+1];
                    uright = U[j+k+1];

                    if (bfi.ndu[j+1][k-1] == 0)
                    {
                        bfi.ndu[j][k] = saved;
                        saved = 0;
                    }
                    else
                    {
                        temp = bfi.ndu[j+1][k-1] / (uright - uleft);
                        bfi.ndu[j][k] = saved + (uright - u)*temp;
                        saved = (u-uleft) * temp;
                    }
                }
            }

            if (der == 0)
                return bfi.ndu[0][pc];    // bfi.ndu[0][pc] is the 0th-order derivative (function value)

            // Copy the necessary basis functions to a new buffer 'dertable'
            // dertable will compute intermediate calculations for the requested derivative
            // We make the copy so that nn is not overwritten (but this may not be necessary)
            // VectorX<T> dertable(der+1);
            // for (int j = 0; j <= der; j++)
            //     dertable(j) = nn(j, pc-der);

            // compute the derivative of order 'der'
            // NOTE: does not compute lower order derivatives
            for (int l = 1; l <= der; l++)
            {
                if (bfi.ndu[0][pc-der] == 0) 
                    saved = 0;
                else
                    saved = bfi.ndu[0][pc-der] / (U[pc - der + l] - U[0]);

                for (int j = 0; j < der - l + 1; j++)
                {
                    uleft = U[j+1];
                    uright = U[j + 1 + pc - der + l];
                    if (bfi.ndu[j+1][pc-der] == 0)
                    {
                        bfi.ndu[j][pc-der] = (pc - der + l)*saved;
                        saved = 0;
                    }
                    else
                    {
                        temp = bfi.ndu[j+1][pc-der] / (uright - uleft);
                        bfi.ndu[j][pc-der] = (pc - der + l)*(saved - temp);
                        saved = temp;
                    }
                }
            }

            return bfi.ndu[0][pc-der];
        }

//         // DEPRECATED
//         // computes and returns one (the ith) basis function value for a given parameter value
//         // algorithm 2.4 of P&T, p. 74
//         //
//         T OneBasisFun(
//                 int                     cur_dim,        // current dimension
//                 T                       u,              // parameter value
//                 int                     i) const        // compute the ith basis function, where span - p(cur_dim) <= i <= span
//         {
//             vector<T> N(p(cur_dim) + 1);                // triangular table result
//             const vector<T>& U = tmesh.all_knots[cur_dim];    // alias for knot vector for current dimension
// 
//             // 1 at edges of global knot vector
//             if ( (i == 0 && u == U[0]) || ( i == U.size() - p(cur_dim) - 2 && u == U.back()) )
//                 return 1.0;
// 
//             // zero outside of local knot vector
//             if (u < U[i] || u >= U[i + p(cur_dim) + 1])
//                 return 0.0;
// 
//             // initialize 0-th degree functions
//             for (auto j = 0; j <= p(cur_dim); j++)
//             {
//                 if (u >= U[i + j] && u < U[i + j + 1])
//                     N[j] = 1.0;
//                 else
//                     N[j] = 0.0;
//             }
// 
//             // compute triangular table
//             T saved, uleft, uright, temp;
//             for (auto k = 1; k <= p(cur_dim); k++)
//             {
//                 if (N[0] == 0.0)
//                     saved = 0.0;
//                 else
//                     saved = ((u - U[i]) * N[0]) / (U[i + k] - U[i]);
//                 for (auto j = 0; j < p(cur_dim) - k + 1; j++)
//                 {
//                     uleft     = U[i + j + 1];
//                     uright    = U[i + j + k + 1];
//                     if (N[j + 1] == 0.0)
//                     {
//                         N[j]    = saved;
//                         saved   = 0.0;
//                     }
//                     else
//                     {
//                         temp    = N[j + 1] / (uright - uleft);
//                         N[j]    = saved + (uright - u) * temp;
//                         saved   = (u - uleft) * temp;
//                     }
//                 }
//             }
//             return N[0];
//         }

        // DEPRECATED
//         // computes one row of basis function values for a given parameter value
//         // writes results in a row of N
//         // algorithm 2.2 of P&T, p. 70
//         // tmesh version for single tensor skipping knots at a level deeper than current tensor
//         //
//         // assumes N has been allocated by caller
//         void BasisFuns(
//                 const TensorProduct<T>& tensor,     // current tensor product
//                 int                     cur_dim,    // current dimension
//                 T                       u,          // parameter value
//                 int                     span,       // index of span in the knots vector containing u, relative to ko
//                 MatrixX<T>&             N,          // matrix of (output) basis function values
//                 int                     row) const  // row in N of result
//         {
//             // initialize row to 0
//             N.row(row).setZero();
// 
//             // init
//             vector<T> scratch(p(cur_dim) + 1);                  // scratchpad, same as N in P&T p. 70
//             scratch[0] = 1.0;
// 
//             // temporary recurrence results
//             // left(j)  = u - knots(span + 1 - j)
//             // right(j) = knots(span + j) - u
//             vector<T> left(p(cur_dim) + 1);
//             vector<T> right(p(cur_dim) + 1);
// 
//             // fill N
//             int j_left = 1;             // j_left and j_right are like j in the loop below but skip over knots not in the right level
//             int j_right = 1;
//             for (int j = 1; j <= p(cur_dim); j++)
//             {
//                 // skip knots not in current level
//                 while (tmesh.all_knot_levels[cur_dim][span + 1 - j_left] != tensor.level)
//                 {
//                     j_left++;
//                     assert(span + 1 - j_left >= 0);
//                 }
//                 // left[j] is u = the jth knot in the correct level to the left of span
//                 left[j]  = u - tmesh.all_knots[cur_dim][span + 1 - j_left];
//                 while (tmesh.all_knot_levels[cur_dim][span + j_right] != tensor.level)
//                 {
//                     j_right++;
//                     assert(span + j_right < tmesh.all_knot_levels[cur_dim].size());
//                 }
//                 // right[j] = the jth knot in the correct level to the right of span - u
//                 right[j] = tmesh.all_knots[cur_dim][span + j_right] - u;
//                 j_left++;
//                 j_right++;
// 
//                 T saved = 0.0;
//                 for (int r = 0; r < j; r++)
//                 {
//                     T temp = scratch[r] / (right[r + 1] + left[j - r]);
//                     scratch[r] = saved + right[r + 1] * temp;
//                     saved = left[j - r] * temp;
//                 }
//                 scratch[j] = saved;
//             }
// 
//             // copy scratch to N
//             for (int j = 0; j < p(cur_dim) + 1; j++)
//                 N(row, span - p(cur_dim) + j) = scratch[j];
//         }

        // computes one row of derivattive of basis function values for a given parameter value
        // writes results in a row of N
        // computes first k derivatives of one row of basis function values for a given parameter value
        // output is ders, with nders + 1 rows, one for each derivative (N, N', N'', ...)
        // including origin basis functions (0-th derivatives)
        // assumes ders has been allocated by caller (nders + 1 rows, # control points cols)
        // Alg. 2.3, p. 72 of P&T
        void DerBasisFuns(
                int         cur_dim,        // current dimension
                T           u,              // parameter value
                int         span,           // index of span in the knots vector containing u, relative to ko
                int         nders,          // number of derivatives
                MatrixX<T>& ders) const     // (output) basis function derivatives
        {
            // matrix from p. 70 of P&T
            // upper triangle is basis functions
            // lower triangle is knot differences
            MatrixX<T> ndu(p(cur_dim) + 1, p(cur_dim) + 1);
            ndu(0, 0) = 1.0;

            // temporary recurrence results
            // left(j)  = u - knots(span + 1 - j)
            // right(j) = knots(span + j) - u
            VectorX<T> left(p(cur_dim) + 1);
            VectorX<T> right(p(cur_dim) + 1);

            // fill ndu
            for (int j = 1; j <= p(cur_dim); j++)
            {
                left(j)  = u - tmesh.all_knots[cur_dim][span + 1 - j];
                right(j) = tmesh.all_knots[cur_dim][span + j] - u;

                T saved = 0.0;
                for (int r = 0; r < j; r++)
                {
                    // lower triangle
                    ndu(j, r) = right(r + 1) + left(j - r);
                    T temp = ndu(r, j - 1) / ndu(j, r);
                    // upper triangle
                    ndu(r, j) = saved + right(r + 1) * temp;
                    saved = left(j - r) * temp;
                }
                ndu(j, j) = saved;
            }

            // two most recently computed rows a_{k,j} and a_{k-1,j}
            MatrixX<T> a(2, p(cur_dim) + 1);

            // initialize ders and set 0-th row with the basis functions = 0-th derivatives
            ders = MatrixX<T>::Zero(ders.rows(), ders.cols());
            for (int j = 0; j <= p(cur_dim); j++)
                ders(0, span - p(cur_dim) + j) = ndu(j, p(cur_dim));

            // compute derivatives according to eq. 2.10
            // 1st row = first derivative, 2nd row = 2nd derivative, ...
            for (int r = 0; r <= p(cur_dim); r++)
            {
                int s1, s2;                             // alternate rows in array a
                s1      = 0;
                s2      = 1;
                a(0, 0) = 1.0;

                for (int k = 1; k <= nders; k++)        // over all the derivatives up to the d_th one
                {
                    T d    = 0.0;
                    int rk = r - k;
                    int pk = p(cur_dim) - k;

                    if (r >= k)
                    {
                        a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                        d        = a(s2, 0) * ndu(rk, pk);
                    }

                    int j1, j2;
                    if (rk >= -1)
                        j1 = 1;
                    else
                        j1 = -rk;
                    if (r - 1 <= pk)
                        j2 = k - 1;
                    else
                        j2 = p(cur_dim) - r;

                    for (int j = j1; j <= j2; j++)
                    {
                        a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                        d += a(s2, j) * ndu(rk + j, pk);
                    }

                    if (r <= pk)
                    {
                        a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r);
                        d += a(s2, k) * ndu(r, pk);
                    }

                    ders(k, span - p(cur_dim) + r) = d;
                    swap(s1, s2);
                }                                       // for k
            }                                           // for r

            // multiply through by the correct factors in eq. 2.10
            int r = p(cur_dim);
            for (int k = 1; k <= nders; k++)
            {
                ders.row(k) *= r;
                r *= (p(cur_dim) - k);
            }
        }

        // compute rational (weighted) NtN from nonrational (unweighted) N
        // ie, convert basis function coefficients to rational ones with weights
        void Rationalize(
                int                 k,                      // current dimension
                const VectorX<T>&   weights,                // weights of control points
                const MatrixX<T>&   N,                      // basis function coefficients
                MatrixX<T>&         NtN_rat) const          // (output) rationalized Nt * N
        {
            // compute rational denominators for input points
            VectorX<T> denom(N.rows());             // rational denomoninator for param of each input point
            for (int j = 0; j < N.rows(); j++)
                denom(j) = (N.row(j).cwiseProduct(weights.transpose())).sum();

            //     cerr << "denom:\n" << denom << endl;

            // "rationalize" N and Nt
            // ie, convert their basis function coefficients to rational ones with weights
            MatrixX<T> N_rat = N;                   // need a copy because N will be reused for other curves
            for (auto i = 0; i < N.cols(); i++)
                N_rat.col(i) *= weights(i);
            for (auto j = 0; j < N.rows(); j++)
                N_rat.row(j) /= denom(j);

            // multiply rationalized Nt and N
            NtN_rat = N_rat.transpose() * N_rat;

            // debug
            //         cerr << "k " << k << " NtN:\n" << NtN << endl;
            //         cerr << " NtN_rat:\n" << NtN_rat << endl;
        }

        // knot insertion into tensor product
        // Boehm's knot insertion algorithm
        // assumes all control points needed are contained within one tensor
        // This version returns new knots, levels, control points, weights as arguments and does not copy them into tmesh
        // This version is for a new knot that does not yet appear in the tmesh
        // TODO: expensive deep copies
        void NewKnotInsertion(const VectorX<T>&        param,                  // new knot value to be inserted
                              TensorIdx                tensor_idx,             // existing tensor product for insertion
                              VectorXi&                new_nctrl_pts,          // (output) new number of control points in each dim., compare with existing tensor to see which dims. changed
                              vector<vector<T>>&       new_all_knots,          // (output) new global all knots
                              vector<vector<int>>&     new_all_knot_levels,    // (output) new global all knot levels
                              MatrixX<T>&              new_ctrl_pts,           // (output) new local control points for this tensor
                              VectorX<T>&              new_weights,            // (output) new local weights for this tensor
                              vector<int>&             new_knot_idxs,          // (output) inserted positions of new knot in each dim (same position as existing if no change)
                              vector<int>&             new_ctrl_pt_idxs) const // (output) inserted position of new ctrl pt in each dim (same position as existing if no change)
        {
            // debug
//             fmt::print(stderr, "NewKnotInsertion(): ctrl_pts before inserting param [{}]:\n{}\n", param.transpose(), tensor.ctrl_pts);

            auto& tensor = tmesh.tensor_prods[tensor_idx];
            new_nctrl_pts = tensor.nctrl_pts;
            NewVolKnotIns(param, tensor_idx, new_all_knots, new_all_knot_levels, new_ctrl_pts,
                    new_weights, new_nctrl_pts, new_knot_idxs, new_ctrl_pt_idxs);

            // debug
//             fmt::print(stderr, "NewKnotInsertion(): ctrl_pts after inserting param [{}]:\n{}\n", param.transpose(), new_ctrl_pts);

        }

        private:

        // curve knot insertion
        // Algorithm 5.1 from P&T p. 151 (Boehm's knot insertion algorithm)
        // this version assumes the new knot does not yet exist in the knot vector; updates both knots and control points
        // not for inserting a duplicate knot (does not handle knot multiplicity > 1)
        // original algorithm from P&T did handle multiplicity, but I simplified
        void NewCurveKnotIns(
                const VectorX<T>&       param,                  // new knot value to be inserted
                TensorIdx               tensor_idx,             // original tensor with existing volume of control points from which curve is derived
                int                     cur_dim,                // current dimension
                const vector<T>&        old_knots,              // old knot vector in cur. dim.
                const vector<int>&      old_knot_levels,        // old knot levels in cur. dim.
                const MatrixX<T>&       old_ctrl_pts,           // old control points of curve
                const VectorX<T>&       old_weights,            // old control point weights of curve
                int                     level,                  // level of new knot to be inserted
                vector<T>&              new_knots,              // (output) new knot vector in cur. dim.
                vector<int>&            new_knot_levels,        // (output) new knot levels in cur. dim.
                MatrixX<T>&             new_ctrl_pts,           // (output) new control points of curve
                VectorX<T>&             new_weights,            // (output) new control point weights of curve
                int&                    new_knot_idx,           // (output) inserted position of new knot
                int&                    new_ctrl_pt_idx) const  // (output)inserted position of new ctrl pt
        {
            // debug
            bool debug = false;
//             if (cur_dim == 1 && fabs(param(0) - 0.27778) < 0.001 && fabs(param(1) - 0.11558) < 0.001)
//                 debug = true;

            auto&   tensor  = tmesh.tensor_prods[tensor_idx];
            T       u       = param(cur_dim);                   // parameter in current dim.

            new_knots.resize(old_knots.size() + 1);
            new_knot_levels.resize(old_knot_levels.size() + 1);
            new_ctrl_pts.resize(old_ctrl_pts.rows() + 1, old_ctrl_pts.cols());
            new_weights.resize(old_weights.size() + 1);
            MatrixX<T> temp_ctrl_pts(p(cur_dim) + 1, old_ctrl_pts.cols());
            VectorX<T> temp_weights(p(cur_dim) + 1);

            // anchor corresponding to param in all dims
            vector<KnotIdx> anchor(dom_dim);
            for (auto i = 0; i < dom_dim; i++)
                anchor[i] = tmesh.FindSpan(i, param(i), tensor);

            int global_span    = anchor[cur_dim];               // global knot span of param in current dim.
            T eps       = 1.0e-8;
            if (fabs(old_knots[global_span] - u) < eps)         // not for multiple knots
                throw MFAError(fmt::format("Error: NewCurveKnotIns attempting to insert duplicate knot in dim {} global_span {} knot {} u {}\n",
                        cur_dim, global_span, tmesh.all_knots[cur_dim][global_span], u));

            // load new knot vector
            for (auto i = 0; i <= global_span; i++)
            {
                new_knots[i]        = old_knots[i];
                new_knot_levels[i]  = old_knot_levels[i];
            }
            new_knots[global_span + 1]         = u;
            new_knot_levels[global_span + 1]   = level;
            for (auto i = global_span + 1; i < old_knots.size(); i++)
            {
                new_knots[i + 1]        = old_knots[i];
                new_knot_levels[i + 1]  = old_knot_levels[i];
            }
            new_knot_idx = global_span + 1;

            // convert span to local knot index and control point index in tensor
            // use global_span for indexing knots in global knot vector
            // use local_span for indexing knots in local tensor
            // use local_span + shift for indexing control points in local tensor
            int local_span = tmesh.global2local_knot_idx(global_span, tensor, cur_dim);
            int shift = tensor.knot_mins[cur_dim] == 0 ? 0 : (p(cur_dim) + 1) / 2;      // shift ctrl pt indices for interior tensors w/o clamped end

            // save unaltered control points and weights
            for (auto i = 0; i <= local_span - p(cur_dim) + shift; i++)     // control points before the p altered ones
            {
                new_ctrl_pts.row(i) = old_ctrl_pts.row(i);
                new_weights(i)      = old_weights(i);
            }
            for (auto i = local_span + shift; i < old_ctrl_pts.rows(); i++) // control points after the p altered ones
            {
                new_ctrl_pts.row(i + 1) = old_ctrl_pts.row(i);
                new_weights(i + 1)      = old_weights(i);
            }

            // set up p+1 temp control points for computing the p altered control points
            int ntemp_ctrl = 0;
            for (auto i = 0; i <= p(cur_dim); i++)
            {
                // check if we run out of control points in this tensor
                if (local_span - p(cur_dim) + i + shift < 0)
                    continue;
                if (local_span - p(cur_dim) + i + shift >= old_ctrl_pts.rows())
                    break;

                temp_ctrl_pts.row(i)    = old_ctrl_pts.row(local_span - p(cur_dim) + i + shift);
                temp_weights(i)         = old_weights(local_span - p(cur_dim) + i + shift);
                ntemp_ctrl++;
            }

            // compute knots needed to recompute modified control points
            int nprev_knots     = (p(cur_dim) + 1) / 2 + 1;         // number of knot intersections before anchor
            int nnext_knots     = p(cur_dim) + 1;                   // number of knot intersections after anchor
            vector<KnotIdx> prev_knots(nprev_knots);                // knot intersections before anchor
            vector<KnotIdx> next_knots(nnext_knots);                // knot intersections after anchor
            tmesh.prev_knot_intersections_dim(anchor, tensor_idx, cur_dim, nprev_knots, 0, prev_knots);
            tmesh.next_knot_intersections_dim(anchor, tensor_idx, cur_dim, nnext_knots, 0, next_knots);

            // compute p(cur_dim) new control points, one of which is newly inserted and rest are modified
            // the newly inserted one is temp_ctrl_pts[p/2], or new_ctrl_pts[local_span - (p+1)/2 + 1 + shift]
            int nrecomp_ctrl    = 0;
            for (auto i = 0; i < p(cur_dim); i++)
            {
                // check if we ran out of control points in this tensor
                if (local_span - p(cur_dim) + i + shift < 0)
                    continue;
                if (local_span - p(cur_dim) + i + shift >= old_ctrl_pts.rows() || i + 1 >= ntemp_ctrl)
                    break;

                auto ofst1 = prev_knots[i];                             // ofst1 = global_span + 1 + i - p(cur_dim)
                auto ofst2 = next_knots[i + 1];                         // ofst2 = global_span + 1 + i
                T alpha    = (u - old_knots[ofst1]) / (old_knots[ofst2] - old_knots[ofst1]);

                // debug
                MatrixX<T> old_ctrl0 = temp_ctrl_pts.row(i);
                MatrixX<T> old_ctrl1 = temp_ctrl_pts.row(i + 1);

                temp_ctrl_pts.row(i)    = alpha * temp_ctrl_pts.row(i + 1) + (1.0 - alpha) * temp_ctrl_pts.row(i);
                temp_weights(i)         = alpha * temp_weights(i + 1) + (1.0 - alpha) * temp_weights(i);
                nrecomp_ctrl++;

                // debug
//                 if (i == p(cur_dim) / 2)
//                     fmt::print(stderr, "new ctrl pt {} at i {} alpha {} param u {} global_span {} old_ctrl0 {} old_ctrl1 {}\n",
//                             temp_ctrl_pts.row(i), i, alpha, u, global_span, old_ctrl0, old_ctrl1);
            }

            // load modified p(cur_dim) control points
            for (auto i = 0; i < nrecomp_ctrl; i++)
            {
                new_ctrl_pts.row(local_span - p(cur_dim) + 1 + i + shift) = temp_ctrl_pts.row(i);
                new_weights(local_span - p(cur_dim) + 1 + i + shift)      = temp_weights(i);
            }
            new_ctrl_pt_idx = local_span - (p(cur_dim) + 1) / 2 + 1 + shift;

            // debug
//             fmt::print(stderr, "NewCurveKnotInsertion(): inserting new curve ctrl pt at idx {} param u {} with value {}\n",
//             new_ctrl_pt_idx, u, new_ctrl_pts.row(new_ctrl_pt_idx));
        }

        // volume knot insertion
        // n-dimensional generalization of Algorithm 5.3 from P&T p. 155 (Boehm's knot insertion algorithm)
        // but without the performance optimizations for now (TODO)
        // not for inserting a duplicate knot (does not handle knot multiplicity > 1)
        // original algorithm from P&T did handle multiplicity, but I simplified
        // this version assumes the new knot does not yet exist, updates both knots and control points
        void NewVolKnotIns(
                const VectorX<T>&           param,                  // new knot value to be inserted
                TensorIdx                   tensor_idx,             // tensor containing existing control points
                vector<vector<T>>&          new_knots,              // (output) new knots
                vector<vector<int>>&        new_knot_levels,        // (output) new knot levels
                MatrixX<T>&                 new_ctrl_pts,           // (output) new control points
                VectorX<T>&                 new_weights,            // (output) new control point weights
                VectorXi&                   nctrl_pts,              // (output) number of control points in all dims, compare with existing tensor to see change in each dim.
                vector<int>&                new_knot_idxs,          // (output) inserted positions of new knot in each dim (same pos as existing if no change)
                vector<int>&                new_ctrl_pt_idxs) const // (output)inserted position of new ctrl pt in each dim (same pos as existing if no change)
        {
            // debug
            bool debug = false;
//             if (fabs(param(0) - 0.27778) < 0.001 && fabs(param(1) - 0.11558) < 0.001)
//                 debug = true;

            auto&                         tensor          = tmesh.tensor_prods[tensor_idx];
            const vector<vector<T>>&      old_knots       = tmesh.all_knots;
            const vector<vector<int>>&    old_knot_levels = tmesh.all_knot_levels;
            const MatrixX<T>&             old_ctrl_pts    = tensor.ctrl_pts;
            const VectorX<T>&             old_weights     = tensor.weights;
            int                           level           = tensor.level;

            size_t old_cs, new_cs;                                  // stride for old and new control points in curve in cur. dim
            new_knot_idxs.resize(dom_dim, -1);
            new_ctrl_pt_idxs.resize(dom_dim, -1);

            // determine new sizes of control points, weights, knots, knot levels in each dim
            // a knot may be new in one dimension (inserted) and same in another dimension (not inserted)
            VectorXi new_nctrl_pts = nctrl_pts;
            for (auto k = 0; k < dom_dim; k++)
            {
                // check if the knot exists in this dimension already
                int span    = tmesh.FindSpan(k, param(k), tensor);  // global knot span for this tensor
                T eps       = 1.0e-8;

                // knot is new
                if (fabs(old_knots[k][span] - param(k)) > eps)      // knot is new in this dim.
                    new_nctrl_pts(k)++;

                // knot exists, set the new_knot_idxs and new_ctrl_pt_idxs to existing values
                else
                {
                    new_knot_idxs[k]    = span;
                    new_ctrl_pt_idxs[k] = tmesh.anchor_ctrl_pt_dim(tensor, k, span);
                }
            }

            new_ctrl_pts.resize(new_nctrl_pts.prod(), old_ctrl_pts.cols());
            new_weights.resize(new_ctrl_pts.rows());
            new_knots.resize(dom_dim);
            new_knot_levels.resize(dom_dim);

            // double buffer for new control points and weights (new_ctrl_pts, new_ctrl_pts1; new_weights, new_weights1)
            // so that in alternating dimensions, the output of previous dimension can be input of next dimension
            MatrixX<T> new_ctrl_pts1(new_ctrl_pts.rows(), new_ctrl_pts.cols());
            VectorX<T> new_weights1(new_weights.size());
            new_ctrl_pts1.block(0, 0, old_ctrl_pts.rows(), old_ctrl_pts.cols()) = old_ctrl_pts;
            new_weights1.segment(0, old_weights.rows())                         = old_weights;

            for (auto k = 0; k < dom_dim; k++)                      // for all domain dimensions
            {
                // debug
//                 fmt::print(stderr, "NewVolKnotInsertion(): param [{}] dim {}\n", param.transpose(), k);

                // check if the knot exists in this dim already (not inserted)
                if (nctrl_pts(k) == new_nctrl_pts(k))
                {
                    if (k % 2 == 0)
                    {
                        new_ctrl_pts    = new_ctrl_pts1;
                        new_weights     = new_weights1;
                    }
                    else
                    {
                        new_ctrl_pts1   = new_ctrl_pts;
                        new_weights1    = new_weights;
                    }

                    old_cs          = (k == 0) ? 1 : old_cs * new_nctrl_pts(k - 1); // stride between curve control points before insertion
                    new_cs          = (k == 0) ? 1 : new_cs * new_nctrl_pts(k - 1); // stride between curve control points before insertion

                    continue;
                }

                // resize new knots, levels, control points, weights
                new_knots[k].resize(old_knots[k].size() + 1);
                new_knot_levels[k].resize(old_knot_levels[k].size() + 1);

                // number of curves in this dimension before knot insertion
                // current dimension contributes no curves, hence the division by number of control points in cur. dim.
                size_t old_ncurves = new_nctrl_pts.prod() / new_nctrl_pts(k);

                vector<size_t> old_co(old_ncurves);                             // old starting curve points in current dim.
                old_co[0]       = 0;
                size_t old_coo  = 0;                                            // old co at start of contiguous sequence
                old_cs          = (k == 0) ? 1 : old_cs * new_nctrl_pts(k - 1); // stride between curve control points before insertion

                // curve offsets for curves before knot insertion
                for (auto j = 1; j < old_ncurves; j++)
                {
                    if (j % old_cs)
                        old_co[j] = old_co[j - 1] + 1;
                    else
                    {
                        old_co[j] = old_coo + old_cs * nctrl_pts(k);
                        old_coo   = old_co[j];
                    }
                }

                // number of curves in this dimension after knot insertion
                // current dimension contributes no curves, hence the division by number of control points in cur. dim.
                size_t new_ncurves = new_nctrl_pts.prod() / new_nctrl_pts(k);
                vector<size_t> new_co(new_ncurves);                             // new starting curve points in current dim.
                new_co[0]       = 0;
                size_t new_coo  = 0;                                            // new co at start of contiguous sequence
                new_cs          = (k == 0) ? 1 : new_cs * new_nctrl_pts(k - 1); // stride between curve control points before insertion

                // curve offsets for curves after knot insertion
                for (auto j = 1; j < new_ncurves; j++)
                {
                    if (j % new_cs)
                        new_co[j] = new_co[j - 1] + 1;
                    else
                    {
                        new_co[j] = new_coo + new_cs * new_nctrl_pts(k);
                        new_coo   = new_co[j];
                    }
                }


#ifdef MFA_TBB      // TBB version

                // thread-local DecodeInfo
                // ref: https://www.threadingbuildingblocks.org/tutorial-intel-tbb-thread-local-storage
                enumerable_thread_specific<MatrixX<T>> old_curve_ctrl_pts, new_curve_ctrl_pts;  // old and new control points for one curve
                enumerable_thread_specific<VectorX<T>> old_curve_weights, new_curve_weights;    // old and new weights for one curve

                parallel_for (size_t(0), old_ncurves, [&] (size_t j)            // for all the curves in this dimension
                {
                    // debug
                    // fprintf(stderr, "j=%ld curve\n", j);

                    int new_knot_idx, new_ctrl_pt_idx;                          // location of inserted knot and control point

                    // copy one curve of old curve control points and weights
                    if (k % 2 == 0)
                    CtrlPts2CtrlCurve(new_ctrl_pts1, new_weights1, old_curve_ctrl_pts.local(),
                            old_curve_weights.local(), nctrl_pts, k, old_co[j], old_cs);
                    else
                    CtrlPts2CtrlCurve(new_ctrl_pts, new_weights, old_curve_ctrl_pts.local(),
                            old_curve_weights.local(), nctrl_pts, k, old_co[j], old_cs);

                    // insert a knot in one curve of control points
                    NewCurveKnotIns(param, tensor_idx, k, old_knots[k], old_knot_levels[k], old_curve_ctrl_pts.local(),
                            old_curve_weights.local(), level, new_knots[k], new_knot_levels[k], new_curve_ctrl_pts.local(),
                            new_curve_weights.local(), new_knot_idx, new_ctrl_pt_idx);

                    // copy new curve control points and weights
                    if (k % 2 == 0)
                        CtrlCurve2CtrlPts(new_curve_ctrl_pts.local(), new_curve_weights.local(),
                                new_ctrl_pts, new_weights, new_nctrl_pts, k, new_co[j], new_cs);
                    else
                        CtrlCurve2CtrlPts(new_curve_ctrl_pts.local(), new_curve_weights.local(),
                                new_ctrl_pts1, new_weights1, new_nctrl_pts, k, new_co[j], new_cs);

                    // record the inserted knot and control point location in this dimension
                    // same for every curve, only need to record first curve
                    if (j == 0)
                    {
                        new_knot_idxs[k]    = new_knot_idx;
                        new_ctrl_pt_idxs[k] = new_ctrl_pt_idx;
                    }
                });

#endif              // end TBB version

#if defined(MFA_SERIAL)  || defined(MFA_KOKKOS)    // serial version or kokkos

                MatrixX<T> old_curve_ctrl_pts, new_curve_ctrl_pts;              // old and new control points for one curve
                VectorX<T> old_curve_weights, new_curve_weights;                // old and new weights for one curve

                for (size_t j = 0; j < old_ncurves; j++)                        // for all curves in this dimension
                {
                    int new_knot_idx, new_ctrl_pt_idx;                          // location of inserted knot and control point

                    // copy one curve of old curve control points and weights
                    if (k % 2 == 0)
                        CtrlPts2CtrlCurve(new_ctrl_pts1, new_weights1, old_curve_ctrl_pts,
                                old_curve_weights, nctrl_pts, k, old_co[j], old_cs);
                    else
                        CtrlPts2CtrlCurve(new_ctrl_pts, new_weights, old_curve_ctrl_pts,
                                old_curve_weights, nctrl_pts, k, old_co[j], old_cs);

                    // debug: print the last curve
//                     if (debug && j == old_ncurves - 1)
//                         fmt::print(stderr, "NewVolKnotIns(): param [{}] old_curve_ctrl_pts:\n [{}]\n", param.transpose(), old_curve_ctrl_pts.transpose());

                    // insert a knot in one curve of control points
                    NewCurveKnotIns(param, tensor_idx, k, old_knots[k], old_knot_levels[k], old_curve_ctrl_pts,
                            old_curve_weights, level, new_knots[k], new_knot_levels[k], new_curve_ctrl_pts,
                            new_curve_weights, new_knot_idx, new_ctrl_pt_idx);

                    // debug: print the last curve
//                     if (debug && j == old_ncurves - 1)
//                         fmt::print(stderr, "NewVolKnotIns(): param [{}] new_curve_ctrl_pts:\n [{}]\n", param.transpose(), new_curve_ctrl_pts.transpose());

                    // copy new curve control points and weights
                    if (k % 2 == 0)
                        CtrlCurve2CtrlPts(new_curve_ctrl_pts, new_curve_weights,
                                new_ctrl_pts, new_weights, new_nctrl_pts, k, new_co[j], new_cs);
                    else
                        CtrlCurve2CtrlPts(new_curve_ctrl_pts, new_curve_weights,
                                new_ctrl_pts1, new_weights1, new_nctrl_pts, k, new_co[j], new_cs);

                    // record the inserted knot and control point location in this dimension
                    // same for every curve, only need to record first curve
                    if (j == 0)
                    {
                        new_knot_idxs[k]    = new_knot_idx;
                        new_ctrl_pt_idxs[k] = new_ctrl_pt_idx;
                    }
                }

#endif              // end serial version

            }   // for all domain dimensions

            // update final output
            nctrl_pts = new_nctrl_pts;
            // odd domain dimensions: result already in the right double buffer, new_ctrl_pts, new_weights
            // even domain dimensions: result ends up in other double buffer, new_ctrl_pts1, new_weights1, and needs to be copied
            if (dom_dim % 2 == 0)
            {
                new_ctrl_pts    = new_ctrl_pts1;
                new_weights     = new_weights1;
            }
        }

        // copy from full set of control points to one control curve
        // TODO: deep copy (expensive)
        void CtrlPts2CtrlCurve(
                const MatrixX<T>&   all_ctrl_pts,       // control points in all dims
                const VectorX<T>&   all_weights,        // weights in all dims
                MatrixX<T>&         curve_ctrl_pts,     // (output) control points for curve in one dim.
                VectorX<T>&         curve_weights,      // (output) weights for curve in one dim.
                const VectorXi&     nctrl_pts,          // number of control points in all dims
                size_t              cur_dim,            // current dimension
                size_t              co,                 // starting ofst for reading domain pts
                size_t              cs) const           // stride for reading domain points
        {
            curve_ctrl_pts.resize(nctrl_pts(cur_dim), all_ctrl_pts.cols());
            curve_weights.resize(nctrl_pts(cur_dim));

            for (auto i = 0; i < nctrl_pts(cur_dim); i++)
            {
                curve_ctrl_pts.row(i)   = all_ctrl_pts.row(co + i * cs);
                curve_weights(i)        = all_weights(co + i * cs);
            }
        }

        // copy from one control curve to full set of control points
        // TODO: deep copy (expensive)
        // assumes full control points and weights are the correct size (does not resize)
        void CtrlCurve2CtrlPts(
                const MatrixX<T>&   curve_ctrl_pts,     // control points for curve in one dim.
                const VectorX<T>&   curve_weights,      // weights for curve in one dim.
                MatrixX<T>&         all_ctrl_pts,       // (output) control points in all dims
                VectorX<T>&         all_weights,        // (output) weights in all dims
                const VectorXi&     nctrl_pts,          // number of control points in all dims
                size_t              cur_dim,            // current dimension
                size_t              co,                 // starting ofst for reading domain pts
                size_t              cs) const           // stride for reading domain points
        {
            for (auto i = 0; i < nctrl_pts(cur_dim); i++)
            {
                all_ctrl_pts.row(co + i * cs)   = curve_ctrl_pts.row(i);
                all_weights(co + i * cs)        = curve_weights(i);
            }
        }

        // compute knots
        // n-d version of eqs. 9.68, 9.69, P&T
        // tmesh version
        // 
        // structured input only
        //
        // the set of knots (called U in P&T) is the set of breakpoints in the parameter space between
        // different basis functions. These are the breaks in the piecewise B-spline approximation
        //
        // nknots = n + p + 2
        // eg, for p = 3 and nctrl_pts = 7, n = nctrl_pts - 1 = 6, nknots = 11
        // let knots = {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}
        // there are p + 1 external knots at each end: {0, 0, 0, 0} and {1, 1, 1, 1}
        // there are n - p internal knots: {0.25, 0.5, 0.75}
        // there are n - p + 1 internal knot spans [0,0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1)
        //
        // resulting knots are same for all curves and stored once for each dimension (1st dim knots, 2nd dim, ...)
        // total number of knots is the sum of number of knots over the dimensions, much less than the product
        // assumes knots were allocated by caller
        void Knots(
                const PointSet<T>&         input,                  // input domain
                // const VectorXi&             ndom_pts,               // number of input points in each dim.
                // const vector<vector<T>>&    params,                 // parameters for input points[dimension][index]
                Tmesh<T>&                   tmesh,                  // (output) tmesh
                int verbose) const                                  // verbosity level
        {
            if (!input.is_structured())
            {
                cerr << "ERROR: Cannot set curve knots from unstructured input" << endl;
                exit(1);
            }

            vector<vector<T>>& params = input.params.param_grid;  // reference to array of params

            for (size_t k = 0; k < dom_dim; k++)                    // for all domain dimensions
            {
                // TODO: hard-coded for first tensor product of the tmesh
                int nctrl_pts = tmesh.tensor_prods[0].nctrl_pts(k);

                int nknots = nctrl_pts + p(k) + 1;                  // number of knots in current dim

                // in P&T, d is the ratio of number of input points (r+1) to internal knot spans (n-p+1)
                //         T d = (T)(ndom_pts(k)) / (nctrl_pts - p(k));         // eq. 9.68, r is P&T's m
                // but I prefer d to be the ratio of input spans r to internal knot spans (n-p+1)
                T d = (T)(input.ndom_pts(k) - 1) / (nctrl_pts - p(k));

                // compute n - p internal knots
                size_t param_idx = 0;                               // index into params
                for (int j = 1; j <= nctrl_pts - p(k) - 1; j++)     // eq. 9.69
                {
                    int   i = j * d;                                // integer part of j steps of d
                    T a = j * d - i;                                // fractional part of j steps of d, P&T's alpha

                    // when using P&T's eq. 9.68, compute knots using the following
                    //             tmesh.all_knots[k][p(k) + j] = (1.0 - a) * params[k][i - 1]+ a * params[k][i];

                    // when using my version of d, use the following
                    tmesh.all_knots[k][p(k) + j] = (1.0 - a) * params[k][i] + a * params[k][i + 1];

                    // parameter span containing the knot
                    while (params[k][param_idx] < tmesh.all_knots[k][p(k) + j])
                        param_idx++;
                    tmesh.all_knot_param_idxs[k][p(k) + j] = param_idx;
                }

                // set external knots
                for (int i = 0; i < p(k) + 1; i++)
                {
                    tmesh.all_knots[k][i] = 0.0;
                    tmesh.all_knots[k][nknots - 1 - i] = 1.0;
                    tmesh.all_knot_param_idxs[k][nknots - 1 - i] = params[k].size() - 1;
                }
            }
        }

        // Set knot vector from user-supplied input
        void customKnots(const vector<vector<T>>& knots)
        {
            // Check that knot vectors are pinned
            for (size_t k = 0; k < dom_dim; k++)
            {
                int last = knots[k].size() - 1;
                bool pinned = true;

                // Check that knots are pinned
                for (int i = 0; i < p(k) + 1; i++)
                {
                    if (knots[k][i] != 0 || knots[k][last - i] != 1)
                    {
                        cerr << "ERROR: Custom knot distribution does not have pinned knots in dimension " << k << endl;
                        cerr << "Exiting." << endl;
                        exit(1);
                    }
                }
            }

            // Resize (and clear) global knot data structures
            VectorXi new_nctrl(dom_dim);
            for (int i = 0; i < dom_dim; i++)
            {
                new_nctrl(i) = knots[i].size() - p(i) - 1;
            }
            tmesh.reinit_knots(new_nctrl);
            
            // Copy knots to tmesh
            tmesh.all_knots = knots;
        }

        // compute knots
        // n-d version of uniform spacing
        // tmesh version
        //
        // nknots = n + p + 2
        // eg, for p = 3 and nctrl_pts = 7, n = nctrl_pts - 1 = 6, nknots = 11
        // let knots = {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}
        // there are p + 1 external knots at each end: {0, 0, 0, 0} and {1, 1, 1, 1}
        // there are n - p internal knots: {0.25, 0.5, 0.75}
        // there are n - p + 1 internal knot spans [0,0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1)
        //
        // resulting knots are same for all curves and stored once for each dimension (1st dim knots, 2nd dim, ...)
        // total number of knots is the sum of number of knots over the dimensions, much less than the product
        // assumes knots were allocated by caller
        void uniformKnots()
        {
            for (size_t k = 0; k < dom_dim; k++)
            {
                // TODO: hard-coded for first tensor product of the tmesh
                int nctrl_pts = tmesh.tensor_prods[0].nctrl_pts(k);
                int nknots    = nctrl_pts + p(k) + 1;              // number of knots in current dim

                // set p + 1 external knots at each end
                for (int i = 0; i < p(k) + 1; i++)
                {
                    tmesh.all_knots[k][i] = 0.0;
                    tmesh.all_knots[k][nknots - 1 - i] = 1.0;
                }

                // compute remaining n - p internal knots
                T step = 1.0 / (nctrl_pts - p(k));              // size of internal knot span
                for (int j = 1; j <= nctrl_pts - p(k) - 1; j++)
                {
                    tmesh.all_knots[k][p(k) + j] = tmesh.all_knots[k][p(k) + j - 1] + step;
                }
            }
        }
    };
}
#endif

