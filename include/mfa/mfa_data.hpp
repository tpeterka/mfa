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

        // constructor for creating an mfa from input points
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

        void set_knots(PointSet<T>& input)
        {
            // TODO move this elsewhere (to encode method?), wrapped in "structured==true" block
            // allocate basis functions
            if (input.structured)
            {
                N.resize(dom_dim);
                for (auto i = 0; i < dom_dim; i++)
                    N[i] = MatrixX<T>::Zero(input.ndom_pts(i), tmesh.all_knots[i].size() - p(i) - 1);
            }

            // initialize first tensor product
            vector<size_t> knot_mins(dom_dim);
            vector<size_t> knot_maxs(dom_dim);
            for (auto i = 0; i < dom_dim; i++)
            {
                knot_mins[i] = 0;
                knot_maxs[i] = tmesh.all_knots[i].size() - 1;
            }
            tmesh.append_tensor(knot_mins, knot_maxs);

#ifdef CURVE_PARAMS
            if (!input.structured)
            {
                cerr << "ERROR: Cannot set curve knots from unstructured input" << endl;
                exit(1);
            }
            Knots(input, tmesh);       // knots spaced according to parameters (per P&T)
#else
            UniformKnots(input, tmesh);                    // knots spaced uniformly
#endif
        }

        //         DEPRECATE: dangerous, can find a span not in the tensor, looks at all knots irrespective of level
//         // binary search to find the span in the knots vector containing a given parameter value
//         // returns span index i s.t. u is in [ knots[i], knots[i + 1] )
//         // NB closed interval at left and open interval at right
//         //
//         // i will be in the range [p, n], where n = number of control points - 1 because there are
//         // p + 1 repeated knots at start and end of knot vector
//         // algorithm 2.1, P&T, p. 68
//         int FindSpan(
//                 int                     p,                  // degree in current dimension
//                 T                       u,                  // parameter value in current dim
//                 const vector<T>&        knots) const        // knots in current dim
//         {
//             int nctrl_pts = knots.size() - p - 1;
//             if (u == knots[nctrl_pts])
//                 return nctrl_pts - 1;
// 
//             // binary search
//             int low = p;
//             int high = nctrl_pts;
//             int mid = (low + high) / 2;
//             while (u < knots[mid] || u >= knots[mid + 1])
//             {
//                 if (u < knots[mid])
//                     high = mid;
//                 else
//                     low = mid;
//                 mid = (low + high) / 2;
//             }
// 
//             return mid;
//         }

        // binary search to find the span in the knots vector containing a given parameter value
        // returns span index i s.t. u is in [ knots[i], knots[i + 1] )
        // NB closed interval at left and open interval at right
        //
        // i will be in the range [p, n], where n = number of control points - 1 because there are
        // p + 1 repeated knots at start and end of knot vector
        // algorithm 2.1, P&T, p. 68
        //
        // CAUTION: can find a span not in the tensor, looks at all knots irrespective of level
        // use only for a t-mesh with one tensor product, where all knots are used in the tensor
        int FindSpan(
                int                     cur_dim,            // current dimension
                T                       u,                  // parameter value
                int                     nctrl_pts) const    // number of control points in current dim
        {
            if (u == tmesh.all_knots[cur_dim][nctrl_pts])
                return nctrl_pts - 1;

            // binary search
            int low = p(cur_dim);
            int high = nctrl_pts;
            int mid = (low + high) / 2;
            while (u < tmesh.all_knots[cur_dim][mid] || u >= tmesh.all_knots[cur_dim][mid + 1])
            {
                if (u < tmesh.all_knots[cur_dim][mid])
                    high = mid;
                else
                    low = mid;
                mid = (low + high) / 2;
            }

            return mid;
        }

        //         DEPRECATE: dangerous, can find a span not in the tensor, looks at all knots irrespective of level
//         // binary search to find the span in the knots vector containing a given parameter value
//         // returns span index i s.t. u is in [ knots[i], knots[i + 1] )
//         // NB closed interval at left and open interval at right
//         //
//         // i will be in the range [p, n], where n = number of control points - 1 because there are
//         // p + 1 repeated knots at start and end of knot vector
//         // algorithm 2.1, P&T, p. 68
//         //
//         // number of control points computed from number of knots
//         int FindSpan(
//                 int                     cur_dim,            // current dimension
//                 T                       u) const            // parameter value
//         {
//             int nctrl_pts = tmesh.all_knots[cur_dim].size() - p(cur_dim) - 1;
// 
//             if (u == tmesh.all_knots[cur_dim][nctrl_pts])
//                 return nctrl_pts - 1;
// 
//             // binary search
//             int low = p(cur_dim);
//             int high = nctrl_pts;
//             int mid = (low + high) / 2;
//             while (u < tmesh.all_knots[cur_dim][mid] || u >= tmesh.all_knots[cur_dim][mid + 1])
//             {
//                 if (u < tmesh.all_knots[cur_dim][mid])
//                     high = mid;
//                 else
//                     low = mid;
//                 mid = (low + high) / 2;
//             }
// 
//             // debug
// //             cerr << "u = " << u << " span = " << mid << endl;
// 
//             return mid;
//         }

        // binary search to find the span in the global all_knots vector containing a given parameter value
        // returns span index i s.t. u is in [ knots[i], knots[i + 1] )
        // NB closed interval at left and open interval at right
        // tmesh version for searching only one tensor
        //
        // i will be in the range [p, n], where n = number of control points - 1 because there are
        // p + 1 repeated knots at start and end of knot vector
        // algorithm 2.1, P&T, p. 68
        //
        // prints an error and aborts if u is not in the min,max range of knots in tensor or if levels of u and the span do not match
        int FindSpan(
                int                     cur_dim,            // current dimension
                T                       u,                  // parameter value
                const TensorProduct<T>& tensor) const       // tensor product in tmesh
        {
            if (u < tmesh.all_knots[cur_dim][tensor.knot_mins[cur_dim]] ||
                    u > tmesh.all_knots[cur_dim][tensor.knot_maxs[cur_dim]])
            {
                fmt::print(stderr, "FindSpan(): Asking for parameter value outside of the knot min/max of the current tensor. This should not happen.\n");
                fmt::print(stderr, "u {} cur_dim {} knot_mins [{}] knot_maxs [{}]\n",
                        u, cur_dim, fmt::join(tensor.knot_mins, ","), fmt::join(tensor.knot_maxs, ","));
                tmesh.print_tensor(tensor, true);
                abort();
            }

            int low, high, mid;
            int found = -1;

            if (tensor.knot_mins[cur_dim] == 0)
                low = p(cur_dim);
            else
                low = 0;
            if (tensor.knot_maxs[cur_dim] == tmesh.all_knots[cur_dim].size() - 1)
                high = tensor.knot_idxs[cur_dim].size() - p(cur_dim) - 1;
            else
                high = tensor.knot_idxs[cur_dim].size() - 1;
            mid = (low + high) / 2;

            if (u >= tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][high]])
                found = high - 1;

            if (found < 0)
            {
                // binary search
                while (u < tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][mid]] ||
                        u >= tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][mid + 1]])
                {
                    if (u < tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][mid]])
                        high = mid;
                    else
                        low = mid;
                    mid = (low + high) / 2;
                }
                found = mid;
            }

            // sanity checks
            if (tmesh.all_knot_levels[cur_dim][tensor.knot_idxs[cur_dim][found]] > tensor.level)
            {
                fmt::print(stderr, "FindSpan(): level mismatch at found span. This should not happen.\n");
                fmt::print(stderr, "u {} dim {} knot idx {} knot value {} knot level {} tensor level {}\n",
                        u, cur_dim, tensor.knot_idxs[cur_dim][found], tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found]],
                        tmesh.all_knot_levels[cur_dim][tensor.knot_idxs[cur_dim][found]], tensor.level);
//                 tmesh.print_tensor(tensor, true);
                abort();
            }
            bool error = false;
            if (u < tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found]])
                error = true;
            if (tensor.knot_maxs[cur_dim] == tmesh.all_knots[cur_dim].size() - 1)               // tensor is at global max end
            {
                if (u > tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                    error = true;
            }
            else                                                                                // tensor is not at global max end
            {
                if (p(cur_dim) % 2 == 0)                                                        // even degree
                {
                    if (u >= tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                        error = true;
                }
                else                                                                            // odd degree
                {
                    if (tensor.knot_maxs[cur_dim] > found + 1 &&                                // right edge of found span is inside the max of the tensor
                            u >= tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                        error = true;
                    if (tensor.knot_maxs[cur_dim] == found + 1 &&                               // right edge of found span is at max of the tensor
                            u > tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                        error = true;
                }
            }
            if (error)
            {
                fmt::print(stderr, "FindSpan(): parameter {} not in span [{}, {}) = knots [{}, {}). This should not happen.\n",
                        u, found, found + 1, tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found]],
                        tmesh.all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]]);
                abort();
            }

            return tensor.knot_idxs[cur_dim][found];
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

//                 // debug
//                 fprintf(stderr, "span = %d ith basis fun = %d row = %d loc_knots: ", span, span - p(cur_dim) + j, row);
//                 for (auto i = 0; i < loc_knots.size(); i++)
//                     fprintf(stderr, "%.3lf ", loc_knots[i]);
//                 fprintf(stderr, "\n");

                // TODO: this is a hack for not having p+1 control points, not sure if this is right
//                 if (span - p(cur_dim) + j >= 0 && span - p(cur_dim) + j < N.cols())
                    N(row, span - p(cur_dim) + j) = OneBasisFun(cur_dim, u, loc_knots);
            }

            // debug
//             cerr << N << "\n---" << endl;
        }

        // computes and returns one (the ith) basis function value for a given parameter value
        // algorithm 2.4 of P&T, p. 74
        //
        T OneBasisFun(
                int                     cur_dim,        // current dimension
                T                       u,              // parameter value
                int                     i) const        // compute the ith basis function, 0 <= i <= p(cur_dim)
        {
            vector<T> N(p(cur_dim) + 1);                // triangular table result
            vector<T>& U = tmesh.all_knots[cur_dim];    // alias for knot vector for current dimension

            // 1 at edges of global knot vector
            if ( (i == 0 && u == U[0]) || ( i == U.size() - p(cur_dim) - 2 && u == U.back()) )
                return 1.0;

            // zero outside of local knot vector
            if (u < U[i] || u >= U[i + p(cur_dim) + 1])
                return 0.0;

            // initialize 0-th degree functions
            for (auto j = 0; j <= p(cur_dim); j++)
            {
                if (u >= U[i + j] && u < U[i + j + 1])
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
                    saved = ((u - U[i]) * N[0]) / (U[i + k] - U[i]);
                for (auto j = 0; j < p(cur_dim) - k + 1; j++)
                {
                    uleft     = U[i + j + 1];
                    uright    = U[i + j + k + 1];
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

        // computes one row of basis function values for a given parameter value
        // writes results in a row of N
        // algorithm 2.2 of P&T, p. 70
        // tmesh version
        //
        // assumes N has been allocated by caller
        void BasisFuns(
                const TensorProduct<T>& tensor,     // current tensor product
                int                     cur_dim,    // current dimension
                T                       u,          // parameter value
                int                     span,       // index of span in the knots vector containing u, relative to ko
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
            int j_left = 1;             // j_left and j_right are like j in the loop below but skip over knots not in the right level
            int j_right = 1;
            for (int j = 1; j <= p(cur_dim); j++)
            {
                // skip knots not in current level
                while (tmesh.all_knot_levels[cur_dim][span + 1 - j_left] != tensor.level)
                {
                    j_left++;
                    assert(span + 1 - j_left >= 0);
                }
                // left[j] is u = the jth knot in the correct level to the left of span
                left[j]  = u - tmesh.all_knots[cur_dim][span + 1 - j_left];
                while (tmesh.all_knot_levels[cur_dim][span + j_right] != tensor.level)
                {
                    j_right++;
                    assert(span + j_right < tmesh.all_knot_levels[cur_dim].size());
                }
                // right[j] = the jth knot in the correct level to the right of span - u
                right[j] = tmesh.all_knots[cur_dim][span + j_right] - u;
                j_left++;
                j_right++;

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
        }

        // computes one row of basis function values for a given parameter value
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
                              TensorIdx                tensor_idx,             // tensor product for insertion
                              VectorXi&                new_nctrl_pts,          // (output) new number of control points in each dim.
                              vector<vector<T>>&       new_all_knots,          // (output) new global all knots
                              vector<vector<int>>&     new_all_knot_levels,    // (output) new global all knot levels
                              MatrixX<T>&              new_ctrl_pts,           // (output) new local control points for this tensor
                              VectorX<T>&              new_weights,            // (output) new local weights for this tensor
                              vector<int>&             inserted_dims) const    // which dims actually added a knot and ctrl pt
        {
            // debug
//             fmt::print(stderr, "NewKnotInsertion(): ctrl_pts before inserting param [{}]:\n{}\n", param.transpose(), tensor.ctrl_pts);

            auto& tensor = tmesh.tensor_prods[tensor_idx];
            new_nctrl_pts = tensor.nctrl_pts;
            NewVolKnotIns(param, tensor_idx, new_all_knots, new_all_knot_levels, new_ctrl_pts, new_weights, new_nctrl_pts, inserted_dims);

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
                const VectorX<T>&       param,              // new knot value to be inserted
                TensorIdx               tensor_idx,         // original tensor with existing volume of control points from which curve is derived
                int                     cur_dim,            // current dimension
                const vector<T>&        old_knots,          // old knot vector in cur. dim.
                const vector<int>&      old_knot_levels,    // old knot levels in cur. dim.
                const MatrixX<T>&       old_ctrl_pts,       // old control points of curve
                const VectorX<T>&       old_weights,        // old control point weights of curve
                int                     level,              // level of new knot to be inserted
                vector<T>&              new_knots,          // (output) new knot vector in cur. dim.
                vector<int>&            new_knot_levels,    // (output) new knot levels in cur. dim.
                MatrixX<T>&             new_ctrl_pts,       // (output) new control points of curve
                VectorX<T>&             new_weights) const  // (output) new control point weights of curve
        {
            auto&   tensor  = tmesh.tensor_prods[tensor_idx];
            T       u       = param(cur_dim);               // parameter in current dim.

            new_knots.resize(old_knots.size() + 1);
            new_knot_levels.resize(old_knot_levels.size() + 1);
            new_ctrl_pts.resize(old_ctrl_pts.rows() + 1, old_ctrl_pts.cols());
            new_weights.resize(old_weights.size() + 1);
            MatrixX<T> temp_ctrl_pts(p(cur_dim) + 1, old_ctrl_pts.cols());
            VectorX<T> temp_weights(p(cur_dim) + 1);

            // anchor corresponding to param in all dims
            vector<KnotIdx> anchor(dom_dim);
            for (auto i = 0; i < dom_dim; i++)
                anchor[i] = FindSpan(i, param(i), tensor);

            int global_span    = anchor[cur_dim];                       // global knot span of param in current dim.
            T eps       = 1.0e-8;
            if (fabs(old_knots[global_span] - u) < eps)                 // not for multiple knots
            {
                fmt::print(stderr, "Error: NewCurveKnotIns attempting to insert duplicate knot in dim {} global_span {} knot {} u {}\n",
                        cur_dim, global_span, tmesh.all_knots[cur_dim][global_span], u);
                exit(0);
            }

            // load new knot vector
            for (auto i = 0; i <= global_span; i++)
            {
                new_knots[i]        = old_knots[i];
                new_knot_levels[i]  = old_knot_levels[i];
            }
            new_knots[global_span + 1]         = u;
            new_knot_levels[global_span + 1]   = level;
            for (auto i = global_span + 1; i < old_ctrl_pts.rows() + p(cur_dim) + 1; i++)
            {
                new_knots[i + 1]        = old_knots[i];
                new_knot_levels[i + 1]  = old_knot_levels[i];
            }

            // convert span to local knot index and control point index in tensor
            // use global_span for indexing knots in global knot vector
            // use local_span for indexing knots in local tensor
            // use local_span + shift for indexing control points in local tensor
            int local_span = tmesh.global2local_knot_idx(global_span, tensor, cur_dim);
            int shift = tensor.knot_mins[cur_dim] == 0 ? 0 : (p(cur_dim) + 1) / 2;      // shift ctrl pt indices for interior tensors w/o clamped end

            // if too close to edge of interior tensor, only compute the newly inserted control point
            // w/o recomputing the others in the changed range of p(cur_dim)
            // this is ok because we only keep the newly inserted control point eventually for our separable local constraints
            // to do a proper knot insertion, would need to go to side neighbor and get more control points from there
            // but the control points there likely won't align, setting up a recursion of knot insertions
            // we don't do this
            if (tmesh.knot_idx_dist(tensor, tensor.knot_mins[cur_dim], global_span, cur_dim, false) < p(cur_dim) - 1 ||
                    tmesh.knot_idx_dist(tensor, global_span, tensor.knot_maxs[cur_dim], cur_dim, false) < p(cur_dim))
            {

                // copy control points before local span
                for (auto i = 0; i <= local_span; i++)
                {
                    new_ctrl_pts.row(i) = old_ctrl_pts.row(i);
                    new_weights(i)      = old_weights(i);
                }

                // copy control points after local span
                for (auto i = local_span + 2; i < tensor.nctrl_pts(cur_dim); i++)
                {
                    new_ctrl_pts.row(i + 1) = old_ctrl_pts.row(i);
                    new_weights(i + 1)      = old_weights(i);
                }

                // set up only the two temp_ctrl points that will be needed
                for (auto i = p(cur_dim) / 2; i <= p(cur_dim) / 2 + 1; i++)
                {
                    // debug
                    // TODO: remove once stable
                    if (i >= temp_ctrl_pts.rows() || local_span - p(cur_dim) + i + shift < 0 || local_span - p(cur_dim) + 1 + shift >= old_ctrl_pts.rows())
                    {
                        fmt::print(stderr, "Error: NewCurveKnotInsertion(): index out of range\n");
                        abort();
                    }

                    temp_ctrl_pts.row(i)    = old_ctrl_pts.row(local_span - p(cur_dim) + i + shift);
                    temp_weights(i)         = old_weights(local_span - p(cur_dim) + i + shift);
                }

                // get knots for interpolation
                // TODO: write a 1-d knot intersection routine; this uses full-d, which is overkill
                vector<vector<KnotIdx>> loc_knots(dom_dim);
                tmesh.knot_intersections(anchor, tensor_idx, loc_knots);

                // pick out the two knots to interpolate from the local knot vector
                KnotIdx left_idx    = loc_knots[cur_dim][1];
                KnotIdx right_idx   = loc_knots[cur_dim].back();

                // interpolate only the one newly inserted control point
                int i                               = p(cur_dim) / 2;
                T alpha                             = (u - old_knots[left_idx]) / (old_knots[right_idx] - old_knots[left_idx]);
                new_ctrl_pts.row(local_span + 1)    = alpha * temp_ctrl_pts.row(i + 1) + (1.0 - alpha) * temp_ctrl_pts.row(i);
                new_weights(local_span + 1)         = alpha * temp_weights(i + 1) + (1.0 - alpha) * temp_weights(i);

//                 fmt::print(stderr, "NewCurveKnotInsertion() 2: inserting new control point at idx {} value [{}]\n", local_span + 1, new_ctrl_pts.row(local_span + 1));

                return;
            }

            // save unaltered control points and weights
            for (auto i = 0; i <= local_span - p(cur_dim) + shift; i++)
            {
                new_ctrl_pts.row(i) = old_ctrl_pts.row(i);
                new_weights(i)      = old_weights(i);
            }
            for (auto i = local_span + shift; i < old_ctrl_pts.rows(); i++)
            {
                new_ctrl_pts.row(i + 1) = old_ctrl_pts.row(i);
                new_weights(i + 1)      = old_weights(i);
            }
            for (auto i = 0; i <= p(cur_dim); i++)
            {
                // debug
                // TODO: remove once stable
                if (i >= temp_ctrl_pts.rows() || local_span - p(cur_dim) + i + shift < 0 || local_span - p(cur_dim) + 1 + shift >= old_ctrl_pts.rows())
                {
                    fmt::print(stderr, "Error: NewCurveKnotInsertion(): index out of range\n");
                    abort();
                }

                temp_ctrl_pts.row(i)    = old_ctrl_pts.row(local_span - p(cur_dim) + i + shift);
                temp_weights(i)         = old_weights(local_span - p(cur_dim) + i + shift);
            }

            // compute p(cur_dim) new control points, one of which is newly inserted and rest are modified
            // the newly inserted one is temp_ctrl_pts(p_cur(dim) / 2) (using integer division, ie, floor)

            for (auto i = 0; i < p(cur_dim); i++)
            {
                bool ofst_success;

                KnotIdx ofst1;               // ofst1 = global_span + 1 + i - p(cur_dim)
                ofst_success = tmesh.knot_idx_ofst(tensor, global_span, i + 1 - p(cur_dim), cur_dim, false, ofst1);
                if (!ofst_success)
                {
                    fmt::print(stderr, "Error: NewCurveKnotInsertion(): unable to offset global_span by 1 + i - p(cur_dim)\n");
                    abort();
                }

                KnotIdx ofst2;               // ofst2 = global_span + 1 + i
                ofst_success = tmesh.knot_idx_ofst(tensor, global_span, i + 1, cur_dim, false, ofst2);
                if (!ofst_success)
                {
                    fmt::print(stderr, "Error: NewCurveKnotInsertion(): unable to offset global span by 1 + i\n");
                    abort();
                }

                T alpha                 = (u - old_knots[ofst1]) / (old_knots[ofst2] - old_knots[ofst1]);
                temp_ctrl_pts.row(i)    = alpha * temp_ctrl_pts.row(i + 1) + (1.0 - alpha) * temp_ctrl_pts.row(i);
                temp_weights(i)         = alpha * temp_weights(i + 1) + (1.0 - alpha) * temp_weights(i);
            }

            // load modified p(cur_dim) control points
            auto L = local_span - p(cur_dim) + 1;
            for (auto i = L; i <= local_span; i++)
            {
                new_ctrl_pts.row(i + shift) = temp_ctrl_pts.row(i - L);
                new_weights(i + shift)      = temp_weights(i - L);

                // debug
//                 if (shift)
//                 {
//                     if (i - L == p(cur_dim) / 2)
//                         fmt::print(stderr, "NewCurveKnotInsertion() 3: inserting new control point at idx {} value [{}]\n", i + shift, new_ctrl_pts.row(i + shift));
//                 }
            }

            // debug
//             fmt::print(stderr, "NewCurveKnotInsertion(): new curve ctrl pts:\n{}\n", new_ctrl_pts);
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
                VectorXi&                   nctrl_pts,              // (input and output) number of control points in all dims
                vector<int>&                inserted_dims) const    // which dims actually added a knot and ctrl pt
        {
            auto&                         tensor          = tmesh.tensor_prods[tensor_idx];
            const vector<vector<T>>&      old_knots       = tmesh.all_knots;
            const vector<vector<int>>&    old_knot_levels = tmesh.all_knot_levels;
            const MatrixX<T>&             old_ctrl_pts    = tensor.ctrl_pts;
            const VectorX<T>&             old_weights     = tensor.weights;
            int                           level           = tensor.level;

            size_t old_cs, new_cs;                                              // stride for old and new control points in curve in cur. dim

            // determine new sizes of control points, weights, knots, knot levels in each dim
            // a knot may be new in one dimension (inserted) and same in another dimension (not inserted)
            VectorXi new_nctrl_pts = nctrl_pts;
            for (auto k = 0; k < dom_dim; k++)
            {
                // check if the knot exists in this dimension already
                int span    = FindSpan(k, param(k), tensor);
                T eps       = 1.0e-8;
                if (fabs(old_knots[k][span] - param(k)) > eps)            // knot is new in this dim.
                    new_nctrl_pts(k)++;
            }

            new_ctrl_pts.resize(new_nctrl_pts.prod(), old_ctrl_pts.cols());
            new_weights.resize(new_ctrl_pts.rows());
            new_knots.resize(dom_dim);
            new_knot_levels.resize(dom_dim);
            inserted_dims.resize(dom_dim);

            // double buffer for new control points and weights (new_ctrl_pts, new_ctrl_pts1; new_weights, new_weights1)
            // so that in alternating dimensions, the output of previous dimension can be input of next dimension
            MatrixX<T> new_ctrl_pts1(new_ctrl_pts.rows(), new_ctrl_pts.cols());
            VectorX<T> new_weights1(new_weights.size());
            new_ctrl_pts1.block(0, 0, old_ctrl_pts.rows(), old_ctrl_pts.cols()) = old_ctrl_pts;
            new_weights1.segment(0, old_weights.rows())                         = old_weights;

            for (auto k = 0; k < dom_dim; k++)                                // for all domain dimensions
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

                    inserted_dims[k] = 0;
                    continue;
                }

                inserted_dims[k] = 1;                                           // inserting a knot in this dimension

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

                        // copy one curve of old curve control points and weights
                        if (k % 2 == 0)
                            CtrlPts2CtrlCurve(new_ctrl_pts1, new_weights1, old_curve_ctrl_pts.local(),
                                    old_curve_weights.local(), nctrl_pts, k, old_co[j], old_cs);
                        else
                            CtrlPts2CtrlCurve(new_ctrl_pts, new_weights, old_curve_ctrl_pts.local(),
                                    old_curve_weights.local(), nctrl_pts, k, old_co[j], old_cs);


                        // insert a knot in one curve of control points
                        NewCurveKnotIns(param, tensor_idx, k, old_knots[k], old_knot_levels[k], old_curve_ctrl_pts.local(), old_curve_weights.local(),
                                level, new_knots[k], new_knot_levels[k], new_curve_ctrl_pts.local(), new_curve_weights.local());

                        // copy new curve control points and weights
                        if (k % 2 == 0)
                            CtrlCurve2CtrlPts(new_curve_ctrl_pts.local(), new_curve_weights.local(),
                                    new_ctrl_pts, new_weights, new_nctrl_pts, k, new_co[j], new_cs);
                        else
                            CtrlCurve2CtrlPts(new_curve_ctrl_pts.local(), new_curve_weights.local(),
                                    new_ctrl_pts1, new_weights1, new_nctrl_pts, k, new_co[j], new_cs);
                        });

#endif              // end TBB version

#ifdef MFA_SERIAL   // serial version

                MatrixX<T> old_curve_ctrl_pts, new_curve_ctrl_pts;              // old and new control points for one curve
                VectorX<T> old_curve_weights, new_curve_weights;                // old and new weights for one curve

                for (size_t j = 0; j < old_ncurves; j++)                        // for all curves in this dimension
                {
                    // copy one curve of old curve control points and weights
                    if (k % 2 == 0)
                        CtrlPts2CtrlCurve(new_ctrl_pts1, new_weights1, old_curve_ctrl_pts,
                                old_curve_weights, nctrl_pts, k, old_co[j], old_cs);
                    else
                        CtrlPts2CtrlCurve(new_ctrl_pts, new_weights, old_curve_ctrl_pts,
                                old_curve_weights, nctrl_pts, k, old_co[j], old_cs);

                    // insert a knot in one curve of control points
                    NewCurveKnotIns(param, tensor_idx, k, old_knots[k], old_knot_levels[k], old_curve_ctrl_pts, old_curve_weights,
                            level, new_knots[k], new_knot_levels[k], new_curve_ctrl_pts, new_curve_weights);

                    // copy new curve control points and weights
                    if (k % 2 == 0)
                        CtrlCurve2CtrlPts(new_curve_ctrl_pts, new_curve_weights,
                                new_ctrl_pts, new_weights, new_nctrl_pts, k, new_co[j], new_cs);
                    else
                        CtrlCurve2CtrlPts(new_curve_ctrl_pts, new_curve_weights,
                                new_ctrl_pts1, new_weights1, new_nctrl_pts, k, new_co[j], new_cs);
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
                Tmesh<T>&                   tmesh) const            // (output) tmesh
        {
            if (!input.structured)
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
        void UniformKnots( const    PointSet<T>&   input,
                                    Tmesh<T>&       tmesh)
        {
            if (input.structured)
            {
                // debug
                cerr << "Using uniform knots (structured input)" << endl;

                uniform_knots_impl_structured(input.params->param_grid, tmesh);
            }
            else
            {
                // debug
                cerr << "Using uniform knots (unstructured input)" << endl;

                uniform_knots_impl_unstructured(tmesh);
            }
        }

        void uniform_knots_impl_structured(
                const vector<vector<T>>&    params,             // parameters for input points[dimension][index]
                Tmesh<T>&                   tmesh) const        // (output) tmesh
        {
            for (size_t k = 0; k < dom_dim; k++)                // for all domain dimensions
            {
                // TODO: hard-coded for first tensor product of the tmesh
                int nctrl_pts = tmesh.tensor_prods[0].nctrl_pts(k);

                int nknots = nctrl_pts + p(k) + 1;              // number of knots in current dim

                // set p + 1 external knots at each end
                for (int i = 0; i < p(k) + 1; i++)
                {
                    tmesh.all_knots[k][i] = 0.0;
                    tmesh.all_knots[k][nknots - 1 - i] = 1.0;
                    tmesh.all_knot_param_idxs[k][nknots - 1 - i] = params[k].size() - 1;
                }

                // compute remaining n - p internal knots
                T step = 1.0 / (nctrl_pts - p(k));              // size of internal knot span
                size_t param_idx = 0;                           // index into params
                for (int j = 1; j <= nctrl_pts - p(k) - 1; j++)
                {
                    tmesh.all_knots[k][p(k) + j] = tmesh.all_knots[k][p(k) + j - 1] + step;

                    // parameter span containing the knot
                    while (params[k][param_idx] < tmesh.all_knots[k][p(k) + j])
                        param_idx++;
                    tmesh.all_knot_param_idxs[k][p(k) + j] = param_idx;
                }
            }
        }

        void uniform_knots_impl_unstructured(Tmesh<T>& tmesh)
        {
            cerr << "Warning: Unstable build, tmesh.all_knot_param_idxs remain uninitialized" << endl; 
            cerr << "  => tmesh.insert_knot() and tmesh.domain_pts() will not be valid" << endl;
            cerr << "  => use of NewKnots class will not be valid" << endl;

            for (size_t k = 0; k < dom_dim; k++)                // for all domain dimensions
            {
                // TODO: hard-coded for first tensor product of the tmesh
                int nctrl_pts = tmesh.tensor_prods[0].nctrl_pts(k);

                int nknots = nctrl_pts + p(k) + 1;              // number of knots in current dim

                // set p + 1 external knots at each end
                for (int i = 0; i < p(k) + 1; i++)
                {
                    tmesh.all_knots[k][i] = 0.0;
                    tmesh.all_knots[k][nknots - 1 - i] = 1.0;
                }

                // compute remaining n - p internal knots
                T step = 1.0 / (nctrl_pts - p(k));              // size of internal knot span
                size_t param_idx = 0;                           // index into params
                for (int j = 1; j <= nctrl_pts - p(k) - 1; j++)
                {
                    tmesh.all_knots[k][p(k) + j] = tmesh.all_knots[k][p(k) + j - 1] + step;
                }
            }
        }

    };
}
#endif

