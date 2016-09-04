//--------------------------------------------------------------
// nurbs encoding algorithms
// ref: [P&T] Piegl & Tiller, The NURBS Book, 1995
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/encode.hpp>

#include <Eigen/Dense>

#include <vector>
#include <iostream>

using namespace std;

// --- data model ---
//
// using Eigen dense MartrixX to represent vectors of n-dimensional points
// rows: points; columns: point coordinates
//
// using Eigen VectorX to represent a single point
// to use a single point from a set of many points,
// explicitly copying from a row of a matrix to a vector before using the vector for math
// (and Eigen matrix row and vector are not interchangeable w/o doing an assignment,
// at least not with the current default column-major matrix ordering, not contiguous)
//
// also using Eigen VectorX to represent a set of scalars such as knots or parameters
//
// TODO: think about row/column ordering of Eigen, choose the most contiguous one
// (current default column ordering of points is not friendly to extracting a single point)
//
// TODO: think about Eigen sparse matrices
// (N and NtN matrices, used for solving for control points, are very sparse)
//
// ------------------

// binary search to find the span in the knots vector containing a given parameter value
// returns span index i s.t. u is in [ knots[i], knots[i + 1] )
// NB closed interval at left and open interval at right
// except when u == knots.last(), in which case the interval is closed at both ends
// i will be in the range [p, n], where n = number of control points - 1 because there are
// p + 1 repeated knots at start and end of knot vector
// algorithm 2.1, P&T, p. 68
int FindSpan(int       p,                    // polynomial degree
             int       n,                    // number of control point spans
             VectorXf& knots,                // knots
             float     u,                    // parameter value
             int       ko)                   // optional starting knot to search (default = 0)
{
    if (u == knots(ko + n + 1))
        return ko + n;

    // binary search
    int low = p;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < knots(ko + mid) || u >= knots(ko + mid + 1))
    {
        if (u < knots(ko + mid))
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }

    return ko + mid;
}

// computes p + 1 nonvanishing basis function values [N_{span - p}, N_{span}]
// of the given parameter value
// keeps only those in the range [N_{start_n}, N_{end_n}]
// writes results in a subset of a row of N starting at index N(start_row, start_col)
// algorithm 2.2 of P&T, p. 70
// assumes N has been allocated by caller
void BasisFuns(int       p,                  // polynomial degree
               VectorXf& knots,              // knots (1st dim changes fastest)
               float     u,                  // parameter value
               int       span,               // index of span in the knots vector containing u
               MatrixXf& N,                  // matrix of (output) basis function values
               int       start_n,            // starting basis function N_{start_n} to compute
               int       end_n,              // ending basis function N_{end_n} to compute
               int       row,                // starting row index in N of result
               int       ko)                 // optional starting knot to search (default = 0)
{
    // init
    vector<float> scratch(p + 1);            // scratchpad, same as N in P&T p. 70
    scratch[0] = 1.0;
    vector<float> left(p + 1);               // temporary recurrence results
    vector<float> right(p + 1);

    // fill N
    for (int j = 1; j <= p; j++)
    {
        left[j]  = u - knots(span + 1 - j);
        right[j] = knots(span + j) - u;
        float saved = 0.0;
        for (int r = 0; r < j; r++)
        {
            float temp = scratch[r] / (right[r + 1] + left[j - r]);
            scratch[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        scratch[j] = saved;
    }

    // debug
    // cerr << "scratch: ";
    // for (int i = 0; i < p + 1; i++)
    //     cerr << scratch[i] << " ";
    // cerr << endl;

    // copy scratch to N
    for (int j = 0; j < p + 1; j++)
    {
        int n_i = span - ko - p + j;              // index of basis function N_{n_i}
        if (n_i >= start_n && n_i <= end_n)
        {
            int col = n_i - start_n;         // column in N where to write result
            if (col >= 0 && col < N.cols())
                N(row, col) = scratch[j];
            else
                cerr << "Note(1): BasisFuns() truncating N_" << n_i << " = " << scratch[j] <<
                    " at (" << row << ", " << col << ")" << endl;
        }
    }
}

// computes R (residual) vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve
void Residual(int       p,                   // polynomial degree
              MatrixXf& domain,              // domain of input data points
              VectorXf& knots,               // knots
              VectorXf& params,              // parameters of input points
              MatrixXf& N,                   // matrix of basis function coefficients
              MatrixXf& R,                   // (output) residual matrix allocated by caller
              int ko = 0,                    // optional index of starting knot (default = 0)
              int po = 0,                    // optional index of starting parameter (default = 0)
              int co = 0,                    // optional index of starting domain pt
                                             // in current curve (default = 0)
              int cs = 1)                    // optional stride of domain pts in current curve
{
    int n      = N.cols() + 1;               // number of control point spans
    int m      = N.rows() + 1;               // number of input data point spans

    // compute the matrix Rk for eq. 9.63 of P&T, p. 411
    MatrixXf Rk(m - 1, domain.cols());       // eigen frees MatrixX when leaving scope
    MatrixXf Nk;                             // basis coefficients for Rk[i]

    // debug
    // cerr << "Residual domain:\n" << domain << endl;

    for (int k = 1; k < m; k++)
    {
        int span = FindSpan(p, n, knots, params(po + k), ko);
        Nk = MatrixXf::Zero(1, n + 1);      // basis coefficients for Rk[i]
        BasisFuns(p, knots, params(po + k), span, Nk, 0, n, 0, ko);

        // debug
        // cerr << "Nk:\n" << Nk << endl;

        // debug
        // cerr << "[" << domain.row(co + k * cs) << "] ["
        //      << domain.row(co) << "] ["
        //      << domain.row(co + m * cs) << "]" << endl;

        Rk.row(k - 1) =
            domain.row(co + k * cs) - Nk(0, 0) * domain.row(co) -
            Nk(0, n) * domain.row(co + m * cs);
    }

    // debug
    // cerr << "Rk:\n" << Rk << endl;

    // compute the matrix R
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < Rk.cols(); j++)
        {
            // debug
            // fprintf(stderr, "3: i %d j %d R.rows %d Rk.rows %d\n", i, j, R.rows(), Rk.rows());
            R(i - 1, j) = (N.col(i - 1).array() * Rk.col(j).array()).sum();
        }
    }
}

// DEPRECATED
// TODO: remove eventually once a more general n-d solution is found or is not needed anymore
//
// // preprocess domain
// // interpolate points on a curve to approximately uniform spacing
// // TODO: normalize domain and range to similar scales
// // new_domain and new_range are resized by Prep1d according to how many new points need to be added
// void Prep1d(MatrixXf& domain,                // domain of input data points
//             MatrixXf& new_domain)            // new domain with interpolated data points
// {
//     vector<float> dists(domain.rows() - 1);  // chord lengths of input data point spans
//     float min_dist;                          // min and max distance

//     // chord lengths

//     // eigen frees following vectors when leaving scope
//     VectorXf a, b, d;
//     for (size_t i = 0; i < domain.rows() - 1; i++)
//     {
//         // TODO: normalize domain and range so they have similar scales
//         a = domain.row(i);
//         b = domain.row(i + 1);
//         d = a - b;
//         dists[i] = d.norm();                 // Euclidean distance (l-2 norm)
//         if (i == 0)
//             min_dist = dists[i];
//         if (dists[i] < min_dist)
//             min_dist = dists[i];
//     }

//     // debug
//     // fprintf(stderr, "min_dist %.3f max_dist %.3f mean_dist %.3f\n",
//     //         min_dist, max_dist, mean_dist);

//     // TODO: experiment with different types (degrees) of interpolation; for now using linear
//     // interpolation based on min distance, so that new data are added, but no original data
//     // points are removed

//     // determine size of new_domain
//     int npts = 0;
//     for (size_t i = 0; i < domain.rows() - 1; i++)
//     {
//         npts++;
//         npts += dists[i] / min_dist - 1;     // number of extra points to insert
//     }
//     npts++;                                  // last point
//     new_domain.resize(npts, domain.cols());

//     // copy domain and range to new versions, adding interpolated points as needed
//     int n = 0;                               // current index in new_domain
//     for (size_t i = 0; i < domain.rows() - 1; i++)
//     {
//         new_domain.row(n++) = domain.row(i);
//         int nextra_pts = dists[i] / min_dist - 1;     // number of extra points to insert
//         for (int j = 0; j < nextra_pts; j++)
//         {
//             float fd = (j + 1) * min_dist / dists[i]; // fraction of distance to add
//             a = domain.row(i);
//             b = domain.row(i + 1);
//             new_domain.row(n++) = a + fd * (b - a);
//         }
//     }
//     // copy last point
//     new_domain.row(n++) = domain.row(domain.rows() - 1);
// }

// precompute curve parameters for input data points using the chord-length method
// n-d version of algorithm 9.3, P&T, p. 377
// params are computed along curves and averaged over all curves at same data point index i,j,k,...
// ie, resulting params for a data point i,j,k,... are same for all curves
// and params are only stored once for each dimension in row-major order (1st dim changes fastest)
// total number of params is the sum of ndom_pts over the dimensions, much less than the total
// number of data points (which would be the product)
// assumes params were allocated by caller
// TODO: investigate other schemes (domain only, normalized domain and range, etc.)
void Params(VectorXi& ndom_pts, // number of input data points in each dim
            MatrixXf& domain,   // input data points in each dim (1st dim changes fastest)
            VectorXf& params)   // (output) curve parameters in each dim (1st dim changes fastest)
{
    float tot_dist;                    // total chord length
    VectorXf dists(ndom_pts.maxCoeff() - 1);  // chord lengths of data point spans for any dim
    params = VectorXf::Zero(params.size());
    VectorXf d;                        // current chord length

    // following are counters for slicing domain and params into curves in different dimensions
    size_t po = 0;                     // starting offset for parameters in current dim
    size_t co = 0;                     // starting offset for curves in domain in current dim
    size_t cs = 1;                     // stride for domain points in curves in current dim

    for (size_t k = 0; k < ndom_pts.size(); k++)         // for all domain dimensions
    {
        co = 0;
        size_t coo = 0;                                  // co at start of contiguous sequence
        size_t ncurves = domain.rows() / ndom_pts(k);    // number of curves in this dimension
        size_t nzero_length_curves = 0;                  // num curves with zero length
        for (size_t j = 0; j < ncurves; j++)             // for all the curves in this dimension
        {
            tot_dist = 0.0;

            // debug
            // fprintf(stderr, "1: k %d j %d po %d co %d cs %d\n", k, j, po, co, cs);

            // chord lengths
            for (size_t i = 0; i < ndom_pts(k) - 1; i++) // for all spans in this curve
            {
                // TODO: normalize domain so that dimensions they have similar scales

                // debug
                // fprintf(stderr, "  i %d co + i * cs = %d co + (i + 1) * cs = %d\n",
                //         i, co + i * cs, co + (i + 1) * cs);

                d = domain.row(co + i * cs) - domain.row(co + (i + 1) * cs);
                dists(i) = d.norm();                     // Euclidean distance (l-2 norm)
                // fprintf(stderr, "dists[%lu] = %.3f\n", i, dists[i]);
                tot_dist += dists(i);
            }

            // accumulate (sum) parameters from this curve into the params for this dim.
            if (tot_dist > 0.0)                          // skip zero length curves
            {
                params(po)                   = 0.0;      // first parameter is known
                params(po + ndom_pts(k) - 1) = 1.0;      // last parameter is known
                float prev_param             = 0.0;      // param value at previous iteration below
                for (size_t i = 0; i < ndom_pts(k) - 2; i++)
                {
                    float dfrac = dists(i) / tot_dist;
                    params(po + i + 1) += prev_param + dfrac;
                    // debug
                    // fprintf(stderr, "k %ld j %ld i %ld po %ld "
                    //         "param %.3f = prev_param %.3f + dfrac %.3f\n",
                    //         k, j, i, po, prev_param + dfrac, prev_param, dfrac);
                    prev_param += dfrac;
                }
            }
            else
                nzero_length_curves++;

            if ((j + 1) % cs)
                co++;
            else
            {
                co = coo + cs * ndom_pts(k);
                coo = co;
            }
        }                                                // curves in this dimension

        // average the params for this dimension by dividing by the number of curves that
        // contributed to the sum (skipped zero length curves)
        for (size_t i = 0; i < ndom_pts(k) - 2; i++)
            params(po + i + 1) /= (ncurves - nzero_length_curves);

        po += ndom_pts(k);
        cs *= ndom_pts(k);
    }                                                    // domain dimensions
}

// compute knots
// n-d version of eqs. 9.68, 9.69, P&T
// eg, for p = 3 and nctrl_pts = 7, n = nctrl_pts - 1 = 6 and nknots = n + p + 2 = 11
// let knots = {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}
// there are p + 1 external knots at each end: {0, 0, 0, 0} and {1, 1, 1, 1}
// there are n - p internal knots: {0.25, 0.5, 0.75}
// there are n - p + 1 internal knot spans [0,0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1)
// resulting knots are same for all curves and stored once for each dimension in
// row-major order (1st dim changes fastest)
// total number of knots is the sum of number of knots over the dimensions,
// much less than the product
// assumes knots were allocated by caller
void Knots(VectorXi& p,               // polynomial degree in each domain dim
           VectorXi& n,               // number of control point spans in each domain dim
           VectorXi& m,               // number of data point spans in each domain dim
           VectorXf& params,          // curve parameters in each dim (1st dim changes fastest)
           VectorXf& knots)           // (output) knots in each dim (1st dim changes fastest)
{
    // following are counters for slicing domain and params into curves in different dimensions
    size_t po = 0;                                // starting offset for params in current dim
    size_t ko = 0;                                // starting offset for knots in current dim

    for (size_t k = 0; k < p.size(); k++)         // for all domain dimensions
    {
        int nknots = n(k) + p(k) + 2;             // number of knots in current dim

        // in P&T, d is the ratio of number of input points (r+1) to internal knot spans (n-p+1)
        // float d = (float)(r + 1) / (n - p + 1);         // eq. 9.68, r is P&T's m
        // but I prefer d to be the ratio of input spans r to internal knot spans (n-p+1)
        float d = (float)m(k) / (n(k) - p(k) + 1);

        // compute n - p internal knots
        for (int j = 1; j <= n(k) - p(k); j++)    // eq. 9.69
        {
            int   i = j * d;                      // integer part of j steps of d
            float a = j * d - i;                  // fractional part of j steps of d, P&T's alpha

            // debug
            // cerr << "d " << d << " j " << j << " i " << i << " a " << a << endl;

            // when using P&T's eq. 9.68, compute knots using the following
            // knots(p + j) = (1.0 - a) * params(i - 1) + a * params(i);

            // when using my version of d, use the following
            knots(ko + p(k) + j) = (1.0 - a) * params(po + i) + a * params(po + i + 1);
        }

        // set external knots
        for (int i = 0; i < p(k) + 1; i++)
        {
            knots(ko + i) = 0.0;
            knots(ko + nknots - 1 - i) = 1.0;
        }

        po += m(k) + 1;
        ko += nknots;
    }
}

// Checks quantities needed for approximation
void Quants(VectorXi& p,                // polynomial degree in each dimension
            VectorXi& ndom_pts,         // number of input data points in each dim
            VectorXi& nctrl_pts,        // desired number of control points in each dim
            VectorXi& n,                // (output) number of control point spans in each dim
            VectorXi& m,                // (output) number of input data point spans in each dim
            int&      tot_nparams,      // (output) total number params in all dims
            int&      tot_nknots,       // (output) total number of knots in all dims
            int&      tot_nctrl)        // (output) total number of control points in all dims
{
    if (p.size() != ndom_pts.size())
    {
        fprintf(stderr, "Error: Approx() size of p must equal size of ndom_pts\n");
        exit(1);
    }
    for (size_t i = 0; i < p.size(); i++)
    {
        if (nctrl_pts(i) <= p(i))
        {
            fprintf(stderr, "Error: Approx() number of control points must be at least p + 1\n");
            exit(1);
        }
        if (nctrl_pts(i) > ndom_pts(i))
        {
            fprintf(stderr, "Error: Approx() number of control points in dimension %ld "
                    "cannot be greater than number of input data points in dimension %ld\n", i, i);
            exit(1);
        }
    }

    n.resize(p.size());
    m.resize(p.size());
    tot_nparams = 0;
    tot_nknots  = 0;
    tot_nctrl   = 1;
    for (size_t i = 0; i < p.size(); i++)
    {
        n(i)        =  nctrl_pts(i) - 1;
        m(i)        =  ndom_pts(i)  - 1;
        tot_nparams += ndom_pts(i);
        tot_nknots  += (n(i) + p(i) + 2);
        tot_nctrl   *= nctrl_pts(i);
    }
}

// append points from P to temporary control points
// init first and last control points and copy rest from solution P
// TODO: any way to avoid this copy?
// last dimension gets copied to final control points
// previous dimensions get copied to alternating double buffers
void CopyCtrl(MatrixXf& P,          // solved points for current dimension and curve
              MatrixXf& domain,     // input data points (1st dim changes fastest)
              VectorXi& ndom_pts,   // number of input domain points in each dimension
              VectorXi& n,          // number of control point spans in each dimension
              int       k,          // current dimension
              size_t   co,          // starting offset for reading domain points
              size_t   cs,          // stride for reading domain points
              size_t   to,          // starting offset for writing control points
              MatrixXf& temp_ctrl0, // first temporary control points buffer
              MatrixXf& temp_ctrl1, // second temporary control points buffer
              MatrixXf& ctrl_pts)   // final ctrl pts after last dim done (1st dim changes fastest)
{
    int ndims = ndom_pts.size();             // number of domain dimensions
    int nctrl_pts = n(k) + 1;                // number of control points in current dim

    // if there is only one dim, copy straight to output
    if (ndims == 1)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        ctrl_pts.row(to) = domain.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + ndom_pts(k) - 1);
        ctrl_pts.row(to + n(k) * cs) = domain.row(co + ndom_pts(k) - 1);
    }
    // first dim copied from domain to temp_ctrl0
    else if (k == 0)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        temp_ctrl0.row(to) = domain.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            temp_ctrl0.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + ndom_pts(k) - 1);
        temp_ctrl0.row(to + n(k) * cs) = domain.row(co + ndom_pts(k) - 1);
    }
    // even numbered dims (but not the last one) copied from temp_ctrl1 to temp_ctrl0
    else if (k % 2 == 0 && k < ndims - 1)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        temp_ctrl0.row(to) = temp_ctrl1.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            temp_ctrl0.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        temp_ctrl0.row(to + n(k) * cs) = temp_ctrl1.row(co + (ndom_pts(k) - 1) * cs);
    }
    // odd numbered dims (but not the last one) copied from temp_ctrl0 to temp_ctrl1
    else if (k % 2 == 1 && k < ndims - 1)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        temp_ctrl1.row(to) = temp_ctrl0.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            temp_ctrl1.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        temp_ctrl1.row(to + n(k) * cs) = temp_ctrl0.row(co + (ndom_pts(k) - 1) * cs);
    }
    // final dim if even is copied from temp_ctrl1 to ctrl_pts
    else if (k == ndims - 1 && k % 2 == 0)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        ctrl_pts.row(to) = temp_ctrl1.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        ctrl_pts.row(to + n(k) * cs) = temp_ctrl1.row(co + (ndom_pts(k) - 1) * cs);
    }
    // final dim if odd is copied from temp_ctrl0 to ctrl_pts
    else if (k == ndims - 1 && k % 2 == 1)
    {
        // debug
        // fprintf(stderr, "t_start[%ld] = d[%ld]\n", to, co);
        ctrl_pts.row(to) = temp_ctrl0.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t_end[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        ctrl_pts.row(to + n(k) * cs) = temp_ctrl0.row(co + (ndom_pts(k) - 1) * cs);
    }
}

// solves for one curve of control points
void CtrlCurve(VectorXi& p,          // polynomial degree in each dimension
               VectorXf& params,     // curve parameters for input points (1st dim changes fastest)
               VectorXf& knots,      // knots (1st dim changes fastest)
               MatrixXf& N,          // basis functions for current dimension
               MatrixXf& NtN,        // N^t * N
               MatrixXf& R,          // residual matrix for current dimension and curve
               MatrixXf& P,          // solved points for current dimension and curve
               MatrixXf& domain,     // input data points (1st dim changes fastest)
               VectorXi& ndom_pts,   // number of input domain points in each dimension
               VectorXi& n,          // number of control point spans in each dimension
               size_t    k,          // current dimension
               size_t    po,         // starting offset for params in current dim
               size_t    ko,         // starting offset for knots in current dim
               size_t    co,         // starting ofst for reading domain pts
               size_t    cs,         // stride for reading domain points
               size_t    to,         // starting ofst for writing control pts
               MatrixXf& temp_ctrl0, // first temporary control points buffer
               MatrixXf& temp_ctrl1, // second temporary control points buffer
               MatrixXf& ctrl_pts)   // final ctrl pts after last dim done (1st dim changes fastest)
{
    // compute R
    // first dimension reads from domain
    // subsequent dims alternate reading temp_ctrl0 and temp_ctrl1
    // even dim reads temp_ctrl1, odd dim reads temp_ctrl0; opposite of writing order
    // because what was written in the previous dimension is read in the current one
    if (k == 0)
        Residual(p(k), domain, knots, params, N, R, ko, po, co, cs);
    else if (k % 2)
        Residual(p(k), temp_ctrl0, knots, params, N, R, ko, po, co, cs);
    else
        Residual(p(k), temp_ctrl1, knots, params, N, R, ko, po, co, cs);

    // solve for P
    P = NtN.ldlt().solve(R);

    // append points from P to control points
    // TODO: any way to avoid this?
    CopyCtrl(P, domain, ndom_pts, n, k, co, cs, to, temp_ctrl0, temp_ctrl1, ctrl_pts);

    // debug
    // int ndims = ndom_pts.size();
    // cerr << "k " << k << " P:\n" << P << endl;
    // if (ndims == 1)
    //     cerr << "ctrl_pts:\n" << ctrl_pts << endl;
    // else if (k == 0)
    //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
    // else if (k % 2 == 0 && k < ndims - 1)
    //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
    // else if (k % 2 == 1 && k < ndims - 1)
    //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;
    // else if (k == ndims - 1)
    //     cerr << "ctrl_pts:\n" << ctrl_pts << endl;
}

// approximate a NURBS hypervolume of arbitrary dimension for a given input data set
// weights are all 1 for now
// n-d version of algorithm 9.7, Piegl & Tiller (P&T) p. 422
//
// the outputs, ctrl_pts and knots, are resized by this function;  caller need not resize them
//
// There are two types of dimensionality:
// 1. The dimensionality of the NURBS tensor product (p.size())
// (1D = NURBS curve, 2D = surface, 3D = volumem 4D = hypervolume, etc.)
// 2. The dimensionality of individual domain and control points (domain.cols())
// p.size() should be <= domain.cols()
void Encode(VectorXi& p,                   // polynomial degree in each dimension
            VectorXi& ndom_pts,            // number of input data points in each dim
            VectorXi& nctrl_pts,           // desired number of control points in each dim
            MatrixXf& domain,              // input data points (1st dim changes fastest)
            MatrixXf& ctrl_pts,            // (output) control points (1st dim changes fastest)
            VectorXf& knots)               // (output) knots (1st dim changes fastest)
{
    // check dimensionality for sanity
    assert(p.size() <= domain.cols());

    // debug
    // cerr << "domain:\n" << domain << endl;

    // TODO: preprocessing n-d domain requires some thought; skipping an preprocessing for now

    // debug
    // cerr << "new_domain:\n" << new_domain << endl;

    // check and assign main quantities
    VectorXi n;                 // number of control point spans in each domain dim
    VectorXi m;                 // number of input data point spans in each domain dim
    int      tot_nparams;       // total number of params = sum of ndom_pts over all dimensions
                                // not the total number of data points, which would be the product
    int      tot_nknots;        // total number of knots = sum of number of knots over all dims
    int      tot_nctrl;         // total number of control points
    int      ndims = ndom_pts.size();        // number of domain dimensions
    Quants(p, ndom_pts, nctrl_pts, n, m, tot_nparams, tot_nknots, tot_nctrl);

    // precompute curve parameters for input points
    VectorXf params(tot_nparams);
    Params(ndom_pts, domain, params);

    // debug
    // cerr << "params:\n" << params << endl;

    // compute knots
    knots.resize(tot_nknots);
    Knots(p, n, m, params, knots);

    // debug
    // cerr << "knots:\n" << knots << endl;

    // following are counters for slicing domain and params into curves in different dimensions
    size_t po = 0;                                // starting offset for params in current dim
    size_t ko = 0;                                // starting offset for knots in current dim
    size_t cs = 1;                                // stride for domain points in curve in cur. dim

    // control points
    ctrl_pts.resize(tot_nctrl, domain.cols());

    // 2 buffers of temporary control points
    // double buffer needed to write output curves of current dim without changing its input pts
    // temporary control points need to begin with size as many as the input domain points
    // except for the first dimension, which can be the correct number of control points
    // because the input domain points are converted to control points one dimension at a time
    // TODO: need to find a more space-efficient way
    size_t tot_ntemp_ctrl = 1;
    for (size_t k = 0; k < ndims; k++)
        tot_ntemp_ctrl *= (k == 0 ? nctrl_pts(k) : ndom_pts(k));
    MatrixXf temp_ctrl0 = MatrixXf::Zero(tot_ntemp_ctrl, domain.cols());
    MatrixXf temp_ctrl1 = MatrixXf::Zero(tot_ntemp_ctrl, domain.cols());

    VectorXi ntemp_ctrl = ndom_pts;               // current num of temp control pts in each dim

    for (size_t k = 0; k < ndims; k++)            // for all domain dimensions
    {
        // compute the matrix N, eq. 9.66 in P&T
        // N is a matrix of (m - 1) x (n - 1) scalars that are the basis function coefficients
        //  _                                _
        // |  N_1(u[1])   ... N_{n-1}(u[1])   |
        // |     ...      ...      ...        |
        // |  N_1(u[m-1]) ... N_{n-1}(u[m-1]) |
        //  -                                -
        // TODO: N is going to be very sparse when it is large: switch to sparse representation
        // N has semibandwidth < p  nonzero entries across diagonal
        MatrixXf N = MatrixXf::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)            // the rows of N
        {
            int span = FindSpan(p(k), n(k), knots, params(po + i), ko);
            // debug
            // fprintf(stderr, "p(k) %d n(k) %d span %d params(po + i) %.3f\n",
            //         p(k), n(k), span, params(po + i));
            assert(span - ko <= n(k));            // sanity
            BasisFuns(p(k), knots, params(po + i), span, N, 1, n(k) - 1, i - 1, ko);
        }

        // debug
        // cerr << "k " << k << " N:\n" << N << endl;

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixXf NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // debug
        // cerr << "k " << k << " NtN:\n" << NtN << endl;

        // R is the residual matrix needed for solving NtN * P = R
        MatrixXf R(n(k) - 1, domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixXf P(n(k) - 1, domain.cols());

        // number of curves in this dimension
        size_t ncurves;
        ncurves = 1;
        for (int i = 0; i < ndims; i++)
        {
            if (i < k)
                ncurves *= nctrl_pts(i);
            else if (i > k)
                ncurves *= ndom_pts(i);
            // NB: current dimension contributes no curves, hence no i == k case
        }
        // debug
        // cerr << "k: " << k << " ncurves: " << ncurves << endl;
        // cerr << "ndom_pts:\n" << ndom_pts << endl;
        // cerr << "ntemp_ctrl:\n" << ntemp_ctrl << endl;
        // if (k > 0 && k % 2 == 1) // input to odd dims is temp_ctrl0
        //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
        // if (k > 0 && k % 2 == 0) // input to even dims is temp_ctrl1
        //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;

        size_t co = 0, to = 0;                    // starting ofst for curve & ctrl pts in cur. dim
        size_t coo = 0, too = 0;                  // co and to at start of contiguous sequence
        for (size_t j = 0; j < ncurves; j++)      // for all the curves in this dimension
        {
            // debug
            // fprintf(stderr, "2: k %ld j %ld co %ld cs %ld to %ld\n", k, j, co, cs, to);

            CtrlCurve(p, params, knots, N, NtN, R, P, domain, ndom_pts, n, k,
                      po, ko, co, cs, to, temp_ctrl0, temp_ctrl1, ctrl_pts);

            // adjust offsets for the next curve
            if ((j + 1) % cs)
                co++;
            else
            {
                co = coo + cs * ntemp_ctrl(k);
                coo = co;
            }
            if ((j + 1) % cs)
                to++;
            else
            {
                to = too + cs * nctrl_pts(k);
                too = to;
            }
        }                                                  // cuves in this dimension

        // debug
        // if (k % 2 == 0 && k < ndims - 1)
        //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
        // else if (k % 2 && k < ndims - 1)
        //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;
        // else
        //     cerr << "ctrl_pts:\n" << ctrl_pts << endl;

        // adjust offsets and strides for next dimension
        po += ndom_pts(k);
        int nknots = n(k) + p(k) + 2;                     // number of knots in current dim
        ko += nknots;
        ntemp_ctrl(k) = nctrl_pts(k);
        cs *= ntemp_ctrl(k);

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // print progress
        fprintf(stderr, "\rdimension %ld of %d encoded", k + 1, ndims);

    }                                                      // domain dimensions

    fprintf(stderr,"\n");

    // debug
    // cerr << "ctrl_pts:\n" << ctrl_pts << endl;
}
