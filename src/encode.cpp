//--------------------------------------------------------------
// nurbs encoding algorithms
// ref: [P&T] Piegl & Tiller, The NURBS Book, 1995
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/pt.hpp>
#include <mfa/encode.hpp>

// eigen objects
#include <Eigen/Dense>

#include <vector>

using namespace std;
using namespace Eigen;

// binary search to find the span in the knots vector containing a given parameter value
// returns span index i s.t. u is in [ knots[i], knots[i + 1] )
// NB closed interval at left and open interval at right
// except when u == knots.last(), in which case the interval is closed at both ends
// i will be in the range [p, n], where n = number of control points - 1 because there are
// p + 1 repeated knots at start and end of knot vector
// algorithm 2.1, P&T, p. 68
int FindSpan(int            p,               // polynomial degree
             int            n,               // number of control point spans
             vector<float>& knots,           // knots
             float          u)               // parameter value
{
    if (u == knots[n + 1])
        return n;

    // binary search
    int low = p;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < knots[mid] || u >= knots[mid + 1])
    {
        if (u < knots[mid])
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

// computes p + 1 nonvanishing basis function values [N_{span - p}, N_{span}]
// of the given parameter value
// keeps only those in the range [N_{start_n}, N_{end_n}]
// writes results in a subset of a row of N starting at index N(start_row, start_col)
// algorithm 2.2 of P&T, p. 70
// assumes N has been allocated by caller
void BasisFuns(int            p,             // polynomial degree
               vector<float>& knots,         // knots
               float          u,             // parameter value
               int            span,          // index of span in the knots vector containing u
               MatrixXf&      N,             // matrix of (output) basis function values
               int            start_n,       // starting basis function N_{start_n} to compute
               int            end_n,         // ending basis function N_{end_n} to compute
               int            row)           // row in N where to write result
{
    // init
    vector<float> scratch(p + 1);            // scratchpad, same as N in P&T p. 70
    scratch[0] = 1.0;
    vector<float> left(p + 1);               // temporary recurrence results
    vector<float> right(p + 1);

    // fill N
    for (int j = 1; j <= p; j++)
    {
        left[j]  = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
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
        int n_i = span - p + j;              // index of basis function N_{n_i}
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

// computes R (residual) vector of P&T eq. 9.63 and 9.67, p. 411-412
void Residual2d(int                   p,     // polynomial degree
                vector<Pt <float> >&  domain,// domain of input data points
                vector<float>&        knots, // knots
                vector<float>&        params,// parameters of input points
                MatrixXf&             N,     // matrix of basis function coefficients
                MatrixXf&             R)     // (output) residual matrix allocated by caller
{
    int n      = N.cols() + 1;               // number of control point spans
    int m      = N.rows() + 1;               // number of input data point spans

    // compute the matrix Rk for eq. 9.63 of P&T, p. 411
    MatrixXf Rk(m - 1, 2);                   // NB, eigen frees dynamic memory when leaving scope
    for (int k = 1; k < m; k++)
    {
        int span = FindSpan(p, n, knots, params[k]);
        MatrixXf Nk = MatrixXf::Zero(1, n + 1); // basis coefficients for Rk[i]
        BasisFuns(p, knots, params[k], span, Nk, 0, n, 0);

        // debug
        // cerr << "Nk:\n" << Nk << endl;

        Rk(k - 1, 0) = domain[k][0] - Nk(0, 0) * domain[0][0] - Nk(0, n) * domain[m][0];
        Rk(k - 1, 1) = domain[k][1] - Nk(0, 0) * domain[0][1] - Nk(0, n) * domain[m][1];
    }

    // debug
    // cerr << "Rk:\n" << Rk << endl;

    // compute the matrix R
    for (int i = 1; i < n; i++)
    {
        R(i - 1, 0) = (N.col(i - 1).array() * Rk.col(0).array()).sum();
        R(i - 1, 1) = (N.col(i - 1).array() * Rk.col(1).array()).sum();
    }
}

// preprocess domain and range
// interpolate (1D) points to approximately uniform spacing
// TODO: normalize domain and range to similar scales
// new_domain and new_range are resized by Prep1d according to how many new points need to be added
void Prep1d(vector<Pt <float> >&  domain,    // domain of input data points
            vector<Pt <float> >&  new_domain)// new domain with interpolated data points
{
    vector<float> dists(domain.size() - 1);  // chord lengths of input data point spans
    float min_dist;                          // min and max distance

    // chord lengths
    for (size_t i = 0; i < domain.size() - 1; i++)
    {
        // TODO: normalize domain and range so they have similar scales
        dists[i] = Pt<float>::dist(domain[i], domain[i + 1]);
        if (i == 0)
            min_dist = dists[i];
        if (dists[i] < min_dist)
            min_dist = dists[i];
    }

    // debug
    // fprintf(stderr, "min_dist %.3f max_dist %.3f mean_dist %.3f\n",
    //         min_dist, max_dist, mean_dist);

    // TODO: experiment with different types (degrees) of interpolation; for now using linear
    // interpolation based on min distance, so that new data are added, but no original data
    // points are removed

    // copy domain and range to new versions, adding interpolated points as needed
    for (size_t i = 0; i < domain.size() - 1; i++)
    {
        new_domain.push_back(domain[i]);
        int nextra_pts = dists[i] / min_dist - 1;     // number of extra points to insert
        for (int j = 0; j < nextra_pts; j++)
        {
            float fd = (j + 1) * min_dist / dists[i]; // fraction of distance to add
            Pt<float> p(domain[i] + fd * (domain[i + 1] - domain[i]));
            new_domain.push_back(p);
        }
    }
    // copy last point
    new_domain.push_back(domain[domain.size() - 1]);
}

// precompute curve parameters for input data points using the chord-length method
// 1D version of algorithm 9.3, P&T, p. 377
// assumes params were allocated by caller
// TODO: investigate other schemes (domain only, normalized domain and range, etc.)
void Params1d(vector<Pt <float> >&  domain,  // domain of input data points
              vector<float>&        params)  // (output) curve parameters
{
    int nparams    = domain.size();          // number of parameters = number of input points
    float tot_dist = 0.0;                    // total chord length
    vector<float> dists(domain.size() - 1);  // chord lengths of input data point spans

    // chord lengths
    for (size_t i = 0; i < nparams - 1; i++)
    {
        // TODO: normalize domain and range so they have similar scales
        dists[i] = Pt<float>::dist(domain[i], domain[i + 1]);
        // fprintf(stderr, "dists[%lu] = %.3f\n", i, dists[i]);
        tot_dist += dists[i];
    }

    // parameters
    params[0]           = 0.0;               // first parameter is known
    params[nparams - 1] = 1.0;               // last parameter is known
    for (size_t i = 0; i < nparams - 2; i++)
        params[i + 1] = params[i] + dists[i] / tot_dist;
}

// approximate a NURBS curve for a given input 1D data set
// weights are all 1 for now
// 1D version of algorithm 9.7, Piegl & Tiller (P&T) p. 422
void Approx1d(int                   p,        // polynomial degree
              int                   nctrl_pts,// desired number of control points
              int                   dim,      // point dimensionality
              vector<Pt <float> >&  domain,   // domain of input data points
              vector<Pt <float> >&  ctrl_pts, // (output) control points
              vector<float>&        knots)    // (output) knots
{
    if (nctrl_pts <= p)
    {
        fprintf(stderr, "Error: Approx1d() number of control points must be at least p + 1\n");
        exit(1);
    }

    // preprocess domain and range
    vector<Pt <float> > new_domain;
    Prep1d(domain, new_domain);

    // copy back TODO: decide whether want to change original data or not
    domain.resize(new_domain.size());
    for (size_t i = 0; i < domain.size(); i++)
        domain[i] = new_domain[i];           // asssigment automatically resizes

    int n      = nctrl_pts - 1;              // number of control point spans
    int m      = domain.size() - 1;          // number of input data point spans
    int nknots = n + p + 2;                  // number of knots

    knots.resize(nknots);
    ctrl_pts.resize(nctrl_pts);
    for (size_t i = 0; i < ctrl_pts.size(); i++)
        ctrl_pts[i].resize(dim);

    // precompute curve parameters for input points
    vector<float> params(domain.size());
    Params1d(domain, params);

    // debug
    // cerr << "params: ";
    // for (int i = 0; i < domain.size(); i++)
    //     cerr << params[i] << " ";
    // cerr << endl;

    // eg, for p = 3 and nctrl_pts = 7, n = nctrl_pts - 1 = 6 and nknots = n + p + 2 = 11
    // let knots = {0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1}
    // there are p + 1 external knots at each end: {0, 0, 0, 0} and {1, 1, 1, 1}
    // there are n - p internal knots: {0.25, 0.5, 0.75}
    // there are n - p + 1 internal knot spans [0,0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1)

    // Eq. 9.68 in P&T
    // d is the ratio of number of input points (r+1) to internal knot spans (n-p+1)
    // float d = (float)(r + 1) / (n - p + 1);         // eq. 9.68, r is P&T's m

    // but I prefer d to be the ratio of input spans r to internal knot spans (n-p+1)
    float d = (float)m / (n - p + 1);

    // compute n - p internal knots
    for (int j = 1; j <= n - p; j++)          // eq. 9.69
    {
        int   i = j * d;                      // integer part of j steps of d
        float a = j * d - i;                  // fractional part of j steps of d, P&T's alpha

        // debug
        // cerr << "d " << d << " j " << j << " i " << i << " a " << a << endl;

        // when using P&T's eq. 9.68, compute knots using the following
        // knots[p + j] = (1.0 - a) * params[i - 1] + a * params[i];

        // when using my version of d, use the following
        knots[p + j] = (1.0 - a) * params[i] + a * params[i + 1];
    }

    // set external knots
    for (int i = 0; i < p + 1; i++)
    {
        knots[i] = 0.0;
        knots[nknots - 1 - i] = 1.0;
    }

    // debug
    // cerr << "knots: ";
    // for (int i = 0; i < nknots; i++)
    //     cerr << knots[i] << " ";
    // cerr << endl;

    // compute the matrix N, eq. 9.66 in P&T
    // N is a matrix of (m - 1) x (n - 1) scalars that are the basis function coefficients
    //  _                                _
    // |  N_1(u[1])   ... N_{n-1}(u[1])   |
    // |     ...      ...      ...        |
    // |  N_1(u[m-1]) ... N_{n-1}(u[m-1]) |
    //  -                                -
    // TODO: N is going to be very sparse when it is large: switch to sparse representation
    // N has semibandwidth < p  nonzero entries across diagonal
    MatrixXf N = MatrixXf::Zero(m - 1, n - 1); // coefficients matrix
                                               // NB, eigen frees dynamic memory when leaving scope
    for (int i = 1; i < m; i++)                // the rows of N
    {
        int span = FindSpan(p, n, knots, params[i]);

        // debug
        // cerr << "params[" << i << "] " << params[i] << " span " << span << endl;

        assert(span <= n);                     // sanity
        BasisFuns(p, knots, params[i], span, N, 1, n - 1, i - 1);
    }

    // debug
    // cerr << "N = \n" << N << endl;

    // compute the product Nt x N
    // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
    // NtN has semibandwidth < p + 1 nonzero entries across diagonal
    MatrixXf NtN(n - 1, n - 1);               // NB, eigen frees dynamic memory when leaving scope
    NtN = N.transpose() * N;

    // TODO: N can be freed at this point
    // learn how to free a matrix in eigen before it goes out of scope

    // debug
    // cerr << "NtN = \n" << NtN << endl;

    // compute R
    MatrixXf R(n - 1, 2);                     // NB, eigen frees dynamic memory when leaving scope
    Residual2d(p, domain, knots, params, N, R);

    // debug
    // cerr << "R = \n" << R << endl;

    // solve NtN * P = R
    // NtN is positive definite -> do not need pivoting
    // P are the unknown interior control points
    // TODO: use a common representation for P and ctrl_pts to avoid copying
    MatrixXf P(n - 1, 2);                     // NB, eigen frees dynamic memory when leaving scope
    P = NtN.ldlt().solve(R);

    // debug
    // cerr << "P = \n" << P << endl;

    // init first and last control points
    ctrl_pts[0] = domain[0];
    ctrl_pts[n] = domain[m];

    // copy rest of control points
    // TODO: use a common representation for P and ctrl_pts to avoid copying
    for (int i = 0; i < n - 1; i++)
    {
        ctrl_pts[i + 1][0] = P(i, 0);
        ctrl_pts[i + 1][1] = P(i, 1);
    }
}
