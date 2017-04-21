//--------------------------------------------------------------
// mfa object
// ref: [P&T] Piegl & Tiller, The NURBS Book, 1995
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#include <mfa/encode.hpp>
#include <mfa/decode.hpp>

#include <Eigen/Dense>

#include <vector>
#include <iostream>
#include <set>

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
mfa::
MFA::
MFA(VectorXi& p_,             // polynomial degree in each dimension
    VectorXi& ndom_pts_,      // number of input data points in each dim
    VectorXi& nctrl_pts_,     // desired number of control points in each dim
    MatrixXf& domain_,        // input data points (1st dim changes fastest)
    MatrixXf& ctrl_pts_,      // (output) control points (1st dim changes fastest)
    VectorXf& knots_,         // (output) knots (1st dim changes fastest)
    float     eps_) :         // minimum difference considered significant
    p(p_),
    ndom_pts(ndom_pts_),
    nctrl_pts(nctrl_pts_),
    domain(domain_),
    ctrl_pts(ctrl_pts_),
    knots(knots_),
    eps(eps_)
{
    // check dimensionality for sanity
    assert(p.size() < domain.cols());

    // max extent of input data points
    dom_range = domain.maxCoeff() - domain.minCoeff();

    // debug
    // cerr << "domain:\n" << domain << endl;

    // total number of params = sum of ndom_pts over all dimensions
    // not the total number of data points, which would be the product
    tot_nparams = ndom_pts.sum();
    // total number of knots = sum of number of knots over all dims
    tot_nknots = 0;
    for (size_t i = 0; i < p.size(); i++)
        tot_nknots  += (nctrl_pts(i) + p(i) + 1);

    // precompute curve parameters for input points
    params.resize(tot_nparams);
    Params();

    // debug
//     cerr << "params:\n" << params << endl;

    // compute knots
    knots.resize(tot_nknots);
    Knots();

    // debug
    // cerr << "knots:\n" << knots << endl;

    // offsets and strides for knots, params, and control points in different dimensions
    // TODO: co for control points currently not used because control points are stored explicitly
    // in future, store them like params, x coords followed by y coords, ...
    // then co will be used
    ko.resize(p.size(), 0);                  // offset for knots
    po.resize(p.size(), 0);                  // offset for params
    co.resize(p.size(), 0);                  // offset for control points
    cs.resize(p.size(), 1);                  // stride for control points
    ds.resize(p.size(), 1);                  // stride for domain points
// DEPRECATED
//     ks.resize(p.size(), 1);                  // stride for knots
    for (size_t i = 1; i < p.size(); i++)
    {
        po[i] = po[i - 1] + ndom_pts[i - 1];
        ko[i] = ko[i - 1] + nctrl_pts[i - 1] + p[i - 1] + 1;
        co[i] = co[i - 1] * nctrl_pts[i - 1];
        ds[i] = ds[i - 1] * ndom_pts[i - 1];
// DEPRECATED
//         ks[i] = ks[i - 1] * (nctrl_pts[i - 1] + p[i - 1] + 1);
    }

    // knot span index table
    KnotSpanIndex();
}

// initialize knot span index
void
mfa::
MFA::
KnotSpanIndex()
{
    // total number of knot spans = product of number of knot spans over all dims
    size_t int_nspans = 1;                  // number of internal (unique) spans
    size_t all_nspans = 1;                  // total number of spans, including repeating 0s and 1s
    for (auto i = 0; i < p.size(); i++)
    {
        int_nspans *= (nctrl_pts(i) - p(i));
        all_nspans *= (nctrl_pts(i) + p(i));
    }

    knot_spans.resize(int_nspans);

    // for all knot spans, fill the KnotSpan fields
    VectorXi ijk   = VectorXi::Zero(p.size());      // i,j,k of start of span
    VectorXi p_ijk = VectorXi::Zero(p.size());      // i,j,k of parameter
    size_t span_idx = 0;                            // current index into knot_spans
    for (auto i = 0; i < all_nspans; i++)           // all knot spans (including repeated 0s and 1s)
    {
        // skip repeating knot spans
        bool skip = false;
        for (auto k = 0; k < p.size(); k++)             // dimensions
            if ((ijk(k) < p[k]) || ijk(k) >= nctrl_pts[k])
            {
                skip = true;
                break;
            }

        // save knot span
        // TODO: may not be necessary to store all the knot span fields, but for now it is
        // convenient; recheck later to see which are actually used
        // unused ones can be computed locally below but not part of the knot span struct
        if (!skip)
        {
            // knot ijk
            knot_spans[span_idx].min_knot_ijk = ijk;
            knot_spans[span_idx].max_knot_ijk = ijk.array() + 1;

            // knot values
            knot_spans[span_idx].min_knot.resize(p.size());
            knot_spans[span_idx].max_knot.resize(p.size());
            for (auto k = 0; k < p.size(); k++)         // dimensions
            {
                knot_spans[span_idx].min_knot(k) = knots(ko[k] + knot_spans[span_idx].min_knot_ijk(k));
                knot_spans[span_idx].max_knot(k) = knots(ko[k] + knot_spans[span_idx].max_knot_ijk(k));
            }

            // parameter ijk and parameter values
            knot_spans[span_idx].min_param.resize(p.size());
            knot_spans[span_idx].max_param.resize(p.size());
            knot_spans[span_idx].min_param_ijk.resize(p.size());
            knot_spans[span_idx].max_param_ijk.resize(p.size());
            VectorXi po_ijk = p_ijk;                    // remember starting param ijk
            for (auto k = 0; k < p.size(); k++)         // dimensions in knot spans
            {
                // min param ijk and value
                knot_spans[span_idx].min_param_ijk(k) = p_ijk(k);
                knot_spans[span_idx].min_param(k)     = params(po[k] + p_ijk(k));

                // max param ijk and value
                // most spans are half open [..., ...)
                while (params(po[k] + p_ijk(k)) < knot_spans[span_idx].max_knot(k))
                {
                    knot_spans[span_idx].max_param_ijk(k) = p_ijk(k);
                    knot_spans[span_idx].max_param(k)     = params(po[k] + p_ijk(k));
                    p_ijk(k)++;
                }
                // the last span in each dimension is fully closed [..., ...]
                if (p_ijk(k) == ndom_pts(k) - 1)
                {
                    knot_spans[span_idx].max_param_ijk(k) = p_ijk(k);
                    knot_spans[span_idx].max_param(k)     = params(po[k] + p_ijk(k));
                }
            }

            // increment param ijk
            for (auto k = 0; k < p.size(); k++)     // dimension in params
            {
                if (p_ijk(k) < ndom_pts[k] - 1)
                {
                    po_ijk(k) = p_ijk(k);
                    break;
                }
                else
                {
                    po_ijk(k) = 0;
                    if (k < p.size() - 1)
                        po_ijk(k + 1)++;
                }
            }
            p_ijk = po_ijk;

            knot_spans[span_idx].last_split_dim = -1;
            knot_spans[span_idx].done           = false;
            ndone_knot_spans                    = 0;

            // debug
//             cerr <<
//                 "spand_idx="         << span_idx                           <<
//                 "\nmin_knot_ijk:\n"  << knot_spans[span_idx].min_knot_ijk  <<
//                 "\nmax_knot_ijk:\n"  << knot_spans[span_idx].max_knot_ijk  <<
//                 "\nmin_knot:\n"      << knot_spans[span_idx].min_knot      <<
//                 "\nmax_knot:\n"      << knot_spans[span_idx].max_knot      <<
//                 "\nmin_param_ijk:\n" << knot_spans[span_idx].min_param_ijk <<
//                 "\nmax_param_ijk:\n" << knot_spans[span_idx].max_param_ijk <<
//                 "\nmin_param:\n"     << knot_spans[span_idx].min_param     <<
//                 "\nmax_param:\n"     << knot_spans[span_idx].max_param     <<
//                 "\n\n"               << endl;

            span_idx++;
        }                                               // !skip

        // increment knot ijk
        for (auto k = 0; k < p.size(); k++)             // dimension in knot spans
        {
            if (ijk(k) < nctrl_pts[k] + p[k] - 1)
            {
                ijk(k)++;
                break;
            }
            else
                ijk(k) = 0;
        }
    }
}

// encode
void
mfa::
MFA::
Encode()
{
    mfa::Encoder encoder(*this);
    encoder.Encode();
}

// re-encode with insertion of new knots into existing knots
// returns whether normalized (relative) error in all knot spans is below err_limit
bool
mfa::
MFA::
Encode(float err_limit)                      // maximum allowable normalized error
{
    VectorXi nnew_knots;                     // number of new knots in each dim
    VectorXf new_knots;                      // new knots (1st dim changes fastest)

    bool done = ErrorSpans(nnew_knots, new_knots, err_limit);

    if (!done)
    {
        mfa::Encoder encoder(*this);
        encoder.Encode();
    }

    return done;
}

// computes error in knot spans and adds knots to be inserted into nnew_knots and new_knots
// returns whether normalized (relative) error in all knot spans is below err_limit
bool
mfa::
MFA::
ErrorSpans(
        VectorXi& nnew_knots,                   // number of new knots in each dim
        VectorXf& new_knots,                    // new knots (1st dim changes fastest)
        float err_limit)                        // max allowable relative error
{
    mfa::Decoder decoder(*this);
    return decoder.ErrorSpans(nnew_knots, new_knots, err_limit);
}

// decode
void
mfa::
MFA::
Decode(MatrixXf& approx)
{
    mfa::Decoder decoder(*this);
    decoder.Decode(approx);
}

// DEPRECATED
// error (distance in normal direction) from a point to the domain points
// (error is signed and not normalized by data range)
// float
// mfa::
// MFA::
// Error(VectorXf& pt,               // point some distance away from domain points
//       int       idx)              // index of point in domain near to the point
//                                   // search for cell containing the point starting at this index
// {
//     mfa::Encoder encoder(*this);
//     return NormalDistance(pt, idx);
// }

// absolute value of error (distance in normal direction) of the mfa at a domain
// point (error is absolute value but not normalized by data range)
float
mfa::
MFA::
Error(size_t idx)                   // index of domain point
{
    mfa::Encoder encoder(*this);
    return encoder.Error(idx);
}

// binary search to find the span in the knots vector containing a given parameter value
// returns span index i s.t. u is in [ knots[i], knots[i + 1] )
// NB closed interval at left and open interval at right
// except when u == knots.last(), in which case the interval is closed at both ends
// i will be in the range [p, n], where n = number of control points - 1 because there are
// p + 1 repeated knots at start and end of knot vector
// algorithm 2.1, P&T, p. 68
int
mfa::
MFA::
FindSpan(int       cur_dim,              // current dimension
         float     u,                    // parameter value
         int       ko)                   // optional starting knot to search (default = 0)
{
    if (u == knots(ko + nctrl_pts(cur_dim)))
        return ko + nctrl_pts(cur_dim) - 1;

    // binary search
    int low = p(cur_dim);
    int high = nctrl_pts(cur_dim);
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
void
mfa::
MFA::
BasisFuns(int       cur_dim,            // current dimension
          float     u,                  // parameter value
          int       span,               // index of span in the knots vector containing u
          MatrixXf& N,                  // matrix of (output) basis function values
          int       start_n,            // starting basis function N_{start_n} to compute
          int       end_n,              // ending basis function N_{end_n} to compute
          int       row,                // starting row index in N of result
          int       ko)                 // optional starting knot to search (default = 0)
{
    // init
    vector<float> scratch(p(cur_dim) + 1);            // scratchpad, same as N in P&T p. 70
    scratch[0] = 1.0;
    vector<float> left(p(cur_dim) + 1);               // temporary recurrence results
    vector<float> right(p(cur_dim) + 1);

    // debug
    // fprintf(stderr, "param=%.3f span=%d\n", u, span);

    // fill N
    for (int j = 1; j <= p(cur_dim); j++)
    {
        left[j]  = u - knots(span + 1 - j);
        right[j] = knots(span + j) - u;

        // debug
        // fprintf(stderr, "min: knot[%d]=%.3f max: knot[%d]=%.3f\n",
        //         span + 1 - j, knots(span + 1 - j), span + j, knots(span + j));

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
    // for (int i = 0; i < p(cur_dim) + 1; i++)
    //     cerr << scratch[i] << " ";
    // cerr << endl;

    // copy scratch to N
    for (int j = 0; j < p(cur_dim) + 1; j++)
    {
        int n_i = span - ko - p(cur_dim) + j;              // index of basis function N_{n_i}
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

// precompute curve parameters for input data points using the chord-length method
// n-d version of algorithm 9.3, P&T, p. 377
// params are computed along curves and averaged over all curves at same data point index i,j,k,...
// ie, resulting params for a data point i,j,k,... are same for all curves
// and params are only stored once for each dimension in row-major order (1st dim changes fastest)
// total number of params is the sum of ndom_pts over the dimensions, much less than the total
// number of data points (which would be the product)
// assumes params were allocated by caller
// TODO: investigate other schemes (domain only, normalized domain and range, etc.)
void
mfa::
MFA::
Params()
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
// resulting knots are same for all curves and stored once for each dimension in
// row-major order (1st dim changes fastest)
// total number of knots is the sum of number of knots over the dimensions,
// much less than the product
// assumes knots were allocated by caller
void
mfa::
MFA::
Knots()
{
    // following are counters for slicing domain and params into curves in different dimensions
    size_t po = 0;                                // starting offset for params in current dim
    size_t ko = 0;                                // starting offset for knots in current dim

    for (size_t k = 0; k < p.size(); k++)         // for all domain dimensions
    {
        int nknots = nctrl_pts(k) + p(k) + 1;    // number of knots in current dim

        // in P&T, d is the ratio of number of input points (r+1) to internal knot spans (n-p+1)
        // float d = (float)(r + 1) / (n - p + 1);         // eq. 9.68, r is P&T's m
        // but I prefer d to be the ratio of input spans r to internal knot spans (n-p+1)
        float d = (float)(ndom_pts(k) - 1) / (nctrl_pts(k) - p(k));

        // compute n - p internal knots
        for (int j = 1; j <= nctrl_pts(k) - p(k) - 1; j++)    // eq. 9.69
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

        po += ndom_pts(k);
        ko += nknots;
    }
}

// DEPRECATED
// insert new knots (one for each domain dim) for the worst control point (the one with the most
// solution error)
//
// void
// mfa::
// MFA::
// InsertKnots()
// {
//     // find values of the knots
//     VectorXf knot(ndom_pts.size());             // the new knot in each domain dimension
//     for (size_t i = 0; i < ndom_pts.size(); i++)
//         knot(i) = InterpolateParams(i, po[i], ds[i], ctrl_pts(worst_ctrl_idx, i));
// 
//     // insert the knots
//     VectorXi nnew_knots = VectorXi::Ones(ndom_pts.size());
//     InsertKnots(nnew_knots, knot);
// 
//     // debug
//     cerr << "knot to be inserted:\n" << knot << endl;
// }

// DEPRECATED
// computes additional knot locations where error threshold is exceeded
//
// original, inserted, and resulting new knots are same for all curves and
// stored once for each dimension in row-major order (1st dim changes fastest)
//
// void
// mfa::
// MFA::
// FindExtraKnots(VectorXi& nnew_knots,    // number of new knots in each dim
//                VectorXf& new_knots,     // new knots (1st dim changes fastest)
//                float     err_limit)     // max error limit
// {
//     // data range for error normalization
//     float min   = domain.minCoeff();
//     float max   = domain.maxCoeff();
//     float range = max - min;
// 
//     // i,j,k domain coordinates for following iteration over approximated points
//     VectorXi ijk = VectorXi::Zero(ndom_pts.size()); // TODO: domain size, not pt size?
// 
//     // i,j,k coordinates of knots to add in each dimension
//     vector < set <int> > new_knot_indices(ndom_pts.size());
// 
//     // normal distance computation
//     // TODO: removed approx, need to decode one point at a time
//     for (size_t i = 0; i < approx.rows(); i++)
//     {
//         // debug
//         // cerr << "ijk:\n" << ijk << endl;
// 
//         VectorXf approx_pos = approx.block(i, 0, 1, p.size()).row(0);
//         VectorXf approx_pt  = approx.row(i);
//         float    err        = Error(approx_pt, i) / range;
// 
//         if (fabs(err) > err_limit)
//         {
//             // add the knot ijk indices
//             for (size_t j = 0; j < ijk.size(); j++)
//                 new_knot_indices[j].insert(ijk(j));
//         }
// 
//         // increment ijk indices
//         for (size_t j = 0; j < ijk.size(); j++)
//         {
//             ijk(j) = (ijk(j) + 1) % ndom_pts(j);
//             if (ijk(j))
//                 break;
//         }
//     }
// 
//     // compute sizes and resize nnew_knots and new_knots
//     int tot_nnew_knots = 0;                 // total number of new knots added
//     for (size_t j = 0; j < ijk.size(); j++) // for each domain dim j
//         tot_nnew_knots += new_knot_indices[j].size();
//     nnew_knots.resize(ndom_pts.size());
//     new_knots.resize(tot_nnew_knots);
// 
//     // convert knot indices to knot values in new_knots, set quantities of nnew_knots
//     size_t n  = 0;                   // linear counter in new_knots
//     size_t p0 = 0;                   // start of current dim in params
//     for (size_t j = 0; j < ijk.size(); j++) // for each domain dim j
//     {
//         nnew_knots(j) = new_knot_indices[j].size();
// 
//         // debug
//         // fprintf(stderr, "nnew_knots(%lu)=%d\n", j, nnew_knots(j));
// 
//         for (set<int>::iterator it = new_knot_indices[j].begin();
//              it != new_knot_indices[j].end(); it++)
//         {
//             // debug
//             // fprintf(stderr, "ijk=%i\n", *it);
// 
//             new_knots[n++] = params[p0 + *it];
//         }
//         p0 += ndom_pts(j);
//     }

    // debug
    // cerr << "nnew_knots:\n" << nnew_knots << endl;
    // cerr << "new_knots:\n" << new_knots << endl;
// }

// inserts a set of knots (in all dimensions) into the original knot set
// also increases the numbers of control points (in all dimensions) that will result
//
// original, inserted, and resulting new knots are same for all curves and
// stored once for each dimension in row-major order (1st dim changes fastest)
void
mfa::
MFA::
InsertKnots(VectorXi& nnew_knots,     // number of new knots in each dim
            VectorXf& new_knots)      // new knots (1st dim changes fastest)
{
    // total number of new knots
    int tot_nnew_knots = 0;                 // total number of new knots added
    for (size_t i = 0; i < nnew_knots.size(); i++) // for each domain dim j
        tot_nnew_knots += nnew_knots(i);

    VectorXf temp_knots(knots.size() + tot_nnew_knots);
    VectorXi nold_knots = VectorXi::Zero(nnew_knots.size());

    size_t ntemp = 0;                             // current number of temp_knots
    size_t n     = 0;                             // counter into knots
    size_t m     = 0;                             // counter into new_knots
    size_t nk    = 0;                             // current number of old knots copied in cur. dim
    size_t mk    = 0;                             // current number of new knots copied in cur. dim

    // copy knots to temp_knots, inserting new_knots along the way
    for (size_t k = 0; k < nnew_knots.size(); k++) // for each domain dimension i
    {
        nold_knots(k) = nctrl_pts(k) + p(k) + 1;  // old number of knots in current dim

        // TODO: in the following, ensure knots are not duplicated (to within epsilon difference?)

        // walk the old knots and insert new ones
        nk = 0;
        while (nk < nold_knots(k))
        {
            if (mk < nnew_knots(k) && new_knots(m) < knots(n))
            {
                // debug
                // fprintf(stderr, "ntemp+1=%d m+1=%d\n", ntemp + 1, m + 1);
                temp_knots(ntemp++) = new_knots(m++);
                mk++;
            }
            else
            {
                temp_knots(ntemp++) = knots(n++);
                nk++;
            }
        }

        mk = 0;
    }

    // debug
//     cerr << "nknots before insertion:\n" << nold_knots << endl;
//     cerr << "knots before insertion:\n" << knots << endl;
//     cerr << "nctrl_pts before insertion:\n" << nctrl_pts << endl;

    // copy temp_knots back to knots
    knots.resize(temp_knots.size());
    knots = temp_knots;

    // debug
//     cerr << "nnew_knots:\n" << nnew_knots << endl;
//     cerr << "knots after insertion:\n" << knots << endl;

    // increase number of control points
    nctrl_pts += nnew_knots;

    // debug
    cerr << "nctrl_pts after insertion:\n" << nctrl_pts << endl;
}

// interpolate parameters to get parameter value for a target coordinate
//
// TODO: experiment whether this is more accurate and/or faster than calling Params
// with a 1-d space of domain pts: min, target, and max
float
mfa::
MFA::
InterpolateParams(int       cur_dim,  // curent dimension
                  size_t    po,       // starting offset for params in current dim
                  size_t    ds,       // stride for domain pts in cuve in cur. dim.
                  float     coord)    // target coordinate
{
    if (coord <= domain(0, cur_dim))
        return params(po);

    if (coord >= domain((ndom_pts(cur_dim) - 1) * ds, cur_dim))
        return params(po + ndom_pts(cur_dim) - 1);

    // binary search
    int low = 0;
    int high = ndom_pts(cur_dim);
    int mid = (low + high) / 2;
    while (coord < domain((mid) * ds, cur_dim) ||
           coord >= domain((mid + 1) * ds, cur_dim))
    {
        if (coord < domain((mid) * ds, cur_dim))
            high = mid;
        else
            low = mid;
        mid = (low + high) / 2;
    }

    // debug
    // fprintf(stderr, "binary search param po=%ld mid=%d param= %.3f\n", po, mid, params(po + mid));

    // interpolate
    // TODO: assumes the domain is ordered in increasing coordinate values
    // very dangerous!
    if (coord <= domain((mid) * ds, cur_dim) && mid > 0)
    {
        assert(coord >= domain((mid - 1) * ds, cur_dim));
        float frac = (coord - domain((mid - 1) * ds, cur_dim)) /
            (domain((mid) * ds, cur_dim) - domain((mid - 1) * ds, cur_dim));
        return params(po + mid - 1) + frac * (params(po + mid) - params(po + mid - 1));
    }
    else if (coord >= domain((mid) * ds, cur_dim) && mid < ndom_pts(cur_dim) - 1)
    {
        assert(coord <= domain((mid + 1) * ds, cur_dim));
        float frac = (coord - domain((mid) * ds, cur_dim)) /
            (domain((mid + 1) * ds, cur_dim) - domain((mid) * ds, cur_dim));
        return params(po + mid) + frac * (params(po + mid + 1) - params(po + mid));
    }
    else
        return params(po + mid, cur_dim);

    // TODO: iterate and get the param to match the target coord even closer
    // resulting coord when the param is used is within about 10^-3
}

// convert linear domain point index into (i,j,k,...) multidimensional index
// number of dimensions is the domain dimensionality
// (not domain * range dimensionality, ie, p.size(), not domain_point.cols())
void
mfa::
MFA::
idx2ijk(size_t    idx,                  // linear cell indx
        VectorXi& ijk)                  // i,j,k,... indices in all dimensions
{
    if (p.size() == 1)
    {
        ijk(0) = idx;
        return;
    }

    for (int i = 0; i < p.size(); i++)
    {
        if (i < p.size() - 1)
            ijk(i) = (idx % ds[i + 1]) / ds[i];
        else
            ijk(i) = idx / ds[i];
    }

    // debug: check the answer
    // int check_idx = 0;
    // for (int i = 0; i < p.size(); i++)
    //     check_idx += ijk(i) * ds[i];
    // assert(idx == check_idx);

    // debug
    // cerr << "idx=" << idx << "ijk\n" << ijk << endl;
}

// convert (i,j,k,...) multidimensional index into linear index into domain
// number of dimension is the domain dimensionality (p.size()), not
// domain + range dimensionality (domain_points.size())
void
mfa::
MFA::
ijk2idx(VectorXi& ijk,                  // i,j,k,... indices to all dimensions
        size_t&   idx)                  // (output) linear index
{
    idx           = 0;
    size_t stride = 1;
    for (int i = 0; i < p.size(); i++)
    {
        idx += ijk(i) * stride;
        stride *= ndom_pts(i);
    }
}

// signed normal distance from a point to the domain
// uses 2-point finite differences (first order linear) method to compute gradient and normal vector
// approximates gradient from 2 points diagonally opposite each other in all
// domain dimensions (not from 2 points in each dimension)
float
mfa::
MFA::
NormalDistance(VectorXf& pt,          // point whose distance from domain is desired
               size_t    idx)         // index of min. corner of cell in the domain
                                      // that will be used to compute partial derivatives
                                      // (linear) search for correct cell will start at this index
{
    // normal vector = [df/dx, df/dy, df/dz, ..., -1]
    // -1 is the last coordinate of the domain points, ie, the range value
    VectorXf normal(domain.cols());
    int      last = domain.cols() - 1;    // last coordinate of a domain pt, ie, the range value

    // convert linear idx to multidim. i,j,k... indices in each domain dimension
    VectorXi ijk(p.size());
    idx2ijk(idx, ijk);

    // compute i0 and i1 1d and ijk0 and ijk1 nd indices for two points in the cell in each dim.
    // even though the caller provided the minimum corner index as idx, it's
    // possible that idx is at the max end of the domain in some dimension
    // in this case we set i1 <- idx and i0 to be one less
    size_t i0, i1;                          // 1-d indices of min, max corner points
    VectorXi ijk0(p.size());                // n-d ijk index of min corner
    VectorXi ijk1(p.size());                // n-d ijk index of max corner
    for (int i = 0; i < p.size(); i++)      // for all domain dimensions
    {
        // at least 2 points needed in each dimension
        // TODO: do something degenerate if not, but probably will never get to this point
        // because there will be insufficient points to encode in the first place
        assert(ndom_pts(i) >= 2);

        // two opposite corners of the cell as i,j,k coordinates
        if (ijk(i) + 1 < ndom_pts(i))
        {
            ijk0(i) = ijk(i);
            ijk1(i) = ijk(i) + 1;
        }
        else
        {
            ijk0(i) = ijk(i) - 1;
            ijk1(i) = ijk(i);
        }
    }

    // set i0 and i1 to be the 1-d indices of the corner points
    ijk2idx(ijk0, i0);
    ijk2idx(ijk1, i1);

    // compute the normal to the domain at i0 and i1
    for (int i = 0; i < p.size(); i++)      // for all domain dimensions
        normal(i) = (domain(i1, last) - domain(i0, last)) / (domain(i1, i) - domain(i0, i));
    normal(last) = -1;
    normal /= normal.norm();

    // project distance from (pt - domain(idx)) to unit normal
    VectorXf dom_pt = domain.row(idx);

    // debug
    // fprintf(stderr, "idx=%d\n", idx);
    // cerr << "unit normal\n" << normal << endl;
    // cerr << "point\n" << pt << endl;
    // cerr << "domain point:\n" << dom_pt << endl;
    // cerr << "pt - dom_pt:\n" << pt - dom_pt << endl;
    // fprintf(stderr, "projection = %e\n\n", normal.dot(pt - dom_pt));

    return normal.dot(pt - dom_pt);
}
