
// decoder object
// ref: [P&T] Piegl & Tiller, The NURBS Book, 1995
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <mfa/decode.hpp>

#include <Eigen/Dense>

#include <vector>
#include <iostream>

using namespace std;

template <typename T>                            // float or double
mfa::
Decoder<T>::
Decoder(MFA<T>& mfa_) :
    mfa(mfa_)
{
    // ensure that encoding was already done
    if (!mfa.p.size()         ||
        !mfa.ndom_pts.size()  ||
        !mfa.nctrl_pts.size() ||
        !mfa.domain.size()    ||
        !mfa.params.size()    ||
        !mfa.ctrl_pts.size()  ||
        !mfa.knots.size())
    {
        fprintf(stderr, "Decoder() error: Attempting to decode before encoding.\n");
        exit(0);
    }

    // initialize decoding data structures
    cs.resize(mfa.p.size(), 1);
    tot_iters = 1;                              // total number of iterations in the flattened decoding loop
    for (size_t i = 0; i < mfa.p.size(); i++)   // for all dims
    {
        tot_iters  *= (mfa.p(i) + 1);
        if (i > 0)
            cs[i] = cs[i - 1] * mfa.nctrl_pts[i - 1];
    }
    ct.resize(tot_iters, mfa.p.size());

    // compute coordinates of first control point of curve corresponding to this iteration
    // these are relative to start of the box of control points located at co
    for (int i = 0; i < tot_iters; i++)      // 1-d flattening all n-d nested loop computations
    {
        int div = tot_iters;
        int i_temp = i;
        for (int j = mfa.p.size() - 1; j >= 0; j--)
        {
            div      /= (mfa.p(j) + 1);
            ct(i, j) =  i_temp / div;
            i_temp   -= (ct(i, j) * div);
        }
    }
}


#if 1

// TBB version, faster (~3X) than serial
//
// computes approximated points from a given set of domain points and an n-d NURBS volume
// P&T eq. 9.77, p. 424
// assumes all vectors have been correctly resized by the caller
template <typename T>
void
mfa::
Decoder<T>::
Decode(MatrixX<T>& approx)                 // pts in approximated volume (1st dim. changes fastest)
{
    vector<size_t> iter(mfa.p.size(), 0);    // parameter index (iteration count) in current dim.
    vector<size_t> ofst(mfa.p.size(), 0);    // start of current dim in linearized params

    for (size_t i = 0; i < mfa.p.size() - 1; i++)
        ofst[i + 1] = ofst[i] + mfa.ndom_pts(i);

    parallel_for (size_t(0), (size_t)mfa.domain.rows(), [&] (size_t i)
    {
        // convert linear idx to multidim. i,j,k... indices in each domain dimension
        VectorXi ijk(mfa.p.size());
        mfa.idx2ijk(i, ijk);

        // compute parameters for the vertices of the cell
        VectorX<T> param(mfa.p.size());
        for (int i = 0; i < mfa.p.size(); i++)
            param(i) = mfa.params(ijk(i) + mfa.po[i]);

        // compute approximated point for this parameter vector
        VectorX<T> cpt(mfa.ctrl_pts.cols());       // approximated point
        VolPt(param, cpt);

        approx.row(i) = cpt;
    });
    fprintf(stderr, "100 %% decoded\n");
}

#else

// serial version
//
// computes approximated points from a given set of domain points and an n-d NURBS volume
// P&T eq. 9.77, p. 424
// assumes all vectors have been correctly resized by the caller
template <typename T>
void
mfa::
Decoder<T>::
Decode(MatrixX<T>& approx)                 // pts in approximated volume (1st dim. changes fastest)
{
    vector<size_t> iter(mfa.p.size(), 0);    // parameter index (iteration count) in current dim.
    vector<size_t> ofst(mfa.p.size(), 0);    // start of current dim in linearized params

    for (size_t i = 0; i < mfa.p.size() - 1; i++)
        ofst[i + 1] = ofst[i] + mfa.ndom_pts(i);

    // eigen frees following temp vectors when leaving scope
    VectorX<T> dpt(mfa.domain.cols());         // original data point
    VectorX<T> cpt(mfa.ctrl_pts.cols());       // approximated point
    VectorX<T> d(mfa.domain.cols());           // apt - dpt
    VectorX<T> param(mfa.p.size());            // parameters for one point
    for (size_t i = 0; i < mfa.domain.rows(); i++)
    {
        // extract parameter vector for one input point from the linearized vector of all params
        for (size_t j = 0; j < mfa.p.size(); j++)
            param(j) = mfa.params(iter[j] + ofst[j]);

        // compute approximated point for this parameter vector
        VolPt(param, cpt);

        // update the indices in the linearized vector of all params for next input point
        for (size_t j = 0; j < mfa.p.size(); j++)
        {
            if (iter[j] < mfa.ndom_pts(j) - 1)
            {
                iter[j]++;
                break;
            }
            else
                iter[j] = 0;
        }

        approx.row(i) = cpt;

        // print progress
        if (i > 0 && mfa.domain.rows() >= 100 && i % (mfa.domain.rows() / 100) == 0)
            fprintf(stderr, "\r%.0f %% decoded", (T)i / (T)(mfa.domain.rows()) * 100);
    }
    fprintf(stderr, "\r100 %% decoded\n");
}

#endif

// compute a point from a NURBS curve at a given parameter value
// algorithm 4.1, Piegl & Tiller (P&T) p.124
template <typename T>
void
mfa::
Decoder<T>::
CurvePt(
        int         cur_dim,                              // current dimension
        T           param,                                // parameter value of desired point
        size_t      to,                                   // offset to start of control points for this curve
        VectorX<T>& out_pt)                               // (output) point
{
//     int n      = (int)mfa.nctrl_pts(cur_dim) - 1;          // number of control point spans
    int span   = mfa.FindSpan(cur_dim, param, mfa.ko[cur_dim]) - mfa.ko[cur_dim];    // relative to ko[cur_dim]
    out_pt     = VectorX<T>::Zero(mfa.ctrl_pts.cols());
    MatrixX<T> N = MatrixX<T>::Zero(1, mfa.nctrl_pts(cur_dim)); // basis coefficients
    mfa.BasisFuns(cur_dim, param, span, N, 0);

    for (int j = 0; j <= mfa.p(cur_dim); j++)
    {
        out_pt += N(0, j + span - mfa.p(cur_dim)) *
            mfa.ctrl_pts.row(to + (span - mfa.p(cur_dim) + j) * cs[cur_dim]) *
            mfa.weights(to + (span - mfa.p(cur_dim) + j) * cs[cur_dim]);
    }

    // clamp dimensions other than cur_dim to same value as first control point
    // eliminates any wiggles in other dimensions due to numerical precision errors
    for (auto j = 0; j < mfa.p.size(); j++)
        if (j != cur_dim)
            out_pt(j) = mfa.ctrl_pts(to + (span - mfa.p(cur_dim)) * cs[cur_dim], j);

    // compute the denominator of the rational curve point and divide
    // basis function and weights arrays must be same size and shape to be multiplied element-wise
    ArrayXX<T> w(1, mfa.nctrl_pts(cur_dim));              // weights for this curve
    ArrayXX<T> b(1, mfa.nctrl_pts(cur_dim));              // basis functions for this curve
    for (auto j = 0; j < mfa.nctrl_pts(cur_dim); j++)
        w(0, j) = mfa.weights(to + j * cs[cur_dim]);
    b = N.row(0).array();
    T denom = (b * w).sum();                        // sum of element-wise products
    out_pt /= denom;

    // debug
//     fprintf(stderr, "1: denom=%.3f\n", denom);
}

// compute a point from a NURBS curve at a given parameter value
// this version takes a temporary set of control points for one curve only rather than
// reading full n-d set of control points from the mfa
// algorithm 4.1, Piegl & Tiller (P&T) p.124
template <typename T>
void
mfa::
Decoder<T>::
CurvePt(
        int         cur_dim,                      // current dimension
        T           param,                        // parameter value of desired point
        MatrixX<T>& temp_ctrl,                    // temporary control points
        VectorX<T>& temp_weights,                 // weights associated with temporary control points
        VectorX<T>& out_pt,                       // (output) point
        int         ko)                           // starting knot offset (default = 0)
{
//     int n      = (int)temp_ctrl.rows() - 1;     // number of control point spans
    int span   = mfa.FindSpan(cur_dim, param, ko) - ko;         // relative to ko
    MatrixX<T> N = MatrixX<T>::Zero(1, temp_ctrl.rows());      // basis coefficients
    mfa.BasisFuns(cur_dim, param, span, N, 0);
    out_pt = VectorX<T>::Zero(temp_ctrl.cols());  // initializes and resizes

    for (int j = 0; j <= mfa.p(cur_dim); j++)
        out_pt += N(0, j + span - mfa.p(cur_dim)) *
            temp_ctrl.row(span - mfa.p(cur_dim) + j) *
            temp_weights(span - mfa.p(cur_dim) + j);

    // clamp dimensions other than cur_dim to same value as first control point
    // eliminates any wiggles in other dimensions due to numerical precision errors
    for (auto j = 0; j < mfa.p.size(); j++)
        if (j != cur_dim)
            out_pt(j) = temp_ctrl(0, j);

    // compute the denominator of the rational curve point and divide by it
    // sum of element-wise multiplication requires transpose so that both arrays are same shape
    // (rows in this case), otherwise eigen cannot multiply them
    T denom = (N.row(0).cwiseProduct(temp_weights.transpose())).sum();
    out_pt /= denom;
}

// compute a point from a NURBS n-d volume at a given parameter value
// algorithm 4.3, Piegl & Tiller (P&T) p.134
template <typename T>
void
mfa::
Decoder<T>::
VolPt(VectorX<T>& param,                       // parameter value in each dim. of desired point
      VectorX<T>& out_pt)                      // (output) point
{
    // check dimensionality for sanity
    assert(mfa.p.size() < mfa.ctrl_pts.cols());

    out_pt = VectorX<T>::Zero(mfa.ctrl_pts.cols());   // initializes and resizes
    vector <MatrixX<T>> N(mfa.p.size());              // basis functions in each dim.
    vector<VectorX<T>>  temp(mfa.p.size());           // temporary point in each dim.
    vector<int>         span(mfa.p.size());           // span in each dim.
    vector<int>         n(mfa.p.size());              // number of control point spans in each dim
    vector<int>         iter(mfa.p.size());           // iteration number in each dim.
    VectorX<T>          ctrl_pt(mfa.ctrl_pts.cols()); // one control point
    int                 ctrl_idx;                     // control point linear ordering index
    VectorX<T>          temp_denom = VectorX<T>::Zero(mfa.p.size());     // temporary rational NURBS denominator in each dim

    // init
    for (size_t i = 0; i < mfa.p.size(); i++)       // for all dims
    {
        temp[i]    = VectorX<T>::Zero(mfa.ctrl_pts.cols());
        iter[i]    = 0;
//         n[i]       = (int)mfa.nctrl_pts(i) - 1;
        span[i]    = mfa.FindSpan(i, param(i), mfa.ko[i]) - mfa.ko[i];  // relative to ko
        N[i]       = MatrixX<T>::Zero(1, mfa.nctrl_pts(i));
        mfa.BasisFuns(i, param(i), span[i], N[i], 0);
    }

    for (int i = 0; i < tot_iters; i++)             // 1-d flattening all n-d nested loop computations
    {
        // control point linear order index
        ctrl_idx = 0;
        for (int j = 0; j < mfa.p.size(); j++)
            ctrl_idx += (span[j] - mfa.p(j) + ct(i, j)) * cs[j];

        // always compute the point in the first dimension
        ctrl_pt = mfa.ctrl_pts.row(ctrl_idx);
        T w            = mfa.weights(ctrl_idx);

#ifdef WEIGH_ALL_DIMS                               // weigh all dimensions
        temp[0]       += (N[0])(0, iter[0] + span[0] - mfa.p(0)) * ctrl_pt * w;
#else                                               // weigh only range dimension
        int last = mfa.ctrl_pts.cols() - 1;
        for (auto j = 0; j < last; j++)
            (temp[0])(j) += (N[0])(0, iter[0] + span[0] - mfa.p(0)) * ctrl_pt(j);
        (temp[0])(last) += (N[0])(0, iter[0] + span[0] - mfa.p(0)) * ctrl_pt(last) * w;
#endif

        temp_denom(0) += w * N[0](0, iter[0] + span[0] - mfa.p(0));
        iter[0]++;

        // for all dimensions except last, check if span is finished
        for (size_t k = 0; k < mfa.p.size() - 1; k++)
        {
            if (iter[k] - 1 == mfa.p(k))
            {
                // compute point in next higher dimension and reset computation for current dim
                temp[k + 1]        += (N[k + 1])(0, iter[k + 1] + span[k + 1] - mfa.p(k + 1)) * temp[k];
                temp_denom(k + 1)  += temp_denom(k) * N[k + 1](0, iter[k + 1] + span[k + 1] - mfa.p(k + 1));
                temp_denom(k)       = 0.0;
                temp[k]             = VectorX<T>::Zero(mfa.ctrl_pts.cols());
                iter[k]             = 0;
                iter[k + 1]++;
            }
        }
    }

    T denom = temp_denom(mfa.p.size() - 1);

#ifdef WEIGH_ALL_DIMS                           // weigh all dimensions
    out_pt = temp[mfa.p.size() - 1] / denom;
#else                                           // weigh only range dimension
    out_pt   = temp[mfa.p.size() - 1];
    int last = mfa.ctrl_pts.cols() - 1;
    out_pt(last) /= denom;
#endif

}

#include    "decode_templates.cpp"
