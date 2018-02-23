
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

// DEPRECATED
#if 0

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

#endif

#include    "decode_templates.cpp"
