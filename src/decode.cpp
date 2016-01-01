//--------------------------------------------------------------
// nurbs decoding algorithms
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

using namespace std;

// compute a point from a 1d NURBS curve at a given parameter value
// algorithm 4.1, Piegl & Tiller (P&T) p.124
// this version recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
void CurvePt1d(int       p,                  // polynomial degree
               MatrixXf& ctrl_pts,           // control points
               VectorXf& knots,              // knots
               float     param,              // parameter value of desired point
               VectorXf& out_pt)             // (output) point
{
    int n      = (int)ctrl_pts.rows() - 1;   // number of control point spans
    int span   = FindSpan(p, n, knots, param);
    MatrixXf N = MatrixXf::Zero(1, n + 1);   // basis coefficients
    BasisFuns(p, knots, param, span, N, 0, n, 0);
    out_pt = VectorXf::Zero(ctrl_pts.cols()); // initializes and resizes
    for (int j = 0; j <= p; j++)
        out_pt += N(0, j + span - p) * ctrl_pts.row(span - p + j);

    // debug
    // cerr << "n " << n << " param " << param << " span " << span << " out_pt " << out_pt << endl;
    // cerr << " N " << N << endl;
}

// max distance from a set of input points to a 1d NURBS curve
// P&T eq. 9.77, p. 424
// this version recomputes parameter values of input points and
// recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
// assumes all vectors have been correctly resized by the caller
void MaxErr1d(int       p,                   // polynomial degree
              MatrixXf& domain,              // domain of input data points
              MatrixXf& ctrl_pts,            // control points
              VectorXf& knots,               // knots
              MatrixXf& approx,              // points on approximated curve
                                             // (same number as input points, for rendering only)
              VectorXf& errs,                // error at each input point
              float&    max_err)             // maximum error
{
    // curve parameters for input points
    VectorXf params(domain.rows());          // curve parameters for input data points
                                             // eigen frees VectorX when leaving scope
    Params1d(domain, params);

    // errors and max error
    max_err = 0;

    // eigen frees following temp vectors when leaving scope
    VectorXf dpt(domain.cols());             // original data point
    VectorXf cpt(ctrl_pts.cols());           // approximated curve point
    VectorXf d(domain.cols());               // apt - dpt
    for (size_t i = 0; i < domain.rows(); i++)
    {
        // TODO: eliminate the folowing copy from cpt to approx.row(i)
        // not straightforward to pass a row to a function expecting a vector
        // because matrix ordering is column order by default
        // not sure whether what is the best combo of usability and performance
        CurvePt1d(p, ctrl_pts, knots, params(i), cpt);
        approx.row(i) = cpt;
        dpt = domain.row(i);
        d = cpt - dpt;
        errs(i) = d.norm();                  // Euclidean distance
        if (i == 0 || errs(i) > max_err)
            max_err = errs(i);
   }
}

// max norm distance from a set of input points to a 1d NURBS curve
// P&T eq. 9.78, p. 424
// usually a smaller and more accurate measure of max error than MaxError
// this version recomputes parameter values of input points and
// recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
// assumes all vectors have been correctly resized by the caller
void MaxNormErr1d(int       p,               // polynomial degree
                  MatrixXf& domain,          // domain of input data points
                  MatrixXf& ctrl_pts,        // control points
                  VectorXf& knots,           // knots
                  int       max_niter,       // max num iterations to search for
                                             // nearest curve pt
                  float     err_bound,       // desired error bound (stop searching if less)
                  int       search_rad,      // number of parameter steps to search path on
                                             // either side of parameter value of input point
                  MatrixXf& approx,          // points on approximated curve (same number as
                                             // input points, for rendering only)
                  VectorXf& errs,            // (output) error at each input point
                  float&    max_err)         // (output) max error from any input pt to curve
{
    // curve parameters for input points
    VectorXf params(domain.rows());          // curve parameters for input data points
    Params1d(domain, params);

    // eigen frees following temp vectors when leaving scope
    VectorXf dpt(domain.cols());             // original data point
    VectorXf cpt(domain.cols());             // point on curve at parameter of input point
    VectorXf d(domain.cols());               // cpt - dpt

    // fit approximated curve (for debugging and rendering only)
    for (size_t i = 0; i < domain.rows(); i++)
    {
        // TODO: eliminate the following copy from cpt to approx.row(i)
        // not straightforward to pass a row to a function expecting a vector
        // because matrix ordering is column order by default
        // not sure whether what is the best combo of usability and performance
        CurvePt1d(p, ctrl_pts, knots, params(i), cpt);
        approx.row(i) = cpt;
    }
    // debug
    // cerr << "approx:\n" << approx << endl;

    // errors and max error
    max_err = 0;

    for (size_t i = 0; i < domain.rows(); i++)
    {
        // find nearest curve point
        float ul, uh, um;                    // low, high, middle  parameter values

        // range of parameter values to search
        if (i < search_rad)
        {
            ul = params(0);
            uh = params(i + search_rad);
        }
        else if (i > domain.rows() - 1 - search_rad)
        {
            uh = params(domain.rows() - 1);
            ul = params(i - search_rad);
        }
        else
        {
            ul = params(i - search_rad);
            uh = params(i + search_rad);
        }
        um = (ul + uh) / 2.0;

        // debug
        // cerr << "i " << i << " ul " << ul << " um " << um << " uh " << uh << endl;

        int j;
        float el, eh, em;                    // low, high, mid errs; dists to C(ul), C(uh), C(um)
        for (j = 0; j < max_niter; j++)
        {
            CurvePt1d(p, ctrl_pts, knots, ul, cpt);
            // eigen frees following temp vectors when leaving scope
            dpt = domain.row(i);             // original data point
            d = cpt - dpt;                   // eigen frees VectorX when leaving scope
            el = d.norm();                   // Euclidean distance to C(ul)
            // cerr << "low " << cpt << " el " << el;         // debug
            CurvePt1d(p, ctrl_pts, knots, um, cpt);
            d = cpt - dpt;
            em = d.norm();                   // Euclidean distance to C(um)
            // cerr << " mid " << cpt << " em " << em;        // debug
            CurvePt1d(p, ctrl_pts, knots, uh, cpt);
            d = cpt - dpt;
            eh = d.norm();                   // Euclidean distance to C(uh)
            // cerr << " hi " << cpt << " eh " << eh << endl; // debug

            if (el < eh && el < em)          // el is the best error
            {                                // shift the search interval to [uh, um]
                if (el <= err_bound)
                {
                    errs(i) = el;
                    // cerr << "el ";        // debug
                    break;
                }
                uh = um;
                um = (ul + uh) / 2.0;
            }

            else if (eh < el && eh < em)     // eh is the best error
            {                                // shift the search interval to [um, uh]
                if (eh <= err_bound)
                {
                    errs(i) = eh;
                    // cerr << "eh ";        // debug
                    break;
                }
                ul = um;
                um = (ul + uh) / 2.0;
            }

            else                             // em is the best error
            {                                // narrower search interval and centered on um
                if (em <= err_bound)
                {
                    errs(i) = em;
                    // cerr << "em ";        // debug
                    break;
                }
                ul = (ul + um) / 2.0;
                uh = (uh + um) / 2.0;
            }
        }
        if (j == max_niter)                  // max number of iterations was reached
        {                                    // pick minimum of el, eh, em
            if (el < eh && el < em)
                errs(i) = el;
            else if (eh < el && eh < em)
                errs(i) = eh;
            else
                errs(i) = em;
            // cerr << "max iter ";          // debug
        }

        // max error
        if (i == 0 || errs(i) > max_err)
            max_err = errs(i);
   }
}
