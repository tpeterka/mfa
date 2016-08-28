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

// compute a point from a NURBS n-d volume at a given parameter value
// algorithm 4.3, Piegl & Tiller (P&T) p.134
// this version recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
//
// There are two types of dimensionality:
// 1. The dimensionality of the NURBS tensor product (p.size())
// (1D = NURBS curve, 2D = surface, 3D = volumem 4D = hypervolume, etc.)
// 2. The dimensionality of individual control points (ctrl_pts.cols())
// p.size() should be <= ctrl_pts.cols()
void VolPt(VectorXi& p,                  // polynomial degree in each dimension
           MatrixXf& ctrl_pts,           // control points (1st dim changes fastest)
           VectorXi& nctrl_pts,          // number of control points in each dim
           VectorXf& knots,              // knots (1st dim changes fastest)
           VectorXf& param,              // parameter value of desired point in each dim.
           VectorXf& out_pt)             // (output) point
{
    // debug
    // cerr << "param:\n" << param << endl;

    // check dimensionality for sanity
    assert(p.size() <= ctrl_pts.cols());

    out_pt = VectorXf::Zero(ctrl_pts.cols());// initializes and resizes
    vector <MatrixXf> N(p.size());           // basis functions in each dim.
    vector<VectorXf>  temp(p.size());        // temporary point in each dim.
    vector<int>       span(p.size());        // span in each dim.
    vector<int>       n(p.size());           // number of control point spans in each dim
    vector<int>       iter(p.size());        // iteration number in each dim.
    int               tot_iters = 1;         // tot. num. iterations in flattened n-d nested loops
    vector<size_t>    ko(p.size(), 0);       // starting offset for knots in current dim
    vector<size_t>    co(p.size(), 0);       // starting offset for control points in current dim
    vector<size_t>    cs(p.size(), 1);       // stride for next co in each dim
    VectorXf          ctrl_pt(ctrl_pts.cols()); // one control point

    for (size_t i = 0; i < p.size(); i++)    // for all dims
    {
        temp[i]    = VectorXf::Zero(ctrl_pts.cols());
        iter[i]    = 0;
        tot_iters  *= (p(i) + 1);
        n[i]       = (int)nctrl_pts(i) - 1;
        cs[i]      *= nctrl_pts(i);
        span[i]    = FindSpan(p(i), n[i], knots, param(i), ko[i]);
        N[i]       = MatrixXf::Zero(1, n[i] + 1);
        BasisFuns(p(i), knots, param(i), span[i], N[i], 0, n[i], 0, ko[i]);
        if (i < p.size() - 1)
        {
            ko[i + 1] = ko[i] + n[i] + p(i) + 2; // n[i]+p(i)+2 =  number of knots in current dim.
            co[i + 1] = cs[i];
        }
    }

    for (int i = 0; i < tot_iters; i++)      // 1-d flattening all n-d nested loop computations
    {
        // always compute the point in the first dimension
        ctrl_pt = ctrl_pts.row(co[0] + span[0] - p(0) + iter[0]);
        temp[0] += (N[0])(0, iter[0] + span[0] - p(0)) * ctrl_pt;
        iter[0]++;

        // for all dimensions except last, check if span is finished
        for (size_t k = 0; k < p.size() - 1; k++)
        {
            if (iter[k] - 1 == p(k))
            {
                // compute point in next higher dimension
                temp[k + 1] += N[k + 1](0, iter[k + 1] + span[k + 1] - ko[k + 1] - p(k + 1)) *
                    temp[k];

                // reset the computation for the current dimension
                temp[k]    = VectorXf::Zero(ctrl_pts.cols());
                iter[k]    = 0;
                co[k]      += cs[k];
                iter[k + 1]++;
            }
        }
    }

    out_pt = temp[p.size() - 1];

    // debug
    // cerr << "out_pt:\n" << out_pt << endl;
}

// DEPRECATED
// but keeping around until a better n-d method is devised
//
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
    VectorXi ndom_pts(1);                    // number of domain points as a vector of one component
    ndom_pts(0) = domain.rows();
    Params(ndom_pts, domain, params);

    // errors and max error
    max_err = 0;

    // eigen frees following temp vectors when leaving scope
    VectorXf dpt(domain.cols());             // original data point
    VectorXf cpt(ctrl_pts.cols());           // approximated curve point
    VectorXf d(domain.cols());               // apt - dpt
    VectorXi p_vec(1);                       // p as a vector of one component
    p_vec(0) = p;
    VectorXi nctrl_pts(1);                   // number of control pts as a vector of one component
    nctrl_pts(0) = ctrl_pts.rows();
    VectorXf param_vec(1);                   // param value as a vector of one component
    for (size_t i = 0; i < domain.rows(); i++)
    {
        // TODO: eliminate the folowing copy from cpt to approx.row(i)
        // not straightforward to pass a row to a function expecting a vector
        // because matrix ordering is column order by default
        // not sure what is the best combo of usability and performance
        param_vec(0) = params(i);
        VolPt(p_vec, ctrl_pts, nctrl_pts, knots, param_vec, cpt);
        approx.row(i) = cpt;
        dpt = domain.row(i);
        d = cpt - dpt;
        errs(i) = d.norm();                  // Euclidean distance
        if (i == 0 || errs(i) > max_err)
            max_err = errs(i);
   }
}

// computes approximated points from a given set of domain points and an n-d NURBS volume
// P&T eq. 9.77, p. 424
// this version recomputes parameter values of input points and
// recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
// assumes all vectors have been correctly resized by the caller
void Decode(VectorXi& p,                   // polynomial degree
            VectorXi& ndom_pts,            // number of input data points in each dim
            MatrixXf& domain,              // domain of input data points (1st dim. changes fastest)
            MatrixXf& ctrl_pts,            // control points (1st dim. changes fastest)
            VectorXi& nctrl_pts,           // number of control points in each dim
            VectorXf& knots,               // knots (1st dim. changes fastest)
            MatrixXf& approx)              // pts in approximated volume (1st dim. changes fastest)
{
    // curve parameters for input points
    // linearized so that 1st dim changes fastest
    // i.e., x params followed by y params followed by z, ...
    // total number of params is the sum of the dimensions of domain points, not the product
    VectorXf params(ndom_pts.sum());          // curve parameters for input data points
    Params(ndom_pts, domain, params);

    vector<size_t> iter(p.size(), 0);        // parameter index (iteration count) in current dim.
    vector<size_t> ofst(p.size(), 0);        // start of current dim in linearized params

    for (size_t i = 0; i < p.size() - 1; i++)
        ofst[i + 1] = ofst[i] + ndom_pts(i);

    // eigen frees following temp vectors when leaving scope
    VectorXf dpt(domain.cols());             // original data point
    VectorXf cpt(ctrl_pts.cols());           // approximated point
    VectorXf d(domain.cols());               // apt - dpt
    VectorXf param(p.size());                // parameters for one point
    for (size_t i = 0; i < domain.rows(); i++)
    {
        // debug
        // cerr << "input point:\n" << domain.row(i) << endl;

        // extract parameter vector for one input point from the linearized vector of all params
        for (size_t j = 0; j < p.size(); j++)
            param(j) = params(iter[j] + ofst[j]);

        // compute approximated point for this parameter vector
        VolPt(p, ctrl_pts, nctrl_pts, knots, param, cpt);

        // update the indices in the linearized vector of all params for next input point
        for (size_t j = 0; j < p.size(); j++)
        {
            if (iter[j] < ndom_pts(j) - 1)
            {
                iter[j]++;
                break;
            }
            else
                iter[j] = 0;
        }

        approx.row(i) = cpt;

        // print progress
        if (i > 0 && i % (domain.rows() / 100) == 0)
            fprintf(stderr, "\r%.0f %% decoded", (float)i / (float)(domain.rows()) * 100);
    }
    fprintf(stderr, "\r100 %% decoded\n");
}

// DEPRECATED
// TODO: keeping around until a better n-d method is devised
//
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
    VectorXi ndom_pts(1);
    ndom_pts(0) = domain.rows();
    Params(ndom_pts, domain, params);

    // eigen frees following temp vectors when leaving scope
    VectorXf dpt(domain.cols());             // original data point
    VectorXf cpt(domain.cols());             // point on curve at parameter of input point
    VectorXf d(domain.cols());               // cpt - dpt
    VectorXi p_vec(1);                       // p as a vector of one component
    p_vec(0) = p;
    VectorXi nctrl_pts(1);                   // number of control pts as a vector of one component
    nctrl_pts(0) = ctrl_pts.rows();
    VectorXf param_vec(1);                   // param value as a vector of one component

    // fit approximated curve (for debugging and rendering only)
    for (size_t i = 0; i < domain.rows(); i++)
    {
        // TODO: eliminate the following copy from cpt to approx.row(i)
        // not straightforward to pass a row to a function expecting a vector
        // because matrix ordering is column order by default
        // not sure what is the best combo of usability and performance
        param_vec(0) = params(i);
        VolPt(p_vec, ctrl_pts, nctrl_pts, knots, param_vec, cpt);
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
            if (i + search_rad < params.size())
                uh = params(i + search_rad);
            else
                uh = params(params.size() - 1);
        }
        else if (i > domain.rows() - 1 - search_rad)
        {
            uh = params(domain.rows() - 1);
            ul = params(i - search_rad);
        }
        else
        {
            ul = params(i - search_rad);
            if (i + search_rad < params.size())
                uh = params(i + search_rad);
            else
                uh = params(params.size() - 1);
        }
        um = (ul + uh) / 2.0;

        // debug
        // cerr << "i " << i << " ul " << ul << " um " << um << " uh " << uh << endl;

        int j;
        float el, eh, em;                    // low, high, mid errs; dists to C(ul), C(uh), C(um)
        for (j = 0; j < max_niter; j++)
        {
            param_vec(0) = ul;
            VolPt(p_vec, ctrl_pts, nctrl_pts, knots, param_vec, cpt);
            dpt = domain.row(i);             // original data point
            d = cpt - dpt;                   // eigen frees VectorX when leaving scope
            el = d.norm();                   // Euclidean distance to C(ul)
            // cerr << "low " << cpt << " el " << el;         // debug

            param_vec(0) = um;
            VolPt(p_vec, ctrl_pts, nctrl_pts, knots, param_vec, cpt);
            d = cpt - dpt;
            em = d.norm();                   // Euclidean distance to C(um)
            // cerr << " mid " << cpt << " em " << em;        // debug

            param_vec(0) = uh;
            VolPt(p_vec, ctrl_pts, nctrl_pts, knots, param_vec, cpt);
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
