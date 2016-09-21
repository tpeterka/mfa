//--------------------------------------------------------------
// nurbs decoding algorithms
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _DECODE_HPP
#define _DECODE_HPP

#include <Eigen/Dense>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;

// DEPRECATE eventually, only for 1d
void MaxErr1d(int       p,                   // polynomial degree
              MatrixXf& domain,              // domain of input data points
              MatrixXf& ctrl_pts,            // control points
              VectorXf& knots,               // knots
              MatrixXf& approx,              // points on approximated curve
                                             // (same number as input points, for rendering only)
              VectorXf& errs,                // error at each input point
              float&    max_err);            // maximum error

void Decode(VectorXi& p,                     // polynomial degree
            VectorXi& ndom_pts,              // number of input data points in each dim
            MatrixXf& domain,                // domain of input data pts (1st dim. changes fastest)
            MatrixXf& ctrl_pts,              // control points (1st dim. changes fastest)
            VectorXi& nctrl_pts,             // number of control points in each dim
            VectorXf& knots,                 // knots (1st dim. changes fastest)
            MatrixXf& approx);               // pts in approx. volume (1st dim. changes fastest)

void DecodeCurve(VectorXi& p,          // polynomial degree in each dimension
                 size_t    cur_dim,    // current dimension
                 MatrixXf& domain,     // input data points (1st dim changes fastest)
                 MatrixXf& ctrl_pts,   // all control points (1st dim changes fastest)
                 VectorXf& knots,      // knots (1st dim changes fastest)
                 VectorXf& params,     // curve parameters for input points (1st dim changes fastest)
                 float     pre_param,  // parameter value in prior dimension of the pts in the curve
                 VectorXi& ndom_pts,   // number of input domain points in each dimension
                 VectorXi& nctrl_pts,  // number of control point spans in each dimension
                 size_t    ko,         // starting offset for knots in current dim
                 size_t    cur_cs,     // stride for control points in current dim
                 size_t    pre_cs,     // stride for control points in prior dim
                 MatrixXf& out_pts);   // output approximated pts for the curve

// DEPRECATE eventually, only for 1d
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
                  float&    max_err);        // (output) max error from any input pt to curve

#endif
