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
