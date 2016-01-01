//--------------------------------------------------------------
// nurbs encoding algorithms
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _ENCODE_HPP
#define _ENCODE_HPP

#include <Eigen/Dense>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;

void Approx1d(int       p,                   // polynomial degree
              int       nctrl_pts,           // desired number of control points
              MatrixXf& domain,              // domain of input data points
              MatrixXf& ctrl_pts,            // (output) control points
              VectorXf& knots);              // (output) knots

int FindSpan(int       p,                    // polynomial degree
             int       n,                    // number of control point spans
             VectorXf& knots,                // knots
             float     u);                   // parameter value

void BasisFuns(int       p,                  // polynomial degree
               VectorXf& knots,              // knots
               float     u,                  // parameter value
               int       span,               // index of span in the knots vector containing u
               MatrixXf& N,                  // matrix of (output) basis function values
               int       start_n,            // starting basis function N_{start_n} to compute
               int       end_n,              // ending basis function N_{end_n} to compute
               int       row);               // starting row index in N of result

void Params1d(MatrixXf&  domain,             // domain of input data points
              VectorXf&  params);            // (output) curve parameters
#endif
