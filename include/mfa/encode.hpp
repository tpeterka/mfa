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
#include <vector>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

// TODO: eventually Approx will replace Approx1d
void Approx1d(int       p,                   // polynomial degree
              int       nctrl_pts,           // desired number of control points
              MatrixXf& domain,              // domain of input data points
              MatrixXf& ctrl_pts,            // (output) control points
              VectorXf& knots);              // (output) knots

void Approx(VectorXi& p,                     // polynomial degree in each dimension
            VectorXi& ndom_pts,              // number of input data points in each dim
            VectorXi& nctrl_pts,             // desired number of control points in each dim
            MatrixXf& domain,                // input data points (1st dim changes fastest)
            MatrixXf& ctrl_pts,              // (output) control points (1st dim changes fastest)
            VectorXf& knots);                // (output) knots (1st dim changes fastest)

int FindSpan(int       p,                    // polynomial degree
             int       n,                    // number of control point spans
             VectorXf& knots,                // knots
             float     u,                    // parameter value
             int       ko = 0);              // optional index of starting knot (default = 0)

void BasisFuns(int       p,                  // polynomial degree
               VectorXf& knots,              // knots
               float     u,                  // parameter value
               int       span,               // index of span in the knots vector containing u
               MatrixXf& N,                  // matrix of (output) basis function values
               int       start_n,            // starting basis function N_{start_n} to compute
               int       end_n,              // ending basis function N_{end_n} to compute
               int       row,                // starting row index in N of result
               int       ko = 0);            // optional index of starting knot (default = 0)

// TODO: eventually Params will replace Params1d
void Params1d(MatrixXf&  domain,             // domain of input data points
              VectorXf&  params);            // (output) curve parameters

void Params(VectorXi& ndom_pts,   // number of input data points in each dim
            MatrixXf& domain,     // input data points in each dim (1st dim changes fastest)
            VectorXf& params);    // (output) curve paramtrs in each dim (1st dim changes fastest)

#endif
