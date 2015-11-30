//--------------------------------------------------------------
// nurbs encoding algorithms
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _ENCODE_HPP
#define _ENCODE_HPP

#include <mfa/types.hpp>
#include <Eigen/Dense>
#include <vector>

using namespace std;

void Approx1d(int                   p,         // polynomial degree
              int                   nctrl_pts, // desired number of control points
              int                   dim,       // point dimensionality
              vector<Pt <float> >&  domain,    // domain of input data points
              vector<Pt <float> >&  ctrl_pts,  // (output) control points
              vector<float>& knots);           // (output) knots

int FindSpan(int            p,                 // polynomial degree
             int            n,                 // number of control point spans
             vector<float>& knots,             // knots
             float          u);                // parameter value

void BasisFuns(int              p,             // polynomial degree
               vector<float>&   knots,         // knots
               float            u,             // parameter value
               int              span,          // index of span in the knots vector containing u
               Eigen::MatrixXf& N,             // matrix of (output) basis function values
               int              start_n,       // starting basis function N_{start_n} to compute
               int              end_n,         // ending basis function N_{end_n} to compute
               int              row);          // starting row index in N of result
void Params1d(vector<Pt <float> >&  domain,    // domain of input data points
              vector<float>&        params);   // (output) curve parameters
#endif
