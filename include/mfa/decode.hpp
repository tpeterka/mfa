//--------------------------------------------------------------
// nurbs decoding algorithms
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _DECODE_HPP
#define _DECODE_HPP

void MaxErr1d(int            p,              // polynomial degree
              vector<Pt1d>&  domain,         // domain of input data points
              vector<float>& range,          // range of input data points, same length as domain
              vector<Pt2d>&  ctrl_pts,       // control points
              vector<float>& knots,          // knots
              vector<Pt2d>&  approx,         // points on approximated curve
                                             // (same number as input points, for rendering only)
              vector<float>& errs,           // error at each input point
              float&         max_err);       // maximum error

void MaxNormErr1d(int            p,          // polynomial degree
                  vector<Pt1d>&  domain,     // domain of input data points
                  vector<float>& range,      // range of input data points, same length as domain
                  vector<Pt2d>&  ctrl_pts,   // control points
                  vector<float>& knots,      // knots
                  int            max_niter,  // max num iterations to search for nearest curve pt
                  float          err_bound,  // desired error bound (stop searching if less)
                  int            search_rad, // number of parameter steps to search path on either
                                             // side of parameter value of input point
                  vector<Pt2d>&  approx,     // points on approximated curve
                                             // (same number as input points, for rendering only)
                  vector<float>& errs,       // (output) error at each input point
                  float&         max_err);   // (output) maximum error from any input point to curve

#endif
