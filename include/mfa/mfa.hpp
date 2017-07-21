//--------------------------------------------------------------
// mfa object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_HPP
#define _MFA_HPP

#include <Eigen/Dense>
#include <vector>
#include <tbb/tbb.h>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;
using namespace tbb;

namespace mfa
{
    class MFA
    {
    public:

        // fixed number of control points version
        MFA(VectorXi& p_,             // polynomial degree in each dimension
            VectorXi& ndom_pts_,      // number of input data points in each dim
            VectorXi& nctrl_pts_,     // desired number of control points in each dim
            MatrixXf& domain_,        // input data points (1st dim changes fastest)
            MatrixXf& ctrl_pts_,      // (output) control points (1st dim changes fastest)
            VectorXf& knots_,         // (output) knots (1st dim changes fastest)
            float     eps_ = 1.0e-6); // minimum difference considered significant

        // adaptive number of control points version
        MFA(VectorXi& p_,             // polynomial degree in each dimension
            VectorXi& ndom_pts_,      // number of input data points in each dim
            MatrixXf& domain_,        // input data points (1st dim changes fastest)
            MatrixXf& ctrl_pts_,      // (output) control points (1st dim changes fastest)
            VectorXf& knots_,         // (output) knots (1st dim changes fastest)
            float     eps_ = 1.0e-6); // minimum difference considered significant

        ~MFA() {}

        void Encode();

        void AdaptiveEncode(
                float     err_limit,        // maximum allowable normalized error
                VectorXi& nctrl_pts_);      // (output) number of control points in each dim

        void Decode(MatrixXf& approx);                  // decode points

        float Error(size_t idx);                        // index of domain point where to compute error of mfa

        float NormalDistance(
                VectorXf& pt,              // point whose distance from domain is desired
                size_t    cell_idx);       // index of min. corner of cell in the domain

        float CurveDistance(
                int       k,               // current dimension in direction of curve
                VectorXf& pt,              // point whose distance from domain is desired
                size_t    cell_idx);       // index of min. corner of cell in the domain
    private:

        int FindSpan(int   cur_dim,       // current dimension
                     float u,             // parameter value
                     int   ko    = 0);    // optional index of starting knot

        void BasisFuns(int       cur_dim, // current dimension
                       float     u,       // parameter value
                       int       span,    // index of span in the knots vector containing u
                       MatrixXf& N,       // matrix of (output) basis function values
                       int       start_n, // starting basis function N_{start_n} to compute
                       int       end_n,   // ending basis function N_{end_n} to compute
                       int       row,     // starting row index in N of result
                       int       ko = 0); // optional index of starting knot

        void Params();

        void Knots();

        void InsertKnots(VectorXi& nnew_knots,     // number of new knots in each dim
                         VectorXf& new_knots);     // new knots (1st dim changes fastest)

        float InterpolateParams(int       cur_dim, // curent dimension
                                size_t    po,      // starting offset for params in cur. dim.
                                size_t    ds,      // stride for domain pts in cuve in cur. dim.
                                float     coord);  // target coordinate

        void idx2ijk(size_t     idx,        // linear index
                    VectorXi&   ijk);       // i,j,k,... indices in all dimensions
        void ijk2idx(VectorXi&  ijk,        // i,j,k,... indices to all dimensions
                    size_t&     idx);       // (output) linear index

       friend class Encoder;
       friend class Decoder;

       VectorXi& p;           // polynomial degree in each dimension
       VectorXi& ndom_pts;    // number of input data points in each dim
       VectorXi  nctrl_pts;   // desired number of control points in each dim
       MatrixXf& domain;      // input data points (1st dim changes fastest)
       VectorXf  params;      // parameters for input points (1st dim changes fastest)
       MatrixXf& ctrl_pts;    // (output) control pts (1st dim changes fastest)
       VectorXf& knots;       // (output) knots (1st dim changes fastest)
       float     dom_range;   // max extent of input data points
       vector<size_t> po;     // starting offset for params in each dim
       vector< vector <size_t> > co; // starting offset for curves in each dim
       vector<size_t> ko;     // starting offset for knots in each dim
       vector<size_t> ds;     // stride for domain points in each dim
       size_t tot_nparams;    // total number of params = sum of ndom_pts over all dims
                              // not the total number of data pts, which would be the prod.
       size_t tot_nknots;     // total nmbr of knots = sum of nmbr of knots over all dims
       size_t tot_nctrl;      // total nmbr of control points = product of control points over all dims
       float eps;             // minimum difference considered significant
   };

}

#endif
