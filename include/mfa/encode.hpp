//--------------------------------------------------------------
// encoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _ENCODE_HPP
#define _ENCODE_HPP

#include <mfa/mfa.hpp>

#include <Eigen/Dense>
#include <vector>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

namespace mfa
{
    class Encoder
    {
    public:

        Encoder(MFA& mfa_);
        ~Encoder() {}
        void Encode();
        void AdaptiveEncode(float err_limit);               // maximum allowable normalized error
        bool FastEncode(
                VectorXi& nnew_knots,                       // number of new knots in each dim
                VectorXf& new_knots,                        // new knots (1st dim changes fastest)
                float     err_limit);                       // max allowable error

   private:

        void RHS(
                int       cur_dim,  // current dimension
                MatrixXf& N,        // matrix of basis function coefficients
                MatrixXf& R,        // (output) residual matrix allocated by caller
                int       ko = 0,   // optional index of starting knot
                int       po = 0,   // optional index of starting parameter
                int       co = 0);  // optional index of starting domain pt in current curve

        void RHS(
                int       cur_dim,  // current dimension
                MatrixXf& in_pts,   // input points (not the default domain stored in the mfa)
                MatrixXf& N,        // matrix of basis function coefficients
                MatrixXf& R,        // (output) residual matrix allocated by caller
                int       ko = 0,   // optional index of starting knot
                int       po = 0,   // optional index of starting parameter
                int       co = 0,   // optional index of starting input pt in current curve
                int       cs = 1);  // optional stride of input pts in current curve

        void Quants(
                VectorXi& n,          // (output) number of control point spans in each dim
                VectorXi& m);         // (output) number of input data point spans in each dim

        void CtrlCurve(
                MatrixXf& N,           // basis functions for current dimension
                MatrixXf& NtN,         // N^t * N
                MatrixXf& R,           // residual matrix for current dimension and curve
                MatrixXf& P,           // solved points for current dimension and curve
                VectorXi& n,           // number of control point spans in each dimension
                size_t    k,           // current dimension
                size_t    co,          // starting ofst for reading domain pts
                size_t    cs,          // stride for reading domain points
                size_t    to,          // starting ofst for writing control pts
                MatrixXf& temp_ctrl0,  // first temporary control points buffer
                MatrixXf& temp_ctrl1); // second temporary control points buffer

        void CopyCtrl(
                MatrixXf& P,          // solved points for current dimension and curve
                VectorXi& n,          // number of control point spans in each dimension
                int       k,          // current dimension
                size_t    co,         // starting offset for reading domain points
                size_t    cs,         // stride for reading domain points
                size_t    to,         // starting offset for writing control points
                MatrixXf& temp_ctrl0, // first temporary control points buffer
                MatrixXf& temp_ctrl1); // second temporary control points buffer

        void CopyCtrl(
                MatrixXf& P,          // solved points for current dimension and curve
                VectorXi& n,          // number of control point spans in each dimension
                int       k,          // current dimension
                size_t    co,         // starting offset for reading domain points
                MatrixXf& temp_ctrl); // temporary control points buffer

        int ErrorCurve(
                size_t       k,             // current dimension
                size_t       co,            // starting ofst for reading domain pts
                MatrixXf&    temp_ctrl,     // first temporary control points buffer
                vector<int>& err_spans,     // spans with error greater than err_limit
                float        err_limit);    // max allowable error

        MFA& mfa;                       // the mfa object
        // following are references the the data in the mfa object
        VectorXi& p;                   // polynomial degree in each dimension
        VectorXi& ndom_pts;            // number of input data points in each dim
        VectorXi& nctrl_pts;           // desired number of control points in each dim
        MatrixXf& domain;              // input data points (1st dim changes fastest)
        VectorXf& params;              // parameters for input points (1st dim changes fastest)
        MatrixXf& ctrl_pts;            // control points (1st dim changes fastest)
        VectorXf& knots;               // knots (1st dim changes fastest)
        float     dom_range;           // max extent of input data points
        vector<size_t>& po;            // starting offset for params in each dim
        vector<size_t>& ko;            // starting offset for knots in each dim
        vector<KnotSpan>& knot_spans;  // not done (greater than max error) knot spans
    };
}

#endif
