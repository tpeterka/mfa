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

// DEPRECATED: moved to mfa.hpp
//         float NormalDistance(VectorXf& pt,        // point whose distance from domain is desired
//                              size_t    cell_idx); // index of min. corner of cell in the domain
                                                  // used to compute partial derivatives
        // DEPRECATED: moved to mfa.hpp
//         float Error(size_t    idx);             // domain point where to compute approximation error

   private:

        void RHS(int       cur_dim,  // current dimension
                 MatrixXf& N,        // matrix of basis function coefficients
                 MatrixXf& R,        // (output) residual matrix allocated by caller
                 int       ko = 0,   // optional index of starting knot
                 int       po = 0,   // optional index of starting parameter
                 int       co = 0,   // optional index of starting domain pt in current curve
                 int       cs = 1);  // optional stride of domain pts in current curve

        void RHS(int       cur_dim,  // current dimension
                 MatrixXf& in_pts,   // input points (not the default domain stored in the mfa)
                 MatrixXf& N,        // matrix of basis function coefficients
                 MatrixXf& R,        // (output) residual matrix allocated by caller
                 int       ko = 0,   // optional index of starting knot
                 int       po = 0,   // optional index of starting parameter
                 int       co = 0,   // optional index of starting input pt in current curve
                 int       cs = 1);  // optional stride of input pts in current curve

        void Quants(VectorXi& n,          // (output) number of control point spans in each dim
                    VectorXi& m,          // (output) number of input data point spans in each dim
                    int&      tot_nparams,// (output) total number params in all dims
                    int&      tot_nknots, // (output) total number of knots in all dims
                    int&      tot_nctrl); // (output) total number of control points in all dims

        void CtrlCurve(MatrixXf& N,           // basis functions for current dimension
                       MatrixXf& NtN,         // N^t * N
                       MatrixXf& R,           // residual matrix for current dimension and curve
                       MatrixXf& P,           // solved points for current dimension and curve
                       VectorXi& n,           // number of control point spans in each dimension
                       size_t    k,           // current dimension
                       vector<size_t> pos,    // starting offsets for params in all dims
                       vector<size_t> kos,    // starting offsets for knots in all dims
                       size_t    co,          // starting ofst for reading domain pts
                       size_t    cs,          // stride for reading domain points
                       size_t    to,          // starting ofst for writing control pts
                       MatrixXf& temp_ctrl0,  // first temporary control points buffer
                       MatrixXf& temp_ctrl1); // second temporary control points buffer

        void CopyCtrl(MatrixXf& P,          // solved points for current dimension and curve
                      VectorXi& n,          // number of control point spans in each dimension
                      int       k,          // current dimension
                      size_t    co,         // starting offset for reading domain points
                      size_t    cs,         // stride for reading domain points
                      size_t    to,         // starting offset for writing control points
                      MatrixXf& temp_ctrl0, // first temporary control points buffer
                      MatrixXf& temp_ctrl1); // second temporary control points buffer

        // DEPRECATED
//         void Gradient(size_t    idx,             // index of min. corner of cell in the domain
//                       VectorXf& grad);           // output gradient

        // DEPECATED
//         void ErrorGradient(size_t    idx,        // index of min. corner of cell in the domain
//                            VectorXf& grad);      // output gradient

        // DEPRECATED
//         void GridSearch(size_t  start_idx,       // starting domain point of search
//                         size_t& end_idx);        // ending domain point of search

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
        vector<size_t>& co;            // starting offset for control points in each dim
        vector<KnotSpan>& knot_spans;  // not done (greater than max error) knot spans
        size_t ndone_knot_spans;       // number of done knot spans
    };
}

#endif
