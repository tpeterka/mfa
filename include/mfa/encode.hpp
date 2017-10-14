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
#include <set>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

namespace mfa
{
    template <typename T>                   // float or double
    class Encoder
    {
    public:

        Encoder(MFA<T>& mfa_);
        ~Encoder() {}
        void Encode();
        void AdaptiveEncode(
                T   err_limit,              // maximum allowable normalized error
                int max_rounds = 0);        // optional maximum number of rounds

   private:

        void RHS(
                int         cur_dim,  // current dimension
                MatrixX<T>& N,        // matrix of basis function coefficients
                MatrixX<T>& R,        // (output) residual matrix allocated by caller
                int         ko = 0,   // optional index of starting knot
                int         po = 0,   // optional index of starting parameter
                int         co = 0);  // optional index of starting domain pt in current curve

        void RHS(
                int         cur_dim,  // current dimension
                MatrixX<T>& in_pts,   // input points (not the default domain stored in the mfa)
                MatrixX<T>& N,        // matrix of basis function coefficients
                MatrixX<T>& R,        // (output) residual matrix allocated by caller
                int         ko = 0,   // optional index of starting knot
                int         po = 0,   // optional index of starting parameter
                int         co = 0,   // optional index of starting input pt in current curve
                int         cs = 1);  // optional stride of input pts in current curve

        void Quants(
                VectorXi& n,          // (output) number of control point spans in each dim
                VectorXi& m);         // (output) number of input data point spans in each dim

        void CtrlCurve(
                MatrixX<T>& N,           // basis functions for current dimension
                MatrixX<T>& NtN,         // N^t * N
                MatrixX<T>& R,           // residual matrix for current dimension and curve
                MatrixX<T>& P,           // solved points for current dimension and curve
                VectorXi&   n,           // number of control point spans in each dimension
                size_t      k,           // current dimension
                size_t      co,          // starting ofst for reading domain pts
                size_t      cs,          // stride for reading domain points
                size_t      to,          // starting ofst for writing control pts
                MatrixX<T>& temp_ctrl0,  // first temporary control points buffer
                MatrixX<T>& temp_ctrl1); // second temporary control points buffer

        void CopyCtrl(
                MatrixX<T>& P,          // solved points for current dimension and curve
                VectorXi&   n,          // number of control point spans in each dimension
                int         k,          // current dimension
                size_t      co,         // starting offset for reading domain points
                size_t      cs,         // stride for reading domain points
                size_t      to,         // starting offset for writing control points
                MatrixX<T>& temp_ctrl0, // first temporary control points buffer
                MatrixX<T>& temp_ctrl1); // second temporary control points buffer

        void CopyCtrl(
                MatrixX<T>& P,          // solved points for current dimension and curve
                VectorXi&   n,          // number of control point spans in each dimension
                int         k,          // current dimension
                size_t      co,         // starting offset for reading domain points
                MatrixX<T>& temp_ctrl); // temporary control points buffer

        // various versions of ErrorCurve follow

        // this version only returns number of erroneous input domain points
        int ErrorCurve(
                size_t         k,             // current dimension
                size_t         co,            // starting ofst for reading domain pts
                MatrixX<T>&    ctrl_pts,      // control points
                VectorX<T>&    weights,       // weights associated with control points
                T              err_limit);    // max allowable error

        // in addition to returning number of erroneous input domain points
        // this version inserts erroneous spans into a set
        // allowing the same span to be inserted multiple times w/o duplicates
        int ErrorCurve(
                size_t         k,             // current dimension
                size_t         co,            // starting ofst for reading domain pts
                MatrixX<T>&    ctrl_pts,      // control points
                VectorX<T>&    weights,       // weights associated with control points
                set<int>&      err_spans,     // spans with error greater than err_limit
                T              err_limit);    // max allowable error

        // compute new knots to be inserted into a curve
        void ErrorCurve(
                size_t           k,           // current dimension
                size_t           co,          // starting ofst for reading domain pts
                MatrixX<T>&      ctrl_pts,    // control points
                VectorX<T>&      weights,     // weights associated with control points
                VectorXi&        nnew_knots,  // number of new knots
                vector<T>&       new_knots,   // new knots
                T                err_limit);  // max allowable error

        // in addition to returning number of erroneous input domain points
        // this version inserts erroneous spans into a set
        // allowing the same span to be inserted multiple times w/o duplicates
        // control points are taken from mfa
        int ErrorCurve(
                size_t       k,             // current dimension
                size_t       co,            // starting ofst for reading domain pts
                size_t       to,            // starting ofst for reading control pts
                set<int>&    err_spans,     // spans with error greater than err_limit
                T            err_limit);    // max allowable error

        // error of points decoded from a curve aligned with a curve of control points
        int ErrorCtrlCurve(
                size_t       k,             // current dimension
                size_t       to,            // starting ofst for reading control pts
                set<int>&    err_spans,     // spans with error greater than err_limit
                T            err_limit);    // max allowable error

        template <typename>
        friend class NewKnots;

        MFA<T>& mfa;                           // the mfa object
    };
}

#endif
