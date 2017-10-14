//--------------------------------------------------------------
// decoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _DECODE_HPP
#define _DECODE_HPP

#include <mfa/mfa.hpp>

#include <Eigen/Dense>

typedef Eigen::MatrixXi MatrixXi;

namespace mfa
{
    template <typename T>                       // float or double
    class Decoder
    {
    public:

        Decoder(MFA<T>& mfa_);
        ~Decoder() {}
        void Decode(MatrixX<T>& approx);         // (output) points (1st dim changes fastest)

        void VolPt(VectorX<T>& param,            // parameter value in each dim. of desired point
                   VectorX<T>& out_pt);          // (output) point

        void CurvePt(
                int         cur_dim,              // current dimension
                T           param,                // parameter value of desired point
                size_t      co,                   // offset to start of control points for this curve
                VectorX<T>& out_pt);              // (output) point

        // DEPRECATED
#if 0
        void CurvePt(
                int         cur_dim,              // current dimension
                T           param,                // parameter value of desired point
                MatrixX<T>& temp_ctrl,            // temporary control points
                VectorX<T>& out_pt,               // (output) point
                int         ko = 0);              // starting knot offset
#endif

        void CurvePt(
                int         cur_dim,              // current dimension
                T           param,                // parameter value of desired point
                MatrixX<T>& temp_ctrl,            // temporary control points
                VectorX<T>& temp_weights,         // weights associate with temporary control points
                VectorX<T>& out_pt,               // (output) point
                int         ko = 0);              // starting knot offset

    private:

        int tot_iters;                          // total iterations in flattened decoding of all dimensions

        MatrixXi  ct;                           // coordinates of first control point of curve for given iteration
                                                // of decoding loop, relative to start of box of
                                                // control points

        vector<size_t>  cs;                     // control point stride (only in decoder, not mfa)

        MFA<T>& mfa;                               // the mfa object
    };
}

#endif
