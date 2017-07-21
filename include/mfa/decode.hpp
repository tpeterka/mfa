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

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXi VectorXi;

namespace mfa
{
    class Decoder
    {
    public:

        Decoder(MFA& mfa_);
        ~Decoder() {}
        void Decode(MatrixXf& approx);         // (output) points (1st dim changes fastest)

        void VolPt(VectorXf& param,            // parameter value in each dim. of desired point
                   VectorXf& out_pt);          // (output) point

        void CurvePt(
                int       cur_dim,              // current dimension
                float     param,                // parameter value of desired point
                MatrixXf& temp_ctrl,            // temporary control points
                VectorXf& out_pt,               // (output) point
                int       ko = 0);              // starting knot offset

    private:

        int tot_iters;                          // total iterations in flattened decoding of all dimensions
        MatrixXi  ct;                           // coordinates of first control point of curve for given iteration
                                                // of decoding loop, relative to start of box of
                                                // control points

        vector<size_t>  cs;                     // control point stride (only in decoder, not mfa)

        MFA& mfa;                               // the mfa object
    };
}

#endif
