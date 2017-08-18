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
                size_t    co,                   // offset to start of control points for this curve
                VectorXf& out_pt);               // (output) point

        void CurvePt(
                int       cur_dim,              // current dimension
                float     param,                // parameter value of desired point
                MatrixXf& temp_ctrl,            // temporary control points
                VectorXf& out_pt,               // (output) point
                int       ko = 0);              // starting knot offset

        bool ErrorSpans(
                VectorXi&      nnew_knots,      // number of new knots in each dim
                vector<float>& new_knots,       // new knots (1st dim changes fastest)
                float          err_limit,       // max. allowed error
                int            iter);           // iteration number

    private:

        void SplitSpan(
                size_t         si,              // id of span to split
                VectorXi&      nnew_knots,      // number of new knots in each dim
                vector<float>& new_knots,       // new knots (1st dim changes fastest)
                int            iter,            // iteration number
                vector<bool>&  split_spans);    // spans that have already been split in this iteration

        int tot_iters;                          // total iterations in flattened decoding of all dimensions

        MatrixXi  ct;                           // coordinates of first control point of curve for given iteration
                                                // of decoding loop, relative to start of box of
                                                // control points

        vector<size_t>  cs;                     // control point stride (only in decoder, not mfa)

        MFA& mfa;                               // the mfa object
    };
}

#endif
