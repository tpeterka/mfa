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

//         bool ErrorSpans(
//                 VectorXi& nnew_knots,           // number of new knots in each dim
//                 VectorXf& new_knots,            // new knots (1st dim changes fastest)
//                 float err_limit);               // max. allowed error

        void CurvePt(
                int       cur_dim,              // current dimension
                float     param,                // parameter value of desired point
                MatrixXf& temp_ctrl,            // temporary control points
                VectorXf& out_pt,               // (output) point
                int       ko = 0);              // starting knot offset

    private:

        void CurvePt(
                int       cur_dim,              // current dimension
                float     param,                // parameter value of desired point
                VectorXf& out_pt,               // (output) point
                int       ko = 0);              // starting knot offset

        // DEPRECATED
//         void DecodeCurve(size_t    cur_dim,    // current dimension
//                          float     pre_param,  // param value in prior dim of the pts in the curve
//                          size_t    ko,         // starting offset for knots in current dim
//                          size_t    cur_cs,     // stride for control points in current dim
//                          size_t    pre_cs,     // stride for control points in prior dim
//                          MatrixXf& out_pts);   // output approximated pts for the curve

//         void SplitSpan(
//                 size_t        si,               // id of span to split
//                 vector<bool>& split_spans,      // spans that were split already in this round, don't split these again
//                 VectorXi&     nnew_knots,       // number of new knots in each dim
//                 VectorXf&     new_knots);       // new knots (1st dim changes fastest)

        int tot_iters;                          // total iterations in flattened decoding of all dimensions
        MatrixXi  ct;                           // coordinates of first control point of curve for given iteration
                                                // of decoding loop, relative to start of box of
                                                // control points

        MFA& mfa;                      // the mfa object
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
        vector<size_t>  cs;            // control point stride (only in decoder, not mfa)
//         vector<KnotSpan>& knot_spans;  // not done (greater than max error) knot spans
//         size_t ndone_knot_spans;       // number of done knot spans
    };
}

#endif
