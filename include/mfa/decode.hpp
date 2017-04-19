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

        void ErrorSpans(float err_limit);      // max. allowed error

    private:

        void CurvePt(int       cur_dim,        // current dimension
                     float     param,          // parameter value of desired point
                     VectorXf& out_pt);        // (output) point

        void DecodeCurve(size_t    cur_dim,    // current dimension
                         float     pre_param,  // param value in prior dim of the pts in the curve
                         size_t    ko,         // starting offset for knots in current dim
                         size_t    cur_cs,     // stride for control points in current dim
                         size_t    pre_cs,     // stride for control points in prior dim
                         MatrixXf& out_pts);   // output approximated pts for the curve

        void SplitSpan(size_t      si,         // id of span to split
        vector<bool>& split_spans);            // spans that were split already in this round, don't split these again

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
        vector<size_t>& co;            // starting offset for control points in each dim
        vector<KnotSpan>& knot_spans;  // not done (greater than max error) knot spans
        size_t ndone_knot_spans;       // number of done knot spans
    };
}

#endif
