//--------------------------------------------------------------
// mfa object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_HPP
#define _MFA_HPP

// comment out the following line for domain parameterization
// domain parameterization is the default method if no method is specified
// #define CURVE_PARAMS

// comment out the following line for low-d knot insertion
// low-d is the default if no method is specified
// #define HIGH_D

// comment out the following line for applying weights to only the range dimension
// weighing the range coordinate only is the default if no method is specified
// #define WEIGH_ALL_DIMS

#include <Eigen/Dense>
#include <vector>
#include <list>
#include <tbb/tbb.h>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::ArrayXXf ArrayXXf;
typedef Eigen::ArrayXXd ArrayXXd;

template <typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using VectorX  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using ArrayXX  = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using ArrayX   = Eigen::Array<T, Eigen::Dynamic, 1>;

using namespace std;
using namespace tbb;

template <typename T>                       // float or double
struct KnotSpan
{
    VectorX<T> min_knot;                  // parameter vector of min knot in this span
    VectorX<T> max_knot;                  // parameter vector of max knot in this span
    VectorXi   min_knot_ijk;              // i,j,k indices of minimum knot in this span
    VectorXi   max_knot_ijk;              // i,j,k indices of maximum knot in this span
    VectorX<T> min_param;                 // parameter vector of minimum domain point in this span
    VectorX<T> max_param;                 // parameter vector of maximum domain point in this span
    VectorXi   min_param_ijk;             // i,j,k indices of minimum domain point in this span
    VectorXi   max_param_ijk;             // i,j,k indices of maximum domain point in this span
    int        last_split_dim;            // last dimension in which this span was subdivided
    bool       done;                      // whether the span has converged (<= error_limit everywhere)
};

namespace mfa
{
    template <typename T>                   // float or double
    class MFA
    {
    public:

        MFA(
                VectorXi&   p_,             // polynomial degree in each dimension
                VectorXi&   ndom_pts_,      // number of input data points in each dim
                MatrixX<T>& domain_,        // input data points (1st dim changes fastest)
                MatrixX<T>& ctrl_pts_,      // (output, optional input) control points (1st dim changes fastest)
                VectorXi&   nctrl_pts_,     // (output, optional input) number of control points in each dim
                VectorX<T>& weights_,       // (output, optional input) weights associated with control points
                VectorX<T>& knots_,         // (output) knots (1st dim changes fastest)
                T           eps_ = 1.0e-6); // minimum difference considered significant

        ~MFA() {}

        void Encode();

        void FixedEncode(
                VectorXi& nctrl_pts_,       // (output) number of control points in each dim
                bool      weighted = true); // solve for and use weights

        void AdaptiveEncode(
                T         err_limit,        // maximum allowable normalized error
                VectorXi& nctrl_pts_,       // (output) number of control points in each dim
                bool      weighted = true,  // solve for and use weights
                int       max_rounds = 0);  // optional maximum number of rounds

        void NonlinearEncode(
                T         err_limit,        // maximum allowable normalized error
                VectorXi& nctrl_pts_);      // (output) number of control points in each dim

        void Decode(MatrixX<T>& approx);    // decode points

        T Error(size_t idx);                // index of domain point where to compute error of mfa

        T RangeError(size_t idx);           // index of domain point where to compute error of mfa

        T NormalDistance(
                VectorX<T>& pt,             // point whose distance from domain is desired
                size_t      cell_idx);      // index of min. corner of cell in the domain

        T CurveDistance(
                int         k,              // current dimension in direction of curve
                VectorX<T>& pt,             // point whose distance from domain is desired
                size_t      cell_idx);      // index of min. corner of cell in the domain

        void KnotSpanDomains(
                VectorXi& span_mins,        // minimum domain points of all knot spans
                VectorXi& span_maxs);       // maximum domain points of all knot spans

        void Rationalize(
                int         k,              // current dimension
                VectorX<T>& weights,        // weights of control points
                MatrixX<T>& N,              // basis function coefficients
                MatrixX<T>& NtN_rat);       // (output) rationalized Nt * N
    private:

        int FindSpan(
                int   cur_dim,              // current dimension
                T     u,                    // parameter value
                int   ko    = 0);           // optional index of starting knot

        void BasisFuns(
                int         cur_dim,        // current dimension
                T           u,              // parameter value
                int         span,           // index of span in the knots vector containing u, relative to ko
                MatrixX<T>& N,              // matrix of (output) basis function values
                int         start_n,        // starting basis function N_{start_n} to compute
                int         end_n,          // ending basis function N_{end_n} to compute
                int         row);           // starting row index in N of result

        void Params();
        void DomainParams();

        void Knots();
        void UniformKnots();

        void KnotSpanIndex();

        void InsertKnots(
                VectorXi&  nnew_knots,      // number of new knots in each dim
                vector<T>& new_knots);      // new knots (1st dim changes fastest)

        T InterpolateParams(
                int       cur_dim,          // curent dimension
                size_t    po,               // starting offset for params in cur. dim.
                size_t    ds,               // stride for domain pts in cuve in cur. dim.
                T         coord);           // target coordinate

        void idx2ijk(
                size_t     idx,             // linear index
                VectorXi&  ijk);            // i,j,k,... indices in all dimensions
        void ijk2idx(
                VectorXi&  ijk,             // i,j,k,... indices to all dimensions
                size_t&    idx);            // (output) linear index

        template <typename>
            friend class Encoder;
        template <typename>
            friend class NL_Encoder;
        template <typename>
            friend class Decoder;
        template <typename>
            friend class NewKnots;
       // TODO: I don't understand why MaxDist can't be a friend of MFA
//        friend class MaxDist;

       // TODO: these members are public until I figure out why MaxDist can't be a friend of MFA
    public:

       VectorXi&                 p;             // polynomial degree in each dimension
       VectorXi&                 ndom_pts;      // number of input data points in each dim
       VectorXi                  nctrl_pts;     // number of control points in each dim
       MatrixX<T>&               domain;        // input data points (row-major order: 1st dim changes fastest)
       VectorX<T>                params;        // parameters for input points (single coords: 1st dim params, 2nd dim, ...)
       MatrixX<T>&               ctrl_pts;      // (output) control pts (row-major order: 1st dim changes fastest)
       VectorX<T>&               weights;       // (output) weights associated with control points
       VectorX<T>&               knots;         // (output) knots (single coords: 1st dim knots, 2nd dim, ...)
       T                         range_extent;  // extent of range value of input data points
       vector<size_t>            po;            // starting offset for params in each dim
       vector< vector <size_t> > co;            // starting offset for curves in each dim
       vector<size_t>            ko;            // starting offset for knots in each dim
       vector<size_t>            ds;            // stride for domain points in each dim
       size_t                    tot_nparams;   // total number of params = sum of ndom_pts over all dims
                                                // not the total number of data pts, which would be the prod.
       size_t                    tot_nknots;    // total nmbr of knots = sum of nmbr of knots over all dims
       size_t                    tot_nctrl;     // total nmbr of control points = product of control points over all dims
       T                         eps;           // minimum difference considered significant
       T                         max_err;       // unnormalized absolute value of maximum error
       vector<KnotSpan <T> >     knot_spans;    // knot spans
    };

}

#endif
