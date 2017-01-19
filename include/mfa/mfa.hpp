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

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

namespace mfa
{
    class MFA
    {
    public:

        MFA(VectorXi& p_,             // polynomial degree in each dimension
            VectorXi& ndom_pts_,      // number of input data points in each dim
            VectorXi& nctrl_pts_,     // desired number of control points in each dim
            MatrixXf& domain_,        // input data points (1st dim changes fastest)
            MatrixXf& ctrl_pts_,      // (output) control points (1st dim changes fastest)
            VectorXf& knots_,         // (output) knots (1st dim changes fastest)
            float     eps_ = 1.0e-6); // minimum difference considered significant

        ~MFA() {}

        void Encode();

        void Encode(float err_limit);  // maximjum allowable normalized error

        void Decode(MatrixXf& approx); // decoded points

        float Error(VectorXf& pt,    // point some distance away from domain points
                    int       idx);  // index of point in domain near to the point
                                     // search for cell containing the point starts at this index

        // TODO: remove approx parameter below (decode points one at a time)
        void FindExtraKnots(VectorXi& nnew_knots,     // number of new knots in each dim
                            VectorXf& new_knots,      // new knots (1st dim changes fastest)
                            float     err_limit,
                            MatrixXf& approx);        // pts in approximated volume (1st dim. changes fastest)

    private:

        int FindSpan(int   cur_dim,       // current dimension
                     float u,             // parameter value
                     int   ko = 0);       // optional index of starting knot

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

        void idx2ijk(int       idx,       // linear index
                     VectorXi& ijk);      // i,j,k,... indices in all dimensions

        friend class Encoder;
        friend class Decoder;

        VectorXi& p;           // polynomial degree in each dimension
        VectorXi& ndom_pts;    // number of input data points in each dim
        VectorXi& nctrl_pts;   // desired number of control points in each dim
        MatrixXf& domain;      // input data points (1st dim changes fastest)
        VectorXf  params;      // parameters for input points (1st dim changes fastest)
        MatrixXf& ctrl_pts;    // (output) control pts (1st dim changes fastest)
        VectorXf& knots;       // (output) knots (1st dim changes fastest)
        vector<size_t> po;     // starting offset for params in each dim
        vector<size_t> ko;     // starting offset for knots in each dim
        vector<size_t> co;     // starting offset for control points in each dim
        vector<size_t> cs;     // stride for control points in each dim
        vector<size_t> ds;     // stride for domain points in each dim
        int tot_nparams;       // total number of params = sum of ndom_pts over all dims
                               // not the total number of data pts, which would be the prod.
        int tot_nknots;        // total nmbr of knots = sum of nmbr of knots over all dims
        float eps;             // minimum difference considered significant
    };
}

#endif
