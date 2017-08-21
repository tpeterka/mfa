//--------------------------------------------------------------
// new knots inserter object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _NEW_KNOTS_HPP
#define _NEW_KNOTS_HPP

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
    class NewKnots
    {
    public:

        NewKnots(MFA& mfa_);
        ~NewKnots() {}
        bool NewKnots_full(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<float>& new_knots,                   // new knots (1st dim changes fastest)
                float          err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)
        bool NewKnots_curve(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<float>& new_knots,                   // new knots (1st dim changes fastest)
                float          err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)
        bool NewKnots_hybrid(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<float>& new_knots,                   // new knots (1st dim changes fastest)
                float          err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)

    private:

        size_t  max_num_curves;             // max num. curves per dimension to check in curve version

        MFA& mfa;                           // the mfa object
    };
}

#endif
