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
    template <typename T>                                   // float or double
    class NewKnots
    {
    public:

        NewKnots(MFA<T>& mfa_);
        ~NewKnots() {}
        bool NewKnots_full(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
                T              err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)
        bool NewKnots_curve1(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
                T              err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)
        bool NewKnots_curve(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
                T              err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)
    private:

        bool ErrorSpans(
                VectorXi&      nnew_knots,      // number of new knots in each dim
                vector<T>&     new_knots,       // new knots (1st dim changes fastest)
                T              err_limit,       // max. allowed error
                int            iter);           // iteration number

        void SplitSpan(
                size_t         si,              // id of span to split
                VectorXi&      nnew_knots,      // number of new knots in each dim
                vector<T>&     new_knots,       // new knots (1st dim changes fastest)
                int            iter,            // iteration number
                vector<bool>&  split_spans);    // spans that have already been split in this iteration

        size_t  max_num_curves;                 // max num. curves per dimension to check in curve version

        MFA<T>& mfa;                               // the mfa object
    };
}

#endif
