//--------------------------------------------------------------
// nonlinear encoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _NL_ENCODE_HPP
#define _NL_ENCODE_HPP

#include <mfa/mfa.hpp>
// #include <mfa/encode.hpp>

#include <cppoptlib/problem.h>
#include <cppoptlib/boundedproblem.h>
#include <Eigen/Dense>
#include <vector>
#include <set>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;
using namespace cppoptlib;

namespace mfa
{
    template<typename T>
        class MaxDist : public Problem<T> {
//         class MaxDist : public BoundedProblem<T> {
            public:

                MaxDist(MFA& mfa_,
                        float err_limit_) :
                    mfa(mfa_),
                    err_limit(err_limit_),
                    niter(0),
                    done(false)
                {
                }

                using typename Problem<T>::TVector;
//                 using typename BoundedProblem<T>::TVector;

                // objective function
                T value(const TVector &x)
                {
                    // debug
//                     return(0.0);

                    if (done)
                        return(0.0);

                    // push old control points
                    MatrixXf old_ctrl_pts = mfa.ctrl_pts;

                    // TODO: is x scalar, vector, or matrix?
                    // how does the MFA get updated

                    // first test, x is a scalar
                    // update all control point range values by the same scaling factor
                    int last = mfa.ctrl_pts.cols() - 1;             // index of last column of control pts
                    mfa.ctrl_pts.col(last) *= x(0);

                    // compute max error
                    // TODO: not checking all points (yet), still using old error computation
                    float max_err = 0.0;
                    for (size_t i = 0; i < (size_t)mfa.domain.rows(); i++)
                    {
                        float err = mfa.Error(i);
                        if (i == 0 || err > max_err)
                            max_err = err;
                    }

                    niter++;

                    // pop old control points
                    mfa.ctrl_pts = old_ctrl_pts;

                    mfa.max_err = max_err;
                    if (max_err / mfa.dom_range < err_limit)
                        done = true;

                    // debug
                    cerr << "iteration=" << niter << " err=" << max_err / mfa.dom_range << " x=" << x.transpose() << endl;

                    return max_err;
                }

                size_t num_iters() { return niter; }

            private:

                MFA&   mfa;                 // the mfa object
                size_t niter;               // number of iterations performed
                float  err_limit;           // user error limit
                bool   done;                // achieved user error limit; stop computing objective
        };

    class NL_Encoder
    {
    public:

        NL_Encoder(MFA& mfa_);
        ~NL_Encoder() {}
        void Encode(float err_limit);       // maximum allowable normalized error

    private:
        MFA& mfa;                           // the mfa object
    };
}

#endif

