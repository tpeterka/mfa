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
    template<typename T>                        // float or double
        class MaxDist : public Problem<T> {
//         class MaxDist : public BoundedProblem<T> {
            public:

                MaxDist(MFA<T>& mfa_,
                        T err_limit_) :
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
                    MatrixX<T> old_ctrl_pts = mfa.ctrl_pts;

                    // x is a vector of scaling factors, one per control point
                    // used to scale the last coordinate of the control point
                    int last = mfa.ctrl_pts.cols() - 1;             // index of last column of control pts
                    for (auto i = 0; i < mfa.ctrl_pts.rows(); i++)
                        mfa.ctrl_pts(i, last) *= x(i);

                    // debug
//                     cerr << "current control points:\n" << mfa.ctrl_pts << endl;

                    // compute max error
                    // TODO: not checking all points (yet), still using old error computation
                    T max_err = 0.0;
                    for (size_t i = 0; i < (size_t)mfa.domain.rows(); i++)
                    {
                        T err = mfa.Error(i);
                        if (i == 0 || err > max_err)
                            max_err = err;
                    }

                    niter++;
                    mfa.max_err = max_err;

                    opt_ctrl_pts = mfa.ctrl_pts;
                    if (max_err / mfa.dom_range < err_limit)
                        done = true;

                    // pop old control points
                    mfa.ctrl_pts = old_ctrl_pts;

                    // debug
                    cerr << "iteration=" << niter << " err=" << max_err / mfa.dom_range << " x=" << x.transpose() << endl;

                    return max_err;
                }

                size_t    num_iters() { return niter; }
                MatrixX<T>& ctrl_pts()  { return opt_ctrl_pts; }

            private:

                MFA<T>&    mfa;                 // the mfa object
                size_t     niter;               // number of iterations performed
                T          err_limit;           // user error limit
                bool       done;                // achieved user error limit; stop computing objective
                MatrixX<T> opt_ctrl_pts;        // control points found by optimization
        };

    template <typename T>                       // float or double
    class NL_Encoder
    {
    public:

        NL_Encoder(MFA<T>& mfa_);
        ~NL_Encoder() {}
        void Encode(T err_limit);               // maximum allowable normalized error

    private:
        MFA<T>& mfa;                           // the mfa object
    };
}

#endif

