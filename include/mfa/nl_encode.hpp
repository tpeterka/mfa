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
            public:

                MaxDist(MFA& mfa_,
                        float err_limit_) :
                    mfa(mfa_),
                    err_limit(err_limit_),
                    niter(0)
                {
                }

                using typename Problem<T>::TVector;

                // objective function
                T value(const TVector &x)
                {
                    // encode the MFA
                    Encoder encoder(mfa);
                    encoder.Encode();


                    // compute max error
                    float max_err = 0.0;
                    for (size_t i = 0; i < (size_t)mfa.domain.rows(); i++)
                    {
                        float err = mfa.Error(i);
                        if (i == 0 || err > max_err)
                            max_err = err;
                    }

                    niter++;

                    // my test objective function
                    return fabs(x[0]);
                }

                size_t num_iters() { return niter; }

            private:

                MFA&   mfa;                 // the mfa object
                size_t niter;               // number of iterations performed
                float  err_limit;           // user error limit
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

