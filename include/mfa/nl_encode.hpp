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
    class NL_Encoder
    {
    public:

        NL_Encoder(MFA& mfa_);
        ~NL_Encoder() {}
        void Encode();

        MFA& mfa;                           // the mfa object
    };
}

#endif

