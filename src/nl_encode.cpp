// --------------------------------------------------------------
// nonlinear encoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <mfa/encode.hpp>
#include <mfa/nl_encode.hpp>
#include <mfa/decode.hpp>
#include <iostream>
#include <cppoptlib/solver/conjugatedgradientdescentsolver.h>

mfa::
NL_Encoder::
NL_Encoder(MFA& mfa_) :
    mfa(mfa_)
{
}

void
mfa::
NL_Encoder::
Encode(float err_limit)                             // maximum allowable normalized error
{
//     // initialize the optimization problem
//     typedef MaxDist<float> TMaxDist;
//     TMaxDist f(mfa, err_limit);
// 
//     // choose a starting point
//     VectorXf x(1); x << -1;
// 
//     // choose a solver
//     ConjugatedGradientDescentSolver<TMaxDist> solver;
// 
//     // and minimize the function
//     solver.minimize(f, x);
// 
//     // print results
//     cerr << "argmin      " << x.transpose() << endl;
//     cerr << "f(argmin)   " << f(x)          << endl;
//     cerr << "num iters   " << f.num_iters() << endl;
}
