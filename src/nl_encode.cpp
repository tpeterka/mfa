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
#include <cppoptlib/solver/bfgssolver.h>
#include <cppoptlib/solver/lbfgsbsolver.h>

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
    // initialize the optimization problem
    MaxDist<float> f(mfa, err_limit);

    // first test: x is a scalar scaling factor on the y-coordinates of the control points

    // initialize starting point
    VectorXf x(1);
    x(0) = 1.0;
//     f.setLowerBound(VectorXf::Ones(1));             // for Lbfgsb solver

    // choose a solver
//     ConjugatedGradientDescentSolver<MaxDist<float>> solver;
    BfgsSolver<MaxDist<float>> solver;
//     LbfgsbSolver<MaxDist<float>> solver;             // bounded

    // and minimize the function
    solver.minimize(f, x);

    // print results
    cerr << "argmin      " << x             << endl;
    cerr << "f(argmin)   " << f(x)          << endl;
    cerr << "num iters   " << f.num_iters() << endl;
}
