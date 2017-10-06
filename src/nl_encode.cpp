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

template <typename T>
mfa::
NL_Encoder<T>::
NL_Encoder(MFA<T>& mfa_) :
    mfa(mfa_)
{
}

template <typename T>
void
mfa::
NL_Encoder<T>::
Encode(float err_limit)                             // maximum allowable normalized error
{
    // initialize the optimization problem
    MaxDist<float> f(mfa, err_limit);

    // x is a vector of scalar scaling factors on the y-coordinates of the control points

    // initialize starting point
    VectorXf x = VectorXf::Ones(mfa.tot_nctrl);
    x *= 1.5;
//     f.setLowerBound(VectorXf::Ones(1));             // for Lbfgsb solver

    // choose a solver
//     ConjugatedGradientDescentSolver<MaxDist<float>> solver;
    BfgsSolver<MaxDist<float>> solver;
//     LbfgsbSolver<MaxDist<float>> solver;             // bounded

    // minimize the function
    solver.minimize(f, x);

    mfa.ctrl_pts = f.ctrl_pts();

    // print results
    cerr << "argmin      " << x.transpose() << endl;
    cerr << "f(argmin)   " << f(x)          << endl;
    cerr << "num iters   " << f.num_iters() << endl;
}

#include    "nl_encode_templates.cpp"
