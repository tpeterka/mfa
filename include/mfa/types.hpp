#ifndef _MFA_TYPES
#define _MFA_TYPES

// Basic includes so we don't need to add them elsewhere
#include    <string>
#include    <vector>
#include    <list>
#include    <cstdio>
#include    <iostream>
#include    <fstream>
#include    <iomanip>
#include    <sstream>

// Eigen
#include    <Eigen/Dense>
#include    <Eigen/Sparse>
#include    <Eigen/OrderingMethods>

// fmt
#include    <diy/thirdparty/fmt/format.h>
#include    <diy/thirdparty/fmt/printf.h>

// set input and ouptut precision here, float or double
#if 0
typedef float                          real_t;
#else
typedef double                         real_t;
#endif

using namespace std;

// Eigen typedefs
using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXXf;
using Eigen::ArrayXXd;
using Eigen::ArrayXi;

// NB, storing matrices and arrays in col-major order
template <typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <typename T>
using VectorX  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using ArrayXX  = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <typename T>
using ArrayX   = Eigen::Array<T, Eigen::Dynamic, 1>;
template <typename T>
using SparseMatrixX = Eigen::SparseMatrix<T, Eigen::ColMajor>;  // Many sparse solvers require column-major format (otherwise, deep copies are made)
template <typename T>
using SpMatTriplet = Eigen::Triplet<T>;

#endif  // _MFA_TYPES