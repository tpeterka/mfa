#ifndef _MFA_TYPES
#define _MFA_TYPES

#include    <Eigen/Dense>
#include    <Eigen/Sparse>
#include    <Eigen/OrderingMethods>
#include    <diy/thirdparty/fmt/format.h>

// set input and ouptut precision here, float or double
#if 0
typedef float                          real_t;
#else
typedef double                         real_t;
#endif

using namespace std;

// Eigen typedefs
using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;
using ArrayXXf = Eigen::ArrayXXf;
using ArrayXXd = Eigen::ArrayXXd;
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