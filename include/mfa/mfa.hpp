//--------------------------------------------------------------
// mfa object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_HPP
#define _MFA_HPP

#define MFA_NAW -1          // an invalid weight, indicating skip this control point

// comment out the following line for unclamped knots (single knot at each end of knot vector)
// clamped knots (repeated at ends) is the default method if no method is specified
// #define UNCLAMPED_KNOTS

// comment out the following line for domain parameterization
// domain parameterization is the default method if no method is specified
// #define CURVE_PARAMS

// comment out the following line for low-d knot insertion
// low-d is the default if no method is specified
// #define HIGH_D

// comment out the following line for applying weights to only the range dimension
// weighing the range coordinate only is the default if no method is specified
// #define WEIGH_ALL_DIMS

// comment out the following line for original single tensor product version
// #define MFA_TMESH

// linear least squares local solve
// #define MFA_LINEAR_LOCAL

#include    <Eigen/Dense>
#include    <Eigen/Sparse>
#include    <vector>
#include    <list>
#include    <iostream>

#ifdef MFA_TBB
#define     TBB_SUPPRESS_DEPRECATED_MESSAGES    1
#include    <tbb/tbb.h>
using namespace tbb;
#endif

using namespace std;

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;
using ArrayXXf = Eigen::ArrayXXf;
using ArrayXXd = Eigen::ArrayXXd;
// NB, storing matrices and arrays in row-major order
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

#include    <mfa/util.hpp>
#include    <mfa/param.hpp>
#include    <mfa/pointset.hpp>
#include    <mfa/tmesh.hpp>
#include    <mfa/mfa_data.hpp>
#include    <mfa/decode.hpp>
#include    <mfa/encode.hpp>

// TODO: Move Model's from BlockBase to MFA
//       Want MFA object to manage construction-destruction of MFA_Data
// a solved and stored MFA_data model (geometry or science variable or both)
template <typename T>
struct Model
{
    int                 min_dim;                // starting coordinate of this model in full-dimensional data
    int                 max_dim;                // ending coordinate of this model in full-dimensional data
    mfa::MFA_Data<T>    *mfa_data;              // MFA model data
};

namespace mfa
{
    template <typename T>                           // float or double
    struct MFA
    {
        int                     dom_dim;            // domain dimensionality

        MFA(size_t dom_dim_) :
            dom_dim(dom_dim_)
        { }

        ~MFA()
        { }

        // fixed number of control points encode
        void FixedEncode(
                MFA_Data<T>&        mfa_data,               // mfa data model
                const PointSet<T>&  input,                  // input points
                const VectorXi      nctrl_pts,              // number of control points in each dim
                int                 verbose,                // debug level
                bool                weighted) const         // solve for and use weights (default = true)
        {
            // fixed encode assumes the tmesh has only one tensor product
            TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];

            t.weights = VectorX<T>::Ones(t.nctrl_pts.prod());
            Encoder<T> encoder(*this, mfa_data, input, verbose);

            if (input.structured)
                encoder.Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);
            else
                encoder.EncodeUnified(0, weighted);  // Assumes only one tensor product
            

            // debug: try inserting a knot
            //             VectorX<T> new_knot(mfa->dom_dim);
            //             for (auto i = 0; i < mfa->dom_dim; i++)
            //                 new_knot(i) = 0.5;
            //             mfa->KnotInsertion(new_knot, tmesh().tensor_prods[0]);
        }

        // adaptive encode
        void AdaptiveEncode(
                MFA_Data<T>&        mfa_data,               // mfa data model
                const PointSet<T>& input,                  // input points
                T                   err_limit,              // maximum allowable normalized error
                int                 verbose,                // debug level
                bool                weighted,               // solve for and use weights (default = true)
                bool                local,                  // solve locally (with constraints) each round
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds) const       // optional maximum number of rounds
        {
            Encoder<T> encoder(*this, mfa_data, input, verbose);

#ifndef MFA_TMESH           // original adaptive encode for one tensor product
            encoder.OrigAdaptiveEncode(err_limit, weighted, extents, max_rounds);
#else                       // adaptive encode for tmesh
            encoder.AdaptiveEncode(err_limit, weighted, local, extents, max_rounds);
#endif
        }

        // decode values at all input points
        void DecodePointSet(
                const MFA_Data<T>&  mfa_data,               // mfa data model
                PointSet<T>&        output,                 // (output) decoded point set
                int                 verbose,                // debug level
                int                 min_dim,                // first dimension to decode
                int                 max_dim,                // last dimension to decode
                bool                saved_basis) const      // whether basis functions were saved and can be reused
        {
            VectorXi no_derivs;                             // size-0 means no derivatives

            DecodePointSet(mfa_data, output, verbose, min_dim, max_dim, saved_basis, no_derivs);
        }

        // decode derivatives at all input points
        void DecodePointSet(
                const MFA_Data<T>&  mfa_data,               // mfa data model
                PointSet<T>&        output,                 // (output) decoded point set
                int                 verbose,                // debug level
                int                 min_dim,                // first dimension to decode
                int                 max_dim,                // last dimension to decode
                bool                saved_basis,            // whether basis functions were saved and can be reused
                const VectorXi&     derivs) const           // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                            // pass size-0 vector if unused
        {
            mfa::Decoder<T> decoder(mfa_data, verbose, saved_basis);
            decoder.DecodePointSet(output, min_dim, max_dim, derivs);
        }

        // decode value of single point at the given parameter location
        void DecodePt(
                const MFA_Data<T>&  mfa_data,               // mfa data model
                const VectorX<T>&   param,                  // parameters of point to decode
                VectorX<T>&         cpt) const              // (output) decoded point
        {
            VectorXi no_derivs;
            int verbose = 0;
            Decoder<T> decoder(mfa_data, verbose);
            // TODO: hard-coded for one tensor product
            decoder.VolPt(param, cpt, mfa_data.tmesh.tensor_prods[0], no_derivs);
        }

        // decode derivative of single point at the given parameter location
        void DecodePt(
                const MFA_Data<T>&  mfa_data,               // mfa data model
                const VectorX<T>&   param,                  // parameters of point to decode
                const VectorXi&     derivs,                 // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                            // pass size-0 vector if unused
                VectorX<T>&         cpt) const              // (output) decoded point
        {
            int verbose = 0;
            Decoder<T> decoder(mfa_data, verbose);
            // TODO: hard-coded for one tensor product
            decoder.VolPt(param, cpt, mfa_data.tmesh.tensor_prods[0], derivs);
        }

        // decode points on grid in parameter space
        void DecodeAtGrid(  const MFA_Data<T>&      mfa_data,               // mfa_data
                            int                     min_dim,                // min index to decode
                            int                     max_dim,                // max index to decode
                            const VectorX<T>&       par_min,                // lower corner of domain in param space
                            const VectorX<T>&       par_max,                // upper corner of domain in param space
                            const VectorXi&         ndom_pts,              // number of points per direction
                            MatrixX<T>&             result)                 // decoded result
        {
            int verbose = 0;
            Decoder<T> decoder(mfa_data, verbose);
            decoder.DecodeGrid(result, min_dim, max_dim, par_min, par_max, ndom_pts);
        }

        // compute the error (absolute value of coordinate-wise difference) of the mfa at a domain point
        // error is not normalized by the data range (absolute, not relative error)
        void AbsCoordError(
                const MFA_Data<T>&  mfa_data,               // mfa data model
                const PointSet<T>&  input,
                size_t              idx,                    // index of domain point
                VectorX<T>&         error,                  // (output) absolute value of error at each coordinate
                int                 verbose) const          // debug level
        {
            VectorX<T> param(dom_dim);
            input.pt_params(idx, param);

            // NB, assumes at least one tensor product exists and that all have the same ctrl pt dimensionality
            int pt_dim = mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols();

            // approximated value
            VectorX<T> cpt(pt_dim);          // approximated point
            Decoder<T> decoder(mfa_data, verbose);
            decoder.VolPt(param, cpt, mfa_data.tmesh.tensor_prods[0]);      // TODO: hard-coded for first tensor product

            for (auto i = 0; i < pt_dim; i++)
                error(i) = fabs(cpt(i) - input.domain(idx, mfa_data.min_dim + i));
        }

        void AbsPointSetDiff(
            const   mfa::PointSet<T>& ps1,
            const   mfa::PointSet<T>& ps2,
                    mfa::PointSet<T>& diff,
                    int               verbose)
        {
            if (!ps1.is_same_layout(ps2) || !ps1.is_same_layout(diff))
            {
                cerr << "ERROR: Incompatible PointSets in AbsPointSetDiff" << endl;
                exit(1);
            }

#ifdef MFA_SERIAL
            int pt_dim = ps1.pt_dim;
            diff.domain.leftCols(dom_dim) = ps1.domain.leftCols(dom_dim);
            diff.domain.rightCols(pt_dim-dom_dim) = (ps1.domain.rightCols(pt_dim-dom_dim) - ps2.domain.rightCols(pt_dim-dom_dim)).cwiseAbs();
#endif // MFA_SERIAL
#ifdef MFA_TBB
            parallel_for (size_t(0), (size_t)diff.npts, [&] (size_t i)
                {
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        diff.domain(i,j) = ps1.domain(i,j); // copy the geometric location of each point
                    }
                });

            parallel_for (size_t(0), (size_t)diff.npts, [&] (size_t i)
                {
                    for (auto j = dom_dim; j < diff.pt_dim; j++)
                    {
                        diff.domain(i,j) = fabs(ps1.domain(i,j) - ps2.domain(i,j)); // compute distance between each science value
                    }
                });
#endif // MFA_TBB
        }

        void AbsPointSetError(
            const   mfa::PointSet<T>& base,
                    mfa::PointSet<T>& error,
                    vector<Model<T>>& vars,
                    int               verbose)
        {
            if (!base.is_same_layout(error))
            {
                cerr << "ERROR: Incompatible PointSets in AbsPointSetError" << endl;
                exit(1);
            }

#ifdef MFA_SERIAL
            // copy geometric point coordinates
            for (size_t i = 0; i < error.npts; i++)
            {   
                for (auto j = 0; j < dom_dim; j++)
                {
                    error.domain(i,j) = base.domain(i,j); // copy the geometric location of each point
                } 
            }

            // compute errors for each point and model
            for (size_t i = 0; i < error.npts; i++)
            {
                VectorX<T> err_vec;                          // errors for all coordinates in current model
                for (auto k = 0; k < vars.size(); k++)      // for all science models
                {
                    err_vec.resize(vars[k].max_dim - vars[k].min_dim);
                    AbsCoordError(*(vars[k].mfa_data), base, i, err_vec, verbose);

                    for (auto j = 0; j < err_vec.size(); j++)
                    {
                        error.domain(i, vars[k].min_dim + j) = err_vec(j);      // error for each science variable
                    }
                }
            }
#endif // MFA_SERIAL
#ifdef MFA_TBB
            parallel_for (size_t(0), (size_t)error.npts, [&] (size_t i)
                {
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        error.domain(i,j) = base.domain(i,j); // copy the geometric location of each point
                    }
                });

            parallel_for (size_t(0), (size_t)error.npts, [&] (size_t i)
                {
                VectorX<T> err_vec;                                 // errors for all coordinates in current model
                for (auto k = 0; k < vars.size() + 1; k++)      // for all models, geometry + science
                {
                    err_vec.resize(vars[k - 1].max_dim - vars[k - 1].min_dim);
                    AbsCoordError(*(vars[k - 1].mfa_data), base, i, err_vec, verbose);

                    for (auto j = 0; j < err_vec.size(); j++)
                    {
                        error.domain(i, vars[k - 1].min_dim + j) = err_vec(j); // error for each science variable
                    }
                }
                });
#endif // MFA_TBB
        }
    };
}                                           // namespace

#endif
