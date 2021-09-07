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

// refine as many knot spans in one iteration as possible
#define MFA_ALL_SPANS

#include    <Eigen/Dense>
#include    <Eigen/Sparse>
#include    <Eigen/OrderingMethods>
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

#include    <diy/thirdparty/fmt/format.h>

#include    <mfa/util.hpp>
#include    <mfa/param.hpp>
#include    <mfa/pointset.hpp>
#include    <mfa/tmesh.hpp>
#include    <mfa/mfa_data.hpp>
#include    <mfa/decode.hpp>
#include    <mfa/encode.hpp>

// forward-declare diy::Serialization so that it can be declared as a friend by MFA
namespace diy
{
    template <typename U>
    struct Serialization;
}

namespace mfa
{
    template <typename T>                           // float or double
    class MFA
    {
        template <typename U> friend struct diy::Serialization;
        
        unique_ptr<MFA_Data<T>>         geometry;
        vector<unique_ptr<MFA_Data<T>>> vars;

        int                             verbose{0};

        // Recomputes pt_dim so that the MFA stays consistent after each model is added.
        // In theory, pt_dim should always equal the max_dim of the last science variable
        // This method emits a warning if this is not the case.
        int recompute_pt_dim()
        {
            pt_dim = geom_dim;

            for (int i = 0; i < vars.size(); i++)
            {
                if (vars[i])
                    pt_dim += vars[i]->dim();
                else
                    cerr << "WARNING: Encountered null variable model in MFA::recompute_pt_dim()" << endl;
            }

            if (vars.size() > 0)
            {
                if (pt_dim != vars.back()->max_dim + 1)
                {
                    cerr << "WARNING: MFA pt_dim is inconsistent" << endl;
                }
            }

            return pt_dim;
        }
    
    public:
        int                 dom_dim{0};            // domain dimensionality
        int                 geom_dim{0};           // dimension of geometry model (physical space)
        int                 pt_dim{0};             // full control point dimensionality

        MFA(int dom_dim_, int verbose_ = 0) :
            dom_dim(dom_dim_),
            verbose(verbose_)
        { }

        // This constructor is intended to be used for loading MFAs in and out of core
        // and can lead to an inconsistent state if the MFA_Data pointers are not set properly
        // 
        // NOTE: this constructor takes ownership of the MFA_Data pointers.
        //       these raw pointers should be set to nullptr after the MFA is constructed 
        //       to avoid double-frees, etc
        MFA(int dom_dim_, int verbose_, MFA_Data<T>* geom_, vector<MFA_Data<T>*> vars_) :
            dom_dim(dom_dim_),
            verbose(verbose_)
        {
            // If vars_ is nonempty, then we should always have a geometry model
            if (geom_ == nullptr && vars_.size() > 0)
                cerr << "WARNING: Constructing MFA with a null geometry model" << endl;

            // Each entry of vars should typically be non-null
            for (int i = 0; i < vars_.size(); i++)
            {
                if (vars_[i] == nullptr)
                    cerr << "WARNING: null variable model added during MFA construction" << endl;
            }

            // Set geometry model and geom_dim
            geometry.reset(geom_);
            if (geometry != nullptr)
                geom_dim = geometry->max_dim + 1;

            // Set variable models and pt_dim
            vars.resize(vars_.size());
            for (int i = 0; i < vars.size(); i++)
            {
                vars[i].reset(vars_[i]);
            }

            recompute_pt_dim();
        }

        ~MFA() { }

        const MFA_Data<T>& geom() const
        {
            return *geometry;
        }

        const MFA_Data<T>& var(int i) const
        {
            if (i < 0 || i >= nvars())
            {
                cerr << "ERROR: var index out of range in MFA::var()" << endl;
                exit(1);
            }

            return *(vars[i]);
        }

        int nvars() const
        {
            return vars.size();
        }

        void AddGeometry(const VectorXi& degree, const VectorXi& nctrl_pts, int dim)
        {
            if (verbose) 
                cout << "MFA: Adding geometry model" << endl;

            if (degree.size() != dom_dim || degree.minCoeff() < 0)
            {
                cerr << "ERROR: AddGeometry failed (degree invalid)" << endl;
                exit(1);
            }
            if (nctrl_pts.size() != dom_dim || (nctrl_pts - degree - VectorXi::Ones(dom_dim)).minCoeff() < 0)
            {
                cerr << "ERROR: AddGeometry failed (nctrl_pts invalid)" << endl;
                exit(1);
            }
            if (dim < 0)
            {
                cerr << "ERROR: AddGeometry failed (dim invalid)" << endl;
                exit(1);
            }

            // set geom_dim
            geom_dim = dim;

            // set up geometry model
            int min_dim = 0;
            int max_dim = dim - 1;
            geometry.reset(new MFA_Data<T>(degree, nctrl_pts, min_dim, max_dim));

            recompute_pt_dim();
        }

        void AddVariable(const VectorXi& degree, const VectorXi& nctrl_pts, int dim)
        {
            if (verbose) 
                cout << "MFA: Adding variable model " << vars.size() << endl;

            if (!geometry)
            {
                cerr << "ERROR: Cannot add variable model before adding geometry model" << endl;
                exit(1);
            }
            if (degree.size() != dom_dim || degree.minCoeff() < 0)
            {
                cerr << "ERROR: AddVariable failed (degree invalid)" << endl;
                exit(1);
            }
            if (nctrl_pts.size() != dom_dim || (nctrl_pts - degree - VectorXi::Ones(dom_dim)).minCoeff() < 0)
            {
                cerr << "ERROR: AddVariable failed (nctrl_pts invalid)" << endl;
                exit(1);
            }
            if (dim < 0)
            {
                cerr << "ERROR: AddVariable failed (dim invalid)" << endl;
                exit(1);
            }

            // Update vars vector
            int id = vars.size();

            // Compute min/max dim
            int min_dim = 0, max_dim = 0;            
            if (id == 0)
            {
                min_dim = geometry->max_dim + 1;
                max_dim = min_dim + dim - 1;
            }
            else
            {
                min_dim = vars[id-1]->max_dim + 1;
                max_dim = min_dim + dim - 1;
            }

            // Set up variable model
            vars.push_back(nullptr);                                                // Increase the size of the vars vector
            vars[id].reset(new MFA_Data<T>(degree, nctrl_pts, min_dim, max_dim));   // Add model to vars vector

            recompute_pt_dim();
        }

        // fixed number of control points encode
        void FixedEncodeImpl(
                MFA_Data<T>&        mfa_data,               // mfa data model
                const PointSet<T>&  input,                  // input points
                bool                weighted)         // solve for and use weights (default = true)
        {
            mfa_data.set_knots(input);

            // fixed encode assumes the tmesh has only one tensor product
            TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];

            t.weights = VectorX<T>::Ones(t.nctrl_pts.prod());
            Encoder<T> encoder(*this, mfa_data, input, verbose);

            if (input.structured)
                encoder.Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);
            else
                encoder.EncodeUnified(0, weighted);  // Assumes only one tensor product
        }

        // adaptive encode
        void AdaptiveEncodeImpl(
                MFA_Data<T>&        mfa_data,               // mfa data model
                const PointSet<T>&  input,                  // input points
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights (default = true)
                bool                local,                  // solve locally (with constraints) each round
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds) const       // optional maximum number of rounds
        {
            mfa_data.set_knots(input);
            
            Encoder<T> encoder(*this, mfa_data, input, verbose);

#ifndef MFA_TMESH           // original adaptive encode for one tensor product
            encoder.OrigAdaptiveEncode(err_limit, weighted, extents, max_rounds);
#else                       // adaptive encode for tmesh
            encoder.AdaptiveEncode(err_limit, weighted, local, extents, max_rounds);
#endif
        }

        // Decode geometry model at set of points
        void DecodeGeom(
                PointSet<T>&        output,
                bool                saved_basis,
                const VectorXi&     derivs = VectorXi()) const
        {
            cout << endl << "--- Decoding geometry ---" << endl << endl;

            mfa::Decoder<T> decoder(*geometry, verbose, saved_basis);
            decoder.DecodePointSet(output, geometry->min_dim, geometry->max_dim, derivs);
        }

        // Decode geometry model at single point
        void DecodeGeom(
                const VectorX<T>&   param,
                VectorX<T>&         out_point,
                const VectorXi&     derivs = VectorXi()) const
        {
            if (out_point.size() != geometry->dim())
            {
                cerr << "ERROR: Incorrect output vector dimension in MFA::DecodeGeom()" << endl;
                exit(1);
            }

            Decoder<T> decoder(*geometry, 0);        // nb. turning off verbose output when decoding single points
            
            // TODO: hard-coded for one tensor product
            decoder.VolPt(param, out_point, geometry->tmesh.tensor_prods[0], derivs);
        }

        // Decode variable model at set of points
        void DecodeVar(
                int                 i,
                PointSet<T>&        output,
                bool                saved_basis,
                const VectorXi&     derivs = VectorXi()) const
        {
            cout << endl << "--- Decoding science variable " << i << " ---" << endl << endl;

            if (i < 0 || i >= nvars())
            {
                cerr << "ERROR: var index out of range in MFA::DecodeVar()" << endl;
                exit(1);
            }

            mfa::Decoder<T> decoder(*(vars[i]), verbose, saved_basis);
            decoder.DecodePointSet(output, vars[i]->min_dim, vars[i]->max_dim, derivs);
        }

        // Decode variable model at single point
        void DecodeVar(
                int                 i,
                const VectorX<T>&   param,
                VectorX<T>&         out_point,
                const VectorXi&     derivs = VectorXi()) const
        {
            if (i < 0 || i >= nvars())
            {
                cerr << "ERROR: var index out of range in MFA::DecodeVar()" << endl;
                exit(1);
            }

            if (out_point.size() != vars[i]->dim())
            {
                cerr << "ERROR: Incorrect output vector dimension in MFA::DecodeVar()" << endl;
                exit(1);
            }

            mfa::Decoder<T> decoder(*(vars[i]), 0);     // nb. turning off verbose output when decoding single points

            // TODO: hard-coded for one tensor product
            decoder.VolPt(param, out_point, vars[i]->tmesh.tensor_prods[0], derivs);
        }

        // Decode all models at set of points
        void Decode(
                PointSet<T>&        output,
                bool                saved_basis,
                const VectorXi&     derivs = VectorXi()) const
        {
            DecodeGeom(output, saved_basis, derivs);
            for (int i = 0; i < nvars(); i++)
            {
                DecodeVar(i, output, saved_basis, derivs);
            }
        }

        // Decode all models at single point
        // 
        // Note: 
        // If desired, we can re-rewrite this to avoid the copies from temp_out into out_point by writing decode
        // functions to take MatrixBase<> objects instead of vectors. This allows "block" expressions like
        // out_point.segment(a,b) to be passed to VolPt, and decoding can be done in-place.
        // However, we would need to refactor DecodeVar, DecodeGeom, and potentially all of the VolPt functions
        // to accept MatrixBase<> inputs. This would not be hard, but could be confusing to read; also this function
        // is not intended to be high-performance anyway, so the benefit may be minimal.
        // 
        // See: http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
        void Decode(
                const VectorX<T>&       param,
                Eigen::Ref<VectorX<T>>  out_point,
                const VectorXi&         derivs = VectorXi()) const
        {
            if (out_point.size() != pt_dim)
            {
                cerr << "ERROR: Incorrect output vector dimension in MFA::Decode()" << endl;
                exit(1);
            }

            // We need an lvalue for passing into DecodeGeom and DecodeVar
            VectorX<T> temp_out = out_point.head(geom_dim);

            // Decode geometry
            DecodeGeom(param, temp_out, derivs);
            out_point.head(geometry->dim()) = temp_out;

            // Decode variables
            for (int i = 0; i < nvars(); i++)
            {
                temp_out.resize(vars[i]->dim());
                DecodeVar(i, param, temp_out, derivs);

                out_point.segment(vars[i]->min_dim, vars[i]->dim()) = temp_out;
            }
        }

        void DefiniteIntegral(
            const MFA_Data<T>&  mfa_data,
                  VectorX<T>&   output,
                  int           verbose,
            const VectorX<T>&   a,
            const VectorX<T>&   b)        
        {
            const TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[0];

            mfa::Decoder<T> decoder(mfa_data, verbose, false);
            decoder.DefiniteIntegral(t, a, b, output);
        }

        // Integrates with respect to parameter space
        // Multiply by extent of each domain dimension to obtain integral wrt physical space
        void IntegratePointSet(
                const MFA_Data<T>&  mfa_data,
                PointSet<T>&        output,
                int                 verbose,
                int                 min_dim,
                int                 max_dim)
        {
            const TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];

            mfa::Decoder<T> decoder(mfa_data, verbose, false);
            decoder.IntegratePointSet(output, t, min_dim, max_dim);
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


        void AbsPointSetError(
            const   mfa::PointSet<T>& base,
                    mfa::PointSet<T>& error,
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
                VectorX<T> err_vec;                             // errors for all coordinates in current model
                for (auto k = 0; k < vars.size(); k++)          // for all science models
                {
                    err_vec.resize(vars[k]->max_dim - vars[k]->min_dim);
                    AbsCoordError(*(vars[k]), base, i, err_vec, verbose);

                    for (auto j = 0; j < err_vec.size(); j++)
                    {
                        error.domain(i, vars[k]->min_dim + j) = err_vec(j);      // error for each science variable
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
                for (auto k = 0; k < vars.size(); k++)              // for all science models
                {
                    err_vec.resize(vars[k]->max_dim - vars[k]->min_dim);
                    AbsCoordError(*(vars[k]), base, i, err_vec, verbose);

                    for (auto j = 0; j < err_vec.size(); j++)
                    {
                        error.domain(i, vars[k]->min_dim + j) = err_vec(j); // error for each science variable
                    }
                }
                });
#endif // MFA_TBB
        }

        //-------------------------------------//
        //-------Convenience Functions---------//

        // Convenience function for adding linear geometry model
        void AddGeometry(int dim)
        {
            VectorXi degree = VectorXi::Ones(dom_dim);
            VectorXi nctrl_pts = VectorXi::Constant(dom_dim, 2);

            AddGeometry(degree, nctrl_pts, dim);
        }

        // Convenience function for adding models with same degree in each dimension
        void AddVariable(int degree, const VectorXi& nctrl_pts, int dim)
        {
            VectorXi degree_vec = VectorXi::Constant(dom_dim, degree);

            AddVariable(degree_vec, nctrl_pts, dim);
        }

        // Fixed encode geometry model only
        // Useful if input arguments vary between models, or different models are encoded by different threads
        void FixedEncodeGeom(const PointSet<T>& input, bool weighted)
        {
            if (verbose)
                cout << "MFA: Encoding geometry model (fixed)" << endl;

            FixedEncodeImpl(*geometry, input, weighted);
        }

        // Fixed encode single variable model only
        // Useful if input arguments vary between models, or different models are encoded by different threads 
        void FixedEncodeVar(int i, const PointSet<T>& input, bool weighted)
        {
            if (i < 0 || i >= nvars())
            {
                cerr << "ERROR: var index out of range in MFA::FixedEncodeVar()" << endl;
                exit(1);
            }

            if (verbose)
                cout << "MFA: Encoding variable model " << i << " (fixed)" << endl;

            FixedEncodeImpl(*(vars[i]), input, weighted);
        }

        // Fixed encode all models simultaneously
        void FixedEncode(const PointSet<T>& input, bool weighted)
        {
            FixedEncodeGeom(input, weighted);

            for (int i = 0; i < vars.size(); i++)
            {
                FixedEncodeVar(i, input, weighted);
            }
        }

        // Adaptive encode geometry model only
        // Useful if input arguments vary between models, or different models are encoded by different threads
        void AdaptiveEncodeGeom(const PointSet<T>&  input,
                                T                   err_limit,
                                bool                weighted,
                                bool                local,
                                const VectorX<T>&   extents,
                                int                 max_rounds)
        {
            if (verbose)
                cout << "MFA: Encoding geometry model (adaptive)" << endl;

            AdaptiveEncodeImpl(*geometry, input, err_limit, weighted, local, extents, max_rounds);
        }

        // Adaptive encode single variable model only
        // Useful if input arguments vary between models, or different models are encoded by different threads
        void AdaptiveEncodeVar( int i,
                                const PointSet<T>&  input,
                                T                   err_limit,
                                bool                weighted,
                                bool                local,
                                const VectorX<T>&   extents,
                                int                 max_rounds)
        {
            if (i < 0 || i >= nvars())
            {
                cerr << "ERROR: var index out of range in MFA::AdaptiveEncodeVar()" << endl;
                exit(1);
            }

            if (verbose)
                cout << "MFA: Encoding variable model " << i << " (adaptive)" << endl;
            
            AdaptiveEncodeImpl(*(vars[i]), input, err_limit, weighted, local, extents, max_rounds);
        }

        // Adaptive encode all models simultaneously
        void AdaptiveEncode(const PointSet<T>&  input,
                            T                   err_limit,
                            bool                weighted,
                            bool                local,
                            const VectorX<T>&   extents,
                            int                 max_rounds)
        {
            AdaptiveEncodeGeom(input, err_limit, weighted, local, extents, max_rounds);

            for (int i = 0; i < vars.size(); i++)
            {
                AdaptiveEncodeVar(i, input, err_limit, weighted, local, extents, max_rounds);
            }
        }
    };      // class MFA
}       // namespace mfa

#endif
