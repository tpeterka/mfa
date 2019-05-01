//--------------------------------------------------------------
// mfa object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_HPP
#define _MFA_HPP

#include    <mfa/data_model.hpp>
#include    <mfa/decode.hpp>
#include    <mfa/encode.hpp>
#include    <mfa/tmesh.hpp>

#include    <Eigen/Dense>
#include    <vector>
#include    <list>

#ifndef MFA_NO_TBB

#include    <tbb/tbb.h>
using namespace tbb;

#endif

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::ArrayXXf ArrayXXf;
typedef Eigen::ArrayXXd ArrayXXd;

template <typename T>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using VectorX  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using ArrayXX  = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using ArrayX   = Eigen::Array<T, Eigen::Dynamic, 1>;

using namespace std;

template <typename T>                       // float or double
class Encoder;

template <typename T>                       // float or double
class Decoder;

namespace mfa
{
    template <typename T>                   // float or double
    class MFA
    {
    public:

        // constructor for creating an mfa from input points
        MFA(
                VectorXi&           p_,             // polynomial degree in each dimension
                VectorXi&           ndom_pts_,      // number of input data points in each dim
                MatrixX<T>&         domain_,        // input data points (1st dim changes fastest)
                VectorXi&           nctrl_pts_,     // (output, optional input) number of control points in each dim
                int                 min_dim_ = -1,  // starting coordinate for input data; -1 = use all coordinates
                int                 max_dim_ = -1,  // ending coordinate for input data; -1 = use all coordinates
                T                   eps_ = 1.0e-6)  // minimum difference considered significant
        {
            if (min_dim_ == -1)
                min_dim_ = 0;
            if (max_dim_ == -1)
                max_dim_ = domain_.cols() - 1;
            mfa = new MFA_Data<T>(p_, ndom_pts_, domain_, nctrl_pts_, min_dim_, max_dim_, eps_);
        }

        // constructor when reading mfa in and knowing nothing about it yet except its degree and dimensionality
        MFA(
                VectorXi&           p_,             // polynomial degree in each dimension
                size_t              ntensor_prods,  // number of tensor products to allocate in tmesh
                int                 min_dim_ = -1,  // starting coordinate for input data; -1 = use all coordinates
                int                 max_dim_ = -1,  // ending coordinate for input data; -1 = use all coordinates
                T                   eps_ = 1.0e-6)  // minimum difference considered significant
        {
            if (min_dim_ == -1)
                min_dim_ = 0;
            if (max_dim_ == -1)
                max_dim_ = 1;
            mfa = new MFA_Data<T>(p_, ntensor_prods, min_dim_, max_dim_, eps_);
        }

        ~MFA()
        {
            delete mfa;
        }

        MFA_Data<T>& mfa_data()
        {
            return *mfa;
        }

        Tmesh<T>& tmesh()
        {
            return mfa->tmesh;
        }

        // encode
        void Encode(int verbose)                         // output level
        {
            Encoder<T> encoder(*mfa, verbose);
            encoder.Encode();
        }

        // fixed number of control points encode
        void FixedEncode(
                MatrixX<T>& domain,                     // input points
                VectorXi    &nctrl_pts_,                // (output) number of control points in each dim
                int         verbose,                    // output level
                bool        weighted)                   // solve for and use weights (default = true)
        {
            // TODO: hard-coded for single tensor
            mfa->tmesh.tensor_prods[0].weights = VectorX<T>::Ones(mfa->tmesh.tensor_prods[0].nctrl_pts.prod());
            Encoder<T> encoder(domain, *mfa, verbose);
            encoder.Encode(weighted);
            // TODO: hard-coded for single tensor
            nctrl_pts_ = mfa->tmesh.tensor_prods[0].nctrl_pts;
        }

        // adaptive encode
        void AdaptiveEncode(
                MatrixX<T>& domain,                    // input points
                T           err_limit,                 // maximum allowable normalized error
                VectorXi&   nctrl_pts_,                // (output) number of control points in each dim
                int         verbose,                   // output level
                bool        weighted,                  // solve for and use weights (default = true)
                VectorX<T>& extents,                   // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int         max_rounds)                // optional maximum number of rounds
        {
            // TODO: update to tmesh
#if 0
            mfa->weights = VectorX<T>::Ones(mfa->tot_nctrl);
            Encoder<T> encoder(domain, *mfa, verbose);
            encoder.AdaptiveEncode(err_limit, weighted, extents, max_rounds);
            nctrl_pts_ = mfa->nctrl_pts;
#endif
        }

        // decode values at all input points
        void DecodeDomain(
                MatrixX<T>& domain,                 // input points
                int         verbose,                // output level
                MatrixX<T>& approx,                 // decoded points
                int         min_dim,                // first dimension to decode
                int         max_dim)                // last dimension to decode
        {
            VectorXi no_derivs;                     // size-0 means no derivatives
            DecodeDomain(domain, verbose, approx, min_dim, max_dim, no_derivs);
        }

        // decode derivatives at all input points
        void DecodeDomain(
                MatrixX<T>& domain,                 // input points
                int         verbose,                // output level
                MatrixX<T>& approx,                 // decoded derivatives
                int         min_dim,                // first dimension to decode
                int         max_dim,                // last dimension to decode
                VectorXi&   derivs)                 // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                    // pass size-0 vector if unused
        {
            mfa::Decoder<T> decoder(*mfa, verbose);
            decoder.DecodeDomain(domain, approx, min_dim, max_dim, derivs);
        }

        // compute the error (absolute value of distance in normal direction) of the mfa at a domain point
        // error is not normalized by the data range (absolute, not relative error)
        T Error(
                size_t idx,               // index of domain point
                int    verbose)           // output level
        {
            // convert linear idx to multidim. i,j,k... indices in each domain dimension
            VectorXi ijk(mfa->p.size());
            mfa->idx2ijk(idx, ijk);

            // compute parameters for the vertices of the cell
            VectorX<T> param(mfa->p.size());
            for (int i = 0; i < mfa->p.size(); i++)
                param(i) = mfa->params(ijk(i) + mfa->po[i]);

            // approximated value
            VectorX<T> cpt(mfa->ctrl_pts.cols());          // approximated point
            Decoder<T> decoder(*mfa, verbose);
            decoder.VolPt(param, cpt);

            T err = fabs(mfa->NormalDistance(cpt, idx));

            return err;
        }

        // decode single point at the given parameter location
        void DecodePt(VectorX<T>& param,        // parameters of point to decode
                      VectorX<T>& cpt)          // (output) decoded point
        {
            int verbose = 0;
            Decoder<T> decoder(*mfa, verbose);
            // TODO: hard-coded for one tensor product
            decoder.VolPt(param, cpt, mfa->tmesh.tensor_prods[0]);
        }

        // compute the error (absolute value of coordinate-wise difference) of the mfa at a domain point
        // error is not normalized by the data range (absolute, not relative error)
        void AbsCoordError(
                MatrixX<T>& domain,             // input points
                size_t      idx,                // index of domain point
                VectorX<T>& error,              // absolute value of error at each coordinate
                int         verbose)            // output level
        {
            // convert linear idx to multidim. i,j,k... indices in each domain dimension
            VectorXi ijk(mfa->p.size());
            mfa->idx2ijk(idx, ijk);

            // compute parameters for the vertices of the cell
            VectorX<T> param(mfa->p.size());
            for (int i = 0; i < mfa->p.size(); i++)
                param(i) = mfa->params(ijk(i) + mfa->po[i]);

            // NB, assumes at least one tensor product exists and that all have the same ctrl pt dimensionality
            int pt_dim = mfa->tmesh.tensor_prods[0].ctrl_pts.cols();

            // approximated value
            VectorX<T> cpt(pt_dim);          // approximated point
            Decoder<T> decoder(*mfa, verbose);
            decoder.VolPt(param, cpt, mfa->tmesh.tensor_prods[0]);      // TODO: hard-coded for first tensor product

            for (auto i = 0; i < pt_dim; i++)
                error(i) = fabs(cpt(i) - domain(idx, mfa->min_dim + i));
        }

        // DEPRECATED, use AbsCoordError instead
        // compute the error (absolute value of difference of range coordinates) of the mfa at a domain point
        // error is not normalized by the data range (absolute, not relative error)
//         T RangeError(
//                 size_t idx,               // index of domain point
//                 int    verbose)           // output level
//         {
//             // convert linear idx to multidim. i,j,k... indices in each domain dimension
//             VectorXi ijk(mfa->p.size());
//             mfa->idx2ijk(idx, ijk);
// 
//             // compute parameters for the vertices of the cell
//             VectorX<T> param(mfa->p.size());
//             for (int i = 0; i < mfa->p.size(); i++)
//                 param(i) = mfa->params(ijk(i) + mfa->po[i]);
// 
//             // approximated value
//             VectorX<T> cpt(mfa->ctrl_pts.cols());          // approximated point
//             Decoder<T> decoder(*mfa, verbose);
//             decoder.VolPt(param, cpt);
// 
//             int last = mfa->domain.cols() - 1;           // range coordinate
//             T err = fabs(cpt(last) - mfa->domain(idx, last));
// 
//             return err;
//         }

        T NormalDistance(
                VectorX<T>& pt,             // point whose distance from domain is desired
                size_t      cell_idx)       // index of min. corner of cell in the domain
        {
            return mfa->NormalDistance(pt, cell_idx);
        }

    private:

        MFA_Data<T>* mfa;                           // the mfa data model
    };

}

#endif
