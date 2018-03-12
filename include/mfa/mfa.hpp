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

        MFA(
                VectorXi&   p_,             // polynomial degree in each dimension
                VectorXi&   ndom_pts_,      // number of input data points in each dim
                MatrixX<T>& domain_,        // input data points (1st dim changes fastest)
                MatrixX<T>& ctrl_pts_,      // (output, optional input) control points (1st dim changes fastest)
                VectorXi&   nctrl_pts_,     // (output, optional input) number of control points in each dim
                VectorX<T>& weights_,       // (output, optional input) weights associated with control points
                VectorX<T>& knots_,         // (output) knots (1st dim changes fastest)
                T           eps_ = 1.0e-6)  // minimum difference considered significant
        {
            mfa = new MFA_Data<T>(p_, ndom_pts_, domain_, ctrl_pts_, nctrl_pts_, weights_, knots_, eps_);
        }

        ~MFA()
        {
            delete mfa;
        }

        // encode
        void Encode(int verbose)                         // output level
        {
            Encoder<T> encoder(*mfa, verbose);
            encoder.Encode();
        }

        // fixed number of control points encode
        void FixedEncode(
                VectorXi &nctrl_pts_,               // (output) number of control points in each dim
                int      verbose,                   // output level
                bool     weighted)                  // solve for and use weights (default = true)
        {
            mfa->weights = VectorX<T>::Ones(mfa->tot_nctrl);
            Encoder<T> encoder(*mfa, verbose);
            encoder.Encode(weighted);
            nctrl_pts_ = mfa->nctrl_pts;
        }

        // adaptive encode
        void AdaptiveEncode(
                T         err_limit,                 // maximum allowable normalized error
                VectorXi& nctrl_pts_,                // (output) number of control points in each dim
                int       verbose,                   // output level
                bool      weighted,                  // solve for and use weights (default = true)
                int       max_rounds)                // optional maximum number of rounds
        {
            mfa->weights = VectorX<T>::Ones(mfa->tot_nctrl);
            Encoder<T> encoder(*mfa, verbose);
            encoder.AdaptiveEncode(err_limit, weighted, max_rounds);
            nctrl_pts_ = mfa->nctrl_pts;
        }

        // decode points
        void Decode(
                int         verbose,                // output level
                MatrixX<T>& approx)                 // decoded points
        {
            VectorXi no_derivs;                     // size-0 means no derivatives
            Decode(approx, verbose, no_derivs);
        }

        // decode derivatives
        void Decode(
                MatrixX<T>& approx,                 // decoded derivatives
                int         verbose,                // output level
                VectorXi&   derivs)                 // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
            // pass size-0 vector if unused
        {
            mfa::Decoder<T> decoder(*mfa, verbose);
            decoder.Decode(approx, derivs);
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

        // compute the error (absolute value of difference of range coordinates) of the mfa at a domain point
        // error is not normalized by the data range (absolute, not relative error)
        T RangeError(
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

            int last = mfa->domain.cols() - 1;           // range coordinate
            T err = fabs(cpt(last) - mfa->domain(idx, last));

            return err;
        }

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
