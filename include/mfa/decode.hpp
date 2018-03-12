//--------------------------------------------------------------
// decoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _DECODE_HPP
#define _DECODE_HPP

#include    <mfa/data_model.hpp>
#include    <mfa/mfa.hpp>

#include    <Eigen/Dense>

typedef Eigen::MatrixXi MatrixXi;

namespace mfa
{
    template <typename T>                           // float or double
    class Decoder
    {
    public:

        Decoder(
                MFA_Data<T>& mfa_,                          // MFA data model
                int          verbose_)                      // output level
            : mfa(mfa_), verbose(verbose_)
        {
            // ensure that encoding was already done
            if (!mfa.p.size()         ||
                !mfa.ndom_pts.size()  ||
                !mfa.nctrl_pts.size() ||
                !mfa.domain.size()    ||
                !mfa.params.size()    ||
                !mfa.ctrl_pts.size()  ||
                !mfa.knots.size())
            {
                fprintf(stderr, "Decoder() error: Attempting to decode before encoding.\n");
                exit(0);
            }

            // initialize decoding data structures
            cs.resize(mfa.p.size(), 1);
            tot_iters = 1;                              // total number of iterations in the flattened decoding loop
            for (size_t i = 0; i < mfa.p.size(); i++)   // for all dims
            {
                tot_iters  *= (mfa.p(i) + 1);
                if (i > 0)
                    cs[i] = cs[i - 1] * mfa.nctrl_pts[i - 1];
            }
            ct.resize(tot_iters, mfa.p.size());

            // compute coordinates of first control point of curve corresponding to this iteration
            // these are relative to start of the box of control points located at co
            for (int i = 0; i < tot_iters; i++)      // 1-d flattening all n-d nested loop computations
            {
                int div = tot_iters;
                int i_temp = i;
                for (int j = mfa.p.size() - 1; j >= 0; j--)
                {
                    div      /= (mfa.p(j) + 1);
                    ct(i, j) =  i_temp / div;
                    i_temp   -= (ct(i, j) * div);
                }
            }
        }

        ~Decoder() {}

        // computes approximated points from a given set of domain points and an n-d NURBS volume
        // P&T eq. 9.77, p. 424
        // assumes all vectors have been correctly resized by the caller
        void Decode(
                MatrixX<T>& approx)                 // (output) points (1st dim changes fastest)
        {
            VectorXi no_ders;                       // size 0 means no derivatives
            Decode(approx, no_ders);
        }

        // computes approximated points from a given set of domain points and an n-d NURBS volume
        // P&T eq. 9.77, p. 424
        // assumes all vectors have been correctly resized by the caller
        void Decode(
                MatrixX<T>& approx,                 // (output) points (1st dim changes fastest)
                VectorXi&   derivs)                 // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                    // pass size-0 vector if unused
        {

#ifndef MFA_NO_TBB                                  // TBB version, faster (~3X) than serial

            vector<size_t> iter(mfa.p.size(), 0);   // parameter index (iteration count) in current dim.
            vector<size_t> ofst(mfa.p.size(), 0);   // start of current dim in linearized params
            int last = mfa.ctrl_pts.cols() - 1;     // last coordinate of control point

            for (size_t i = 0; i < mfa.p.size() - 1; i++)
                ofst[i + 1] = ofst[i] + mfa.ndom_pts(i);

            parallel_for (size_t(0), (size_t)mfa.domain.rows(), [&] (size_t i)
            {
                // convert linear idx to multidim. i,j,k... indices in each domain dimension
                VectorXi ijk(mfa.p.size());
                mfa.idx2ijk(i, ijk);

                // compute parameters for the vertices of the cell
                VectorX<T> param(mfa.p.size());
                for (int i = 0; i < mfa.p.size(); i++)
                    param(i) = mfa.params(ijk(i) + mfa.po[i]);

                // compute approximated point for this parameter vector
                VectorX<T> cpt(mfa.ctrl_pts.cols());        // evaluated point
                VolPt(param, cpt, derivs);
                approx.row(i) = cpt;
            });
            if (verbose)
                fprintf(stderr, "100 %% decoded\n");

#else                                               // serial version

            vector<size_t> iter(mfa.p.size(), 0);   // parameter index (iteration count) in current dim.
            vector<size_t> ofst(mfa.p.size(), 0);   // start of current dim in linearized params
            int last = mfa.ctrl_pts.cols() - 1;     // last coordinate of control point

            for (size_t i = 0; i < mfa.p.size() - 1; i++)
                ofst[i + 1] = ofst[i] + mfa.ndom_pts(i);

            VectorX<T> cpt(mfa.ctrl_pts.cols());    // evaluated point
            VectorX<T> param(mfa.p.size());         // parameters for one point

            for (size_t i = 0; i < mfa.domain.rows(); i++)
            {
                // extract parameter vector for one input point from the linearized vector of all params
                for (size_t j = 0; j < mfa.p.size(); j++)
                    param(j) = mfa.params(iter[j] + ofst[j]);

                // compute approximated point for this parameter vector
                VolPt(param, cpt, derivs);

                // update the indices in the linearized vector of all params for next input point
                for (size_t j = 0; j < mfa.p.size(); j++)
                {
                    if (iter[j] < mfa.ndom_pts(j) - 1)
                    {
                        iter[j]++;
                        break;
                    }
                    else
                        iter[j] = 0;
                }

                approx.row(i) = cpt;

                // print progress
                if (verbose)
                    if (i > 0 && mfa.domain.rows() >= 100 && i % (mfa.domain.rows() / 100) == 0)
                        fprintf(stderr, "\r%.0f %% decoded", (T)i / (T)(mfa.domain.rows()) * 100);
            }

#endif

        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // algorithm 4.3, Piegl & Tiller (P&T) p.134
        void VolPt(
                VectorX<T>& param,              // parameter value in each dim. of desired point
                VectorX<T>& out_pt)             // (output) point
        {
            VectorXi no_ders;                   // size 0 vector means no derivatives
            VolPt(param, out_pt, no_ders);
        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // algorithm 4.3, Piegl & Tiller (P&T) p.134
        void VolPt(
                VectorX<T>& param,              // parameter value in each dim. of desired point
                VectorX<T>& out_pt,             // (output) point
                VectorXi&   derivs)             // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                // pass size-0 vector if unused
        {
            // check dimensionality for sanity
            assert(mfa.p.size() < mfa.ctrl_pts.cols());
            if (derivs.size())                  // extra check for derivatives, won't slow down normal point evaluation
            {
                if (derivs.size() != mfa.p.size())
                {
                    fprintf(stderr, "Error: size of derivatives vector is not the same as the number of domain dimensions\n");
                    exit(0);
                }
                for (auto i = 0; i < mfa.p.size(); i++)
                    if (derivs(i) > mfa.p(i))
                        fprintf(stderr, "Warning: In dimension %d, trying to take derivative %d of an MFA with degree %d will result in 0. This may not be what you want",
                                i, derivs(i), mfa.p(i));
            }

            out_pt = VectorX<T>::Zero(mfa.ctrl_pts.cols());   // initializes and resizes
            vector <MatrixX<T>> N(mfa.p.size());              // basis functions in each dim.
            vector<VectorX<T>>  temp(mfa.p.size());           // temporary point in each dim.
            vector<int>         span(mfa.p.size());           // span in each dim.
            vector<int>         n(mfa.p.size());              // number of control point spans in each dim
            vector<int>         iter(mfa.p.size());           // iteration number in each dim.
            VectorX<T>          ctrl_pt(mfa.ctrl_pts.cols()); // one control point
            int                 ctrl_idx;                     // control point linear ordering index
            VectorX<T>          temp_denom = VectorX<T>::Zero(mfa.p.size());     // temporary rational NURBS denominator in each dim

            // init
            for (size_t i = 0; i < mfa.p.size(); i++)       // for all dims
            {
                temp[i]    = VectorX<T>::Zero(mfa.ctrl_pts.cols());
                iter[i]    = 0;
                span[i]    = mfa.FindSpan(i, param(i), mfa.ko[i]) - mfa.ko[i];  // relative to ko
                N[i]       = MatrixX<T>::Zero(1, mfa.nctrl_pts(i));
                if (derivs.size() && derivs(i))
                {
                    MatrixX<T> Ders = MatrixX<T>::Zero(derivs(i) + 1, mfa.nctrl_pts(i));
                    mfa.DerBasisFuns(i, param(i), span[i], derivs(i), Ders);
                    N[i].row(0) = Ders.row(derivs(i));
                }
                else
                    mfa.BasisFuns(i, param(i), span[i], N[i], 0);
            }

            for (int i = 0; i < tot_iters; i++)             // 1-d flattening all n-d nested loop computations
            {
                // control point linear order index
                ctrl_idx = 0;
                for (int j = 0; j < mfa.p.size(); j++)
                    ctrl_idx += (span[j] - mfa.p(j) + ct(i, j)) * cs[j];

                // always compute the point in the first dimension
                ctrl_pt = mfa.ctrl_pts.row(ctrl_idx);
                T w     = mfa.weights(ctrl_idx);

#ifdef WEIGH_ALL_DIMS                               // weigh all dimensions
                temp[0] += (N[0])(0, iter[0] + span[0] - mfa.p(0)) * ctrl_pt * w;
#else                                               // weigh only range dimension
                int last = mfa.ctrl_pts.cols() - 1;
                for (auto j = 0; j < last; j++)
                    (temp[0])(j) += (N[0])(0, iter[0] + span[0] - mfa.p(0)) * ctrl_pt(j);
                (temp[0])(last) += (N[0])(0, iter[0] + span[0] - mfa.p(0)) * ctrl_pt(last) * w;
#endif

                temp_denom(0) += w * N[0](0, iter[0] + span[0] - mfa.p(0));
                iter[0]++;

                // for all dimensions except last, check if span is finished
                for (size_t k = 0; k < mfa.p.size() - 1; k++)
                {
                    if (iter[k] - 1 == mfa.p(k))
                    {
                        // compute point in next higher dimension and reset computation for current dim
                        temp[k + 1]        += (N[k + 1])(0, iter[k + 1] + span[k + 1] - mfa.p(k + 1)) * temp[k];
                        temp_denom(k + 1)  += temp_denom(k) * N[k + 1](0, iter[k + 1] + span[k + 1] - mfa.p(k + 1));
                        temp_denom(k)       = 0.0;
                        temp[k]             = VectorX<T>::Zero(mfa.ctrl_pts.cols());
                        iter[k]             = 0;
                        iter[k + 1]++;
                    }
                }
            }

            T denom;                                // rational denominator
            if (derivs.size() && derivs.sum())
                denom = 1.0;                        // TODO: weights for derivatives not implemented yet
            else
                denom = temp_denom(mfa.p.size() - 1);

#ifdef WEIGH_ALL_DIMS                           // weigh all dimensions
            out_pt = temp[mfa.p.size() - 1] / denom;
#else                                           // weigh only range dimension
            out_pt   = temp[mfa.p.size() - 1];
            int last = mfa.ctrl_pts.cols() - 1;
            out_pt(last) /= denom;
#endif

        }

        // compute a point from a NURBS curve at a given parameter value
        // this version takes a temporary set of control points for one curve only rather than
        // reading full n-d set of control points from the mfa
        // algorithm 4.1, Piegl & Tiller (P&T) p.124
        void CurvePt(
                int         cur_dim,            // current dimension
                T           param,              // parameter value of desired point
                MatrixX<T>& temp_ctrl,          // temporary control points
                VectorX<T>& temp_weights,       // weights associate with temporary control points
                VectorX<T>& out_pt,             // (output) point
                int         ko = 0)             // starting knot offset
        {
            int span   = mfa.FindSpan(cur_dim, param, ko) - ko;         // relative to ko
            MatrixX<T> N = MatrixX<T>::Zero(1, temp_ctrl.rows());      // basis coefficients
            mfa.BasisFuns(cur_dim, param, span, N, 0);
            out_pt = VectorX<T>::Zero(temp_ctrl.cols());  // initializes and resizes

            for (int j = 0; j <= mfa.p(cur_dim); j++)
                out_pt += N(0, j + span - mfa.p(cur_dim)) *
                    temp_ctrl.row(span - mfa.p(cur_dim) + j) *
                    temp_weights(span - mfa.p(cur_dim) + j);

            // clamp dimensions other than cur_dim to same value as first control point
            // eliminates any wiggles in other dimensions due to numerical precision errors
            for (auto j = 0; j < mfa.p.size(); j++)
                if (j != cur_dim)
                    out_pt(j) = temp_ctrl(0, j);

            // compute the denominator of the rational curve point and divide by it
            // sum of element-wise multiplication requires transpose so that both arrays are same shape
            // (rows in this case), otherwise eigen cannot multiply them
            T denom = (N.row(0).cwiseProduct(temp_weights.transpose())).sum();
            out_pt /= denom;
        }

    private:

        int             tot_iters;                      // total iterations in flattened decoding of all dimensions
        MatrixXi        ct;                             // coordinates of first control point of curve for given iteration
                                                        // of decoding loop, relative to start of box of
                                                        // control points
        vector<size_t>  cs;                             // control point stride (only in decoder, not mfa)
        int             verbose;                        // output level
        MFA_Data<T>&    mfa;                            // the mfa object
    };
}

#endif
