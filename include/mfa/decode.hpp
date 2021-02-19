//--------------------------------------------------------------
// decoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _DECODE_HPP
#define _DECODE_HPP

#include    <mfa/mfa_data.hpp>
#include    <mfa/mfa.hpp>

#include    <Eigen/Dense>

typedef Eigen::MatrixXi MatrixXi;

namespace mfa
{
    template <typename T>                                   // float or double
    struct MFA;

    template <typename T>                                   // float or double
    struct DecodeInfo
    {
        vector<MatrixX<T>>  N;                              // basis functions in each dim.
        vector<VectorX<T>>  temp;                           // temporary point in each dim.
        vector<int>         span;                           // current knot span in each dim.
        vector<int>         n;                              // number of control point spans in each dim
        vector<int>         iter;                           // iteration number in each dim.
        VectorX<T>          ctrl_pt;                        // one control point
        int                 ctrl_idx;                       // control point linear ordering index
        VectorX<T>          temp_denom;                     // temporary rational NURBS denominator in each dim
        vector<MatrixX<T>>  ders;                           // derivatives in each dim.

        DecodeInfo(const MFA_Data<T>&   mfa_data,           // current mfa
                   const VectorXi&      derivs)             // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                            // pass size-0 vector if unused
        {
            N.resize(mfa_data.p.size());
            temp.resize(mfa_data.p.size());
            span.resize(mfa_data.p.size());
            n.resize(mfa_data.p.size());
            iter.resize(mfa_data.p.size());
            ctrl_pt.resize(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());
            temp_denom = VectorX<T>::Zero(mfa_data.p.size());
            ders.resize(mfa_data.p.size());

            for (size_t i = 0; i < mfa_data.dom_dim; i++)
            {
                temp[i]    = VectorX<T>::Zero(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());
                // TODO: hard-coded for one tensor product
                N[i]       = MatrixX<T>::Zero(1, mfa_data.tmesh.tensor_prods[0].nctrl_pts(i));
                if (derivs.size() && derivs(i))
                    // TODO: hard-coded for one tensor product
                    ders[i] = MatrixX<T>::Zero(derivs(i) + 1, mfa_data.tmesh.tensor_prods[0].nctrl_pts(i));
            }
        }

        // reset decode info
        // version for recomputing basis functions
        void Reset(const MFA_Data<T>&   mfa_data,           // current mfa
                   const VectorXi&      derivs)             // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                            // pass size-0 vector if unused
        {
            temp_denom.setZero();
            for (auto i = 0; i < mfa_data.dom_dim; i++)
            {
                temp[i].setZero();
                iter[i] = 0;
                N[i].setZero();
                if (derivs.size() && derivs(i))
                    ders[i].setZero();
            }
        }

        // reset decode info
        // version for saved basis functions
        void Reset_saved_basis(const MFA_Data<T>&   mfa_data)    // current mfa
        {
            temp_denom.setZero();
            for (auto i = 0; i < mfa_data.dom_dim; i++)
            {
                temp[i].setZero();
                iter[i] = 0;
            }
        }
    };

    template <typename T>                               // float or double
    class Decoder
    {
    private:

        int                 tot_iters;                  // total iterations in flattened decoding of all dimensions
        MatrixXi            ct;                         // coordinates of first control point of curve for given iteration
                                                        // of decoding loop, relative to start of box of
                                                        // control points
        vector<size_t>      cs;                         // control point stride (only in decoder, not mfa)
        int                 verbose;                    // output level
        const MFA<T>&       mfa;                        // the mfa top-level object
        const MFA_Data<T>&  mfa_data;                   // the mfa data model

    public:

        Decoder(
                const MFA<T>&       mfa_,               // MFA top-level object
                const MFA_Data<T>&  mfa_data_,          // MFA data model
                int                 verbose_) :         // debug level
            mfa(mfa_),
            mfa_data(mfa_data_),
            verbose(verbose_)
        {
            // ensure that encoding was already done
            if (!mfa_data.p.size()                               ||
                !mfa_data.tmesh.all_knots.size()                 ||
                !mfa_data.tmesh.tensor_prods.size()              ||
                !mfa_data.tmesh.tensor_prods[0].nctrl_pts.size() ||
                !mfa_data.tmesh.tensor_prods[0].ctrl_pts.size())
            {
                fprintf(stderr, "Decoder() error: Attempting to decode before encoding.\n");
                exit(0);
            }

            // initialize decoding data structures
            // TODO: hard-coded for first tensor product only
            // needs to be expanded for multiple tensor products, maybe moved into the tensor product
            cs.resize(mfa_data.p.size(), 1);
            tot_iters = 1;                              // total number of iterations in the flattened decoding loop
            for (size_t i = 0; i < mfa_data.p.size(); i++)   // for all dims
            {
                tot_iters  *= (mfa_data.p(i) + 1);
                if (i > 0)
                    cs[i] = cs[i - 1] * mfa_data.tmesh.tensor_prods[0].nctrl_pts[i - 1];
            }
            ct.resize(tot_iters, mfa_data.p.size());

            // compute coordinates of first control point of curve corresponding to this iteration
            // these are relative to start of the box of control points located at co
            for (int i = 0; i < tot_iters; i++)      // 1-d flattening all n-d nested loop computations
            {
                int div = tot_iters;
                int i_temp = i;
                for (int j = mfa_data.p.size() - 1; j >= 0; j--)
                {
                    div      /= (mfa_data.p(j) + 1);
                    ct(i, j) =  i_temp / div;
                    i_temp   -= (ct(i, j) * div);
                }
            }
        }

        ~Decoder() {}

        // computes approximated points from a given set of domain points and an n-d NURBS volume
        // P&T eq. 9.77, p. 424
        // assumes all vectors have been correctly resized by the caller
        void DecodeDomain(
                const   InputInfo<T>&   input,                   // information for domain to be decoded
                        MatrixX<T>&     approx,                 // decoded output points (1st dim changes fastest)
                        int             min_dim,                // first dimension to decode
                        int             max_dim,                // last dimension to decode
                        bool            saved_basis)            // whether basis functions were saved and can be reused
        {
            VectorXi no_ders;                       // size 0 means no derivatives
            DecodeDomain(input, approx, min_dim, max_dim, saved_basis, no_ders);
        }

        // computes approximated points from a given set of domain points and an n-d NURBS volume
        // P&T eq. 9.77, p. 424
        // assumes all vectors have been correctly resized by the caller
        void DecodeDomain(
                const   InputInfo<T>&   input,              // information for domain to be decoded
                        MatrixX<T>&     approx,             // decoded output points (1st dim changes fastest)
                        int             min_dim,            // first dimension to decode
                        int             max_dim,            // last dimension to decode
                        bool            saved_basis,        // whether basis functions were saved and can be reused
                const   VectorXi&       derivs)             // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                        // pass size-0 vector if unused
        {
            int last = mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols() - 1;       // last coordinate of control point

#ifdef MFA_TBB                                          // TBB version, faster (~3X) than serial
            // thread-local DecodeInfo
            // ref: https://www.threadingbuildingblocks.org/tutorial-intel-tbb-thread-local-storage
            enumerable_thread_specific<DecodeInfo<T>> thread_decode_info(mfa_data, derivs);
            
            // parallel_for (size_t(0), (size_t)approx.rows(), [&] (size_t i)
            // {
            parallel_for (blocked_range<size_t>(0, approx.rows()), [&](blocked_range<size_t>& r)
            {
                auto pt_it  = input.iterator(r.begin());
                auto pt_end = input.iterator(r.end());
                for (; pt_it != pt_end; ++pt_it)
                {
                    // // convert linear idx to multidim. i,j,k... indices in each domain dimension
                    // VectorXi ijk(mfa_data.dom_dim);
                    // mfa_data.idx2ijk(mfa.ds(), i, ijk);

                    // // compute parameters for the vertices of the cell
                    // VectorX<T> param(mfa_data.dom_dim);
                    // for (int j = 0; j < mfa_data.dom_dim; j++)
                    //     param(j) = mfa.params()[j][ijk(j)];

                    VectorX<T>  cpt(last + 1);               // evaluated point
                    VectorX<T>  param(mfa_data.dom_dim);    // vector of param values
                    VectorXi    ijk(mfa_data.dom_dim);      // vector of param indices (structured grid only)
                    pt_it.params(param);  
                    // compute approximated point for this parameter vector

#ifndef MFA_TMESH   // original version for one tensor product

                    if (saved_basis && input.structured)
                    {
                        pt_it.ijk(ijk);
                        VolPt_saved_basis(ijk, param, cpt, thread_decode_info.local(), mfa_data.tmesh.tensor_prods[0]);

                        // debug
                        if (pt_it.idx() == 0)
                            fprintf(stderr, "Using VolPt_saved_basis\n");
                    }
                    else
                    {
                        VolPt(param, cpt, thread_decode_info.local(), mfa_data.tmesh.tensor_prods[0], derivs);

                        // debug
                        if (pt_it.idx() == 0)
                            fprintf(stderr, "Using VolPt\n");
                        if (saved_basis && !input.structured)
                            cerr << "Warning: Saved basis decoding not implemented with unstructured input. Proceeding with standard decoding" << endl;
                    }

#else           // tmesh version

                    if (pt_it.idx() == 0)
                        fprintf(stderr, "Using VolPt_tmesh\n");
                    VolPt_tmesh(param, cpt);

#endif

                    approx.block(pt_it.idx(), min_dim, 1, max_dim - min_dim + 1) = cpt.transpose();
                }
            });
            if (verbose)
                fprintf(stderr, "100 %% decoded\n");

#endif              // end TBB version

#ifdef MFA_SERIAL   // serial version

            DecodeInfo<T> decode_info(mfa_data, derivs);    // reusable decode point info for calling VolPt repeatedly

            VectorX<T> cpt(last + 1);                       // evaluated point
            VectorX<T> param(mfa_data.dom_dim);            // parameters for one point
            VectorXi   ijk(mfa_data.dom_dim);      // vector of param indices (structured grid only)

            // for (size_t i = 0; i < approx.rows(); i++)
            auto pt_it  = input.begin();
            auto pt_end = input.end();
            for (; pt_it != pt_end; ++pt_it)
            {
                // Get parameter values and indices at current point
                pt_it.params(param);

                // // extract parameter vector for one input point of all params
                // for (size_t j = 0; j < mfa_data.dom_dim; j++)
                //     param(j) = mfa.params()[j][iter(j)];

                // compute approximated point for this parameter vector

#ifndef MFA_TMESH   // original version for one tensor product

                if (saved_basis && input.structured)
                {
                    pt_it.ijk(ijk);
                    VolPt_saved_basis(ijk, param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0]);

                    // debug
                    if (pt_it.idx() == 0)
                        fprintf(stderr, "Using VolPt_saved_basis\n");
                }
                else
                {
                    VolPt(param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0], derivs);

                    // debug
                    if (pt_it.idx() == 0)
                        fprintf(stderr, "Using VolPt\n");
                    if (saved_basis && !input.structured)
                        cerr << "Warning: Saved basis decoding not implemented with unstructured input. Proceeding with standard decoding" << endl;
                }

#else           // tmesh version

                if (pt_it.idx() == 0)
                    fprintf(stderr, "Using VolPt_tmesh\n");
                VolPt_tmesh(param, cpt);

#endif          // end serial version

                // This code block no longer needed since using PtIterator
                //
                // // update the indices in the linearized vector of all params for next input point
                // for (size_t j = 0; j < mfa_data.dom_dim; j++)
                // {
                //     if (iter(j) < mfa.ndom_pts()(j) - 1)
                //     {
                //         iter(j)++;
                //         break;
                //     }
                //     else
                //         iter(j) = 0;
                // }

                approx.block(pt_it.idx(), min_dim, 1, max_dim - min_dim + 1) = cpt.transpose();

                // print progress
                if (verbose)
                    if (pt_it.idx() > 0 && approx.rows() >= 100 && pt_it.idx() % (approx.rows() / 100) == 0)
                        fprintf(stderr, "\r%.0f %% decoded", (T)pt_it.idx() / (T)(approx.rows()) * 100);
            }

#endif

        }

        // decode at a regular grid using saved basis that is computed once by this function
        // and then used to decode all the points in the grid
        void DecodeGrid(MatrixX<T>&         result,         // output
                        int                 min_dim,        // min dimension to decode
                        int                 max_dim,        // max dimension to decode
                        const VectorX<T>&   min_params,     // lower corner of decoding points
                        const VectorX<T>&   max_params,     // upper corner of decoding points
                        const VectorXi&     ndom_pts)       // number of points to decode in each direction
        {
            // precompute basis functions
            vector<vector<T>>   params(mfa_data.dom_dim);   // params for points to be decoded
            const VectorXi&     nctrl_pts = mfa_data.tmesh.tensor_prods[0].nctrl_pts;   // reference to control points (assume only one tensor)
            vector<MatrixX<T>>  NN(mfa_data.dom_dim);       // basis functions for points to be decoded
            for (int k = 0; k < mfa_data.dom_dim; k++)
                NN[k] = MatrixX<T>::Zero(ndom_pts(k), nctrl_pts(k));

            // compute params for points to be decoded
            for (int k = 0; k < mfa_data.dom_dim; k++)
            {
                params[k].resize(ndom_pts(k));
                T step = 0;
                if (ndom_pts(k) > 1)
                    step = (max_params(k) - min_params(k)) / (ndom_pts(k) - 1);
                params[k][0] = min_params(k);
                for (int i = 1; i < ndom_pts(k); i++)
                {
                    params[k][i] = min_params(k) + i * step;
                    // make sure we are between 0 and 1 !!
                    if (params[k][i] < 0)
                        params[k][i] = 0.;
                    if (params[k][i] > 1.0)
                        params[k][i] = 1.0;
                }
            }

            // compute basis functions for points to be decoded
            for (int k = 0; k < mfa_data.dom_dim; k++)
            {
                for (int i = 0; i < NN[k].rows(); i++)
                {
                    int span = mfa_data.FindSpan(k, params[k][i], nctrl_pts(k));
#ifndef MFA_TMESH   // original version for one tensor product

                    mfa_data.OrigBasisFuns(k, params[k][i], span, NN[k], i);

#else               // tmesh version

                    // TODO: TBD

#endif              // tmesh version
                }
            }

            VectorXi    derivs;                             // do not use derivatives yet, pass size 0
            VolIterator vol_it(ndom_pts);

#ifdef MFA_SERIAL   // serial version

            DecodeInfo<T>   decode_info(mfa_data, derivs);  // reusable decode point info for calling VolPt repeatedly
            VectorX<T>      cpt(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());                      // evaluated point
            VectorX<T>      param(mfa_data.p.size());       // parameters for one point
            VectorXi        ijk(mfa_data.dom_dim);          // multidim index in grid

            while (!vol_it.done())
            {
                int j = (int) vol_it.cur_iter();
                for (auto i = 0; i < mfa_data.dom_dim; i++)
                {
                    ijk[i] = vol_it.idx_dim(i);             // index along direction i in grid
                    param(i) = params[i][ijk[i]];
                }

#ifndef MFA_TMESH   // original version for one tensor product

                VolPt_saved_basis_grid(ijk, param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0], NN);

#else               // tmesh version

                    // TODO: TBD

#endif              // tmesh version

                vol_it.incr_iter();
                result.block(j, min_dim, 1, max_dim - min_dim + 1) = cpt.transpose();
            }

#endif              // serial version

#ifdef MFA_TBB      // TBB version

            // thread-local objects
            // ref: https://www.threadingbuildingblocks.org/tutorial-intel-tbb-thread-local-storage
            enumerable_thread_specific<DecodeInfo<T>>   thread_decode_info(mfa_data, derivs);
            enumerable_thread_specific<VectorXi>        thread_ijk(mfa_data.dom_dim);              // multidim index in grid
            enumerable_thread_specific<VectorX<T>>      thread_cpt(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());                      // evaluated point
            enumerable_thread_specific<VectorX<T>>      thread_param(mfa_data.p.size());           // parameters for one point

            parallel_for (size_t(0), (size_t)vol_it.tot_iters(), [&] (size_t j)
            {
                vol_it.idx_ijk(j, thread_ijk.local());
                for (auto i = 0; i < mfa_data.dom_dim; i++)
                    thread_param.local()(i)    = params[i][thread_ijk.local()[i]];

#ifndef MFA_TMESH   // original version for one tensor product

                VolPt_saved_basis_grid(thread_ijk.local(), thread_param.local(), thread_cpt.local(), thread_decode_info.local(), mfa_data.tmesh.tensor_prods[0], NN);

#else           // tmesh version

                // TODO: TBD

#endif          // tmesh version

                result.block(j, min_dim, 1, max_dim - min_dim + 1) = thread_cpt.local().transpose();
            });

#endif      // TBB version

        }

#ifdef MFA_TMESH

        // decode a point in the t-mesh
        // TODO: serial implementation, no threading
        // TODO: checks local support of individual control points, but could also check bounds of tensors and skip entire tensors
        // TODO: no derivatives as yet
        // TODO: weighs all dims, whereas other versions of VolPt have a choice of all dims or only last dim
        void VolPt_tmesh(const VectorX<T>&      param,      // parameters of point to decode
                         VectorX<T>&            out_pt)     // (output) point, allocated by caller
        {
            // debug
//             cerr << "VolPt_tmesh(): decoding point with param: " << param.transpose() << endl;

            // init
            out_pt = VectorX<T>::Zero(out_pt.size());
            T B_sum = 0.0;                                                          // sum of multidim basis function products
            T w_sum = 0.0;                                                          // sum of control point weights

            // compute range of anchors covering decoded point
            vector<vector<KnotIdx>> anchors(mfa_data.dom_dim);

            mfa_data.tmesh.anchors(param, true, anchors);

            // debug
//             if (param(0) > 0.7 && param(0) < 0.8 && param(1) > 0.6 && param(1) < 0.7)
//             {
//                 for (auto i = 0; i < mfa_data.dom_dim; i++)
//                 {
//                     fprintf(stderr, "anchors[%d]: [ ", i);
//                     for (auto j = 0; j < anchors[i].size(); j++)
//                         fprintf(stderr, "%lu ", anchors[i][j]);
//                     fprintf(stderr, "] ");
//                 }
//                 fprintf(stderr, "\n");
//             }

            for (auto k = 0; k < mfa_data.tmesh.tensor_prods.size(); k++)           // for all tensor products
            {
                const TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[k];

                // TODO: skip entire tensor if knot mins, maxs are too far away from decoded point
                // don't need to check individual control points in this case

                // debug
//                 cerr << "tensor " << k << endl;

                VolIterator         vol_iterator(t.nctrl_pts);                      // iterator over control points in the current tensor
                vector<KnotIdx>     anchor(mfa_data.dom_dim);                       // one anchor in (global, ie, over all tensors) index space
                VectorXi            ijk(mfa_data.dom_dim);                          // multidim index of current control point

                while (!vol_iterator.done())
                {
                    // get anchor of the current control point
                    vol_iterator.idx_ijk(vol_iterator.cur_iter(), ijk);
                    mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);

                    // skip odd degree duplicated control points, indicated by invalid weight
                    if (t.weights(vol_iterator.cur_iter()) == MFA_NAW)
                    {
                        // debug
//                         cerr << "skipping ctrl pt (MFA_NAW) " << t.ctrl_pts.row(vol_iterator.cur_iter()) << endl;

                        vol_iterator.incr_iter();
                        continue;
                    }

                    // debug
//                     bool skip = false;

                    // skip control points too far away from the decoded point
                    if (!mfa_data.tmesh.in_anchors(anchor, anchors))
                    {
                        // debug
//                         cerr << "skipping ctrl pt (too far away) [" << ijk.transpose() << "] " << t.ctrl_pts.row(vol_iterator.cur_iter()) << endl;

                        // debug
//                         skip = true;

                        vol_iterator.incr_iter();
                        continue;
                    }

                    // compute product of basis functions in each dimension
                    T                       B = 1.0;                                // product of basis function values in each dimension
                    vector<vector<KnotIdx>> local_knot_idxs(mfa_data.dom_dim);      // local knot indices
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        // local knot vector
                        local_knot_idxs[i].resize(mfa_data.p(i) + 2);               // local knot vector for current dim in index space
                        vector<T> local_knots(mfa_data.p(i) + 2);                   // local knot vector for current dim in parameter space
                        // TODO: why are knot_intersections being computed for each dimension?
                        mfa_data.tmesh.knot_intersections(anchor, k, true, local_knot_idxs);
                        for (auto n = 0; n < local_knot_idxs[i].size(); n++)
                            local_knots[n] = mfa_data.tmesh.all_knots[i][local_knot_idxs[i][n]];

                        // debug: print local knot idxs
//                         fprintf(stderr, "tensor = %d iter = %ld anchor[%d] = %ld local knot idxs = [", k, vol_iterator.cur_iter(), i, anchor[i]);
//                         for (auto j = 0; j < local_knot_idxs[i].size(); j++)
//                             fprintf(stderr, "%ld ", local_knot_idxs[i][j]);
//                         fprintf(stderr, "]\n");

                        // debug
//                         cerr << "OneBasisFun[dim " << i << " ]: " << mfa_data.OneBasisFun(i, param(i), local_knots) << endl;

                        B *= mfa_data.OneBasisFun(i, param(i), local_knots);
                    }
                    // compute the point
                    out_pt += B * t.ctrl_pts.row(vol_iterator.cur_iter()) * t.weights(vol_iterator.cur_iter());

                    // debug
//                     if (skip && B != 0.0)
//                     {
//                         cerr << "\nVolPt_tmesh(): Error: incorrect skip. decoding point with param: " << param.transpose() << endl;
//                         cerr << "tensor " << k << " skipping ctrl pt [" << ijk.transpose() << "] " << endl;
//                     }

                    B_sum += B * t.weights(vol_iterator.cur_iter());
                    // debug
//                     cerr << "Tensor: " << k << " CtrlPt: " << vol_iterator.cur_iter() << " B[all dims]: " << B << " B_sum: " << B_sum << endl;


                    vol_iterator.incr_iter();                                       // must increment volume iterator at the bottom of the loop
                }       // volume iterator
            }       // tensors

            // debug
//             cerr << "out_pt: " << out_pt.transpose() << " B_sum: " << B_sum << "\n" << endl;

            // divide by sum of weighted basis functions to make a partition of unity
            out_pt /= B_sum;

            // debug
//             cerr << "out_pt: " << out_pt.transpose() << "\n" << endl;
        }

#endif      // MFA_TMESH

        // compute a point from a NURBS n-d volume at a given parameter value
        // slower version for single points
        void VolPt(
                const VectorX<T>&       param,      // parameter value in each dim. of desired point
                VectorX<T>&             out_pt,     // (output) point, allocated by caller
                const TensorProduct<T>& tensor)     // tensor product to use for decoding
        {
            VectorXi no_ders;                   // size 0 vector means no derivatives
            VolPt(param, out_pt, tensor, no_ders);
        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // faster version for multiple points, reuses decode info
        void VolPt(
                const VectorX<T>&       param,      // parameter value in each dim. of desired point
                VectorX<T>&             out_pt,     // (output) point, allocated by caller
                DecodeInfo<T>&          di,         // reusable decode info allocated by caller (more efficient when calling VolPt multiple times)
                const TensorProduct<T>& tensor)     // tensor product to use for decoding
        {
            VectorXi no_ders;                   // size 0 vector means no derivatives
            VolPt(param, out_pt, di, tensor, no_ders);
        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // slower version for single points
        // algorithm 4.3, Piegl & Tiller (P&T) p.134
        void VolPt(
                const VectorX<T>&       param,      // parameter value in each dim. of desired point
                VectorX<T>&             out_pt,     // (output) point, allocated by caller
                const TensorProduct<T>& tensor,     // tensor product to use for decoding
                const VectorXi&         derivs)     // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                    // pass size-0 vector if unused
        {
            int last = mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols() - 1;      // last coordinate of control point
            if (derivs.size())                                                  // extra check for derivatives, won't slow down normal point evaluation
            {
                if (derivs.size() != mfa_data.p.size())
                {
                    fprintf(stderr, "Error: size of derivatives vector is not the same as the number of domain dimensions\n");
                    exit(0);
                }
                for (auto i = 0; i < mfa_data.p.size(); i++)
                    if (derivs(i) > mfa_data.p(i))
                        fprintf(stderr, "Warning: In dimension %d, trying to take derivative %d of an MFA with degree %d will result in 0. This may not be what you want",
                                i, derivs(i), mfa_data.p(i));
            }

            // init
            vector <MatrixX<T>> N(mfa_data.p.size());                           // basis functions in each dim.
            vector<VectorX<T>>  temp(mfa_data.p.size());                        // temporary point in each dim.
            vector<int>         span(mfa_data.p.size());                        // span in each dim.
            VectorX<T>          ctrl_pt(last + 1);                              // one control point
            int                 ctrl_idx;                                       // control point linear ordering index
            VectorX<T>          temp_denom = VectorX<T>::Zero(mfa_data.p.size());// temporary rational NURBS denominator in each dim

            // set up the volume iterator
            VectorXi npts = mfa_data.p + VectorXi::Ones(mfa_data.dom_dim);      // local support is p + 1 in each dim.
            VolIterator vol_iter(npts);                                         // for iterating in a flat loop over n dimensions

            // basis funs
            for (size_t i = 0; i < mfa_data.dom_dim; i++)                       // for all dims
            {
                temp[i]    = VectorX<T>::Zero(last + 1);
                span[i]    = mfa_data.FindSpan(i, param(i), tensor);
                N[i]       = MatrixX<T>::Zero(1, tensor.nctrl_pts(i));
                if (derivs.size() && derivs(i))
                {
#ifndef MFA_TMESH   // original version for one tensor product
                    MatrixX<T> Ders = MatrixX<T>::Zero(derivs(i) + 1, tensor.nctrl_pts(i));
                    mfa_data.DerBasisFuns(i, param(i), span[i], derivs(i), Ders);
                    N[i].row(0) = Ders.row(derivs(i));
#endif
                }
                else
                {
#ifndef MFA_TMESH   // original version for one tensor product
                    mfa_data.OrigBasisFuns(i, param(i), span[i], N[i], 0);
#else               // tmesh version
                    mfa_data.BasisFuns(i, param(i), span[i], N[i], 0);
#endif
                }
            }

            // linear index of first control point
            ctrl_idx = 0;
            for (int j = 0; j < mfa_data.p.size(); j++)
                ctrl_idx += (span[j] - mfa_data.p(j) + ct(0, j)) * cs[j];
            size_t start_ctrl_idx = ctrl_idx;

            while (!vol_iter.done())
            {
                // always compute the point in the first dimension
                ctrl_pt = tensor.ctrl_pts.row(ctrl_idx);
                T w     = tensor.weights(ctrl_idx);

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
                temp[0] += (N[0])(0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0)) * ctrl_pt * w;
#else                                                                           // weigh only range dimension
                for (auto j = 0; j < last; j++)
                    (temp[0])(j) += (N[0])(0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0)) * ctrl_pt(j);
                (temp[0])(last) += (N[0])(0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0)) * ctrl_pt(last) * w;
#endif

                temp_denom(0) += w * N[0](0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0));

                vol_iter.incr_iter();                                           // must call near bottom of loop, but before checking for done span below

                // for all dimensions except last, check if span is finished
                ctrl_idx = start_ctrl_idx;
                for (size_t k = 0; k < mfa_data.p.size(); k++)
                {
                    if (vol_iter.cur_iter() < vol_iter.tot_iters())
                        ctrl_idx += ct(vol_iter.cur_iter(), k) * cs[k];         // ctrl_idx for the next iteration
                    if (k < mfa_data.dom_dim - 1 && vol_iter.done(k))
                    {
                        // compute point in next higher dimension and reset computation for current dim
                        // use prev_idx_dim because iterator was already incremented above
                        temp[k + 1]        += (N[k + 1])(0, vol_iter.prev_idx_dim(k + 1) + span[k + 1] - mfa_data.p(k + 1)) * temp[k];
                        temp_denom(k + 1)  += temp_denom(k) * N[k + 1](0, vol_iter.prev_idx_dim(k + 1) + span[k + 1] - mfa_data.p(k + 1));
                        temp_denom(k)       = 0.0;
                        temp[k]             = VectorX<T>::Zero(last + 1);
                    }
                }
            }

            T denom;                                                            // rational denominator
            if (derivs.size() && derivs.sum())
                denom = 1.0;                                                    // TODO: weights for derivatives not implemented yet
            else
                denom = temp_denom(mfa_data.p.size() - 1);

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            out_pt = temp[mfa_data.p.size() - 1] / denom;
#else                                                                           // weigh only range dimension
            out_pt   = temp[mfa_data.p.size() - 1];
            out_pt(last) /= denom;
#endif

        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // slower version for single points
        // explicit full set of control points and weights
        // used only for testing tmesh during development (deprecate/remove eventually)
        // algorithm 4.3, Piegl & Tiller (P&T) p.134
        void VolPt(
                const VectorX<T>&       param,              // parameter value in each dim. of desired point
                VectorX<T>&             out_pt,             // (output) point, allocated by caller
                const VectorXi&         nctrl_pts,          // number of control points
                const MatrixX<T>&       ctrl_pts,           // p+1 control points per dimension, linearized
                const VectorX<T>&       weights)            // p+1 weights per dimension, linearized
        {
            int last = ctrl_pts.cols() - 1;                 // last coordinate of control point

            // init
            vector <MatrixX<T>> N(mfa_data.p.size());       // basis functions in each dim.
            vector<VectorX<T>>  temp(mfa_data.p.size());    // temporary point in each dim.
            vector<int>         span(mfa_data.p.size());    // span in each dim.
            VectorX<T>          ctrl_pt(last + 1);          // one control point
            int                 ctrl_idx;                   // control point linear ordering index
            VectorX<T>          temp_denom = VectorX<T>::Zero(mfa_data.p.size());     // temporary rational NURBS denominator in each dim

            // set up the volume iterator
            VectorXi npts = mfa_data.p + VectorXi::Ones(mfa_data.dom_dim);      // local support is p + 1 in each dim.
            VolIterator vol_iter(npts);                                         // for iterating in a flat loop over n dimensions

            // basis funs
            for (size_t i = 0; i < mfa_data.dom_dim; i++)   // for all dims
            {
                temp[i]    = VectorX<T>::Zero(last + 1);
                span[i]    = mfa_data.FindSpan(i, param(i), nctrl_pts(i));
                N[i]       = MatrixX<T>::Zero(1, nctrl_pts(i));

                mfa_data.OrigBasisFuns(i, param(i), span[i], N[i], 0);
            }

            // linear index of first control point
            ctrl_idx = 0;
            for (int j = 0; j < mfa_data.p.size(); j++)
                ctrl_idx += (span[j] - mfa_data.p(j) + ct(0, j)) * cs[j];
            size_t start_ctrl_idx = ctrl_idx;

            while (!vol_iter.done())
            {
                // always compute the point in the first dimension
                ctrl_pt = ctrl_pts.row(ctrl_idx);
                T w     = weights(ctrl_idx);

#ifdef WEIGH_ALL_DIMS                                       // weigh all dimensions
                temp[0] += (N[0])(0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0)) * ctrl_pt * w;
#else                                                       // weigh only range dimension
                for (auto j = 0; j < last; j++)
                    (temp[0])(j) += (N[0])(0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0)) * ctrl_pt(j);
                (temp[0])(last) += (N[0])(0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0)) * ctrl_pt(last) * w;
#endif

                temp_denom(0) += w * N[0](0, vol_iter.idx_dim(0) + span[0] - mfa_data.p(0));

                vol_iter.incr_iter();                                           // must call near bottom of loop, but before checking for done span below

                // for all dimensions except last, check if span is finished
                ctrl_idx = start_ctrl_idx;
                for (size_t k = 0; k < mfa_data.p.size(); k++)
                {
                    if (vol_iter.cur_iter() < vol_iter.tot_iters())
                        ctrl_idx += ct(vol_iter.cur_iter(), k) * cs[k];         // ctrl_idx for the next iteration
                    if (k < mfa_data.dom_dim - 1 && vol_iter.done(k))
                    {
                        // compute point in next higher dimension and reset computation for current dim
                        temp[k + 1]        += (N[k + 1])(0, vol_iter.prev_idx_dim(k + 1) + span[k + 1] - mfa_data.p(k + 1)) * temp[k];
                        temp_denom(k + 1)  += temp_denom(k) * N[k + 1](0, vol_iter.prev_idx_dim(k + 1) + span[k + 1] - mfa_data.p(k + 1));
                        temp_denom(k)       = 0.0;
                        temp[k]             = VectorX<T>::Zero(last + 1);
                    }
                }
            }

            T denom;                                        // rational denominator
            denom = temp_denom(mfa_data.p.size() - 1);

#ifdef WEIGH_ALL_DIMS                                       // weigh all dimensions
            out_pt = temp[mfa_data.p.size() - 1] / denom;
#else                                                       // weigh only range dimension
            out_pt   = temp[mfa_data.p.size() - 1];
            out_pt(last) /= denom;
#endif

        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // fastest version for multiple points, reuses saved basis functions
        // only values, no derivatives, because basis functions were not saved for derivatives
        // algorithm 4.3, Piegl & Tiller (P&T) p.134
        void VolPt_saved_basis(
                const VectorXi&             ijk,        // ijk index of input domain point being decoded
                const VectorX<T>&           param,      // parameter value in each dim. of desired point
                VectorX<T>&                 out_pt,     // (output) point, allocated by caller
                DecodeInfo<T>&              di,         // reusable decode info allocated by caller (more efficient when calling VolPt multiple times)
                const TensorProduct<T>&     tensor)     // tensor product to use for decoding
        {
            int last = tensor.ctrl_pts.cols() - 1;

            di.Reset_saved_basis(mfa_data);

            // set up the volume iterator
            VectorXi npts = mfa_data.p + VectorXi::Ones(mfa_data.dom_dim);      // local support is p + 1 in each dim.
            VolIterator vol_iter(npts);                                         // for iterating in a flat loop over n dimensions

            // linear index of first control point
            di.ctrl_idx = 0;
            for (int j = 0; j < mfa_data.dom_dim; j++)
            {
                di.span[j]    = mfa_data.FindSpan(j, param(j), tensor);
                di.ctrl_idx += (di.span[j] - mfa_data.p(j) + ct(0, j)) * cs[j];
            }
            size_t start_ctrl_idx = di.ctrl_idx;

            while (!vol_iter.done())
            {
                // always compute the point in the first dimension
                di.ctrl_pt  = tensor.ctrl_pts.row(di.ctrl_idx);
                T w         = tensor.weights(di.ctrl_idx);

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
                di.temp[0] += (mfa_data.N[0])(ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt * w;
#else                                                                           // weigh only range dimension
                for (auto j = 0; j < last; j++)
                    (di.temp[0])(j) += (mfa_data.N[0])(ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt(j);
                (di.temp[0])(last) += (mfa_data.N[0])(ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt(last) * w;
#endif

                di.temp_denom(0) += w * mfa_data.N[0](ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0));

                vol_iter.incr_iter();                                           // must call near bottom of loop, but before checking for done span below

                // for all dimensions except last, check if span is finished
                di.ctrl_idx = start_ctrl_idx;
                for (size_t k = 0; k < mfa_data.dom_dim; k++)
                {
                    if (vol_iter.cur_iter() < vol_iter.tot_iters())
                        di.ctrl_idx += ct(vol_iter.cur_iter(), k) * cs[k];      // ctrl_idx for the next iteration
                    if (k < mfa_data.dom_dim - 1 && vol_iter.done(k))
                    {
                        // compute point in next higher dimension and reset computation for current dim
                        // use prev_idx_dim because iterator was already incremented above
                        di.temp[k + 1]        += (mfa_data.N[k + 1])(ijk(k + 1), vol_iter.prev_idx_dim(k + 1) + di.span[k + 1] - mfa_data.p(k + 1)) * di.temp[k];
                        di.temp_denom(k + 1)  += di.temp_denom(k) * mfa_data.N[k + 1](ijk(k + 1), vol_iter.prev_idx_dim(k + 1) + di.span[k + 1] - mfa_data.p(k + 1));
                        di.temp_denom(k)       = 0.0;
                        di.temp[k].setZero();
                    }
                }
            }

            T denom = di.temp_denom(mfa_data.dom_dim - 1);                      // rational denominator

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            out_pt = di.temp[mfa_data.dom_dim - 1] / denom;
#else                                                                           // weigh only range dimension
            out_pt   = di.temp[mfa_data.dom_dim - 1];
            out_pt(last) /= denom;
#endif
        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // fastest version for multiple points, reuses computed basis functions
        // only values, no derivatives, because basis functions were not saved for derivatives
        // algorithm 4.3, Piegl & Tiller (P&T) p.134
        void VolPt_saved_basis_grid(
                const VectorXi&             ijk,        // ijk index of grid domain point being decoded
                const VectorX<T>&           param,      // parameter value in each dim. of desired point
                VectorX<T>&                 out_pt,     // (output) point, allocated by caller
                DecodeInfo<T>&              di,         // reusable decode info allocated by caller (more efficient when calling VolPt multiple times)
                const TensorProduct<T>&     tensor,     // tensor product to use for decoding
                vector<MatrixX<T>>&         NN )        // precomputed basis functions at grid
        {
            int last = tensor.ctrl_pts.cols() - 1;

            di.Reset_saved_basis(mfa_data);

            // set up the volume iterator
            VectorXi npts = mfa_data.p + VectorXi::Ones(mfa_data.dom_dim);      // local support is p + 1 in each dim.
            VolIterator vol_iter(npts);                                         // for iterating in a flat loop over n dimensions

            // linear index of first control point
            di.ctrl_idx = 0;
            for (int j = 0; j < mfa_data.dom_dim; j++)
            {
                di.span[j]    = mfa_data.FindSpan(j, param(j), tensor);
                di.ctrl_idx += (di.span[j] - mfa_data.p(j) + ct(0, j)) * cs[j];
            }
            size_t start_ctrl_idx = di.ctrl_idx;

            while (!vol_iter.done())
            {
                // always compute the point in the first dimension
                di.ctrl_pt  = tensor.ctrl_pts.row(di.ctrl_idx);
                T w         = tensor.weights(di.ctrl_idx);

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
                di.temp[0] += (NN[0])(ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt * w;
#else                                                                           // weigh only range dimension
                for (auto j = 0; j < last; j++)
                    (di.temp[0])(j) += (NN[0])(ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt(j);
                (di.temp[0])(last) += (NN[0])(ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt(last) * w;
#endif

                di.temp_denom(0) += w * NN[0](ijk(0), vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0));

                vol_iter.incr_iter();                                           // must call near bottom of loop, but before checking for done span below

                // for all dimensions except last, check if span is finished
                di.ctrl_idx = start_ctrl_idx;
                for (size_t k = 0; k < mfa_data.dom_dim; k++)
                {
                    if (vol_iter.cur_iter() < vol_iter.tot_iters())
                        di.ctrl_idx += ct(vol_iter.cur_iter(), k) * cs[k];      // ctrl_idx for the next iteration
                    if (k < mfa_data.dom_dim - 1 && vol_iter.done(k))
                    {
                        // compute point in next higher dimension and reset computation for current dim
                        // use prev_idx_dim because iterator was already incremented above
                        di.temp[k + 1]        += (NN[k + 1])(ijk(k + 1), vol_iter.prev_idx_dim(k + 1) + di.span[k + 1] - mfa_data.p(k + 1)) * di.temp[k];
                        di.temp_denom(k + 1)  += di.temp_denom(k) * NN[k + 1](ijk(k + 1), vol_iter.prev_idx_dim(k + 1) + di.span[k + 1] - mfa_data.p(k + 1));
                        di.temp_denom(k)       = 0.0;
                        di.temp[k].setZero();
                    }
                }
            }

            T denom = di.temp_denom(mfa_data.dom_dim - 1);                      // rational denominator

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            out_pt = di.temp[mfa_data.dom_dim - 1] / denom;
#else                                                                           // weigh only range dimension
            out_pt   = di.temp[mfa_data.dom_dim - 1];
            out_pt(last) /= denom;
#endif
        }

        // compute a point from a NURBS n-d volume at a given parameter value
        // faster version for multiple points, reuses decode info, but recomputes basis functions
        // algorithm 4.3, Piegl & Tiller (P&T) p.134
        void VolPt(
                const VectorX<T>&       param,      // parameter value in each dim. of desired point
                VectorX<T>&             out_pt,     // (output) point, allocated by caller
                DecodeInfo<T>&          di,         // reusable decode info allocated by caller (more efficient when calling VolPt multiple times)
                const TensorProduct<T>& tensor,     // tensor product to use for decoding
                const VectorXi&         derivs)     // derivative to take in each domain dim. (0 = value, 1 = 1st deriv, 2 = 2nd deriv, ...)
                                                    // pass size-0 vector if unused
        {
            int last = tensor.ctrl_pts.cols() - 1;
            if (derivs.size())                                                  // extra check for derivatives, won't slow down normal point evaluation
            {
                if (derivs.size() != mfa_data.dom_dim)
                {
                    fprintf(stderr, "Error: size of derivatives vector is not the same as the number of domain dimensions\n");
                    exit(0);
                }
                for (auto i = 0; i < mfa_data.dom_dim; i++)
                    if (derivs(i) > mfa_data.p(i))
                        fprintf(stderr, "Warning: In dimension %d, trying to take derivative %d of an MFA with degree %d will result in 0. This may not be what you want",
                                i, derivs(i), mfa_data.p(i));
            }

            di.Reset(mfa_data, derivs);

            // set up the volume iterator
            VectorXi npts = mfa_data.p + VectorXi::Ones(mfa_data.dom_dim);      // local support is p + 1 in each dim.
            VolIterator vol_iter(npts);                                         // for iterating in a flat loop over n dimensions

            // basis funs
            for (size_t i = 0; i < mfa_data.dom_dim; i++)                       // for all dims
            {
                di.span[i]    = mfa_data.FindSpan(i, param(i), tensor);

                if (derivs.size() && derivs(i))
                {
#ifndef MFA_TMESH   // original version for one tensor product
                    mfa_data.DerBasisFuns(i, param(i), di.span[i], derivs(i), di.ders[i]);
                    di.N[i].row(0) = di.ders[i].row(derivs(i));
#endif
                }
                else
                {
#ifndef MFA_TMESH   // original version for one tensor product
                    mfa_data.OrigBasisFuns(i, param(i), di.span[i], di.N[i], 0);
#else               // tmesh version
                    mfa_data.BasisFuns(i, param(i), di.span[i], di.N[i], 0);
#endif
                }
            }

            // linear index of first control point
            di.ctrl_idx = 0;
            for (int j = 0; j < mfa_data.dom_dim; j++)
                di.ctrl_idx += (di.span[j] - mfa_data.p(j) + ct(0, j)) * cs[j];
            size_t start_ctrl_idx = di.ctrl_idx;

            while (!vol_iter.done())
            {
                // always compute the point in the first dimension
                di.ctrl_pt  = tensor.ctrl_pts.row(di.ctrl_idx);
                T w         = tensor.weights(di.ctrl_idx);

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
                di.temp[0] += (di.N[0])(0, vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt * w;
#else                                                                           // weigh only range dimension
                for (auto j = 0; j < last; j++)
                    (di.temp[0])(j) += (di.N[0])(0, vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt(j);
                (di.temp[0])(last) += (di.N[0])(0, vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0)) * di.ctrl_pt(last) * w;
#endif

                di.temp_denom(0) += w * di.N[0](0, vol_iter.idx_dim(0) + di.span[0] - mfa_data.p(0));

                vol_iter.incr_iter();                                           // must call near bottom of loop, but before checking for done span below

                // for all dimensions except last, check if span is finished
                di.ctrl_idx = start_ctrl_idx;
                for (size_t k = 0; k < mfa_data.dom_dim; k++)
                {
                    if (vol_iter.cur_iter() < vol_iter.tot_iters())
                        di.ctrl_idx += ct(vol_iter.cur_iter(), k) * cs[k];      // ctrl_idx for the next iteration
                    if (k < mfa_data.dom_dim - 1 && vol_iter.done(k))
                    {
                        // compute point in next higher dimension and reset computation for current dim
                        // use prev_idx_dim because iterator was already incremented above
                        di.temp[k + 1]        += (di.N[k + 1])(0, vol_iter.prev_idx_dim(k + 1) + di.span[k + 1] - mfa_data.p(k + 1)) * di.temp[k];
                        di.temp_denom(k + 1)  += di.temp_denom(k) * di.N[k + 1](0, vol_iter.prev_idx_dim(k + 1) + di.span[k + 1] - mfa_data.p(k + 1));
                        di.temp_denom(k)       = 0.0;
                        di.temp[k].setZero();
                    }
                }
            }

            T denom;                                                            // rational denominator
            if (derivs.size() && derivs.sum())
                denom = 1.0;                                                    // TODO: weights for derivatives not implemented yet
            else
                denom = di.temp_denom(mfa_data.dom_dim - 1);

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            out_pt = di.temp[mfa_data.dom_dim - 1] / denom;
#else                                                                           // weigh only range dimension
            out_pt   = di.temp[mfa_data.dom_dim - 1];
            out_pt(last) /= denom;
#endif
        }

        // compute a point from a NURBS curve at a given parameter value
        // this version takes a temporary set of control points for one curve only rather than
        // reading full n-d set of control points from the mfa
        // algorithm 4.1, Piegl & Tiller (P&T) p.124
        void CurvePt(
                int                             cur_dim,        // current dimension
                T                               param,          // parameter value of desired point
                const MatrixX<T>&               temp_ctrl,      // temporary control points
                const VectorX<T>&               temp_weights,   // weights associate with temporary control points
                const TensorProduct<T>&         tensor,         // current tensor product
                VectorX<T>&                     out_pt)         // (output) point
        {
            int span   = mfa_data.FindSpan(cur_dim, param, tensor);
            MatrixX<T> N = MatrixX<T>::Zero(1, temp_ctrl.rows());// basis coefficients

#ifndef MFA_TMESH                                               // original version for one tensor product

            mfa_data.OrigBasisFuns(cur_dim, param, span, N, 0);

#else                                                           // tmesh version

            mfa_data.BasisFuns(cur_dim, param, span, N, 0);

#endif
            out_pt = VectorX<T>::Zero(temp_ctrl.cols());        // initializes and resizes

            for (int j = 0; j <= mfa_data.p(cur_dim); j++)
                out_pt += N(0, j + span - mfa_data.p(cur_dim)) *
                    temp_ctrl.row(span - mfa_data.p(cur_dim) + j) *
                    temp_weights(span - mfa_data.p(cur_dim) + j);

            // compute the denominator of the rational curve point and divide by it
            // sum of element-wise multiplication requires transpose so that both arrays are same shape
            // (rows in this case), otherwise eigen cannot multiply them
            T denom = (N.row(0).cwiseProduct(temp_weights.transpose())).sum();
            out_pt /= denom;
        }

    };
}

#endif
