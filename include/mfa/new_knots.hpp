//--------------------------------------------------------------
// new knots inserter object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _NEW_KNOTS_HPP
#define _NEW_KNOTS_HPP

#include    <mfa/mfa_data.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/encode.hpp>
#include    <mfa/decode.hpp>

#include    <Eigen/Dense>
#include    <vector>
#include    <set>
#include    <iostream>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

namespace mfa
{
    template <typename T>                                   // float or double
    class NewKnots
    {
    private:
        size_t                  dom_dim;                            // dimensionality of parameter space
        const PointSet<T>&      input;                              // input points
        MFA_Data<T>&            mfa_data;                           // mfa data

    public:

        NewKnots(
                MFA_Data<T>&        mfa_data_,
                const PointSet<T>&  input_) :
            dom_dim(mfa_data_.dom_dim),
            input(input_),
            mfa_data(mfa_data_) {}

        ~NewKnots() {}

        // inserts a set of knots (in all dimensions) into the original knot set
        void OrigInsertKnots(
                vector<vector<T>>&          new_knots,              // new knots
                vector<vector<int>>&        new_levels,             // new knot levels
                vector<vector<KnotIdx>>&    inserted_knot_idxs)     // (output) indices in each dim. of inserted knots in full knot vector after insertion
        {
            // debug
//             fmt::print(stderr, "Inserting knots with OrigInsertKnots()\n");

            vector<vector<T>> temp_knots(dom_dim);
            vector<vector<int>> temp_levels(dom_dim);
            inserted_knot_idxs.resize(dom_dim);

            // insert new_knots into knots: replace old knots with union of old and new (in temp_knots)
            for (size_t k = 0; k < dom_dim; k++)                // for all domain dimensions
            {
                inserted_knot_idxs[k].clear();
                auto ninserted = mfa_data.tmesh.all_knots[k].size();      // number of new knots inserted

                // manual walk along old and new knots so that levels can be inserted along with knots
                // ie, that's why std::set_union cannot be used
                auto ak = mfa_data.tmesh.all_knots[k].begin();
                auto al = mfa_data.tmesh.all_knot_levels[k].begin();
                auto nk = new_knots[k].begin();
                auto nl = new_levels[k].begin();
                while (ak != mfa_data.tmesh.all_knots[k].end() || nk != new_knots[k].end())
                {
                    if (ak == mfa_data.tmesh.all_knots[k].end())
                    {
                        temp_knots[k].push_back(*nk++);
                        temp_levels[k].push_back(*nl++);
                        inserted_knot_idxs[k].push_back(temp_knots[k].size() - 1);
                    }
                    else if (nk == new_knots[k].end())
                    {
                        temp_knots[k].push_back(*ak++);
                        temp_levels[k].push_back(*al++);
                    }
                    else if (*ak < *nk)
                    {
                        temp_knots[k].push_back(*ak++);
                        temp_levels[k].push_back(*al++);
                    }
                    else if (*nk < *ak)
                    {
                        temp_knots[k].push_back(*nk++);
                        temp_levels[k].push_back(*nl++);
                        inserted_knot_idxs[k].push_back(temp_knots[k].size() - 1);
                    }
                    else if (*ak == *nk)
                    {
                        temp_knots[k].push_back(*ak++);
                        temp_levels[k].push_back(*al++);
                        nk++;
                        nl++;
                    }
                }

                // in case of single tensor and structured data, don't allow more control points than input points
                if (mfa_data.tmesh.tensor_prods.size() == 1 && input.structured &&
                        mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + inserted_knot_idxs[k].size() > input.ndom_pts(k))
                {
                    fmt::print(stderr, "OrigInsertKnots(): Unable to insert {} knots in dimension {} because {} control points would outnumber {} input points.\n",
                            inserted_knot_idxs[k].size(), k, mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + inserted_knot_idxs[k].size(), input.ndom_pts(k));
                    if (mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) > input.ndom_pts(k))
                    {
                        fmt::print(stderr, "Error: OrigInsertKnots(): control points already outnumber input points in dimension {}. This should not happen.\n", k);
                        abort();
                    }
                    size_t nknots = input.ndom_pts(k) - mfa_data.tmesh.tensor_prods[0].nctrl_pts(k);
                    inserted_knot_idxs[k].resize(nknots);
                    new_knots[k].resize(nknots);
                    new_levels[k].resize(nknots);
                    fmt::print(stderr, "Inserting {} knots instead.\n", nknots);
                }

                for (auto i = 0; i < inserted_knot_idxs[k].size(); i++)
                {
                    auto idx = inserted_knot_idxs[k][i];
                    mfa_data.tmesh.insert_knot_at_pos(k, idx, temp_levels[k][idx], temp_knots[k][idx], input.params->param_grid);
                }
            }   // for all domain dimensions
        }

        // inserts a set of knots (in all dimensions) into all_knots of the tmesh
        // this version is for global solve, also called inside of local solve
        // returns idx of parent tensor containing new knot to be inserted (assuming single knot insertion)
        int InsertKnots(
                vector<vector<T>>&          new_knots,              // new knots
                vector<vector<int>>&        new_levels,             // new knot levels
                vector<vector<KnotIdx>>&    inserted_knot_idxs)     // (output) indices in each dim. of inserted knots in full knot vector after insertion
        {
            vector<vector<T>> temp_knots(dom_dim);
            vector<vector<int>> temp_levels(dom_dim);
            inserted_knot_idxs.resize(dom_dim);

            // insert new_knots into knots: replace old knots with union of old and new (in temp_knots)
            for (size_t k = 0; k < dom_dim; k++)                // for all domain dimensions
            {
                inserted_knot_idxs[k].clear();
                auto ninserted = mfa_data.tmesh.all_knots[k].size();      // number of new knots inserted

                // manual walk along old and new knots so that levels can be inserted along with knots
                // ie, that's why std::set_union cannot be used
                auto ak = mfa_data.tmesh.all_knots[k].begin();
                auto al = mfa_data.tmesh.all_knot_levels[k].begin();
                auto nk = new_knots[k].begin();
                auto nl = new_levels[k].begin();
                while (ak != mfa_data.tmesh.all_knots[k].end() || nk != new_knots[k].end())
                {
                    if (ak == mfa_data.tmesh.all_knots[k].end())
                    {
                        temp_knots[k].push_back(*nk++);
                        temp_levels[k].push_back(*nl++);
                        inserted_knot_idxs[k].push_back(temp_knots[k].size() - 1);
                    }
                    else if (nk == new_knots[k].end())
                    {
                        temp_knots[k].push_back(*ak++);
                        temp_levels[k].push_back(*al++);
                    }
                    else if (*ak < *nk)
                    {
                        temp_knots[k].push_back(*ak++);
                        temp_levels[k].push_back(*al++);
                    }
                    else if (*nk < *ak)
                    {
                        temp_knots[k].push_back(*nk++);
                        temp_levels[k].push_back(*nl++);
                        inserted_knot_idxs[k].push_back(temp_knots[k].size() - 1);
                    }
                    else if (*ak == *nk)
                    {
                        temp_knots[k].push_back(*ak++);
                        temp_levels[k].push_back(*al++);
                        nk++;
                        nl++;
                    }
                }

                // in case of single tensor and structured data, don't allow more control points than input points
                if (mfa_data.tmesh.tensor_prods.size() == 1 && input.structured &&
                        mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + inserted_knot_idxs[k].size() > input.ndom_pts(k))
                {
                    fmt::print(stderr, "InsertKnots(): Unable to insert {} knots in dimension {} because {} control points would outnumber {} input points.\n",
                            inserted_knot_idxs[k].size(), k, mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + inserted_knot_idxs[k].size(), input.dom_pts(k));
                    if (mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) > input.ndom_pts(k))
                    {
                        fmt::print(stderr, "Error: InsertKnots(): control points already outnumber input points in dimension {}. This should not happen.\n", k);
                        abort();
                    }
                    size_t nknots = input.ndom_pts(k) - mfa_data.tmesh.tensor_prods[0].nctrl_pts(k);
                    inserted_knot_idxs[k].resize(nknots);
                    new_knots[k].resize(nknots);
                    new_levels[k].resize(nknots);
                    fmt::print(stderr, "Inserting {} knots instead.\n", nknots);
                }

                // insert the knots into the tmesh
                for (auto i = 0; i < inserted_knot_idxs[k].size(); i++)
                {
                    auto idx = inserted_knot_idxs[k][i];
                    mfa_data.tmesh.insert_knot(k, idx, temp_levels[k][idx], temp_knots[k][idx], input.params->param_grid);
                }
            }   // for all domain dimensions

            // find parent tensor (assumes only one knot being inserted)
            vector<KnotIdx> inserted_knot_idx(dom_dim);
            for (auto j = 0; j < dom_dim; j++)
                inserted_knot_idx[j] = inserted_knot_idxs[j][0];    // there is only one inserted knot, index 0, but dimensions are switched, hence the copy
            return mfa_data.tmesh.search_tensors(inserted_knot_idx, mfa_data.p / 2);
        }

        // inserts a set of knots (in all dimensions) into the original knot set
        // this version is for local solve
        // returns idx of parent tensor containing new knot to be inserted (assuming single knot insertion)
        int InsertKnots(
                vector<vector<T>>&          new_knots,              // new knots
                vector<vector<int>>&        new_levels,             // new knot levels
                vector<vector<KnotIdx>>&    inserted_knot_idxs,     // (output) indices in each dim. of inserted knots in full knot vector after insertion
                vector<VectorXi>&           new_nctrl_pts,          // (output) new number of control points in each dim. from P&T knot insertion, one std::vector element for each knot inserted
                vector<MatrixX<T>>&         new_ctrl_pts,           // (output) new control points from P&T knot insertion, one std::vector element for each knot inserted
                vector<VectorX<T>>&         new_weights)            // (output) new weights from P&T knot insertion, one std::vector element for each knot inserted
        {
            // insert new_knots into knots: replace old knots with union of old and new (in temp_knots)
            int parent_tensor_idx = InsertKnots(new_knots, new_levels, inserted_knot_idxs);

            int nnew_knots = new_knots[0].size();                   // number of new knots being inserted
            VectorX<T> param(dom_dim);                          // current knot to be inserted
            new_nctrl_pts.resize(nnew_knots);
            new_ctrl_pts.resize(nnew_knots);
            new_weights.resize(nnew_knots);
            vector<KnotIdx> inserted_idx(dom_dim);

#ifdef MFA_LINEAR_LOCAL

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[parent_tensor_idx];
            for (auto i = 0; i < nnew_knots; i++)
            {
                new_nctrl_pts[i].resize(dom_dim);
                for (auto k = 0; k < dom_dim; k++)
                    (new_nctrl_pts[i])(k) = t.nctrl_pts(k) + 1;
                new_ctrl_pts[i].resize(new_nctrl_pts[i].prod(), t.ctrl_pts.cols());
                new_weights[i].resize(new_ctrl_pts[i].rows());
                // linear local solve does not solve for weights; set to 1
                new_weights[i] = VectorX<T>::Ones(new_weights[i].size());
            }

#else

            // call P&T knot insertion
            for (auto i = 0; i < nnew_knots; i++)
            {
                for (auto j = 0; j < dom_dim; j++)
                {
                    param(j) = new_knots[j][i];
                    inserted_idx[j] = inserted_knot_idxs[j][i];
                }
                mfa_data.ExistKnotInsertion(inserted_idx,
                                            param,
                                            mfa_data.tmesh.tensor_prods[parent_tensor_idx],
                                            new_nctrl_pts[i],
                                            new_ctrl_pts[i],
                                            new_weights[i]);
            }

#endif

            return parent_tensor_idx;
        }

//         // DEPRECATE eventually in favor of AllErrorSpans
//         // computes error in knot spans and finds first new knot (in all dimensions at once) that should be inserted
//         // returns true if all done, ie, no new knots inserted
//         //
//         // This version is for the tmesh with global solve
//         // It takes a full tensor product of ctrl_pts and weights of the same quantities as all_knots in the tmesh
//         // and decodes that in order to determine first error span.
//         //
//         // TODO: lots of optimizations possible; this is completely naive so far
//         // optimizations:
//         // each time called, resume search at next domain point
//         // step through domain points fineer each time the end is reached
//         // increment ijk in the loop over domain points instead of calling idx2ijk
//         // TBB? (currently serial)
//         bool FirstErrorSpan(
//                 VectorX<T>                  extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
//                 T                           err_limit,              // max. allowed error
//                 int                         iter,                   // iteration number
//                 int&                        parent_tensor_idx,      // (output) idx of parent tensor of error span
//                 vector<vector<KnotIdx>>&    inserted_knot_idxs)     // (output) indices in each dim. of inserted knots in full knot vector after insertion
//         {
//             // debug
//             fprintf(stderr, "*** Using global solve in FirstErrorSpan ***\n");
// 
//             Decoder<T>          decoder(mfa_data, 1);
//             VectorXi            ijk(dom_dim);                   // i,j,k of domain point
//             VectorX<T>          param(dom_dim);                 // parameters of domain point
//             vector<int>         span(dom_dim);                  // knot span in each dimension
//             VectorXi            derivs;                             // size 0 means unused
//             DecodeInfo<T>       decode_info(mfa_data, derivs);      // reusable decode point info for calling VolPt repeatedly
//             vector<vector<T>>   new_knots(dom_dim);             // new knots
//             vector<vector<int>> new_levels(dom_dim);            // new knot levels
// 
//             if (!extents.size())
//                 extents = VectorX<T>::Ones(input.pt_dim);
// 
//             for (auto pt_it = input.begin(), pt_end = input.end(); pt_it != pt_end; ++pt_it)
//             {
//                 pt_it.params(param);
//                 VectorX<T> cpt(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());
// 
// #ifdef MFA_TMESH
//                 decoder.VolPt_tmesh(param, cpt);
// #else
//                 decoder.VolPt(param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0]);
// #endif
// 
// 
//                 // debug
// //                 cerr << "cpt: " << cpt.transpose() << endl;
// 
//                 int last = input.pt_dim - 1;                       // range coordinate
// 
//                 // error
//                 T max_err = 0.0;
//                 for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
//                 {
//                     T err = fabs(cpt(j) - input.domain(pt_it.idx(), mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
//                     max_err = err > max_err ? err : max_err;
//                 }
// 
//                 if (max_err > err_limit)
//                 {
//                     // check that new spans will contain at least one input point
//                     bool empty_span = false;
//                     for (auto k = 0; k < dom_dim; k++)
//                     {
//                         span[k] = mfa_data.FindSpan(k, param(k));
// 
//                         // debug: hard code span
// //                         span[k] = 5;
// 
//                         // span should never be the last knot because of the repeated knots at end
//                         assert(span[k] < mfa_data.tmesh.all_knots[k].size() - 1);
// 
//                         // current span must contain at least two input points
//                         size_t low_idx  = mfa_data.tmesh.all_knot_param_idxs[k][span[k]];
//                         size_t high_idx = mfa_data.tmesh.all_knot_param_idxs[k][span[k] + 1];
//                         if (high_idx - low_idx < 2)
//                         {
//                             empty_span = true;
//                             break;
//                         }
// 
//                         // new knot would be the midpoint of the span containing the domain point parameters
//                         T new_knot_val = (mfa_data.tmesh.all_knots[k][span[k]] + mfa_data.tmesh.all_knots[k][span[k] + 1]) / 2.0;
// 
//                         // if the current span were to be split, check whether the resulting spans will have an input point
//                         ParamIdx param_idx  = low_idx;
//                         while (input.params->param_grid[k][param_idx] < new_knot_val)
//                             param_idx++;
//                         if (param_idx - low_idx == 0 || high_idx - param_idx == 0)
//                         {
//                             empty_span =  true;
//                             break;
//                         }
//                     }
// 
//                     if (empty_span)
//                         continue;               // next input point
// 
//                     for (auto k = 0; k < dom_dim; k++)
//                     {
//                         // new knot is the midpoint of the span containing the domain point parameters
//                         T new_knot_val = (mfa_data.tmesh.all_knots[k][span[k]] + mfa_data.tmesh.all_knots[k][span[k] + 1]) / 2.0;
//                         new_knots[k].push_back(new_knot_val);
//                         new_levels[k].push_back(iter + 1);  // adapt at the next level, for now every iteration is a new level
//                     }
// 
//                     parent_tensor_idx = InsertKnots(new_knots, new_levels, inserted_knot_idxs);
//                     return false;
//                 }   // max_err > err_limit
//             }   // for all input points
// 
//             return true;
//         }
// 
//         // DEPRECATE eventually in favor of AllErrorSpans
//         // computes error in knot spans and finds first new knot (in all dimensions at once) that should be inserted
//         // returns true if all done, ie, no new knots inserted
//         //
//         // This version is for tmesh with local solve
//         //
//         // TODO: lots of optimizations possible; this is completely naive so far
//         // optimizations:
//         // each time called, resume search at next domain point
//         // step through domain points fineer each time the end is reached
//         // increment ijk in the loop over domain points instead of calling idx2ijk
//         // TBB? (currently serial)
//         bool FirstErrorSpan(
//                 VectorX<T>                  extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
//                 T                           err_limit,              // max. allowed error
//                 int                         iter,                   // iteration number
//                 int&                        parent_tensor_idx,      // (output) idx of parent tensor of error span
//                 vector<vector<KnotIdx>>&    inserted_knot_idxs,     // (output) indices in each dim. of inserted knots in full knot vector after insertion
//                 vector<VectorXi>&           new_nctrl_pts,          // (output) new number of control points in each dim. from P&T knot insertion, one std:vector element per knot inserted
//                 vector<MatrixX<T>>&         new_ctrl_pts,           // (output) new control points from P&T knot insertion, one std::vector element per knot inserted
//                 vector<VectorX<T>>&         new_weights,            // (output) new weights from P&T knot insertion, one std::vector element per knot inserted
//                 bool&                       local)                  // (output) span allows local solve to proceed (sufficient constraints exist)
//         {
//             // debug
//             fprintf(stderr, "*** Using local solve in FirstErrorSpan ***\n");
// 
//             Decoder<T>          decoder(mfa_data, 1);
//             VectorXi            ijk(dom_dim);                   // i,j,k of domain point
//             VectorX<T>          param(dom_dim);                 // parameters of domain point
//             VectorXi            derivs;                             // size 0 means unused
//             DecodeInfo<T>       decode_info(mfa_data, derivs);      // reusable decode point info for calling VolPt repeatedly
//             vector<vector<T>>   new_knots(dom_dim);             // new knots
//             vector<vector<int>> new_levels(dom_dim);            // new knot levels
// 
//             if (!extents.size())
//                 extents = VectorX<T>::Ones(input.pt_dim);
// 
//             for (auto pt_it = input.begin(), pt_end = input.end(); pt_it != pt_end; ++pt_it)
//             {
//                 pt_it.params(param);
//                 VectorX<T> cpt(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());
// 
// #ifdef MFA_TMESH
//                 decoder.VolPt_tmesh(param, cpt);
// #else
//                 decoder.VolPt(param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0]);
// #endif
// 
//                 // debug
// //                 cerr << "cpt: " << cpt.transpose() << endl;
// 
//                 int last = input.pt_dim - 1;                       // range coordinate
// 
//                 // error
//                 T max_err = 0.0;
//                 for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
//                 {
//                     T err = fabs(cpt(j) - input.domain(pt_it.idx(), mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
//                     max_err = err > max_err ? err : max_err;
//                 }
// 
//                 if (max_err > err_limit)
//                 {
//                     vector<KnotIdx> span(dom_dim);              // index space coordinates of span where knot will be inserted
//                     for (auto k = 0; k < dom_dim; k++)
//                         span[k] = mfa_data.FindSpan(k, param(k));
// 
//                     // check that new spans will contain at least one input point
//                     bool empty_span = false;
//                     for (auto k = 0; k < dom_dim; k++)
//                     {
//                         // current span must contain at least two input points
//                         size_t low_idx  = mfa_data.tmesh.all_knot_param_idxs[k][span[k]];
//                         size_t high_idx = mfa_data.tmesh.all_knot_param_idxs[k][span[k] + 1];
//                         if (high_idx - low_idx < 2)
//                         {
//                             // debug
// //                             fprintf(stderr, "Insufficient input points when attempting to split span %lu: low_idx = %lu high_idx = %lu\n",
// //                                     span[k], low_idx, high_idx);
// 
//                             empty_span = true;
//                             break;
//                         }
// 
//                         // new knot would be the midpoint of the span containing the domain point parameters
//                         T new_knot_val = (mfa_data.tmesh.all_knots[k][span[k]] + mfa_data.tmesh.all_knots[k][span[k] + 1]) / 2.0;
// 
//                         // if the current span were to be split, check whether the resulting spans will have an input point
//                         ParamIdx param_idx  = low_idx;
//                         while (input.params->param_grid[k][param_idx] < new_knot_val)
//                             param_idx++;
//                         if (param_idx - low_idx == 0 || high_idx - param_idx == 0)
//                         {
//                             empty_span =  true;
//                             break;
//                         }
//                     }
// 
//                     if (empty_span)
//                         continue;               // next input point
// 
//                     // debug: hard code span
// //                     for (auto k = 0; k < dom_dim; k++)
// //                         span[k] = 5;
// 
//                     for (auto k = 0; k < dom_dim; k++)
//                     {
//                         // span should never be the last knot because of the repeated knots at end
//                         assert(span[k] < mfa_data.tmesh.all_knots[k].size() - 1);
// 
//                         // new knot is the midpoint of the span containing the domain point parameters
//                         T new_knot_val = (mfa_data.tmesh.all_knots[k][span[k]] + mfa_data.tmesh.all_knots[k][span[k] + 1]) / 2.0;
//                         new_knots[k].push_back(new_knot_val);
//                         new_levels[k].push_back(iter + 1);  // adapt at the next level, for now every iteration is a new level
//                     }
// 
//                     if (local)
//                         parent_tensor_idx = InsertKnots(new_knots,
//                                                         new_levels,
//                                                         inserted_knot_idxs,
//                                                         new_nctrl_pts,
//                                                         new_ctrl_pts,
//                                                         new_weights);
//                     else
//                         parent_tensor_idx = InsertKnots(new_knots,
//                                                         new_levels,
//                                                         inserted_knot_idxs);
// 
//                     return false;
//                 }   // max_err > err_limit
//             }   // for all input points
// 
//             return true;
//         }
// 
//         // DEPRECATE eventually in favor of AllErrorSpans
//         // computes error in knot spans and finds best new knot (span w/ highest error, in all dimensions at once) that should be inserted
//         // returns true if all done, ie, no new knots inserted
//         //
//         // This version is for the tmesh with global solve
//         // It takes a full tensor product of ctrl_pts and weights of the same quantities as all_knots in the tmesh
//         // and decodes that in order to determine first error span.
//         //
//         // TODO: lots of optimizations possible; this is completely naive so far
//         // optimizations:
//         // each time called, resume search at next domain point
//         // step through domain points finer each time the end is reached
//         // increment ijk in the loop over domain points instead of calling idx2ijk
//         // TBB? (currently serial)
//         bool MaxErrorSpan(
//                 VectorX<T>                  extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
//                 T                           err_limit,              // max. allowed error
//                 int                         iter,                   // iteration number
//                 int&                        parent_tensor_idx,      // (output) idx of parent tensor of error span
//                 vector<vector<KnotIdx>>&    inserted_knot_idxs)     // (output) indices in each dim. of inserted knots in full knot vector after insertion
//         {
//             // debug
//             fprintf(stderr, "*** Using global solve in MaxErrorSpan ***\n");
// 
//             Decoder<T>          decoder(mfa_data, 1);
//             VectorXi            ijk(dom_dim);                   // i,j,k of domain point
//             VectorX<T>          param(dom_dim);                 // parameters of domain point
//             vector<int>         span(dom_dim);                  // knot span in each dimension
//             VectorXi            derivs;                             // size 0 means unused
//             DecodeInfo<T>       decode_info(mfa_data, derivs);      // reusable decode point info for calling VolPt repeatedly
//             vector<vector<T>>   new_knots(dom_dim);             // new knots
//             vector<vector<int>> new_levels(dom_dim);            // new knot levels
// 
//             if (!extents.size())
//                 extents = VectorX<T>::Ones(input.pt_dim);
// 
//             T Max_err = -1.0;                                       // overall max error
//             vector<int> Max_span(dom_dim);                      // span at location of overall max error
// 
//             for (auto pt_it = input.begin(), pt_end = input.end(); pt_it != pt_end; ++pt_it)
//             {
//                 pt_it.params(param);
//                 VectorX<T> cpt(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());
// 
// #ifdef MFA_TMESH
//                 decoder.VolPt_tmesh(param, cpt);
// #else
//                 decoder.VolPt(param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0]);
// #endif
// 
//                 // debug
// //                 cerr << "cpt: " << cpt.transpose() << endl;
// 
//                 int last = input.pt_dim - 1;                       // range coordinate
// 
//                 // error
//                 T max_err = 0.0;                                    // max error over dimensions (eg, science variables)
//                 for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
//                 {
//                     T err = fabs(cpt(j) - input.domain(pt_it.idx(), mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
//                     max_err = err > max_err ? err : max_err;
//                 }
// 
//                 if (max_err > err_limit && max_err > Max_err)
//                 {
//                     // check that new spans will contain at least one input point
//                     bool empty_span = false;
//                     for (auto k = 0; k < dom_dim; k++)
//                     {
//                         span[k] = mfa_data.FindSpan(k, param(k));
// 
//                         // span should never be the last knot because of the repeated knots at end
//                         assert(span[k] < mfa_data.tmesh.all_knots[k].size() - 1);
// 
//                         // current span must contain at least two input points
//                         size_t low_idx  = mfa_data.tmesh.all_knot_param_idxs[k][span[k]];
//                         size_t high_idx = mfa_data.tmesh.all_knot_param_idxs[k][span[k] + 1];
//                         if (high_idx - low_idx < 2)
//                         {
//                             empty_span = true;
//                             break;
//                         }
// 
//                         // new knot would be the midpoint of the span containing the domain point parameters
//                         T new_knot_val = (mfa_data.tmesh.all_knots[k][span[k]] + mfa_data.tmesh.all_knots[k][span[k] + 1]) / 2.0;
// 
//                         // if the current span were to be split, check whether the resulting spans will have an input point
//                         ParamIdx param_idx  = low_idx;
//                         while (input.params->param_grid[k][param_idx] < new_knot_val)
//                             param_idx++;
//                         if (param_idx - low_idx == 0 || high_idx - param_idx == 0)
//                         {
//                             empty_span =  true;
//                             break;
//                         }
//                     }
// 
//                     if (empty_span)
//                         continue;               // next input point
// 
//                     // record Max_err and Max_span
//                     Max_err = max_err;
//                     for (auto k = 0; k < dom_dim; k++)
//                         Max_span[k] = span[k];
//                 }   // max_err > err_limit && max_err > Max_err
//             }   // for all input points
// 
//             // debug: hard code span
// //             if (Max_err >= 0.0)
// //             {
// //                 for (auto k = 0; k < dom_dim; k++)
// //                     Max_span[k] = 4;
// //             }
// 
//             if (Max_err >= 0.0)
//             {
//                 for (auto k = 0; k < dom_dim; k++)
//                 {
//                     // new knot is the midpoint of the span containing the domain point parameters
//                     T new_knot_val = (mfa_data.tmesh.all_knots[k][Max_span[k]] + mfa_data.tmesh.all_knots[k][Max_span[k] + 1]) / 2.0;
//                     new_knots[k].push_back(new_knot_val);
//                     new_levels[k].push_back(iter + 1);  // adapt at the next level, for now every iteration is a new level
//                 }
// 
//                 parent_tensor_idx = InsertKnots(new_knots, new_levels, inserted_knot_idxs);
//                 return false;
//             }
//             return true;
//         }
// 
//         // DEPRECATE eventually in favor of AllErrorSpans
//         // computes error in knot spans and finds best new knot (span w/ highest error, in all dimensions at once) that should be inserted
//         // returns true if all done, ie, no new knots inserted
//         //
//         // This version is for tmesh with local solve
//         //
//         // TODO: lots of optimizations possible; this is completely naive so far
//         // optimizations:
//         // each time called, resume search at next domain point
//         // step through domain points fineer each time the end is reached
//         // increment ijk in the loop over domain points instead of calling idx2ijk
//         // TBB? (currently serial)
//         bool MaxErrorSpan(
//                 VectorX<T>                  extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
//                 T                           err_limit,              // max. allowed error
//                 int                         iter,                   // iteration number
//                 int&                        parent_tensor_idx,      // (output) idx of parent tensor of error span
//                 vector<vector<KnotIdx>>&    inserted_knot_idxs,     // (output) indices in each dim. of inserted knots in full knot vector after insertion
//                 vector<VectorXi>&           new_nctrl_pts,          // (output) new number of control points in each dim. from P&T knot insertion, one std:vector element per knot inserted
//                 vector<MatrixX<T>>&         new_ctrl_pts,           // (output) new control points from P&T knot insertion, one std::vector element per knot inserted
//                 vector<VectorX<T>>&         new_weights,            // (output) new weights from P&T knot insertion, one std::vector element per knot inserted
//                 bool&                       local)                  // (output) span allows local solve to proceed (sufficient constraints exist)
//         {
//             // debug
//             fprintf(stderr, "*** Using local solve in MaxErrorSpan ***\n");
// 
//             Decoder<T>          decoder(mfa_data, 1);
//             VectorXi            ijk(dom_dim);                   // i,j,k of domain point
//             VectorX<T>          param(dom_dim);                 // parameters of domain point
//             VectorXi            derivs;                             // size 0 means unused
//             DecodeInfo<T>       decode_info(mfa_data, derivs);      // reusable decode point info for calling VolPt repeatedly
//             vector<vector<T>>   new_knots(dom_dim);             // new knots
//             vector<vector<int>> new_levels(dom_dim);            // new knot levels
// 
//             if (!extents.size())
//                 extents = VectorX<T>::Ones(input.pt_dim);
// 
//             T Max_err = -1.0;                                       // overall max error
//             vector<KnotIdx> Max_span(dom_dim);                  // span at location of overall max error
// 
//             for (auto pt_it = input.begin(), pt_end = input.end(); pt_it != pt_end; ++pt_it)
//             {
//                 pt_it.params(param);
//                 VectorX<T> cpt(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());
// 
// #ifdef MFA_TMESH
//                 decoder.VolPt_tmesh(param, cpt);
// #else
//                 decoder.VolPt(param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0]);
// #endif
// 
//                 // debug
// //                 cerr << "cpt: " << cpt.transpose() << endl;
// 
//                 int last = input.pt_dim - 1;                       // range coordinate
// 
//                 // error
//                 T max_err = 0.0;
//                 for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
//                 {
//                     T err = fabs(cpt(j) - input.domain(pt_it.idx(), mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
//                     max_err = err > max_err ? err : max_err;
//                 }
// 
//                 if (max_err > err_limit && max_err > Max_err)
//                 {
//                     vector<KnotIdx> span(dom_dim);              // index space coordinates of span where knot will be inserted
//                     for (auto k = 0; k < dom_dim; k++)
//                         span[k] = mfa_data.FindSpan(k, param(k));
// 
//                     // check that new spans will contain at least one input point
//                     bool empty_span = false;
//                     for (auto k = 0; k < dom_dim; k++)
//                     {
//                         // current span must contain at least two input points
//                         size_t low_idx  = mfa_data.tmesh.all_knot_param_idxs[k][span[k]];
//                         size_t high_idx = mfa_data.tmesh.all_knot_param_idxs[k][span[k] + 1];
//                         if (high_idx - low_idx < 2)
//                         {
//                             // debug
// //                             fprintf(stderr, "Insufficient input points when attempting to split span %lu: low_idx = %lu high_idx = %lu\n",
// //                                     span[k], low_idx, high_idx);
// 
//                             empty_span = true;
//                             break;
//                         }
// 
//                         // new knot would be the midpoint of the span containing the domain point parameters
//                         T new_knot_val = (mfa_data.tmesh.all_knots[k][span[k]] + mfa_data.tmesh.all_knots[k][span[k] + 1]) / 2.0;
// 
//                         // if the current span were to be split, check whether the resulting spans will have an input point
//                         ParamIdx param_idx  = low_idx;
//                         while (input.params->param_grid[k][param_idx] < new_knot_val)
//                             param_idx++;
//                         if (param_idx - low_idx == 0 || high_idx - param_idx == 0)
//                         {
//                             empty_span =  true;
//                             break;
//                         }
//                     }
// 
//                     if (empty_span)
//                         continue;               // next input point
// 
//                     // record Max_err and Max_span
//                     Max_err = max_err;
//                     for (auto k = 0; k < dom_dim; k++)
//                         Max_span[k] = span[k];
//                 }   // max_err > err_limit
//             }   // for all input points
// 
//             // debug: hard code span
// //             if (Max_err >= 0.0)
// //             {
// //                 for (auto k = 0; k < dom_dim; k++)
// //                     Max_span[k] = 4;
// //             }
// 
//             if (Max_err >= 0.0)
//             {
//                 for (auto k = 0; k < dom_dim; k++)
//                 {
//                     // span should never be the last knot because of the repeated knots at end
//                     assert(Max_span[k] < mfa_data.tmesh.all_knots[k].size() - 1);
// 
//                     // new knot is the midpoint of the span containing the domain point parameters
//                     T new_knot_val = (mfa_data.tmesh.all_knots[k][Max_span[k]] + mfa_data.tmesh.all_knots[k][Max_span[k] + 1]) / 2.0;
//                     new_knots[k].push_back(new_knot_val);
//                     new_levels[k].push_back(iter + 1);  // adapt at the next level, for now every iteration is a new level
//                 }
// 
//                 if (local)
//                     parent_tensor_idx = InsertKnots(new_knots,
//                                                     new_levels,
//                                                     inserted_knot_idxs,
//                                                     new_nctrl_pts,
//                                                     new_ctrl_pts,
//                                                     new_weights);
//                 else
//                     parent_tensor_idx = InsertKnots(new_knots,
//                                                     new_levels,
//                                                     inserted_knot_idxs);
// 
//                 return false;
//             }   // Max_err >= 0.0
//             return true;
//         }

        // for debugging: checks all knot spans for at least one input point and for nondecreasing order
        // returns true if all spans check out
        bool CheckAllSpans()
        {
            // typing shortcuts
            Tmesh<T>&                   tmesh                   = mfa_data.tmesh;
            vector<vector<T>>&          all_knots               = tmesh.all_knots;
            vector<vector<ParamIdx>>&   all_knot_param_idxs     = tmesh.all_knot_param_idxs;
            int&                        dom_dim                 = mfa_data.dom_dim;
            VectorXi&                   p                       = mfa_data.p;

            for (auto k = 0; k < dom_dim; k++)
            {
                for (auto j = p(k); j < all_knots[k].size() - p(k) - 1; j++)
                {
                    size_t min = mfa_data.tmesh.all_knot_param_idxs[k][j];
                    size_t max = mfa_data.tmesh.all_knot_param_idxs[k][j + 1];
                    T min_param = input.params->param_grid[k][min];
                    T max_param = input.params->param_grid[k][max];

                    if (all_knots[k][j] > all_knots[k][j + 1])
                    {
                        fmt::print(stderr, "CheckAllSpans(): Error: knots are out of order (should be monotone nondecreasing)\n");
                        fmt::print(stderr, "span [{} - {}]\n", all_knots[k][j], all_knots[k][j + 1]);
                    }
                    if (max - min <= 0)
                    {
                        cerr << "CheckAllSpans(): Error: dim " << k << " span " << j << " does not have an input point" << endl;
                        fmt::print(stderr, "span [{} - {}] min {} max {} min_param {} max_param {}\n",
                                all_knots[k][j], all_knots[k][j + 1], min, max, min_param, max_param);
                        return false;
                    }
                    if (min_param < all_knots[k][j])
                    {
                        cerr << "CheckAllSpans(): Error: dim " << k << " span " << j << " min param < range of knot span. This should not happen.\n" << endl;
                        fmt::print(stderr, "span [{} - {}] min {} max {} min_param {} max_param {}\n",
                                all_knots[k][j], all_knots[k][j + 1], min, max, min_param, max_param);
                        return false;
                    }
                    if (min_param >= all_knots[k][j + 1])
                    {
                        cerr << "CheckAllSpans(): Error: dim " << k << " span " << j << " min param > range of knot span. This should not happen.\n" << endl;
                        fmt::print(stderr, "span [{} - {}] min {} max {} min_param {} max_param {}\n",
                                all_knots[k][j], all_knots[k][j + 1], min, max, min_param, max_param);
                        return false;
                    }
                    if (max < input.params->param_grid[k].size() - 1 && max_param < all_knots[k][j + 1])
                    {
                        cerr << "CheckAllSpans(): Error: dim " << k << " span " << j << " max param < range of next knot span. This should not happen.\n" << endl;
                        fmt::print(stderr, "span [{} - {}] min {} max {} min_param {} max_param {}\n",
                                all_knots[k][j], all_knots[k][j + 1], min, max, min_param, max_param);
                        return false;
                    }
                }
            }
            return true;
        }

        // computes error in knot spans and finds all new knots (in all dimensions at once) that should be inserted at one level
        // returns true if all done, ie, no new knots inserted
        bool AllErrorSpans(
                int                         parent_level,           // level of parent tensor to check
                VectorX<T>                  extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                T                           err_limit,              // max. allowed error, assumed to be normalized
                bool                        saved_basis,            // whether basis functions were saved and can be re-used
                vector<TensorIdx>&          parent_tensor_idxs,     // (output) idx of parent tensor of each new knot to be inserted
                vector<vector<KnotIdx>>&    new_knot_idxs,          // (output) indices in each dim. of (unique) new knots in full knot vector after insertion
                vector<vector<T>>&          new_knots,              // (output) knot values in each dim. of knots to be inserted (unique)
                ErrorStats<T>&              error_stats)            // (output) error statistics
        {
            bool retval                     = true;
            error_stats.max_abs_err         = 0.0;
            error_stats.max_norm_err        = 0.0;
            error_stats.sum_sq_abs_errs     = 0.0;
            error_stats.sum_sq_norm_errs    = 0.0;

            VectorXi            derivs;                             // size 0 means unused
            DecodeInfo<T>       decode_info(mfa_data, derivs);      // reusable decode point info for calling VolPt repeatedly

            // typing shortcuts
            Tmesh<T>&                   tmesh                   = mfa_data.tmesh;
            vector<vector<T>>&          all_knots               = tmesh.all_knots;
            vector<vector<int>>&        all_knot_levels         = tmesh.all_knot_levels;
            vector<vector<ParamIdx>>&   all_knot_param_idxs     = tmesh.all_knot_param_idxs;
            int&                        dom_dim                 = mfa_data.dom_dim;
            VectorXi&                   p                       = mfa_data.p;

            // debug
//             fprintf(stderr, "*** Using AllErrorSpans ***\n");
//             fmt::print(stderr, "Tmesh in AllErrorSpans\n\n");
//             tmesh.print(true, false, true);

            Decoder<T>          decoder(mfa_data, 1);
            VectorX<T>          param(dom_dim);                             // parameters of domain point
            VectorX<T>          cpt(tmesh.tensor_prods[0].ctrl_pts.cols()); // decoded point

            // new knot when splitting a knot span
            vector<KnotIdx> new_knot_idx(dom_dim);
            vector<T>       new_knot_val(dom_dim);

            vector<size_t> nnew_knots(dom_dim, 0);                          // number of new knots inserted so far in each dim.

            if (!extents.size())
                extents = VectorX<T>::Ones(input.pt_dim);

            // parameters for vol iterator over knot spans in a tensor product and parameters in a knot span
            VectorXi sub_npts(dom_dim);
            VectorXi sub_starts(dom_dim);
            VectorXi all_npts(dom_dim);
            VectorXi span_ijk(dom_dim);
            VectorXi param_ijk(dom_dim);

            VolIterator dom_iter(input.ndom_pts);                           // iterator over input domain points

            for (auto tidx = 0; tidx < tmesh.tensor_prods.size(); tidx++)   // for all tensors
            {
                // debug
//                 cerr << "tensor " << tidx << endl;

                TensorProduct<T>& t = tmesh.tensor_prods[tidx];

                if (t.level > parent_level || t.done)
                    continue;

                bool tensor_done    = true;                         // no new knots added in the current tensor

                // setup vol iterator over knot spans

                // adjust range of knot spans to include interior spans with input points, skipping repeated knots at global edges
                for (auto j = 0; j < dom_dim; j++)
                {
                    KnotIdx min = t.knot_mins[j] == 0 ? p(j) : 0;
                    KnotIdx max = t.knot_maxs[j] == all_knots[j].size() - 1 ?
                        t.knot_idxs[j].size() - p(j) - 1 : t.knot_idxs[j].size() - 1;

                    sub_npts(j)     = max - min;
                    all_npts(j)     = t.knot_idxs[j].size() - 1;
                    sub_starts(j)   = min;
                }

                // debug
//                 cerr << "AllErrorSpans(): tidx " << tidx  << " sub_npts: " << sub_npts.transpose() << " all_npts: " << all_npts.transpose() << " sub_starts: " << sub_starts.transpose() << endl;

                VolIterator span_iter(sub_npts, sub_starts, all_npts);

                // iterate over knot spans
                while (!span_iter.done())
                {
                    span_iter.idx_ijk(span_iter.cur_iter(), span_ijk);

                    // debug
//                     cerr << "span_ijk: " << span_ijk.transpose() << endl;

                    // setup vol iterator over parameters inside of current knot span
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        KnotIdx min_knot_idx    = t.knot_idxs[j][span_ijk(j)];                      // knot idx at start of span
                        KnotIdx max_knot_idx    = t.knot_idxs[j][span_ijk(j) + 1];                  // knot idx at end of span
                        ParamIdx min_param_idx  = all_knot_param_idxs[j][min_knot_idx];             // first parameter index of the knot span
                        ParamIdx max_param_idx  = all_knot_param_idxs[j][max_knot_idx] - 1;         // last parameter index of the knot span
                        if (max_param_idx == input.params->param_grid[j].size() - 1)                // include last parameter in last knot span
                            max_param_idx++;

                        // debug
//                         cerr << "AllErrorSpans(): tidx " << tidx << " j: " << j << " min_knot_idx: " << min_knot_idx << " max_knot_idx: " << max_knot_idx <<
//                         " min_param_idx: " << min_param_idx << " max_param_idx: " << max_param_idx << endl;

                        sub_npts(j)     = max_param_idx - min_param_idx;
                        all_npts(j)     = max_param_idx;
                        sub_starts(j)   = min_param_idx;
                    }
                    VolIterator param_iter(sub_npts, sub_starts, all_npts);

                    // debug
//                     cerr << "dom_iter sub_npts: "   << sub_npts.transpose()
//                          << " all_npts: "           << all_npts.transpose()
//                          << " sub_starts: "         << sub_starts.transpose() << endl;

                    // iterate over input domain points
                    while (!param_iter.done())
                    {
                        param_iter.idx_ijk(param_iter.cur_iter(), param_ijk);

                        // debug
//                         cerr << "param_ijk: " << param_ijk.transpose() << endl;

                        // decode a point at the input point parameters
                        for (auto j = 0; j < dom_dim; j++)
                            param(j) = input.params->param_grid[j][param_ijk(j)];

#ifdef MFA_TMESH
                        decoder.VolPt_tmesh(param, cpt);
#else
                        if (saved_basis)
                            decoder.VolPt_saved_basis(param_ijk, param, cpt, decode_info, t);
                        else
                            decoder.VolPt(param, cpt, decode_info, t);
#endif

                        // error between decoded point and input point
                        size_t dom_idx = dom_iter.ijk_idx(param_ijk);
                        T max_abs_err   = 0.0;                    // max over dims. of one point of absolute error
                        T max_norm_err  = 0.0;                    // max over dims. of one point of normalized error
                        for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
                        {
                            T abs_err   = fabs(cpt(j) - input.domain(dom_idx, mfa_data.min_dim + j));
                            T norm_err  = abs_err / extents(mfa_data.min_dim + j);
                            max_abs_err = abs_err > max_abs_err ? abs_err : max_abs_err;                            // max over dims. of current point
                            max_norm_err = norm_err > max_norm_err ? norm_err : max_norm_err;                       // max over dims. of current point
                            error_stats.max_abs_err = max_abs_err > error_stats.max_abs_err ? max_abs_err : error_stats.max_abs_err;        // max over all points
                            error_stats.max_norm_err = max_norm_err > error_stats.max_norm_err ? max_norm_err : error_stats.max_norm_err;   // max over all points
                            error_stats.sum_sq_abs_errs += max_abs_err * max_abs_err;
                            error_stats.sum_sq_norm_errs += max_norm_err * max_norm_err;
                        }

                        if (max_norm_err > err_limit)               // assumes err_limit is normalized
                        {
                            // debug
//                             cerr << "tensor tidx " << tidx << " has error above limit in span_ijk: " << span_ijk.transpose() << " param_ijk: " << param_ijk.transpose() << " dom_idx: " << dom_idx << endl;
//                             cerr << "param: " << param.transpose() << " decoded pt: " << cpt.transpose() << " input pt: " << input.domain.row(dom_idx) << endl;

                            if (valid_split_span_local(span_ijk, t, nnew_knots, new_knot_idx, new_knot_val))      // splitting span will have input points
                            {
                                // record new knot to be inserted
                                for (auto j = 0; j < dom_dim; j++)
                                {
                                    new_knot_idxs[j].push_back(new_knot_idx[j]);
                                    new_knots[j].push_back(new_knot_val[j]);
                                    nnew_knots[j]++;
                                }
                                parent_tensor_idxs.push_back(tidx);
                                retval      = false;
                                tensor_done = false;

                                // debug
//                                 cerr << "inserting knot in tensor tidx " << tidx << " span_ijk: " << span_ijk.transpose() << endl;
//                                 fmt::print(stderr, "new_knot_idx [{}] new_knot_val [{}]\n", fmt::join(new_knot_idx, ","), fmt::join(new_knot_val, ","));
                            }
//                             else
//                                 // debug
//                                 cerr << "tensor tidx " << tidx << " has error above limit in span_ijk: " << span_ijk.transpose() << " but is empty of input points" << endl;
                            break;
                        }

                        param_iter.incr_iter();
                    }   // iterator over domain points in a knot span

                    span_iter.incr_iter();
                }   // iterator over knot spans

                if (tensor_done)
                    t.done = true;
            }   // for all tensors

            // debug
//             for (auto j = 0; j < dom_dim; j++)
//                 fmt::print(stderr, "AllErrorSpans(): dim {} new_knot_idxs[{}] = [{}]\nnew_knots[{}] = [{}]\n",
//                         j, j, fmt::join(new_knot_idxs[j], ","), j, fmt::join(new_knots[j], ","));
//             fmt::print(stderr, "AllErrorSpans(): parent_tensor_idxs [{}]\n", fmt::join(parent_tensor_idxs,","));

            return retval;
        }

//         // DEPRECATE and use valid_split_span_local instead
//         // checks whether splitting a knot span will be empty of input points in all dimensions of splitting
//         // if the return value is false (an empty, invalid split in all dims), then new_knot_idx and new_knot_val are invalid
//         // if the return value is true (a valid split in one or more dims), then new_knot_idx and new_knot_val can be used
//         // new_knot_idx and new_knot_val allocated by caller, size not checked here
//         bool valid_split_span(VectorXi&             span,           // indices of knot span in all dims
//                               TensorProduct<T>&     t,              // current tensor
//                               const vector<size_t>& nnew_knots,     // number of new knots inserted so far in each dim.
//                               vector<KnotIdx>&      new_knot_idx,   // (output) index of new knot in all dims of all_knots
//                               vector<T>&            new_knot_val)   // (output) value of new knot in all dims
//         {
//             bool debug = false;
// 
//             // typing shortcuts
//             Tmesh<T>&                   tmesh                   = mfa_data.tmesh;
//             vector<vector<T>>&          all_knots               = tmesh.all_knots;
//             vector<vector<int>>&        all_knot_levels         = tmesh.all_knot_levels;
//             vector<vector<ParamIdx>>&   all_knot_param_idxs     = tmesh.all_knot_param_idxs;
//             int&                        dom_dim                 = mfa_data.dom_dim;
// 
//             bool retval = false;
//             for (auto k = 0; k < dom_dim; k++)
//             {
//                 // in case of single tensor, don't allow more control points than input points
//                 // only for structured data for now
//                 if (tmesh.tensor_prods.size() == 1 &&
//                         t.nctrl_pts(k) + nnew_knots[k] >= input.ndom_pts(k))
//                 {
//                     new_knot_idx[k] = span(k);
//                     new_knot_val[k] = all_knots[k][span(k)];
//                     continue;
//                 }
// 
//                 // skip higher level knots to get right edge of span
//                 KnotIdx next_span;
//                 tmesh.knot_idx_ofst(t, span(k), 1, k, false, next_span);
// 
// //                 if (debug )
// //                     fmt::print(stderr, "0: dim {} tensor level {} span {} knot {} next_span {} next_knot {}\n",
// //                             k, t.level, span(k), all_knots[k][span(k)], next_span, all_knots[k][next_span]);
// 
//                 // current span must contain at least two input points
//                 size_t low_idx  = all_knot_param_idxs[k][span(k)];
//                 size_t high_idx = all_knot_param_idxs[k][next_span];
// 
//                 if (high_idx - low_idx < 2)
//                 {
//                     new_knot_idx[k] = span(k);
//                     new_knot_val[k] = all_knots[k][span(k)];
//                     continue;
//                 }
// 
//                 bool split_span;
//                 // check if an existing knot already splits the span (at a deeper level of refinement)
//                 // if so, use it
//                 if (next_span - span(k) > 1)
//                 {
//                     new_knot_idx[k] = (next_span + span(k)) / 2;
//                     new_knot_val[k] = all_knots[k][new_knot_idx[k]];
//                     split_span = false;
//                 }
// 
//                 // otherwise insert a new knot
//                 else
//                 {
//                     // new knot value would is the midpoint of the span
//                     new_knot_val[k] = (all_knots[k][span(k)] + all_knots[k][next_span]) / 2.0;
//                     // new knot index found by keeping all_knots sorted by knot value
//                     new_knot_idx[k] = span(k);
//                     int i = span(k);
//                     while (new_knot_val[k] > all_knots[k][i])
//                         new_knot_idx[k] = ++i;
//                     split_span = true;
//                 }
// 
//                 // if the current span were to be split, check whether the resulting spans will have an input point
//                 if (split_span)
//                 {
//                     ParamIdx param_idx  = low_idx;
//                     while (input.params->param_grid[k][param_idx] < new_knot_val[k])
//                         param_idx++;
// 
//                     // check spans of immediate neighboring knots for input points
//                     // so that every span at the finest level always has input
//                     low_idx     = all_knot_param_idxs[k][new_knot_idx[k] - 1];
//                     high_idx    = all_knot_param_idxs[k][new_knot_idx[k]];
// 
//                     if (param_idx - low_idx == 0 || high_idx - param_idx == 0)
//                     {
//                         new_knot_idx[k] = span(k);
//                         new_knot_val[k] = all_knots[k][span(k)];
//                         continue;
//                     }
//                 }
// 
//                 retval |= true;
//             }
// 
//             return retval;
//         }

        // checks whether splitting a knot span will be empty of input points in all dimensions of splitting
        // knot span is the span in the local tensor knot_idxs, not the global all_knot_idxs
        // if the return value is false (an empty, invalid split in all dims), then new_knot_idx and new_knot_val are invalid
        // if the return value is true (a valid split in one or more dims), then new_knot_idx and new_knot_val can be used
        // new_knot_idx and new_knot_val allocated by caller, size not checked here
        bool valid_split_span_local(
                VectorXi&               span,           // indices of knot span in all dims
                TensorProduct<T>&       t,              // current tensor
                const vector<size_t>&   nnew_knots,     // number of new knots inserted so far in each dim.
                vector<KnotIdx>&        new_knot_idx,   // (output) index of new knot in all dims of all_knots
                vector<T>&              new_knot_val)   // (output) value of new knot in all dims
        {
            bool debug = false;

            // typing shortcuts
            Tmesh<T>&                   tmesh                   = mfa_data.tmesh;
            vector<vector<T>>&          all_knots               = tmesh.all_knots;
            vector<vector<int>>&        all_knot_levels         = tmesh.all_knot_levels;
            vector<vector<ParamIdx>>&   all_knot_param_idxs     = tmesh.all_knot_param_idxs;
            int&                        dom_dim                 = mfa_data.dom_dim;

            bool retval = false;
            for (auto k = 0; k < dom_dim; k++)
            {
                KnotIdx cur_span    = t.knot_idxs[k][span(k)];
                KnotIdx next_span   = t.knot_idxs[k][span(k) + 1];

                // in case of single tensor, don't allow more control points than input points
                // only for structured data for now
                if (tmesh.tensor_prods.size() == 1 &&
                        t.nctrl_pts(k) + nnew_knots[k] >= input.ndom_pts(k))
                {
                    new_knot_idx[k] = cur_span;
                    new_knot_val[k] = all_knots[k][cur_span];
                    continue;
                }

                // current span must contain at least two input points
                size_t low_idx  = all_knot_param_idxs[k][cur_span];
                size_t high_idx = all_knot_param_idxs[k][next_span];

                if (high_idx - low_idx < 2)
                {
                    new_knot_idx[k] = cur_span;
                    new_knot_val[k] = all_knots[k][cur_span];
                    continue;
                }

                bool split_span;
                // check if an existing knot already splits the span (at a deeper level of refinement)
                // if so, use it
                if (next_span - cur_span > 1)
                {
                    new_knot_idx[k] = (next_span + cur_span) / 2;
                    new_knot_val[k] = all_knots[k][new_knot_idx[k]];
                    split_span = false;
                }

                // otherwise insert a new knot
                else
                {
                    // new knot value would is the midpoint of the span
                    new_knot_val[k] = (all_knots[k][cur_span] + all_knots[k][next_span]) / 2.0;
                    // new knot index found by keeping all_knots sorted by knot value
                    new_knot_idx[k] = cur_span;
                    int i = cur_span;
                    while (new_knot_val[k] > all_knots[k][i])
                        new_knot_idx[k] = ++i;
                    split_span = true;
                }

                // if the current span were to be split, check whether the resulting spans will have an input point
                if (split_span)
                {
                    ParamIdx param_idx  = low_idx;
                    while (input.params->param_grid[k][param_idx] < new_knot_val[k])
                        param_idx++;

                    // check spans of immediate neighboring knots for input points
                    // so that every span at the finest level always has input
                    low_idx     = all_knot_param_idxs[k][new_knot_idx[k] - 1];
                    high_idx    = all_knot_param_idxs[k][new_knot_idx[k]];

                    if (param_idx - low_idx == 0 || high_idx - param_idx == 0)
                    {
                        new_knot_idx[k] = cur_span;
                        new_knot_val[k] = all_knots[k][cur_span];
                        continue;
                    }
                }

                retval |= true;
            }

            return retval;
        }

        // computes error in knot spans and returns first new knot (in all dimensions at once) that should be inserted
        // returns true if all done, ie, no new knots inserted
        // TODO: lots of optimizations possible; this is completely naive so far
        // optimizations:
        // each time called, resume search at next domain point
        // step through domain points fineer each time the end is reached
        // increment ijk in the loop over domain points instead of calling idx2ijk
        // TBB? (currently serial)
        bool FirstErrorSpan(
                       VectorX<T>                  extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                       T                           err_limit,              // max. allowed error
                       int                         iter,                   // iteration number
                       const VectorXi&             nctrl_pts,              // number of control points
                       vector<vector<KnotIdx>>&    inserted_knot_idxs,     // (output) indices in each dim. of inserted knots in full knot vector after insertion
                       int&                        parent_tensor_idx)      // (output) parent tensor containing new knot to be inserted (assuming only one insertion)
       {
           Decoder<T>          decoder(mfa_data, 1);
           VectorXi            ijk(dom_dim);                   // i,j,k of domain point
           VectorX<T>          param(dom_dim);                 // parameters of domain point
           vector<int>         span(dom_dim);                  // knot span in each dimension
           VectorXi            derivs;                             // size 0 means unused
           DecodeInfo<T>       decode_info(mfa_data, derivs);      // reusable decode point info for calling VolPt repeatedly
           vector<vector<T>>   new_knots(dom_dim);             // new knots
           vector<vector<int>> new_levels(dom_dim);            // new knot levels

           if (!extents.size())
               extents = VectorX<T>::Ones(input.pt_dim);

            for (auto pt_it = input.begin(), pt_end = input.end(); pt_it != pt_end; ++pt_it)
            {
               pt_it.params(param);
               VectorX<T> cpt(input.pt_dim);                      // approximated point

#ifdef MFA_TMESH
                decoder.VolPt_tmesh(param, cpt);
#else
                decoder.VolPt(param, cpt, decode_info, mfa_data.tmesh.tensor_prods[0]);
#endif

               int last = input.pt_dim - 1;                       // range coordinate

               // error
               T max_err = 0.0;
               for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
               {
                   T err = fabs(cpt(j) - input.domain(pt_it.idx(), mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
                   max_err = err > max_err ? err : max_err;
               }

               if (max_err > err_limit)
               {
                   for (auto k = 0; k < dom_dim; k++)
                   {
                       span[k] = mfa_data.FindSpan(k, param(k), nctrl_pts(k));

                       // span should never be the last knot because of the repeated knots at end
                       assert(span[k] < mfa_data.tmesh.all_knots[k].size() - 1);

                       // new knot is the midpoint of the span containing the domain point parameters
                       T new_knot_val = (mfa_data.tmesh.all_knots[k][span[k]] + mfa_data.tmesh.all_knots[k][span[k] + 1]) / 2.0;
                       new_knots[k].push_back(new_knot_val);
                       new_levels[k].push_back(iter + 1);  // adapt at the next level, for now every iteration is a new level
                   }
                   parent_tensor_idx = InsertKnots(new_knots, new_levels, inserted_knot_idxs, parent_tensor_idx);
                   return false;
               }
           }
           return true;
       }

        // TODO: not ported to tmesh yet, remove nnew_knots and change new_knots to vector<vector<T>>
//         //
//         // computes error in knot spans
//         // marks the knot spans that are done (error <= max_error in the entire span)
//         // assumes caller allocated new_knots to number of spans and nnew_knots to domain dimensions
//         // (does no resizing of new_knots and nnew_knots) and zeroed nnew_knots
//         // returns true if all done, ie, no new knots inserted
//         bool ErrorSpans(
//                 VectorXi&      nnew_knots,                  // number of new knots in each dim
//                 vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
//                 T              err_limit,                   // max. allowed error
//                 int            iter)                        // iteration number
//         {
// 
// #ifdef MFA_TBB                                          // TBB version
// 
//             Decoder<T> decoder(mfa, 1);
// 
//             // initialize all knot spans to not done
//             for (auto i = 0; i < mfa.knot_spans.size(); i++)
//                 mfa.knot_spans[i].done = false;
// 
//             // spans that have already been split in this round (to prevent splitting twice)
//             vector<bool> split_spans(mfa.knot_spans.size());                // intialized to false by default
// 
//             parallel_for(size_t(0), mfa.knot_spans.size(), [&] (size_t i)          // knot spans
//                     {
//                     if (!mfa.knot_spans[i].done)
//                     {
//                     size_t nspan_pts = 1;                                   // number of domain points in the span
//                     for (auto k = 0; k < mfa.p.size(); k++)
//                     nspan_pts *= (mfa.knot_spans[i].max_param_ijk(k) - mfa.knot_spans[i].min_param_ijk(k) + 1);
// 
//                     VectorXi p_ijk = mfa.knot_spans[i].min_param_ijk;           // indices of current parameter in the span
//                     VectorX<T> param(mfa.p.size());                               // value of current parameter
//                     bool span_done = true;                                  // span is done until error > err_limit
// 
//                     // TODO:  consider binary search of the points in the span?
//                     // (error likely to be higher in the center of the span?)
//                     for (auto j = 0; j < nspan_pts; j++)                    // parameters in the span
//                     {
//                     for (auto k = 0; k < mfa.p.size(); k++)
//                     param(k) = mfa.params(mfa.po[k] + p_ijk(k));
// 
//                     // approximate the point and measure error
//                     size_t idx;
//                     mfa.ijk2idx(p_ijk, idx);
//                     VectorX<T> cpt(mfa.ctrl_pts.cols());       // approximated point
//                     decoder.VolPt(param, cpt);
// 
// //                     T err = fabs(mfa.NormalDistance(cpt, idx)) / mfa.range_extent;     // normalized by data range
// 
//                     // range error
//                     int last = mfa.domain.cols() - 1;           // range coordinate
//                     T err = fabs(cpt(last) - mfa.domain(i, last)) / mfa.range_extent;
// 
//                     // span is not done
//                     if (err > err_limit)
//                     {
//                         span_done = false;
//                         break;
//                     }
// 
//                     // increment param ijk
//                     for (auto k = 0; k < mfa.p.size(); k++)                 // dimensions in the parameter
//                     {
//                         if (p_ijk(k) < mfa.knot_spans[i].max_param_ijk(k))
//                         {
//                             p_ijk(k)++;
//                             break;
//                         }
//                         else
//                             p_ijk(k) = mfa.knot_spans[i].min_param_ijk(k);
//                     }                                                   // dimension in parameter
//                     }                                                       // parameters in the span
// 
//                     if (span_done)
//                         mfa.knot_spans[i].done = true;
//                     }                                                           // knot span not done
//                     });                                                           // knot spans
// 
//             // split spans that are not done
//             auto norig_spans = mfa.knot_spans.size();
//             bool new_knot_found = false;
//             for (auto i = 0; i < norig_spans; i++)
//             {
//                 if (!mfa.knot_spans[i].done && !split_spans[i])
//                 {
//                     new_knots.resize(1);
//                     nnew_knots = VectorXi::Zero(mfa.p.size());
//                     SplitSpan(i, nnew_knots, new_knots, iter, split_spans);
//                     if (nnew_knots.sum())
//                     {
//                         new_knot_found = true;
//                         InsertKnots(nnew_knots, new_knots);
//                     }
//                 }
//             }
// 
//             // debug
//             //     for (auto i = 0; i < mfa.knot_spans.size(); i++)                  // knot spans
//             //     {
//             //         cerr <<
//             //             "span_idx="          << i                           <<
//             //             "\nmin_knot_ijk:\n"  << mfa.knot_spans[i].min_knot_ijk  <<
//             //             "\nmax_knot_ijk:\n"  << mfa.knot_spans[i].max_knot_ijk  <<
//             //             "\nmin_knot:\n"      << mfa.knot_spans[i].min_knot      <<
//             //             "\nmax_knot:\n"      << mfa.knot_spans[i].max_knot      <<
//             //             "\nmin_param_ijk:\n" << mfa.knot_spans[i].min_param_ijk <<
//             //             "\nmax_param_ijk:\n" << mfa.knot_spans[i].max_param_ijk <<
//             //             "\nmin_param:\n"     << mfa.knot_spans[i].min_param     <<
//             //             "\nmax_param:\n"     << mfa.knot_spans[i].max_param     <<
//             //             "\n"                 << endl;
//             //     }
// 
//             return !nnew_knots.sum();
// 
// #endif               // end TBB version
//
// #ifdef MFA_SERIAL    // serial version
// 
//             Decoder<T> decoder(mfa, 1);
// 
//             // initialize all knot spans to not done
//             for (auto i = 0; i < mfa.knot_spans.size(); i++)
//                 mfa.knot_spans[i].done = false;
// 
//             // spans that have already been split in this round (to prevent splitting twice)
//             vector<bool> split_spans(mfa.knot_spans.size());                // intialized to false by default
// 
//             for (auto i = 0; i < mfa.knot_spans.size(); i++)                // knot spans
//             {
//                 size_t nspan_pts = 1;                                       // number of domain points in the span
//                 for (auto k = 0; k < mfa.p.size(); k++)
//                     nspan_pts *= (mfa.knot_spans[i].max_param_ijk(k) - mfa.knot_spans[i].min_param_ijk(k) + 1);
// 
//                 VectorXi p_ijk = mfa.knot_spans[i].min_param_ijk;           // indices of current parameter in the span
//                 VectorX<T> param(mfa.p.size());                               // value of current parameter
//                 bool span_done = true;                                      // span is done until error > err_limit
// 
//                 // TODO:  consider binary search of the points in the span?
//                 // (error likely to be higher in the center of the span?)
//                 for (size_t j = 0; j < nspan_pts; j++)                      // parameters in the span
//                 {
//                     for (auto k = 0; k < mfa.p.size(); k++)
//                         param(k) = mfa.params(mfa.po[k] + p_ijk(k));
// 
//                     // approximate the point and measure error
//                     size_t idx;
//                     mfa.ijk2idx(p_ijk, idx);
//                     VectorX<T> cpt(mfa.ctrl_pts.cols());                      // approximated point
//                     decoder.VolPt(param, cpt);
// 
// //                     T err = fabs(mfa.NormalDistance(cpt, idx)) / mfa.range_extent;     // normalized by data range
// 
//                     // range error
//                     int last = mfa.domain.cols() - 1;           // range coordinate
//                     T err = fabs(cpt(last) - mfa.domain(i, last)) / mfa.range_extent;
// 
//                     // span is not done
//                     if (err > err_limit)
//                     {
//                         span_done = false;
//                         break;
//                     }
// 
//                     // increment param ijk
//                     for (auto k = 0; k < mfa.p.size(); k++)                 // dimensions in the parameter
//                     {
//                         if (p_ijk(k) < mfa.knot_spans[i].max_param_ijk(k))
//                         {
//                             p_ijk(k)++;
//                             break;
//                         }
//                         else
//                             p_ijk(k) = mfa.knot_spans[i].min_param_ijk(k);
//                     }                                                       // dimension in parameter
//                 }                                                           // parameters in the span
// 
//                 if (span_done)
//                     mfa.knot_spans[i].done = true;
//             }                                                               // knot spans
// 
//             // split spans that are not done
//             auto norig_spans = mfa.knot_spans.size();
//             bool new_knot_found = false;
//             for (auto i = 0; i < norig_spans; i++)
//             {
//                 if (!mfa.knot_spans[i].done && !split_spans[i])
//                 {
//                     new_knots.resize(1);
//                     nnew_knots = VectorXi::Zero(mfa.p.size());
//                     SplitSpan(i, nnew_knots, new_knots, iter, split_spans);
//                     if (nnew_knots.sum())
//                     {
//                         new_knot_found = true;
//                         InsertKnots(nnew_knots, new_knots);
//                     }
//                 }
//             }
// 
//             // debug
//             //     fprintf(stderr, "\nspans after splitting:\n-----\n");
//             //     for (auto i = 0; i < mfa.knot_spans.size(); i++)                  // knot spans
//             //         fprintf(stderr, "i=%d min_knot=[%.3f %.3f] max_knot=[%.3f %.3f]\n", i,
//             //                 mfa.knot_spans[i].min_knot(0), mfa.knot_spans[i].min_knot(1),
//             //                 mfa.knot_spans[i].max_knot(0), mfa.knot_spans[i].max_knot(1));
// 
//             return !new_knot_found;
// 
// #endif       // end serial version
// 
//         }
// 
//         // splits a knot span into two
//         // also splits all other spans sharing the same knot values
//         void SplitSpan(
//                 size_t         si,                          // id of span to split
//                 VectorXi&      nnew_knots,                  // number of new knots in each dim
//                 vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
//                 int            iter,                        // iteration number
//                 vector<bool>&  split_spans)                 // spans that have already been split in this iteration
//         {
//             // new split dimension based on alternating dimension per span
//             // check if span can be split (both halves would have domain points in its range)
//             // if not, check other split directions
//             int sd = mfa.knot_spans[si].last_split_dim;         // alternating per knot span
//             T new_knot;                                     // new knot value in the split dimension
//             size_t k;                                           // dimension
//             for (k = 0; k < mfa.p.size(); k++)
//             {
//                 sd       = (sd + 1) % mfa.p.size();
//                 new_knot = (mfa.knot_spans[si].min_knot(sd) + mfa.knot_spans[si].max_knot(sd)) / 2;
//                 if (mfa.params(mfa.po[sd] + mfa.knot_spans[si].min_param_ijk(sd)) < new_knot &&
//                         mfa.params(mfa.po[sd] + mfa.knot_spans[si].max_param_ijk(sd)) > new_knot)
//                     break;
//             }
//             if (k == mfa.p.size())                                  // a split direction could not be found
//             {
//                 mfa.knot_spans[si].done = true;
//                 split_spans[si]         = true;
// 
//                 // debug
//                 fprintf(stderr, "--- SplitSpan(): span %ld could not be split further ---\n", si);
// 
//                 return;
//             }
// 
//             // find all spans with the same min_knot_ijk as the span to be split and that are not done yet
//             // those will be split too (NB, in the same dimension as the original span to be split)
//             bool new_split = false;                             // the new knot was used to actually split a span
//             for (auto j = 0; j < split_spans.size(); j++)       // original number of spans in this round
//             {
//                 if (split_spans[j] || mfa.knot_spans[j].min_knot_ijk(sd) != mfa.knot_spans[si].min_knot_ijk(sd))
//                     continue;
// 
//                 // debug
//                 //      fprintf(stderr, "splitting span %d in sd=%d by new_knot=%.3f\n", j, sd, new_knot);
// 
//                 new_split = true;
// 
//                 // copy span to the back
//                 mfa.knot_spans.push_back(mfa.knot_spans[j]);
// 
//                 // modify old span
//                 auto pi = mfa.knot_spans[j].min_param_ijk(sd);          // one coordinate of ijk index into params
//                 if (mfa.params(mfa.po[sd] + pi) < new_knot)                 // at least one param (domain pt) in the span
//                 {
//                     while (mfa.params(mfa.po[sd] + pi) < new_knot)          // pi - 1 = max_param_ijk(sd) in the modified span
//                         pi++;
//                     mfa.knot_spans[j].last_split_dim    = sd;
//                     mfa.knot_spans[j].max_knot(sd)      = new_knot;
//                     mfa.knot_spans[j].max_param_ijk(sd) = pi - 1;
//                     mfa.knot_spans[j].max_param(sd)     = mfa.params(mfa.po[sd] + pi - 1);
// 
//                     // modify new span
//                     mfa.knot_spans.back().last_split_dim     = -1;
//                     mfa.knot_spans.back().min_knot(sd)       = new_knot;
//                     mfa.knot_spans.back().min_param_ijk(sd)  = pi;
//                     mfa.knot_spans.back().min_param(sd)      = mfa.params(mfa.po[sd] + pi);
//                     mfa.knot_spans.back().min_knot_ijk(sd)++;
// 
//                     split_spans[j] = true;
//                 }
//             }
// 
//             if (!new_split)
//                 return;
// 
//             // increment min and max knot ijk for any knots after the inserted one
//             for (auto j = 0; j < mfa.knot_spans.size(); j++)
//             {
//                 if (mfa.knot_spans[j].min_knot(sd) > mfa.knot_spans[si].max_knot(sd))
//                     mfa.knot_spans[j].min_knot_ijk(sd)++;
//                 if (mfa.knot_spans[j].max_knot(sd) > mfa.knot_spans[si].max_knot(sd))
//                     mfa.knot_spans[j].max_knot_ijk(sd)++;
//             }
// 
//             // add the new knot to nnew_knots and new_knots (only a single knot inserted at a time)
//             new_knots.resize(1);
//             new_knots[0]    = new_knot;
//             nnew_knots      = VectorXi::Zero(mfa.p.size());
//             nnew_knots(sd)  = 1;
// 
//             // debug
//             //     fprintf(stderr, "inserted new knot value=%.3f dim=%d\n", new_knot, sd);
//         }
    };
}

#endif
