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

            return parent_tensor_idx;
        }

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
        // returns true if no change in knots; all tensors at the parent level are done
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
                TensorProduct<T>& t = tmesh.tensor_prods[tidx];

                if (t.level != parent_level || t.done)
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

                VolIterator span_iter(sub_npts, sub_starts, all_npts);

#if 0            // debug: turn off TBB

// #ifdef MFA_TBB      // TBB version

                // thread-local objects
                // ref: https://www.threadingbuildingblocks.org/tutorial-intel-tbb-thread-local-storage
                enumerable_thread_specific<DecodeInfo<T>>           thread_decode_info(mfa_data, derivs);   // decode info
                enumerable_thread_specific<VectorXi>                thread_param_ijk(mfa_data.dom_dim);     // multidim index of parameter
                enumerable_thread_specific<VectorXi>                thread_span_ijk(mfa_data.dom_dim);      // multidim index of knot span
                enumerable_thread_specific<VectorX<T>>              thread_cpt(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols()); // evaluated point
                enumerable_thread_specific<VectorX<T>>              thread_param(mfa_data.dom_dim);         // parameters for one point
                enumerable_thread_specific<ErrorStats<T>>           thread_error_stats(0.0, 0.0, 0.0, 0.0); // parameters for one point
                enumerable_thread_specific<vector<TensorIdx>>       thread_parent_tensor_idxs;              // idx of parent tensor of each new knot to be inserted
                enumerable_thread_specific<vector<vector<KnotIdx>>> thread_new_knot_idxs(mfa_data.dom_dim); // indices in each dim. of (unique) new knots in full knot vector after insertion
                enumerable_thread_specific<vector<vector<T>>>       thread_new_knots(mfa_data.dom_dim);     // knot values in each dim. of knots to be inserted (unique)
                enumerable_thread_specific<vector<KnotIdx>>         thread_new_knot_idx(mfa_data.dom_dim);  // new knot idx
                enumerable_thread_specific<vector<T>>               thread_new_knot_val(mfa_data.dom_dim);  // new knot value
                enumerable_thread_specific<vector<size_t>>          thread_nnew_knots(mfa_data.dom_dim, 0); // number of new knots inserted so far in each dim.
                enumerable_thread_specific<VectorXi>                thread_sub_npts(mfa_data.dom_dim);      // for defining param_iter VolIterator
                enumerable_thread_specific<VectorXi>                thread_sub_starts(mfa_data.dom_dim);    // for defining param_iter VolIterator
                enumerable_thread_specific<VectorXi>                thread_all_npts(mfa_data.dom_dim);      // for defining param_iter VolIterator

                // iterate over spans
                static affinity_partitioner ap;
                parallel_for (blocked_range<size_t>(0, span_iter.tot_iters()), [&] (blocked_range<size_t>& r)
                {
                    for (auto k = r.begin(); k < r.end(); k++)
                    {
                        span_iter.idx_ijk(k, thread_span_ijk.local());

                        // setup vol iterator over parameters inside of current knot span
                        for (auto j = 0; j < dom_dim; j++)
                        {
                            KnotIdx min_knot_idx    = t.knot_idxs[j][thread_span_ijk.local()(j)];       // knot idx at start of span
                            KnotIdx max_knot_idx    = t.knot_idxs[j][thread_span_ijk.local()(j) + 1];   // knot idx at end of span
                            ParamIdx min_param_idx  = all_knot_param_idxs[j][min_knot_idx];             // first parameter index of the knot span
                            ParamIdx max_param_idx  = all_knot_param_idxs[j][max_knot_idx] - 1;         // last parameter index of the knot span
                            if (max_param_idx == input.params->param_grid[j].size() - 1)                // include last parameter in last knot span
                                max_param_idx++;

                            thread_sub_npts.local()(j)      = max_param_idx - min_param_idx;
                            thread_all_npts.local()(j)      = max_param_idx;
                            thread_sub_starts.local()(j)    = min_param_idx;
                        }
                        VolIterator param_iter(thread_sub_npts.local(), thread_sub_starts.local(), thread_all_npts.local());

                        // iterate over input domain points
                        while (!param_iter.done())
                        {
                            param_iter.idx_ijk(param_iter.cur_iter(), thread_param_ijk.local());    // ijk of current input point

                            // parameters of current input point
                            for (auto j = 0; j < dom_dim; j++)
                                thread_param.local()(j) = input.params->param_grid[j][thread_param_ijk.local()(j)];

                            // decode the point
#ifdef MFA_TMESH
                            decoder.VolPt_tmesh(thread_param.local(), thread_cpt.local());
#else
                            if (saved_basis)
                                decoder.VolPt_saved_basis(thread_param_ijk.local(), thread_param.local(), thread_cpt.local(), thread_decode_info.local(), t);
                            else
                                decoder.VolPt(thread_param.local(), thread_cpt.local(), thread_decode_info.local(), t);
#endif

                            // error between decoded point and input point
                            size_t dom_idx = dom_iter.ijk_idx(thread_param_ijk.local());
                            T max_abs_err   = 0.0;                              // max over dims. of one point of absolute error
                            T max_norm_err  = 0.0;                              // max over dims. of one point of normalized error
                            for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
                            {
                                T abs_err       = fabs(thread_cpt.local()(j) - input.domain(dom_idx, mfa_data.min_dim + j));
                                T norm_err      = abs_err / extents(mfa_data.min_dim + j);
                                max_abs_err     = std::max(abs_err, max_abs_err);
                                max_norm_err    = std::max(norm_err, max_norm_err);
                                thread_error_stats.local().max_abs_err      = std::max(max_abs_err, thread_error_stats.local().max_abs_err);
                                thread_error_stats.local().max_norm_err     = std::max(max_norm_err, thread_error_stats.local().max_norm_err);
                                thread_error_stats.local().sum_sq_abs_errs  += max_abs_err * max_abs_err;
                                thread_error_stats.local().sum_sq_norm_errs += max_norm_err * max_norm_err;
                            }

                            if (max_norm_err > err_limit)                       // assumes err_limit is normalized
                            {
                                if (valid_split_span_local(thread_span_ijk.local(), t, thread_nnew_knots.local(), thread_new_knot_idx.local(), thread_new_knot_val.local()))    // splitting span will have input points
                                {
                                    // debug
//                                     fmt::print(stderr, "inserting new knot in span  {}\n", k);

                                    // record new knot to be inserted
                                    for (auto j = 0; j < dom_dim; j++)
                                    {
                                        thread_new_knot_idxs.local()[j].push_back(thread_new_knot_idx.local()[j]);
                                        thread_new_knots.local()[j].push_back(thread_new_knot_val.local()[j]);
                                        thread_nnew_knots.local()[j]++;
                                    }
                                    thread_parent_tensor_idxs.local().push_back(tidx);
                                    retval      = false;
                                    tensor_done = false;
                                }
                                break;
                            }

                            param_iter.incr_iter();
                        }   // iterator over domain input points in a knot span
                    }   // for k

                }, ap); // parallel for all knot spans

                // combine thread-safe error_stats
                thread_error_stats.combine_each([&](const ErrorStats<T>& err)
                {
                    error_stats.max_abs_err         = std::max(error_stats.max_abs_err, err.max_abs_err);
                    error_stats.max_norm_err        = std::max(error_stats.max_norm_err, err.max_norm_err);
                    error_stats.sum_sq_abs_errs     += err.sum_sq_abs_errs;
                    error_stats.sum_sq_norm_errs    += err.sum_sq_norm_errs;
                });

                // combine thread-safe new_knot_idxs, new_knots, parent_idxs
                thread_new_knot_idxs.combine_each([&](const vector<vector<KnotIdx>>& knot_idxs)
                {
                    for (auto j = 0; j < dom_dim; j++)
                        for (auto k = 0; k < knot_idxs[j].size(); k++)
                            new_knot_idxs[j].push_back(knot_idxs[j][k]);
                });
                thread_new_knots.combine_each([&](const vector<vector<T>>& knot_vals)
                {
                    for (auto j = 0; j < dom_dim; j++)
                        for (auto k = 0; k < knot_vals[j].size(); k++)
                            new_knots[j].push_back(knot_vals[j][k]);
                });
                thread_parent_tensor_idxs.combine_each([&](const vector<TensorIdx>& parent_idxs)
                {
                    for (auto k = 0; k < parent_idxs.size(); k++)
                    {
                        parent_tensor_idxs.push_back(parent_idxs[k]);
                        retval      = false;
                        tensor_done = false;
                    }
                });

#else               // serial version

                // debug
//                 fmt::print(stderr, "span_iter.tot_iters() = {}\n", span_iter.tot_iters());

                // iterate over knot spans
                while (!span_iter.done())
                {
                    span_iter.idx_ijk(span_iter.cur_iter(), span_ijk);

                    // setup vol iterator over parameters inside of current knot span
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        KnotIdx min_knot_idx    = t.knot_idxs[j][span_ijk(j)];                      // knot idx at start of span
                        KnotIdx max_knot_idx    = t.knot_idxs[j][span_ijk(j) + 1];                  // knot idx at end of span
                        ParamIdx min_param_idx  = all_knot_param_idxs[j][min_knot_idx];             // first parameter index of the knot span
                        ParamIdx max_param_idx  = all_knot_param_idxs[j][max_knot_idx] - 1;         // last parameter index of the knot span
                        if (max_param_idx == input.params->param_grid[j].size() - 1)                // include last parameter in last knot span
                            max_param_idx++;

                        sub_npts(j)     = max_param_idx - min_param_idx;
                        all_npts(j)     = max_param_idx;
                        sub_starts(j)   = min_param_idx;
                    }
                    VolIterator param_iter(sub_npts, sub_starts, all_npts);

                    // iterate over input domain points
                    while (!param_iter.done())
                    {
                        param_iter.idx_ijk(param_iter.cur_iter(), param_ijk);

                        // parameters of the current input point
                        for (auto j = 0; j < dom_dim; j++)
                            param(j) = input.params->param_grid[j][param_ijk(j)];

                        // decode the point
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
                            T abs_err       = fabs(cpt(j) - input.domain(dom_idx, mfa_data.min_dim + j));
                            T norm_err      = abs_err / extents(mfa_data.min_dim + j);
                            max_abs_err     = std::max(abs_err, max_abs_err);
                            max_norm_err    = std::max(norm_err, max_norm_err);
                            error_stats.max_abs_err         = std::max(max_abs_err, error_stats.max_abs_err);
                            error_stats.max_norm_err        = std::max(max_norm_err, error_stats.max_norm_err);
                            error_stats.sum_sq_abs_errs     += max_abs_err * max_abs_err;
                            error_stats.sum_sq_norm_errs    += max_norm_err * max_norm_err;
                        }

                        if (max_norm_err > err_limit)               // assumes err_limit is normalized
                        {
                            if (valid_split_span_local(span_ijk, t, nnew_knots, new_knot_idx, new_knot_val))      // splitting span will have input points
                            {
                                // debug
//                                 fmt::print(stderr, "inserting new knot in span  {}\n", span_iter.cur_iter());

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
                            }
                            break;
                        }

                        param_iter.incr_iter();
                    }   // iterator over domain points in a knot span

                    span_iter.incr_iter();
                }   // iterator over knot spans

#endif              // TBB or serial

                // debug
//                 if (tidx == 0)
//                 {
//                     fmt::print(stderr, "tensor {}:\n", tidx);
//                     fmt::print(stderr, "error_stats: max_abs {} max_norm {} sum_sq_abs {} sum_sq_norm {}\n",
//                             error_stats.max_abs_err, error_stats.max_norm_err, error_stats.sum_sq_abs_errs, error_stats.sum_sq_norm_errs);
//                     fmt::print(stderr, "nnew_knots {}\n", parent_tensor_idxs.size());

//                     for (auto j = 0; j < dom_dim; j++)
//                     {
//                         fmt::print(stderr, "dim {} new_knot_idxs [{}]\n", j, fmt::join(new_knot_idxs[j], ","));
//                         fmt::print(stderr, "dim {} new_knots [{}]\n", j, fmt::join(new_knots[j], ","));
//                     }
//                     fmt::print(stderr, "parent_tensor_idxs [{}]\n", fmt::join(parent_tensor_idxs, ","));
//                     fmt::print(stderr, "retval {} tensor_done {}\n", retval, tensor_done);
//                 }

                if (tensor_done)
                    t.done = true;
            }   // for all tensors

            return retval;
        }

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

#ifndef MFA_TMESH
#ifndef MFA_TBB

                // not for t-mesh, don't allow more control points than input points
                // only for structured data for now
                // won't work for TBB because nnew_knots needs to be global, not per thread
                if (tmesh.tensor_prods.size() == 1 &&
                        t.nctrl_pts(k) + nnew_knots[k] >= input.ndom_pts(k))
                {
                    new_knot_idx[k] = cur_span;
                    new_knot_val[k] = all_knots[k][cur_span];
                    continue;
                }

#endif
#endif

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

    };
}

#endif
