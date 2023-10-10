//--------------------------------------------------------------
// Custom iterators for MFA
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_ITER_HPP
#define _MFA_ITER_HPP

#include <vector>
#include <iostream>
#include <cstdio>
#include <mfa/types.hpp>

namespace mfa
{
    // object for iterating in a flat loop over an n-dimensional volume
    // a few member functions are thread-safe (marked); rest are not
    struct VolIterator
    {
        friend struct SliceIterator;
        friend struct CurveIterator;

        // TODO these variables ideally should be const
        size_t          dom_dim_;                   // number of domain dimensions
        VectorXi        npts_dim_;                  // size of volume or subvolume in each dimension
        VectorXi        starts_dim_;                // offset to start of subvolume in each dimension
        VectorXi        all_npts_dim_;              // size of total volume in each dimension
        VectorXi        ds_;                        // stride for domain points in each dim.
        size_t          tot_iters_;                 // total number of flattened iterations

        // These variables are non-const
        VectorXi        idx_dim_;                   // current iteration number in each dimension
        VectorXi        prev_idx_dim_;              // previous iteration number in each dim., before incrementing
        size_t          cur_iter_;                  // current flattened iteration number
        vector<bool>    done_dim_;                  // whether row, col, etc. in each dimension is done

    public:

        void init(size_t idx = 0)
        {
            // sanity checks
            if (npts_dim_.size() != dom_dim_ || starts_dim_.size() != dom_dim_ || all_npts_dim_.size() != dom_dim_)
            {
                fprintf(stderr, "Error: VolIterator sizes of sub_npts, sub_starts, all_npts are not equal.\n");
                abort();
            }
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (starts_dim_(i) < 0)
                {
                    fprintf(stderr, "Error: VolIterator sub_starts[%d] < 0.\n", i);
                    abort();
                }
                if (starts_dim_(i) + npts_dim_(i) > all_npts_dim_(i))
                {
                    fprintf(stderr, "Error: VolIterator sub_starts[%d] + sub_npts[%d] > all_npts[%d].\n", i, i, i);
                    abort();
                }
            }

            ds_ = VectorXi::Ones(dom_dim_);
            for (size_t i = 1; i < dom_dim_; i++)
                ds_(i) = ds_(i - 1) * npts_dim_(i - 1);

            cur_iter_   = idx;
            idx_dim_.setZero();
            prev_idx_dim_.setZero();
            std::fill(done_dim_.begin(), done_dim_.end(), false);
            if (idx > 0)
            {
                idx_ijk(idx, idx_dim_);
                prev_idx_dim_ = idx_dim_;

                // set done_dim_ for dims which have been traversed up to this point
                for (size_t i = 0; i < dom_dim_ && ds_(i) <= idx; i++)
                {
                    if (idx_dim_(i) == starts_dim_(i))
                        done_dim_[i] = true;
                }
            }
            else
            {
                idx_dim_ = starts_dim_;
                prev_idx_dim_ = starts_dim_;
            }
        }

        // subvolume version
        VolIterator(const   VectorXi& sub_npts,             // size of subvolume in each dimension
                    const   VectorXi& sub_starts,           // offset to start of subvolume in each dimension
                    const   VectorXi& all_npts,             // size of total volume in each dimension
                            size_t idx = 0) :               // linear iteration count within subvolume
                    dom_dim_(sub_npts.size()),
                    npts_dim_(sub_npts),
                    starts_dim_(sub_starts),
                    all_npts_dim_(all_npts),
                    tot_iters_(npts_dim_.prod()),
                    idx_dim_(sub_npts.size()),
                    prev_idx_dim_(sub_npts.size()),
                    cur_iter_(idx),
                    done_dim_(sub_npts.size())                 { init(idx); }

        // full volume version
        VolIterator(const   VectorXi& npts,                 // size of volume in each dimension
                            size_t idx = 0) :               // linear iteration count within volume
                    dom_dim_(npts.size()),
                    npts_dim_(npts),
                    starts_dim_(VectorXi::Zero(npts.size())),
                    all_npts_dim_(npts),
                    tot_iters_(npts_dim_.prod()),
                    idx_dim_(npts.size()),
                    prev_idx_dim_(npts.size()),
                    cur_iter_(idx),
                    done_dim_(npts.size())                 { init(idx); }

        // null iterator
        VolIterator() :
                    dom_dim_(0),
                    tot_iters_(0)                           { }

        // copy constructor
        VolIterator(const VolIterator& other) = default;

        // move constructor
        VolIterator(VolIterator&& other) :
            VolIterator()
        {
            swap(*this, other);
        }

        // move & copy assignment (pass by value for copy-and-swap)
        VolIterator& operator=(VolIterator other)
        {
            swap(*this, other);
            return *this;
        }

        friend void swap(VolIterator& first, VolIterator& second)
        {
            std::swap(first.dom_dim_, second.dom_dim_);
            first.npts_dim_.swap(second.npts_dim_);
            first.starts_dim_.swap(second.starts_dim_);
            first.all_npts_dim_.swap(second.all_npts_dim_);
            first.ds_.swap(second.ds_);
            std::swap(first.tot_iters_, second.tot_iters_);
            first.idx_dim_.swap(second.idx_dim_);
            first.prev_idx_dim_.swap(second.prev_idx_dim_);
            std::swap(first.cur_iter_, second.cur_iter_);
            std::swap(first.done_dim_, second.done_dim_);
        }

        // reset the iterator, possibly to a given iteration count
        void reset(size_t idx = 0)      { init(idx); }

        // return total number of iterations in the volume
        // thread-safe
        size_t tot_iters() const        { return tot_iters_; }

        // return whether all iterations are done
        bool done() const              { return cur_iter_ >= tot_iters_; }

        // return current index in a dimension
        // in case of a subvolume, index is w.r.t. entire volume
        int idx_dim(int dim) const      { return idx_dim_[dim]; }

        // return vector of indices in each dimension
        // in case of a subvolume, index is w.r.t entire volume
        VectorXi idx_dim() const        { return idx_dim_; }

        // return previous index, what it was before incrementing, in a dimension
        // in case of a subvolume, index is w.r.t. entire volume
        int prev_idx_dim(int dim) const   { return prev_idx_dim_[dim]; }

        // return whether a row, col, etc. in a dimension is done
        // call after incr_iter(), not before, because incr_iter()
        // updates the done flags
        bool done(int dim) const          { return done_dim_[dim]; }

        // return current total iteration count
        size_t cur_iter() const       { return cur_iter_; }

        // convert linear domain point index into (i,j,k,...) multidimensional index
        // in case of subvolume, idx is w.r.t. subvolume but ijk is w.r.t entire volume
        // thread-safe
        void idx_ijk(
                size_t                  idx,            // linear cell index in subvolume
                VectorXi&               ijk) const      // (output) i,j,k,... indices in all dimensions
        {
            if (dom_dim_ == 1)
            {
                ijk(0) = idx + starts_dim_(0);
                return;
            }

            for (int i = 0; i < dom_dim_; i++)
            {
                if (i < dom_dim_ - 1)
                    ijk(i) = (idx % ds_[i + 1]) / ds_[i];
                else
                    ijk(i) = idx  / ds_[i];
            }
            ijk += starts_dim_;
        }

        // convert (i,j,k,...) multidimensional index into linear index into domain
        // in the case of subvolume, both ijk and idx are w.r.t. entire volume
        // thread-safe
        size_t ijk_idx(const VectorXi& ijk) const       // i,j,k,... indices to all dimensions
        {
            size_t idx          = 0;
            size_t stride       = 1;
            for (int i = 0; i < dom_dim_; i++)
            {
                idx     += ijk(i) * stride;
                stride  *= all_npts_dim_(i);
            }
            return idx;
        }

        size_t ijk_idx(const vector<int>& ijk) const
        {
            size_t idx          = 0;
            size_t stride       = 1;
            for (int i = 0; i < dom_dim_; i++)
            {
                idx     += ijk[i] * stride;
                stride  *= all_npts_dim_(i);
            }
            return idx;
        }

        // convert subvolume index into full volume index
        // thread-safe
        size_t sub_full_idx(size_t sub_idx) const
        {
            VectorXi ijk(dom_dim_);
            idx_ijk(sub_idx, ijk);
            return ijk_idx(ijk);
        }

        // return current iteration count within full volume
        size_t cur_iter_full() const
        {
            return ijk_idx(idx_dim_);
        }

        // increment iteration; user must call incr_iter() near the bottom of the flattened loop
        void incr_iter()
        {
            // save the previous state
            for (int i = 0; i < dom_dim_; i++)
                prev_idx_dim_[i] = idx_dim_[i];

            // increment the iteration, flipping to false any done flags that were true
            cur_iter_++;
            idx_dim_[0]++;
            for (int i = 0; i < dom_dim_ - 1; i++)
            {
                if (done_dim_[i] == true)
                    done_dim_[i] = false;
                else
                    break;
            }

            // check for last point, flipping any done flags to true
            bool done_prev_dims = true;                         // logical and of done state of all previous dims
            for (int i = 0; i < dom_dim_ - 1; i++)
            {
                // reset iteration for current dim and increment next dim.
                if (done_prev_dims && idx_dim_[i] - starts_dim_[i] >= npts_dim_[i])
                {
                    done_dim_[i]    = true;
                    idx_dim_[i]     = starts_dim_[i];
                    idx_dim_[i + 1]++;
                    done_dim_[i + 1] = false;
                }
                done_prev_dims &= done_dim_[i];
            }

            // special case for last dimension; prevent index overflow when previous dimension was zero size
            if (idx_dim_[dom_dim_ - 1] - starts_dim_[dom_dim_ - 1] >= npts_dim_[dom_dim_ - 1])
            {
                idx_dim_[dom_dim_ - 1] = 0;
                done_dim_[dom_dim_ - 1] = false;
            }

        }
    };  // VolIterator

    // a slice of a VolIterator missing one dimension
    struct SliceIterator
    {
        friend struct CurveIterator;

        private:

        VolIterator*        vol_iter_;          // the original full-dim vol iterator from which this slice derives
        int                 missing_dim_;       // the dimension missing in the slice
        size_t              dom_dim_;           // number of domain dimensions in original volume
        size_t              cur_iter_;          // current flattened iteration number
        VectorXi            idx_dim_;           // current index in each dimension in original volume
        size_t              tot_iters_;         // total number of iterations in slice (not original volume)
        VolIterator         sub_vol_iter_;      // subvolume iterator for the slice

        public:

        SliceIterator(VolIterator& vol_iter, int missing_dim) :
            vol_iter_(&vol_iter),
            missing_dim_(missing_dim),
            cur_iter_(0),
            dom_dim_(vol_iter_->dom_dim_)
        {
            VectorXi sub_npts       = vol_iter_->npts_dim_;
            sub_npts(missing_dim_)  = 1;
            sub_vol_iter_           = VolIterator(sub_npts, vol_iter_->starts_dim_, vol_iter_->all_npts_dim_);
            idx_dim_                = vol_iter_->starts_dim_;
            tot_iters_              = sub_vol_iter_.tot_iters();
        }

        // null iterator
        SliceIterator() :
            dom_dim_(0),
            tot_iters_(0)                       {}

        // copy constructor
        SliceIterator(const SliceIterator& other) = default;

        // move constructor
        SliceIterator(SliceIterator&& other) :
            SliceIterator()
        {
            swap(*this, other);
        }

        // move & copy assignment (pass by value for copy-and-swap)
        SliceIterator& operator=(SliceIterator other)
        {
            swap(*this, other);
            return *this;
        }

        friend void swap(SliceIterator& first, SliceIterator& second)
        {
            std::swap(first.vol_iter_, second.vol_iter_);
            std::swap(first.missing_dim_, second.missing_dim_);
            std::swap(first.dom_dim_, second.dom_dim_);
            std::swap(first.cur_iter_, second.cur_iter_);
            first.idx_dim_.swap(second.idx_dim_);
            std::swap(first.tot_iters_, second.tot_iters_);
            std::swap(first.sub_vol_iter_, second.sub_vol_iter_);
        }

        ~SliceIterator()
        {
        }

        // reset the iterator
        void reset()
        {
            // vol_iter_->reset();      D.L.: I don't think we ever want to reset the parent volIter,
            // since it may be incrementing independently after SliceIterator is constructed
            cur_iter_ = 0;
            sub_vol_iter_.reset();
            idx_dim_ = vol_iter_->starts_dim_;
        }

        // return whether all iterations in slice (not original volume) are done
        bool done() const
        {
            return cur_iter_ >= tot_iters_;
        }

        // increment iteration; user must call incr_iter() near the bottom of the flattened loop
        void incr_iter()
        {
            sub_vol_iter_.incr_iter();
            cur_iter_   = sub_vol_iter_.cur_iter();
            idx_dim_    = sub_vol_iter_.idx_dim();
        }

        // return ijk of current iterator location w.r.t. full volume
        VectorXi cur_ijk() const
        {
            return idx_dim_;
        }

        // return one dimension of ijk of current iterator location w.r.t. full volume
        int cur_ijk(int dim) const
        {
            return idx_dim_(dim);
        }

        // return current iteration count
        size_t cur_iter() const         { return cur_iter_; }

        // return total number of elements in slice
        size_t tot_iters() const        { return tot_iters_; }

        // return missing dimension (i.e. the dimension perpendicular to the slice)
        int missing_dim() const         { return missing_dim_; }

        // return total number of iterations in the slice (not in original volume)
        size_t tot_iters() const        { return tot_iters_; }

    };  // SliceIterator

    // a one-dimension curve of a VolIterator
    struct CurveIterator
    {


        SliceIterator*          slice_iter_;        // the slice iterator containing the start of this curve
        
        private:
        size_t                  dom_dim_;           // number of domain dimensions in original volume
        size_t                  cur_iter_;          // current flattened iteration number
        VectorXi                idx_dim_;           // current index in each dimension in original volume
        int                     curve_dim_;         // dimension of curve
        size_t                  tot_iters_;         // total number of iterations in curve (not original slice)

        public:

        CurveIterator(SliceIterator& slice_iter) :
            slice_iter_(&slice_iter),
            cur_iter_(0),
            curve_dim_(slice_iter_->missing_dim_),
            dom_dim_(slice_iter_->vol_iter_->dom_dim_)
        {
            tot_iters_  = slice_iter_->vol_iter_->npts_dim_(curve_dim_);
            idx_dim_    = VectorXi::Zero(dom_dim_);
        }

        // null iterator
        CurveIterator() :
                    dom_dim_(0),
                    cur_iter_(0),
                    tot_iters_(0)                           { }

        // copy constructor
        CurveIterator(const CurveIterator& other) = default;

        // move constructor
        CurveIterator(CurveIterator&& other) :
            CurveIterator()
        {
            swap(*this, other);
        }

        // move & copy assignment (pass by value for copy-and-swap)
        CurveIterator& operator=(CurveIterator other)
        {
            swap(*this, other);
            return *this;
        }

        friend void swap(CurveIterator& first, CurveIterator& second)
        {
            std::swap(first.slice_iter_, second.slice_iter_);
            std::swap(first.dom_dim_, second.dom_dim_);
            std::swap(first.cur_iter_, second.cur_iter_);
            first.idx_dim_.swap(second.idx_dim_);
            std::swap(first.curve_dim_, second.curve_dim_);
            std::swap(first.tot_iters_, second.tot_iters_);
        }

        // reset the iterator
        void reset()
        {
            cur_iter_ = 0;
            idx_dim_ = VectorXi::Zero(dom_dim_);
        }

        // return whether all iterations in curve (not original volume) are done
        bool done() const              { return cur_iter_ >= tot_iters_; }

        // return ijk of current iterator location w.r.t. full volume
        VectorXi cur_ijk() const
        {
            return slice_iter_->idx_dim_ + idx_dim_;
        }

        // return one dimension of ijk of current iterator location w.r.t. full volume
        int cur_ijk(int dim) const
        {
            return slice_iter_->idx_dim_(dim) + idx_dim_(dim);
        }

        // convert (i,j,k,...) multidimensional index into linear index into domain
        // both ijk and idx are w.r.t. entire volume
        // thread-safe
        size_t ijk_idx(const VectorXi& ijk) const       // i,j,k,... indices to all dimensions
        {
            return slice_iter_->vol_iter_->ijk_idx(ijk);
        }

        int cur_iter_full()
        {
            return ijk_idx(cur_ijk());
        }

        // increment iteration; user must call incr_iter() near the bottom of the flattened loop
        void incr_iter()
        {
            cur_iter_++;
            if (idx_dim_[curve_dim_] < slice_iter_->vol_iter_->npts_dim_[curve_dim_] - 1)
                idx_dim_[curve_dim_]++;
            else
                idx_dim_[curve_dim_] = 0;
        }

        // return current total iteration count
        size_t cur_iter() const     { return cur_iter_; }

        // return curve dimension
        int curve_dim() const       { return curve_dim_; }

        // return number of points on the curve
        int tot_iters() const       { return tot_iters_; }

        // return total number of iterations in the curve (not in original slice)
        size_t tot_iters() const        { return tot_iters_; }

    };  // CurveIterator
}   // namespace mfa

#endif  // _MFA_ITER_HPP