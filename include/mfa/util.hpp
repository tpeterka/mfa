//--------------------------------------------------------------
// mfa utilities
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _UTIL_HPP
#define _UTIL_HPP

namespace mfa
{
    // error statistics
    template <typename T>
        struct ErrorStats
        {
            T max_abs_err;          // max of absolute errors (absolute value)
            T max_norm_err;         // max of normalized errors (absolute value)
            T sum_sq_abs_errs;      // sum of squared absolute errors
            T sum_sq_norm_errs;     // sum of squared normalized errors

            ErrorStats()
            {
                max_abs_err         = 0.0;
                max_norm_err        = 0.0;
                sum_sq_abs_errs     = 0.0;
                sum_sq_norm_errs    = 0.0;
            }
            ErrorStats(T max_abs_err_, T max_norm_err_, T sum_sq_abs_errs_, T sum_sq_norm_errs_) :
                max_abs_err(max_abs_err_),
                max_norm_err(max_norm_err_),
                sum_sq_abs_errs(sum_sq_abs_errs_),
                sum_sq_norm_errs(sum_sq_norm_errs_)
            {}
        };

    // object for iterating in a flat loop over an n-dimensional volume
    // a few member functions are thread-safe (marked); rest are not
    struct VolIterator
    {
        template<typename>
        friend class PtIterator;

        private:

        size_t          dom_dim_;                   // number of domain dimensions
        VectorXi        npts_dim_;                  // size of volume or subvolume in each dimension
        VectorXi        starts_dim_;                // offset to start of subvolume in each dimension
        VectorXi        all_npts_dim_;              // size of total volume in each dimension
        VectorXi        ds_;                        // stride for domain points in each dim.
        size_t          tot_iters_;                 // total number of flattened iterations

        VectorXi        idx_dim_;                   // current iteration number in each dimension
        VectorXi        prev_idx_dim_;              // previous iteration number in each dim., before incrementing
        size_t          cur_iter_;                  // current flattened iteration number
        vector<bool>    done_dim_;                  // whether row, col, etc. in each dimension is done

        void init(size_t idx = 0)
        {
            dom_dim_    = npts_dim_.size();

            // sanity checks
            if (starts_dim_.size() != dom_dim_ || all_npts_dim_.size() != dom_dim_)
            {
                fprintf(stderr, "Error: VolIterator sizes of sub_npts sub_starts, all_npts are not equal.\n");
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

            // tot_iters_  = npts_dim_.prod();
            idx_dim_.resize(dom_dim_);
            prev_idx_dim_.resize(dom_dim_);
            cur_iter_   = idx;
            done_dim_.resize(dom_dim_);
            if (idx > 0)
            {
                idx_ijk(idx, idx_dim_);
                prev_idx_dim_ = idx_dim_;

                // Set done_dim_ for dims which have been traversed up to this point
                for (size_t i = 0; i < dom_dim_ && ds_(i) <= idx; i++)
                {
                    if(idx_dim_(i)==starts_dim_(i))
                        done_dim_[i] = true;
                }
            }
            else
            {
                idx_dim_ = starts_dim_;
                prev_idx_dim_ = starts_dim_;
            }
        }

        public:

        // subvolume (slice) version
        VolIterator(const   VectorXi& sub_npts,             // size of subvolume in each dimension
                    const   VectorXi& sub_starts,           // offset to start of subvolume in each dimension
                    const   VectorXi& all_npts,             // size of total volume in each dimension
                            size_t idx = 0) :               // linear iteration count within subvolume
                    dom_dim_(sub_npts.size()),
                    npts_dim_(sub_npts),
                    starts_dim_(sub_starts),
                    all_npts_dim_(all_npts),
                    tot_iters_(npts_dim_.prod())                 { init(idx); }

        VolIterator(const   VectorXi& npts,                 // size of volume in each dimension
                            size_t idx = 0) :               // linear iteration count within volume
                    dom_dim_(npts.size()),
                    npts_dim_(npts),
                    starts_dim_(VectorXi::Zero(npts.size())),
                    all_npts_dim_(npts),
                    tot_iters_(npts_dim_.prod())               { init(idx); }

        // null iterator
        VolIterator() :
                    dom_dim_(0),
                    tot_iters_(0)                                { }

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
            std::swap(first.dom_dim_, second.dom_dim_);
        }


        // return total number of iterations in the volume
        // thread-safe
        size_t tot_iters() const        { return tot_iters_; }

        // return whether all iterations are done
        bool done() const              { return cur_iter_ >= tot_iters_; }

        // return current index in a dimension
        // in case of a subvolume (slice), index is w.r.t. entire volume
        int idx_dim(int dim) const      { return idx_dim_[dim]; }

        // return vector of indices in each dimension
        // in case of a subvolume (slice), index is w.r.t entire volume
        VectorXi idx_dim() const        { return idx_dim_; }

        // return previous index, what it was before incrementing, in a dimension
        // in case of a subvolume (slice), index is w.r.t. entire volume
        int prev_idx_dim(int dim) const   { return prev_idx_dim_[dim]; }

        // return whether a row, col, etc. in a dimension is done
        // call after incr_iter(), not before, because incr_iter()
        // updates the done flags
        bool done(int dim) const          { return done_dim_[dim]; }

        // return current total iteration count
        size_t cur_iter() const       { return cur_iter_; }

        // convert linear domain point index into (i,j,k,...) multidimensional index
        // in case of subvolume (slice), idx is w.r.t. subvolume but ijk is w.r.t entire volume
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
        // in the case of subvolume (slice), both ijk and idx are w.r.t. entire volume
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
            for (int i = 0; i < dom_dim_ - 1; i++)
            {
                // reset iteration for current dim and increment next dim.
                if (idx_dim_[i] - starts_dim_[i] >= npts_dim_[i])
                {
                    done_dim_[i]    = true;
                    idx_dim_[i]     = starts_dim_[i];
                    idx_dim_[i + 1]++;
                    done_dim_[i + 1] = false;
                }
            }
        }
    };  // VolIterator

    struct GridInfo
    {
        size_t              dom_dim{0};
        VectorXi            ndom_pts{};
        VectorXi            ds{};
        vector<VectorXi>    co{};
        // bool                initialized{false};

        friend void swap(GridInfo& first, GridInfo& second)
        {
            swap(first.dom_dim, second.dom_dim);
            first.ndom_pts.swap(second.ndom_pts);
            first.ds.swap(second.ds);
            swap(first.co, second.co);
        }

        void init(size_t dom_dim_, VectorXi& ndom_pts_ ) 
        {
            dom_dim = dom_dim_;
            ndom_pts = ndom_pts_;

            size_t npts = ndom_pts.prod();

            // stride for domain points in different dimensions
            ds = VectorXi::Ones(dom_dim);
            for (size_t k = 1; k < dom_dim; k++)
                ds(k) = ds(k - 1) * ndom_pts(k - 1);

            // offsets for curve starting (domain) points in each dimension
            co.resize(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                size_t ncurves  = npts / ndom_pts(k);    // number of curves in this dimension
                size_t coo      = 0;                                // co at start of contiguous sequence
                co[k].resize(ncurves);

                co[k][0] = 0;

                for (auto j = 1; j < ncurves; j++)
                {
                    // adjust offsets for the next curve
                    if (j % ds(k))
                        co[k][j] = co[k][j - 1] + 1;
                    else
                    {
                        co[k][j] = coo + ds(k) * ndom_pts(k);
                        coo = co[k][j];
                    }
                }
            }
        }

        void idx2ijk(size_t idx, VectorXi& ijk) const
        {
            if (dom_dim == 1)
            {
                ijk(0) = idx;
                return;
            }

            for (int k = 0; k < dom_dim; k++)
            {
                if (k < dom_dim - 1)
                    ijk(k) = (idx % ds(k + 1)) / ds(k);
                else
                    ijk(k) = idx  / ds(k);
            }
        }

        size_t ijk2idx(const VectorXi& ijk) const
        {
            size_t idx          = 0;
            size_t stride       = 1;
            for (int k = 0; k < dom_dim; k++)
            {
                idx     += ijk(k) * stride;
                stride  *= ndom_pts(k);
            }
            return idx;
        }
    };  // GridInfo

}   // namespace mfa
#endif

