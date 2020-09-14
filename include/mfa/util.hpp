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
    // object for iterating in a flat loop over an n-dimensional volume
    struct VolIterator
    {
        private:

        int                     dom_dim_;                   // number of domain dimensions
        VectorXi                npts_dim_;                  // size of volume or subvolume in each dimension
        VectorXi                starts_dim_;                // offset to start of subvolume in each dimension
        VectorXi                all_npts_dim_;              // size of total volume in each dimension
        vector<int>             idx_dim_;                   // current iteration number in each dimension
        vector<int>             prev_idx_dim_;              // previous iteration number in each dim., before incrementing
        size_t                  cur_iter_;                  // current flattened iteration number
        size_t                  tot_iters_;                 // total number of flattened iterations
        vector<bool>            done_dim_;                  // whether row, col, etc. in each dimension is done
        vector<size_t>          ds_;                        // stride for domain points in each dim.

        void init()
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

            cur_iter_   = 0;
            tot_iters_  = npts_dim_.prod();
            idx_dim_.resize(dom_dim_);
            prev_idx_dim_.resize(dom_dim_);
            done_dim_.resize(dom_dim_);
            ds_.resize(dom_dim_, 1);
            for (auto i = 0; i < dom_dim_; i++)
            {
                idx_dim_[i]         = starts_dim_(i);
                prev_idx_dim_[i]    = starts_dim_(i);
            }
            for (size_t i = 1; i < dom_dim_; i++)
                ds_[i] = ds_[i - 1] * npts_dim_(i - 1);
        }

        public:

        // full volume version
        VolIterator(const VectorXi& npts) :                 // size of volume in each dimension
                        npts_dim_(npts),
                        all_npts_dim_(npts),
                        starts_dim_(VectorXi::Zero(npts.size()))    { init(); }

        // subvolume (slice) version
        VolIterator(const VectorXi& sub_npts,               // size of subvolume in each dimension
                    const VectorXi& sub_starts,             // offset to start of subvolume in each dimension
                    const VectorXi& all_npts) :             // size of total volume in each dimension
                        starts_dim_(sub_starts),
                        npts_dim_(sub_npts),
                        all_npts_dim_(all_npts)                     { init(); }

        // return total number of iterations in the volume
        size_t tot_iters()          { return tot_iters_; }

        // return whether all iterations are done
        bool done()                 { return cur_iter_ >= tot_iters_; }

        // return current index in a dimension
        // in case of a subvolume (slice), index is w.r.t. entire volume
        int idx_dim(int dim)        { return idx_dim_[dim]; }

        // return previous index, what it was before incrementing, in a dimension
        // in case of a subvolume (slice), index is w.r.t. entire volume
        int prev_idx_dim(int dim)   { return prev_idx_dim_[dim]; }

        // return whether a row, col, etc. in a dimension is done
        // call after incr_iter(), not before, because incr_iter()
        // updates the done flags
        bool done(int dim)          { return done_dim_[dim]; }

        // return current total iteration count
        size_t cur_iter()           { return cur_iter_; }

        // convert linear domain point index into (i,j,k,...) multidimensional index
        // in case of subvolume (slice), idx is w.r.t. subvolume but ijk is w.r.t entire volume
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

        // convert subvolume index into full volume index
        size_t sub_full_idx(size_t sub_idx) const
        {
            VectorXi ijk(dom_dim_);
            idx_ijk(sub_idx, ijk);
            return ijk_idx(ijk);
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
    };
}
#endif

