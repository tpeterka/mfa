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
        VectorXi                npts_dim_;                  // size of volume in each dimension
        vector<int>             idx_dim_;                   // current iteration number in each dimension
        vector<int>             prev_idx_dim_;              // previous iteration number in each dim., before incrementing
        size_t                  cur_iter_;                  // current flattened iteration number
        size_t                  tot_iters_;                 // total number of flattened iterations
        vector<bool>            done_dim_;                  // whether row, col, etc. in each dimension is done

        public:

        VolIterator(const VectorXi& npts) : npts_dim_(npts) // sizes of volume in each dimension
        {
            dom_dim_    = npts_dim_.size();
            cur_iter_   = 0;
            tot_iters_  = npts_dim_.prod();
            idx_dim_.resize(dom_dim_);
            prev_idx_dim_.resize(dom_dim_);
            done_dim_.resize(dom_dim_);
        }

        // return total number of iterations in the volume
        size_t tot_iters()          { return tot_iters_; }

        // return whether all iterations are done
        bool done()                 { return cur_iter_ >= tot_iters_; }

        // return current index in a dimension
        int idx_dim(int dim)        { return idx_dim_[dim]; }

        // return previous index, what it was before incrementing, in a dimension
        int prev_idx_dim(int dim)   { return prev_idx_dim_[dim]; }

        // return whether a row, col, etc. in a dimension is done
        bool done(int dim)          { return done_dim_[dim]; }

        // return current total iteration count
        size_t cur_iter()           { return cur_iter_; }

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
                if (idx_dim_[i] >= npts_dim_[i])
                {
                    done_dim_[i] = true;
                    idx_dim_[i] = 0;
                    idx_dim_[i + 1]++;
                    done_dim_[i + 1] = false;
                }
            }
        }
    };
}
#endif

