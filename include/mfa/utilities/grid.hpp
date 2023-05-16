//--------------------------------------------------------------
// Grid traversal helper for MFA
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_GRID_HPP
#define _MFA_GRID_HPP

#include <utility>
#include <mfa/types.hpp>

namespace mfa
{
    struct GridInfo
    {
        bool                initialized{false};
        int                 dom_dim{0};
        VectorXi            ndom_pts{};
        VectorXi            ds{};
        vector<VectorXi>    co{};

        friend void swap(GridInfo& first, GridInfo& second)
        {
            swap(first.initialized, second.initialized);
            swap(first.dom_dim, second.dom_dim);
            first.ndom_pts.swap(second.ndom_pts);
            first.ds.swap(second.ds);
            swap(first.co, second.co);
        }

        void init(int dom_dim_, VectorXi& ndom_pts_ ) 
        {
            initialized = true;
            dom_dim = dom_dim_;
            ndom_pts = ndom_pts_;

            int npts = ndom_pts.prod();

            // stride for domain points in different dimensions
            ds = VectorXi::Ones(dom_dim);
            for (int k = 1; k < dom_dim; k++)
                ds(k) = ds(k - 1) * ndom_pts(k - 1);

            // offsets for curve starting (domain) points in each dimension
            co.resize(dom_dim);
            for (int k = 0; k < dom_dim; k++)
            {
                int ncurves  = npts / ndom_pts(k);    // number of curves in this dimension
                int coo      = 0;                                // co at start of contiguous sequence
                co[k].resize(ncurves);

                co[k][0] = 0;

                for (int j = 1; j < ncurves; j++)
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

        void idx2ijk(int idx, VectorXi& ijk) const
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

        int ijk2idx(const VectorXi& ijk) const
        {
            int idx          = 0;
            int stride       = 1;
            for (int k = 0; k < dom_dim; k++)
            {
                idx     += ijk(k) * stride;
                stride  *= ndom_pts(k);
            }
            return idx;
        }
    };  // GridInfo
}   // namespace mfa

#endif // _MFA_GRID_HPP