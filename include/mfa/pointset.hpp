//--------------------------------------------------------------
// mfa point set data structure
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
// 
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _POINTSET_HPP
#define _POINTSET_HPP

#include <memory>
#include <mfa/types.hpp>
#include <mfa/param.hpp>
#include <mfa/utilities/util.hpp>

namespace mfa
{
    // PointSets may have an optional grid structure imposed on them, which must
    // be set at the time of construction. A grid structure can be asserted
    // by passing a non-empty vector for 'ndom_pts,' or constructing from a
    // Params object which is structured. In a structured PointSet, the class
    // member GridInfo is initialized to be nontrivial. Once a PointSet has
    // been constructed with a particular grid, any Params that are 
    // subsequently constructed will inherit this structure. If a Param 
    // object is manually set to a structured Param object, it must have the
    // same grid structure.
    template <typename T>
    struct PointSet
    {
        // Defined during construction
        int         dom_dim;
        int         pt_dim;
        int         npts;

        // Members that track how the columns of domain correspond to different models
        VectorXi    mdims;      // Dimensionality of each model (including geometry!)
        VectorXi    dim_mins;   // Start column index for each science variable model
        VectorXi    dim_maxs;   // End column index for each science variable model

        // List of points
        MatrixX<T>  domain;

        // Parameter values corresponding to domain
        shared_ptr<Param<T>>    params;
        mutable VectorX<T>      dom_mins;
        mutable VectorX<T>      dom_maxs;
        mutable bool            bounds_cached{false};   // Flag if existing domain bounds are valid
        Bbox<T>                 bbox;

        // Optional grid data
        GridInfo    g;

        // This constructs an unfilled Param object that shares its
        // dom_dim and grid structure with the PointSet. The Param object can
        // be filled with a subsequent call to the set_###_params() methods
        PointSet(
                size_t          dom_dim_,
                const VectorXi& mdims_,
                size_t          npts_,
                const VectorXi& ndom_pts_ = VectorXi()) :
            dom_dim(dom_dim_),
            pt_dim(mdims_.sum()),
            npts(npts_),
            mdims(mdims_)
        {
            domain.resize(npts, pt_dim);

            // Fill dim_mins/maxs
            dim_mins.resize(nvars());
            dim_maxs.resize(nvars());

            if (nvars() > 0)
            {
                dim_mins[0] = geom_dim();
                dim_maxs[0] = dim_mins[0] + var_dim(0) - 1;
            }

            for (int k = 1; k < nvars(); k++)
            {
                dim_mins[k] = dim_maxs[k-1] + 1;
                dim_maxs[k] = dim_mins[k] + var_dim(k) - 1;
            }

            // Does nothing if ndom_pts_ is empty
            add_grid(ndom_pts_);

            // n.b. Param object must be constructed after grid is added
            params = make_shared<Param<T>>(dom_dim, ndom_pts());

            validate();
        }

        // Constructor for a PointSet mapped from existing parameters
        //   N.B. this is useful when decoding at the same params as encoding,
        //        or when Params are constructed ahead of time
        PointSet(
                shared_ptr<Param<T>>    params_,
                const VectorXi&         mdims_) :
            dom_dim(params_->dom_dim),
            pt_dim(mdims_.sum()),
            npts(params_->npts()),
            mdims(mdims_),
            params(params_)
            // structured(params->structured)
        {
            domain.resize(npts, pt_dim);

            // Fill dim_mins/maxs
            dim_mins.resize(nvars());
            dim_maxs.resize(nvars());
            dim_mins[0] = geom_dim();
            dim_maxs[0] = dim_mins[0] + var_dim(0) - 1;
            for (int k = 1; k < nvars(); k++)
            {
                dim_mins[k] = dim_maxs[k-1] + 1;
                dim_maxs[k] = dim_mins[k] + var_dim(k) - 1;
            }

            if (params_->structured)
                add_grid(params_->ndom_pts);
        }

        // Manually set (or overwrite) domain bounding box.
        // If bounds are not set manually, they will be computed during the 
        // first call to dom_mins() or dom_maxs() by searching 'domain'
        // for the min/max coordinates in each domain dimension.
        void set_bounds(const VectorX<T>& mins_, const VectorX<T>& maxs_)
        {
            if ( (mins_.size() != geom_dim()) || (mins_.size() != maxs_.size()) )
            {
                fmt::print(stderr, "ERROR: Invalid bounds passed to PointSet\n");
                fmt::print(stderr, "  mins: [{}]\n", fmt::join(mins_, " "));
                fmt::print(stderr, "  maxs: [{}]\n", fmt::join(maxs_, " "));
                exit(1);
            }

            dom_mins = mins_;
            dom_maxs = maxs_;
            bounds_cached = true;
        }

        VectorX<T> mins() const
        {
            if (!bounds_cached)
            {
                dom_mins = domain.leftCols(geom_dim()).colwise().minCoeff();
                dom_maxs = domain.leftCols(geom_dim()).colwise().maxCoeff();
                bounds_cached = true;
            }

            return dom_mins;
        }
        
        VectorX<T> maxs() const
        {
            if (!bounds_cached)
            {
                dom_mins = domain.leftCols(geom_dim()).colwise().minCoeff();
                dom_maxs = domain.leftCols(geom_dim()).colwise().maxCoeff();
                bounds_cached = true;
            }

            return dom_maxs;
        }

        T mins(int i) const {return mins()(i);}

        T maxs(int i) const {return maxs()(i);}

        int nvars() const {return mdims.size() - 1;}

        int geom_dim() const {return mdims[0];}

        int var_dim(int k) const {return mdims[k+1];}

        int var_min(int k) const {return dim_mins[k];}

        int var_max(int k) const {return dim_maxs[k];}

        VectorXi model_dims() const {return mdims;}

        VectorXi ndom_pts() const {return g.ndom_pts;}

        int ndom_pts(int i) const {return g.ndom_pts(i);}

        bool is_structured() const {return g.initialized;}

        shared_ptr<Param<T>> get_params_ptr() const {return params;}

        void set_params(shared_ptr<Param<T>> params_)
        {
            if (!check_param_domain_agreement(*params_))
            {
                throw MFAError("Attempted to add mismatched Params to PointSet");
            }
 
            params = params_;
        }

        void set_params(const PointSet<T>& ps)
        {
            set_params(ps.get_params_ptr());
        }

        // Create Param object with a domain parametrization
        void set_domain_params()
        {
            // n.b. A structured grid which has been rotated will still have its parameters computed correctly.
            //      dom mins/maxs are not used in the computation of structured parameters, so the parameters
            //      are computed to be the correct "rotated" grid
            // params->make_domain_params(geom_dim(), domain);
            params->makeDomainParams(geom_dim(), domain);
        }

        // Create Param object with a domain parametrization, with a bounding
        // box specified by [domain_mins, domain_maxs]
        void set_domain_params(const VectorX<T>& domain_mins, const VectorX<T>& domain_maxs)
        {
            set_bounds(domain_mins, domain_maxs);

            // params->make_domain_params(geom_dim(), domain, domain_mins, domain_maxs);
            params->makeDomainParams(domain_mins, domain_maxs, domain);
        }

        // Define parametrizations directly from a bounding box which may be rotated (not axis-aligned)
        void set_domain_params(const Bbox<T>& box)
        {
            params->makeDomainParams(box, domain);
        }

        // Create Param object that is equispaced over all parameter space
        void set_grid_params()
        {
            if (!is_structured())
            {
                throw MFAError("Cannot set grid parametrization to unstructured PointSet");
            }

            params->make_grid_params();
        }

        // Create Param object that is equispaced on a given subvolume of parameter space
        void set_grid_params(const VectorX<T>& param_mins, const VectorX<T>& param_maxs)
        {
            if (!is_structured())
            {
                throw MFAError("Cannot set grid parametrization to unstructured PointSet");
            }

            params->make_grid_params(param_mins, param_maxs);
        }

        void set_curve_params()
        {
            if (!is_structured())
            {
                throw MFAError("Cannot set curve parametrization to unstructured PointSet");
            }

            params->make_curve_params(domain);
        }

        // Checks that the Param object does not contradict existing members of PointSet
        bool check_param_domain_agreement(const Param<T>& params_) const
        {
            bool agreement =    (dom_dim == params_.dom_dim)
                            &&  (npts == params_.npts())
                            &&  (is_structured() == params_.structured)
                            &&  (ndom_pts() == params_.ndom_pts)
                            ;

            return agreement;
        }

        // Add a grid structure to point set
        // Does nothing if ndom_pts_ is empty
        void add_grid(VectorXi ndom_pts_)
        {
            if (ndom_pts_.size() == 0)
            {
                return;
            }
            else
            {
                if (npts != ndom_pts_.prod())
                {
                    fmt::print(stderr, "ERROR: Invalid grid added to PointSet. Total points do not match.\n");
                    fmt::print(stderr, "       npts = {}\n", npts);
                    fmt::print(stderr, "       ndom_pts = [{}]\n", fmt::join(ndom_pts_, " "));
                    exit(1);
                }

                g.init(dom_dim, ndom_pts_);

                validate(); // Prints warning and aborts if invalid
            }
        }

        // Test that user-provided data meets basic sanity checks
        bool validate() const
        { 
            bool is_valid =     (dom_dim > 0)
                            &&  (geom_dim() > 0)
                            &&  (pt_dim >= dom_dim)
                            &&  (pt_dim == domain.cols())
                            &&  (npts == domain.rows())
                            &&  (is_structured() ? ndom_pts().size() == dom_dim : true)
                            &&  (is_structured() ? ndom_pts().prod() == domain.rows() : true)
                            ;

            for (int k = 0; k < nvars(); k++)
            {
                is_valid = is_valid && (mdims[k+1] > 0);
                is_valid = is_valid && (dim_maxs[k] - dim_mins[k] + 1 == mdims[k+1]);
            }

            if (nvars() > 0)
            {
                is_valid = is_valid && (dim_mins[0] == geom_dim()) && (dim_maxs[nvars()-1] == pt_dim-1);
            }

            if (is_valid) return is_valid;
            else 
            {
                string err_message = 
                    fmt::format("PointSet initialized with incompatible data\n"
                                "       structured: {}\n"
                                "       dom_dim: {}, geom_dim: {},  pt_dim: {}\n"
                                "       npts: {}\n"
                                "       ndom_pts: [{}]\n"
                                "       domain matrix dims: {} x {}\n"
                                "       nvars: {}\n"
                                "       model_dims: [{}]\n"
                                "       dim_mins: [{}]\n"
                                "       dim_maxs: [{}]\n",
                                is_structured(),
                                dom_dim, geom_dim(), pt_dim,
                                npts,
                                fmt::join(ndom_pts(), " "),
                                domain.rows(), domain.cols(),
                                nvars(),
                                fmt::join(mdims, " "),
                                fmt::join(dim_mins, " "),
                                fmt::join(dim_maxs, " "));
                throw MFAError(err_message);
                
                return is_valid;    // never reached since we throw above
            }
        }

        bool is_same_layout(const PointSet<T>& ps, int verbose = 1) const
        {
            bool is_same =      (dom_dim        == ps.dom_dim)
                            &&  (pt_dim         == ps.pt_dim)
                            &&  (npts           == ps.npts)
                            &&  (is_structured()== ps.is_structured())
                            &&  (mdims          == ps.mdims);
            
            if (is_structured())
            {
                is_same = is_same && (ndom_pts() == ps.ndom_pts());
            }

            if (!is_same)
            {
                if (verbose >= 2)
                {
                    fmt::print(stderr, "DEBUG: Pair of PointSets do not have matching layout\n");
                    fmt::print(stderr, "       dom_dim    = {},\t{}\n", dom_dim, ps.dom_dim);
                    fmt::print(stderr, "       pt_dim     = {},\t{}\n", pt_dim, ps.pt_dim);
                    fmt::print(stderr, "       npts       = {},\t{}\n", npts, ps.npts);
                    fmt::print(stderr, "       structured = {},\t{}\n", is_structured(), ps.is_structured());
                    if (is_structured() || ps.is_structured())
                    {
                        fmt::print(stderr, "       ndom_pts: [{}] [{}]\n", fmt::join(ndom_pts(), " "), fmt::join(ps.ndom_pts(), " "));
                    }
                    fmt::print(stderr, "       model_dims = [{}] [{}]\n", fmt::join(mdims, " "), fmt::join(ps.mdims, " "));
                }
            }

            return is_same;
        }

        void abs_diff(
            const   mfa::PointSet<T>& other,
                    mfa::PointSet<T>& diff) const
        {
            if (!this->is_same_layout(other) || !this->is_same_layout(diff))
            {
                throw MFAError("Incompatible PointSets in PointSet::abs_diff");
            }

#ifdef MFA_TBB
            parallel_for (size_t(0), (size_t)diff.npts, [&] (size_t i)
                {
                    for (auto j = 0; j < geom_dim(); j++)
                    {
                        diff.domain(i,j) = this->domain(i,j); // copy the geometric location of each point
                    }
                });

            parallel_for (size_t(0), (size_t)npts, [&] (size_t i)
                {
                    for (auto j = geom_dim(); j < pt_dim; j++)
                    {
                        diff.domain(i,j) = fabs(this->domain(i,j) - other.domain(i,j)); // compute distance between each science value
                    }
                });
#else
            diff.domain.leftCols(geom_dim()) = this->domain.leftCols(geom_dim());
            diff.domain.rightCols(pt_dim-geom_dim()) = (this->domain.rightCols(pt_dim-geom_dim()) - other.domain.rightCols(pt_dim-geom_dim())).cwiseAbs();
#endif
        }

        PointSet(const PointSet&) = delete;
        PointSet(PointSet&&) = delete;
        PointSet& operator=(const PointSet&) = delete;
        PointSet& operator=(PointSet&&) = delete;

        // PointSet& operator=(PointSet&& other)
        // {
        //     swap(*this, other);
        //     return *this;
        // }

        // friend void swap(PointSet& first, PointSet& second)
        // {
        //     swap(first.dom_dim, second.dom_dim);
        //     swap(first.pt_dim, second.pt_dim);
        //     swap(first.structured, second.structured);
        //     first.ndom_pts.swap(second.ndom_pts);
        //     first.dom_mins.swap(second.dom_mins);
        //     first.dom_maxs.swap(second.dom_maxs);
        //     swap(first.g, second.g);
        //     first.domain.swap(second.domain);
        //     swap(first.params, second.params);
        // }

        class PtIterator
        {
            const bool  structured;
            size_t      lin_idx;
            mfa::VolIterator vol_it;
            const PointSet&  pset;

        public:
            PtIterator(const PointSet& pset_, size_t idx_) :
                structured(pset_.is_structured()),
                lin_idx(idx_),
                vol_it(structured ? VolIterator(pset_.ndom_pts(), idx_) : VolIterator()),
                pset(pset_)
            { }

            // prefix increment
            PtIterator operator++()
            {
                if(structured)
                    vol_it.incr_iter();
                
                lin_idx++;

                return *this;
            }


            bool operator!=(const PtIterator& other)
            {
                return (structured != other.structured) || (lin_idx != other.lin_idx);
            }

            bool operator==(const PtIterator& other)
            {
                return !(*this!=other);
            }

            // Full set of coordinates at this point
            void coords(VectorX<T>& coord_vec)
            {
                coord_vec = pset.domain.row(lin_idx);
            }

            // Geometric coordinates at this point
            void geom_coords(VectorX<T>& coord_vec)
            {
                coord_vec = pset.domain.row(lin_idx).head(pset.geom_dim());
            }

            // Coordinates for variable k at this point
            void var_coords(int k, VectorX<T>& coord_vec)
            {
                coord_vec = pset.domain.row(lin_idx).segment(pset.var_min(k), pset.var_dim(k));
            }

            void coords(VectorX<T>& coord_vec, size_t min_dim, size_t max_dim)
            {
                coord_vec = pset.domain.block(idx(), min_dim, 1, max_dim - min_dim + 1).transpose();
            }

            void params(VectorX<T>& param_vec)
            {
                if (structured) pset.params->pt_params(vol_it, param_vec);
                else            pset.params->pt_params(lin_idx, param_vec);
            }

            void ijk(VectorXi& ijk_vec)
            {
                if (!structured)
                {
                    throw MFAError("No ijk values in PtIterator for unstructured input");
                }

                ijk_vec = vol_it.idx_dim();
            }

            int ijk(int k)
            {
                if (!structured)
                {
                    throw MFAError("No ijk values in PtIterator for unstructured input");
                }

                return vol_it.idx_dim(k);
            }

            int idx()
            {
                return lin_idx;
            }
        };  // PtIterator

        PtIterator iterator(size_t idx) const
        {
            return PtIterator(*this, idx);
        }

        PtIterator begin() const
        {
            return PtIterator(*this, 0);
        }

        PtIterator end() const
        {
            return PtIterator(*this, npts);
        }
        PtIterator last() const
        {
            return PtIterator(*this, npts - 1);
        }

        void pt_coords(size_t idx, VectorX<T>& coord_vec) const
        {
            coord_vec = domain.row(idx);
        }

        void geom_coords(size_t idx, VectorX<T>& coord_vec) const
        {
            coord_vec = domain.row(idx).head(geom_dim());
        }

        void var_coords(size_t idx, size_t k, VectorX<T>& coord_vec) const
        {
            coord_vec = domain.row(idx).segment(var_min(k), var_dim(k));
        }

        // This should not be called in a large loop because it involves
        // dynamic memory allocation when VectorXi is constructed
        // Could pre-allocate a VectorXi in GridInfo to hold the
        // output of idx2ijk for cases like this.
        void pt_params(size_t idx, VectorX<T>& param_vec) const
        {
            if(is_structured())
            {
                VectorXi ijk(dom_dim);
                g.idx2ijk(idx, ijk);
                params->pt_params(ijk, param_vec);
            }
            else
            {
                params->pt_params(idx, param_vec);   
            }
        }
    };
}   // namespace mfa

#endif // _POINTSET_HPP
