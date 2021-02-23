//--------------------------------------------------------------
// mfa input data structure
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
// 
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _INPUT_HPP
#define _INPUT_HPP

namespace mfa
{
    template <typename T>
    struct PointSet
    {
        PointSet(
                        size_t      dom_dim_,
                        size_t      pt_dim_,
                const   VectorX<T>& mins_,
                const   VectorX<T>& maxs_,
                        bool        structured_,
                const   VectorXi&   ndom_pts_) :
            dom_dim(dom_dim_),
            pt_dim(pt_dim_),
            dom_mins(mins_),
            dom_maxs(maxs_),
            structured(structured_),
            ndom_pts(ndom_pts_)
        {
            // Check that ndom_pts matches structured flag
            if ( (!structured && ndom_pts.size() != 0) ||
                (structured && ndom_pts.size() == 0)     ) 
            {
                cerr << "ERROR: Conflicting constructor arguments for PointSet" << endl;    
                cerr << "  structured: " << boolalpha << structured << endl;
                cerr << "  ndom_pts: ";
                for (size_t k = 0; k < ndom_pts.size(); k++) cerr << ndom_pts(k) << " ";
                cerr << endl;
            }
        }

        void init()
        {
            if (is_initialized)
            {
                cerr << "Warning: Attempting to initialize a previously initialized PointSet" << endl;
                return;
            }
            if (!validate())
            {
                cerr << "ERROR: Improper setup of PointSet" << endl;
                exit(1);
            }
            else
            {
                // set total number of points
                tot_ndom_pts = domain.rows();
                
                // set parameters
                Param<T> temp_param(dom_dim, dom_mins, dom_maxs, ndom_pts, domain, structured);
                swap(params, temp_param);
                
                // set grid data structure if needed
                if (structured)
                {
                    g.init(dom_dim, ndom_pts);
                }
            }

            is_initialized = true;
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

        // Defined during construction
        int         dom_dim;
        int         pt_dim;
        bool        structured;
        VectorXi    ndom_pts;           
        VectorX<T>  dom_mins;           // Minimal extents of bounding box (optional: for parametrization)
        VectorX<T>  dom_maxs;           // Maximal extents of bounding box (optional: for parametrization)
        // VectorXi    model_dims;

        // Defined by user
        MatrixX<T>  domain;

        // Defined automatically during init()
        int             tot_ndom_pts{0};
        mfa::GridInfo   g;
        mfa::Param<T>   params;
        bool            is_initialized{false};

        

        class PtIterator
        {
            const bool  structured;
            size_t      lin_idx;
            mfa::VolIterator vol_it;
            const PointSet&  pset;

        public:
            PtIterator(const PointSet& pset_, size_t idx_) :
                structured(pset_.structured),
                lin_idx(structured ? 0 : idx_),
                vol_it(structured ? VolIterator(pset_.ndom_pts, idx_) : VolIterator()),
                pset(pset_)
            { }

            // prefix increment
            PtIterator operator++()
            {
                if(structured)
                    vol_it.incr_iter();
                else
                    lin_idx++;

                return *this;
            }


            bool operator!=(const PtIterator& other)
            {
                return structured ? (vol_it.cur_iter() != other.vol_it.cur_iter()) :
                                    (lin_idx != other.lin_idx);
            }

            bool operator==(const PtIterator& other)
            {
                return !(*this!=other);
            }

            void coords(VectorX<T>& coord_vec)
            {
                if(structured)
                    coord_vec = pset.domain.row(vol_it.cur_iter());
                else
                    coord_vec = pset.domain.row(lin_idx);
            }

            void coords(VectorX<T>& coord_vec, size_t min_dim, size_t max_dim)
            {
                coord_vec = pset.domain.block(idx(), min_dim, 1, max_dim - min_dim + 1).transpose();
            }

            void params(VectorX<T>& param_vec)
            {
                if(structured)
                    param_vec = pset.params.pt_params(vol_it);
                else
                    param_vec = pset.params.pt_params(lin_idx);
            }

            void ijk(VectorXi& ijk_vec)
            {
                if (!structured)
                {
                    cerr << "ERROR: No ijk values in PtIterator for unstructured input" << endl;
                    exit(1);
                }

                ijk_vec = vol_it.idx_dim();
            }

            int idx()
            {
                return structured ? vol_it.cur_iter() : lin_idx;
            }
        };

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
            return PtIterator(*this, tot_ndom_pts);
        }

        void pt_coords(size_t idx, VectorX<T>& coord_vec) const
        {
            coord_vec = domain.row(idx);
        }

        void pt_params(size_t idx, VectorX<T>& param_vec) const
        {
            if(structured)
            {
                VectorXi ijk(dom_dim);
                g.idx2ijk(idx, ijk);
                param_vec = params.pt_params(ijk);
            }
            else
            {
                param_vec = params.pt_params(idx);   
            }
        }

        // Test that user-provided data meets basic sanity checks
        bool validate() const
        { 
            bool is_valid =     (dom_dim > 0)
                            &&  (pt_dim > dom_dim)
                            &&  (pt_dim == domain.cols())
                            &&  (structured ? ndom_pts.size() == dom_dim : true)
                            &&  (structured ? ndom_pts.prod() == domain.rows() : true)
                            &&  (dom_mins.size() == dom_maxs.size())
                            ;

            if (is_valid) return is_valid;
            else 
            {
                cerr << "PointSet initialized with incompatible data" << endl;
                cerr << "  structured: " << boolalpha << structured << endl;
                cerr << "  dom_dim: " << dom_dim << ",  pt_dim: " << endl;
                cerr << "  ndom_pts: ";
                for (size_t k=0; k < ndom_pts.size(); k++) 
                    cerr << ndom_pts(k) << " ";
                cerr << endl;
                cerr << "  domain matrix dims: " << domain.rows() << " x " << domain.cols() << endl;
                cerr << "  dom_mins: ";
                for (size_t k=0; k < dom_mins.size(); k++)
                    cerr << dom_mins(k) << " ";
                cerr << endl;
                cerr << "  dom_maxs: ";
                for (size_t k=0; k < dom_maxs.size(); k++)
                    cerr << dom_maxs(k) << " ";
                cerr << endl;

                return is_valid;
            }
        }
    };
}   // namespace mfa

#endif // _INPUT_HPP