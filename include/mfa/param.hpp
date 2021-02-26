//--------------------------------------------------------------
// parameterization object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _PARAMS_HPP
#define _PARAMS_HPP

#include    "mfa/util.hpp"

#include    <Eigen/Dense>
#include    <vector>
#include    <list>
#include    <iostream>

#ifdef MFA_TBB
#include    <tbb/tbb.h>
using namespace tbb;
#endif

using namespace std;

namespace mfa
{
    template <typename T>                           // float or double
    struct Param
    {
        VectorXi                ndom_pts;           // number of domain points in each dimension
        vector<vector<T>>       param_grid;         // parameters for input points[dimension][index] (for structured case)
        MatrixX<T>              param_list;         // list of parameters for each input pt (for unstructured case)
        T                       range_extent;       // extent of range value of input data points  // TODO: what does this have to do with parameters?
        // vector<vector<size_t>>  co;                 // starting offset for curves in each dim
        // vector<size_t>          ds;                 // stride for domain points in each dim
        int                     dom_dim;            // dimensionality of domain
        bool                    structured;         // true if points lie on structured grid

        // Default constructor
        Param() : range_extent(0), dom_dim(0), structured(true) { }

//         // Construcutor for unstructured input
//         Param(  int                 dom_dim_,
//                 const MatrixX<T>&   domain_) :
//             dom_dim(dom_dim_),
//             ndom_pts(VectorXi::Zero(dom_dim)),
//             structured(false)
//         {
// #ifdef CURVE_PARAMS
//             cerr << "ERROR: Cannot set curve parametrization to unstructured input" << endl;
// #else
//             setDomainParamsUnstructured(domain_);
// #endif
//         }

        // Constructor for structured input
        Param(  int                 dom_dim_,           // domain dimensionality (excluding science variables)
                const VectorX<T>&   dom_mins_,          // minimal extents of bounding box in each dimension (optional, important when data does not cover domain)
                const VectorX<T>&   dom_maxs_,          // maximal extents of bounding box in each dimension (see above)
                const VectorXi&     ndom_pts_,          // number of input data points in each dim
                const MatrixX<T>&   domain_,
                bool                structured_) :          // input data points (1st dim changes fastest)
            dom_dim(dom_dim_),
            ndom_pts(ndom_pts_),
            structured(structured_)
        {
            if (structured == true)
                assert(ndom_pts.size() > 0);
            if (structured == false)
                assert(ndom_pts.size() == 0);

            // check dimensionality for sanity
            assert(dom_dim < domain_.cols());

            // precompute curve parameters and knots for input points
            param_grid.resize(dom_dim);

#ifdef CURVE_PARAMS
            if (structured)
                CurveParams(domain_, param_grid);           // params spaced according to the curve length (per P&T)
            else
            {
                cerr << "ERROR: Cannot set curve parametrization to unstructured input" << endl;
                exit(1);
            }
#else
            if (structured)
                setDomainParamsStructured(domain_, dom_mins_, dom_maxs_, param_grid);          // params spaced according to domain spacing
            else
                setDomainParamsUnstructured(domain_, dom_mins_, dom_maxs_);
#endif

            // debug
//             fprintf(stderr, "----- params -----\n");
//             for (auto i = 0; i < params.size(); i++)
//             {
//                 fprintf(stderr, "dimension %d:\n", i);
//                 for (auto j = 0; j < params[i].size(); j++)
//                     fprintf(stderr, "params[%d][%d] = %.3lf\n", i, j, params[i][j]);
//             }
//             fprintf(stderr, "-----\n");

            // max extent of input data points
            int last     = domain_.cols() - 1;
            range_extent = domain_.col(last).maxCoeff() - domain_.col(last).minCoeff();

            // NB: Moved to GridInfo struct within InputInfo

            // // stride for domain points in different dimensions
            // ds.resize(dom_dim, 1);
            // for (size_t i = 1; i < dom_dim; i++)
            //     ds[i] = ds[i - 1] * ndom_pts_[i - 1];

            // // offsets for curve starting (domain) points in each dimension
            // co.resize(dom_dim);
            // for (auto k = 0; k < dom_dim; k++)
            // {
            //     size_t ncurves  = domain_.rows() / ndom_pts_(k);    // number of curves in this dimension
            //     size_t coo      = 0;                                // co at start of contiguous sequence
            //     co[k].resize(ncurves);

            //     co[k][0] = 0;

            //     for (auto j = 1; j < ncurves; j++)
            //     {
            //         // adjust offsets for the next curve
            //         if (j % ds[k])
            //             co[k][j] = co[k][j - 1] + 1;
            //         else
            //         {
            //             co[k][j] = coo + ds[k] * ndom_pts_(k);
            //             coo = co[k][j];
            //         }
            //     }
            // }
        }

        size_t npts()
        {
            return structured ? ndom_pts.prod() : param_list.rows();
        }

        friend void swap(Param& first, Param& second)
        {
            first.ndom_pts.swap(second.ndom_pts);
            swap(first.param_grid, second.param_grid);
            first.param_list.swap(second.param_list);
            swap(first.range_extent, second.range_extent);
            // swap(first.co, second.co);
            // swap(first.ds, second.ds);
            swap(first.dom_dim, second.dom_dim);
            swap(first.structured, second.structured);
        }

        // Structured data only.
        // Get params from VolIterator
        VectorX<T> pt_params(const VolIterator& it) const
        {
            VectorX<T> ret(dom_dim);
            for(int k = 0; k < dom_dim; k++)
            {
                ret(k) = param_grid[k][it.idx_dim(k)];
            }

            return ret;
        }

        // Structured data only.
        // Get params from param indices in each dimension
        VectorX<T> pt_params(const VectorXi& idxs) const
        {
            VectorX<T> ret(dom_dim);
            for(int k = 0; k < dom_dim; k++)
            {
                ret(k) = param_grid[k][idxs(k)];
            }

            return ret;
        }

        // Unstructured data only.
        // Get params from linear index
        VectorX<T> pt_params(int i) const
        {
            return param_list.row(i);
        }

        // precompute curve parameters for input data points using the chord-length method
        // n-d version of algorithm 9.3, P&T, p. 377
        // params are computed along curves and averaged over all curves at same data point index i,j,k,...
        // ie, resulting params for a data point i,j,k,... are same for all curves
        // and params are only stored once for each dimension (1st dim params, 2nd dim params, ...)
        // total number of params is the sum of ndom_pts over the dimensions, much less than the total
        // number of data points (which would be the product)
        // assumes params were allocated by caller
        void setCurveParamsStructured(
                const MatrixX<T>&   domain,                 // input data points (1st dim changes fastest)
                vector<vector<T>>&  params)                 // (output) parameters for input points[dimension][index]
        {
            T          tot_dist;                          // total chord length
            VectorX<T> dists(ndom_pts.maxCoeff() - 1);    // chord lengths of data point spans for any dim
            params = VectorX<T>::Zero(params.size());
            VectorX<T> d;                                 // current chord length

            // following are counters for slicing domain and params into curves in different dimensions
            size_t co = 0;                     // starting offset for curves in domain in current dim
            size_t cs = 1;                     // stride for domain points in curves in current dim

            for (size_t k = 0; k < ndom_pts.size(); k++)         // for all domain dimensions
            {
                params[k].resize(ndom_pts(k));
                co = 0;
                size_t coo = 0;                                  // co at start of contiguous sequence
                size_t ncurves = domain.rows() / ndom_pts(k);    // number of curves in this dimension
                size_t nzero_length_curves = 0;                  // num curves with zero length
                for (size_t j = 0; j < ncurves; j++)             // for all the curves in this dimension
                {
                    tot_dist = 0.0;

                    // chord lengths
                    for (size_t i = 0; i < ndom_pts(k) - 1; i++) // for all spans in this curve
                    {
                        // TODO: normalize domain so that dimensions they have similar scales
                        d = domain.row(co + i * cs) - domain.row(co + (i + 1) * cs);
                        dists(i) = d.norm();                     // Euclidean distance (l-2 norm)
                        tot_dist += dists(i);
                    }

                    // accumulate (sum) parameters from this curve into the params for this dim.
                    if (tot_dist > 0.0)                          // skip zero length curves
                    {
                        params[k][0]                 = 0.0;      // first parameter is known
                        params[k][ndom_pts(k) - 1]   = 1.0;      // last parameter is known
                        T prev_param                 = 0.0;      // param value at previous iteration below
                        for (size_t i = 0; i < ndom_pts(k) - 2; i++)
                        {
                            T dfrac             = dists(i) / tot_dist;
                            params[k][i + 1]    += prev_param + dfrac;
                            prev_param += dfrac;
                        }
                    }
                    else
                        nzero_length_curves++;

                    if ((j + 1) % cs)
                        co++;
                    else
                    {
                        co = coo + cs * ndom_pts(k);
                        coo = co;
                    }
                }                                                // curves in this dimension

                // average the params for this dimension by dividing by the number of curves that
                // contributed to the sum (skipped zero length curves)
                for (size_t i = 0; i < ndom_pts(k) - 2; i++)
                    params[k][i + 1] /= (ncurves - nzero_length_curves);

                cs *= ndom_pts(k);
            }                                                    // domain dimensions
            // debug
            //     cerr << "params:\n" << params << endl;
        }

        // precompute parameters for input data points using domain spacing only (not length along curve)
        // params are only stored once for each dimension (1st dim params, 2nd dim params, ...)
        // total number of params is the sum of ndom_pts over the dimensions, much less than the total
        // number of data points (which would be the product)
        // assumes params were allocated by caller
        void setDomainParamsStructured(
                const MatrixX<T>&     domain,                   // input data points (1st dim changes fastest)
                const VectorX<T>&     dom_mins,
                const VectorX<T>&     dom_maxs,
                vector<vector<T>>&    params)                   // (output) parameters for input points[dimension][index]
        {
            VectorX<T> mins;
            VectorX<T> maxs;

            // dom mins/maxs should either be empty or of size dom_dim
            if (dom_mins.size() > 0 && dom_mins.size() != dom_dim)
                cerr << "Warning: Invalid size of dom_mins in Param construction" << endl;
            if (dom_maxs.size() > 0 && dom_maxs.size() != dom_dim)
                cerr << "Warning: Invalid size of dom_maxs in Param construction" << endl;

            // Set min/max extents in each domain dimension
            if (dom_mins.size() != dom_dim || dom_maxs.size() != dom_dim)
            {
                mins = domain.leftCols(dom_dim).colwise().minCoeff();
                maxs = domain.leftCols(dom_dim).colwise().maxCoeff();
            }
            else
            {
                mins = dom_mins;
                maxs = dom_maxs;
            }

            VectorX<T> diff = maxs - mins;

            size_t cs = 1;                                      // stride for domain points in current dim.
            for (size_t k = 0; k < dom_dim; k++)                // for all domain dimensions
            {
                param_grid[k].resize(ndom_pts(k));
                for (size_t i = 0; i < ndom_pts(k); i++)
                    param_grid[k][i]= (domain(cs * i, k) - mins(k)) / diff(k);

                cs *= ndom_pts(k);
            }                                                    // domain dimensions

            // Check for parameter values outside of [0,1]
            // This could happen if dom_mins, dom_maxs are set incorrectly, e.g.
            // Also catches any unanticipated floating point arithmetic issues
            for (size_t k = 0; k < dom_dim; k++)
            {
                for (size_t j = 0; j < ndom_pts(k); j++)
                {
                    if (param_grid[k][j] < 0.0 || param_grid[k][j] > 1.0)
                    {
                        cerr << "ERROR: Domain parametrization contains values outside of [0,1]. Exiting program." << endl;
                        cerr << "         parameter for dim " << k << ", pt #" << j << " = " << param_grid[k][j] << endl;
                        exit(1);
                    }
                }
            }
            // debug
            //     cerr << "params:\n" << params << endl;
        }

        void setDomainParamsUnstructured(
            const MatrixX<T>&   domain,
            const VectorX<T>&   dom_mins,
            const VectorX<T>&   dom_maxs)
        {
            VectorX<T> mins;
            VectorX<T> maxs;

            // dom mins/maxs should either be empty or of size dom_dim
            if (dom_mins.size() > 0 && dom_mins.size() != dom_dim)
                cerr << "Warning: Invalid size of dom_mins in Param construction" << endl;
            if (dom_maxs.size() > 0 && dom_maxs.size() != dom_dim)
                cerr << "Warning: Invalid size of dom_maxs in Param construction" << endl;

            // Set min/max extents in each domain dimension
            if (dom_mins.size() != dom_dim || dom_maxs.size() != dom_dim)
            {
                mins = domain.leftCols(dom_dim).colwise().minCoeff();
                maxs = domain.leftCols(dom_dim).colwise().maxCoeff();
            }
            else    // Use domain bounds provided by block (input data need not extend to bounds)
            {
                mins = dom_mins;
                maxs = dom_maxs;
            }
            
            int npts = domain.rows();
            param_list.resize(npts, dom_dim);

            VectorX<T> diff = maxs - mins;

            // Rescale domain values to the interval [0,1], column-by-column
            for (size_t k = 0; k < dom_dim; k++)
            {
                param_list.col(k) = (domain.col(k).array() - mins(k)) * (1/diff(k));
            }
        
            // Check for parameter values outside of [0,1]
            // This could happen if dom_mins, dom_maxs are set incorrectly, e.g.
            // Also catches any unanticipated floating point arithmetic issues
            if ((param_list.array() < 0.0).any() || (param_list.array() > 1.0).any())
            {
                cerr << "ERROR: Domain parametrization contains values outside of [0,1]. Exiting program." << endl;
                exit(1);
            }
        }
    };

}                                               // namespace

#endif
