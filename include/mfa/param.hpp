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
#include    <iomanip>

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
        int                     dom_dim{0};         // dimensionality of domain
        bool                    structured{false};  // true if points lie on structured grid

        // // Default constructor
        // Param() : dom_dim(0), structured(true) { }

        // // Constructor for equispaced grid over all of parameter space
        // Param(const VectorXi& ndom_pts_) :
        //     ndom_pts(ndom_pts_),
        //     dom_dim(ndom_pts.size()),
        //     structured(true)
        // {
        //     VectorX<T>  param_mins = VectorX<T>::Zero(dom_dim);
        //     VectorX<T>  param_maxs = VectorX<T>::Ones(dom_dim);

        //     make_grid_params(ndom_pts, param_mins, param_maxs);
        // }

        // DEPRECATED in favor of having a single constructor and then calling make_grid_params()
        // 
        // // Constructor for equispaced grid over subset of parameter space
        // // N.B. If ndom_pts_(k) = 1 in any dimension, then we expect param_mins_(k) == param_maxs_(k)
        // Param(  const VectorXi&     ndom_pts_,
        //         const VectorX<T>&   param_mins_ = VectorX<T>::Zero(ndom_pts_.size()),
        //         const VectorX<T>&   param_maxs_ = VectorX<T>::Ones(ndom_pts_.size())) :
        //     ndom_pts(ndom_pts_),
        //     dom_dim(ndom_pts.size()),
        //     structured(true)
        // {
        //     make_grid_params(param_mins_, param_maxs_);
        // }

        // DEPRECATED see above
        //
        // // Convenience constructor to make equispaced grid of params
        // // from an STL vector
        // Param(const vector<int>& ndom_pts_) :
        //     Param(Eigen::Map<VectorXi>(&ndom_pts_[0], ndom_pts_.size()))
        // { }

        Param(int dom_dim_, const vector<int>& ndom_pts_) :
            Param(dom_dim_, Eigen::Map<VectorXi>(&ndom_pts_[0], ndom_pts_.size()))
        { }

        // General constructor for creating unspecified  params with an optional grid structure
        Param(  int                 dom_dim_,           // domain dimensionality
                // const VectorX<T>&   dom_mins_,          // minimal extents of bounding box in each dimension (optional, important when data does not cover domain)
                // const VectorX<T>&   dom_maxs_,          // maximal extents of bounding box in each dimension (see above)
                const VectorXi&     ndom_pts_ = VectorX<T>()) :          // number of input data points in each dim
                // const MatrixX<T>&   domain_,            // physical coordinates of points
                // bool                structured_) :
            dom_dim(dom_dim_),
            ndom_pts(ndom_pts_),
            structured(ndom_pts_.size() > 0)
        { }
//             // TODO: replace with warnings?
//             if (structured == true)
//                 assert(ndom_pts.size() > 0);
//             if (structured == false)
//                 assert(ndom_pts.size() == 0);

//             // check dimensionality for sanity
//             assert(dom_dim < domain_.cols());

//             // precompute curve parameters and knots for input points
//             param_grid.resize(dom_dim);

// #ifdef CURVE_PARAMS
//             if (structured)
//                 CurveParams(domain_, param_grid);           // params spaced according to the curve length (per P&T)
//             else
//             {
//                 cerr << "ERROR: Cannot set curve parametrization to unstructured input" << endl;
//                 exit(1);
//             }
// #else
//             if (structured)
//                 make_domain_params_structured(domain_);          // params spaced according to domain spacing
//             else
//                 make_domain_params_unstructured(domain_, dom_mins_, dom_maxs_);
// #endif
//             // print();
        // }

        int npts() const
        {
            return structured ? ndom_pts.prod() : param_list.rows();
        }

        friend void swap(Param& first, Param& second)
        {
            first.ndom_pts.swap(second.ndom_pts);
            swap(first.param_grid, second.param_grid);
            first.param_list.swap(second.param_list);
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

        void make_grid_params()
        {
            make_grid_params(VectorX<T>::Zero(dom_dim), VectorX<T>::Ones(dom_dim));
        }

        // Set parameters to be a rectangular, equispaced grid bounded by [param_mins, param_maxs]
        void make_grid_params(  const VectorX<T>&   param_mins,   // Minimum param in each dimension
                                const VectorX<T>&   param_maxs)   // Maximum param in each dimension
        {
            if (!structured)
            {
                cerr << "\nWarning: Setting grid params to unstructured Param object\n" << endl;
            }

            T step = 0;
            param_grid.resize(dom_dim);

            for (int k = 0; k < dom_dim; k++)
            {
                param_grid[k].resize(ndom_pts(k));

                param_grid[k][0] = param_mins(k);

                if (ndom_pts(k) > 1)
                {
                    param_grid[k][ndom_pts(k)-1] = param_maxs(k);

                    step = (param_maxs(k) - param_mins(k)) / (ndom_pts(k)-1);
                    for (int j = 1; j < ndom_pts(k)-1; j++)
                    {
                        param_grid[k][j] = param_mins(k) + j * step;
                    }
                }
            }

            check_param_bounds();
        }

        // precompute curve parameters for input data points using the chord-length method
        // n-d version of algorithm 9.3, P&T, p. 377
        // params are computed along curves and averaged over all curves at same data point index i,j,k,...
        // ie, resulting params for a data point i,j,k,... are same for all curves
        // and params are only stored once for each dimension (1st dim params, 2nd dim params, ...)
        // total number of params is the sum of ndom_pts over the dimensions, much less than the total
        // number of data points (which would be the product)
        void make_curve_params(const MatrixX<T>&   domain)           // input data points (1st dim changes fastest)
        {
            if (!structured)
            {
                cerr << "Error: Cannot set curve parametrization to unstructured data. Aborting" << endl;
                exit(1);
            }

            T          tot_dist;                          // total chord length
            VectorX<T> dists(ndom_pts.maxCoeff() - 1);    // chord lengths of data point spans for any dim
            param_grid = VectorX<T>::Zero(dom_dim);
            VectorX<T> d;                                 // current chord length

            // following are counters for slicing domain and params into curves in different dimensions
            size_t co = 0;                     // starting offset for curves in domain in current dim
            size_t cs = 1;                     // stride for domain points in curves in current dim

            param_grid.resize(dom_dim);
            for (size_t k = 0; k < ndom_pts.size(); k++)         // for all domain dimensions
            {
                param_grid[k].resize(ndom_pts(k));
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
                        param_grid[k][0]                 = 0.0;      // first parameter is known
                        param_grid[k][ndom_pts(k) - 1]   = 1.0;      // last parameter is known
                        T prev_param                 = 0.0;      // param value at previous iteration below
                        for (size_t i = 0; i < ndom_pts(k) - 2; i++)
                        {
                            T dfrac             = dists(i) / tot_dist;
                            param_grid[k][i + 1]    += prev_param + dfrac;
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
                    param_grid[k][i + 1] /= (ncurves - nzero_length_curves);

                cs *= ndom_pts(k);
            }                                                    // domain dimensions
            // debug
            // print();
            check_param_bounds();
        }

        // If dom_mins/maxs are passed, they define an axis-aligned bounding
        // box with which to compute the domain parametrization. If they are
        // not passed, we assume the bounding box is defined by the extents
        // of the grid.
        // If the data is structured, we always use the grid bounds to 
        // define the extents.
        void make_domain_params(const MatrixX<T>&   domain,
                                const VectorX<T>&   dom_mins = VectorX<T>(),
                                const VectorX<T>&   dom_maxs = VectorX<T>())
        {
            if (structured)
            {
                if (dom_mins == VectorX<T>() || dom_maxs == VectorX<T>())
                {
                    cerr << "Warning: Requested dom_mins, dom_maxs variables are unused in a structured domain parametrization." << endl;
                    cerr << "         Domain bounds will be determine from grid structure." << endl;
                }
                make_domain_params_structured(domain);
            }
            else
            {
                make_domain_params_unstructured(domain, dom_mins, dom_maxs);
            }
        }

        // TODO: Does not properly support geom_dim!
        // This only works if the first dom_dim dimensions of the 
        // geometric coordinates should be used for the parametrization
        // 
        // precompute parameters for input data points using domain spacing only (not length along curve)
        // params are only stored once for each dimension (1st dim params, 2nd dim params, ...)
        // total number of params is the sum of ndom_pts over the dimensions, much less than the total
        // number of data points (which would be the product)
        void make_domain_params_structured(const MatrixX<T>& domain)     // input data points (1st dim changes fastest)
        {
            param_grid.resize(dom_dim);

            int cs = 1;                                      // stride for domain points in current dim.
            for (int k = 0; k < dom_dim; k++)                // for all domain dimensions
            {
                param_grid[k].resize(ndom_pts(k));
                T width_recip = 1 / (domain(cs * (ndom_pts(k) - 1), k) - domain(0, k));

                for (int i = 0; i < ndom_pts(k); i++)
                    param_grid[k][i]= (domain(cs * i, k) - domain(0, k)) * width_recip;

                cs *= ndom_pts(k);
            }

            check_param_bounds();
        }

        // TODO: Does not properly support geom_dim!
        // This only works if the first dom_dim dimensions of the 
        // geometric coordinates should be used for the parametrization
        void make_domain_params_unstructured(
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

            // truncate floating-point roundoffs to [0,1]
            for (int i = 0; i < param_list.rows(); i++)
            {
                for (int j = 0; j < param_list.cols(); j++)
                {
                    if (param_list(i,j) > 1.0)
                    {
                        if (param_list(i,j) - 1.0 < 1e-12)
                        {
                            param_list(i,j) = 1.0;
                            // cerr << "Debug: truncated a parameter value" << endl;
                        }
                        else
                        {
                            cerr << "ERROR: Construction of Param object contains out-of-bounds entries" << endl;
                            cerr << "       Bad Value: " << setprecision(9) << scientific << param_list(i,j) << endl;
                            cerr << "       Out of Tolerance: " << scientific << param_list(i,j) - 1.0 << endl;
                            cerr << i << " " << j << endl;
                            cerr << domain(i,j) << endl;
                            exit(1);
                        }
                    }
                    if (param_list(i,j) < 0.0)
                    {
                        if (0.0 - param_list(i,j) < 1e-12)
                        {
                            param_list(i,j) = 0.0;
                            // cerr << "Debug: truncated a parameter value" << endl;
                        }
                        else
                        {
                            cerr << "ERROR: Construction of Param object contains out-of-bounds entries" << endl;
                            cerr << "       Bad Value: " << setprecision(9) << scientific << param_list(i,j) << endl;
                            cerr << "       Out of Tolerance: " << scientific << 0.0 - param_list(i,j) << endl;
                            exit(1);
                        }
                    }
                }
            }
        
            check_param_bounds();
        }

        // Checks for any parameter values outside the range [0,1].
        // If found, prints an error message and quits the program.
        bool check_param_bounds()
        {
            bool valid = true;
            T minp = 0, maxp = 0;
            T badval = 42;

            if (structured)
            {
                // Check sizes of param_grid are correct
                if (param_grid.size() != dom_dim) valid = false;
                for (int i = 0; i < dom_dim; i++)
                    if (param_grid[i].size() != ndom_pts(i)) valid = false;
                
                if (valid)
                {
                    for (int k = 0; k < dom_dim; k++)
                    {
                        for (int j = 0; j < ndom_pts(k); j++)
                        {
                            if (param_grid[k][j] < 0.0 || param_grid[k][j] > 1.0)
                            {
                                valid = false;
                                badval = param_grid[k][j];
                                break;
                            }
                        }
                    }
                }
            }
            else
            {
                for (int k = 0; k < dom_dim; k++)
                {
                    minp = param_list.col(k).minCoeff();
                    if (minp < 0.0)
                    {
                        valid = false;
                        badval = minp;
                        break;
                    }

                    maxp = param_list.col(k).maxCoeff();
                    if (maxp > 1.0)
                    {
                        valid = false;
                        badval = maxp;
                        break;
                    }
                }
            }
            
            if (valid == false)
            {
                if (badval == 42)
                {
                    cerr << "ERROR: Param object contains invalid param_grid" << endl;
                }
                else
                {
                    cerr << "ERROR: Construction of Param object contains out-of-bounds entries" << endl;
                    cerr << "       Bad Value: " << setprecision(9) << scientific << badval << endl;
                    if (badval > 1.0)
                        cerr << "       Out of Tolerance: " << scientific << badval - 1.0 << endl;
                    else if (badval < 0.0)
                        cerr << "       Out of Tolerance: " << scientific << 0.0 - badval << endl;
                }
                exit(1);
            }

            return valid;
        }

        // Print all parameters (for debugging)
        void print()
        {
            cerr << "----- params -----" << endl;
            if (structured)
            {
                for (int i = 0; i < param_grid.size(); i++)
                {
                    cerr << "Dimension " << i << ":" << endl;
                    for (int j = 0; j < param_grid[i].size(); j++)
                        cerr << "params[" << i << "][" << j << "] = " << param_grid[i][j] << endl;
                }
            }
            else
            {
                cerr << param_list << endl;
            }
            cerr << "------------------" << endl;
        }
    };  // struct Param
}  // namespace mfa

#endif  // _PARAMS_HPP
