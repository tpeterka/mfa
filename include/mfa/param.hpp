//--------------------------------------------------------------
// parameter container object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _PARAMS_HPP
#define _PARAMS_HPP

#include    <mfa/utilities/geom.hpp>
#include    <mfa/utilities/util.hpp>
#include    <mfa/parameterization.hpp>

#ifdef MFA_TBB
#include    <tbb/tbb.h>
using namespace tbb;
#endif

namespace mfa
{
    template <typename T>
    struct Param
    {
        VectorXi                ndom_pts;           // number of domain points in each dimension
        vector<vector<T>>       param_grid;         // parameters for input points[dimension][index] (for structured case)
        MatrixX<T>              param_list;         // list of parameters for each input pt (for unstructured case)
        int                     dom_dim{0};         // dimensionality of domain
        bool                    structured{false};  // true if points lie on structured grid

        // Convenience constructor that accepts a STL vector
        Param(int dom_dim_, vector<int>& ndom_pts_) :
            Param(dom_dim_, Eigen::Map<VectorXi>(&ndom_pts_[0], ndom_pts_.size()))
        { }

        // General constructor for creating unspecified  params with an optional grid structure
        Param(  int                 dom_dim_,                       // domain dimensionality
                const VectorXi&     ndom_pts_ = VectorXi()) :     // number of input data points in each dim (optional)

            dom_dim(dom_dim_),
            ndom_pts(ndom_pts_),
            structured(ndom_pts_.size() > 0)
        { 
            if (structured)
            {
                if (ndom_pts.size() != dom_dim)
                {
                    cerr << "ERROR: Dimension mismatch in Param constructor.\nExiting." << endl;
                    exit(1);
                }
                
                param_grid.resize(dom_dim);
                for (int i = 0; i < dom_dim; i++)
                {
                    param_grid[i].resize(ndom_pts(i));
                }
            }
        }

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
        void pt_params(const VolIterator& it, VectorX<T>& ret) const
        {
            for(int i = 0; i < dom_dim; i++)
            {
                ret(i) = param_grid[i][it.idx_dim(i)];
            }
        }

        // Structured data only.
        // Get params from param indices in each dimension
        void pt_params(const VectorXi& idxs, VectorX<T>& ret) const
        {
            for(int i = 0; i < dom_dim; i++)
            {
                ret(i) = param_grid[i][idxs(i)];
            }
        }

        // Unstructured data only.
        // Get params from linear index
        // TODO: Allow this to work for structured data to?
        //       Would need a GridInfo class member to 
        //       convert linear indices to ijk-format
        void pt_params(int i, VectorX<T>& ret) const
        {
            for (int j = 0; j < dom_dim; j++)
            {
                ret(j) = param_list(i, j);
            }
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

            for (int k = 0; k < dom_dim; k++)
            {
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

            checkParamBounds();
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
            VectorX<T> d;                                 // current chord length

            // following are counters for slicing domain and params into curves in different dimensions
            size_t co = 0;                     // starting offset for curves in domain in current dim
            size_t cs = 1;                     // stride for domain points in curves in current dim

            for (size_t k = 0; k < ndom_pts.size(); k++)         // for all domain dimensions
            {
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
            checkParamBounds();
        }

        // Make domain parameterization, domain bounds are computed from the data
        void makeDomainParams(int geomDim, const MatrixX<T>& domain)
        {
            if (structured)
            {
                makeDomainParamsStructured(geomDim, domain);
            }
            else
            {
                makeDomainParamsUnstructured(geomDim, domain);
            }

            truncateRoundoff();
            checkParamBounds();
        }

        // Structured data; domain is inferred from the grid
        void makeDomainParamsStructured(int geomDim, const MatrixX<T>& domain)
        {
            AffMap<T> map(dom_dim, geomDim, domain, ndom_pts);
            makeDomainParamsStructuredImpl(map, domain);
        }

        // Make domain parameterization, where the domain is given by `box`
        void makeDomainParams(const Bbox<T>& box, const MatrixX<T>& domain)
        {
            if (structured)
            {
                makeDomainParamsStructured(box, domain);
            }
            else
            {
                makeDomainParamsUnstructured(box, domain);
            }

            truncateRoundoff();
            checkParamBounds();
        }

        // Convenience operator for min/max vectors. In this case, dom_dim must match the
        // dimensionality of the bounding box. This is to catch errors quickly when a user 
        // doesn't know what they are doing. 
        //  
        // For cases when dom_dim < geom_dim is intended, the user can define a Bbox ahead of 
        // time and pass that in instead.
        void makeDomainParams(const VectorX<T>& mins, const VectorX<T>& maxs, const MatrixX<T>& domain)
        {
            if (dom_dim != mins.size()) throw MFAError("Incorrect dimensions in makeDomainParams");

            makeDomainParams(Bbox<T>(mins, maxs), domain);
        }

        // Structured data; domain is defined by the given bounding box
        void makeDomainParamsStructured(const Bbox<T>& box, const MatrixX<T>& domain)
        {
            BoxMap<T> map(dom_dim, box);
            makeDomainParamsStructuredImpl(map, domain);
        }

        void makeDomainParamsUnstructured(const Bbox<T>& box, const MatrixX<T>& domain)
        {
            if (dom_dim == box.geomDim)
            {
                BoxMap<T> map(dom_dim, box);
                makeDomainParamsUnstructuredImpl(map, domain);
            }
            else if (dom_dim + 1 == box.geomDim) 
            {
                BoxMapProjected<T> map(dom_dim, box);
                makeDomainParamsUnstructuredImpl(map, domain);
            }
            else
            {
                throw MFAError("Dimension error in make_domain_params_unstructured");
            }
        }

        // Parameterize unstructured data, inferring bounds from the data
        // When dom_dim==geom_dim, assume data is oriented to the cardinal axes
        // When dom_dim==geom_dim-1, assume data is planar, approximate the plane, 
        //     and then compute bounds within that plane.
        void makeDomainParamsUnstructured(
                  int           geomDim,
            const MatrixX<T>&   domain)
        {
            if (dom_dim == geomDim)
            {
                Bbox<T> box(MatrixX<T>::Identity(geomDim), domain);
                make_domain_params_unstructured(box, domain);
                // BoxMap<T> map(dom_dim, box);    // TODO since this is the identity, replace with a simpler map that does no linear transformation
                // make_domain_params_unstructured_impl(map, domain);
            }
            else if (dom_dim + 1 == geomDim)
            {
                // Assume the 2D data is roughly planar, and estimate the normal to this plane
                VectorX<T> n = estimateSurfaceNormal<T>(domain.leftCols(geomDim));
                auto [a, b] = getPlaneVectors<T>(n);

                // Create a box oriented to the plane
                Bbox<T> box({a, b, n}, domain);

                make_domain_params_unstructured(box, domain);
            }
            else
            {
                throw MFAError("Dimension error in make_domain_params_unstructured");
            }
        }

        // Implementation for structured points. P is the type of the parameterization function
        template <typename P>
        void makeDomainParamsStructuredImpl(const P& map, const MatrixX<T>& domain)
        {
            // Helper class to manage grid indices
            GridInfo grid;
            grid.init(dom_dim, ndom_pts);

            VectorX<T> x(map.geomDim);
            VectorX<T> u(dom_dim);
            VectorXi ijk(dom_dim);
            int idx = 0;
            for (int i = 0; i < dom_dim; i++)
            {
                for (int j = 0; j < ndom_pts(i); j++)
                {
                    ijk.setZero();
                    ijk(i) = j;
                    idx = grid.ijk2idx(ijk);
                    x = domain.row(idx).head(map.geomDim);

                    map.transform(x, u);
                    param_grid[i][j] = u(i);
                }
            }
        }

        template <typename P>
        void makeDomainParamsUnstructuredImpl(const P& map, const MatrixX<T>& domain)
        {
            // Resize the parameter list and fill
            param_list.resize(domain.rows(), dom_dim);
            map.transform(domain.leftCols(map.geomDim), param_list, true);    // warning: this could resize param_list!
        }

        // truncate floating-point roundoffs to [0,1]
        // unclear what this precision value should be in general
        // have observed cases where floating point error exceeds 1e-12
        void truncateRoundoff(double prec = 1e-10)
        {
            for (int i = 0; i < param_list.rows(); i++)
            {
                for (int j = 0; j < param_list.cols(); j++)
                {
                    if (param_list(i,j) > 1.0)
                    {
                        if (param_list(i,j) - 1.0 < prec)
                        {
                            param_list(i,j) = 1.0;
                            // cerr << "Debug: truncated a parameter value" << endl;
                        }
                        else
                        {
                            cerr << "ERROR: Construction of Param object contains out-of-bounds entries" << endl;
                            cerr << "       Bad Value: " << setprecision(9) << scientific << param_list(i,j) << endl;
                            cerr << "       Out of Tolerance: " << scientific << param_list(i,j) - 1.0 << endl;
                            cerr << "       Index: " << i << " " << j << endl;
                            exit(1);
                        }
                    }
                    if (param_list(i,j) < 0.0)
                    {
                        if (0.0 - param_list(i,j) < prec)
                        {
                            param_list(i,j) = 0.0;
                            // cerr << "Debug: truncated a parameter value" << endl;
                        }
                        else
                        {
                            cerr << "ERROR: Construction of Param object contains out-of-bounds entries" << endl;
                            cerr << "       Bad Value: " << setprecision(9) << scientific << param_list(i,j) << endl;
                            cerr << "       Out of Tolerance: " << scientific << 0.0 - param_list(i,j) << endl;
                            cerr << "       Index: " << i << " " << j << endl;
                            exit(1);
                        }
                    }
                }
            }
        }

        // Checks for any parameter values outside the range [0,1].
        // If found, prints an error message and quits the program.
        bool checkParamBounds()
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
                            if (param_grid[k][j] < 0.0 || param_grid[k][j] > 1.0 || std::isnan(param_grid[k][j]))
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
                if (param_list.cols() != dom_dim) throw MFAError("Incorrect column number in param_list");

                for (int k = 0; k < dom_dim; k++)
                {
                    minp = param_list.col(k).minCoeff();
                    if (minp < 0.0 || std::isnan(minp))
                    {
                        valid = false;
                        badval = minp;
                        break;
                    }

                    maxp = param_list.col(k).maxCoeff();
                    if (maxp > 1.0 || std::isnan(maxp))
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
