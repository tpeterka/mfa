//--------------------------------------------------------------
// parameterization object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _PARAMS_HPP
#define _PARAMS_HPP

#include    <mfa/utilities/geom.hpp>
#include    <mfa/utilities/util.hpp>

#ifdef MFA_TBB
#include    <tbb/tbb.h>
using namespace tbb;
#endif

namespace mfa
{
    // Parametrization function representing a rotated rectangle. Essentially, this is a
    // affine transformation with no shearing.
    template <typename T>
    struct BoxMap
    {
        const int domDim;
        const int geomDim;
        const Bbox<T> box;
        const VectorX<T> extentsRecip;

        BoxMap(int domDim_, const Bbox<T>& box_) :
            domDim(domDim_),
            geomDim(box_.geomDim),
            box(box_),
            extentsRecip((box_.rotatedMaxs-box_.rotatedMins).cwiseInverse())
        {
            if (geomDim != domDim) throw MFAError("Incorrect dimensions in BoxMap");
        }

        // Compute the parameterizations for a collection of points.
        // 
        // If transpose==false, x is a matrix where each column is the geometric coordinates of a point to be parameterized
        //                      x is geom_dim-by-N, where N is the number of points
        //                      u is returned as a dom_dim-by-N matrix
        // If transpose==true, x is a matrix where each row is the geometric coordinates of a point to be parameterized
        //                     x is N-by-geom_dim
        //                     u is returned as an N-by-dom_dim matrix
        template <typename Derived, typename OtherDerived>
        void transform(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<OtherDerived>& u_, bool transpose = false) const
        {
            Eigen::MatrixBase<OtherDerived>& u = const_cast<Eigen::MatrixBase<OtherDerived>&>(u_);

            // create transpose views for solving (n.b. these are views, so no data movement)
            auto uT = u.transpose();
            auto xT = x.transpose();

            if (transpose)
            {
                box.toRotatedSpace(xT, uT);
                uT = extentsRecip.asDiagonal() * (uT.colwise() - box.rotatedMins);
            }
            else
            {
                box.toRotatedSpace(x, u);
                u = extentsRecip.asDiagonal() * (u.colwise() - box.rotatedMins);
            }
        }
    };

    // Parameterization where coordinates are mapped to a skew box and then a dimension (usually the final dimension) is ignored
    template <typename T>
    struct BoxMapProjected
    {
        const int domDim;
        const int geomDim;
        const int flattenDim;
        BoxMap<T> boxmap;
        std::vector<int> indices;

        BoxMapProjected(int domDim_, const Bbox<T>& box_, int flattenDim_) :
            domDim(domDim_),
            geomDim(box_.geomDim),
            boxmap(domDim_ + 1, box_),
            flattenDim(flattenDim_)
        { 
            if (geomDim != domDim + 1) throw MFAError("Incorrect dimensions in BoxMapProjected");

            // indices[i] describes which dimensions to keep when flattening
            indices.resize(domDim);
            for (int i = 0; i < domDim; i++)
            {
                if (i < flattenDim) indices[i] = i;
                if (i >= flattenDim) indices[i] = i+1;
            }
        }

        // Convenience overload for squashing the final dimension
        BoxMapProjected(int domDim_, const Bbox<T>& box_) :
            BoxMapProjected(domDim_, box_, domDim_)
        { }

        // Parameterize x with a box parameterization, and then delete a dimension
        //
        // This could be faster by only computing the desired dimensions in the first place,
        // but it is likely not worth the speedup
        template <typename Derived, typename OtherDerived>
        void transform(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<OtherDerived>& u_, bool transpose = false) const
        {
            // Temporary matrix to store parameters before removing a dimension
            MatrixX<T> temp;

            // Cast const-ness off of u_
            Eigen::MatrixBase<OtherDerived>& u = const_cast<Eigen::MatrixBase<OtherDerived>&>(u_);

            // Compute transpose views before calling boxmap.transform()
            auto uT = u.transpose();
            auto xT = x.transpose();
            auto tempT = temp.transpose();

            if (transpose)
            {
                boxmap.transform(xT, tempT, false);     // Don't do any further transpose
                uT = tempT(indices, Eigen::all);        // Copy over a subset of columns
            }
            else
            {
                boxmap.transform(x, temp, false);       // Don't do any further transpose
                u = temp(indices, Eigen::all);          // Copy over a subset of columns
            }
        }
    };

    // Parametrization function representing a general affine transformation
    template <typename T>
    struct AffMap
    {
        // Defines an affine transformation from parameter space to physical space
        // x = tMat*u + tVec
        // u = tMat^-1*(x-tVec)
        int                     dom_dim;    // dimension of parameter space
        int                     geom_dim;   // dimension of physical space
        MatrixX<T>              mat;        // Affine transform matrix mapping parameters to physical coords
        VectorX<T>              vec;        // Affine tranform vector mapping parameters to physical coords
        bool                    init{false};// flag that transformation has been initialized

        Eigen::ColPivHouseholderQR<MatrixX<T>> qr;

        AffMap(int dom_dim_, int geom_dim_, const MatrixX<T>& domain, const VectorXi& ndom_pts) :
            dom_dim(dom_dim_),
            geom_dim(geom_dim_)
        {
            // Helper class to manage grid indices
            GridInfo grid;
            grid.init(dom_dim, ndom_pts);

            // Set translation vector for affine transform
            vec = domain.row(0).head(geom_dim);

            // Set linear operator for affine transform
            mat.resize(geom_dim, dom_dim);
            for (int i = 0; i < dom_dim; i++)
            {
                // Get cardinal direction vectors; e.g. (1,0,0), (0,1,0), (0,0,1)
                VectorXi ijk = VectorXi::Zero(dom_dim);
                ijk(i) = ndom_pts(i) - 1;

                // Get physical point at this vector
                int idx = grid.ijk2idx(ijk);
                VectorX<T> edge = domain.row(idx).head(geom_dim);

                mat.col(i) = edge - vec;
            }

            qr = mat.colPivHouseholderQr();

            // Mark transformation as initialized
            init = true;
        }

        AffMap(int dom_dim_, int geom_dim_, const VectorX<T>& vec_, const MatrixX<T>& mat_) :
            dom_dim(dom_dim_),
            geom_dim(geom_dim_),
            vec(vec_),
            mat(mat_)
        {
            qr = mat.colPivHouseholderQr();
            init = true;
        }

        // Computes the parameter u corresponding to point x
        // 
        // Note: In cases where the physical space has higher dimension than paramter space
        //       (e.g., a 2D surface embedded in 3D space), we should only attempt to 
        //       compute parameter values for points that lie on the affine surface.
        //       However, this method ALWAYS produces an answer, even for points not on
        //       the surface. For efficiency, we only check if our answer is valid with
        //       an assert (that is, in a Debug build). So, this method assumes that the 
        //       user is passing in a valid value for x.
        void transform(const VectorX<T>& x, VectorX<T>& u)
        {
            assert(init);
            u = qr.solve(x-vec);
            assert(x.isApprox(mat*u + vec));
        }

        // We want to create a whole new function here (not overload the transform function)
        // because a Vector and a RowVector can both be interpreted as a Matrix. However, 
        // it is likely that a user may extract a row vector from a matrix and pass it 
        // to transform().  In this case, we want to treat it as a (column) Vector.
        // If we had a function overload with Matrix inputs, Eigen could interpret that 
        // row vector as a 1xN matrix, which would cause undefined behavior as we expect
        // x to have 'geom_dim' rows.
        void transformSet(const MatrixX<T>& x, MatrixX<T>& u)
        {
            assert(init);

            // Subtract vec from every row of x
            MatrixX<T> y = x;
            for (int i = 0; i < y.cols(); i++)
            {
                y.col(i) = x.col(i) - vec;
            }

            u = qr.solve(y);
            assert(y.isApprox(mat*u));            
        }

        // Convenience function to transpose matrices before computing parameters
        // No deep copy is made in order to transform
        // 
        // transformSet expects x to be (geom_dim x N) and u to be (dom_dim x N)
        // However, we often store coordinates in matrices of size (N x geom_dim)
        void transformTransposeSet(const MatrixX<T>& x, MatrixX<T>& u)
        {
            assert(init);

            // create transpose views for solving
            Eigen::Transpose<MatrixX<T>> uT = u.transpose();
            Eigen::Transpose<const MatrixX<T>> xT = x.transpose();

            // Subtract vec from every row of xT
            MatrixX<T> yT = xT;
            for (int i = 0; i < yT.cols(); i++)
            {
                yT.col(i) = xT.col(i) - vec;
            }

            uT = qr.solve(yT);
            assert(yT.isApprox(mat*uT));
        }
    };


    template <typename T>                           // float or double
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

        // // Create a parametrization based on physical coordinates of the data.
        // // If dom_mins/maxs are passed, they define an axis-aligned bounding
        // // box with which to compute the domain parametrization. If they are
        // // not passed, we assume the bounding box is defined by the extents
        // // of the grid. 
        // // Note: the dom bounds are only used for unstructured data sets, 
        // //       since domain bounds can be inferred from structured data.
        // void make_domain_params(      int           geom_dim,
        //                         const MatrixX<T>&   domain,
        //                         const VectorX<T>&   dom_mins = VectorX<T>(),
        //                         const VectorX<T>&   dom_maxs = VectorX<T>())
        // {
        //     if (structured)
        //     {
        //         // Warn that bounds are unused when parameterizing structured data
        //         if (dom_mins != VectorX<T>() || dom_maxs != VectorX<T>())
        //         {
        //             cerr << "Warning: dom_mins/maxs variables are unused in a structured domain parametrization." << endl;
        //             cerr << "         Domain bounds will be determine from grid structure." << endl;
        //         }
        //         make_domain_params_structured(geom_dim, domain);
        //     }
        //     else
        //     {
        //         make_domain_params_unstructured(geom_dim, domain, dom_mins, dom_maxs);
        //     }

        //     truncateRoundoff();
        //     checkParamBounds();
        // }

        // Make domain parameterization, domain bounds are computed from the data
        void makeDomainParams(int geomDim, const MatrixX<T>& domain)
        {
            if (structured)
            {
                make_domain_params_structured(geomDim, domain);
            }
            else
            {
                make_domain_params_unstructured(geomDim, domain);
            }

            truncateRoundoff();
            checkParamBounds();
        }

        // Structured data; domain is inferred from the grid
        void make_domain_params_structured(int geom_dim, const MatrixX<T>& domain)
        {
            AffMap<T> map(dom_dim, geom_dim, domain, ndom_pts);
            make_domain_params_structured_impl(map, domain);
        }

        // Make domain parameterization, where the domain is given by `box`
        void makeDomainParams(const Bbox<T>& box, const MatrixX<T>& domain)
        {
            if (structured)
            {
                make_domain_params_structured(box, domain);
            }
            else
            {
                make_domain_params_unstructured(box, domain);
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
        void make_domain_params_structured(const Bbox<T>& box, const MatrixX<T>& domain)
        {
            BoxMap<T> map(dom_dim, box);
            make_domain_params_structured_impl(map, domain);
        }

        void make_domain_params_unstructured(const Bbox<T>& box, const MatrixX<T>& domain)
        {
            if (dom_dim == box.geomDim)
            {
                BoxMap<T> map(dom_dim, box);
                make_domain_params_unstructured_impl(map, domain);
            }
            else if (dom_dim + 1 == box.geomDim) 
            {
                BoxMapProjected<T> map(dom_dim, box);
                make_domain_params_unstructured_impl(map, domain);
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
        void make_domain_params_unstructured(
                  int           geom_dim,
            const MatrixX<T>&   domain)
        {
            if (dom_dim == geom_dim)
            {
                Bbox<T> box(MatrixX<T>::Identity(geom_dim), domain);
                make_domain_params_unstructured(box, domain);
                // BoxMap<T> map(dom_dim, box);    // TODO since this is the identity, replace with a simpler map that does no linear transformation
                // make_domain_params_unstructured_impl(map, domain);
            }
            else if (dom_dim + 1 == geom_dim)
            {
                // Assume the 2D data is roughly planar, and estimate the normal to this plane
                VectorX<T> n, a, b;
                n = estimateSurfaceNormal<T>(domain.leftCols(geom_dim));
                auto vecs = getPlaneVectors<T>(n);
                a = vecs.first;
                b = vecs.second;

                // Create a box oriented to the plane
                Bbox<T> box({a, b, n}, domain);

                make_domain_params_unstructured(box, domain);
                // Parameterize data relative to the plane computed above
                // BoxMapProjected<T> map(dom_dim, box);
                // make_domain_params_unstructured_impl(map, domain);
            }
            else
            {
                throw MFAError("Dimension error in make_domain_params_unstructured");
            }
            
            // // First, determine if dom_mins/maxs were set manually or if
            // // they need to be computed from the domain
            // VectorX<T> mins;
            // VectorX<T> maxs;

            // // dom mins/maxs should either be empty or of size geom_dim
            // if (dom_mins.size() > 0 && dom_mins.size() != geom_dim)
            //     cerr << "Warning: Invalid size of dom_mins in Param construction" << endl;
            // if (dom_maxs.size() > 0 && dom_maxs.size() != geom_dim)
            //     cerr << "Warning: Invalid size of dom_maxs in Param construction" << endl;

            // // Set min/max extents in each domain dimension
            // if (dom_mins.size() != geom_dim || dom_maxs.size() != geom_dim)
            // {
            //     mins = domain.leftCols(geom_dim).colwise().minCoeff();
            //     maxs = domain.leftCols(geom_dim).colwise().maxCoeff();
            // }
            // else    // Use domain bounds provided by block (input data need not extend to bounds)
            // {
            //     mins = dom_mins;
            //     maxs = dom_maxs;
            // }

            // // Create the affine map for the special case of an axis-aligned bounding box
            // VectorX<T> edge(geom_dim);
            // VectorX<T> translation = mins;
            // MatrixX<T> linear(geom_dim, dom_dim);
            // for (int i = 0; i < dom_dim; i++)
            // {
            //     edge.setZero();
            //     edge(i) = maxs(i) - mins(i);
            //     linear.col(i) = edge;
            // }
            // AffMap<T> map(dom_dim, geom_dim, translation, linear);

            // make_domain_params_unstructured_impl(map, domain);
        }

        // Implementation for structured points. P is the type of the parameterization function
        template <typename P>
        void make_domain_params_structured_impl(const P& map, const MatrixX<T>& domain)
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
        void make_domain_params_unstructured_impl(const P& map, const MatrixX<T>& domain)
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
