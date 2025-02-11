//--------------------------------------------------------------
// parameterization functions
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _PARAMETERIZATION_HPP
#define _PARAMETERIZATION_HPP

#include <mfa/types.hpp>
#include <mfa/utilities/util.hpp>

namespace mfa {
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
} // namespace mfa
#endif // _PARAMETERIZATION_HPP