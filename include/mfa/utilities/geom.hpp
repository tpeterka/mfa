//--------------------------------------------------------------
// Geometry utilities for MFA
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_GEOM_HPP
#define _MFA_GEOM_HPP

#include <mfa/pointset.hpp>
#include <mfa/utilities/util.hpp>
#include <mfa/types.hpp>

namespace mfa
{
    template <typename T>
    struct PointSet;

    // Object representing a bounding box that may be rotated and translated in space.
    // This object can apply the rotation transformation to a given set of points
    // (it can also apply the inverse transformation). When there is no rotation of the
    // box, both of these transformations avoid multiplying by the rotation matrix.
    template <typename T>
    struct Bbox
    {
        int         geomDim;
        VectorX<T>  mins;       // Minimal corner of bounding box
        VectorX<T>  maxs;       // Maximal corner of bounding box
        VectorX<T>  rotatedMins;
        VectorX<T>  rotatedMaxs;
        MatrixX<T>  basis;      // Orthonormal vectors defining the reference frame (each column is a vector)
        bool        aligned;    // Flag if box is aligned to cardinal axes
    
    public:
        // Default constructor
        Bbox() :
            geomDim(0),
            aligned(true)
        { }

        // Constructor for box with no rotation
        Bbox(const VectorX<T>& mins_, const VectorX<T>& maxs_) :
            Bbox(mins_, maxs_, MatrixX<T>::Identity(mins_.size()))
        { }

        // Constructor for box with no rotation, bounds inferred from data
        Bbox(const PointSet<T>& ps) :
            Bbox(MatrixX<T>::Identity(ps.geom_dim()), ps)
        { }

        // Construct with minimum corner, maximum corner, and orientation
        Bbox(const VectorX<T>& mins_, const VectorX<T>& maxs_, const MatrixX<T>& basis_) :
            geomDim(mins_.size()),
            mins(mins_),
            maxs(maxs_),
            basis(basis_),
            aligned(basis_.isIdentity())
        { 
            if (mins.size() != maxs.size()) throw MFAError("min/max dimension mismatch in Bbox ctor");
            
            makeOrthonormal();

            // compute quantities in rotation space
            toRotatedSpace(mins, rotatedMins);
            toRotatedSpace(maxs, rotatedMaxs);
        }

        // Construct with orientation and a dataset
        // Computes the min/max corners from the data
        Bbox(const MatrixX<T>& basis_, const PointSet<T>& ps) :
            Bbox(basis_, ps.domain)
        { }

        // Construct from orientation and list of points to bound
        Bbox(const MatrixX<T>& basis_, const MatrixX<T>& points) :
            geomDim(basis_.rows()),
            basis(basis_),
            aligned(basis_.isIdentity())
        {
            if (basis.rows() != basis.cols()) throw MFAError("Incorrect basis size in Bbox constructor");

            makeOrthonormal();
            setBounds(points);
        }

        // Construct when basis is given by list of vectors
        Bbox(std::initializer_list<VectorX<T>> basis_, const PointSet<T>& ps) :
            Bbox(basis_, ps.domain)
        { }

        // Convenience constructor if basis is specified from a list of vectors
        Bbox(std::initializer_list<VectorX<T>> basis_, const MatrixX<T>& points) :
            geomDim(basis_.size())
        {
            if (geomDim != basis_.begin()->size()) throw MFAError("Bbox dimension mismatch\n");

            basis.resize(geomDim, geomDim);

            int i = 0;
            for (auto vec : basis_)
            {
                basis.col(i) = vec;
                i++;
            }

            // Check if box is axis-aligned
            aligned = basis.isIdentity();

            makeOrthonormal();                  // ensure orthonormal system
            setBounds(points);                      // compute min/max corners from data
        }

        // Rescales a set of basis vectors to be orthonormal
        // If rescaling takes place, an info message is printed
        // If vectors are not orthogonal to begin with, a runtime error is raised
        void makeOrthonormal()
        {
            bool adjusted = false;

            for (int i = 0; i < basis.cols(); i++)
            {
                if (!Eigen::internal::isApprox(basis.col(i).squaredNorm(), 1.0))
                {
                    basis.col(i).normalize();
                    adjusted = true;
                }

                for (int j = 0; j < i; j++)
                {
                    if (!basis.col(i).isOrthogonal(basis.col(j))) 
                    {
                        throw MFAError("Bbox basis cannot be made orthonormal by rescaling");
                    }
                }
            }

            if (adjusted)
            {
                fmt::print("MFA (info): Bbox basis adjusted by makeOrthonormal\n");
            }
        }

        // Solves basis*w = x for w
        // 
        // Note: We assume the basis is orthonormal. So, basis^-1 = basis^T
        //
        // Templated with Eigen-speak to allow matrices/submatrices to be passed in addition to vectors.
        // w is passed as a const-reference, and then the constness is cast away. If we pass
        // w by non-const reference, and then use an expression as input like:
        //   changeBasis(x, a+b)
        // then the compiler will have to evaluate the addition expression into a temporary, which is
        // not always ideal. (Again, this evaluation to a temporary only occurs when passing by non-const
        // reference). See https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
        template <typename Derived, typename OtherDerived>
        void toRotatedSpace(const Eigen::MatrixBase<Derived>& x, const Eigen::MatrixBase<OtherDerived>& w_) const
        {
            if (!basis.isUnitary()) throw MFAError("Bbox basis is not unitary, cannot change basis");
            if (basis.rows() != x.rows()) throw MFAError("Incompatible matrix dimensions in Bbox::toRotatedSpace");

            // Remove const-ness and resize (we explicityly resize w because it is a MatrixBase<> object,
            // which does not get automatically resized during assignment)
            Eigen::MatrixBase<OtherDerived>& w = const_cast<Eigen::MatrixBase<OtherDerived>&>(w_);
            w.derived().resize(x.rows(), x.cols());
            
            if (aligned) // case where basis is the identity
            {
                w = x;
            }
            else
            {
                w = basis.transpose()*x;
            }
        }

        // Given a set of rotated coordinates, convert them to Cartesian (xyz) coordinates.
        // This is the inverse of changeBasis()
        template <typename Derived, typename OtherDerived>
        void toCartesian(const Eigen::MatrixBase<Derived>& w, const Eigen::MatrixBase<OtherDerived>& x_) const
        {
            if (!basis.isUnitary()) throw MFAError("Bbox basis is not unitary, cannot change basis");
            if (basis.cols() != w.rows()) throw MFAError("Incompatible matrix dimensions in Bbox::toRotatedSpace");

            // Remove const-ness and resize (we explicityly resize w because it is a MatrixBase<> object,
            // which does not get automatically resized during assignment)
            Eigen::MatrixBase<OtherDerived>& x = const_cast<Eigen::MatrixBase<OtherDerived>&>(x_);
            x.derived().resize(w.rows(), w.cols());

            if (aligned)
            {
                x = w;
            }
            else
            {
                x = basis*w;
            }
        }

        // Compute box bounds from a set of points.
        // We assume that points is a NxD matrix, where D >= geomDim 
        // and each row represents a different point. The first geomDim 
        // columns are assumed to contain the point coordinates. This is the
        // exact structure given by the matrix PointSet::domain.
        void setBounds(const MatrixX<T>& points)
        {
            if (aligned)    // if basis==I, there is no rotation
            {
                rotatedMins = points.leftCols(geomDim).transpose().rowwise().minCoeff();
                rotatedMaxs = points.leftCols(geomDim).transpose().rowwise().maxCoeff();
                mins = rotatedMins;
                maxs = rotatedMaxs;
            }
            else
            {
                // Compute point locations in rotated basis
                MatrixX<T> skewCoords;
                toRotatedSpace(points.leftCols(geomDim).transpose(), skewCoords);

                // Compute corners in rotated coordinates
                rotatedMins = skewCoords.rowwise().minCoeff();
                rotatedMaxs = skewCoords.rowwise().maxCoeff();

                // Transform corners back to Cartesian coordinates
                toCartesian(rotatedMins, mins);
                toCartesian(rotatedMaxs, maxs);
            }

        }

        // Tests if the bounding box contains every point in a PointSet
        bool doesContain(const PointSet<T>& ps, int verbose = 0, T prec = 1e-12) const
        {
            VectorX<T> dataMins, dataMaxs;
            if (aligned)
            {
                dataMins = ps.domain.leftCols(geomDim).transpose().rowwise().minCoeff();
                dataMaxs = ps.domain.leftCols(geomDim).transpose().rowwise().maxCoeff();
            }
            else
            {
                MatrixX<T> skewCoords;
                toRotatedSpace(ps.domain.leftCols(geomDim).transpose(), skewCoords);
    
                dataMins = skewCoords.rowwise().minCoeff();
                dataMaxs = skewCoords.rowwise().maxCoeff();
            }
            
            // Component wise comparison between dataMins/Maxs and box mins/maxs
            // If any component test fails, there is some point not contained in the box
            if ((dataMins.array() < rotatedMins.array() - prec).any() || 
                    (dataMaxs.array() > rotatedMaxs.array() + prec).any())
            {
                if (verbose >= 2)
                {
                    fmt::print("Bbox::doesContain failed: \n");
                    fmt::print("  rotated data mins: {}\n", dataMins.transpose());
                    fmt::print("  rotated box mins:  {}\n", rotatedMins.transpose());
                    fmt::print("  rotated data maxs: {}\n", dataMaxs.transpose());
                    fmt::print("  rotated box maxs:  {}\n", rotatedMaxs.transpose());
                }

                return false;
            }

            return true;
        }

        // Given another box with the same orientation, returns the smallest bounding box
        // with the same orientation that contains both boxes.
        Bbox<T> merge(const Bbox<T>& other) const
        {
            if (!basis.isApprox(other.basis))
            {
                throw MFAError("Bounding box orientations do not match in Bbox::merge");
            }

            // We have to take care here because *this and other have the same orientation
            // but different minimum corners. (n.b. the min corner defines the translation
            // inside the affine map). So, we start by mapping other's corners
            // into *this's rotation space.
            VectorX<T> otherRotatedMins, otherRotatedMaxs;
            toRotatedSpace(other.mins, otherRotatedMins);
            toRotatedSpace(other.maxs, otherRotatedMaxs);

            // Next, compute the superset in rotated space
            VectorX<T> mergeRotatedMins = rotatedMins.cwiseMin(otherRotatedMins);
            VectorX<T> mergeRotatedMaxs = rotatedMaxs.cwiseMax(otherRotatedMaxs);

            // Finally, transform these new points back into Cartesian coords
            // taking care to use *this's mapping, like before
            VectorX<T> mergeMins, mergeMaxs;
            toCartesian(mergeRotatedMins, mergeMins);
            toCartesian(mergeRotatedMaxs, mergeMaxs);

            // Construct the new bounding box
            return Bbox<T>(mergeMins, mergeMaxs, basis);
        }

        // Print basic information
        void print(string title = "box") const
        {
            fmt::print("{} minimum corner: {}\n", title, print_vec(mins));
            fmt::print("{} maximum corner: {}\n", title, print_vec(maxs));
            fmt::print("{} orientation:\n", title);
            fmt::print("{}\n", print_mat(basis));
        }
    };

    // In the case where the data define a 2D surface embedded in 3D space, it is common
    // that the normal to this surface is not aligned with a cardinal direction (x, y, or z).
    // However, an easy way to parametrize data on a "loosely" planar surface is to project
    // the points on to a plane and compute their parameterization from the location of 
    // of their projection within the plane. However, if the projection plane is not 
    // roughly parallel to the surface defined by the data, this parameterization can fail.
    //
    // This method very roughly determines an "average" normal direction to the data
    // surface, which can be used to define a plane for parameterization of the data.
    template <typename T>
    VectorX<T> estimateSurfaceNormal(const MatrixX<T>& pts)
    {
        assert(pts.cols() == 3);

        int nsamples = 10000;   // chosen empirically 
        VectorX<T> normalAvg = VectorX<T>::Zero(3);

        // Randomly generate integers up to the number of points in the PointSet
        random_device dev;
        mt19937 rng(dev());
        uniform_int_distribution<std::mt19937::result_type> dist(0, pts.rows() - 1);

        int id0, id1, id2;
        Vector3<T> p0, p1, p2, v1, v2, normal;
        for (int i = 0; i < nsamples; i++)
        {
            id0 = dist(rng);
            id1 = dist(rng);
            id2 = dist(rng);
            p0 = pts.row(id0);
            p1 = pts.row(id1);
            p2 = pts.row(id2);
            v1 = p1 - p0;
            v2 = p2 - p0;

            normal = v1.cross(v2).normalized();

            // Ensure that we are always average normal vectors with the same orientation
            if (normal.dot(normalAvg) < 0)
            {
                normal = -1 * normal;
            }

            normalAvg = normalAvg + normal;
        }

        normalAvg = 1.0/nsamples * normalAvg;
        normalAvg.normalize();

        return normalAvg;
    }

    // Given a surface normal, generate two perpendicular vectors within the plane
    template <typename T>
    pair<VectorX<T>, VectorX<T>> getPlaneVectors(const VectorX<T>& normal)
    {
        Vector3<T> a(3), b(3);  // spanning vectors to be computed
        Vector3<T> n = normal.normalized();

        // There are in infinite number of plane-spanning vectors we can choose.
        // For now, consider a spherical coordinate system, and choose one
        // vector to be the azimuthal direction. Then choose the second vector
        // accordingly.
        T nx = n(0), ny = n(1), nz = n(2);
        if (nz < 0.95)  // n is not close to being z-aligned, compute the azimuthal direction
        {
            T phi = asin(nz);

            // In the spherical coordinate system,
            // n = [cos(theta)*cos(phi), sin(theta)*cos(phi), sin(phi)]
            // a = [-1*cos(theta)*sin(phi), -1*sin(theta)*sin(phi), cos(phi)]
            //   = [-1*(nx/cos(phi))*sin(phi), -1*(ny/cos(phi))*sin(phi), cos(phi)]
            //   = [-1*nx*tan(phi), -1*ny*tan(phi), cos(phi)]
            //
            // NOTE: We intentionally do NOT use arccos() in the definition of vector a, because
            //       the range of arccos() is [0, pi], but we need to consider angles in the full
            //       circle [0, 2pi]
            a << -1*nx*tan(phi), -1*ny*tan(phi), cos(phi); 
            a.normalize();
        }
        else        // Gram-Schmidt orthogonalization starting with (1,0,0)
        {
            a << 1, 0, 0;
            a -= nx*n;
            a.normalize();
        }
       
        b = n.cross(a);

        // debug
        if (fabs(a.norm() - 1.0) > 1e-12) throw MFAError(fmt::format("a norm = {}\n", a.norm()));
        if (fabs(b.norm() - 1.0) > 1e-12) throw MFAError(fmt::format("b norm = {}\n", b.norm()));
        if (fabs(n.norm() - 1.0) > 1e-12) throw MFAError(fmt::format("n norm = {}\n", n.norm()));
        if (fabs(a.dot(b)) > 1e-12) throw MFAError(fmt::format("a.b value = {}\n", fabs(a.dot(b))));
        if (fabs(a.dot(n)) > 1e-12) throw MFAError(fmt::format("a.n value = {}\n", fabs(a.dot(n))));
        if (fabs(b.dot(n)) > 1e-12) throw MFAError(fmt::format("b.n value = {}\n", fabs(b.dot(n))));

        return make_pair(a,b);
    }
}

#endif // _MFA_GEOM_HPP