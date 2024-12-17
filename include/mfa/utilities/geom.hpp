//--------------------------------------------------------------
// Geometry utilities for MFA
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_GEOM_HPP
#define _MFA_GEOM_HPP

#include <mfa/utilities/util.hpp>
#include <mfa/types.hpp>


    #include <bitset>

namespace mfa
{
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
        T phi = asin(nz);
        // T theta = acos(nx/cos(phi)); // do not use theta, see below

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