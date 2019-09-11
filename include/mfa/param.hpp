//--------------------------------------------------------------
// parameterization object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _PARAMS_HPP
#define _PARAMS_HPP

#include    <Eigen/Dense>
#include    <vector>
#include    <list>
#include    <iostream>

#ifndef MFA_NO_TBB
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
        vector<vector<T>>       params;             // parameters for input points[dimension][index]
        T                       range_extent;       // extent of range value of input data points
        vector<vector<size_t>>  co;                 // starting offset for curves in each dim
        vector<size_t>          ds;                 // stride for domain points in each dim
        int                     dom_dim;            // dimensionality of domain

        Param(
            int                 dom_dim_,           // domain dimensionality (excluding science variables)
            const VectorXi&     ndom_pts_,          // number of input data points in each dim
            const MatrixX<T>&   domain_) :          // input data points (1st dim changes fastest)
            dom_dim(dom_dim_),
            ndom_pts(ndom_pts_)
        {
            // check dimensionality for sanity
            assert(dom_dim < domain_.cols());

            // max extent of input data points
            int last     = domain_.cols() - 1;
            range_extent = domain_.col(last).maxCoeff() - domain_.col(last).minCoeff();

            // stride for domain points in different dimensions
            ds.resize(dom_dim, 1);
            for (size_t i = 1; i < dom_dim; i++)
                ds[i] = ds[i - 1] * ndom_pts_[i - 1];

            // precompute curve parameters and knots for input points
            params.resize(dom_dim);

#ifdef CURVE_PARAMS
            CurveParams(domain_, params);           // params spaced according to the curve length (per P&T)
#else
            DomainParams(domain_, params);          // params spaced according to domain spacing
#endif

            // debug
//             cerr << "Params:\n" << params_ << endl;

            // offsets for curve starting (domain) points in each dimension
            co.resize(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                size_t ncurves  = domain_.rows() / ndom_pts_(k);    // number of curves in this dimension
                size_t coo      = 0;                                // co at start of contiguous sequence
                co[k].resize(ncurves);

                co[k][0] = 0;

                for (auto j = 1; j < ncurves; j++)
                {
                    // adjust offsets for the next curve
                    if (j % ds[k])
                        co[k][j] = co[k][j - 1] + 1;
                    else
                    {
                        co[k][j] = coo + ds[k] * ndom_pts_(k);
                        coo = co[k][j];
                    }
                }
            }
        }

        // precompute curve parameters for input data points using the chord-length method
        // n-d version of algorithm 9.3, P&T, p. 377
        // params are computed along curves and averaged over all curves at same data point index i,j,k,...
        // ie, resulting params for a data point i,j,k,... are same for all curves
        // and params are only stored once for each dimension (1st dim params, 2nd dim params, ...)
        // total number of params is the sum of ndom_pts over the dimensions, much less than the total
        // number of data points (which would be the product)
        // assumes params were allocated by caller
        void CurveParams(
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
        void DomainParams(
                const MatrixX<T>&     domain,                   // input data points (1st dim changes fastest)
                vector<vector<T>>&    params)                   // (output) parameters for input points[dimension][index]
        {
            size_t cs = 1;                                      // stride for domain points in current dim.
            for (size_t k = 0; k < dom_dim; k++)                // for all domain dimensions
            {
                params[k].resize(ndom_pts(k));
                for (size_t i = 1; i < ndom_pts(k) - 1; i++)
                    params[k][i]= fabs( (domain(cs * i, k) - domain(0, k)) /
                            (domain(cs * (ndom_pts(k) - 1), k) - domain(0, k)) );

                params[k][ndom_pts(k) - 1] = 1.0;
                cs *= ndom_pts(k);
            }                                                    // domain dimensions

            // debug
            //     cerr << "params:\n" << params << endl;
        }
    };

}                                               // namespace

#endif
