// --------------------------------------------------------------
// new knots inserter object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <mfa/new_knots.hpp>
#include <mfa/encode.hpp>
#include <mfa/decode.hpp>
#include <iostream>
#include <set>

template <typename T>                               // float or double
mfa::
NewKnots<T>::
NewKnots(MFA<T>& mfa_) :
    mfa(mfa_),
    max_num_curves(1.0e4)                           // max num. curves to check in one dimension of curve version
{
}

// encodes at full dimensionality and decodes at full dimensionality
// decodes full-d points in each knot span and adds new knot spans where error > err_limit
// returns true if done, ie, no knots are inserted
template <typename T>
bool
mfa::
NewKnots<T>::
NewKnots_full(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
        T              err_limit,                   // max allowable error
        int            iter)                        // iteration number of caller (for debugging)
{
    mfa::Encoder<T> encoder(mfa);

    // resize control points and weights
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());
    mfa.weights = VectorX<T>::Ones(mfa.ctrl_pts.rows());

    // full n-d encoding
    encoder.Encode();

    // find new knots
    nnew_knots = VectorXi::Zero(mfa.p.size());
    new_knots.resize(0);
    bool done = ErrorSpans(nnew_knots, new_knots, err_limit, iter);

    return done;
}

// TBB version
//
// 1d encoding and 1d decoding
// for each dimension, finds worst curve for new knots
// then inserts knots in spans at locations of highest error
//
// returns true if done, ie, no knots are inserted
template <typename T>
bool
mfa::
NewKnots<T>::
NewKnots_curve1(
        VectorXi&      nnew_knots,                       // number of new knots in each dim
        vector<T>&     new_knots,                        // new knots (1st dim changes fastest)
        T              err_limit,                        // max allowable error
        int            iter)                             // iteration number of caller (for debugging)
{
    mfa::Encoder<T> encoder(mfa);

    // check and assign main quantities
    int  ndims = mfa.ndom_pts.size();                   // number of domain dimensions
    VectorXi n = mfa.nctrl_pts - VectorXi::Ones(ndims); // number of control point spans in each domain dim
    VectorXi m = mfa.ndom_pts  - VectorXi::Ones(ndims); // number of input data point spans in each domain dim
    nnew_knots = VectorXi::Zero(mfa.p.size());
    new_knots.resize(0);

    // resize control points and weights
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());
    mfa.weights = VectorX<T>::Ones(mfa.ctrl_pts.rows());

    for (size_t k = 0; k < ndims; k++)              // for all domain dimensions
    {
        // maximum number of domain points with error greater than err_limit
        size_t max_nerr     =  0;

        // hard-coded weights all equal to 1 for now
        VectorX<T> temp_weights = VectorX<T>::Ones(mfa.nctrl_pts(k));

        // compute the matrix N, eq. 9.66 in P&T
        // N is a matrix of (m - 1) x (n - 1) scalars that are the basis function coefficients
        //  _                                _
        // |  N_1(u[1])   ... N_{n-1}(u[1])   |
        // |     ...      ...      ...        |
        // |  N_1(u[m-1]) ... N_{n-1}(u[m-1]) |
        //  -                                -
        // TODO: N is going to be very sparse when it is large: switch to sparse representation
        // N has semibandwidth < p  nonzero entries across diagonal
        MatrixX<T> N = MatrixX<T>::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)                  // the rows of N
        {
            int span = mfa.FindSpan(k, mfa.params(mfa.po[k] + i), mfa.ko[k]) - mfa.ko[k];   // relative to ko
            assert(span <= n(k));            // sanity
            mfa.BasisFuns(k, mfa.params(mfa.po[k] + i), span, N, 1, n(k) - 1, i - 1);
        }

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixX<T> NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // debug
        //         cerr << "k " << k << " NtN:\n" << NtN << endl;

        size_t ncurves         = mfa.domain.rows() / mfa.ndom_pts(k);   // number of curves in this dimension
        int nsame_steps        = 0;                                     // number of steps with same number of erroneous points
        int n_step_sizes       = 0;                                     // number of step sizes so far
        size_t worst_curve_idx = 0;                                     // index of worst curve in this dimension

        // starting step size over curves
        size_t s0 = ncurves / 2 > 0 ? ncurves / 2 : 1;

        // debug, only one step size s=1
//         s0 = 1;

        for (size_t s = s0; s >= 1 && ncurves / s < max_num_curves; s /= 2)  // for all step sizes over curves up to max allowed
        {
            // debug
            fprintf(stderr, "k=%ld s=%ld\n", k, s);

            bool new_max_nerr = false;                      // this step size changed the max_nerr
            size_t ncurves_s  = static_cast<size_t>(ceil(static_cast<T>(ncurves) / s));
            vector<size_t> nerrs(ncurves_s);                    // number of erroneous points (error > err_limit) in the curve

            parallel_for (size_t (0), ncurves_s, [&] (size_t j) // for all the curves in this dimension (given the curve step)
            {
                // R is the right hand side needed for solving NtN * P = R
                MatrixX<T> R(n(k) - 1, mfa.domain.cols());

                // P are the unknown interior control points and the solution to NtN * P = R
                // NtN is positive definite -> do not need pivoting
                // TODO: use a common representation for P and ctrl_pts to avoid copying
                MatrixX<T> P(n(k) - 1, mfa.domain.cols());

                if (N.cols())
                {
                    // compute R from input domain points
                    encoder.RHS(k, N, R, mfa.ko[k], mfa.po[k], mfa.co[k][j * s]);

                    // solve for P for one curve of control points
                    P = NtN.ldlt().solve(R);
                }

                // append points from P to control points
                // TODO: any way to avoid this?
                MatrixX<T> temp_ctrl = MatrixX<T>::Zero(mfa.nctrl_pts(k), mfa.domain.cols());   // temporary control points for one curve
                encoder.CopyCtrl(P, n, k, mfa.co[k][j * s], temp_ctrl);

                // compute the error on the curve (number of input points with error > err_limit)
                nerrs[j] = encoder.ErrorCurve(k, mfa.co[k][j * s], temp_ctrl, temp_weights, err_limit);

            });                                               // parallel for over curves in this dimension

            vector<size_t>::iterator worst_curve = max_element(nerrs.begin(), nerrs.end());
            if (*worst_curve > max_nerr)
            {
                max_nerr        = *worst_curve;
                worst_curve_idx = (worst_curve - nerrs.begin()) * s;
                new_max_nerr    = true;

                // debug
//                 fprintf(stderr, "k=%ld s=%ld worst_curve_idx=%ld nerr=%ld max_nerr=%ld\n",
//                 k, s, worst_curve_idx, nerrs[worst_curve_idx / s], max_nerr);
            }

            // stop refining step if no change
            if (max_nerr && !new_max_nerr)
                nsame_steps++;
            if (nsame_steps == 2)
                break;

            n_step_sizes++;
        }                                               // step sizes over curves

        // --- TODO: change recomputing worst curve to inserting error spans into set ---

        // recompute the worst curve
        set<int> err_spans;                          // error spans for one curve

        // debug
        fprintf(stderr, "k=%ld worst_curve_idx=%ld co=%ld\n", k, worst_curve_idx, mfa.co[k][worst_curve_idx]);

        // R is the right hand side needed for solving NtN * P = R
        MatrixX<T> R(n(k) - 1, mfa.domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixX<T> P(n(k) - 1, mfa.domain.cols());

        if (N.cols())
        {
            // compute R from input domain points
            encoder.RHS(k, N, R, mfa.ko[k], mfa.po[k], mfa.co[k][worst_curve_idx]);

            // solve for P for one curve of control points
            P = NtN.ldlt().solve(R);
        }

        // append points from P to control points
        // TODO: any way to avoid this?
        MatrixX<T> temp_ctrl = MatrixX<T>::Zero(mfa.nctrl_pts(k), mfa.domain.cols());   // temporary control points for one curve
        encoder.CopyCtrl(P, n, k, mfa.co[k][worst_curve_idx], temp_ctrl);

        // --- TODO: end of recomputing worst curve ---

        // compute the new knots on the worst curve in this dimension
        encoder.ErrorCurve(k, mfa.co[k][worst_curve_idx], temp_ctrl, temp_weights, nnew_knots, new_knots, err_limit);

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // print progress
        fprintf(stderr, "\rdimension %ld of %d encoded\n", k + 1, ndims);
    }                                                      // domain dimensions

    mfa.InsertKnots(nnew_knots, new_knots);

    // debug
//     cerr << "\nnnew_knots:\n" << nnew_knots << endl;
//     cerr << "new_knots:\n"  << new_knots  << endl;

    return(nnew_knots.sum() ? 0 : 1);
}

#if 1

// single thread version
//
// 1d encoding and 1d decoding
// adds knots error spans from all curves in all directions (into a set)
// adds knots in middles of spans that have error higher than the limit
// returns true if done, ie, no knots are inserted
template <typename T>
bool
mfa::
NewKnots<T>::
NewKnots_curve(
        VectorXi&      nnew_knots,                       // number of new knots in each dim
        vector<T>&     new_knots,                        // new knots (1st dim changes fastest)
        T              err_limit,                        // max allowable error
        int            iter)                             // iteration number of caller (for debugging)
{
    mfa::Encoder<T> encoder(mfa);

    // check and assign main quantities
    int  ndims = mfa.ndom_pts.size();                   // number of domain dimensions
    VectorXi n = mfa.nctrl_pts - VectorXi::Ones(ndims); // number of control point spans in each domain dim
    VectorXi m = mfa.ndom_pts  - VectorXi::Ones(ndims); // number of input data point spans in each domain dim
    nnew_knots = VectorXi::Zero(mfa.p.size());
    new_knots.resize(0);

    // resize control points and weights
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());
    mfa.weights = VectorX<T>::Ones(mfa.ctrl_pts.rows());

    for (size_t k = 0; k < ndims; k++)              // for all domain dimensions
    {
        // temporary control points for one curve
        MatrixX<T> temp_ctrl = MatrixX<T>::Zero(mfa.nctrl_pts(k), mfa.domain.cols());

        // error spans for one curve and for worst curve
        set<int> err_spans;

        // maximum number of domain points with error greater than err_limit and their curves
        size_t max_nerr     =  0;

        // hard-coded weights all equal to 1 for now
        VectorX<T> temp_weights = VectorX<T>::Ones(mfa.nctrl_pts(k));

        // compute the matrix N, eq. 9.66 in P&T
        // N is a matrix of (m - 1) x (n - 1) scalars that are the basis function coefficients
        //  _                                _
        // |  N_1(u[1])   ... N_{n-1}(u[1])   |
        // |     ...      ...      ...        |
        // |  N_1(u[m-1]) ... N_{n-1}(u[m-1]) |
        //  -                                -
        // TODO: N is going to be very sparse when it is large: switch to sparse representation
        // N has semibandwidth < p  nonzero entries across diagonal
        MatrixX<T> N = MatrixX<T>::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)                  // the rows of N
        {
            int span = mfa.FindSpan(k, mfa.params(mfa.po[k] + i), mfa.ko[k]) - mfa.ko[k];   // relative to ko
            assert(span <= n(k));            // sanity
            mfa.BasisFuns(k, mfa.params(mfa.po[k] + i), span, N, 1, n(k) - 1, i - 1);
        }

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixX<T> NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // R is the right hand side needed for solving NtN * P = R
        MatrixX<T> R(n(k) - 1, mfa.domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixX<T> P(n(k) - 1, mfa.domain.cols());

        size_t ncurves   = mfa.domain.rows() / mfa.ndom_pts(k); // number of curves in this dimension
        int nsame_steps  = 0;                                   // number of steps with same number of erroneous points
        int n_step_sizes = 0;                                   // number of step sizes so far

        // starting step size over curves
        size_t s0 = ncurves / 2 > 0 ? ncurves / 2 : 1;

        // debug, only one step size s=1
//         s0 = 1;

        for (size_t s = s0; s >= 1 && ncurves / s < max_num_curves; s /= 2)        // for all step sizes over curves
        {
            // debug
//             fprintf(stderr, "k=%ld s=%ld\n", k, s);

            bool new_max_nerr = false;                      // this step size changed the max_nerr

            for (size_t j = 0; j < ncurves; j++)            // for all the curves in this dimension
            {
                // each time the step changes, shift start of s-th curves by one (by subtracting
                // n_step-sizes below)
                if (j >= n_step_sizes && (j - n_step_sizes) % s == 0)   // this is one of the s-th curves; compute it
                {
                    if (N.cols())
                    {
                        // compute R from input domain points
                        encoder.RHS(k, N, R, mfa.ko[k], mfa.po[k], mfa.co[k][j]);

                        // solve for P for one curve of control points
                        P = NtN.ldlt().solve(R);
                    }

                    // append points from P to control points
                    // TODO: any way to avoid this?
                    encoder.CopyCtrl(P, n, k, mfa.co[k][j], temp_ctrl);

                    // compute the error on the curve (number of input points with error > err_limit)
                    size_t nerr = encoder.ErrorCurve(k, mfa.co[k][j], temp_ctrl, temp_weights, err_spans, err_limit);

                    if (nerr > max_nerr)
                    {
                        max_nerr     = nerr;
                        new_max_nerr = true;
                    }
                }
            }                                               // curves in this dimension

            // stop refining step if no change
            if (max_nerr && !new_max_nerr)
                nsame_steps++;
            if (nsame_steps == 2)
                break;

            n_step_sizes++;
        }                                                   // step sizes over curves

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // add new knots in the middle of spans with errors
        nnew_knots(k) = err_spans.size();
        auto old_size = new_knots.size();
        new_knots.resize(old_size + err_spans.size());      // existing values are preserved
        size_t i = 0;                                       // index into new_knots
        for (set<int>::iterator it = err_spans.begin(); it != err_spans.end(); ++it)
        {
            // debug
            assert(*it < mfa.nctrl_pts[k]);                          // not trying to go beyond the last span

            new_knots[old_size + i] = (mfa.knots(mfa.ko[k] + *it) + mfa.knots(mfa.ko[k] + *it + 1)) / 2.0;
            i++;
        }

        // print progress
//         fprintf(stderr, "\rdimension %ld of %d encoded\n", k + 1, ndims);
    }                                                      // domain dimensions

    mfa.InsertKnots(nnew_knots, new_knots);

    // debug
//     cerr << "\nnnew_knots:\n" << nnew_knots << endl;
//     cerr << "new_knots:\n"  << new_knots  << endl;

    return(nnew_knots.sum() ? 0 : 1);
}

#else

// TBB version
//
// 1d encoding and 1d decoding
// for each dimension, finds worst curve for new knots (not new knots from all curves)
// this is less accurate than inserting all new knots from all curves into a set (as in the serial
// version)
// TODO: need to figure out how to do this in this version
//
// returns true if done, ie, no knots are inserted
template <typename T>
bool
mfa::
NewKnots<T>::
NewKnots_curve(
        VectorXi&      nnew_knots,                       // number of new knots in each dim
        vector<T>&     new_knots,                        // new knots (1st dim changes fastest)
        T              err_limit,                        // max allowable error
        int            iter)                             // iteration number of caller (for debugging)
{
    mfa::Encoder encoder(mfa);

    // check and assign main quantities
    int  ndims = mfa.ndom_pts.size();                   // number of domain dimensions
    VectorXi n = mfa.nctrl_pts - VectorXi::Ones(ndims); // number of control point spans in each domain dim
    VectorXi m = mfa.ndom_pts  - VectorXi::Ones(ndims); // number of input data point spans in each domain dim
    nnew_knots = VectorXi::Zero(mfa.p.size());
    new_knots.resize(0);

    // resize control points and weights
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());
    mfa.weights = VectorX<T>::Ones(mfa.ctrl_pts.rows());

    for (size_t k = 0; k < ndims; k++)              // for all domain dimensions
    {
        // maximum number of domain points with error greater than err_limit
        size_t max_nerr     =  0;

        // compute the matrix N, eq. 9.66 in P&T
        // N is a matrix of (m - 1) x (n - 1) scalars that are the basis function coefficients
        //  _                                _
        // |  N_1(u[1])   ... N_{n-1}(u[1])   |
        // |     ...      ...      ...        |
        // |  N_1(u[m-1]) ... N_{n-1}(u[m-1]) |
        //  -                                -
        // TODO: N is going to be very sparse when it is large: switch to sparse representation
        // N has semibandwidth < p  nonzero entries across diagonal
        MatrixX<T> N = MatrixX<T>::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)                  // the rows of N
        {
            int span = mfa.FindSpan(k, mfa.params(mfa.po[k] + i), mfa.ko[k]) - mfa.ko[k];   // relative to ko
            assert(span <= n(k));            // sanity
            mfa.BasisFuns(k, mfa.params(mfa.po[k] + i), span, N, 1, n(k) - 1, i - 1);
        }

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixX<T> NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // debug
        //         cerr << "k " << k << " NtN:\n" << NtN << endl;

        size_t ncurves         = mfa.domain.rows() / mfa.ndom_pts(k);   // number of curves in this dimension
        int nsame_steps        = 0;                                     // number of steps with same number of erroneous points
        int n_step_sizes       = 0;                                     // number of step sizes so far
        size_t worst_curve_idx = 0;                                     // index of worst curve in this dimension

        // starting step size over curves
        size_t s0 = ncurves / 2 > 0 ? ncurves / 2 : 1;

        // debug, only one step size s=1
//         s0 = 1;

        for (size_t s = s0; s >= 1 && ncurves / s < max_num_curves; s /= 2)  // for all step sizes over curves up to max allowed
        {
            // debug
            fprintf(stderr, "k=%ld s=%ld\n", k, s);

            bool new_max_nerr = false;                      // this step size changed the max_nerr
            size_t ncurves_s  = static_cast<size_t>(ceil(static_cast<T>(ncurves) / s));
            vector<size_t> nerrs(ncurves_s);                    // number of erroneous points (error > err_limit) in the curve

            parallel_for (size_t (0), ncurves_s, [&] (size_t j) // for all the curves in this dimension (given the curve step)
            {
                // R is the right hand side needed for solving NtN * P = R
                MatrixX<T> R(n(k) - 1, mfa.domain.cols());

                // P are the unknown interior control points and the solution to NtN * P = R
                // NtN is positive definite -> do not need pivoting
                // TODO: use a common representation for P and ctrl_pts to avoid copying
                MatrixX<T> P(n(k) - 1, mfa.domain.cols());

                if (N.cols())
                {
                    // compute R from input domain points
                    RHS(k, N, R, mfa.ko[k], mfa.po[k], mfa.co[k][j * s]);

                    // solve for P for one curve of control points
                    P = NtN.ldlt().solve(R);
                }

                // append points from P to control points
                // TODO: any way to avoid this?
                MatrixX<T> temp_ctrl = MatrixX<T>::Zero(mfa.nctrl_pts(k), mfa.domain.cols());   // temporary control points for one curve
                CopyCtrl(P, n, k, mfa.co[k][j * s], temp_ctrl);

                // compute the error on the curve (number of input points with error > err_limit)
                nerrs[j] = ErrorCurve(k, mfa.co[k][j * s], temp_ctrl, err_limit);

            });                                               // parallel for over curves in this dimension

            vector<size_t>::iterator worst_curve = max_element(nerrs.begin(), nerrs.end());
            if (*worst_curve > max_nerr)
            {
                max_nerr        = *worst_curve;
                worst_curve_idx = (worst_curve - nerrs.begin()) * s;
                new_max_nerr    = true;

                // debug
//                 fprintf(stderr, "k=%ld s=%ld worst_curve_idx=%ld nerr=%ld max_nerr=%ld\n",
//                 k, s, worst_curve_idx, nerrs[worst_curve_idx / s], max_nerr);
            }

            // stop refining step if no change
            if (max_nerr && !new_max_nerr)
                nsame_steps++;
            if (nsame_steps == 2)
                break;

            n_step_sizes++;
        }                                               // step sizes over curves

        // --- TODO: change recomputing worst curve to inserting error spans into set ---

        // recompute the worst curve
        set<int> err_spans;                          // error spans for one curve

        // debug
//         fprintf(stderr, "k=%ld worst_curve_idx=%ld co=%ld\n", k, worst_curve_idx, co);

        // R is the right hand side needed for solving NtN * P = R
        MatrixX<T> R(n(k) - 1, mfa.domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixX<T> P(n(k) - 1, mfa.domain.cols());

        if (N.cols())
        {
            // compute R from input domain points
            RHS(k, N, R, mfa.ko[k], mfa.po[k], mfa.co[k][worst_curve_idx]);

            // solve for P for one curve of control points
            P = NtN.ldlt().solve(R);
        }

        // append points from P to control points
        // TODO: any way to avoid this?
        MatrixX<T> temp_ctrl = MatrixX<T>::Zero(mfa.nctrl_pts(k), mfa.domain.cols());   // temporary control points for one curve
        CopyCtrl(P, n, k, mfa.co[k][worst_curve_idx], temp_ctrl);

        // --- TODO: end of recomputing worst curve ---

        // compute the error spans on the worst curve in this dimension
        ErrorCurve(k, mfa.co[k][worst_curve_idx], temp_ctrl, err_spans, err_limit);

        // add new knots in the middle of spans with errors
        nnew_knots(k) = err_spans.size();
        auto old_size = new_knots.size();
        new_knots.resize(old_size + err_spans.size());    // existing values are preserved
        size_t i = 0;
        for (set<int>::iterator it = err_spans.begin(); it != err_spans.end(); ++it)
        {
            // debug
            assert(*it < mfa.nctrl_pts[k]);                          // not trying to go beyond the last span

            new_knots[old_size + i] = (mfa.knots(mfa.ko[k] + *it) + mfa.knots(mfa.ko[k] + *it + 1)) / 2.0;
            i++;
        }

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // print progress
        fprintf(stderr, "\rdimension %ld of %d encoded\n", k + 1, ndims);
    }                                                      // domain dimensions

    mfa.InsertKnots(nnew_knots, new_knots);

    // debug
//     cerr << "\nnnew_knots:\n" << nnew_knots << endl;
//     cerr << "new_knots:\n"  << new_knots  << endl;

    return(nnew_knots.sum() ? 0 : 1);
}

#endif

// encodes at full dimensionality and decodes in 1d curves
// decodes 1d curves at control points and adds knot spans from all curves in all directions (into a set)
// returns true if done, ie, no knots are inserted
//
// this version produces an excessive number of knots and control points and is not recommended
// remove eventually
template <typename T>
bool
mfa::
NewKnots<T>::
NewKnots_hybrid(
        VectorXi&      nnew_knots,                       // number of new knots in each dim
        vector<T>&     new_knots,                        // new knots (1st dim changes fastest)
        T              err_limit,                        // max allowable error
        int            iter)                             // iteration number of caller (for debugging)
{
    mfa::Encoder<T> encoder(mfa);

    // resize control points and weights
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());
    mfa.weights = VectorX<T>::Ones(mfa.ctrl_pts.rows());

    // full n-d encoding
    encoder.Encode();

    size_t ts = 1;                                              // control point stride

    // find new knots
    nnew_knots = VectorXi::Zero(mfa.p.size());
    new_knots.resize(0);
    for (size_t k = 0; k < mfa.p.size(); k++)                   // for all domain dimensions
    {
        set<int> err_spans;                                     // all error spans so far in this dim.
        size_t ncurves = mfa.ctrl_pts.rows() / mfa.nctrl_pts(k);    // number of curves in this dimension

        // offsets for starting control point for each curve in this dimension
        vector<size_t> to(ncurves);
        size_t too     = 0;                                     // to at start of contiguous sequence
        to[0]          = 0;

        for (auto j = 1; j < ncurves; j++)
        {
            // adjust offsets for the next curve
            if (j % ts)
                to[j] = to[j - 1] + 1;
            else
            {
                to[j] = too + ts * mfa.nctrl_pts(k);
                too   = to[j];
            }
        }

        // find spans with error > err_limit
        for (size_t j = 0; j < ncurves; j++)
            size_t nerr = encoder.ErrorCtrlCurve(k, to[j], err_spans, err_limit);

        // add new knots in the middle of spans with errors
        nnew_knots(k) = err_spans.size();
        auto old_size = new_knots.size();
        new_knots.resize(old_size + err_spans.size());    // existing values are preserved
        size_t i = 0;                                           // index into new_knots
        for (set<int>::iterator it = err_spans.begin(); it != err_spans.end(); ++it)
        {
            // debug
            assert(*it < mfa.nctrl_pts[k]);                     // not trying to go beyond the last span

            new_knots[old_size + i] = (mfa.knots(mfa.ko[k] + *it) + mfa.knots(mfa.ko[k] + *it + 1)) / 2.0;
            i++;
        }

        // print progress
        fprintf(stderr, "found total %d new knots after dimension %ld of %ld\n", nnew_knots.sum(), k + 1, mfa.p.size());

        ts *= mfa.nctrl_pts(k);
    }                                                          // domain dimensions
    fprintf(stderr, "\n");

    mfa.InsertKnots(nnew_knots, new_knots);

    // debug
//     cerr << "\nnnew_knots:\n" << nnew_knots << endl;
//     cerr << "new_knots:\n"  << new_knots  << endl;

    return(nnew_knots.sum() ? 0 : 1);
}

#if 1

// TBB version
// computes error in knot spans
// marks the knot spans that are done (error <= max_error in the entire span)
// assumes caller allocated new_knots to number of spans and nnew_knots to domain dimensions
// (does no resizing of new_knots and nnew_knots) and zeroed nnew_knots
// returns true if all done, ie, no new knots inserted
template <typename T>
bool
mfa::
NewKnots<T>::
ErrorSpans(
        VectorXi&      nnew_knots,                          // number of new knots in each dim
        vector<T>&     new_knots,                           // new knots (1st dim changes fastest)
        T              err_limit,                           // max allowable error
        int            iter)                                // iteration number
{
    mfa::Decoder<T> decoder(mfa);

    // initialize all knot spans to not done
    for (auto i = 0; i < mfa.knot_spans.size(); i++)
        mfa.knot_spans[i].done = false;

    // spans that have already been split in this round (to prevent splitting twice)
    vector<bool> split_spans(mfa.knot_spans.size());                // intialized to false by default

    parallel_for(size_t(0), mfa.knot_spans.size(), [&] (size_t i)          // knot spans
    {
        if (!mfa.knot_spans[i].done)
        {
            size_t nspan_pts = 1;                                   // number of domain points in the span
            for (auto k = 0; k < mfa.p.size(); k++)
                nspan_pts *= (mfa.knot_spans[i].max_param_ijk(k) - mfa.knot_spans[i].min_param_ijk(k) + 1);

            VectorXi p_ijk = mfa.knot_spans[i].min_param_ijk;           // indices of current parameter in the span
            VectorX<T> param(mfa.p.size());                               // value of current parameter
            bool span_done = true;                                  // span is done until error > err_limit

            // TODO:  consider binary search of the points in the span?
            // (error likely to be higher in the center of the span?)
            for (auto j = 0; j < nspan_pts; j++)                    // parameters in the span
            {
                // debug
//                 fprintf(stderr, "ErrorSpans(): span %ld point %d\n", i, j);

                for (auto k = 0; k < mfa.p.size(); k++)
                param(k) = mfa.params(mfa.po[k] + p_ijk(k));

                // approximate the point and measure error
                size_t idx;
                mfa.ijk2idx(p_ijk, idx);
                VectorX<T> cpt(mfa.ctrl_pts.cols());       // approximated point
                decoder.VolPt(param, cpt);
                T err = fabs(mfa.NormalDistance(cpt, idx)) / mfa.range_extent;     // normalized by data range

                // span is not done
                if (err > err_limit)
                {
                    span_done = false;
                    break;
                }

                // increment param ijk
                for (auto k = 0; k < mfa.p.size(); k++)                 // dimensions in the parameter
                {
                    if (p_ijk(k) < mfa.knot_spans[i].max_param_ijk(k))
                    {
                        p_ijk(k)++;
                        break;
                    }
                    else
                        p_ijk(k) = mfa.knot_spans[i].min_param_ijk(k);
                }                                                   // dimension in parameter
            }                                                       // parameters in the span

            if (span_done)
                mfa.knot_spans[i].done = true;
        }                                                           // knot span not done
    });                                                           // knot spans

    // split spans that are not done
    auto norig_spans = mfa.knot_spans.size();
    bool new_knot_found = false;
    for (auto i = 0; i < norig_spans; i++)
    {
        if (!mfa.knot_spans[i].done && !split_spans[i])
        {
            new_knots.resize(1);
            nnew_knots = VectorXi::Zero(mfa.p.size());
            SplitSpan(i, nnew_knots, new_knots, iter, split_spans);
            if (nnew_knots.sum())
            {
                new_knot_found = true;
                mfa.InsertKnots(nnew_knots, new_knots);

                // debug
//             cerr << "inserting nnew_knots:\n" << nnew_knots << endl;
            }
        }
    }

    // debug
//     for (auto i = 0; i < mfa.knot_spans.size(); i++)                  // knot spans
//     {
//         cerr <<
//             "span_idx="          << i                           <<
//             "\nmin_knot_ijk:\n"  << mfa.knot_spans[i].min_knot_ijk  <<
//             "\nmax_knot_ijk:\n"  << mfa.knot_spans[i].max_knot_ijk  <<
//             "\nmin_knot:\n"      << mfa.knot_spans[i].min_knot      <<
//             "\nmax_knot:\n"      << mfa.knot_spans[i].max_knot      <<
//             "\nmin_param_ijk:\n" << mfa.knot_spans[i].min_param_ijk <<
//             "\nmax_param_ijk:\n" << mfa.knot_spans[i].max_param_ijk <<
//             "\nmin_param:\n"     << mfa.knot_spans[i].min_param     <<
//             "\nmax_param:\n"     << mfa.knot_spans[i].max_param     <<
//             "\n"                 << endl;
//     }

    return !nnew_knots.sum();
}

#else

// computes error in knot spans, single-threaded version
// marks the knot spans that are done (error <= max_error in the entire span)
// assumes caller allocated new_knots to number of spans and nnew_knots to domain dimensions
// (does no resizing of new_knots and nnew_knots) and zeroed nnew_knots
// returns true if all done, ie, no new knots inserted
template <typename T>
bool
mfa::
NewKnots<T>::
ErrorSpans(
        VectorXi&      nnew_knots,                          // number of new knots in each dim
        vector<T>&     new_knots,                           // new knots (1st dim changes fastest)
        T              err_limit,                           // max allowable error
        int            iter)                                // iteration number
{
    mfa::Decoder decoder(mfa);

    // initialize all knot spans to not done
    for (auto i = 0; i < mfa.knot_spans.size(); i++)
        mfa.knot_spans[i].done = false;

    // spans that have already been split in this round (to prevent splitting twice)
    vector<bool> split_spans(mfa.knot_spans.size());                // intialized to false by default

    for (auto i = 0; i < mfa.knot_spans.size(); i++)                // knot spans
    {
        size_t nspan_pts = 1;                                       // number of domain points in the span
        for (auto k = 0; k < mfa.p.size(); k++)
            nspan_pts *= (mfa.knot_spans[i].max_param_ijk(k) - mfa.knot_spans[i].min_param_ijk(k) + 1);

        VectorXi p_ijk = mfa.knot_spans[i].min_param_ijk;           // indices of current parameter in the span
        VectorX<T> param(mfa.p.size());                               // value of current parameter
        bool span_done = true;                                      // span is done until error > err_limit

        // TODO:  consider binary search of the points in the span?
        // (error likely to be higher in the center of the span?)
        for (size_t j = 0; j < nspan_pts; j++)                      // parameters in the span
        {
            for (auto k = 0; k < mfa.p.size(); k++)
                param(k) = mfa.params(mfa.po[k] + p_ijk(k));

            // approximate the point and measure error
            size_t idx;
            mfa.ijk2idx(p_ijk, idx);
            VectorX<T> cpt(mfa.ctrl_pts.cols());                      // approximated point
            decoder.VolPt(param, cpt);
            T err = fabs(mfa.NormalDistance(cpt, idx)) / mfa.dom_range;     // normalized by data range

            // span is not done
            if (err > err_limit)
            {
                span_done = false;
                break;
            }

            // increment param ijk
            for (auto k = 0; k < mfa.p.size(); k++)                 // dimensions in the parameter
            {
                if (p_ijk(k) < mfa.knot_spans[i].max_param_ijk(k))
                {
                    p_ijk(k)++;
                    break;
                }
                else
                    p_ijk(k) = mfa.knot_spans[i].min_param_ijk(k);
            }                                                       // dimension in parameter
        }                                                           // parameters in the span

        if (span_done)
            mfa.knot_spans[i].done = true;
    }                                                               // knot spans

    // debug
//     fprintf(stderr, "\nspans to be split:\n-----\n");
//     for (auto i = 0; i < mfa.knot_spans.size(); i++)                  // knot spans
//         if (!mfa.knot_spans[i].done && !split_spans[i])
//             fprintf(stderr, "i=%d min_knot=[%.3f %.3f] max_knot=[%.3f %.3f]\n", i,
//                     mfa.knot_spans[i].min_knot(0), mfa.knot_spans[i].min_knot(1),
//                     mfa.knot_spans[i].max_knot(0), mfa.knot_spans[i].max_knot(1));

    // split spans that are not done
    auto norig_spans = mfa.knot_spans.size();
    bool new_knot_found = false;
    for (auto i = 0; i < norig_spans; i++)
    {
        if (!mfa.knot_spans[i].done && !split_spans[i])
        {
            new_knots.resize(1);
            nnew_knots = VectorXi::Zero(mfa.p.size());
            SplitSpan(i, nnew_knots, new_knots, iter, split_spans);
            if (nnew_knots.sum())
            {
                new_knot_found = true;
                mfa.InsertKnots(nnew_knots, new_knots);
            }
        }
    }

    // debug
//     fprintf(stderr, "\nspans after splitting:\n-----\n");
//     for (auto i = 0; i < mfa.knot_spans.size(); i++)                  // knot spans
//         fprintf(stderr, "i=%d min_knot=[%.3f %.3f] max_knot=[%.3f %.3f]\n", i,
//                 mfa.knot_spans[i].min_knot(0), mfa.knot_spans[i].min_knot(1),
//                 mfa.knot_spans[i].max_knot(0), mfa.knot_spans[i].max_knot(1));

    return !new_knot_found;
}

#endif

// splits a knot span into two
// also splits all other spans sharing the same knot values
template <typename T>
void
mfa::
NewKnots<T>::
SplitSpan(
        size_t         si,                   // id of span to split
        VectorXi&      nnew_knots,           // number of new knots in each dim
        vector<T>&     new_knots,            // new knots (1st dim changes fastest)
        int            iter,                 // iteration number
        vector<bool>&  split_spans)          // spans that have already been split in this iteration
{
    // debug
//     fprintf(stderr, "Calling SplitSpan on span %ld\n", si);

    // new split dimension based on alternating dimension per span
    // check if span can be split (both halves would have domain points in its range)
    // if not, check other split directions
    int sd = mfa.knot_spans[si].last_split_dim;         // alternating per knot span
    T new_knot;                                     // new knot value in the split dimension
    size_t k;                                           // dimension
    for (k = 0; k < mfa.p.size(); k++)
    {
        sd       = (sd + 1) % mfa.p.size();
        new_knot = (mfa.knot_spans[si].min_knot(sd) + mfa.knot_spans[si].max_knot(sd)) / 2;
        if (mfa.params(mfa.po[sd] + mfa.knot_spans[si].min_param_ijk(sd)) < new_knot &&
                mfa.params(mfa.po[sd] + mfa.knot_spans[si].max_param_ijk(sd)) > new_knot)
            break;
    }
    if (k == mfa.p.size())                                  // a split direction could not be found
    {
        mfa.knot_spans[si].done = true;
        split_spans[si]         = true;

        // debug
        fprintf(stderr, "--- SplitSpan(): span %ld could not be split further ---\n", si);

        return;
    }

    // find all spans with the same min_knot_ijk as the span to be split and that are not done yet
    // those will be split too (NB, in the same dimension as the original span to be split)
    bool new_split = false;                             // the new knot was used to actually split a span
    for (auto j = 0; j < split_spans.size(); j++)       // original number of spans in this round
    {
        if (split_spans[j] || mfa.knot_spans[j].min_knot_ijk(sd) != mfa.knot_spans[si].min_knot_ijk(sd))
            continue;

        // debug
//      fprintf(stderr, "splitting span %d in sd=%d by new_knot=%.3f\n", j, sd, new_knot);

        new_split = true;

        // copy span to the back
        mfa.knot_spans.push_back(mfa.knot_spans[j]);

        // modify old span
        auto pi = mfa.knot_spans[j].min_param_ijk(sd);          // one coordinate of ijk index into params
        if (mfa.params(mfa.po[sd] + pi) < new_knot)                 // at least one param (domain pt) in the span
        {
            while (mfa.params(mfa.po[sd] + pi) < new_knot)          // pi - 1 = max_param_ijk(sd) in the modified span
                pi++;
            mfa.knot_spans[j].last_split_dim    = sd;
            mfa.knot_spans[j].max_knot(sd)      = new_knot;
            mfa.knot_spans[j].max_param_ijk(sd) = pi - 1;
            mfa.knot_spans[j].max_param(sd)     = mfa.params(mfa.po[sd] + pi - 1);

            // modify new span
            mfa.knot_spans.back().last_split_dim     = -1;
            mfa.knot_spans.back().min_knot(sd)       = new_knot;
            mfa.knot_spans.back().min_param_ijk(sd)  = pi;
            mfa.knot_spans.back().min_param(sd)      = mfa.params(mfa.po[sd] + pi);
            mfa.knot_spans.back().min_knot_ijk(sd)++;

            split_spans[j] = true;
        }
    }

    if (!new_split)
        return;

    // increment min and max knot ijk for any knots after the inserted one
    for (auto j = 0; j < mfa.knot_spans.size(); j++)
    {
        if (mfa.knot_spans[j].min_knot(sd) > mfa.knot_spans[si].max_knot(sd))
            mfa.knot_spans[j].min_knot_ijk(sd)++;
        if (mfa.knot_spans[j].max_knot(sd) > mfa.knot_spans[si].max_knot(sd))
            mfa.knot_spans[j].max_knot_ijk(sd)++;
    }

    // add the new knot to nnew_knots and new_knots (only a single knot inserted at a time)
    new_knots.resize(1);
    new_knots[0]    = new_knot;
    nnew_knots      = VectorXi::Zero(mfa.p.size());
    nnew_knots(sd)  = 1;

    // debug
//     fprintf(stderr, "inserted new knot value=%.3f dim=%d\n", new_knot, sd);
}

#include    "new_knots_templates.cpp"
