// --------------------------------------------------------------
// encoder object
// ref: [P&T] Piegl & Tiller, The NURBS Book, 1995
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <mfa/encode.hpp>
#include <mfa/decode.hpp>
#include <iostream>
#include <set>

mfa::
Encoder::
Encoder(MFA& mfa_) :
    mfa(mfa_),
//     p(mfa_.p),
//     ndom_pts(mfa_.ndom_pts),
//     nctrl_pts(mfa_.nctrl_pts),
//     domain(mfa_.domain),
//     params(mfa_.params),
//     ctrl_pts(mfa_.ctrl_pts),
//     knots(mfa_.knots),
//     dom_range(mfa_.dom_range),
//     po(mfa_.po),
//     ko(mfa_.ko),
    max_num_curves(1.0e4)                           // max num. curves to check in one dimension of fast encode
{
}

// adaptive encode
void
mfa::
Encoder::
AdaptiveEncode(float err_limit)                     // maximum allowable normalized error
{
    VectorXi nnew_knots = VectorXi::Zero(mfa.p.size()); // number of new knots in each dim
    VectorXf new_knots;                             // new knots (1st dim changes fastest)

    // knot insertion in 1-d
    fprintf(stderr, "Inserting knots in 1D curves...\n\n");
    for (int iter = 0; ; iter++)
    {
        fprintf(stderr, "\nIteration %d...\n", iter);
        bool done = FastEncode(nnew_knots, new_knots, err_limit, iter);

        // no new knots to be added
        if (done)
        {
            fprintf(stderr, "\n\nKnot insertion done after %d iterations; no new knots added.\n\n", iter + 1);
            break;
        }

        // check if the new knots would make the number of control points >= number of input points in any dim
        done = false;
        for (auto k = 0; k < mfa.p.size(); k++)
            if (mfa.ndom_pts(k) <= mfa.nctrl_pts(k) + nnew_knots(k))
            {
                done = true;
                break;
            }
        if (done)
        {
            fprintf(stderr, "\n\nKnot insertion done after %d iterations; control points would outnumber input points.\n", iter + 1);
            break;
        }

        mfa.InsertKnots(nnew_knots, new_knots);
    }

    // slow encoding full-d mfa
    fprintf(stderr, "Encoding in full %ldD\n", mfa.p.size());
    Encode();
}

#if 1

// single thread version
// adds knots error spans from all curves in all directions (into a set)
// fast encode using curves instead of high volume in early rounds to determine knot insertions
// returns true if done, ie, no knots are inserted
bool
mfa::
Encoder::
FastEncode(
        VectorXi& nnew_knots,                       // number of new knots in each dim
        VectorXf& new_knots,                        // new knots (1st dim changes fastest)
        float     err_limit,                        // max allowable error
        int       iter)                             // iteration number of caller (for debugging)
{
    // check and assign main quantities
    int  ndims = mfa.ndom_pts.size();                   // number of domain dimensions
    VectorXi n = mfa.nctrl_pts - VectorXi::Ones(ndims); // number of control point spans in each domain dim
    VectorXi m = mfa.ndom_pts  - VectorXi::Ones(ndims); // number of input data point spans in each domain dim
    nnew_knots = VectorXi::Zero(mfa.p.size());
    new_knots.resize(0);

    // control points
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());

    for (size_t k = 0; k < ndims; k++)              // for all domain dimensions
    {
        // temporary control points for one curve
        MatrixXf temp_ctrl = MatrixXf::Zero(mfa.nctrl_pts(k), mfa.domain.cols());

        // error spans for one curve and for worst curve
        set<int> err_spans;

        // maximum number of domain points with error greater than err_limit and their curves
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
        MatrixXf N = MatrixXf::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)                  // the rows of N
        {
            int span = mfa.FindSpan(k, mfa.params(mfa.po[k] + i), mfa.ko[k]);
            assert(span - mfa.ko[k] <= n(k));            // sanity
            mfa.BasisFuns(k, mfa.params(mfa.po[k] + i), span, N, 1, n(k) - 1, i - 1, mfa.ko[k]);
        }

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixXf NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // R is the right hand side needed for solving NtN * P = R
        MatrixXf R(n(k) - 1, mfa.domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixXf P(n(k) - 1, mfa.domain.cols());

        size_t ncurves  = mfa.domain.rows() / mfa.ndom_pts(k);      // number of curves in this dimension
        int nsame_steps = 0;                                // number of steps with same number of erroneous points
        int n_step_sizes = 0;                               // number of step sizes so far

//         for (size_t s = 1; s >= 1; s /= 2)        // debug only, one step size of s=1
        for (size_t s = ncurves / 2; s >= 1 && ncurves / s < max_num_curves; s /= 2)        // for all step sizes over curves
        {
            // debug
            fprintf(stderr, "k=%ld s=%ld\n", k, s);

            bool new_max_nerr = false;                      // this step size changed the max_nerr

            for (size_t j = 0; j < ncurves; j++)            // for all the curves in this dimension
            {
                // each time the step changes, shift start of s-th curves by one (by subtracting
                // n_step-sizes below)
                if (j >= n_step_sizes && (j - n_step_sizes) % s == 0)   // this is one of the s-th curves; compute it
                {
                    // compute R from input domain points
                    RHS(k, N, R, mfa.ko[k], mfa.po[k], mfa.co[k][j]);

                    // solve for P for one curve of control points
                    P = NtN.ldlt().solve(R);

                    // append points from P to control points
                    // TODO: any way to avoid this?
                    CopyCtrl(P, n, k, mfa.co[k][j], temp_ctrl);

                    // compute the error on the curve (number of input points with error > err_limit)
                    size_t nerr = ErrorCurve(k, mfa.co[k][j], temp_ctrl, err_spans, err_limit);

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
        }                                               // step sizes over curves

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // add new knots in the middle of spans with errors
        nnew_knots(k) = err_spans.size();
        auto old_size = new_knots.size();
        new_knots.conservativeResize(old_size + err_spans.size());    // existing values are preserved
        size_t i = 0;                                   // index into new_knots
        for (set<int>::iterator it = err_spans.begin(); it != err_spans.end(); ++it)
        {
            // debug
            assert(*it < mfa.nctrl_pts[k]);                          // not trying to go beyond the last span

            new_knots(old_size + i) = (mfa.knots(mfa.ko[k] + *it) + mfa.knots(mfa.ko[k] + *it + 1)) / 2.0;
            i++;
        }

        // print progress
        fprintf(stderr, "\rdimension %ld of %d encoded\n", k + 1, ndims);
    }                                                      // domain dimensions

    // debug
//     cerr << "\nnnew_knots:\n" << nnew_knots << endl;
//     cerr << "new_knots:\n"  << new_knots  << endl;

    return(nnew_knots.sum() ? 0 : 1);
}

#else

// NB: the following TBB version is slower (~1.5X) than the serial version
// probably the offset calculations that must be repeated each time to make the loops reentrant are
// too expensive and not worth it
// Don't use this version for now
// TODO: figure out how to compute the offsets in closed form
//
// does not include accuracy improvements later added to single threaded version (multiple
// indentical results before exiting step loop, shifted curve start in step loop, etc.)

// TBB version
// fast encode using curves instead of high volume in early rounds to determine knot insertions
// returns true if done, ie, no knots are inserted
bool
mfa::
Encoder::
FastEncode(
        VectorXi& nnew_knots,                       // number of new knots in each dim
        VectorXf& new_knots,                        // new knots (1st dim changes fastest)
        float     err_limit)                        // max allowable error
{
    // check and assign main quantities
    int  ndims = mfa.ndom_pts.size();                   // number of domain dimensions
    VectorXi n = mfa.nctrl_pts - VectorXi::Ones(ndims); // number of control point spans in each domain dim
    VectorXi m = mfa.ndom_pts  - VectorXi::Ones(ndims); // number of input data point spans in each domain dim
    nnew_knots = VectorXi::Zero(mfa.p.size());
    new_knots.resize(0);

    // control points
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());

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
        MatrixXf N = MatrixXf::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)                  // the rows of N
        {
            int span = mfa.FindSpan(k, mfa.params(mfa.po[k] + i), mfa.ko[k]);
            assert(span - mfa.ko[k] <= n(k));            // sanity
            mfa.BasisFuns(k, mfa.params(po[k] + i), span, N, 1, n(k) - 1, i - 1, mfa.ko[k]);
        }

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixXf NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // debug
        //         cerr << "k " << k << " NtN:\n" << NtN << endl;

        size_t ncurves         = mfa.domain.rows() / mfa.ndom_pts(k);   // number of curves in this dimension
        size_t worst_curve_idx = 0;                             // index of worst curve in this dimension

//         for (size_t s = 1; s >= 1; s /= 2)        // debug only, one step size of s=1
        for (size_t s = ncurves / 2; s >= 1 && ncurves / s < max_num_curves; s /= 2)        // for all step sizes over curves up to max allowed
        {
            // debug
            fprintf(stderr, "k=%ld s=%ld\n", k, s);

            size_t ncurves_s = static_cast<size_t>(ceil(static_cast<float>(ncurves) / s));
            vector<size_t> nerrs(ncurves_s);                    // number of erroneous points (error > err_limit) in the curve

            parallel_for (size_t (0), ncurves_s, [&] (size_t j) // for all the curves in this dimension (given the curve step)
            {
                // compute co, the curve domain point starting offset
                size_t co         = 0;                          // starting ofst for domain curve pts in cur. dim
                size_t coo        = 0;                          // co at start of contiguous sequence
                for (auto n = 0; n < j * s; n++)
                {
                    // adjust offsets for the next curve
                    if ((n + 1) % mfa.ds[k])
                        co++;
                    else
                    {
                        co = coo + mfa.ds[k] * mfa.ndom_pts(k);
                        coo = co;
                    }
                }

                // R is the right hand side needed for solving NtN * P = R
                MatrixXf R(n(k) - 1, mfa.domain.cols());

                // P are the unknown interior control points and the solution to NtN * P = R
                // NtN is positive definite -> do not need pivoting
                // TODO: use a common representation for P and ctrl_pts to avoid copying
                MatrixXf P(n(k) - 1, mfa.domain.cols());

                // compute R from input domain points
                RHS(k, N, R, mfa.ko[k], mfa.po[k], co);

                // solve for P for one curve of control points
                P = NtN.ldlt().solve(R);

                // append points from P to control points
                // TODO: any way to avoid this?
                MatrixXf temp_ctrl = MatrixXf::Zero(mfa.nctrl_pts(k), mfa.domain.cols());   // temporary control points for one curve
                CopyCtrl(P, n, k, co, temp_ctrl);

                // compute the error on the curve (number of input points with error > err_limit)
                nerrs[j] = ErrorCurve(k, co, temp_ctrl, err_limit);

            });                                               // parallel for over curves in this dimension

            vector<size_t>::iterator worst_curve = max_element(nerrs.begin(), nerrs.end());
            if (*worst_curve > max_nerr)
            {
                max_nerr        = *worst_curve;
                worst_curve_idx = (worst_curve - nerrs.begin()) * s;

                // debug
//                 fprintf(stderr, "k=%ld s=%ld worst_curve_idx=%ld nerr=%ld max_nerr=%ld\n",
//                 k, s, worst_curve_idx, nerrs[worst_curve_idx / s], max_nerr);
            }
            else                                            // stop refining step if no change
                if (max_nerr)
                    break;
        }                                               // step sizes over curves

        // recompute the worst curve
        vector<int> err_spans;                          // error spans for one curve

        // --- TODO: factor this out into a function ---

        // compute co, the curve domain point starting offset
        size_t co         = 0;                          // starting ofst for domain curve pts in cur. dim
        size_t coo        = 0;                          // co at start of contiguous sequence
        for (auto n = 0; n < worst_curve_idx; n++)
        {
            // adjust offsets for the next curve
            if ((n + 1) % mfa.ds[k])
                co++;
            else
            {
                co = coo + mfa.ds[k] * mfa.ndom_pts(k);
                coo = co;
            }
        }

        // debug
//         fprintf(stderr, "k=%ld worst_curve_idx=%ld co=%ld\n", k, worst_curve_idx, co);

        // R is the right hand side needed for solving NtN * P = R
        MatrixXf R(n(k) - 1, mfa.domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixXf P(n(k) - 1, mfa.domain.cols());

        // compute R from input domain points
        RHS(k, N, R, mfa.ko[k], mfa.po[k], co);

        // solve for P for one curve of control points
        P = NtN.ldlt().solve(R);

        // append points from P to control points
        // TODO: any way to avoid this?
        MatrixXf temp_ctrl = MatrixXf::Zero(mfa.nctrl_pts(k), mfa.domain.cols());   // temporary control points for one curve
        CopyCtrl(P, n, k, co, temp_ctrl);

        // --- TODO: end of factored out function ---

        // compute the error spans on the worst curve in this dimension and add new knots in their middles
        ErrorCurve(k, co, temp_ctrl, err_spans, err_limit);
        nnew_knots(k) = err_spans.size();
        auto old_size = new_knots.size();
        new_knots.conservativeResize(old_size + err_spans.size());    // existing values are preserved
        for (auto i = 0; i < err_spans.size(); i++)
            new_knots(old_size + i) = (knots(mfa.ko[k] + err_spans[i]) + knots(mfa.ko[k] + err_spans[i] + 1)) / 2.0;

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // print progress
        fprintf(stderr, "\rdimension %ld of %d encoded", k + 1, ndims);
    }                                                      // domain dimensions

    // debug
//     cerr << "\nnnew_knots:\n" << nnew_knots << endl;
//     cerr << "new_knots:\n"  << new_knots  << endl;

    return(nnew_knots.sum() ? 0 : 1);
}

#endif

// approximate a NURBS hypervolume of arbitrary dimension for a given input data set
// weights are all 1 for now
// n-d version of algorithm 9.7, Piegl & Tiller (P&T) p. 422
//
// the outputs, ctrl_pts and knots, are resized by this function;  caller need not resize them
//
// There are two types of dimensionality:
// 1. The dimensionality of the NURBS tensor product (p.size())
// (1D = NURBS curve, 2D = surface, 3D = volumem 4D = hypervolume, etc.)
// 2. The dimensionality of individual domain and control points (domain.cols())
// p.size() should be < domain.cols()
void
mfa::
Encoder::
Encode()
{
    // TODO: some of these quantities mirror this in the mfa

    // check and assign main quantities
    VectorXi n;                             // number of control point spans in each domain dim
    VectorXi m;                             // number of input data point spans in each domain dim
    int      ndims = mfa.ndom_pts.size();       // number of domain dimensions
    size_t   cs    = 1;                     // stride for domain points in curve in cur. dim

    Quants(n, m);

    // control points
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());

    // 2 buffers of temporary control points
    // double buffer needed to write output curves of current dim without changing its input pts
    // temporary control points need to begin with size as many as the input domain points
    // except for the first dimension, which can be the correct number of control points
    // because the input domain points are converted to control points one dimension at a time
    // TODO: need to find a more space-efficient way
    size_t tot_ntemp_ctrl = 1;
    for (size_t k = 0; k < ndims; k++)
        tot_ntemp_ctrl *= (k == 0 ? mfa.nctrl_pts(k) : mfa.ndom_pts(k));
    MatrixXf temp_ctrl0 = MatrixXf::Zero(tot_ntemp_ctrl, mfa.domain.cols());
    MatrixXf temp_ctrl1 = MatrixXf::Zero(tot_ntemp_ctrl, mfa.domain.cols());

    VectorXi ntemp_ctrl = mfa.ndom_pts;         // current num of temp control pts in each dim

    float  max_err_val;                     // maximum solution error in final dim of all curves

    for (size_t k = 0; k < ndims; k++)      // for all domain dimensions
    {
        fprintf(stderr, "k=%ld\n", k);

        // TODO:
        // Investigate whether in later dimensions, when input data points are replaced by
        // control points, need new knots and params computed.
        // In the next dimension, the coordinates of the dimension didn't change,
        // but the chord length did, as control points moved away from the data points in
        // the prior dim. Need to determine how much difference it would make to recompute
        // params and knots for the new input points

        // compute the matrix N, eq. 9.66 in P&T
        // N is a matrix of (m - 1) x (n - 1) scalars that are the basis function coefficients
        //  _                                _
        // |  N_1(u[1])   ... N_{n-1}(u[1])   |
        // |     ...      ...      ...        |
        // |  N_1(u[m-1]) ... N_{n-1}(u[m-1]) |
        //  -                                -
        // TODO: N is going to be very sparse when it is large: switch to sparse representation
        // N has semibandwidth < p  nonzero entries across diagonal
        MatrixXf N = MatrixXf::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)            // the rows of N
        {
            int span = mfa.FindSpan(k, mfa.params(mfa.po[k] + i), mfa.ko[k]);
            assert(span - mfa.ko[k] <= n(k));            // sanity
            mfa.BasisFuns(k, mfa.params(mfa.po[k] + i), span, N, 1, n(k) - 1, i - 1, mfa.ko[k]);
        }

        // debug
//         cerr << "k " << k << " N:\n" << N << endl;

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixXf NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // debug
//         cerr << "k " << k << " NtN:\n" << NtN << endl;

        // R is the residual matrix needed for solving NtN * P = R
        MatrixXf R(n(k) - 1, mfa.domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixXf P(n(k) - 1, mfa.domain.cols());

        // number of curves in this dimension
        size_t ncurves;
        ncurves = 1;
        for (int i = 0; i < ndims; i++)
        {
            if (i < k)
                ncurves *= mfa.nctrl_pts(i);
            else if (i > k)
                ncurves *= mfa.ndom_pts(i);
            // NB: current dimension contributes no curves, hence no i == k case
        }
        // debug
        // cerr << "k: " << k << " ncurves: " << ncurves << endl;
        // cerr << "ndom_pts:\n" << mfa.ndom_pts << endl;
        // cerr << "ntemp_ctrl:\n" << ntemp_ctrl << endl;
        // if (k > 0 && k % 2 == 1) // input to odd dims is temp_ctrl0
        //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
        // if (k > 0 && k % 2 == 0) // input to even dims is temp_ctrl1
        //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;

        size_t co = 0, to = 0;                    // starting ofst for curve & ctrl pts in cur. dim
        size_t coo = 0, too = 0;                  // co and to at start of contiguous sequence
        for (size_t j = 0; j < ncurves; j++)      // for all the curves in this dimension
        {
            // print progress
            if (j > 0 && j > 100 && j % (ncurves / 100) == 0)
                fprintf(stderr, "\r dimension %ld: %.0f %% encoded (%ld out of %ld curves)",
                        k, (float)j / (float)ncurves * 100, j, ncurves);

            // compute the one curve of control points
            CtrlCurve(N, NtN, R, P, n, k, co, cs, to, temp_ctrl0, temp_ctrl1);

            // adjust offsets for the next curve
            if ((j + 1) % cs)
                co++;
            else
            {
                co = coo + cs * ntemp_ctrl(k);
                coo = co;
            }
            if ((j + 1) % cs)
                to++;
            else
            {
                to = too + cs * mfa.nctrl_pts(k);
                too = to;
            }
        }                                                  // curves in this dimension

        // adjust offsets and strides for next dimension
        ntemp_ctrl(k) = mfa.nctrl_pts(k);
        cs *= ntemp_ctrl(k);

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // print progress
        fprintf(stderr, "\33[2K\rdimension %ld of %d encoded\n", k + 1, ndims);

    }                                                      // domain dimensions

    fprintf(stderr,"\n");

    // debug
//     cerr << "ctrl_pts:\n" << mfa.ctrl_pts << endl;
}

// NB: the following TBB version is slower (~8X) than the serial version
// probably the offset calculations that must be repeated each time to make the loops reentrant are
// too expensive and not worth it
// Don't use this version for now
// TODO: figure out how to compute the offsets in closed form
//

#if 0                                       // TBB version

// approximate a NURBS hypervolume of arbitrary dimension for a given input data set
// weights are all 1 for now
// n-d version of algorithm 9.7, Piegl & Tiller (P&T) p. 422
//
// the outputs, ctrl_pts and knots, are resized by this function;  caller need not resize them
//
// There are two types of dimensionality:
// 1. The dimensionality of the NURBS tensor product (p.size())
// (1D = NURBS curve, 2D = surface, 3D = volumem 4D = hypervolume, etc.)
// 2. The dimensionality of individual domain and control points (domain.cols())
// p.size() should be < domain.cols()
void
mfa::
Encoder::
Encode()
{
    // TODO: some of these quantities mirror this in the mfa

    // check and assign main quantities
    VectorXi n;                             // number of control point spans in each domain dim
    VectorXi m;                             // number of input data point spans in each domain dim
    int      ndims = mfa.ndom_pts.size();       // number of domain dimensions
    size_t   cs    = 1;                     // stride for input points in curve in cur. dim
                                            // domain points in first dim and ctrl pts from previous
                                            // dim for later dims.
    Quants(n, m);

    // control points
    mfa.ctrl_pts.resize(mfa.tot_nctrl, mfa.domain.cols());

    // 2 buffers of temporary control points
    // double buffer needed to write output curves of current dim without changing its input pts
    // temporary control points need to begin with size as many as the input domain points
    // except for the first dimension, which can be the correct number of control points
    // because the input domain points are converted to control points one dimension at a time
    // TODO: need to find a more space-efficient way
    size_t tot_ntemp_ctrl = 1;
    for (size_t k = 0; k < ndims; k++)
        tot_ntemp_ctrl *= (k == 0 ? mfa.nctrl_pts(k) : mfa.ndom_pts(k));
    MatrixXf temp_ctrl0 = MatrixXf::Zero(tot_ntemp_ctrl, mfa.domain.cols());
    MatrixXf temp_ctrl1 = MatrixXf::Zero(tot_ntemp_ctrl, mfa.domain.cols());

    VectorXi ntemp_ctrl = mfa.ndom_pts;         // current num of temp control pts in each dim

    float  max_err_val;                     // maximum solution error in final dim of all curves

    for (size_t k = 0; k < ndims; k++)      // for all domain dimensions
    {
        // TODO:
        // Investigate whether in later dimensions, when input data points are replaced by
        // control points, need new knots and params computed.
        // In the next dimension, the coordinates of the dimension didn't change,
        // but the chord length did, as control points moved away from the data points in
        // the prior dim. Need to determine how much difference it would make to recompute
        // params and knots for the new input points

        // compute the matrix N, eq. 9.66 in P&T
        // N is a matrix of (m - 1) x (n - 1) scalars that are the basis function coefficients
        //  _                                _
        // |  N_1(u[1])   ... N_{n-1}(u[1])   |
        // |     ...      ...      ...        |
        // |  N_1(u[m-1]) ... N_{n-1}(u[m-1]) |
        //  -                                -
        // TODO: N is going to be very sparse when it is large: switch to sparse representation
        // N has semibandwidth < p  nonzero entries across diagonal
        MatrixXf N = MatrixXf::Zero(m(k) - 1, n(k) - 1); // coefficients matrix

        for (int i = 1; i < m(k); i++)            // the rows of N
        {
            int span = mfa.FindSpan(k, mfa.params(mfa.po[k] + i), mfa.ko[k]);
            assert(span - mfa.ko[k] <= n(k));            // sanity
            mfa.BasisFuns(k, mfa.params(po[k] + i), span, N, 1, n(k) - 1, i - 1, mfa.ko[k]);
        }

        // debug
//         cerr << "k " << k << " N:\n" << N << endl;

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixXf NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // debug
//         cerr << "k " << k << " NtN:\n" << NtN << endl;

        // number of curves in this dimension
        size_t ncurves;
        ncurves = 1;
        for (int i = 0; i < ndims; i++)
        {
            if (i < k)
                ncurves *= mfa.nctrl_pts(i);
            else if (i > k)
                ncurves *= mfa.ndom_pts(i);
            // NB: current dimension contributes no curves, hence no i == k case
        }
        // debug
        // cerr << "k: " << k << " ncurves: " << ncurves << endl;
        // cerr << "ndom_pts:\n" << mfa.ndom_pts << endl;
        // cerr << "ntemp_ctrl:\n" << ntemp_ctrl << endl;
        // if (k > 0 && k % 2 == 1) // input to odd dims is temp_ctrl0
        //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
        // if (k > 0 && k % 2 == 0) // input to even dims is temp_ctrl1
        //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;

        parallel_for (size_t(0), ncurves, [&] (size_t j)      // for all the curves in this dimension
        {
            // debug
            // fprintf(stderr, "j=%ld curve\n", j);

            // compute starting offsets for curve and control points
            size_t co  = 0, to  = 0;                // starting ofst for curve & ctrl pts in cur. dim
            size_t coo = 0, too = 0;                // co and to at start of contiguous sequence
            for (auto n = 0; n < j; n++)
            {
                // adjust offsets for the next curve
                if ((n + 1) % cs)
                    co++;
                else
                {
                    co = coo + cs * mfa.ndom_pts(k);
                    coo = co;
                }
                if ((n + 1) % cs)
                    to++;
                else
                {
                    to = too + cs * mfa.nctrl_pts(k);
                    too = to;
                }
            }

            // R is the right hand side needed for solving NtN * P = R
            MatrixXf R(n(k) - 1, mfa.domain.cols());

            // P are the unknown interior control points and the solution to NtN * P = R
            // NtN is positive definite -> do not need pivoting
            // TODO: use a common representation for P and ctrl_pts to avoid copying
            MatrixXf P(n(k) - 1, mfa.domain.cols());

            // compute the one curve of control points
            CtrlCurve(N, NtN, R, P, n, k, co, cs, to, temp_ctrl0, temp_ctrl1);
        });                                                  // curves in this dimension

        // adjust offsets and strides for next dimension
        ntemp_ctrl(k) = mfa.nctrl_pts(k);
        cs *= ntemp_ctrl(k);

        NtN.resize(0, 0);                           // free NtN

        // print progress
        fprintf(stderr, "\rdimension %ld of %d encoded", k + 1, ndims);

    }                                                      // domain dimensions

    fprintf(stderr,"\n");

    // debug
//     cerr << "ctrl_pts:\n" << mfa.ctrl_pts << endl;
}

#endif

// computes right hand side vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from the
// original input domain points
void
mfa::
Encoder::
RHS(int       cur_dim,             // current dimension
    MatrixXf& N,                   // matrix of basis function coefficients
    MatrixXf& R,                   // (output) residual matrix allocated by caller
    int       ko,                  // optional index of starting knot
    int       po,                  // optional index of starting parameter
    int       co)                  // optional index of starting domain pt in current curve
{
    int n      = N.cols() + 1;               // number of control point spans
    int m      = N.rows() + 1;               // number of input data point spans

    // compute the matrix Rk for eq. 9.63 of P&T, p. 411
    MatrixXf Rk(m - 1, mfa.domain.cols());       // eigen frees MatrixX when leaving scope
    MatrixXf Nk;                             // basis coefficients for Rk[i]

    for (int k = 1; k < m; k++)
    {
        int span = mfa.FindSpan(cur_dim, mfa.params(po + k), ko);
        Nk = MatrixXf::Zero(1, n + 1);      // basis coefficients for Rk[i]
        mfa.BasisFuns(cur_dim, mfa.params(po + k), span, Nk, 0, n, 0, ko);

        Rk.row(k - 1) =
            mfa.domain.row(co + k * mfa.ds[cur_dim]) - Nk(0, 0) * mfa.domain.row(co) -
            Nk(0, n) * mfa.domain.row(co + m * mfa.ds[cur_dim]);
    }

    // compute the matrix R
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < Rk.cols(); j++)
        {
            // debug
            // fprintf(stderr, "3: i %d j %d R.rows %d Rk.rows %d\n", i, j, R.rows(), Rk.rows());
            R(i - 1, j) = (N.col(i - 1).array() * Rk.col(j).array()).sum();
        }
    }
}

// computes right hand side vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from a
// new set of input points, not the default input domain
void
mfa::
Encoder::
RHS(int       cur_dim,             // current dimension
    MatrixXf& in_pts,              // input points (not the default domain stored in the mfa)
    MatrixXf& N,                   // matrix of basis function coefficients
    MatrixXf& R,                   // (output) residual matrix allocated by caller
    int       ko,                  // optional index of starting knot
    int       po,                  // optional index of starting parameter
    int       co,                  // optional index of starting input pt in current curve
    int       cs)                  // optional stride of input pts in current curve
{
    int n      = N.cols() + 1;               // number of control point spans
    int m      = N.rows() + 1;               // number of input data point spans

    // compute the matrix Rk for eq. 9.63 of P&T, p. 411
    MatrixXf Rk(m - 1, in_pts.cols());       // eigen frees MatrixX when leaving scope
    MatrixXf Nk;                             // basis coefficients for Rk[i]

    // debug
    // cerr << "RHS in_pts:\n" << in_pts << endl;

    for (int k = 1; k < m; k++)
    {
        int span = mfa.FindSpan(cur_dim, mfa.params(po + k), ko);
        Nk = MatrixXf::Zero(1, n + 1);      // basis coefficients for Rk[i]
        mfa.BasisFuns(cur_dim, mfa.params(po + k), span, Nk, 0, n, 0, ko);

        // debug
        // cerr << "Nk:\n" << Nk << endl;

        // debug
        // cerr << "[" << in_pts.row(co + k * cs) << "] ["
        //      << in_pts.row(co) << "] ["
        //      << in_pts.row(co + m * cs) << "]" << endl;

        Rk.row(k - 1) =
            in_pts.row(co + k * cs) - Nk(0, 0) * in_pts.row(co) -
            Nk(0, n) * in_pts.row(co + m * cs);
    }

    // debug
    // cerr << "Rk:\n" << Rk << endl;

    // compute the matrix R
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < Rk.cols(); j++)
        {
            // debug
            // fprintf(stderr, "3: i %d j %d R.rows %d Rk.rows %d\n", i, j, R.rows(), Rk.rows());
            R(i - 1, j) = (N.col(i - 1).array() * Rk.col(j).array()).sum();
        }
    }
}

// Checks quantities needed for approximation
void
mfa::
Encoder::
Quants(
        VectorXi& n,                // (output) number of control point spans in each dim
        VectorXi& m)                // (output) number of input data point spans in each dim
{
    if (mfa.p.size() != mfa.ndom_pts.size())
    {
        fprintf(stderr, "Error: Encode() size of p must equal size of ndom_pts\n");
        exit(1);
    }
    for (size_t i = 0; i < mfa.p.size(); i++)
    {
        if (mfa.nctrl_pts(i) <= mfa.p(i))
        {
            fprintf(stderr, "Error: Encode() number of control points in dimension %ld"
                    "must be at least p + 1 for dimension %ld\n", i, i);
            exit(1);
        }
        if (mfa.nctrl_pts(i) > mfa.ndom_pts(i))
        {
            fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                    "exceeds number of input data points (%d) in dimension %ld. "
                    "Technically, this is not an error, but it could be a sign something is wrong and "
                    "probably not desired if you want compression. You may not be able to get the "
                    "desired error limit and compression simultaneously. Try increasing error limit?\n",
                    mfa.nctrl_pts(i), i, mfa.ndom_pts(i), i);
//             exit(1);
        }
    }

    n.resize(mfa.p.size());
    m.resize(mfa.p.size());
    for (size_t i = 0; i < mfa.p.size(); i++)
    {
        n(i)        =  mfa.nctrl_pts(i) - 1;
        m(i)        =  mfa.ndom_pts(i)  - 1;
    }
}

// append points from P to temporary control points
// init first and last control points and copy rest from solution P
// TODO: any way to avoid this copy?
// last dimension gets copied to final control points
// previous dimensions get copied to alternating double buffers
void
mfa::
Encoder::
CopyCtrl(MatrixXf& P,          // solved points for current dimension and curve
         VectorXi& n,          // number of control point spans in each dimension
         int       k,          // current dimension
         size_t    co,         // starting offset for reading domain points
         size_t    cs,         // stride for reading domain points
         size_t    to,         // starting offset for writing control points
         MatrixXf& temp_ctrl0, // first temporary control points buffer
         MatrixXf& temp_ctrl1) // second temporary control points buffer
{
    int ndims = mfa.ndom_pts.size();             // number of domain dimensions

    // if there is only one dim, copy straight to output
    if (ndims == 1)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        mfa.ctrl_pts.row(to) = mfa.domain.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            mfa.ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + mfa.ndom_pts(k) - 1);
        mfa.ctrl_pts.row(to + n(k) * cs) = mfa.domain.row(co + mfa.ndom_pts(k) - 1);
    }
    // first dim copied from domain to temp_ctrl0
    else if (k == 0)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        temp_ctrl0.row(to) = mfa.domain.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            temp_ctrl0.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + mfa.ndom_pts(k) - 1);
        temp_ctrl0.row(to + n(k) * cs) = mfa.domain.row(co + mfa.ndom_pts(k) - 1);
    }
    // even numbered dims (but not the last one) copied from temp_ctrl1 to temp_ctrl0
    else if (k % 2 == 0 && k < ndims - 1)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        temp_ctrl0.row(to) = temp_ctrl1.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            temp_ctrl0.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (mfa.ndom_pts(k) - 1) * cs);
        temp_ctrl0.row(to + n(k) * cs) = temp_ctrl1.row(co + (mfa.ndom_pts(k) - 1) * cs);
    }
    // odd numbered dims (but not the last one) copied from temp_ctrl0 to temp_ctrl1
    else if (k % 2 == 1 && k < ndims - 1)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        temp_ctrl1.row(to) = temp_ctrl0.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            temp_ctrl1.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (mfa.ndom_pts(k) - 1) * cs);
        temp_ctrl1.row(to + n(k) * cs) = temp_ctrl0.row(co + (mfa.ndom_pts(k) - 1) * cs);
    }
    // final dim if even is copied from temp_ctrl1 to ctrl_pts
    else if (k == ndims - 1 && k % 2 == 0)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        mfa.ctrl_pts.row(to) = temp_ctrl1.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            mfa.ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (mfa.ndom_pts(k) - 1) * cs);
        mfa.ctrl_pts.row(to + n(k) * cs) = temp_ctrl1.row(co + (mfa.ndom_pts(k) - 1) * cs);
    }
    // final dim if odd is copied from temp_ctrl0 to ctrl_pts
    else if (k == ndims - 1 && k % 2 == 1)
    {
        // debug
        // fprintf(stderr, "t_start[%ld] = d[%ld]\n", to, co);
        mfa.ctrl_pts.row(to) = temp_ctrl0.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            mfa.ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t_end[%ld] = d[%ld]\n", to + n(k) * cs, co + (mfa.ndom_pts(k) - 1) * cs);
        mfa.ctrl_pts.row(to + n(k) * cs) = temp_ctrl0.row(co + (mfa.ndom_pts(k) - 1) * cs);
    }
}

// append points from P to temporary control points
// init first and last control points and copy rest from solution P
// TODO: any way to avoid this copy?
// just simple copy to one temporary buffer, no alternating double buffers
// nor copy to final control points
void
mfa::
Encoder::
CopyCtrl(MatrixXf& P,          // solved points for current dimension and curve
         VectorXi& n,          // number of control point spans in each dimension
         int       k,          // current dimension
         size_t    co,         // starting offset for reading domain points
         MatrixXf& temp_ctrl)  // temporary control points buffer
{
    // copy first point straight from domain
    temp_ctrl.row(0) = mfa.domain.row(co);

    // copy intermediate points
    // clamp all dimensions other than k to the same as the domain points
    // this eliminates any wiggles in other dimensions from the computation of P (typically ~10^-5)
    for (int i = 1; i < n(k); i++)
    {
        for (auto j = 0; j < mfa.domain.cols(); j++)
        {
            if (j < mfa.p.size() && j != k)
                temp_ctrl(i, j) = mfa.domain(co, j);
            else
                temp_ctrl(i, j) = P(i - 1, j);
        }

        // DEPRECATED: unclamped copy
//         temp_ctrl.row(i) = P.row(i - 1);
    }

    // copy last point straight from domain
    temp_ctrl.row(n(k)) = mfa.domain.row(co + (mfa.ndom_pts(k) - 1) * mfa.ds[k]);
}

// solves for one curve of control points
void
mfa::
Encoder::
CtrlCurve(MatrixXf& N,          // basis functions for current dimension
          MatrixXf& NtN,        // N^t * N
          MatrixXf& R,          // residual matrix for current dimension and curve
          MatrixXf& P,          // solved points for current dimension and curve
          VectorXi& n,          // number of control point spans in each dimension
          size_t    k,          // current dimension
          size_t    co,         // starting ofst for reading domain pts
          size_t    cs,         // stride for reading domain points
          size_t    to,         // starting ofst for writing control pts
          MatrixXf& temp_ctrl0, // first temporary control points buffer
          MatrixXf& temp_ctrl1) // second temporary control points buffer
{
    // compute R
    // first dimension reads from domain
    // subsequent dims alternate reading temp_ctrl0 and temp_ctrl1
    // even dim reads temp_ctrl1, odd dim reads temp_ctrl0; opposite of writing order
    // because what was written in the previous dimension is read in the current one
    if (k == 0)
        RHS(k, N, R, mfa.ko[k], mfa.po[k], co);                 // input points = default domain
    else if (k % 2)
        RHS(k, temp_ctrl0, N, R, mfa.ko[k], mfa.po[k], co, cs); // input points = temp_ctrl0
    else
        RHS(k, temp_ctrl1, N, R, mfa.ko[k], mfa.po[k], co, cs); // input points = temp_ctrl1

    // solve for P
    P = NtN.ldlt().solve(R);

    // debug
    // cerr << "P:\n" << P << endl;
    // Eigen::FullPivLU<MatrixXf> lu_decomp(NtN);
    // cerr << "Rank of NtN = " << lu_decomp.rank() << endl;

    // append points from P to control points
    // TODO: any way to avoid this?
    CopyCtrl(P, n, k, co, cs, to, temp_ctrl0, temp_ctrl1);

    // debug
    // int ndims = mfa.ndom_pts.size();
    // cerr << "k " << k << " P:\n" << P << endl;
    // if (ndims == 1)
    //     cerr << "ctrl_pts:\n" << mfa.ctrl_pts << endl;
    // else if (k == 0)
    //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
    // else if (k % 2 == 0 && k < ndims - 1)
    //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
    // else if (k % 2 == 1 && k < ndims - 1)
    //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;
    // else if (k == ndims - 1)
    //     cerr << "ctrl_pts:\n" << mfa.ctrl_pts << endl;
}

// returns number of points in a curve that have error greater than err_limit
int
mfa::
Encoder::
ErrorCurve(
        size_t       k,                         // current dimension
        size_t       co,                        // starting ofst for reading domain pts
        MatrixXf&    ctrl_pts,                  // control points
        float        err_limit)                 // max allowable error
{
    mfa::Decoder decoder(mfa);
    VectorXf cpt(mfa.domain.cols());            // decoded curve point
    int nerr = 0;                               // number of points with error greater than err_limit
    int span = mfa.p[k];                        // current knot span of the domain point being checked

    for (auto i = 0; i < mfa.ndom_pts[k]; i++)      // all domain points in the curve
    {
        while (mfa.knots(mfa.ko[k] + span + 1) < 1.0 && mfa.knots(mfa.ko[k] + span + 1) <= mfa.params(mfa.po[k] + i))
            span++;
        // debug
//         fprintf(stderr, "param=%.3f span=[%.3f %.3f]\n", mfa.params(po[k] + i), knots(mfa.ko[k] + span), knots(mfa.ko[k] + span + 1));

        decoder.CurvePt(k, mfa.params(mfa.po[k] + i), ctrl_pts, cpt, mfa.ko[k]);
        float err = fabs(mfa.NormalDistance(cpt, co + i * mfa.ds[k])) / mfa.dom_range;     // normalized by data range
//         float err = fabs(mfa.CurveDistance(k, cpt, co + i * mfa.ds[k])) / mfa.dom_range;     // normalized by data range
        if (err > err_limit)
        {
            nerr++;

            // debug
//             VectorXf dpt = mfa.domain.row(co + i * mfa.ds[k]);
//             cerr << "\ndomain point:\n" << dpt << endl;
//             cerr << "approx point:\n" << cpt << endl;
//             fprintf(stderr, "k=%ld i=%d co=%ld err=%.3e\n\n", k, i, co, err);
        }
    }

    return nerr;
}

// returns number of points in a curve that have error greater than err_limit
// fills err_spans with the span indices of spans that have at least one point with such error
//      and that have at least one inut point in each half of the span (assuming eventually
//      the span would be split in half with a knot added in the middle, and an input point would
//      need to be in each span after splitting)
// this version takes a set instead of a vector for error_spans so that the same span can be
// added iteratively multiple times without creating duplicates
int
mfa::
Encoder::
ErrorCurve(
        size_t       k,                         // current dimension
        size_t       co,                        // starting ofst for reading domain pts
        MatrixXf&    ctrl_pts,                  // control points
        set<int>&    err_spans,                 // spans with error greater than err_limit
        float        err_limit)                 // max allowable error
{
    mfa::Decoder decoder(mfa);
    VectorXf cpt(mfa.domain.cols());            // decoded curve point
    int nerr = 0;                               // number of points with error greater than err_limit
    int span = mfa.p[k];                        // current knot span of the domain point being checked

    for (auto i = 0; i < mfa.ndom_pts[k]; i++)      // all domain points in the curve
    {
        while (mfa.knots(mfa.ko[k] + span + 1) < 1.0 && mfa.knots(mfa.ko[k] + span + 1) <= mfa.params(mfa.po[k] + i))
            span++;

        decoder.CurvePt(k, mfa.params(mfa.po[k] + i), ctrl_pts, cpt, mfa.ko[k]);

        float err = fabs(mfa.NormalDistance(cpt, co + i * mfa.ds[k])) / mfa.dom_range;     // normalized by data range
//         float err = fabs(mfa.CurveDistance(k, cpt, co + i * mfa.ds[k])) / mfa.dom_range;     // normalized by data range

        if (err > err_limit)
        {
            // don't duplicate spans
            set<int>::iterator it = err_spans.find(span);
            if (!err_spans.size() || it == err_spans.end())
            {
                // ensure there would be a domain point in both halves of the span if it were split
                bool split_left = false;
                for (auto j = i; mfa.params(mfa.po[k] + j) >= mfa.knots(mfa.ko[k] + span); j--)
                    if (mfa.params(mfa.po[k] + j) < (mfa.knots(mfa.ko[k] + span) + mfa.knots(mfa.ko[k] + span + 1)) / 2.0)
                    {
                        // debug
//                         fprintf(stderr, "split_left: param=%.3f span[%d]=[%.3f, %.3f)\n",
//                                 mfa.params(po[k] + j), span, knots(mfa.ko[k] + span), knots(mfa.ko[k] + span + 1));
                        split_left = true;
                        break;
                    }
                bool split_right = false;
                for (auto j = i; mfa.params(mfa.po[k] + j) < mfa.knots(mfa.ko[k] + span + 1); j++)
                    if (mfa.params(mfa.po[k] + j) >= (mfa.knots(mfa.ko[k] + span) + mfa.knots(mfa.ko[k] + span + 1)) / 2.0)
                    {
//                         fprintf(stderr, "split_right: param=%.3f span[%d]=[%.3f, %.3f)\n",
//                                 mfa.params(mfa.po[k] + j), span, knots(mfa.ko[k] + span), knots(mfa.ko[k] + span + 1));
                        split_right = true;
                        break;
                    }
                // mark the span and count the point if the span can (later) be split
                if (split_left && split_right)
                    err_spans.insert(it, span);
            }
            // count the point in the total even if the span is not marked for spitting
            // total used to find worst curve, defined as the curve with the most domain points in
            // error (including multiple domain points per span and points in spans that can't be
            // split further)
            nerr++;

            // debug
//             VectorXf dpt = mfa.domain.row(co + i * mfa.ds[k]);
//             cerr << "\ndomain point:\n" << dpt << endl;
//             cerr << "approx point:\n" << cpt << endl;
//             fprintf(stderr, "k=%ld i=%d co=%ld err=%.3e\n\n", k, i, co, err);
        }
    }

    return nerr;
}
