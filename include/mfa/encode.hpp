//--------------------------------------------------------------
// encoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _ENCODE_HPP
#define _ENCODE_HPP

#include    <mfa/mfa_data.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/decode.hpp>
#include    <mfa/new_knots.hpp>

#include    <Eigen/Dense>
#include    <vector>
#include    <set>
#include    <iostream>

#ifndef      MFA_NO_WEIGHTS

#include    "coin/ClpSimplex.hpp"
#include    "coin/ClpInterior.hpp"

#endif

#include    <cppoptlib/problem.h>
#include    <cppoptlib/boundedproblem.h>
#include    <cppoptlib/solver/bfgssolver.h>
#include    <cppoptlib/solver/lbfgsbsolver.h>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

template <typename T>                                   // float or double
class NewKnots;


namespace mfa
{
using namespace cppoptlib;

template <typename T>                        // float or double
    class LocalLSQ : public Problem<T>
    {
    private:
        const MFA<T>&       mfa;         // the mfa object
        MFA_Data<T>&        mfa_data;    // the mfa data object
        const MatrixX<T>&   domain;      // input points
        const MatrixX<T>&   cons;        // control point constraints Matrix
        size_t              start_idx;   // start and end of the local subdomain
        size_t              end_idx;     // in input point space (1D for now)
        int                 verbose;     // more output

    public:
        LocalLSQ(const MFA<T>&      mfa_,
                 MFA_Data<T>&       mfa_data_,
                 const MatrixX<T>&  domain_,
                 const MatrixX<T>&  cons_,
                 size_t             start_,
                 size_t             end_,
                 int                verb_): mfa(mfa_),
                                            mfa_data(mfa_data_),
                                            domain(domain_),
                                            cons(cons_),
                                            start_idx(start_),
                                            end_idx(end_),
                                            verbose(verb_)

        {}

        ~LocalLSQ()
        {}

        using typename Problem<T>::TVector;

        void setConstraints(const MatrixX<T> c) {cons = c;}
        // objective function
        T value(const TVector &x);
//        void gradient(const TVector &x, TVector &grad);
    };

    template <typename T>                               // float or double
    struct MFA;

    template <typename T>                               // float or double
    class Encoder
    {
    private:

        template <typename>
        friend class NewKnots;

        const MatrixX<T>&   domain;                         // input points
        const MFA<T>&       mfa;                            // the mfa top-level object
        MFA_Data<T>&        mfa_data;                       // the mfa data model
        int                 verbose;                        // output level
        size_t              max_num_curves;                 // max num. curves per dimension to check in curve version

    public:

        Encoder(
                const MFA<T>&       mfa_,                   // MFA top-level object
                MFA_Data<T>&        mfa_data_,              // MFA data model
                const MatrixX<T>&   domain_,                // input points
                int                 verbose_) :             // debug level
            mfa(mfa_),
            mfa_data(mfa_data_),
            verbose(verbose_),
            domain(domain_),
            max_num_curves(1.0e4)                           // max num. curves to check in one dimension of curve version
        {}

        ~Encoder() {}

        // approximate a NURBS hypervolume of arbitrary dimension for a given input data set
        // n-d version of algorithm 9.7, Piegl & Tiller (P&T) p. 422
        // output control points can be specified by caller, does not have to be those in the tmesh
        // the output ctrl_pts are resized by this function;  caller need not resize them
        void Encode(const VectorXi& nctrl_pts,              // number of control points in each dim.
                    MatrixX<T>&     ctrl_pts,               // (output) control points
                    VectorX<T>&     weights,                // (output) weights
                    bool            weighted = true)        // solve for and use weights
        {
            // check quantities
            if (mfa_data.p.size() != mfa.ndom_pts().size())
            {
                fprintf(stderr, "Error: Encode() size of p must equal size of ndom_pts\n");
                exit(1);
            }
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                if (nctrl_pts(i) <= mfa_data.p(i))
                {
                    fprintf(stderr, "Error: Encode() number of control points in dimension %ld "
                            "must be at least p + 1 for dimension %ld\n", i, i);
                    exit(1);
                }
                if (nctrl_pts(i) > mfa.ndom_pts()(i))
                {
                    fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                            "exceeds number of input data points (%d) in dimension %ld.\n", nctrl_pts(i), i, mfa.ndom_pts()(i), i);
                }
            }

            int      ndims  = mfa.ndom_pts().size();          // number of domain dimensions
            size_t   cs     = 1;                            // stride for input points in curve in cur. dim
            int      pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;// control point dimensonality

            // resize matrices in case number of control points changed
            ctrl_pts.resize(nctrl_pts.prod(), pt_dim);
            weights.resize(ctrl_pts.rows());
            for (auto k = 0; k < ndims; k++)
                mfa_data.N[k] = MatrixX<T>::Zero(mfa.ndom_pts()(k), nctrl_pts(k));  // basis functions need to be resized and initialized to 0

            // 2 buffers of temporary control points
            // double buffer needed to write output curves of current dim without changing its input pts
            // temporary control points need to begin with size as many as the input domain points
            // except for the first dimension, which can be the correct number of control points
            // because the input domain points are converted to control points one dimension at a time
            // TODO: need to find a more space-efficient way
            size_t tot_ntemp_ctrl = 1;
            for (size_t k = 0; k < ndims; k++)
                tot_ntemp_ctrl *= (k == 0 ? nctrl_pts(k) : mfa.ndom_pts()(k));
            MatrixX<T> temp_ctrl0 = MatrixX<T>::Zero(tot_ntemp_ctrl, pt_dim);
            MatrixX<T> temp_ctrl1 = MatrixX<T>::Zero(tot_ntemp_ctrl, pt_dim);

            VectorXi ntemp_ctrl = mfa.ndom_pts();     // current num of temp control pts in each dim

            for (size_t k = 0; k < ndims; k++)      // for all domain dimensions
            {
                // number of curves in this dimension
                size_t ncurves;
                ncurves = 1;
                for (int i = 0; i < ndims; i++)
                {
                    if (i < k)
                        ncurves *= nctrl_pts(i);
                    else if (i > k)
                        ncurves *= mfa.ndom_pts()(i);
                    // NB: current dimension contributes no curves, hence no i == k case
                }

                // compute local version of co
                vector<size_t> co(ncurves);                     // starting curve points in current dim.
                vector<size_t> to(ncurves);                     // starting control points in current dim.
                co[0]      = 0;
                to[0]      = 0;
                size_t coo = 0;                                 // co at start of contiguous sequence
                size_t too = 0;                                 // to at start of contiguous sequence

                // TODO: allocate P and R once for all curves in an EncodeInfo struct similar to DecodeInfo
                for (auto j = 1; j < ncurves; j++)
                {
                    if (j % cs)
                    {
                        co[j] = co[j - 1] + 1;
                        to[j] = to[j - 1] + 1;
                    }
                    else
                    {
                        co[j] = coo + cs * ntemp_ctrl(k);
                        coo   = co[j];
                        to[j] = too + cs * nctrl_pts(k);
                        too   = to[j];
                    }
                }

                // TODO:
                // Investigate whether in later dimensions, when input data points are replaced by
                // control points, need new knots and params computed.
                // In the next dimension, the coordinates of the dimension didn't change,
                // but the chord length did, as control points moved away from the data points in
                // the prior dim. Need to determine how much difference it would make to recompute
                // params and knots for the new input points
                // (moot when using domain decomposition)

                // N is a matrix of (m + 1) x (n + 1) scalars that are the basis function coefficients
                //  _                          _
                // |  N_0(u[0])   ... N_n(u[0]) |
                // |     ...      ...      ...  |
                // |  N_0(u[m])   ... N_n(u[m]) |
                //  -                          -
                // TODO: N is going to be very sparse when it is large: switch to sparse representation
                // N has semibandwidth < p  nonzero entries across diagonal

                for (int i = 0; i < mfa_data.N[k].rows(); i++)
                {
                    int span = mfa_data.FindSpan(k, mfa.params()[k][i], nctrl_pts(k));

#ifndef TMESH       // original version for one tensor product

                    mfa_data.OrigBasisFuns(k, mfa.params()[k][i], span, mfa_data.N[k], i);

#else               // tmesh version

                    mfa_data.BasisFuns(k, mfa.params()[k][i], span, mfa_data.N[k], i);

#endif
                }

                // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
                // NtN has semibandwidth < p + 1 nonzero entries across diagonal
                MatrixX<T> NtN  = mfa_data.N[k].transpose() * mfa_data.N[k];

                // debug
//                 cerr << "N[k]:\n" << mfa_data.N[k] << endl;
//                 cerr << "NtN:\n" << NtN << endl;

#ifdef MFA_TBB  // TBB version

                parallel_for (size_t(0), ncurves, [&] (size_t j)      // for all the curves in this dimension
                        {
                        // debug
                        // fprintf(stderr, "j=%ld curve\n", j);

                        // R is the right hand side needed for solving NtN * P = R
                        MatrixX<T> R(mfa_data.N[k].cols(), pt_dim);

                        // P are the unknown control points and the solution to NtN * P = R
                        // NtN is positive definite -> do not need pivoting
                        // TODO: use a common representation for P and ctrl_pts to avoid copying
                        MatrixX<T> P(mfa_data.N[k].cols(), pt_dim);

                        // compute the one curve of control points
                        CtrlCurve(mfa_data.N[k], NtN, R, P, k, co[j], cs, to[j], temp_ctrl0, temp_ctrl1, -1, ctrl_pts, weights, weighted);
                        });                                                  // curves in this dimension

#endif              // end TBB version

#ifdef MFA_SERIAL   // serial version

                // R is the right hand side needed for solving NtN * P = R
                MatrixX<T> R(mfa_data.N[k].cols(), pt_dim);

                // P are the unknown control points and the solution to NtN * P = R
                // NtN is positive definite -> do not need pivoting
                // TODO: use a common representation for P and ctrl_pts to avoid copying
                MatrixX<T> P(mfa_data.N[k].cols(), pt_dim);

                // encode curves in this dimension
                for (size_t j = 0; j < ncurves; j++)
                {
                    // print progress
                    if (verbose)
                    {
                        if (j > 0 && j > 100 && j % (ncurves / 100) == 0)
                            fprintf(stderr, "\r dimension %ld: %.0f %% encoded (%ld out of %ld curves)",
                                    k, (T)j / (T)ncurves * 100, j, ncurves);
                    }

                    // compute the one curve of control points
                    CtrlCurve(mfa_data.N[k], NtN, R, P, k, co[j], cs, to[j], temp_ctrl0, temp_ctrl1, j, ctrl_pts, weights, weighted);
                }

#endif          // end serial version

                // adjust offsets and strides for next dimension
                ntemp_ctrl(k) = nctrl_pts(k);
                cs *= ntemp_ctrl(k);

                // print progress
                if (verbose)
                    fprintf(stderr, "\ndimension %ld of %d encoded\n", k + 1, ndims);
            }                                                      // domain dimensions

            // debug
//             cerr << "Encode() ctrl_pts:\n" << ctrl_pts << endl;
        }

        // original adaptive encoding for first tensor product only
        void OrigAdaptiveEncode(
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights
                bool                local,                  // solve locally (with constraints) each round
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)         // optional maximum number of rounds
        {
            vector<vector<T>> new_knots;                               // new knots in each dim.

            // TODO: use weights for knot insertion
            // for now, weights are only used for final full encode

            // debug
//             mfa_data.tmesh.print();

            // debug: local not being used in OrigAdaptiveEncode (no tmesh, no local solve)
            if (local)
                fprintf(stderr, "*** Not using local solve in OrigAdaptiveEncode ***\n");

            // loop until no change in knots
            for (int iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                    break;

                if (verbose)
                    fprintf(stderr, "Iteration %d...\n", iter);

                // low-d w/ splitting spans in the middle
                bool done = NewKnots_curve(new_knots, err_limit, extents, iter);

                // no new knots to be added
                if (done)
                {
                    if (verbose)
                        fprintf(stderr, "\nKnot insertion done after %d iterations; no new knots added.\n\n", iter + 1);
                    break;
                }

                // check if the new knots would make the number of control points >= number of input points in any dim
                done = false;
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    // hard-coded for first tensor
                    if (mfa.ndom_pts()(k) <= mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + new_knots[k].size())
                    {
                        done = true;
                        break;
                    }
                if (done)
                {
                    if (verbose)
                        fprintf(stderr, "\nKnot insertion done after %d iterations; control points would outnumber input points.\n", iter + 1);
                    break;
                }

                // debug
//                 mfa_data.tmesh.print();
            }

            // final full encoding needed after last knot insertion above
            if (verbose)
                fprintf(stderr, "Encoding in full %ldD\n", mfa_data.p.size());
            TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];        // fixed encode assumes the tmesh has only one tensor product
            Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);
        }

        // adaptive encoding for tmesh with optional local solve
        void AdaptiveEncode(
                T                   err_limit,                  // maximum allowable normalized error
                bool                weighted,                   // solve for and use weights
                bool                local,                      // solve locally (with constraints) each round
                const VectorX<T>&   extents,                    // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)             // optional maximum number of rounds
        {
            // debug
            if (local)
                fprintf(stderr, "*** Using local solve in AdaptiveEncode ***\n");

            // temporary control points and weights for global encode
            // TODO: replace for local encode
            VectorXi nctrl_pts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
                nctrl_pts(k) = mfa_data.tmesh.all_knots[k].size() - mfa_data.p(k) - 1;
            MatrixX<T> ctrl_pts(nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
            VectorX<T> weights(ctrl_pts.rows());

            // Initial global encode and scattering of control points to tensors
            // TODO: replace for local encode
            Encode(nctrl_pts, ctrl_pts, weights);
            mfa_data.tmesh.scatter_ctrl_pts(nctrl_pts, ctrl_pts, weights);

            // debug: print tmesh
            fprintf(stderr, "\n----- initial T-mesh -----\n\n");
            mfa_data.tmesh.print();
            fprintf(stderr, "--------------------------\n\n");

            // loop until no change in knots or number of control points >= input points
            for (int iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)       // optional cap on number of rounds
                    break;

                if (verbose)
                    fprintf(stderr, "Iteration %d...\n", iter);

                // using NewKnots_full high-d span splitting with tmesh (for now)
                int retval = NewKnots_full(err_limit, extents, iter, local, nctrl_pts, ctrl_pts, weights);

                // debug: print tmesh
                fprintf(stderr, "\n----- T-mesh after LocalNewKnots_full -----\n\n");
                mfa_data.tmesh.print();
                fprintf(stderr, "--------------------------\n\n");

                // resize temporary control points and weights and global encode and scattering of control points to tensors
                // TODO: replace for local encode
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    nctrl_pts(k) = mfa_data.tmesh.all_knots[k].size() - mfa_data.p(k) - 1;
                ctrl_pts.resize(nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
                weights.resize(ctrl_pts.rows());

                Encode(nctrl_pts, ctrl_pts, weights);
                mfa_data.tmesh.scatter_ctrl_pts(nctrl_pts, ctrl_pts, weights);

                // debug: print tmesh
                fprintf(stderr, "\n----- T-mesh at the end of iteration %d-----\n\n", iter);
                mfa_data.tmesh.print();
                fprintf(stderr, "--------------------------\n\n");

                // no new knots to be added
                if (retval == 0)
                {
                    if (verbose)
                        fprintf(stderr, "\nKnot insertion done after %d iterations; no new knots added.\n\n", iter + 1);
                    break;
                }

                // new knots would make the number of control points >= number of input points in any dim
                if (retval == -1)
                {
                    if (verbose)
                        fprintf(stderr, "\nKnot insertion done after %d iterations; control points would outnumber input points.\n", iter + 1);
                    break;
                }
            }
        }

    private:

#ifndef      MFA_NO_WEIGHTS

        bool Weights(
                int                 k,              // current dimension
                const MatrixX<T>&   Q,              // input points
                const MatrixX<T>&   N,              // basis functions
                const MatrixX<T>&   NtN,            // N^T * N
                int                 curve_id,       // debugging
                VectorX<T>&         weights)        // (output) weights
        {
            bool success;

            // Nt, NtNi
            // TODO: offer option of saving time or space by comuting Nt and NtN each time it is needed?
            MatrixX<T> Nt   = N.transpose();
            MatrixX<T> NtNi = NtN.partialPivLu().inverse();

            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                     // dimensionality of input and control points (domain and range)

            MatrixX<T> NtQ2  = MatrixX<T>::Zero(Nt.rows(),   Nt.cols());    // N^T * Q^2
            MatrixX<T> NtQ   = MatrixX<T>::Zero(Nt.rows(),   Nt.cols());    // N^T * Q
            MatrixX<T> NtQ2N = MatrixX<T>::Zero(NtN.rows(),  NtN.cols());   // N^T * Q^2 * N
            MatrixX<T> NtQN  = MatrixX<T>::Zero(NtN.rows(),  NtN.cols());   // N^T * Q   * N

            // temporary matrices NtQ and NtQ2
            for (auto i = 0; i < Nt.cols(); i++)
            {
                NtQ.col(i)  = Nt.col(i) * Q(i, pt_dim - 1);
                NtQ2.col(i) = Nt.col(i) * Q(i, pt_dim - 1) * Q(i, pt_dim - 1);
            }

            // final matrices NtQN and NtQ2N
            NtQN  = NtQ  * N;
            NtQ2N = NtQ2 * N;

            // compute the matrix M according to eq.3 and eq. 4 of M&K95
            MatrixX<T> M = NtQ2N - NtQN * NtNi * NtQN;

            // debug: output the matrix M
            //     cerr << M << endl;
            //     Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
            //     ofstream M_out;
            //     M_out.open("M.txt");
            //     M_out << M.format(OctaveFmt) << endl;
            //     M_out.close();

            // compute the eigenvalues and eigenvectors of M (eq. 9 of M&K95)
            // NB: M is self-adjoint (ie, real and symmetric)
            // Eigen's SelfAdjointEigenSolver is faster than its normal EigenSolver in this case
            Eigen::SelfAdjointEigenSolver<MatrixX<T>> eigensolver(M);
            if (eigensolver.info() != Eigen::Success)
            {
                fprintf(stderr, "Error: Encoder::Weights(), computing eigenvalues of M failed, perhaps M is not self-adjoint?\n");
                return false;
            }

            const MatrixX<T>& EV    = eigensolver.eigenvectors();          // typing shortcut
            const VectorX<T>& evals = eigensolver.eigenvalues();           // typing shortcut

            // eigenvalues should be positive and distinct
            for (auto i = 0; i < evals.size() - 1; i++)
                if (evals(i) == 0.0 || evals(i) == evals(i + 1))
                {
                    fprintf(stderr, "Warning: Weights(): eigenvalues should be positive and distinct.\n");
                    fprintf(stderr, "Aborting weights calculation\n");
                    return false;
                }

            // if smallest eigenvector is all positive or all negative, those are the weights
            if ( (EV.col(0).array() > 0.0).all() )
            {
                weights = EV.col(0);
                weights *= (1.0 / weights.maxCoeff());  // scale to max weight = 1
                success = true;
            }
            else if ( (EV.col(0).array() < 0.0).all() )
            {
                weights = -EV.col(0);
                weights *= (1.0 / weights.maxCoeff());  // scale to max weight = 1
                success = true;
            }

            // if smallest eigenvector is mixed sign, then expand eigen space
            else
            {
                //         fprintf(stderr, "\nExpanding eigenspace using linear solver\n");
                success      = false;
                T min_weight = 1.0;
                T max_weight = 1.0e4;
                // minimum eigenvector element, if less, clamp to 0.0 (between 1e-6 and 1e-12 seems to help)
                T min_ev_val = 1.0e-12;

                auto nweights = weights.size();

                // column indices
                vector<int> col_idx(2 * nweights);
                for (auto j = 0; j < 2 * nweights; j++)
                    col_idx[j] = j;

                ClpSimplex model;                           // simplex method, ~10X faster than interior point
                //         ClpInterior model;                          // interior point method
                model.resize(2 * nweights, 0);              // all the rows and no columns (columns are added incrementally)

                // row upper bounds
                for (auto i = 0; i < nweights; i++)
                {
                    model.setRowUpper(i, max_weight);
                    model.setRowUpper(nweights + i, -min_weight);
                }

                // first column of the matrix
                vector<double> newcol(nweights * 2);
                for (auto j = 0; j < nweights; j++)
                {
                    newcol[j]            = fabs(EV(j, 0)) < min_ev_val ? 0.0 : EV(j, 0);
                    newcol[nweights + j] = -newcol[j];
                }
                model.addColumn(2 * nweights, &col_idx[0], &newcol[0], -COIN_DBL_MAX, COIN_DBL_MAX);

                for (auto i = 2; i <= EV.cols(); i++)        // expand from 2 eigenvectors to all, one at a time
                {
                    // add another column
                    for (auto j = 0; j < nweights; j++)
                    {
                        newcol[j]            = fabs(EV(j, i - 1)) < min_ev_val ? 0.0 : EV(j, i - 1);
                        newcol[nweights + j] = -newcol[j];
                    }
                    model.addColumn(2 * nweights, &col_idx[0], &newcol[0], -COIN_DBL_MAX, COIN_DBL_MAX);

                    // solve
                    model.setLogLevel(0);
                    model.initialSolve();               // simplex method, ~10X faster than interior point
                    //             model.primalDual();              // interior point method

                    if (!model.isProvenPrimalInfeasible() && !model.isIterationLimitReached())
                    {
                        // copy out the solution
                        VectorX<T>    solved_weights = VectorX<T>::Zero(nweights);
                        int           ncols          = model.getNumCols();
                        const double* colSol         = model.getColSolution();
                        for (auto k = 0; k < ncols; k++)
                            solved_weights += colSol[k] * EV.col(k);

                        // check if the solution was found successfully
                        if ( (solved_weights.array() > 0.0).all() )
                        {
                            weights = solved_weights;
                            weights *= (1.0 / weights.maxCoeff());  // scale to max weight = 1
                            success = true;
                            if (verbose)
                                fprintf(stderr, "curve %d: successful linear solve from %d eigenvectors\n", curve_id, i);
                        }
                    }

                    if (success)
                        break;
                }                                               // increasing number of eigenvectors

                if (!success)
                {
                    weights = VectorX<T>::Ones(nweights);
                    if (verbose)
                        fprintf(stderr, "curve %d: linear solver could not find positive weights; setting to 1\n", curve_id);
                }
            }                                                   // else need to expand eigenspace

            return success;
        }

#endif

        // computes right hand side vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from the
        // original input domain points
        // includes multiplication by weights
        // R is column vector of n + 1 elements, each element multiple coordinates of the input points
        void RHS(
                int                 cur_dim,  // current dimension
                const MatrixX<T>&   N,        // matrix of basis function coefficients
                MatrixX<T>&         R,        // (output) residual matrix allocated by caller
                const VectorX<T>&   weights,  // precomputed weights for n + 1 control points on this curve
                int                 co)       // index of starting domain pt in current curve
        {
            int last   = R.cols() - 1;                                  // column of range value TODO: weighing only the last column does not make much sense in the split model
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);     // one row for each input point
            VectorX<T> denom(N.rows());                                 // rational denomoninator for param of each input point

            for (int k = 0; k < N.rows(); k++)                          // for all input points
            {
                denom(k) = (N.row(k).cwiseProduct(weights.transpose())).sum();
#ifdef UNCLAMPED_KNOTS
                if (denom(k) == 0.0)
                    denom(k) = 1.0;
#endif
                Rk.row(k) = domain.block(co + k * mfa.ds()[cur_dim], mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
            }

#ifdef WEIGH_ALL_DIMS                               // weigh all dimensions
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                 // ith basis functions for input pts
                         weights(i) / denom.array() *       // rationalized
                         Rk.col(j).array()).sum();          // input points
#else                                               // don't weigh domain coordinate (only range)
            // debug
//             cerr << "N in RHS:\n" << N << endl;
//             cerr << "weights in RHS:\n" << weights << endl;
//             cerr << "denom in RHS:\n" << denom << endl;

            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
            {
                // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                for (int j = 0; j < R.cols() - 1; j++)
                    R(i, j) =
                        (N.col(i).array() *                 // ith basis functions for input pts
                         Rk.col(j).array()).sum();          // input points
                R(i, last) =
                    (N.col(i).array() *                     // ith basis functions for input pts
                     weights(i) / denom.array() *           // rationalized
                     Rk.col(last).array()).sum();           // input points
            }
#endif

            // debug
//             cerr << "R:\n" << R << endl;
        }


        // computes right hand side vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from a
        // new set of input points, not the default input domain
        // includes multiplication by weights
        // R is column vector of n + 1 elements, each element multiple coordinates of the input points
        void RHS(
                int                 cur_dim,  // current dimension
                const MatrixX<T>&   in_pts,   // input points (not the default domain stored in the mfa)
                const MatrixX<T>&   N,        // matrix of basis function coefficients
                MatrixX<T>&         R,        // (output) residual matrix allocated by caller
                const VectorX<T>&   weights,  // precomputed weights for n + 1 control points on this curve
                int                 co,       // index of starting input pt in current curve
                int                 cs)       // stride of input pts in current curve
        {
            int last   = R.cols() - 1;                                  // column of range value TODO: weighing only the last column does not make much sense in the split model
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);     // one row for each input point
            VectorX<T> denom(N.rows());                                 // rational denomoninator for param of each input point

            for (int k = 0; k < N.rows(); k++)
            {
                denom(k) = (N.row(k).cwiseProduct(weights.transpose())).sum();
                Rk.row(k) = in_pts.row(co + k * cs);
            }

#ifdef WEIGH_ALL_DIMS                               // weigh all dimensions
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                 // ith basis functions for input pts
                         weights(i) / denom.array() *       // rationalized
                         Rk.col(j).array()).sum();          // input points
#else                                               // don't weigh domain coordinate (only range)
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
            {
                // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                for (int j = 0; j < R.cols() - 1; j++)
                    R(i, j) =
                        (N.col(i).array() *                 // ith basis functions for input pts
                         Rk.col(j).array()).sum();          // input points
                R(i, last) =
                    (N.col(i).array() *                     // ith basis functions for input pts
                     weights(i) / denom.array() *           // rationalized
                     Rk.col(last).array()).sum();           // input points
            }
#endif

            // debug
            //     cerr << "R:\n" << R << endl;
        }

        // Checks quantities needed for approximation
        void Quants(
                const VectorXi& nctrl_pts,      // number of control points
                VectorXi&       n,              // (output) number of control point spans in each dim
                VectorXi&       m)              // (output) number of input data point spans in each dim
        {
            if (mfa_data.p.size() != mfa.ndom_pts().size())
            {
                fprintf(stderr, "Error: Encode() size of p must equal size of ndom_pts\n");
                exit(1);
            }
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                if (nctrl_pts(i) <= mfa_data.p(i))
                {
                    fprintf(stderr, "Error: Encode() number of control points in dimension %ld "
                            "must be at least p + 1 for dimension %ld\n", i, i);
                    exit(1);
                }
                if (nctrl_pts(i) > mfa.ndom_pts()(i))
                {
                    fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                            "exceeds number of input data points (%d) in dimension %ld.\n", nctrl_pts(i), i, mfa.ndom_pts()(i), i);
                }
            }

            n.resize(mfa_data.p.size());
            m.resize(mfa_data.p.size());
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                n(i)        =  nctrl_pts(i) - 1;
                m(i)        =  mfa.ndom_pts()(i)  - 1;
            }
        }

        // solves for one curve of control points
        // outputs go to specified control points and weights matrix and vector rather than default mfa
        void CtrlCurve(
                const MatrixX<T>&   N,                  // basis functions for current dimension
                const MatrixX<T>&   NtN,                // Nt * N
                MatrixX<T>&         R,                  // (output) residual matrix for current dimension and curve
                MatrixX<T>&         P,                  // (output) solved points for current dimension and curve
                size_t              k,                  // current dimension
                size_t              co,                 // starting ofst for reading domain pts
                size_t              cs,                 // stride for reading domain points
                size_t              to,                 // starting ofst for writing control pts
                MatrixX<T>&         temp_ctrl0,         // first temporary control points buffer
                MatrixX<T>&         temp_ctrl1,         // second temporary control points buffer
                int                 curve_id,           // debugging
                MatrixX<T>&         ctrl_pts,           // (output) control points
                VectorX<T>&         weights,            // (output) weights
                bool                weighted = true)    // solve for and use weights
        {
            // solve for weights
            // TODO: avoid copying into Q by passing temp_ctrl0, temp_ctrl1, co, cs to Weights()
            // TODO: check that this is right, using co and cs for copying control points and domain points
            MatrixX<T> Q;
            Q.resize(mfa.ndom_pts()(k), ctrl_pts.cols());
            if (k == 0)
            {
                for (auto i = 0; i < mfa.ndom_pts()(k); i++)
                    Q.row(i) = domain.block(co + i * cs, mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
            }
            else if (k % 2)
            {
                for (auto i = 0; i < mfa.ndom_pts()(k); i++)
                    Q.row(i) = temp_ctrl0.row(co + i * cs);
            }
            else
            {
                for (auto i = 0; i < mfa.ndom_pts()(k); i++)
                    Q.row(i) = temp_ctrl1.row(co + i * cs);
            }

            VectorX<T> temp_weights = VectorX<T>::Ones(N.cols());

#ifndef MFA_NO_WEIGHTS

            if (weighted)
                if (k == mfa_data.dom_dim - 1)                               // only during last dimension of separable iteration over dimensions
                    Weights(k, Q, N, NtN, curve_id, temp_weights);      // solve for weights

#endif

            // compute R
            // first dimension reads from domain
            // subsequent dims alternate reading temp_ctrl0 and temp_ctrl1
            // even dim reads temp_ctrl1, odd dim reads temp_ctrl0; opposite of writing order
            // because what was written in the previous dimension is read in the current one
            if (k == 0)
                RHS(k, N, R, temp_weights, co);                         // input points = default domain
            else if (k % 2)
                RHS(k, temp_ctrl0, N, R, temp_weights, co, cs);         // input points = temp_ctrl0
            else
                RHS(k, temp_ctrl1, N, R, temp_weights, co, cs);         // input points = temp_ctrl1

            // rationalize NtN, ie, weigh the basis function coefficients
            MatrixX<T> NtN_rat = NtN;
            mfa_data.Rationalize(k, temp_weights, N, NtN_rat);

            // solve for P
#ifdef WEIGH_ALL_DIMS                                   // weigh all dimensions
            P = NtN_rat.ldlt().solve(R);
#else                                                   // don't weigh domain coordinate (only range)
            // TODO: avoid 2 solves?
            MatrixX<T> P2(P.rows(), P.cols());
            P = NtN.ldlt().solve(R);                            // nonrational domain coordinates
            P2 = NtN_rat.ldlt().solve(R);                       // rational range coordinate
            for (auto i = 0; i < P.rows(); i++)
                P(i, P.cols() - 1) = P2(i, P.cols() - 1);
#endif

            // append points from P to control points that will become inputs for next dimension
            // TODO: any way to avoid this?
            CopyCtrl(P, k, co, cs, to, ctrl_pts, temp_ctrl0, temp_ctrl1);

            // copy weights of final dimension to mfa
            if (k == mfa_data.dom_dim - 1)
            {
                for (auto i = 0; i < temp_weights.size(); i++)
                    weights(to + i * cs) = temp_weights(i);
            }
        }

        // solves for one curve of control points
        void CtrlCurve(
                const MatrixX<T>&   N,                  // basis functions for current dimension
                const MatrixX<T>&   NtN,                // Nt * N
                MatrixX<T>&         R,                  // residual matrix for current dimension and curve
                MatrixX<T>&         P,                  // solved points for current dimension and curve
                size_t              k,                  // current dimension
                size_t              co,                 // starting ofst for reading domain pts
                size_t              cs,                 // stride for reading domain points
                size_t              to,                 // starting ofst for writing control pts
                MatrixX<T>&         temp_ctrl0,         // first temporary control points buffer
                MatrixX<T>&         temp_ctrl1,         // second temporary control points buffer
                int                 curve_id,           // debugging
                TensorProduct<T>&   tensor,             // (output) tensor product containing result
                bool                weighted = true)    // solve for and use weights
        {
            // solve for weights
            // TODO: avoid copying into Q by passing temp_ctrl0, temp_ctrl1, co, cs to Weights()
            // TODO: check that this is right, using co and cs for copying control points and domain points
            MatrixX<T> Q;
            Q.resize(mfa.ndom_pts()(k), tensor.ctrl_pts.cols());
            if (k == 0)
            {
                for (auto i = 0; i < mfa.ndom_pts()(k); i++)
                    Q.row(i) = domain.block(co + i * cs, mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
            }
            else if (k % 2)
            {
                for (auto i = 0; i < mfa.ndom_pts()(k); i++)
                    Q.row(i) = temp_ctrl0.row(co + i * cs);
            }
            else
            {
                for (auto i = 0; i < mfa.ndom_pts()(k); i++)
                    Q.row(i) = temp_ctrl1.row(co + i * cs);
            }

            VectorX<T> weights = VectorX<T>::Ones(N.cols());

#ifndef MFA_NO_WEIGHTS

            if (weighted)
                if (k == mfa_data.dom_dim - 1)                      // only during last dimension of separable iteration over dimensions
                    Weights(k, Q, N, NtN, curve_id, weights);   // solve for weights

#endif

            // compute R
            // first dimension reads from domain
            // subsequent dims alternate reading temp_ctrl0 and temp_ctrl1
            // even dim reads temp_ctrl1, odd dim reads temp_ctrl0; opposite of writing order
            // because what was written in the previous dimension is read in the current one
            if (k == 0)
                RHS(k, N, R, weights, co);                 // input points = default domain
            else if (k % 2)
                RHS(k, temp_ctrl0, N, R, weights, co, cs); // input points = temp_ctrl0
            else
                RHS(k, temp_ctrl1, N, R, weights, co, cs); // input points = temp_ctrl1

            // rationalize NtN, ie, weigh the basis function coefficients
            MatrixX<T> NtN_rat = NtN;
            mfa_data.Rationalize(k, weights, N, NtN_rat);

            // solve for P
#ifdef WEIGH_ALL_DIMS                                   // weigh all dimensions
            P = NtN_rat.ldlt().solve(R);
#else                                                   // don't weigh domain coordinate (only range)
            // TODO: avoid 2 solves?
            MatrixX<T> P2(P.rows(), P.cols());
            P = NtN.ldlt().solve(R);                            // nonrational domain coordinates
            P2 = NtN_rat.ldlt().solve(R);                       // rational range coordinate
            for (auto i = 0; i < P.rows(); i++)
                P(i, P.cols() - 1) = P2(i, P.cols() - 1);
#endif

            // append points from P to control points that will become inputs for next dimension
            // TODO: any way to avoid this?
            CopyCtrl(P, k, co, cs, to, tensor, temp_ctrl0, temp_ctrl1);

            // copy weights of final dimension to mfa
            if (k == mfa_data.dom_dim - 1)
            {
                for (auto i = 0; i < weights.size(); i++)
                    tensor.weights(to + i * cs) = weights(i);
            }
        }

        // append solved control points from P to become inputs for next dimension
        // TODO: any way to avoid this copy?
        // last dimension gets copied to final control points
        // This version specifies a location for ctrl_pts rather than default one in mfa
        // previous dimensions get copied to alternating double buffers
        void CopyCtrl(
                const MatrixX<T>&   P,              // solved points for current dimension and curve
                int                 k,              // current dimension
                size_t              co,             // starting offset for reading domain points
                size_t              cs,             // stride for reading domain points
                size_t              to,             // starting offset for writing control points
                MatrixX<T>&         ctrl_pts,       // (output) control points
                MatrixX<T>&         temp_ctrl0,     // (output) first temporary control points buffer
                MatrixX<T>&         temp_ctrl1)     // (output) second temporary control points buffer
        {
            int ndims = mfa.ndom_pts().size();    // number of domain dimensions

            // if there is only one dim, copy straight to output
            if (ndims == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // first dim copied from domain to temp_ctrl0
            else if (k == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // even numbered dims (but not the last one) copied from P to temp_ctrl0
            else if (k % 2 == 0 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // odd numbered dims (but not the last one) copied from P to temp_ctrl1
            else if (k % 2 == 1 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl1.row(to + i * cs) = P.row(i);
            }
            // final dim if even is copied from temp_ctrl1 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // final dim if odd is copied from temp_ctrl0 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    ctrl_pts.row(to + i * cs) = P.row(i);
            }
        }

        // append solved control points from P to become inputs for next dimension
        // TODO: any way to avoid this copy?
        // last dimension gets copied to final control points
        // previous dimensions get copied to alternating double buffers
        void CopyCtrl(
                const MatrixX<T>&   P,              // solved points for current dimension and curve
                int                 k,              // current dimension
                size_t              co,             // starting offset for reading domain points
                size_t              cs,             // stride for reading domain points
                size_t              to,             // starting offset for writing control points
                TensorProduct<T>&   tensor,         // (output) tensor product containing result
                MatrixX<T>&         temp_ctrl0,     // first temporary control points buffer
                MatrixX<T>&         temp_ctrl1)     // second temporary control points buffer
        {
            int ndims = mfa.ndom_pts().size();    // number of domain dimensions

            // if there is only one dim, copy straight to output
            if (ndims == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    tensor.ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // first dim copied from domain to temp_ctrl0
            else if (k == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // even numbered dims (but not the last one) copied from P to temp_ctrl0
            else if (k % 2 == 0 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl0.row(to + i * cs) = P.row(i);
            }
            // odd numbered dims (but not the last one) copied from P to temp_ctrl1
            else if (k % 2 == 1 && k < ndims - 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    temp_ctrl1.row(to + i * cs) = P.row(i);
            }
            // final dim if even is copied from temp_ctrl1 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 0)
            {
                for (int i = 0; i < P.rows(); i++)
                    tensor.ctrl_pts.row(to + i * cs) = P.row(i);
            }
            // final dim if odd is copied from temp_ctrl0 to ctrl_pts
            else if (k == ndims - 1 && k % 2 == 1)
            {
                for (int i = 0; i < P.rows(); i++)
                    tensor.ctrl_pts.row(to + i * cs) = P.row(i);
            }
        }

        // computes new knots to be inserted into a curve
        // for each current knot span where the error is greater than the limit, finds the domain point
        // where the error is greatest and adds the knot at that parameter value
        //
        // this version takes a set of control points as input instead of mfa_data.ctrl_pts
        int ErrorCurve(
                size_t                  k,          // current dimension
                const TensorProduct<T>& tensor,     // current tensor product
                size_t                  co,         // starting ofst for reading domain pts
                const MatrixX<T>&       ctrl_pts,   // control points
                const VectorX<T>&       weights,    // weights associated with control points
                VectorX<T>              extents,    // extents in each dimension, for normalizing error (size 0 means do not normalize)
                set<int>&               err_spans,  // (output) spans with error greater than err_limit
                T                       err_limit)  // max allowable error
        {
            mfa::Decoder<T> decoder(mfa, mfa_data, verbose);
            int pt_dim = tensor.ctrl_pts.cols();            // control point dimensonality
            VectorX<T> cpt(pt_dim);                         // decoded curve point
            int nerr = 0;                                   // number of points with error greater than err_limit
            int span = mfa_data.p[k];                            // current knot span of the domain point being checked
            if (!extents.size())
                extents = VectorX<T>::Ones(domain.cols());

            for (auto i = 0; i < mfa.ndom_pts()[k]; i++)      // all domain points in the curve
            {
                while (mfa_data.tmesh.all_knots[k][span + 1] < 1.0 && mfa_data.tmesh.all_knots[k][span + 1] <= mfa.params()[k][i])
                    span++;

                decoder.CurvePt(k, mfa.params()[k][i], ctrl_pts, weights, tensor, cpt);


                // error
                T max_err = 0.0;
                for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
                {
                    T err = fabs(cpt(j) - domain(co + i * mfa.ds()[k], mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
                    max_err = err > max_err ? err : max_err;
                }

                if (max_err > err_limit)
                {
                    // don't duplicate spans
                    set<int>::iterator it = err_spans.find(span);
                    if (!err_spans.size() || it == err_spans.end())
                    {
                        // ensure there would be a domain point in both halves of the span if it were split
                        bool split_left = false;
                        for (auto j = i; mfa.params()[k][j] >= mfa_data.tmesh.all_knots[k][span]; j--)
                            if (mfa.params()[k][j] < (mfa_data.tmesh.all_knots[k][span] + mfa_data.tmesh.all_knots[k][span + 1]) / 2.0)
                            {
                                split_left = true;
                                break;
                            }
                        bool split_right = false;
                        for (auto j = i; mfa.params()[k][j] < mfa_data.tmesh.all_knots[k][span + 1]; j++)
                            if (mfa.params()[k][j] >= (mfa_data.tmesh.all_knots[k][span] + mfa_data.tmesh.all_knots[k][span + 1]) / 2.0)
                            {
                                split_right = true;
                                break;
                            }
                        // mark the span and count the point if the span can (later) be split
                        if (split_left && split_right)
                            err_spans.insert(it, span);
                    }
                    // count the point in the total even if the span is not marked for splitting
                    // total used to find worst curve, defined as the curve with the most domain points in
                    // error (including multiple domain points per span and points in spans that can't be
                    // split further)
                    nerr++;
                }
            }

            return nerr;
        }

#ifdef TMESH

        // this is the version used currently for tmesh and optional local solve
        // encodes at full dimensionality and decodes at full dimensionality
        // decodes full-d points in each knot span and adds new knot spans where error > err_limit
        // returns 1 if knots were added, 0 if no knots were added, -1 if number of control points >= input points
        int NewKnots_full(
                T                   err_limit,                  // max allowable error
                const VectorX<T>&   extents,                    // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 iter,                       // iteration number of caller (for debugging)
                bool                local,                      // solve locally (with constraints) each round
                const VectorXi&     nctrl_pts,                  // number of control points in each dimension
                const MatrixX<T>&   ctrl_pts,                   // control points
                const VectorX<T>&   weights)                    // weights
        {
            // debug
            if (local)
                fprintf(stderr, "*** Using local solve in NewKnots_full ***\n");

            bool done = true;

            // indices in tensor, in each dim. of inserted knots in full knot vector after insertion
            vector<vector<KnotIdx>> inserted_knot_idxs(mfa_data.dom_dim);

            VectorX<T> myextents = extents.size() ? extents : VectorX<T>::Ones(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());

            // find new knots
            mfa::NewKnots<T> nk(mfa, mfa_data);

            // vectors of new_nctrl_pts, new_ctrl_pts, new_weights, one instance for each knot to be inserted
            // we're only inserting one knot at a time, but the NewKnots object supports multiple knot insertions, hence std::vector
            vector<VectorXi>    new_nctrl_pts;
            vector<MatrixX<T>>  new_ctrl_pts;
            vector<VectorX<T>>  new_weights;

            if (local)
                done &= nk.FirstErrorSpan(domain, myextents, err_limit, iter, nctrl_pts, ctrl_pts, weights, inserted_knot_idxs, new_nctrl_pts, new_ctrl_pts, new_weights);
            else
            {
            // TODO: temporarily call TempFirstErrorSpan insted of FirstErrorSpan, passing full set of control points and weights
            // Once adaptive algorithm is in place, call FirstErrorSpan instead
//             done &= nk.FirstErrorSpan(domain, myextents, err_limit, iter, nctrl_pts, inserted_knot_idxs);
                done &= nk.TempFirstErrorSpan(domain, myextents, err_limit, iter, nctrl_pts, ctrl_pts, weights, inserted_knot_idxs);
            }

            if (local)
                assert(inserted_knot_idxs[0].size() == new_ctrl_pts.size());     // sanity, number of inserted knots is consistent across things that depend on it

            if (done)
                return 0;

            // append new tensors
            vector<KnotIdx> knot_mins(mfa_data.dom_dim);
            vector<KnotIdx> knot_maxs(mfa_data.dom_dim);
            auto ntensors = mfa_data.tmesh.tensor_prods.size();         // number of tensors before any additional appends
            for (auto i = 0; i < ntensors; i++)                         // for all existing tensors
            {
                TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[i];

                for (auto j = 0; j < mfa_data.dom_dim; j++)
                {
                    KnotIdx min_idx, max_idx;
                    for (auto k = 0; k < inserted_knot_idxs[j].size(); k++)
                    {
                        // inserted knot falls into the mins, maxs of this tensor
                        if (inserted_knot_idxs[j][k] - mfa_data.p(j) / 2 >= t.knot_mins[k] && inserted_knot_idxs[j][k] + mfa_data.p(j) / 2 <= t.knot_maxs[k])
                        {
                            // expand knot mins and maxs by p / 2 index lines on each side of added knot
                            // so that the new tensor has p anchors (control pts) in each dimension (needed for local adaptive solve)
                            assert(inserted_knot_idxs[j][k] - mfa_data.p(j) / 2 >= 0);
                            assert(inserted_knot_idxs[j][k] + mfa_data.p(j) / 2 < mfa_data.tmesh.all_knots[j].size());
                            if (k == 0 || inserted_knot_idxs[j][k] < min_idx)
                            {
                                knot_mins[j] = inserted_knot_idxs[j][k] - mfa_data.p(j) / 2;
                                min_idx = inserted_knot_idxs[j][k];
                            }
                            if (k == 0 || inserted_knot_idxs[j][k] > max_idx)
                            {
                                knot_maxs[j] = inserted_knot_idxs[j][k] + mfa_data.p(j) / 2;
                                max_idx = inserted_knot_idxs[j][k];
                            }
                        }
                    }
                }

                // debug
                if (mfa_data.dom_dim == 1)
                    fprintf(stderr, "appending tensor with knot_mins [%ld] knot_maxs [%ld]\n", knot_mins[0], knot_maxs[0]);
                else if (mfa_data.dom_dim == 2)
                    fprintf(stderr, "appending tensor with knot_mins [%ld %ld] knot_maxs [%ld %ld]\n", knot_mins[0], knot_mins[1], knot_maxs[0], knot_maxs[1]);
                else if (mfa_data.dom_dim == 3)
                    fprintf(stderr, "appending tensor with knot_mins [%ld %ld %ld] knot_maxs [%ld %ld %ld]\n",
                            knot_mins[0], knot_mins[1], knot_mins[2], knot_maxs[0], knot_maxs[1], knot_maxs[2]);

                // only doing one new knot insertion, hence the [0] index on new_nctrl_pts, new_ctrl_pts, new_weights
                mfa_data.tmesh.append_tensor(knot_mins, knot_maxs, new_nctrl_pts[0], new_ctrl_pts[0], new_weights[0]);

                // debug
                mfa_data.tmesh.print();

                if (local)      // local solve for the newly appended tensor
                {
                    // TODO hard-coded for 1D and even degree
                    // TODO: check all of the below when p is odd (not too bad)
                    // TODO: expand all of the below for higher dimensions (considerably more work)
                    int p = mfa_data.p[0];
                    const TensorProduct<T>& tc = mfa_data.tmesh.tensor_prods.back();            // current (newly appended) tensor
                    const TensorProduct<T>& tp = mfa_data.tmesh.tensor_prods[tc.prev[0][0]];    // previous tensor
                    const TensorProduct<T>& tn = mfa_data.tmesh.tensor_prods[tc.next[0][0]];    // next tensor
                    MatrixX<T> ctrlpts_tosolve(3 * p, tc.ctrl_pts.cols());                      // control points to solve, p interior and p constraints on each side
                    ctrlpts_tosolve.block(0, 0, p, 1)       = tp.ctrl_pts.block(tp.ctrl_pts.rows() - p, 0, p, 1);   // left constraint
                    ctrlpts_tosolve.block(p, 0, p, 1)       = tc.ctrl_pts;                                          // unconstrained interior
                    ctrlpts_tosolve.block(2 * p, 0, p, 1)   = tn.ctrl_pts.block(0, 0, p, 1);                        // right constraint

                    // set the constraints
                    MatrixX<T> cons         = ctrlpts_tosolve;
                    cons.block(p, 0, p, 1)  = MatrixX<T>::Zero(p, 1);       // zero out the unconstrained interior

                    // get the subset of the domain points needed for the local solve

                    vector<KnotIdx> anchor;                                 // anchor for the edge basis functions of the new tensor
                    vector<vector<KnotIdx>> local_knot_idxs;                // local knot vector for an anchor

                    // left edge
                    anchor.push_back(tp.knot_maxs[0] - p);
                    mfa_data.tmesh.local_knot_vector(anchor, local_knot_idxs);
                    KnotIdx start_knot_idx = local_knot_idxs[0][0];
                    T start_knot = mfa_data.tmesh.all_knots[0][start_knot_idx];

                    anchor.clear();
                    local_knot_idxs.clear();

                    // right edge
                    anchor.push_back(tn.knot_mins[0] + p - 1);
                    mfa_data.tmesh.local_knot_vector(anchor, local_knot_idxs);
                    KnotIdx end_knot_idx = local_knot_idxs[0].back();
                    T end_knot = mfa_data.tmesh.all_knots[0][end_knot_idx];

                    // search params for start and end knot values
                    // TODO: use ijk2idx to get the actual indices
                    auto it = std::lower_bound(mfa.params()[0].begin(), mfa.params()[0].end(), start_knot);
                    size_t subdomain_start_idx = it - mfa.params()[0].begin();
                    it = std::upper_bound(mfa.params()[0].begin(), mfa.params()[0].end(), end_knot);
                    size_t subdomain_end_idx = it - mfa.params()[0].begin() - 1;

                    // set up the optimization
                    LocalLSQ<T> f(mfa, mfa_data, domain, cons, subdomain_start_idx, subdomain_end_idx, verbose);
                    BfgsSolver<LocalLSQ<T>> solver;

                    // minimize the function
                    VectorX<T> x1(Eigen::Map<VectorX<T>>(ctrlpts_tosolve.data(), ctrlpts_tosolve.size()));
                    solver.minimize(f, x1);

                }       // local solve for the newly appended tensor
            }       // for all existing tensors

            for (auto k = 0; k < mfa_data.dom_dim; k++)
                if (mfa.ndom_pts()(k) <= nctrl_pts(k))
                    return -1;

            return 1;
        }

#endif      // TMESH

        // 1d encoding and 1d decoding
        // adds knots error spans from all curves in all directions (into a set)
        // adds knots in middles of spans that have error higher than the limit
        // returns true if done, ie, no knots are inserted
        bool NewKnots_curve(
                vector<vector<T>>&  new_knots,                              // (output) new knots
                T                   err_limit,                              // max allowable error
                const VectorX<T>&   extents,                                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 iter)                                   // iteration number of caller (for debugging)
        {
            int     pt_dim          = mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols();    // control point dimensonality
            size_t  tot_nnew_knots  = 0;                                            // total number of new knots found
            new_knots.resize(mfa_data.dom_dim);
            vector<vector<int>> new_levels(mfa_data.dom_dim);

            for (TensorProduct<T>& t : mfa_data.tmesh.tensor_prods)             // for all tensor products in the tmesh
            {
                // check and assign main quantities
                VectorXi n = t.nctrl_pts - VectorXi::Ones(mfa_data.dom_dim);     // number of control point spans in each domain dim
                VectorXi m = mfa.ndom_pts()  - VectorXi::Ones(mfa_data.dom_dim);   // number of input data point spans in each domain dim

                // resize control points and weights
                t.ctrl_pts.resize(t.nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
                t.weights = VectorX<T>::Ones(t.ctrl_pts.rows());

                for (size_t k = 0; k < mfa_data.dom_dim; k++)                    // for all domain dimensions
                {
                    new_knots[k].resize(0);
                    new_levels[k].resize(0);

                    // for now set weights to 1, TODO: get weights from elsewhere
                    // NB: weights are for all n + 1 control points, not just the n -1 interior ones
                    VectorX<T> weights = VectorX<T>::Ones(n(k) + 1);

                    // temporary control points for one curve
                    MatrixX<T> temp_ctrl = MatrixX<T>::Zero(t.nctrl_pts(k), pt_dim);

                    // error spans for one curve and for worst curve
                    set<int> err_spans;

                    // maximum number of domain points with error greater than err_limit and their curves
                    size_t max_nerr     =  0;

                    // N is a matrix of (m + 1) x (n + 1) scalars that are the basis function coefficients
                    //  _                          _
                    // |  N_0(u[0])   ... N_n(u[0]) |
                    // |     ...      ...      ...  |
                    // |  N_0(u[m])   ... N_n(u[m]) |
                    //  -                          -
                    // TODO: N is going to be very sparse when it is large: switch to sparse representation
                    // N has semibandwidth < p  nonzero entries across diagonal
                    MatrixX<T> N = MatrixX<T>::Zero(m(k) + 1, n(k) + 1);    // coefficients matrix

                    for (int i = 0; i < N.rows(); i++)                      // the rows of N
                    {
                        // TODO: hard-coded for single tensor
                        int span = mfa_data.FindSpan(k, mfa.params()[k][i], mfa_data.tmesh.tensor_prods[0]);
#ifndef TMESH           // original version for one tensor product
                        mfa_data.OrigBasisFuns(k, mfa.params()[k][i], span, N, i);
#else                   // tmesh version
                        mfa_data.BasisFuns(k, mfa.params()[k][i], span, N, i);
#endif
                    }

                    // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
                    // NtN has semibandwidth < p + 1 nonzero entries across diagonal
                    MatrixX<T> NtN = N.transpose() * N;

                    // R is the right hand side needed for solving NtN * P = R
                    MatrixX<T> R(N.cols(), pt_dim);

                    // P are the unknown interior control points and the solution to NtN * P = R
                    // NtN is positive definite -> do not need pivoting
                    // TODO: use a common representation for P and ctrl_pts to avoid copying
                    MatrixX<T> P(N.cols(), pt_dim);

                    size_t ncurves   = domain.rows() / mfa.ndom_pts()(k);     // number of curves in this dimension
                    int nsame_steps  = 0;                                   // number of steps with same number of erroneous points
                    int n_step_sizes = 0;                                   // number of step sizes so far

                    // starting step size over curves
                    size_t s0 = ncurves / 2 > 0 ? ncurves / 2 : 1;

                    // debug, only one step size s=1
                    //         s0 = 1;

                    for (size_t s = s0; s >= 1 && ncurves / s < max_num_curves; s /= 2)     // for all step sizes over curves
                    {
                        bool new_max_nerr = false;                          // this step size changed the max_nerr

                        for (size_t j = 0; j < ncurves; j++)                // for all the curves in this dimension
                        {
                            // each time the step changes, shift start of s-th curves by one (by subtracting
                            // n_step-sizes below)
                            if (j >= n_step_sizes && (j - n_step_sizes) % s == 0)           // this is one of the s-th curves; compute it
                            {
                                // compute R from input domain points
                                RHS(k, N, R, weights, mfa.co()[k][j]);

                                // rationalize NtN
                                MatrixX<T> NtN_rat = NtN;
                                mfa_data.Rationalize(k, weights, N, NtN_rat);

                                // solve for P
#ifdef WEIGH_ALL_DIMS                                                       // weigh all dimensions
                                P = NtN_rat.ldlt().solve(R);
#else                                                                       // don't weigh domain coordinate (only range)
                                // TODO: avoid 2 solves?
                                MatrixX<T> P2(P.rows(), P.cols());
                                P = NtN.ldlt().solve(R);                    // nonrational domain coordinates
                                P2 = NtN_rat.ldlt().solve(R);               // rational range coordinate
                                for (auto i = 0; i < P.rows(); i++)
                                    P(i, P.cols() - 1) = P2(i, P.cols() - 1);
#endif

                                // compute the error on the curve (number of input points with error > err_limit)
                                size_t nerr = ErrorCurve(k, t, mfa.co()[k][j], P, weights, extents, err_spans, err_limit);

                                if (nerr > max_nerr)
                                {
                                    max_nerr     = nerr;
                                    new_max_nerr = true;
                                }
                            }
                        }                                                   // curves in this dimension

                        // stop refining step if no change
                        if (max_nerr && !new_max_nerr)
                            nsame_steps++;
                        if (nsame_steps == 2)
                            break;

                        n_step_sizes++;
                    }                                                       // step sizes over curves

                    // free R, NtN, and P
                    R.resize(0, 0);
                    NtN.resize(0, 0);
                    P.resize(0, 0);

                    // add new knots in the middle of spans with errors
                    new_knots[k].resize(err_spans.size());
                    new_levels[k].resize(new_knots[k].size());
                    tot_nnew_knots += new_knots[k].size();
                    size_t i = 0;                                           // index into new_knots
                    for (set<int>::iterator it = err_spans.begin(); it != err_spans.end(); ++it)
                    {
                        // debug
                        assert(*it < t.nctrl_pts[k]);                       // not trying to go beyond the last span

                        new_knots[k][i] = (mfa_data.tmesh.all_knots[k][*it] + mfa_data.tmesh.all_knots[k][*it + 1]) / 2.0;
                        new_levels[k][i] = t.level;
                        i++;
                    }

                    // print progress
                    //         fprintf(stderr, "\rdimension %ld of %d encoded\n", k + 1, mfa_data.dom_dim);
                }                                                           // domain dimensions

                // insert the new knots
                mfa::NewKnots<T> nk(mfa, mfa_data);
                vector<vector<KnotIdx>> unused(mfa_data.dom_dim);
                nk.InsertKnots(new_knots, new_levels, unused);

                // increase number of control points, weights, basis functions
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                {
                    t.nctrl_pts(k) += new_knots[k].size();
                    mfa_data.N[k] = MatrixX<T>::Zero(mfa_data.N[k].rows(), t.nctrl_pts(k));
                }
                auto tot_nctrl_pts = t.nctrl_pts.prod();
                t.ctrl_pts.resize(tot_nctrl_pts, t.ctrl_pts.cols());
                t.weights =  VectorX<T>::Ones(tot_nctrl_pts);
            }                                                               // tensor products

            // debug
//             cerr << "new_knots:\n"  << endl;
//             for (auto i = 0; i < new_knots.size(); i++)
//             {
//                 for (auto j = 0; j < new_knots[i].size(); j++)
//                     cerr << new_knots[i][j] << " ";
//                  cerr << endl;
//             }
//             cerr << endl;

            return(tot_nnew_knots ? 0 : 1);
        }
    };

    template <typename T>
    T LocalLSQ<T>::value(const TVector &x)
    {
        // TODO: hard-coded for 1-d and possibly for even degree (need to check odd degree)

        // upack the candidate solution vector x into tensor_prods
        int p = mfa_data.p[0];
        TensorProduct<T>& tc = mfa_data.tmesh.tensor_prods.back();          // current (newly appended) tensor
        TensorProduct<T>& tp = mfa_data.tmesh.tensor_prods[tc.prev[0][0]];  // previous tensor
        TensorProduct<T>& tn = mfa_data.tmesh.tensor_prods[tc.next[0][0]];  // next tensor
        // TODO: Youssef had the next two lines, which I changed from resize to conservativeResize, but regardless, does not properly reshape a vector to a matrix of > 1 column
        // Eigen::Map is the right way to do this, but I can't get Map to compile
        MatrixX<T> ctrlpts_tosolve(x);
        ctrlpts_tosolve.conservativeResize(3 * p, tc.ctrl_pts.cols());
//         const Eigen::Map<MatrixX<T>> ctrlpts_tosolve(x.data(), 3 * p, tc.ctrl_pts.cols());   // TODO: the right way, but will not compile for some reason

        // debug
//         cerr << "x:\n" << x << endl;
//         cerr << "ctrlpts_to_solve ( " << ctrlpts_tosolve.rows() << " x " << ctrlpts_tosolve.cols() << " ):\n" << ctrlpts_tosolve << endl;

        ctrlpts_tosolve = x;                // TODO: check if this right
        tp.ctrl_pts.block(tp.ctrl_pts.rows() - p, 0, p, 1)  = ctrlpts_tosolve.block(0, 0, p, 1);        // left constraint
        tc.ctrl_pts                                         = ctrlpts_tosolve.block(p, 0, p, 1);        // unconstrained
        tn.ctrl_pts.block(0, 0, p, 1)                       = ctrlpts_tosolve.block(2 * p, 0, p, 1);    // right constraint

        // loop from substart to subend, decode.volptTmesh(param(subIdx), cpt) - domain(ijk2idx(subIdx))
        T sum_sq_err = 0;
        mfa::Decoder<T> decoder(mfa, mfa_data, verbose);
        VectorX<T> cpt(tc.ctrl_pts.cols());                                 // decoded curve point
        VectorX<T> param(mfa_data.p.size());                                // parameters for one point

        for (size_t idx = start_idx; idx <= end_idx; ++idx)
        {
            param(0) = mfa.params()[0][idx];
            decoder.VolPt_tmesh(param, cpt);
            T diff = cpt[0] - domain(idx, 0);
            sum_sq_err += (diff * diff);
        }

        fprintf(stderr, "least squares error: %e\n", sum_sq_err);
        if (cons.rows() == ctrlpts_tosolve.rows() && cons.cols() == ctrlpts_tosolve.cols())
        {
            T cons_residual = (ctrlpts_tosolve - cons).squaredNorm();
            fprintf(stderr, "constraints residual: %e\n", cons_residual);
            sum_sq_err += 1e8 * cons_residual;                              // multiplying by 1e8 forces constraints to be satisfied
        }
        return sum_sq_err;
    }
}

#endif
