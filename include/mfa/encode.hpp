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
#include    <cppoptlib/solver/lbfgssolver.h>
#include    <cppoptlib/solver/lbfgsbsolver.h>
#include    <cppoptlib/solver/newtondescentsolver.h>
#include    <cppoptlib/solver/cmaessolver.h>
#include    <cppoptlib/solver/neldermeadsolver.h>
#include    <cppoptlib/solver/conjugatedgradientdescentsolver.h>
#include    <cppoptlib/solver/gradientdescentsolver.h>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

template <typename T>                       // float or double
class NewKnots;

template <typename T>                               // float or double
struct MFA;

namespace mfa
{
    using namespace cppoptlib;

    template <typename T>                   // float or double
    class LocalLSQ : public Problem<T>
    {
    private:
        const MFA<T>&       mfa;            // the mfa object
        MFA_Data<T>&        mfa_data;       // the mfa data object
        const MatrixX<T>&   domain;         // input points
        const MatrixX<T>&   cons;           // control point constraints Matrix
        vector<size_t>      start_idxs;     // start and end of the local subdomain
        vector<size_t>      end_idxs;       // in input point space
        int                 verbose;        // more output
        size_t              niters;         // number of iterations
        T                   lsq_error;      // error of decoding points
        T                   cons_error;     // error of constraints
        T                   tot_error;      // total error = lsq_error + weight * cons_error

    public:
        LocalLSQ(const MFA<T>&      mfa_,
                 MFA_Data<T>&       mfa_data_,
                 const MatrixX<T>&  domain_,
                 const MatrixX<T>&  cons_,
                 vector<size_t>     starts_,
                 vector<size_t>     ends_,
                 int                verb_): mfa(mfa_),
                                            mfa_data(mfa_data_),
                                            domain(domain_),
                                            cons(cons_),
                                            start_idxs(starts_),
                                            end_idxs(ends_),
                                            verbose(verb_),
                                            niters(0)

        {}

        ~LocalLSQ()                         {}

        using typename Problem<T>::TVector;

        void setConstraints(const MatrixX<T> c) {cons = c;}
        size_t iters() { return niters; }

        // objective function evaluation
        // n-d version
        T value(const TVector &x)
        {
            niters++;
            Tmesh<T>&           tmesh   = mfa_data.tmesh;
            TensorProduct<T>&   tc      = tmesh.tensor_prods.back();                            // current (newly appended) tensor
            int                 cols    = tc.ctrl_pts.cols();

            // convert candidate solution vector back to a matrix
            VectorX<T>  x1      = x;                                                            // need non-const vector to pass to Map, cannot find other way than deep copy
            Eigen::Map<MatrixX<T>> ctrlpts_tosolve(x1.data(), x1.size() / cols, cols);          // matrix version of the vector x

            // copy candidate solution back to current tensor control points
            tc.ctrl_pts = ctrlpts_tosolve.block(0, 0, tc.ctrl_pts.rows(), cols);

            // debug
            //         cerr << "x:\n" << x << endl;
            //         cerr << "ctrlpts_to_solve ( " << ctrlpts_tosolve.rows() << " x " << ctrlpts_tosolve.cols() << " ):\n" << ctrlpts_tosolve << endl;

            // loop from substart to subend, decode.VolPt_tmesh(param(idx), cpt) - domain(idx)
            lsq_error = 0.0;
            mfa::Decoder<T> decoder(mfa, mfa_data, verbose);
            VectorX<T> cpt(cols);                                               // decoded curve point
            VectorX<T> param(mfa_data.dom_dim);                                 // parameters for one point
            VectorXi npts(mfa_data.dom_dim);                                    // number of points to decode
            VectorXi starts(mfa_data.dom_dim);                                  // starting indices of points to decode
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                starts(k)   = start_idxs[k];
                npts(k)     = end_idxs[k] - start_idxs[k] + 1;
            }
            VolIterator vol_iter(npts, starts, mfa.ndom_pts());
            VectorXi ijk(mfa_data.dom_dim);

            while(!vol_iter.done())
            {
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    param(k) = mfa.params()[k][vol_iter.idx_dim(k)];

                decoder.VolPt_tmesh(param, cpt);
                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);                     // multi-dim index into domain points
                size_t dom_idx = vol_iter.ijk_idx(ijk);                         // linear index into domain points
                for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
                {
                    T diff = cpt[j] - domain(dom_idx, mfa_data.dom_dim + j);
                    lsq_error += (diff * diff);
                }
                vol_iter.incr_iter();
            }

            // try making lsq_error RMSE
            // don't do this: doubles number of iterations without changing result
            //         lsq_error = sqrt(lsq_error / vol_iter.tot_iters());

            //         fprintf(stderr, "least squares error: %e\n", lsq_error);
            if (cons.rows() && cons.cols())                                     // nonzero size indicates constraints are being used
            {
                cons_error =
                    (ctrlpts_tosolve.block(tc.ctrl_pts.rows(), 0, ctrlpts_tosolve.rows() - tc.ctrl_pts.rows(), cols) - cons).squaredNorm();

                // try making cons_error RMSE
                // don't do this: doubles number of iterations without changing result
                //             cons_error = sqrt(cons_error / cons.size());

                // debug
                //             cerr << "ctrlpts_tosolve:\n" << ctrlpts_tosolve << endl;
                //             cerr << "cons:\n" << cons << endl;
                //             cerr << "ctrlpts_tosolve - cons:\n" << ctrlpts_tosolve.block(p, 0, ctrlpts_tosolve.rows() - p) - cons << endl;

                T cons_weight = 1.0e8;                                          // multiplying by large weight forces constraints to be satisfied 1e8 was Youssef's factor when interior not 0'd
                tot_error = lsq_error + cons_weight * cons_error;

                // debug
//                 fprintf(stderr, "niters = %lu lsq_error = %.3e cons_error = %.3e tot_error (lsq_error + %.3e * cons_error) = %e\n",
//                         niters, lsq_error, cons_error, cons_weight, tot_error);
            }
            else
            {
                tot_error = lsq_error;

                // debug
//                 fprintf(stderr, "niters = %lu lsq_error = tot_error = %e\n", niters, tot_error);
            }
            return tot_error;
        }

//         // objective function gradient
//         // TODO: toy test to see if having a gradient speeds up the evaluation
//         void gradient(const TVector &x, TVector &grad)
//         {
//             Tmesh<T>&           tmesh   = mfa_data.tmesh;
//             TensorProduct<T>&   tc      = tmesh.tensor_prods.back();                            // current (newly appended) tensor
// 
//             // set a fake gradient for the free variables
//             // TODO: caution: this will mess up the result!
//             for (auto i = 0; i < tc.ctrl_pts.rows(); i++)
//                 grad[i] = 1.0;
// 
//             // set the gradient of the constraints to be 0 to try to dissuade the optimizer from moving them
//             for (auto i = 0; i < cons.rows(); i++)
//                 grad[tc.ctrl_pts.rows() + i] = 0.0;
//         }
    };          // LocalLSQ class

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
//             cerr << "Encode() weights:\n" << weights << endl;
        }

        // encodes the control points for one tensor product of a tmesh
        // takes a subset of input points from the global domain, covered by basis functions of this tensor product
        // solves all dimensions together (not separably)
        // does not encode weights for now
        void EncodeTensor(TensorIdx                 t_idx,                  // index of tensor product being encoded
                          bool                      pad,                    // pad input points by degree p in each dimension
                          bool                      constrained,            // whether to impose p control point constraints on each side
                          bool                      weighted = true)        // solve for and use weights
        {
            // debug
            fprintf(stderr, "EncodeTensor\n");

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];

            // get input domain points covered by the tensor
            vector<size_t> start_idxs(mfa_data.dom_dim);
            vector<size_t> end_idxs(mfa_data.dom_dim);
            mfa_data.tmesh.domain_pts(t_idx, pad, mfa.params(), start_idxs, end_idxs);

            // debug: hard-code start and end idxs
//             for (auto i = 0; i < mfa_data.dom_dim; i++)
//             {
//                 start_idxs[i]   = 0;
//                 end_idxs[i]     = mfa.ndom_pts()(i) - 1;
//             }

            VectorXi ndom_pts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
                ndom_pts(k) = end_idxs[k] - start_idxs[k] + 1;

            // debug: try unconstrained
//             constrained = false;

            // get constraints
            MatrixX<T>  cons;

#ifdef TMESH

            if (constrained)
            {
                MatrixX<T> ctrlpts_tosolve;
                // TODO: make a version of LocalSolverCtrlPts that only finds the constraints
                LocalSolveCtrlPts(t, ctrlpts_tosolve);
                cons = ctrlpts_tosolve.block(t.nctrl_pts.prod(), 0, ctrlpts_tosolve.rows() - t.ctrl_pts.rows(), ctrlpts_tosolve.cols());
            }

#endif

            // debug
//             cerr << "cons:\n" << cons << endl;

            // resize matrices in case number of control points changed
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                           // control point dimensonality
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());

            // matrix of basis functions
            // using 0th dimension for full-d
            // N is a matrix of m x n scalars that are the basis function coefficients
            //  _                                      _
            // |  N_0(u[0])     ...     N_n-1(u[0])     |
            // |     ...        ...      ...            |
            // |  N_0(u[m-1])   ...     N_n-1(u[m-1])   |
            //  -                                      -
            MatrixX<T>& N = mfa_data.N[0];
            N = MatrixX<T>::Constant(ndom_pts.prod(), t.nctrl_pts.prod(), -1);              // basis functions, -1 means unassigned so far
            VectorXi dom_starts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
                dom_starts(k) = start_idxs[k];

            vector<int>         spans(mfa_data.dom_dim);
            vector<MatrixX<T>>  B(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
                B[k].resize(1, t.nctrl_pts(k));
            VolIterator dom_vol_iter(ndom_pts, dom_starts, mfa.ndom_pts());                 // iterator over input points
            while (!dom_vol_iter.done())
            {
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                {
                    int p   = mfa_data.p(k);
                    spans[k] = mfa_data.FindSpan(k, mfa.params()[k][dom_vol_iter.idx_dim(k)]);

                    // basis functions
                    vector<T> loc_knots(p + 2);
                    B[k].row(0).setZero();
                    T u = mfa.params()[k][dom_vol_iter.idx_dim(k)];                         // parameter of current input point

                    for (auto j = 0; j < p + 1; j++)
                    {
                        for (auto i = 0; i < p + 2; i++)
                            loc_knots[i] = mfa_data.tmesh.all_knots[k][spans[k] - p + j + i];
                        int col = spans[k] - p + j - t.knot_mins[k];
                        if (col >= 0 && col < B[k].cols())
                            B[k](0, col) = mfa_data.OneBasisFun(k, u, loc_knots);
                    }

                    // debug
//                     cerr << "B[" << k <<"]: " << B[k] << endl;
                }

                VolIterator ctrl_vol_iter(t.nctrl_pts);                                     // iterator over control points
                while (!ctrl_vol_iter.done())
                {
                    VectorXi ijk(mfa_data.dom_dim);                                         // ijk of current control point
                    ctrl_vol_iter.idx_ijk(ctrl_vol_iter.cur_iter(), ijk);

                    // check if current control point is covered by the basis funcs. in all dims
                    bool local_support = true;
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                    {
                        if (ijk(k) < spans[k] - mfa_data.p(k) - t.knot_mins[k] || ijk(k) > spans[k] - t.knot_mins[k])
                        {
                            local_support = false;
                            break;
                        }
                    }

                    if (local_support)
                    {
                        for (auto k = 0; k < mfa_data.dom_dim; k++)
                        {
                            for (auto j = 0; j < mfa_data.p(k) + 1; j++)                    // nonzero basis function values at this parameter, in this dim.
                            {
                                int col = spans[k] - mfa_data.p(k) + j - t.knot_mins[k];
                                if (ijk(k) == col)
                                {
                                    // debug
//                                     cerr << "ctrl_cur_iter = " << ctrl_vol_iter.cur_iter() << " k = " << k << " ijk(k) = col = " << col << endl;

                                    if (N(dom_vol_iter.cur_iter(), ctrl_vol_iter.cur_iter()) == -1.0)   // unassigned so far
                                        N(dom_vol_iter.cur_iter(), ctrl_vol_iter.cur_iter()) = B[k](0, col);
                                    else
                                        N(dom_vol_iter.cur_iter(), ctrl_vol_iter.cur_iter()) *= B[k](0, col);
                                }
                            }
                        }
                    }

                    // debug
//                     cerr << "N.row(" << dom_vol_iter.cur_iter() << "): " << N.row(dom_vol_iter.cur_iter()) << endl;

                    ctrl_vol_iter.incr_iter();
                }
                dom_vol_iter.incr_iter();
            }

            // set any unassigned values in N to 0
            for (auto i = 0; i < N.rows(); i++)
                for (auto j = 0; j < N.cols(); j++)
                    if (N(i, j) == -1.0)
                        N(i, j) = 0.0;

            // debug
//             cerr << "N:\n" << mfa_data.N[0] << endl;

            // normal form
            MatrixX<T> NtN = N.transpose() * N;

            // pad NtN on the right and bottom for the constraints
            if (constrained)
            {
                int pad_rows = NtN.rows() + cons.rows();
                int pad_cols = NtN.cols() + cons.rows();
                NtN.conservativeResize(pad_rows, pad_cols);
                // 1's along diagonal of NtN for constraints, rest of padding is 0
                NtN.block(0, t.ctrl_pts.rows(), NtN.rows(), cons.rows()) =
                    MatrixX<T>::Zero(NtN.rows(), cons.rows());
                NtN.block(t.ctrl_pts.rows(), 0, cons.rows(), t.ctrl_pts.rows()) =
                    MatrixX<T>::Zero(cons.rows(), t.ctrl_pts.rows());
                for (auto i = 0; i < cons.rows(); i++)
                    NtN(t.ctrl_pts.rows() + i, t.ctrl_pts.rows() + i) = 1.0;
            }

            // debug
//             cerr << "NtN:\n" << NtN << endl;

            // R is the right hand side needed for solving NtN * P = R
            MatrixX<T> R(NtN.cols(), pt_dim);
            RHSTensor(N, start_idxs, end_idxs, t, cons, R);

            // P is the solution vector
            MatrixX<T> P(NtN.cols(), pt_dim);
            P = NtN.ldlt().solve(R);
            t.ctrl_pts = P.block(0, 0, t.ctrl_pts.rows(), pt_dim);

            // debug
//             cerr << "\nEncodeTensor() P:\n" << P << endl;
//             cerr << "\nEncodeTensor() ctrl_pts:\n" << t.ctrl_pts << endl;
//             cerr << "\nEncodeTensor() weights:\n" << t.weights << endl;
        }

#ifdef TMESH

        // encodes the control points for one tensor product of a tmesh
        // takes a subset of input points from the global domain, covered by basis functions of this tensor product
        // solves all dimensions together (not separably)
        // does not encode weights for now
        // latest linear constrained formulation as proposed by David Lenz (see wiki/notes/linear-constrained-fit.pdf)
        void EncodeTensorLocalLinear(
                TensorIdx                 t_idx,                  // index of tensor product being encoded
                bool                      weighted = true)        // solve for and use weights
        {
            // debug
            fprintf(stderr, "EncodeTensorLocalLinear\n");

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];                           // current tensor product
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                               // control point dimensionality

            // get input domain points covered by the tensor
            vector<size_t> start_idxs(mfa_data.dom_dim);
            vector<size_t> end_idxs(mfa_data.dom_dim);
            mfa_data.tmesh.domain_pts(t_idx, false, mfa.params(), start_idxs, end_idxs);

            // Q matrix of relevant input domain points
            VectorXi ndom_pts(mfa_data.dom_dim);
            VectorXi dom_starts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];                                                // need Eigen vector from STL vector
            }
            MatrixX<T> Q(ndom_pts.prod(), pt_dim);
            VolIterator dom_iter(ndom_pts, dom_starts, mfa.ndom_pts());
            while (!dom_iter.done())
            {
                Q.block(dom_iter.cur_iter(), 0, 1, pt_dim) =
                    domain.block(dom_iter.sub_full_idx(dom_iter.cur_iter()), mfa_data.min_dim, 1, pt_dim);
                dom_iter.incr_iter();
            }

            // resize control poiunts and weights in case number of control points changed
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());
            t.weights = VectorX<T>::Ones(t.weights.size());                                     // linear solve does not solve for weights; set to 1

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(mfa_data.dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(mfa_data.dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

            vector<KnotIdx>     anchor(mfa_data.dom_dim);                                       // control point anchor

            // iterator over free control points
            MatrixX<T> Nfree = MatrixX<T>::Constant(ndom_pts.prod(), t.ctrl_pts.rows(), -1);    // basis functions, -1 means unassigned so far
            VolIterator free_iter(t.nctrl_pts);
            while (!free_iter.done())
            {
                VectorXi ijk(mfa_data.dom_dim);                                                 // ijk of current control point
                free_iter.idx_ijk(free_iter.cur_iter(), ijk);

                // anchor of control point
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    anchor[k] = free_iter.idx_dim(k) + mfa_data.tmesh.tensor_prods[t_idx].knot_mins[k];

                // tensor containing anchor
                TensorIdx t_idx_anchor;
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                {
                    t_idx_anchor = mfa_data.tmesh.in_prev_next(anchor, t_idx, k, true);
                    if (t_idx_anchor < mfa_data.tmesh.tensor_prods.size())
                        break;
                }
                assert(t_idx_anchor < mfa_data.tmesh.tensor_prods.size());

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchor, t_idx_anchor, true, local_knot_idxs);
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, mfa.ndom_pts());
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < mfa_data.dom_dim; k++)                                 // for all dims
                    {
                        int p = mfa_data.p(k);                                                  // degree of current dim.
                        T u = mfa.params()[k][dom_iter.idx_dim(k)];                             // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        if (Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) == -1.0)           // unassigned so far
                            Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) = B;
                        else
                            Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) *= B;
                    }           // for all dims
                    dom_iter.incr_iter();
                }           // domain point iterator
                free_iter.incr_iter();
            }           // free control point iterator

            // set any unassigned values in Nfree to 0
            for (auto i = 0; i < Nfree.rows(); i++)
            {
                for (auto j = 0; j < Nfree.cols(); j++)
                    if (Nfree(i, j) == -1.0)
                        Nfree(i, j) = 0.0;
            }

            // find constraint control points and their anchors
            MatrixX<T>              Pcons;                                                      // constraint control points
            vector<vector<KnotIdx>> anchors;                                                    // corresponding anchors
            LocalSolveConstraints(t, Pcons, anchors);

            // iterator over constraint control points
            MatrixX<T> Ncons = MatrixX<T>::Constant(ndom_pts.prod(), Pcons.rows(), -1);         // basis functions, -1 means unassigned so far
            for (auto i = 0; i < Pcons.rows(); i++)                                             // for all constraint control points
            {
                // tensor containing anchor
                TensorIdx t_idx_anchor;
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                {
                    t_idx_anchor = mfa_data.tmesh.in_prev_next(anchors[i], t_idx, k, true);
                    if (t_idx_anchor < mfa_data.tmesh.tensor_prods.size())
                        break;
                }
                assert(t_idx_anchor < mfa_data.tmesh.tensor_prods.size());

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchors[i], t_idx_anchor, true, local_knot_idxs);
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, mfa.ndom_pts());
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < mfa_data.dom_dim; k++)                                 // for all dims
                    {
                        int p = mfa_data.p(k);                                                  // degree of current dim.
                        T u = mfa.params()[k][dom_iter.idx_dim(k)];                             // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        if (Ncons(dom_iter.cur_iter(), i) == -1.0)                              // unassigned so far
                            Ncons(dom_iter.cur_iter(), i) = B;
                        else
                            Ncons(dom_iter.cur_iter(), i) *= B;
                    }           // for all dims
                    dom_iter.incr_iter();
                }           // domain points iterator
            }           // for all constraint control points

            // set any unassigned values in Ncons to 0
            for (auto i = 0; i < Ncons.rows(); i++)
            {
                for (auto j = 0; j < Ncons.cols(); j++)
                    if (Ncons(i, j) == -1.0)
                        Ncons(i, j) = 0.0;
            }

            // R is the right hand side needed for solving N * P = R
            MatrixX<T>  R = Q - Ncons* Pcons;

            // debug
//             cerr << "input:\n"  << domain   << endl;
//             cerr << "Nfree:\n"  << Nfree    << endl;
//             cerr << "Ncons:\n"  << Ncons    << endl;
//             cerr << "Pcons:\n"  << Pcons    << endl;
//             cerr << "Q:\n"      << Q        << endl;
//             cerr << "R:\n"      << R        << endl;

            // sanity check that the row sums of Nfree + Ncons should be ~ 1
            for (auto i = 0; i < Nfree.rows(); i++)
            {
                double eps = 1.0e-1;                // not sure how close is close enough
                if (fabs(Nfree.row(i).sum() + Ncons.row(i).sum() - 1.0) > eps)
                {
                    fprintf(stderr, "\nWarning: Nfree + Ncons row(%d) sum = %.4lf which should be ~ 1.0\n", i, Nfree.row(i).sum() + Ncons.row(i).sum());
                    fprintf(stderr, "This could indicate that the constraints (control point values or anchors) or input points covered by the constraints are wrong.\n");
                }

//                 assert(fabs(Nfree.row(i).sum() + Ncons.row(i).sum() - 1.0) < eps);
            }

            // solve
            t.ctrl_pts = Nfree.colPivHouseholderQr().solve(R);

            // debug
//             cerr << "\EncodeTensorLocalLinear() ctrl_pts:\n" << t.ctrl_pts << endl;
//             cerr << "\nEncodeTensorLocalLinear() weights:\n" << t.weights << endl;
        }

#endif

        // original adaptive encoding for first tensor product only
        void OrigAdaptiveEncode(
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)         // optional maximum number of rounds
        {
            vector<vector<T>> new_knots;                               // new knots in each dim.

            // TODO: use weights for knot insertion
            // for now, weights are only used for final full encode

            // loop until no change in knots
            for (int iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                    break;

                if (verbose)
                    fprintf(stderr, "Iteration %d...\n", iter);

                // low-d w/ splitting spans in the middle
                bool done = OrigNewKnots_curve(new_knots, err_limit, extents, iter);

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

            // temporary control points and weights for global encode or first round of local encode
            VectorXi nctrl_pts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
                nctrl_pts(k) = mfa_data.tmesh.all_knots[k].size() - mfa_data.p(k) - 1;
            MatrixX<T> ctrl_pts(nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
            VectorX<T> weights(ctrl_pts.rows());

            // Initial global encode and scattering of control points to tensors
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
                bool temp_local = local;                        // ability to do local solve temporarily for this round depends on whether new knots permits local solve
                int retval = NewKnots_full(err_limit, extents, iter, temp_local);

                // if not doing local solve,
                // resize temporary control points and weights and global encode and scatter of control points to tensors
                if (!local || !temp_local)
                {
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                        nctrl_pts(k) = mfa_data.tmesh.all_knots[k].size() - mfa_data.p(k) - 1;
                    ctrl_pts.resize(nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
                    weights.resize(ctrl_pts.rows());

                    Encode(nctrl_pts, ctrl_pts, weights);
                    mfa_data.tmesh.scatter_ctrl_pts(nctrl_pts, ctrl_pts, weights);
                }

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
//             cerr << "Rk:\n" << Rk << endl;
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

        // computes right hand side vector for encoding one tensor product of control points
        // takes subset of original input points
        // solves all dimensions at once (not seperable dimensions)
        // allows for additional constraints
        // no weights as yet
        // R is column vector of n elements, each element multiple coordinates of the input points
        void RHSTensor(
                const MatrixX<T>&       N,          // matrix of basis function coefficients
                const vector<size_t>&   start_idxs, // starting indices of subset of domain points
                const vector<size_t>&   end_idxs,   // ending indices of subset of domain points
                const TensorProduct<T>& t,          // current tensor product
                const MatrixX<T>&       cons,       // constraints
                MatrixX<T>&             R)          // (output) residual matrix allocated by caller
        {
            VectorXi ndom_pts(mfa_data.dom_dim);
            VectorXi dom_starts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];
            }

            // fill Rk, the matrix of input points
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);           // one row for each input point
            VolIterator vol_iter(ndom_pts, dom_starts, mfa.ndom_pts());                 // iterator over input points
            VectorXi ijk(mfa_data.dom_dim);
            while (!vol_iter.done())
            {
                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);
                Rk.row(vol_iter.cur_iter()) = domain.block(vol_iter.ijk_idx(ijk), mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
                vol_iter.incr_iter();
            }

            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                 // ith basis functions for input pts
                         Rk.col(j).array()).sum();          // input points

            // append the constraints to R
            for (auto i = 0; i < R.rows() - N.cols(); i++)
                R.row(N.cols() + i) = cons.row(i);

            // debug
//             cerr << "Rk:\n" << Rk << endl;
//             cerr << "\nR:\n" << R << endl;
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

        // this is the version used currently for tmesh global or local solve
        // encodes at full dimensionality and decodes at full dimensionality
        // decodes full-d points in each knot span and adds new knot spans where error > err_limit
        // returns 1 if knots were added, 0 if no knots were added, -1 if number of control points >= input points
        int NewKnots_full(
                T                   err_limit,                  // max allowable error
                const VectorX<T>&   extents,                    // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 iter,                       // iteration number of caller (for debugging)
                bool&               local)                      // (input, output) solve locally (with constraints)
        {
            // debug
            if (local)
                fprintf(stderr, "*** Using local solve in NewKnots_full ***\n");
            else
                fprintf(stderr, "*** Using global solve in NewKnots_full ***\n");

            bool done = true;

            // indices in tensor, in each dim. of inserted knots in full knot vector after insertion
            vector<vector<KnotIdx>> inserted_knot_idxs(mfa_data.dom_dim);

            VectorX<T> myextents = extents.size() ? extents : VectorX<T>::Ones(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());

            int parent_tensor_idx;                              // idx of parent tensor of new knot (assuming only one new knot)

            // find new knots
            mfa::NewKnots<T> nk(mfa, mfa_data);

            // vectors of new_nctrl_pts, new_ctrl_pts, new_weights, one instance for each knot to be inserted
            // we're only inserting one knot at a time, but the NewKnots object supports multiple knot insertions, hence std::vector
            vector<VectorXi>    new_nctrl_pts;
            vector<MatrixX<T>>  new_ctrl_pts;
            vector<VectorX<T>>  new_weights;

            if (local)
//                 done &= nk.FirstErrorSpan(domain,
                done &=   nk.MaxErrorSpan(domain,
                                          myextents,
                                          err_limit,
                                          iter,
                                          parent_tensor_idx,
                                          inserted_knot_idxs,
                                          new_nctrl_pts,
                                          new_ctrl_pts,
                                          new_weights,
                                          local);
            else
//                 done &= nk.FirstErrorSpan(domain,
                done &=   nk.MaxErrorSpan(domain,
                                          myextents,
                                          err_limit,
                                          iter,
                                          parent_tensor_idx,
                                          inserted_knot_idxs);

            if (local)
                assert(inserted_knot_idxs[0].size() == new_ctrl_pts.size());    // sanity check: number of inserted knots is consistent across things that depend on it

            if (done)                                                           // nothing inserted
                return 0;

            // knot mins and maxs of tensor to be appended
            // this is where we decide how many knots and control points the added tensor has
            vector<KnotIdx> knot_mins(mfa_data.dom_dim);
            vector<KnotIdx> knot_maxs(mfa_data.dom_dim);
            for (auto j = 0; j < mfa_data.dom_dim; j++)
            {
                // following makes p control points in the added tensor
                knot_mins[j] = inserted_knot_idxs[j][0] - mfa_data.p(j) / 2;
                knot_maxs[j] = inserted_knot_idxs[j][0] + mfa_data.p(j) / 2;

                // debug: try making a bigger tensor with p + 1 control points
                knot_maxs[j]++;             // correct for both even and odd degree

                // check that we don't leave the parent tensor with less than p control points anywhere
                TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[parent_tensor_idx];
                int odd_degree = mfa_data.p(j) % 2 == 0 ? 0 : 1;
                if (t.knot_maxs[j] - knot_maxs[j] < mfa_data.p(j) - odd_degree)
                    knot_maxs[j] = t.knot_maxs[j];
                if (knot_mins[j] - t.knot_mins[j] < mfa_data.p(j) - odd_degree)
                    knot_mins[j] = t.knot_mins[j];
            }

            // debug
            if (mfa_data.dom_dim == 1)
                fprintf(stderr, "inserting knot with idxs [%ld] and appending tensor with knot_mins [%ld] knot_maxs [%ld]\n",
                        inserted_knot_idxs[0][0], knot_mins[0], knot_maxs[0]);
            else if (mfa_data.dom_dim == 2)
                fprintf(stderr, "inserting knot with idxs [%ld %ld] and appending tensor with knot_mins [%ld %ld] knot_maxs [%ld %ld]\n",
                        inserted_knot_idxs[0][0], inserted_knot_idxs[1][0], knot_mins[0], knot_mins[1], knot_maxs[0], knot_maxs[1]);
            else if (mfa_data.dom_dim == 3)
                fprintf(stderr, "inserting knot with idxs [%ld %ld %ld] and appending tensor with knot_mins [%ld %ld %ld] knot_maxs [%ld %ld %ld]\n",
                        inserted_knot_idxs[0][0], inserted_knot_idxs[1][0], inserted_knot_idxs[2][0], knot_mins[0], knot_mins[1], knot_mins[2], knot_maxs[0], knot_maxs[1], knot_maxs[2]);

            // append the tensor
            // only doing one new knot insertion, hence the [0] index on new_nctrl_pts, new_ctrl_pts, new_weights
            if (local)
                mfa_data.tmesh.append_tensor(knot_mins,
                                             knot_maxs,
                                             new_nctrl_pts[0],
                                             new_ctrl_pts[0],
                                             new_weights[0],
                                             parent_tensor_idx);
            else
                mfa_data.tmesh.append_tensor(knot_mins,
                                             knot_maxs);

            // debug
//             mfa_data.tmesh.print();

            // local solve newly appended tensor
            // TODO: experimenting with the difference between iterative and linear least squares
            if (local)
#ifdef MFA_LINEAR_LOCAL

//                 EncodeTensor(mfa_data.tmesh.tensor_prods.size() - 1, true, true);
                EncodeTensorLocalLinear(mfa_data.tmesh.tensor_prods.size() - 1);

#else

                LocalSolve();

#endif

            for (auto k = 0; k < mfa_data.dom_dim; k++)
                if (mfa.ndom_pts()(k) <= mfa_data.tmesh.all_knots[k].size() - (mfa_data.p(k) + 1))
                    return -1;

            return 1;
        }

        // set up and run iterative solver for constrained local solve
        // of a previously added tensor to the back of the tensor products in the tmesh
        // n-d version
        void LocalSolve()
        {
            const Tmesh<T>&         tmesh   = mfa_data.tmesh;
            const TensorProduct<T>& tc      = tmesh.tensor_prods.back();                                    // current (newly appended) tensor
            int                     cols    = tc.ctrl_pts.cols();

            // fill control points to solve (both interior and constraints)
            MatrixX<T> ctrlpts_tosolve;

            // control points to solve are only free control points
            // constraints come from decoding input points that are covered by the constrained control points
            ctrlpts_tosolve = tc.ctrl_pts;

            // debug
//             cerr << "ctrlpts_tosolve:\n" << ctrlpts_tosolve << endl;

            // set the constraints
            MatrixX<T> cons = ctrlpts_tosolve.block(tc.ctrl_pts.rows(), 0, ctrlpts_tosolve.rows() - tc.ctrl_pts.rows(), cols);              // set the constraints

            // debug
//             cerr << "cons:\n" << cons << endl;

            // get the subset of the domain points needed for the local solve
            vector<size_t> start_idxs(mfa_data.dom_dim);
            vector<size_t> end_idxs(mfa_data.dom_dim);
            tmesh.domain_pts(tmesh.tensor_prods.size() - 1, true, mfa.params(), start_idxs, end_idxs);        // true = pad by degree on each side

            // debug: decode all domain points
//             fprintf(stderr, "Debug: overriding domain points to include all: ");
//             for (auto i = 0; i < mfa_data.dom_dim; i++){
//                 start_idxs[i] = 0;
//                 end_idxs[i] = mfa.ndom_pts()(i) - 1;
//                 fprintf(stderr, "start_idxs[%d] = %lu end_idxs[%d] = %lu\n", i, start_idxs[i], i, end_idxs[i]);
//             }
//             fprintf(stderr, "\n");

            // set up the optimization
            LocalLSQ<T> llsq(mfa, mfa_data, domain, cons, start_idxs, end_idxs, verbose);
            // trying various different solvers
            BfgsSolver<LocalLSQ<T>> solver;
//             LbfgsSolver<LocalLSQ<T>> solver;
//             NewtonDescentSolver<LocalLSQ<T>> solver;
//             CMAesSolver<LocalLSQ<T>> solver;         // does not compile
//             NelderMeadSolver<LocalLSQ<T>> solver;
//             ConjugatedGradientDescentSolver<LocalLSQ<T>> solver;
//             GradientDescentSolver<LocalLSQ<T>> solver;

            // minimize the function
            VectorX<T> x1(Eigen::Map<VectorX<T>>(ctrlpts_tosolve.data(), ctrlpts_tosolve.size()));  // size() = rows() * cols()
            fprintf(stderr, "\nIterative solver optimizing control points...\n");
            solver.minimize(llsq, x1);

            // debug
            fprintf(stderr, "\nSolver converged in %lu iterations.\n", llsq.iters());
        }

        // constraint control points and corresponding anchors for local solve
        void LocalSolveConstraints(
                const TensorProduct<T>&     tc,                 // current tensor product being solved
                MatrixX<T>&                 ctrl_pts,           // (output) constraing control points
                vector<vector<KnotIdx>>&    anchors)            // (output) corresponding anchors
        {
            const Tmesh<T>&         tmesh   = mfa_data.tmesh;
            int                     cols    = tc.ctrl_pts.cols();
            KnotIdx                 min, max;                   // temporaries

            // get required sizes

            int rows = 0;                                       // number of rows required in ctrl_pts
            VectorXi npts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                for (auto j = 0; j < tc.prev[k].size(); j++)    // previous tensors
                {
                    const TensorProduct<T>& tp = tmesh.tensor_prods[tc.prev[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of prev
                            npts(i) = p;
                        else                                    // direction orthogonal to prev
                        {
                            mfa_data.tmesh.knot_idx_ofst(tp, tc.knot_mins[i], -p, i, min);
                            mfa_data.tmesh.knot_idx_ofst(tp, tc.knot_maxs[i], p, i, max);
                            if (p % 2)                          // odd degree
                                npts(i) = mfa_data.tmesh.knot_idx_dist(tp, min, max, i, true);
                            else                                // even degree
                                npts(i) = mfa_data.tmesh.knot_idx_dist(tp, min, max, i, false);
                        }
                    }
                    rows += npts.prod();
                }   // previous tensors
                for (auto j = 0; j < tc.next[k].size(); j++)    // next tensors
                {
                    const TensorProduct<T>& tn = tmesh.tensor_prods[tc.next[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of next
                            npts(i) = p;
                        else                                    // direction orthogonal to next
                        {
                            mfa_data.tmesh.knot_idx_ofst(tn, tc.knot_mins[i], -p, i, min);
                            mfa_data.tmesh.knot_idx_ofst(tn, tc.knot_maxs[i], p, i, max);
                            if (p % 2)                          // odd degree
                                npts(i) = mfa_data.tmesh.knot_idx_dist(tn, min, max, i, true);
                            else                                // even degree
                                npts(i) = mfa_data.tmesh.knot_idx_dist(tn, min, max, i, false);
                        }
                    }
                    rows += npts.prod();
                }
            }   // next tensors
            ctrl_pts.resize(rows, cols);
            anchors.resize(rows);

            // get control points and anchors

            int cur_row = 0;
            VectorXi sub_starts(mfa_data.dom_dim);
            VectorXi sub_npts(mfa_data.dom_dim);
            VectorXi all_npts(mfa_data.dom_dim);
            vector<KnotIdx> anchor(mfa_data.dom_dim);           // one anchor
            for (auto k = 0; k < mfa_data.dom_dim; k++)         // for all dimensions
            {
                // previous tensors
                // assumes that all previous tensors have at least the required number of control points (p) in the prev direction
                for (auto j = 0; j < tc.prev[k].size(); j++)
                {
                    const TensorProduct<T>& tp = tmesh.tensor_prods[tc.prev[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of prev
                        {
                            sub_starts(i)   = tp.nctrl_pts(i) - p;
                            if (p % 2)                          // odd degree, skip border point
                                sub_starts(i)--;
                            sub_npts(i)     = p;
                        }
                        else                                    // direction orthogonal to prev
                        {
                            mfa_data.tmesh.knot_idx_ofst(tp, tc.knot_mins[i], -p, i, min);
                            mfa_data.tmesh.knot_idx_ofst(tp, tc.knot_maxs[i], p, i, max);

                            sub_starts(i) = mfa_data.tmesh.knot_idx_dist(tp, tp.knot_mins[i], min, i, false);
                            if (tp.knot_mins[i] == 0)
                                sub_starts(i) -= (p + 1) / 2;
                            if (p % 2)                          // odd degree
                                sub_npts(i) = mfa_data.tmesh.knot_idx_dist(tp, min, max, i, true);
                            else                                // even degree
                                sub_npts(i) = mfa_data.tmesh.knot_idx_dist(tp, min, max, i, false);
                        }
                        all_npts(i)         = tp.nctrl_pts(i);

                        // debug
//                         fprintf(stderr, "prev tensor: dim = %d sub_npts = %d sub_starts = %d all_npts = %d\n",
//                                 i, sub_npts(i), sub_starts(i), all_npts(i));
                    }
                    VolIterator voliter_prev(sub_npts, sub_starts, all_npts);
                    VectorXi ijk(mfa_data.dom_dim);
                    while (!voliter_prev.done())
                    {
                        // control point
                        ctrl_pts.row(cur_row) = tp.ctrl_pts.row(voliter_prev.sub_full_idx(voliter_prev.cur_iter()));

                        // anchor
                        anchors[cur_row].resize(mfa_data.dom_dim);
                        voliter_prev.idx_ijk(voliter_prev.cur_iter(), ijk);
                        mfa_data.tmesh.ctrl_pt_anchor(tp, ijk, anchor);
                        for (auto i = 0; i < mfa_data.dom_dim; i++)
                            anchors[cur_row][i] = anchor[i];
                        cur_row++;
                        voliter_prev.incr_iter();
                    }
                }   // previous tensors

                // next tensors
                // assumes that all next tensors have at least the required number of control points (p) in the next direction
                for (auto j = 0; j < tc.next[k].size(); j++)
                {
                    const TensorProduct<T>& tn = tmesh.tensor_prods[tc.next[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of next
                        {
                            if (p % 2)                          // odd degree, skip border point
                                sub_starts(i)   = 1;
                            else
                                sub_starts(i)   = 0;
                            sub_npts(i)         = p;
                        }
                        else                                    // direction orthogonal to next
                        {
                            mfa_data.tmesh.knot_idx_ofst(tn, tc.knot_mins[i], -p, i, min);
                            mfa_data.tmesh.knot_idx_ofst(tn, tc.knot_maxs[i], p, i, max);

                            sub_starts(i) = mfa_data.tmesh.knot_idx_dist(tn, tn.knot_mins[i], min, i, false);
                            if (tn.knot_mins[i] == 0)
                                sub_starts(i) -= (p + 1) / 2;
                            if (p % 2)                          // odd degree
                                sub_npts(i) = mfa_data.tmesh.knot_idx_dist(tn, min, max, i, true);
                            else                                // even degree
                                sub_npts(i) = mfa_data.tmesh.knot_idx_dist(tn, min, max, i, false);
                        }
                        all_npts(i)         = tn.nctrl_pts(i);

                        // debug
//                         fprintf(stderr, "next tensor: dim = %d sub_npts = %d sub_starts = %d all_npts = %d\n",
//                                 i, sub_npts(i), sub_starts(i), all_npts(i));
                    }
                    VolIterator voliter_next(sub_npts, sub_starts, all_npts);
                    VectorXi ijk(mfa_data.dom_dim);
                    while (!voliter_next.done())
                    {
                        // control point
                        ctrl_pts.row(cur_row) = tn.ctrl_pts.row(voliter_next.sub_full_idx(voliter_next.cur_iter()));

                        // anchor
                        anchors[cur_row].resize(mfa_data.dom_dim);
                        voliter_next.idx_ijk(voliter_next.cur_iter(), ijk);
                        mfa_data.tmesh.ctrl_pt_anchor(tn, ijk, anchor);
                        for (auto i = 0; i < mfa_data.dom_dim; i++)
                            anchors[cur_row][i] = anchor[i];
                        cur_row++;
                        voliter_next.incr_iter();
                    }
                }   // next tensors
            }   // for all dimensions
        }

        // set up control points to solve for local solver
        void LocalSolveCtrlPts(const TensorProduct<T>&  tc,                 // current tensor product being solved
                               MatrixX<T>&              ctrlpts_tosolve)    // (output) control points to solv
        {
            const Tmesh<T>&         tmesh   = mfa_data.tmesh;
            int                     cols    = tc.ctrl_pts.cols();
            KnotIdx                 min, max;                   // temporaries

            // get required sizes

            int rows = tc.ctrl_pts.rows();                      // number of rows required in ctrlpts_tosolve
            VectorXi npts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                for (auto j = 0; j < tc.prev[k].size(); j++)
                {
                    const TensorProduct<T>& tp = tmesh.tensor_prods[tc.prev[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of prev
                            npts(i) = p;
                        else                                    // direction orthogonal to prev
                        {
                            if (tp.knot_mins[i] <= tc.knot_mins[i] - p)
                                min = tc.knot_mins[i] - p;
                            else
                                min = tp.knot_mins[i];
                            if (tp.knot_maxs[i] >= tc.knot_maxs[i] + p)
                                max = tc.knot_maxs[i] + p;
                            else
                                max = tp.knot_maxs[i];
                            npts(i) = p % 2 ? max - min : max - min - 1;
                        }
                    }
                    rows += npts.prod();
                }
                for (auto j = 0; j < tc.next[k].size(); j++)
                {
                    const TensorProduct<T>& tn = tmesh.tensor_prods[tc.next[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of next
                            npts(i) = p;
                        else                                    // direction orthogonal to next
                        {
                            if (tn.knot_mins[i] <= tc.knot_mins[i] - p)
                                min = tc.knot_mins[i] - p;
                            else
                                min = tn.knot_mins[i];
                            if (tn.knot_maxs[i] >= tc.knot_maxs[i] + p)
                                max = tc.knot_maxs[i] + p;
                            else
                                max = tn.knot_maxs[i];
                            npts(i) = p % 2 ? max - min : max - min - 1;
                        }
                    }
                    rows += npts.prod();
                }
            }
            ctrlpts_tosolve.resize(rows, cols);

            // ctrlpts_tosolve has the new control points being solved at the front
            // followed by the constraint control points that are intended to be held constant

            // copy original control points into the front of ctrlpts_tosolve
            ctrlpts_tosolve.block(0, 0, tc.ctrl_pts.rows(), cols) = tc.ctrl_pts;

            // copy constraints into ctrlpts_tosolve after the original control points

            int cur_row = tc.ctrl_pts.rows();
            VectorXi sub_starts(mfa_data.dom_dim);
            VectorXi sub_npts(mfa_data.dom_dim);
            VectorXi all_npts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                // previous tensors
                // assumes that all previous tensors have at least the required number of control points (p)
                for (auto j = 0; j < tc.prev[k].size(); j++)
                {
                    const TensorProduct<T>& tp = tmesh.tensor_prods[tc.prev[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of prev
                        {
                            sub_starts(i)   = tp.nctrl_pts(i) - p;
                            if (p % 2)                          // odd degree, skip border point
                                sub_starts(i)--;
                            sub_npts(i)     = p;
                        }
                        else                                    // direction orthogonal to prev
                        {
                            if (tp.knot_mins[i] <= tc.knot_mins[i] - p)
                                min = tc.knot_mins[i] - p;
                            else
                                min = tp.knot_mins[i];
                            if (tp.knot_maxs[i] >= tc.knot_maxs[i] + p)
                                max = tc.knot_maxs[i] + p;
                            else
                                max = tp.knot_maxs[i];
                            if (tp.knot_mins[i] == 0)
                                sub_starts(i)   = min - (p + 1) / 2;
                            else
                                sub_starts(i)   = min - tp.knot_mins[i];
                            sub_npts(i)     = p % 2 ? max - min : max - min - 1;
                        }
                        all_npts(i)         = tp.nctrl_pts(i);
                    }
                    VolIterator voliter_prev(sub_npts, sub_starts, all_npts);
                    while (!voliter_prev.done())
                    {
                        ctrlpts_tosolve.row(cur_row) = tp.ctrl_pts.row(voliter_prev.sub_full_idx(voliter_prev.cur_iter()));
                        cur_row++;
                        voliter_prev.incr_iter();
                    }
                }

                // next tensors
                // assumes that all next tensors have at least the required number of control points (p)
                for (auto j = 0; j < tc.next[k].size(); j++)
                {
                    const TensorProduct<T>& tn = tmesh.tensor_prods[tc.next[k][j]];
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        if (i == k)                             // direction of next
                        {
                            if (p % 2)                          // odd degree, skip border point
                                sub_starts(i)   = 1;
                            else
                                sub_starts(i)   = 0;
                            sub_npts(i)         = p;
                        }
                        else                                    // direction orthogonal to next
                        {
                            if (tn.knot_mins[i] <= tc.knot_mins[i] - p)
                                min = tc.knot_mins[i] - p;
                            else
                                min = tn.knot_mins[i];
                            if (tn.knot_maxs[i] >= tc.knot_maxs[i] + p)
                                max = tc.knot_maxs[i] + p;
                            else
                                max = tn.knot_maxs[i];
                            if (tn.knot_mins[i] == 0)
                                sub_starts(i)   = min - (p + 1) / 2;
                            else
                                sub_starts(i)   = min - tn.knot_mins[i];
                            sub_npts(i)     = p % 2 ? max - min : max - min - 1;
                        }
                        all_npts(i)         = tn.nctrl_pts(i);
                    }
                    VolIterator voliter_next(sub_npts, sub_starts, all_npts);
                    while (!voliter_next.done())
                    {
                        ctrlpts_tosolve.row(cur_row) = tn.ctrl_pts.row(voliter_next.sub_full_idx(voliter_next.cur_iter()));
                        cur_row++;
                        voliter_next.incr_iter();
                    }
                }
            }
        }

#endif      // TMESH

        // 1d encoding and 1d decoding
        // adds knots error spans from all curves in all directions (into a set)
        // adds knots in middles of spans that have error higher than the limit
        // returns true if done, ie, no knots are inserted
        bool OrigNewKnots_curve(
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
                nk.OrigInsertKnots(new_knots, new_levels, unused);

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
    };          // Encoder class
}           // mfa namespace

#endif
