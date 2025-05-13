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

#include    <set>
#include    <algorithm>

// temporary utilities for testing
#include    <ctime>
#include    <chrono>
#include    <mpi.h>     // for MPI_Wtime() only

#ifndef      MFA_NO_WEIGHTS
#include    "coin/ClpSimplex.hpp"
#include    "coin/ClpInterior.hpp"
#endif

template <typename T>                                       // float or double
class NewKnots;

template <typename T>                                       // float or double
struct MFA;

namespace mfa
{
    template <typename T>                                   // float or double
    class Encoder
    {
        enum class ConsType
        {
            MFA_NO_CONSTRAINT,
            MFA_LEFT_ONLY_CONSTRAINT,
            MFA_RIGHT_ONLY_CONSTRAINT,
            MFA_BOTH_CONSTRAINT
        };

    private:

        template <typename>
        friend class NewKnots;

        MFA_Data<T>&        mfa_data;                       // the mfa data model
        int                 dom_dim;                        // domain dimension of mfa_data
        int                 verbose;                        // output level
        const PointSet<T>&  input;                          // input points
        size_t              max_num_curves;                 // max num. curves per dimension to check in curve version

    public:

        Encoder(MFA_Data<T>&        mfa_data_,              // MFA data model
                const PointSet<T>&  input_,                 // input points
                int                 verbose_) :             // debug level
            mfa_data(mfa_data_),
            dom_dim(mfa_data.dom_dim),
            verbose(verbose_),
            input(input_),
            max_num_curves(1.0e5)                           // max num. curves to check in one dimension of curve version
        { }

        ~Encoder() {}

        void Encode(const VectorXi& nctrl_pts,              // number of control points in each dim.
            MatrixX<T>&     ctrl_pts,                       // (output) control points
            VectorX<T>&     weights,                        // (output) weights
            bool            weighted = true)                // solve for and use weights
        {
            if (weighted)
            {
                encodeNURBS(nctrl_pts, ctrl_pts, weights, true);
            }
            else
            {
                encodeBSpline(nctrl_pts, ctrl_pts);
            }
        }

        // approximate a NURBS hypervolume of arbitrary dimension for a given input data set
        // n-d version of algorithm 9.7, Piegl & Tiller (P&T) p. 422
        // output control points can be specified by caller, does not have to be those in the tmesh
        // the output ctrl_pts are resized by this function;  caller need not resize them
        void encodeNURBS(const VectorXi& nctrl_pts,              // number of control points in each dim.
                    MatrixX<T>&     ctrl_pts,               // (output) control points
                    VectorX<T>&     weights,                // (output) weights
                    bool            weighted = true)        // solve for and use weights
        {
            // check quantities
            if (mfa_data.p.size() != input.ndom_pts().size())
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
                if (nctrl_pts(i) > input.ndom_pts(i))
                {
                    fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                            "exceeds number of input data points (%d) in dimension %ld.\n", nctrl_pts(i), i, input.ndom_pts(i), i);
                }
            }

            int      ndims  = input.ndom_pts().size();          // number of domain dimensions
            size_t   cs     = 1;                            // stride for input points in curve in cur. dim
            int      pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;// control point dimensonality
            // resize matrices in case number of control points changed
            ctrl_pts.resize(nctrl_pts.prod(), pt_dim);
            weights.resize(ctrl_pts.rows());

            // resize basis function matrices and initialize to 0; will only fill nonzeros later
            vector<MatrixX<T>> N(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
                N[k] = MatrixX<T>::Zero(input.ndom_pts(k), nctrl_pts(k));

            // 2 buffers of temporary control points
            // double buffer needed to write output curves of current dim without changing its input pts
            // temporary control points need to begin with size as many as the input domain points
            // except for the first dimension, which can be the correct number of control points
            // because the input domain points are converted to control points one dimension at a time
            // TODO: need to find a more space-efficient way
            size_t tot_ntemp_ctrl = 1;
            for (size_t k = 0; k < ndims; k++)
                tot_ntemp_ctrl *= (k == 0 ? nctrl_pts(k) : input.ndom_pts(k));
            MatrixX<T> temp_ctrl0 = MatrixX<T>::Zero(tot_ntemp_ctrl, pt_dim);
            MatrixX<T> temp_ctrl1 = MatrixX<T>::Zero(tot_ntemp_ctrl, pt_dim);

            VectorXi ntemp_ctrl = input.ndom_pts();     // current num of temp control pts in each dim

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
                        ncurves *= input.ndom_pts(i);
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

                // N is a matrix of m (# input points) x n (# control points) basis function coefficients
                //  _                              _
                // |  N_0(u[0])   ... N_n-1(u[0])   |
                // |     ...      ...      ...      |
                // |  N_0(u[m-1]) ... N_n-1(u[m-1]) |
                //  -                              -
                // TODO: N is going to be very sparse when it is large: switch to sparse representation
                // N has semibandwidth < p  nonzero entries across diagonal

                for (int i = 0; i < N[k].rows(); i++)
                {
                    int span = mfa_data.tmesh.FindSpan(k, input.params->param_grid[k][i], nctrl_pts(k));

#ifndef MFA_TMESH   // original version for one tensor product

                    mfa_data.OrigBasisFuns(k, input.params->param_grid[k][i], span, N[k], i);

#else               // tmesh version

                    mfa_data.BasisFuns(k, input.params->param_grid[k][i], span, N[k], i);

#endif
                }

                // debug
//                 fmt::print(stderr, "N[{}]:\n {}\n", k, N[k]);

                // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
                // NtN has semibandwidth < p + 1 nonzero entries across diagonal
                MatrixX<T> NtN  = N[k].transpose() * N[k];

                // debug
//                 cerr << "N[k]:\n" << N[k] << endl;
//                 cerr << "NtN:\n" << NtN << endl;

#ifdef MFA_TBB  // TBB version
                static affinity_partitioner ap;
                parallel_for (blocked_range<size_t>(0, ncurves), [&] (blocked_range<size_t>& r)
                {
                    for (auto j = r.begin(); j < r.end(); j++)        // for all the curves in this dimension
                    {
                        // debug
                        // fprintf(stderr, "j=%ld curve\n", j);

                        // R is the right hand side needed for solving NtN * P = R
                        MatrixX<T> R(N[k].cols(), pt_dim);
                        // P are the unknown control points and the solution to NtN * P = R
                        // NtN is positive definite -> do not need pivoting
                        // TODO: use a common representation for P and ctrl_pts to avoid copying
                        MatrixX<T> P(N[k].cols(), pt_dim);
                        // compute the one curve of control points
                        CtrlCurve(N[k], NtN, R, P, k, co[j], cs, to[j], temp_ctrl0, temp_ctrl1, -1, ctrl_pts, weights, weighted);
                    }   // curves in this dimension
                }, ap);

#endif              // end TBB version

#if defined(MFA_SERIAL) || defined(MFA_KOKKOS)
                // R is the right hand side needed for solving NtN * P = R
                MatrixX<T> R(N[k].cols(), pt_dim);

                // P are the unknown control points and the solution to NtN * P = R
                // NtN is positive definite -> do not need pivoting
                // TODO: use a common representation for P and ctrl_pts to avoid copying
                MatrixX<T> P(N[k].cols(), pt_dim);

                // encode curves in this dimension
                for (size_t j = 0; j < ncurves; j++)
                {
                    // print progress
                    if (verbose >= 2)
                    {
                        if (j > 0 && j > 100 && j % (ncurves / 100) == 0)
                            fprintf(stderr, "\r dimension %ld: %.0f %% encoded (%ld out of %ld curves)",
                                    k, (T)j / (T)ncurves * 100, j, ncurves);
                    }

                    // compute the one curve of control points
                    CtrlCurve(N[k], NtN, R, P, k, co[j], cs, to[j], temp_ctrl0, temp_ctrl1, j, ctrl_pts, weights, weighted);
                }

#endif          // end serial version

                // adjust offsets and strides for next dimension
                ntemp_ctrl(k) = nctrl_pts(k);
                cs *= ntemp_ctrl(k);

                // print progress
                if (verbose >= 2)
                    fprintf(stderr, "\r dimension %ld: 100%% encoded                                       \n", k);
            }  // domain dimensions

            // debug
//             cerr << "Encode() ctrl_pts:\n" << ctrl_pts << endl;
//             cerr << "Encode() weights:\n" << weights << endl;
        }

        void encodeBSpline( 
            const VectorXi& nctrl_pts,                          // number of control points in each dim.
            MatrixX<T>&     ctrl_pts)                           // (output) control points)
        {
            // check quantities
            if (mfa_data.p.size() != input.ndom_pts().size())
            {
                fprintf(stderr, "Error: encodeBSpline() size of p must equal size of ndom_pts\n");
                exit(1);
            }
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                if (nctrl_pts(i) <= mfa_data.p(i))
                {
                    fprintf(stderr, "Error: encodeBSpline() number of control points in dimension %ld "
                            "must be at least p + 1 for dimension %ld\n", i, i);
                    exit(1);
                }
                if (nctrl_pts(i) > input.ndom_pts(i))
                {
                    fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                            "exceeds number of input data points (%d) in dimension %ld.\n", nctrl_pts(i), i, input.ndom_pts(i), i);
                }
            }

            int      pt_dim = mfa_data.dim();               // control point dimensonality
            
            // resize matrices in case number of control points changed
            ctrl_pts.resize(nctrl_pts.prod(), pt_dim);

            // resize basis function matrices and initialize to 0; will only fill nonzeros later
            vector<MatrixX<T>> N(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
                N[k] = MatrixX<T>::Zero(input.ndom_pts(k), nctrl_pts(k));

            // 2 buffers of temporary control points
            // double buffer needed to write output curves of current dim without changing its input pts
            // temporary control points need to begin with size as many as the input domain points
            // except for the first dimension, which can be the correct number of control points
            // because the input domain points are converted to control points one dimension at a time
            // TODO: need to find a more space-efficient way
            size_t tot_ntemp_ctrl = 1;
            for (size_t k = 0; k < dom_dim; k++)
                tot_ntemp_ctrl *= (k == 0 ? nctrl_pts(k) : input.ndom_pts(k));
            MatrixX<T> temp_ctrl0 = input.domain.middleCols(mfa_data.min_dim, mfa_data.dim());
            MatrixX<T> temp_ctrl1 = MatrixX<T>::Zero(tot_ntemp_ctrl, pt_dim);

            // Define sizes of volumes before and after encoding each dimension
            VectorXi    inputSize = input.ndom_pts();
            VectorXi    outputSize = input.ndom_pts();

            // Encode each dimension independently (separably)
            for (size_t k = 0; k < dom_dim; k++)
            {
                // Update volume sizes
                inputSize = outputSize;
                outputSize(k) = nctrl_pts(k);

                // Iterators for each volume
                VolIterator inputVolumeIter(inputSize);
                VolIterator outputVolumeIter(outputSize);

                // number of curves in this dimension
                int ncurves = inputSize.prod() / inputSize(k);

                // N is a matrix of m (# input points) x n (# control points) basis function coefficients
                //  _                              _
                // |  N_0(u[0])   ... N_n-1(u[0])   |
                // |     ...      ...      ...      |
                // |  N_0(u[m-1]) ... N_n-1(u[m-1]) |
                //  -                              -
                // TODO: N is going to be very sparse when it is large: switch to sparse representation
                // N has semibandwidth < p  nonzero entries across diagonal
                for (int i = 0; i < N[k].rows(); i++)
                {
                    int span = mfa_data.tmesh.FindSpan(k, input.params->param_grid[k][i], nctrl_pts(k));

#ifndef MFA_TMESH
                    // original version for one tensor product
                    mfa_data.OrigBasisFuns(k, input.params->param_grid[k][i], span, N[k], i);
#else
                    mfa_data.BasisFuns(k, input.params->param_grid[k][i], span, N[k], i);
#endif
                }

                // Compute system matrix
                MatrixX<T> NtN  = N[k].transpose() * N[k];
                Eigen::LLT<MatrixX<T>> ntnLLT(NtN);

#ifdef MFA_TBB  // TBB version
                enumerable_thread_specific<SliceIterator> inputSliceIter(inputVolumeIter, k);
                enumerable_thread_specific<SliceIterator> outputSliceIter(outputVolumeIter, k);
                static affinity_partitioner ap;
                parallel_for (blocked_range<size_t>(0, ncurves), [&] (blocked_range<size_t>& r)
                {
                    // R is the right hand side needed for solving NtN * P = Nt*R
                    // P are the unknown control points and the solution to NtN * P = R
                    MatrixX<T> R(inputSize(k), pt_dim);
                    MatrixX<T> P(outputSize(k), pt_dim);
                    
                    for (auto j = r.begin(); j < r.end(); j++)        // for all the curves in this dimension
                    {
                        inputSliceIter.local().seek(j);
                        outputSliceIter.local().seek(j);
                        CurveIterator   inputCurveIter(inputSliceIter.local());       // one curve of the input points in the current dim
                        CurveIterator   outputCurveIter(outputSliceIter.local());      // one curve of the output points in the current dim

                        // compute the one curve of control points
                        ctrlCurveBSpline(inputCurveIter, R, temp_ctrl0, temp_ctrl1, N[k].transpose(), ntnLLT, P);

                        // Copy P to the intermediate volumes, temp_ctrl0 or temp_ctrl1
                        copyCtrl(outputCurveIter, P, temp_ctrl0, temp_ctrl1);

                        inputSliceIter.local().incr_iter();
                        outputSliceIter.local().incr_iter();
                    }   // curves in this dimension
                }, ap);

#endif              // end TBB version

#if defined(MFA_SERIAL) || defined(MFA_KOKKOS)
                // R is the right hand side needed for solving NtN * P = Nt*R
                MatrixX<T> R(inputSize(k), pt_dim);

                // P are the unknown control points and the solution to NtN * P = R
                MatrixX<T> P(outputSize(k), pt_dim);

                // encode curves in this dimension
                SliceIterator inputSliceIter(inputVolumeIter, k);
                SliceIterator outputSliceIter(outputVolumeIter, k);
                while (!inputSliceIter.done())
                {
                    // print progress
                    if (verbose >= 2)
                    {
                        int j = inputSliceIter.cur_iter();
                        if (j > 0 && j > 100 && j % (ncurves / 100) == 0)
                            fprintf(stderr, "\r dimension %ld: %.0f %% encoded (%ld out of %ld curves)",
                                    k, (T)j / (T)ncurves * 100, j, ncurves);
                    }

                    // TODO: check if we can move this out of inner loop (may need to reset iterators inside loop)
                    CurveIterator   inputCurveIter(inputSliceIter);       // one curve of the input points in the current dim
                    CurveIterator   outputCurveIter(outputSliceIter);      // one curve of the output points in the current dim

                    // compute the one curve of control points
                    ctrlCurveBSpline(inputCurveIter, R, temp_ctrl0, temp_ctrl1, N[k].transpose(), ntnLLT, P);

                    // Copy P to the intermediate volumes, temp_ctrl0 or temp_ctrl1
                    copyCtrl(outputCurveIter, P, temp_ctrl0, temp_ctrl1);

                    inputSliceIter.incr_iter();
                    outputSliceIter.incr_iter();
                }

#endif          // end serial version

                // copy final result back to tensor product
                if (dom_dim % 2 == 0)
                    ctrl_pts = temp_ctrl0.block(0, 0, nctrl_pts.prod(), pt_dim);    // TODO: change to Q0.topRows(t.nctrl_pts.prod());
                else
                    ctrl_pts = temp_ctrl1.block(0, 0, nctrl_pts.prod(), pt_dim);

                // print progress
                if (verbose >= 2)
                    fprintf(stderr, "\r dimension %ld: 100%% encoded                                       \n", k);
            }        
        }

        void ConsMatrix(    TensorIdx               t_idx,
                            int                     deriv,
                            T                       c_target,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            const VectorX<T>&       dom_mins,
                            const VectorX<T>&       dom_maxs,
                            SparseMatrixX<T>&       Ct)
        {
            clock_t fill_time = clock();
            if (verbose >= 2)
                cerr << "Adjusting matrix for regularization..." << endl;

            const int num_points = N.rows();
            const int num_ctrl = N.cols();
            const int num_cons = dom_dim * num_ctrl;
            const VectorX<T> extents = dom_maxs - dom_mins;

            if (Ct.rows() != num_ctrl || Ct.cols() != num_cons)
            {
                fmt::print("ERROR: Incorrect matrix dimensions of Ct in ConsMatrix()\nExiting.\n");
                exit(1);
            }

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
            VectorXi spans(dom_dim);

            vector<MatrixX<T>> B(dom_dim);
            for (int k = 0; k < dom_dim; k++)
            {
                B[k].resize(deriv+1, t.nctrl_pts(k));
            }

            int n_entries = (mfa_data.p.array() + 1).prod();
            Ct.reserve(VectorXi::Constant(Ct.cols(), n_entries));

            // Compute parameters for each derivative constraint
            // Compute basis functions and derivatives at each parameter
            vector<vector<T>> anchor_pts(dom_dim);
            for (int i = 0; i < dom_dim; i++) 
            {
                MatrixX<T> temp_ders(3, t.nctrl_pts(i)); // 3 rows stores up to 2nd deriv

                int pc = mfa_data.p(i);     // current p for this dimension

                anchor_pts[i].resize(t.nctrl_pts(i));
                anchor_pts[i][0] = 0;
                anchor_pts[i][t.nctrl_pts(i)-1] = 1.0;
                for (int j = 1; j < t.nctrl_pts(i) - 1; j++)
                {              
                    T low = mfa_data.tmesh.all_knots[i][j];
                    T high = mfa_data.tmesh.all_knots[i][j+pc+1];
                    T mid = (low + high) / 2;
                    int temp_span = 0;

                    // Find parameter for max value of basis function with binary search
                    temp_ders.setZero();
                    temp_span = mfa_data.tmesh.FindSpan(i, mid);
                    mfa_data.DerBasisFuns(i, mid, temp_span, 1, temp_ders);

                    // TODO speed up by using Newton's method instead.
                    //      Warning: Need to ensure convergence to the right vanishing derivative when using Newton's method (there are three per basis fxn)
                    while (temp_ders(1,j) < -0.01 || temp_ders(1,j) > 0.01)
                    {
                        if (temp_ders(1,j) > 0.01)    // if function is increasing, we are left of max value
                            low = mid;
                        else
                            high = mid;
                        mid = (low + high) / 2;

                        temp_span = mfa_data.tmesh.FindSpan(i, mid);
                        mfa_data.DerBasisFuns(i, mid, temp_span, 1, temp_ders);
                    }

                    anchor_pts[i][j] = mid;
                }
            }

            VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

            // Loop through each parameter corresponding to derivative constraint
            VolIterator pt_it(t.nctrl_pts);
            while (!pt_it.done())
            {
                // Compute basis function values and derivatives at each anchor
                for (int k = 0; k < dom_dim; k++)
                {
                    T u = anchor_pts[k][pt_it.idx_dim(k)];
                    int p = mfa_data.p(k);
                    spans[k] = mfa_data.tmesh.FindSpan(k, u);
                    ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

                    B[k].setZero();
                    mfa_data.DerBasisFuns(k, u, spans[k], deriv, B[k]);
                }

                // Compute constraints for each directional derivative
                for (int dk = 0; dk < dom_dim; dk++)
                {
                    VolIterator ctrl_vol_iter(nctrl_patch, ctrl_starts, t.nctrl_pts);
                    while (!ctrl_vol_iter.done())   // for each nonzero basis function at the location for the constraint
                    {
                        int ctrl_idx_full = ctrl_vol_iter.cur_iter_full();

                        T current_deriv = 1;
                        for (auto k = 0; k < dom_dim; k++) // compute tensor-product basis function
                        {
                            int idx = ctrl_vol_iter.idx_dim(k);

                            T mult = B[k]((k==dk) ? deriv : 0, idx);    // deriv if k==dk, otherwise normal basis function

                            if (k == dk)    // if this is a derivative, scale properly
                                mult /= pow(extents(k), deriv);

                            current_deriv *= mult;
                        }

                        Ct.insertBackUncompressed(ctrl_idx_full, pt_it.cur_iter() * dom_dim + dk) = current_deriv;

                        ctrl_vol_iter.incr_iter();
                    }
                }
                pt_it.incr_iter();
            }

            VectorX<T> reg_strengths(num_ctrl);

            // uniform_reg(t, N, Nt, Ct, reg_strengths, deriv, c_target);
            yeh_reg(t, N, Nt, Ct, reg_strengths, deriv, c_target);
            // vazquez_reg(t, N, Nt, Ct, reg_strengths, deriv, c_target);

            write_reg_strength(t, deriv, reg_strengths, anchor_pts);            

            fill_time = clock() - fill_time;
            if (verbose >= 2)
                cerr << "Regularization Total Time: " << setprecision(3) << ((double)fill_time)/CLOCKS_PER_SEC << "s." << endl;
        }

        void uniform_reg(TensorProduct<T>& t,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            SparseMatrixX<T>& Ct,
                            VectorX<T>& reg_strengths,
                            int deriv,
                            T c_target)
        {
            reg_strengths = VectorX<T>::Ones(t.nctrl_pts.prod());
            reg_strengths = c_target*reg_strengths;

            Ct = c_target*Ct;

            return;
        }

        void yeh_reg(TensorProduct<T>& t,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            SparseMatrixX<T>& Ct,
                            VectorX<T>& reg_strengths,
                            int deriv,
                            T c_target)
        {
#if 0 // OLD regularization technique (column-specific)
            // Compute regularization strengths
            SparseMatrixX<T> C(Ct.transpose());                             // create col-major transpose for fast column sums

            Eigen::DiagonalMatrix<T, Eigen::Dynamic> lambda(C.cols());      // regularization strengths
            for (int i = 0; i < C.cols(); i++)
            {
                T c_sum = N.col(i).sum();
                T c_add = (c_sum < c_target) ? c_target - c_sum : 0;
                if (deriv > 1)
                {
                    lambda.diagonal()(i) = c_add / C.col(i).cwiseAbs().sum();
                }
                else    
                {
                    // For 1st deriv constraints, only use when there are no value constraints
                    if (c_sum == 0)
                        lambda.diagonal()(i) = c_add / C.col(i).cwiseAbs().sum();
                    else
                        lambda.diagonal()(i) = 0;
                }

                reg_strengths(i) = lambda.diagonal()(i);
            }

            // Scale the constraint matrix Ct by the regularization strengths
            // Each row of Ct (col of C) is multiplied by a diagonal entry of lambda
            Ct = lambda * Ct;

#else // NEW regularization technique (row-specific)
            // Compute regularization strengths
            SparseMatrixX<T> C(Ct.transpose());                             // create col-major transpose for fast column sums
            Eigen::DiagonalMatrix<T, Eigen::Dynamic> lambda(C.rows());       // NEW test case
            T str = 0;
            for (int i = 0; i < C.cols(); i++)
            { 
                T n_sum = N.col(i).sum();
                T c_sum = C.col(i).cwiseAbs().sum();
                if (deriv == 1 && n_sum > 0)
                    str = 0;
                else
                    str = max((c_target - n_sum)/c_sum, (T)0);

                reg_strengths(i) = str; // nb. for debug below only

                for (int k = 0; k < dom_dim; k++)
                {
                    lambda.diagonal()(dom_dim*i + k) = str;
                }
            }

            // Scale the constraint matrix Ct by the regularization strengths
            // Each col of Ct (row of C) is multiplied by a diagonal entry of lambda
            Ct = Ct*lambda;
#endif

            return;
        }

        void vazquez_reg(TensorProduct<T>& t,
                            const SparseMatrixX<T>& N,
                            const SparseMatrixX<T>& Nt,
                            SparseMatrixX<T>& Ct,
                            VectorX<T>& reg_strengths,
                            int deriv,
                            T c_target) // should always be =2 for vazquez's method
        {
            if (deriv != 2)
            {
                cerr << "ERROR: deriv order not equal to 2 in Encoder::vazquez_reg()\nExiting" << endl;
                exit(0);
            }

            T str = 0;
            int lid = 0;
            VectorXi ijk(dom_dim);
            VectorXi ijk_r(dom_dim);
            GridInfo g;
            g.init(dom_dim, t.nctrl_pts);

            SparseMatrixX<T> NtN = Nt * Nt.transpose();

            int* op = NtN.outerIndexPtr();
            int* ip = NtN.innerIndexPtr();
            T*   vp = NtN.valuePtr();
            int oi = 0, ii = 0;
            T val = 0;


            Eigen::DiagonalMatrix<T, Eigen::Dynamic> lambda(Ct.cols());

            for (int j = 0; j < t.nctrl_pts.prod(); j++) // loop thru cols of NtN
            {
                T sum0 = 0, sum1 = 0;
                // oo = op[j];

                g.idx2ijk(j, ijk_r); // ijk coords of basis functions for col j
                
                VolIterator pt_it(t.nctrl_pts);
                while (!pt_it.done())
                {
                    lid = pt_it.cur_iter();
                    ijk = pt_it.idx_dim();


                    if ((ijk_r-ijk)(0) % 2 == 1) // if x offset between basis funs is odd
                    {
                        sum0 -= NtN.coeff(lid, j);
                    }
                    else
                    {
                        sum0 += NtN.coeff(lid, j);
                    }

                    if ((ijk_r-ijk)(1) % 2 == 1) // if y offset between basis funs is odd
                    {
                        sum1 -= NtN.coeff(lid, j);
                    }
                    else
                    {
                        sum1 += NtN.coeff(lid, j);
                    }

                    pt_it.incr_iter();
                }
                // cerr << "VazSum0: " << sum0 << endl;
                // cerr << "VazSum1: " << sum1 << endl;

                str = c_target * max(0.0, max(.11111 - abs(sum0), .11111 - abs(sum1)));
                reg_strengths(j) = str;
                for (int k = 0; k < dom_dim; k++)
                {
                    lambda.diagonal()(dom_dim*j + k) = str;
                }
            }
            
            Ct = Ct*lambda;     // right multiply for vazquez method

            return;
        }

        void write_reg_strength(TensorProduct<T>& t,
                                int deriv, 
                                VectorX<T>& reg_strengths, 
                                vector<vector<T>>& anchor_pts)
        {
            // DEBUG: Export the regularization strength and position for each term
            ofstream reg_st_out;
            string reg_st_fname = "reg-strength-" + to_string(deriv) + ".txt";
            reg_st_out.open(reg_st_fname);

            VolIterator pt_it(t.nctrl_pts);
            while (!pt_it.done())
            {
                for (int i = 0; i < dom_dim; i++)
                {
                    int id = pt_it.idx_dim(i);
                    reg_st_out << anchor_pts[i][id] << " ";
                }
                reg_st_out << reg_strengths(pt_it.cur_iter()) << endl;

                pt_it.incr_iter();
            }
            reg_st_out.close();
        }

        // Assemble B-spline collocation matrix for a full tensor, using sparse matrices
        // Here we are filling Nt (transpose of N)
        // Nt is a matrix of n x m scalars that are the basis function coefficients
        //  _                                      _
        // |  N_0(u[0])     ...     N_0(u[m-1])     |
        // |     ...        ...      ...            |
        // |  N_n-1(u[0])   ...     N_n-1(u[m-1])   |
        //  -                                      -
        void CollMatrixUnified( TensorIdx           t_idx,    // index of tensor product containing input points and control points
                                // vector<size_t>&     start_idxs,
                                // vector<size_t>&     end_idxs,
                                SparseMatrixX<T>&   N, // (output) collocation matrix 
                                SparseMatrixX<T>&   Nt)  // (output) transpose of collocation matrix
        {
            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];
            assert(Nt.rows() == t.nctrl_pts.prod());
            assert(Nt.cols() == input.npts);

            clock_t fill_time = clock();

            // resize matrices in case number of control points changed
            const int           pt_dim  = mfa_data.dim();        // dimensionality of current model
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());

            // control point strides
            VectorXi cps = VectorXi::Ones(dom_dim); 
            for (int k = 1; k < dom_dim; k++)
            {
                cps(k) = cps(k-1) * t.nctrl_pts(k-1);
            }

            // Prep for evaluating basis functions
            VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
            BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
            vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
            for (int k = 0; k < dom_dim; k++)
            {
                lB[k].resize(q[k]);
            }

            VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi spans(dom_dim);                                       // current knot span in each dimension
            VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension

            // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
            int bf_per_pt = (mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
            Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

            int         tot_nnzs = 0;
            vector<int> row_nnzs(Nt.rows(), 0);

            // Iterate thru every point in subvolume given by tensor
            VectorX<T> param(input.dom_dim);
            for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
            {
                input_it.params(param);

                // Compute basis functions at input point
                for (auto k = 0; k < dom_dim; k++)
                {
                    int p   = mfa_data.p(k);
                    T   u   = param(k);

                    spans[k] = mfa_data.tmesh.FindSpan(k, u);

                    ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

                    mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi);
                }

                // Compute matrix value and insert into Nt
                VolIterator ctrl_vol_iter(nctrl_patch);
                while (!ctrl_vol_iter.done())
                {
                    int ctrl_idx_full = 0;
                    T coeff_prod = 1;
                    for (int k = 0; k < dom_dim; k++)
                    {
                        int idx = ctrl_vol_iter.idx_dim(k);
                        ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
                        coeff_prod *= lB[k][idx];
                    }

                    // increment number of entries in this row (for use in constructing N)
                    row_nnzs[ctrl_idx_full]++;
                    tot_nnzs++;

                    Nt.insertBackUncompressed(ctrl_idx_full, input_it.idx()) = coeff_prod;  
                    // FRAGILE command; reserve() must be called prior, **with vector signature**
                    // entries must be inserted only once, and entries within a column must be inserted
                    // in order of increasing row index.
                    // This is an internal SparseMatrix command and no consistency checks are done.
                    // 
                    // Empirical testing indicates that insertBackUncompressed is faster than insert, which
                    // is faster than setFromTriplets(); however, space must be reserved/allocated properly ahead of time.
                    // setFromTriplets also requires a bit more memory, but is the most robust/user-friendly.

                    ctrl_vol_iter.incr_iter();
                }
            }

            Nt.makeCompressed();  // not necessary if using prune(), as prune always returns compressed form
            // Nt.prune(1,1e-5);  // remove entries less than a given value (and compress matrix)


            // this code block fills N as the transpose of Nt.
            // it is marginally faster than:
            //      N = Nt.transpose()
            {
                // Must do this after Nt is in compressed mode!
                N.reserve(row_nnzs);    // allocate innerIndexPtr and valuePtr to store each row. row_nnzs in num nnzs per row of Nt

                int* col_starts     = N.outerIndexPtr();
                int* row_ids        = N.innerIndexPtr();
                T*   tr_values      = N.valuePtr();
                int* col_inner_nnzs = N.innerNonZeroPtr();

                int* oip    = Nt.outerIndexPtr();  // oip has Nt.cols() + 1 entries
                int* iip    = Nt.innerIndexPtr();  // iip[oip[o]] is the row index of the first entry in the o^th column
                T*   values = Nt.valuePtr();   

                for (int o = 0; o < Nt.cols(); o++) // loop through columns of Nt (rows of N)
                {
                    for (int i = 0; i < oip[o+1] - oip[o]; i++) // loop through entries in column o of Nt
                    {
                        int row = o;                // row in N
                        int col = iip[oip[o]+i];    // col in N
                        row_ids[col_starts[col] + col_inner_nnzs[col]] = row;
                        tr_values[col_starts[col] + col_inner_nnzs[col]] = values[oip[o]+i];

                        // col_nnzs tracks how many entries we have inserted into each col buffer
                        // Since we loop through the rows of N in order, the row indices within
                        //     each col buffer are strictly increasing
                        col_inner_nnzs[col]++;
                    }
                }

                N.makeCompressed();
            }

            fill_time = clock() - fill_time;
            if (verbose >= 2)
                cerr << "Matrix Construction Time: " << setprecision(3) << ((double)fill_time)/CLOCKS_PER_SEC << "s." << endl;
        }

#ifdef MFA_TBB
        // Computes the product of two Eigen sparse matrices using TBB.  All matrices must be in column major format.
        void MatProdThreaded(   Eigen::SparseMatrix<T, Eigen::ColMajor>& lhs,
                                Eigen::SparseMatrix<T, Eigen::ColMajor>& rhs,
                                Eigen::SparseMatrix<T, Eigen::ColMajor>& res,
                                int reserveSizes )
        {
            using LhsInIt = typename Eigen::SparseMatrix<T, Eigen::ColMajor>::InnerIterator;
            using RhsInIt = typename Eigen::SparseMatrix<T, Eigen::ColMajor>::InnerIterator;

            int rows = lhs.innerSize();
            int cols = rhs.outerSize();

            res.setZero();
            res.reserve(VectorXi::Constant(rows, reserveSizes));

            // Internal SparseMatrix buffers
            int* outer_index = res.outerIndexPtr();
            int* inner_nnz = res.innerNonZeroPtr();
            auto& res_data = res.data();

            tbb::affinity_partitioner ap;
            parallel_for(blocked_range<size_t>(0, cols), [&](blocked_range<size_t>& r)
            {
                int j_begin = r.begin();
                int j_end = r.end();

                std::vector<char> mask(rows, 0);
                std::vector<T> values(rows, 0);

                // we compute each column of the result, one after the other
                for (int j=j_begin; j<j_end; ++j)
                {
                    for (RhsInIt rhsIt(rhs, j); rhsIt; ++rhsIt)
                    {
                        T y = rhsIt.value();
                        int k = rhsIt.index();
                        for (LhsInIt lhsIt(lhs, k); lhsIt; ++lhsIt)
                        {
                            int i = lhsIt.index();
                            T x = lhsIt.value();
                            if(mask[i] == 0)
                            {
                                mask[i] = 1;
                                values[i] = x * y;
                            }
                            else
                                values[i] += x * y;
                        }
                    }

                    // TODO: can loop over nonzeros only, but this involves sorted insertion, which may be slower
                    for(int i=0; i<rows; ++i)
                    {
                        if(mask[i] == 1)
                        {
                            mask[i] = 0;

                            // We never want this (slow) reallocation to happen. Always OVER-estimate reserveSizes (input variable)
                            if( inner_nnz[j] >= outer_index[j+1] - outer_index[j] )
                            {
                                cerr << "Warning: Reallocating memory in threaded matrix product. Did you reserve space properly?" << endl;
                                VectorXi extra_size = VectorXi::Zero(cols);
                                extra_size[j] = std::max<int>(2, inner_nnz[j]);
                                res.reserve(extra_size);  // NB: reserve() always adds to existing buffer
                            }

                            // Insert entry 
                            int p = outer_index[j] + inner_nnz[j];
                            res_data.index(p) = i;
                            res_data.value(p) = values[i];
                            inner_nnz[j]++;
                        }
                    }
                }

            }, ap );

            res.makeCompressed();
        }

#endif // MFA_TBB

        // Encodes ctrl points in each dimensions simultaneously
        // Necessary for encoding unstructured input data, where parameter values
        // corresponding to input do not lie on structured grid
        // 
        // TODO: Currently only valid for one tensor product. To extend to general
        // tmesh, need a way to compute the total number of input points in a given
        // tensor. At the moment, this is only possible with the domain_pts() 
        // method, which does not work for unstructured data
        void EncodeUnified( TensorIdx   t_idx,                      // tensor product being encoded
                            T           regularization=0,           // parameter to set smoothing of artifacts from non-uniform data
                            bool        reg1and2=false,
                            bool        weighted=true)                   // solve for and use weights 
        {
            if (verbose >= 2)
            {
                fmt::print("Encoding dimensions all at once\n");
            }
            if (verbose >= 3)
            {
                fmt::print("  => NOTE: Unified encode only valid for single tensor product!\n");
            }
            
            if (weighted)  // We want weighted encoding to be default behavior eventually. However, not currently implemented.
            {
                fmt::print("ERROR: NURBS (nonuniform weights) are not implemented for unified-dimensional encoding!\n");
                exit(1);
            }

            // additional constraints
            // it turns out these don't change anything, leave set to 0
            int extra_cons = 0;

            const int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                           // control point dimensonality
            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];

            // // TODO: REQUIRED for Tmesh
            // // Compute total number of points in tensor product
            // // Does not work for unstructured data yet, see note at top of function
            // int tot_dom_pts = 1;
            // vector<size_t> start_idxs(dom_dim);
            // vector<size_t> end_idxs(dom_dim);
            // mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, true, start_idxs, end_idxs);
            // for (int k=0; k < dom_dim; k++)
            //     tot_dom_pts *= end_idxs[k] - start_idxs[k] + 1;

            // Assemble collocation matrix
            SparseMatrixX<T> N(input.npts, t.nctrl_pts.prod());
            SparseMatrixX<T> Nt(t.nctrl_pts.prod(), input.npts);
            CollMatrixUnified(t_idx, /*start_idxs, end_idxs,*/ N, Nt);

            // Construct RHS of linear system
            MatrixX<T>  R(Nt.cols(), pt_dim);
            RHSUnified(Nt, R);

            if (regularization > 0)
            {
                if (verbose >= 2)
                    fmt::print("Applying model regularization with strength r={}\n", regularization);

                int num_reg_conds = t.nctrl_pts.prod();
                SparseMatrixX<T> Ct(Nt.rows(), num_reg_conds * dom_dim);
                SparseMatrixX<T> Ct1(Nt.rows(), num_reg_conds * dom_dim);
                SparseMatrixX<T> Ct2(Nt.rows(), num_reg_conds * dom_dim);

                if (reg1and2)   // constrain 1st and 2nd derivs
                {
                    ConsMatrix(t_idx, 1, regularization, N, Nt, input.mins(), input.maxs(), Ct1);
                    ConsMatrix(t_idx, 2, regularization, N, Nt, input.mins(), input.maxs(), Ct2);

                    Ct.conservativeResize(Ct1.rows(), Ct1.cols() + Ct2.cols());
                    Ct.leftCols(Ct1.cols()) = Ct1;
                    Ct.rightCols(Ct2.cols()) = Ct2;
                    Ct.makeCompressed();
                }
                else            // constrain only 2nd derivs
                {
                    ConsMatrix(t_idx, 2, regularization, N, Nt, input.mins(), input.maxs(), Ct2);
                    Ct = Ct2;            // only C2 regularization
                }

                // Concatenate Ct to the right end of Nt
                Nt.conservativeResize(Nt.rows(), Nt.cols() + Ct.cols());
                Nt.rightCols(Ct.cols()) = Ct;
                Nt.makeCompressed();
            }

            // Set up linear system
            SparseMatrixX<T> Mat(Nt.rows(), Nt.rows()); // Mat will be the matrix on the LHS

#ifdef MFA_TBB  // TBB version
            Eigen::SparseMatrix<T, Eigen::ColMajor> NCol = Nt.transpose();

            int ntn_sparsity = (2*mfa_data.p + VectorXi::Ones(dom_dim)).prod();       // nonzero basis functions per input point
            MatProdThreaded(Nt, NCol, Mat, ntn_sparsity);
#else
            Mat = Nt * Nt.transpose();
#endif

            // Solve Linear System
            // Eigen::ConjugateGradient<SparseMatrixX<T>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteLUT<T>>  solver;
            Eigen::ConjugateGradient<SparseMatrixX<T>, Eigen::Lower|Eigen::Upper>  solver;  // Default preconditioner is Jacobi

            // // Optional parameters for solver    
            solver.setTolerance(1e-8);
            // solver.preconditioner().setDroptol(0.001);
            // solver.preconditioner().setFillfactor(1);

            solver.compute(Mat);
            if (solver.info() != Eigen::Success) 
                fmt::print("WARNING: Matrix decomposition failed in EncodeTensor\n");
            else if (verbose >= 2)
                fmt::print("Sparse matrix factorization successful\n");

            t.ctrl_pts = solver.solve(R); 
            if (solver.info() != Eigen::Success)
            {
                fmt::print("WARNING: Least-squares solve failed in EncodeTensor\n");
                fmt::print("  error: {} (tolerance = {})\n", solver.error(), solver.tolerance());
                fmt::print("  # iterations: {} (max iterations = {})\n", solver.iterations(), solver.maxIterations());
            }
            else if (verbose >= 2)
            {
                fmt::print("Sparse matrix solve successful\n");
                fmt::print("  # iterations: {}\n", solver.iterations());
            }
        }

#ifdef MFA_TMESH

        // whether a curve of input points intersects a tensor product
        // only checks dimensions above the current dim
        // higher dims are for input point space, lower dims are in control point space
        // lower dims are guranteed to intersect already
        // NB: not general purpose because does not check all dims
        bool CurveIntersectsTensor(
                TensorIdx               t_idx,                          // index of current tensor
                int                     dim,                            // current dimension of curve
                const VectorXi&         nin_pts,                        // number of input points
                const VectorXi&         start_ijk)                      // i,j,k of start of input points
        {
            auto& t = mfa_data.tmesh.tensor_prods[t_idx];

            // ijk and param of point in the middle of the curve
            VectorXi mid_ijk = start_ijk;
            mid_ijk(dim) += nin_pts(dim) / 2;
            VectorX<T> mid_param(dom_dim);

            for (auto i = dim + 1; i < dom_dim; i++)           // only checks dimensions after current dim; earlier dims guranteed to intersect
            {
                mid_param(i) = input.params->param_grid[i][mid_ijk(i)];
                if (mid_param(i) < mfa_data.tmesh.all_knots[i][t.knot_mins[i]] || mid_param(i) > mfa_data.tmesh.all_knots[i][t.knot_maxs[i]])
                    return false;
            }

            return true;
        }

        // computes curve of free control points in one dimension
        // helper function for EncodeTensorLocalSeparable
        void ComputeCtrlPtCurve(
                CurveIterator&          in_curve_iter,                  // current curve
                TensorIdx               t_idx,                          // index of current tensor
                int                     dim,                            // current curve dimension
                MatrixX<T>&             R,                              // right hand side, allocated by caller
                MatrixX<T>&             Q,                              // first matrix of input points, allocated by caller
                MatrixX<T>&             Q1,                             // second matrix of input points, allocated by caller
                MatrixX<T>&             Nfree,                          // free basis functions, allocated by caller
                MatrixX<T>&             Ncons,                          // constraint basis functions, allocated by caller
                MatrixX<T>&             Pcons,                          // constraint control points, allocated by caller
                MatrixX<T>&             P,                              // (output) solution control points, allocated by caller
                ConsType                cons_type,                      // constraint type
                const VectorXi&         nin_pts,                        // number of input points
                const VectorXi&         start_ijk,                      // i,j,k of start of input points
                int                     ncons,                          // number of constraints
                double                  free_time,                      // time to find free basis functions
                double                  cons_time,                      // time to find constraints
                double                  norm_time,                      // time to normalize basis functions
                double                  solve_time)                     // time to solve the matrix
        {
            auto& t = mfa_data.tmesh.tensor_prods[t_idx];

            // copy one curve of input points to right hand side
            while (!in_curve_iter.done())
            {
                if (dim == 0)
                    R.row(in_curve_iter.cur_iter()) =
                        input.domain.block(in_curve_iter.ijk_idx(in_curve_iter.cur_ijk()), mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
                else if (dim % 2 == 0)
                    R.row(in_curve_iter.cur_iter()) = Q.row(in_curve_iter.ijk_idx(in_curve_iter.cur_ijk()));
                else
                    R.row(in_curve_iter.cur_iter()) = Q1.row(in_curve_iter.ijk_idx(in_curve_iter.cur_ijk()));
                in_curve_iter.incr_iter();
            }

            // find matrix of free control point basis functions
            double t0 = MPI_Wtime();
            FreeCtrlPtCurve(dim, t_idx, start_ijk, nin_pts(dim), Nfree);

            // timing
            free_time   += (MPI_Wtime() - t0);
            t0          = MPI_Wtime();

            // normalize Nfree and Ncons such that the row sum of Nfree + Ncons = 1.0
            t0          = MPI_Wtime();              // timing

#ifdef MFA_TBB

            static affinity_partitioner ap;
            parallel_for (blocked_range<size_t>(0, Nfree.rows()), [&] (blocked_range<size_t>& r)
            {
                for (auto i = r.begin(); i < r.end(); i++)

#else

                for (auto i = 0; i < Nfree.rows(); i++)

#endif

                {
                    bool error = false;
                    T sum = Nfree.row(i).sum();

                    // debug
//                     if (cons_type != ConsType::MFA_NO_CONSTRAINT && Ncons.rows() == 0)
//                         fmt::print(stderr, "ComputeCtrlPtCurve(): did not find any constraints\n");

                    if (cons_type != ConsType::MFA_NO_CONSTRAINT && Ncons.rows() > i)
                        sum += Ncons.row(i).sum();
                    if (sum > 0.0)
                    {
                        Nfree.row(i) /= sum;
                        if (cons_type != ConsType::MFA_NO_CONSTRAINT && Ncons.rows() > i)
                            Ncons.row(i) /= sum;
                    }
                    else
//                         throw MFAError(fmt::format("ComputeCtrlPtCurve(): row {} has 0.0 row sum for free and constraint basis functions in dim {}",
//                                 i, dim));
                        fmt::print(stderr, "ComputeCtrlPtCurve(): row {} has 0.0 row sum for free and constraint basis functions in dim {}\n",
                                i, dim);
                }

#ifdef MFA_TBB

            }, ap);

#endif

            norm_time += (MPI_Wtime() - t0);                    // timing

            // R is the right hand side needed for solving N * P = R
            if (Pcons.rows())
                R -= Ncons * Pcons;

            // solve
            t0 = MPI_Wtime();
            P = (Nfree.transpose() * Nfree).ldlt().solve(Nfree.transpose() * R);
            solve_time += (MPI_Wtime() - t0);
        }

        // computes curve of free control points basis functions in one dimension
        // matrix Nfree of basis functions is m rows x n columns, where
        //  m is number of input points covered by constraints and free control points
        //  n is number of free control points
        // the curve of input points may not be in the tensor of control points, but can be in a neighbor if the
        //  curve is in the constraint region in a dim. orthogonal to the current dim.
        // helper function for EncodeTensorLocalSeparable
        void FreeCtrlPtCurve(
                int                 dim,                // current dimension
                TensorIdx           t_idx,              // index of tensor of control points
                const VectorXi&     start_ijk,          // multidim index of first input point for the curve, including constraints
                size_t              npts,               // number of input points in current dim, including constraints
                MatrixX<T>&         Nfree)              // (output) matrix of free control points basis functions
        {
            auto&               t = mfa_data.tmesh.tensor_prods[t_idx];
            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor
            Nfree = MatrixX<T>::Zero(npts, t.nctrl_pts(dim));

            // local knot vector
            vector<KnotIdx> local_knot_idxs(mfa_data.p(dim) + 2);                               // local knot indices for current dim
            vector<T> local_knots(mfa_data.p(dim) + 2);                                         // local knot vector for current dim

#if 0

            // TODO: decide whether there is enough parallelism, if so, update TBB for serial version below
            // TODO: if this is used, needs to be carefully reviewed and updated, not current

// #ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts()); // iterator over input points
            VolIterator free_iter(t.nctrl_pts);                         // iterator over free control points

            enumerable_thread_specific<vector<KnotIdx>>         thread_local_knot_idxs(mfa_data.p(dim) + 2);// local knot idices
            enumerable_thread_specific<vector<T>>               thread_local_knots(mfa_data.p(dim) + 2);    // local knot values
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(mfa_data.dom_dim);           // multidim index of domain point
            enumerable_thread_specific<VectorXi>                thread_free_ijk(mfa_data.dom_dim);          // multidim index of control point
            enumerable_thread_specific<vector<KnotIdx>>         thread_anchor(mfa_data.dom_dim);            // anchor of control point
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, dom_iter.tot_iters(), 0, free_iter.tot_iters()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)                                // for control points
                {
                    free_iter.idx_ijk(i, thread_free_ijk.local());                                      // ijk of domain point
                    mfa_data.tmesh.ctrl_pt_anchor(t, thread_free_ijk.local(), thread_anchor.local());   // anchor of control point

                    // local knot vector
                    mfa_data.tmesh.knot_intersections_dim(thread_anchor.local(), t_idx, true, thread_local_knot_idxs.local(), dim);
                    for (auto n = 0; n < thread_local_knot_idxs.local().size(); n++)
                        thread_local_knots.local()[n] = mfa_data.tmesh.all_knots[dim][thread_local_knot_idxs.local()[n]];

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)                            // for input domain points
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                                    // ijk of domain point
                        T u = input.params->param_grid[dim][thread_dom_ijk.local()(dim)];               // parameter of current input point
                        Nfree(j, i) = mfa_data.OneBasisFun(dim, u, thread_local_knots.local());         // basis function
                    }       // for blocked range rows, ie, input domain points
                }       // for blocked range cols, ie, control points
            }, ap); // parallel for

#else           // serial

            VectorX<T> param(dom_dim);

            // for start of the curve, for dims prior to current dim, find anchor and param
            // those dims are in control point index space for the current tensor
            for (auto i = 0; i < dim; i++)
            {
                mfa_data.tmesh.knot_idx_ofst(t, t.knot_mins[i], start_ijk(i), i, true, anchor[i]);                     // computes anchor as offset from start of tensor
                param(i)    = mfa_data.tmesh.all_knots[i][anchor[i]];
            }

            // for the start of the curve, for current dim. and higher, find param
            // these dims are in the input point index space
            for (auto i = dim; i < dom_dim; i++)
                param(i) = input.params->param_grid[i][start_ijk(i)];

            // for the start of the curve, for higher than the current dim, find anchor
            // these dims are in the input point space
            // in the current dim, the anchor coordinate will be replaced below by the control point anchor
            for (auto i = dim + 1; i < dom_dim; i++)
            {
                // if param == 0, FindSpan finds the last 0-value knot span, but we want the first control point anchor, which is an earlier span
                if (param(i) == 0.0)
                    anchor[i] = (mfa_data.p(i) + 1) / 2;
                else if (param(i) == 1.0)
                    anchor[i] = mfa_data.tmesh.all_knots[i].size() - 2 - (mfa_data.p(i) + 1) / 2;
                else
                    anchor[i] = mfa_data.tmesh.FindSpan(i, param(i), t);
            }

            for (auto i = 0; i < t.nctrl_pts(dim); i++)                                                 // for all control points in current dim
            {
                // anchor of control point in current dim
                anchor[dim] = mfa_data.tmesh.ctrl_pt_anchor_dim(dim, t, i);

                // local knot vector in currrent dimension
                mfa_data.tmesh.knot_intersections_dim(anchor, t_idx, local_knot_idxs, dim);             // local knot indices in current dim
                for (auto n = 0; n < local_knot_idxs.size(); n++)                                       // local knots in only current dim
                    local_knots[n] = mfa_data.tmesh.all_knots[dim][local_knot_idxs[n]];

                for (auto j = 0; j < npts; j++)                                                         // for all input points (for this tensor) in current dim
                {
                    T u = input.params->param_grid[dim][start_ijk(dim) + j];                            // parameter of current input point
                    Nfree(j, i) = mfa_data.OneBasisFun(dim, u, local_knots);                            // basis function
                }
            }       // control points

#endif          // TBB or serial

        }

        // free control points matrix of basis functions
        // matrix Nfree of basis functions is m rows x n columns, where
        // m is number of input points covered by constraints and free control points
        // n is number of free control points
        // helper function for EncodeTensorLocalUnified
        // returns max number of nonzeros in any column
        int  FreeCtrlPtMat(TensorIdx            t_idx,              // index of tensor of control points
                           VectorXi&            ndom_pts,           // number of relevant input points in each dim, covering constraints
                           VectorXi&            dom_starts,         // starting offsets of relevant input points in each dim, covering constraints
                           MatrixX<T>&          Nfree)              // (output) matrix of free control points basis functions
        {
            TensorProduct<T>&   t = mfa_data.tmesh.tensor_prods[t_idx];
            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor
            Nfree = MatrixX<T>::Zero(ndom_pts.prod(), t.ctrl_pts.rows());
            int max_nnz_col = 0;                                                                // max num nonzeros in any column

            // debug
            fmt::print(stderr, "Nfree has {} rows and {} columns\n", Nfree.rows(), Nfree.cols());

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

#ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts()); // iterator over input points
            VolIterator free_iter(t.nctrl_pts);                         // iterator over free control points

            enumerable_thread_specific<vector<vector<KnotIdx>>> thread_local_knot_idxs(dom_dim);   // local knot idices
            enumerable_thread_specific<vector<vector<T>>>       thread_local_knots(dom_dim);       // local knot idices
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(dom_dim);           // multidim index of domain point
            enumerable_thread_specific<VectorXi>                thread_free_ijk(dom_dim);          // multidim index of control point
            enumerable_thread_specific<vector<KnotIdx>>         thread_anchor(dom_dim);            // anchor of control point
            enumerable_thread_specific<vector<int>>             thread_nnz_col(Nfree.cols(), 0);            // number of nonzeros in each column
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, dom_iter.tot_iters(), 0, free_iter.tot_iters()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)                                // for control points
                {
                    if (i == r.cols().begin())
                    {
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            thread_local_knot_idxs.local()[k].resize(mfa_data.p(k) + 2);
                            thread_local_knots.local()[k].resize(mfa_data.p(k) + 2);
                        }
                    }

                    free_iter.idx_ijk(i, thread_free_ijk.local());                                      // ijk of domain point
                    mfa_data.tmesh.ctrl_pt_anchor(t, thread_free_ijk.local(), thread_anchor.local());   // anchor of control point

                    // local knot vector
                    mfa_data.tmesh.knot_intersections(thread_anchor.local(), t_idx, thread_local_knot_idxs.local());
                    for (auto k = 0; k < dom_dim; k++)
                    {
                        for (auto n = 0; n < thread_local_knot_idxs.local()[k].size(); n++)
                            thread_local_knots.local()[k][n] = mfa_data.tmesh.all_knots[k][thread_local_knot_idxs.local()[k][n]];
                    }

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)                            // for input domain points
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                                    // ijk of domain point
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            T u = input.params->param_grid[k][thread_dom_ijk.local()(k)];               // parameter of current input point
                            T B = mfa_data.OneBasisFun(k, u, thread_local_knots.local()[k]);                           // basis function
                            Nfree(j, i) = (k == 0 ? B : Nfree(j, i) * B);
                        }
                        if (Nfree(j, i) != 0.0)
                            thread_nnz_col.local()[i]++;
                    }       // for blocked range rows, ie, input domain points
                }       // for blocked range cols, ie, control points
            }, ap); // parallel for

            // combine thread-safe nnz_col
            vector<int> nnz_col(Nfree.cols(), 0);       // nonzeros in each column of Nfree, summed from threads
            thread_nnz_col.combine_each([&](const vector<int>& nnzs)
            {
                for (auto i = 0; i < Nfree.cols(); i++)
                {
                    nnz_col[i]  += nnzs[i];
                    if (nnz_col[i] > max_nnz_col)
                        max_nnz_col = nnz_col[i];
                }
            });

#else           // serial

            VolIterator free_iter(t.nctrl_pts);                         // iterator over free control points
            while (!free_iter.done())
            {
                VectorXi ijk(dom_dim);                                                 // ijk of current control point
                free_iter.idx_ijk(free_iter.cur_iter(), ijk);

                // anchor of control point
                mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchor, t_idx, local_knot_idxs);
                for (auto k = 0; k < dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                int nnz_col = 0;                                                                // num nonzeros in current column

                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());                     // iterator over input points
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < dom_dim; k++)                                 // for all dims
                    {
                        T u = input.params->param_grid[k][dom_iter.idx_dim(k)];                 // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) =
                            (k == 0 ? B : Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) * B);
                    }       // for all dims
                    if (Nfree(dom_iter.cur_iter(), free_iter.cur_iter()) != 0.0)
                        nnz_col++;
                    dom_iter.incr_iter();
                }       // domain point iterator

                if (nnz_col > max_nnz_col)
                    max_nnz_col = nnz_col;
                free_iter.incr_iter();
            }       // free control point iterator

#endif          // TBB or serial

            return max_nnz_col;
        }

        // free control points matrix of basis functions
        // helper function for EncodeTensorLocalUnified
        // sparse matrix version; not used for now, but might be in the future
        void FreeCtrlPtMatSparse(TensorIdx          t_idx,              // index of tensor of control points
                                 VectorXi&          ndom_pts,           // number of relevant input points in each dim
                                 VectorXi&          dom_starts,         // starting offsets of relevant input points in each dim
                                 SparseMatrixX<T>&  Nfree_sparse)       // (output) matrix of free control points basis functions
        {
            typedef Eigen::Triplet<T> Triplet;                                                  // (row, column, value)
            vector<Triplet> coeffs;                                                             // nonzero coefficients

            TensorProduct<T>&   t = mfa_data.tmesh.tensor_prods[t_idx];
            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

            // iterator over free control points
            VolIterator free_iter(t.nctrl_pts);
            while (!free_iter.done())
            {
                VectorXi ijk(dom_dim);                                                 // ijk of current control point
                free_iter.idx_ijk(free_iter.cur_iter(), ijk);

                // anchor of control point
                mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchor, t_idx, true, local_knot_idxs);
                for (auto k = 0; k < dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());
                while (!dom_iter.done())
                {
                    T v;                                                                        // basis function value
                    for (auto k = 0; k < dom_dim; k++)                                 // for all dims
                    {
                        int p = mfa_data.p(k);                                                  // degree of current dim.
                        T u = input.params->param_grid[k][dom_iter.idx_dim(k)];                             // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        v = (k == 0 ? B : v * B);
                    }       // for all dims

                    coeffs.push_back(Triplet(dom_iter.cur_iter(), free_iter.cur_iter(), v));
                    dom_iter.incr_iter();
                }       // domain point iterator

                free_iter.incr_iter();
            }       // free control point iterator

            Nfree_sparse.setFromTriplets(coeffs.begin(), coeffs.end());
        }

        // constraint control points matrix of basis functions
        // helper function for EncodeTensorLocalUnified and EncodeTensorLocalSeparable
        // Ncons needs to be sized correctly by caller
        void ConsCtrlPtMat(VectorXi&                ndom_pts,           // number of relevant input points in each dim
                           VectorXi&                dom_starts,         // starting offsets of relevant input points in each dim
                           vector<vector<KnotIdx>>& anchors,            // anchors of constraint control points                                                    // corresponding anchors
                           vector<TensorIdx>&       t_idx_anchors,      // tensors containing corresponding anchors
                           MatrixX<T>&              Ncons)              // (output) matrix of constraint control points basis functions
        {
            Ncons = MatrixX<T>::Constant(ndom_pts.prod(), Ncons.cols(), -1);         // basis functions, -1 means unassigned so far

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

#ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());                         // iterator over input points

            enumerable_thread_specific<vector<vector<KnotIdx>>> thread_local_knot_idxs(dom_dim);   // local knot idices
            enumerable_thread_specific<vector<vector<T>>>       thread_local_knots(dom_dim);       // local knot idices
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(dom_dim);   // multidim index of domain point
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, Ncons.rows(), 0, Ncons.cols()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)
                {
                    if (i == r.cols().begin())
                    {
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            thread_local_knot_idxs.local()[k].resize(mfa_data.p(k) + 2);
                            thread_local_knots.local()[k].resize(mfa_data.p(k) + 2);
                        }
                    }

                    // local knot vector
                    mfa_data.tmesh.knot_intersections(anchors[i], t_idx_anchors[i], thread_local_knot_idxs.local());
                    for (auto k = 0; k < dom_dim; k++)
                    {
                        for (auto n = 0; n < thread_local_knot_idxs.local()[k].size(); n++)
                            thread_local_knots.local()[k][n] = mfa_data.tmesh.all_knots[k][thread_local_knot_idxs.local()[k][n]];
                    }

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                        // ijk of domain point
                        for (auto k = 0; k < dom_dim; k++)
                        {
                            T u = input.params->param_grid[k][thread_dom_ijk.local()(k)];   // parameter of current input point
                            T B = mfa_data.OneBasisFun(k, u, thread_local_knots.local()[k]);// basis function
                            if (Ncons(j, i) == -1.0)                                        // unassigned so far
                                Ncons(j, i) = B;
                            else
                                Ncons(j, i) *= B;
                        }
                    }       // for blocked range rows
                }       // for blocked range cols
            }, ap); // parallel for

#else       // serial

            for (auto i = 0; i < Ncons.cols(); i++)                                             // for all constraint control points
            {
                // local knot vector
                mfa_data.tmesh.knot_intersections(anchors[i], t_idx_anchors[i], local_knot_idxs);
                for (auto k = 0; k < dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < dom_dim; k++)                                 // for all dims
                    {
                        T u = input.params->param_grid[k][dom_iter.idx_dim(k)];                             // parameter of current input point
                        T B = mfa_data.OneBasisFun(k, u, local_knots[k]);                       // basis function
                        if (Ncons(dom_iter.cur_iter(), i) == -1.0)                              // unassigned so far
                            Ncons(dom_iter.cur_iter(), i) = B;
                        else
                            Ncons(dom_iter.cur_iter(), i) *= B;
                    }           // for all dims
                    dom_iter.incr_iter();
                }           // domain points iterator
            }       // for all constraint control points

#endif      // TBB or serial

            // set any unassigned values in Ncons to 0
            for (auto i = 0; i < Ncons.rows(); i++)
            {
                for (auto j = 0; j < Ncons.cols(); j++)
                    if (Ncons(i, j) == -1.0)
                        Ncons(i, j) = 0.0;
            }
        }

        // encodes the control points for one tensor product of a tmesh
        // takes a subset of input points from the global domain, covered by basis functions of this tensor product
        // solves all dimensions together (not separably)
        // does not encode weights for now
        // latest linear constrained formulation as proposed by David Lenz (see wiki/notes/linear-constrained-fit.pdf)
        void EncodeTensorLocalUnified(
                TensorIdx                 t_idx,                  // index of tensor product being encoded
                bool                      weighted = true)        // solve for and use weights
        {
            // debug
            bool debug = false;

            // additional constraints
            // it turns out these don't change anything, leave set to 0
            int extra_cons = 0;

            // timing
            double setup_time   = MPI_Wtime();
            double q_time       = MPI_Wtime();
            fmt::print(stderr, "\nSetting up...\n");

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];                               // current tensor product
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                                   // control point dimensionality

            // get input domain points covered by the tensor and the constraints
            vector<size_t> start_idxs(mfa_data.dom_dim);
            vector<size_t> end_idxs(mfa_data.dom_dim);
            mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, true, extra_cons, start_idxs, end_idxs);

            // Q matrix of relevant input domain points
            VectorXi ndom_pts(dom_dim);
            VectorXi dom_starts(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];                                                    // need Eigen vector from STL vector
            }
            MatrixX<T> Q(ndom_pts.prod(), pt_dim);
            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts());
            while (!dom_iter.done())
            {
                Q.block(dom_iter.cur_iter(), 0, 1, pt_dim) =
                    input.domain.block(dom_iter.sub_full_idx(dom_iter.cur_iter()), mfa_data.min_dim, 1, pt_dim);
                dom_iter.incr_iter();
            }

            // resize control points and weights in case number of control points changed
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());
            t.weights = VectorX<T>::Ones(t.weights.size());                                         // linear solve does not solve for weights; set to 1

            // timing
            q_time                  = MPI_Wtime() - q_time;
            double free_time        = MPI_Wtime();

            // matrix of free control point basis functions

            // a dense matrix copied to a sparse matrix just prior to solving
            // To fill sparse matrix directly, skipping dense matrix, we would need to use row-major order
            // to do row-sum normalization efficiently later. However, row-major is 2X slower to fill
            // and 3-4X slower to solve than column-major. Hence, for now, we fill a dense matrix,
            // do the row normalization, then copy dense matrix to column-major sparse matrix just before solving.

            MatrixX<T> Nfree;
            int max_nnz_col = FreeCtrlPtMat(t_idx, ndom_pts, dom_starts, Nfree);                    // returns max num nonzeros in any column

            // timing
            free_time           = MPI_Wtime() - free_time;
            double cons_time    = MPI_Wtime();

            // find constraint control points and their anchors
            MatrixX<T>                  Pcons;                                                      // constraint control points
            vector<vector<KnotIdx>>     anchors;                                                    // corresponding anchors
            vector<TensorIdx>           t_idx_anchors;                                              // tensors containing corresponding anchors

#ifndef MFA_NO_CONSTRAINTS      // for debugging can disable constraints; normally not used

            LocalSolveAllConstraints(t, Pcons, anchors, t_idx_anchors);

#endif

            MatrixX<T> Ncons = MatrixX<T>::Constant(Q.rows(), Pcons.rows(), -1);                    // basis functions, -1 means unassigned so far
            if (Pcons.rows())
                ConsCtrlPtMat(ndom_pts, dom_starts, anchors, t_idx_anchors, Ncons);

            // timing
            cons_time           = MPI_Wtime() - cons_time;

            // normalize Nfree and Ncons such that the row sum of Nfree + Ncons = 1.0
            double norm_time    = MPI_Wtime();              // timing

#ifdef MFA_TBB

            static affinity_partitioner ap;
            parallel_for (blocked_range<size_t>(0, Nfree.rows()), [&] (blocked_range<size_t>& r)
            {
            for (auto i = r.begin(); i < r.end(); i++)

#else

            for (auto i = 0; i < Nfree.rows(); i++)

#endif

            {
                bool error = false;
                T sum = Nfree.row(i).sum();
                if (Pcons.rows())
                    sum += Ncons.row(i).sum();

                if (sum > 0.0)
                {
                    Nfree.row(i) /= sum;
                    if (Pcons.rows())
                        Ncons.row(i) /= sum;
                }
                else
                {
                    if (Pcons.rows())
                        fmt::print(stderr, "Warning: EncodeTensorLocalUnified(): row {} Nfree row sum = {} Ncons row sum = {}, Nfree + Ncons row sum = {}. This should not happen.\n",
                            i, Nfree.row(i).sum(), Ncons.row(i).sum(), sum);
                    else
                        fmt::print(stderr, "Warning: EncodeTensorLocalUnified(): row {} Nfree row sum = {}. This should not happen.\n",
                            i, sum);
                    error = true;
                }
                if (error)
                {
                    VectorXi ijk(dom_dim);
                    dom_iter.idx_ijk(i, ijk);
                    cerr << "ijk = " << ijk.transpose() << endl;
                    fmt::print(stderr, "params = [ ");
                    for (auto k = 0; k < dom_dim; k++)
                        fmt::print(stderr, "{} ", input.params->param_grid[k][ijk(k)]);
                    fmt::print(stderr, "]\n");
                }
            }

#ifdef MFA_TBB

            }, ap);

#endif

            norm_time = MPI_Wtime() - norm_time;                // timing

            // multiply by transpose to make the matrix square and smaller
            double mult_time    = MPI_Wtime();                  // timing
            MatrixX<T> NtNfree  = Nfree.transpose() * Nfree;
            mult_time           = MPI_Wtime() - mult_time;      // timing

// for comparing sparse with dense solve
// #define MFA_DENSE

#ifndef MFA_DENSE

            // compute max nonzero columns in NtNfree
            max_nnz_col = 0;
            for (auto i = 0; i < NtNfree.rows(); i++)
            {
                int nnz_col = 0;
                for (auto j = 0; j < NtNfree.cols(); j++)
                {
                    if (NtNfree(i, j) != 0.0)
                        nnz_col++;
                }
                if (nnz_col > max_nnz_col)
                    max_nnz_col = nnz_col;
            }

            // copy from dense to sparse
            SparseMatrixX<T> NtNfree_sparse(NtNfree.rows(), NtNfree.cols());
            NtNfree_sparse.reserve(VectorXi::Constant(NtNfree.cols(), max_nnz_col));
            for (auto i = 0; i < NtNfree.rows(); i++)
            {
                for (auto j = 0; j < NtNfree.cols(); j++)
                {
                    if (NtNfree(i, j) != 0.0)
                        NtNfree_sparse.insert(i, j) = NtNfree(i,j);
                }
            }
            NtNfree_sparse.makeCompressed();

#endif

            // timing
            double r_time       = MPI_Wtime();

            // R is the right hand side needed for solving N * P = R
            MatrixX<T> R = Q;
            if (Pcons.rows())
                R -= Ncons * Pcons;

            // timing
            r_time                  = MPI_Wtime() - r_time;
            setup_time              = MPI_Wtime() - setup_time;

            fmt::print(stderr, "Solving...\n");

            // for debugging, compute condition number of NtNfree
            // unfortunately SVD takes too long when the matrix is ill-conditioned
            // ref: https://forum.kde.org/viewtopic.php?f=74&t=117430
//             Eigen::JacobiSVD<Eigen::MatrixXd> svd(NtNfree);
//             double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
//             fmt::print(stderr, "NtNfree has condition number {}\n", cond);


#ifdef MFA_DENSE    // dense solve

            double dense_solve_time = MPI_Wtime();                  // timing
            t.ctrl_pts = (Nfree.transpose() * Nfree).ldlt().solve(Nfree.transpose() * R);
            dense_solve_time = MPI_Wtime() - dense_solve_time;

#else               // sparse solve

            double sparse_solve_time    = MPI_Wtime();              // timing

            Eigen::SparseQR<SparseMatrixX<T>, Eigen::COLAMDOrdering<int>>  solver;

            // TODO: iterative least squares conjugate gradient is faster but sometimes fails
            // NtN not necessarily symmetric positive definite?
//             Eigen::LeastSquaresConjugateGradient<SparseMatrixX<T>>  solver;

            solver.compute(NtNfree_sparse);

            if (solver.info() != Eigen::Success)
            {
                cerr << "EncodeTensorLocalUnified(): Error: Matrix decomposition failed" << endl;
                abort();
            }

            t.ctrl_pts = solver.solve(Nfree.transpose() * R);
            if (solver.info() != Eigen::Success)
            {
                cerr << "EncodeTensorLocalUnified(): Error: Least-squares solve failed" << endl;
                abort();
            }

            sparse_solve_time = MPI_Wtime() - sparse_solve_time;

#endif

            // timing
            fmt::print(stderr, "EncodeTensorLocalUnified() timing:\n");
            fmt::print(stderr, "setup time: {} s.\n", setup_time);
            fmt::print(stderr, "    = free time {} + cons time {} + mult time {} s.\n",
                    free_time, cons_time, mult_time);
//             fmt::print(stderr, "    = q time {} + free time {} + cons time {} + norm time {} + mult time {} r time {} s.\n",
//                     q_time, free_time, cons_time, norm_time, mult_time, r_time);
//             fmt::print(stderr, "free_time {} = free_iter_time {} + dom_iter_time {} s.\n",
//                     free_time, free_iter_time, dom_iter_time);

#ifdef MFA_DENSE
            fmt::print(stderr, "dense_solve time: {} s.\n", dense_solve_time);
#else
            fmt::print(stderr, "sparse_solve time: {} s.\n", sparse_solve_time);
#endif

            // debug: check relative error of solution
            double relative_error = (Nfree * t.ctrl_pts - R).norm() / R.norm(); // norm() is L2 norm
            cerr << "EncodeTensorLocalUnified(): The relative error is " << relative_error << endl;
        }

        // encodes the control points for one tensor product of a tmesh
        // takes a subset of input points from the global domain, covered by basis functions of this tensor product
        // solves dimensions separably
        // does not encode weights for now
        // linear constrained formulation as proposed by David Lenz (see wiki/notes/linear-constrained-fit.pdf)
        void EncodeTensorLocalSeparable(
                TensorIdx                 t_idx,                  // index of tensor product being encoded
                bool                      weighted = true)        // solve for and use weights
        {
            // additional constraints
            // it turns out these don't change anything, leave set to 0
            int extra_cons = 0;

            double t0 = MPI_Wtime();

            // typing shortcuts
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;

            auto& t = tensor_prods[t_idx];                                                          // current tensor product
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                                   // control point dimensionality

            // get input domain points covered by the tensor, including constraints
            vector<size_t> start_idxs(dom_dim);                                                     // start of input points including constraints
            vector<size_t> end_idxs(dom_dim);                                                       // end of input points including constraints
//             mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, true, extra_cons, start_idxs, end_idxs); // true: extend to cover constraints
            mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, false, extra_cons, start_idxs, end_idxs); // false: do not extend to cover constraints

            // relevant input domain points covering constraints
            VectorXi ndom_pts(dom_dim);
            VectorXi dom_starts(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];                                                    // need Eigen vector from STL vector
            }
 
            // resize control points and weights in case number of control points changed
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());
            t.weights = VectorX<T>::Ones(t.weights.size());                                         // linear solve does not solve for weights; set to 1

            // max size of input point matrices is the product of the max of input points and control points in each dim
            size_t max_npts = 1;
            for (auto k = 0; k < dom_dim; k++)
                max_npts *= std::max(ndom_pts(k), t.nctrl_pts(k));

            // two matrices of input points for subsequent dimensions after dimension 0, which draws directly from input domain
            // input points cover constraints as well as free control point basis functions
            // (double buffering output control points to input points)
            MatrixX<T> Q(max_npts, pt_dim);
            MatrixX<T> Q1(max_npts, pt_dim);

            // number of output points in current dimension and number of input points of next dimension
            VectorXi npts = ndom_pts;
            npts(0) = t.nctrl_pts(0);

            // input and output number of points
            VectorXi nin_pts    = ndom_pts;
            VectorXi nout_pts   = npts;
            VectorXi in_starts  = dom_starts;
            VectorXi in_all_pts = input.ndom_pts();
            VectorXi start_ijk  = in_starts;

            // timing
            double free_time    = 0.0;
            double cons_time    = 0.0;
            double norm_time    = 0.0;
            double solve_time   = 0.0;

            for (auto dim = 0; dim < dom_dim; dim++)                                                // for all domain dimensions
            {
                MatrixX<T> R(nin_pts(dim), pt_dim);                                                 // RHS for solving N * P = R

                VolIterator     in_iter(nin_pts, in_starts, in_all_pts);                            // volume of current input points
                VolIterator     out_iter(nout_pts);                                                 // volume of current output points
                SliceIterator   in_slice_iter(in_iter, dim);                                        // slice of the input points volume missing current dim
                SliceIterator   out_slice_iter(out_iter, dim);                                      // slice of the output points volume missing current dim

                // allocate matrices of free and constraint control points and constraint basis functions
                MatrixX<T>  Nfree(nin_pts(dim), t.nctrl_pts(dim));
                ConsType    cons_type;                                                              // none, left, right, both
                MatrixX<T>  Ncons;
                MatrixX<T>  Pcons;
                MatrixX<T>  P(t.nctrl_pts(dim), pt_dim);

                // debug: turn off constraints

#ifdef MFA_NO_CONSTRAINTS

                cons_type = ConsType::MFA_NO_CONSTRAINT;
                int ncons = 0;

#else       // MFA_NO_CONSTRAINTS

                int ncons = mfa_data.p(dim) + extra_cons;

                if (t.knot_mins[dim] == 0 && t.knot_maxs[dim] == tmesh.all_knots[dim].size() - 1)
                    cons_type = ConsType::MFA_NO_CONSTRAINT;

                vector<TensorIdx>       t_idx_anchors;                                                  // tensors containing corresponding anchors
                vector<vector<KnotIdx>> anchors;                                                        // corresponding anchors
                VectorXi                ncurve_pts = ndom_pts;

                if (dim > 0)
                    cons_type = ConsType::MFA_NO_CONSTRAINT;
                else if (dim == 0 && t.knot_mins[dim] == 0)
                    cons_type   = ConsType::MFA_RIGHT_ONLY_CONSTRAINT;
                else if (dim == 0 && t.knot_maxs[dim] == tmesh.all_knots[dim].size() - 1)
                    cons_type   = ConsType::MFA_LEFT_ONLY_CONSTRAINT;
                else if (dim == 0)
                    cons_type   = ConsType::MFA_BOTH_CONSTRAINT;

                if (dim == 0)
                {
                    // fill Pcons
                    LocalSolveAllConstraints(t, Pcons, anchors, t_idx_anchors);

                    // number of input points in one curve
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        if (j != dim)
                            ncurve_pts[j] = 1;
                    }
                }

#endif      // MFA_NO_CONSTRAINTS

                // for all curves in the current dimension
                while (!in_slice_iter.done())                                                       // for all curves
                {

                    CurveIterator   in_curve_iter(in_slice_iter);                                   // one curve of the input points in the current dim
                    start_ijk = in_curve_iter.cur_ijk();
                    // add original input point starting offsets to higher dims start_ijk
                    // but don't offset the previous dims: start_ijk for previous dims refers to control points, not input points
                    if (dim > 0)
                    {
                        for (auto j = dim; j < dom_dim; j++)
                            start_ijk(j) += dom_starts(j);
                    }

                    if (CurveIntersectsTensor(t_idx, dim, nin_pts, start_ijk))
                    {
                        // fill Ncons
                        if (dim == 0 && Pcons.rows())
                        {
                            Ncons = MatrixX<T>::Constant(ncurve_pts[0], Pcons.rows(), -1);                    // basis functions, -1 means unassigned so far
                            ConsCtrlPtMat(ncurve_pts, start_ijk, anchors, t_idx_anchors, Ncons);
                        }

                        ComputeCtrlPtCurve(in_curve_iter, t_idx, dim, R, Q, Q1, Nfree, Ncons, Pcons, P,
                                cons_type, nin_pts, start_ijk, ncons, free_time, cons_time, norm_time, solve_time);
                    }

                    // copy solution to one curve of output points
                    CurveIterator   out_curve_iter(out_slice_iter);                                 // one curve of the output points in the current dim
                    while (!out_curve_iter.done())
                    {
                        if (dim % 2 == 0)
                            Q1.row(out_curve_iter.ijk_idx(out_curve_iter.cur_ijk())) = P.row(out_curve_iter.cur_iter());
                        else
                            Q.row(out_curve_iter.ijk_idx(out_curve_iter.cur_ijk())) = P.row(out_curve_iter.cur_iter());
                        out_curve_iter.incr_iter();
                    }
                    out_slice_iter.incr_iter();
                    in_slice_iter.incr_iter();
                }       // for all curves in current dim

                // adjust input, output numbers of points for next iteration
                nin_pts(dim) = t.nctrl_pts(dim);
                if (dim < dom_dim - 1)
                    nout_pts(dim + 1)   = t.nctrl_pts(dim + 1);
                in_starts   = VectorXi::Zero(dom_dim);                  // subvolume = full volume for later dims
                in_all_pts  = nin_pts;
            }       // for all domain dimensions

            // copy final result back to tensor product
            if (dom_dim % 2 == 0)
                t.ctrl_pts = Q.block(0, 0, t.nctrl_pts.prod(), pt_dim);
            else
                t.ctrl_pts = Q1.block(0, 0, t.nctrl_pts.prod(), pt_dim);

            // timing
            fmt::print(stderr, "EncodeTensorLocalSeparable() tidx {} time {:.3e} s.\n", t_idx, MPI_Wtime() - t0);
        }

#endif  // MFA_TMESH

#ifdef MFA_LOW_D

        // original adaptive encoding for first tensor product only
        // older version using 1-d curves for new knots
        void OrigAdaptiveEncode(
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)         // optional maximum number of rounds
        {
            vector<vector<T>> new_knots;                               // new knots in each dim.

            // debug
            if (verbose >= 2)
            {
                fmt::print(stderr, "Using OrigAdaptiveEncode() w/ 1-d curve knot splitting\n");
            }
            if (verbose >= 3)
            {
#ifdef MFA_CHECK_ALL_CURVES
                fmt::print(stderr, "Checking all curves (slower but more accurate)\n");
#else
                fmt::print(stderr, "Checking a sampling of curves (faster but less accurate)\n");
#endif
            }


            // TODO: use weights for knot insertion
            // for now, weights are only used for final full encode

            // loop until no change in knots
            for (int iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                {
                    if (verbose >= 2)
                        fprintf(stderr, "\nDone; max iterations reached.\n\n");
                    break;
                }

                if (verbose >= 2)
                    fprintf(stderr, "--- Iteration %d ---\n", iter);

                // low-d w/ splitting spans in the middle
                bool done = OrigNewKnots_curve(new_knots, err_limit, extents, iter);

                // no new knots to be added
                if (done)
                {
                    if (verbose >= 2)
                        fprintf(stderr, "\nDone; no new knots added.\n\n");
                    break;
                }

                // check if the new knots would make the number of control points >= number of input points in any dim
                done = false;
                for (auto k = 0; k < dom_dim; k++)
                    // hard-coded for first tensor
                    if (input.ndom_pts(k) <= mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + new_knots[k].size())
                    {
                        done = true;
                        break;
                    }
                if (done)
                {
                    if (verbose >= 2)
                        fprintf(stderr, "\nDone; control points would outnumber input points.\n\n");
                    break;
                }

                // debug
//                 mfa_data.tmesh.print();
            }

            // final full encoding needed after last knot insertion above
            if (verbose >= 2)
                fmt::print("Performing final encoding\n");
            TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];        // fixed encode assumes the tmesh has only one tensor product
            Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);
        }

#else       // full-dimensional knot insertion

        // original adaptive encoding for first tensor product only
        // latest version using full knot spans (5/19/21)
        void OrigAdaptiveEncode(
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)         // optional maximum number of rounds
        {
            vector<vector<T>> new_knots;                               // new knots in each dim.
            ErrorStats<T> error_stats;

            // debug
            fmt::print(stderr, "Using OrigAdaptiveEncode() w/ full-d knot splitting\n\n");

            VectorX<T> myextents = extents.size() ? extents : VectorX<T>::Ones(mfa_data.tmesh.tensor_prods[0].ctrl_pts.cols());

            mfa::NewKnots<T> nk(mfa_data, input);

            // loop until no change in knots
            for (int iter = 0; ; iter++)
            {
                // encode tensor product 0
                TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];
                for (auto j = 0; j < dom_dim; j++)
                    t.nctrl_pts[j] = mfa_data.tmesh.all_knots[j].size() - mfa_data.p(j) - 1;
                Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);

                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; max iterations reached.\n\n");
                    break;
                }

                if (verbose)
                    fprintf(stderr, "\n--- Iteration %d ---\n", iter);

                // check all knots spans for error
                vector<vector<KnotIdx>>     inserted_knot_idxs(dom_dim);   // indices in each dim. of inserted knots in full knot vector after insertion
                vector<vector<T>>           inserted_knots(dom_dim);       // knots to be inserted in each dim.
                vector<TensorIdx>           parent_tensor_idxs;                     // tensors having knots inserted
                vector<TensorProduct<T>>    new_tensors;                            // empty, unused
                bool done = nk.AllErrorSpans(
                        0,
                        myextents,
                        err_limit,
                        true,
                        new_tensors,
                        parent_tensor_idxs,
                        inserted_knot_idxs,
                        inserted_knots,
                        error_stats);

                // no new knots to be added
                if (done)
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; no new knots added.\n\n");
                    break;
                }

                // insert new knots into knot vectors
                int n_insertions = parent_tensor_idxs.size();                       // number of knots to insert
                for (auto j = 0; j < dom_dim; j++)
                    assert(inserted_knot_idxs[j].size() == n_insertions &&
                            inserted_knots[j].size() == n_insertions);

                vector<bool> inserted(dom_dim);                            // whether the current insertion succeed (in each dim)

                for (auto i = 0; i < n_insertions; i++)                             // for all knots to be inserted
                {
                    // insert the new knots into tmesh all_knots
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        inserted[j] = false;
                        if (mfa_data.tmesh.insert_knot(
                                    j,
                                    0,                                              // all knots at level 0 in this version
                                    inserted_knots[j][i],
                                    input.params->param_grid,
                                    inserted_knot_idxs[j][i]))
                        {
                            inserted[j] = true;
                            // increment subsequent insertions
                            for (auto k = 0; k < n_insertions; k++)
                            {
                                if (inserted_knot_idxs[j][k] > inserted_knot_idxs[j][i])
                                    inserted_knot_idxs[j][k]++;
                            }
                        }
                    }   // dimension
                }   // knot insertions

                if (verbose)
                    PrintAdaptiveStats(error_stats);
            }   // iterations
        }

#endif

        // adaptive encoding for T-mesh
        void AdaptiveEncode(
                T                   err_limit,                          // maximum allowable normalized error
                bool                weighted,                           // solve for and use weights
                const VectorX<T>&   extents,                            // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)                     // optional maximum number of rounds
        {
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;
            int parent_level = 0;                                       // parent level currently being refined

            // timing
            double t0 = MPI_Wtime();

            TensorProduct<T>&t = tensor_prods[0];                       // fixed encode assumes the tmesh has only one tensor product
            Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);

            // timing
            fmt::print(stderr, "\nInitial full encode time:   {:.3e} s.\n", MPI_Wtime() - t0);

            // debug: print tmesh
//             fprintf(stderr, "\n----- initial T-mesh -----\n\n");
//             tmesh.print();
//             fprintf(stderr, "--------------------------\n\n");

            vector<TensorProduct<T>>    new_tensors;                    // newly refined tensors to be added

            // intersection proximity between tensors (assumes same for all dims)
            VectorXi& p = mfa_data.p;

            // padding for all tensors
            // following was found empirically to be best: even p: pad = p + 1; odd p: pad = p + 2
            // only tested p=2 and p=3, however
            int pad         = p(0) % 2 == 0 ? p(0) + 1 : p(0) + 2;
//             int pad         = p(0) % 2 == 0 ? p(0) + 0 : p(0) + 1;

            // tightest pad possible, leaving only p control points in a tensor
            int tightest_pad = (p(0) % 2 == 0 ? p(0) : p(0) - 1);

            // extra padding for tensor at global edge
            int edge_pad    = (p(0) / 2) + 2;

            // loop until all tensors are done
            int iter;
            for (iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                    break;

                if (verbose)
                    fmt::print(stderr, "\n--- Iteration {}, refining level {} ---\n", iter, parent_level);

                // debug
//                 fmt::print(stderr, "\nTmesh before refinement\n\n");
//                 tmesh.print(true, true, false, false);

                Refine(parent_level, iter + 1, err_limit, extents, pad, edge_pad, new_tensors);

                UpdateExistingTensors(new_tensors);

                // check if number of control points exceeds input points in any dimension (in the case of a single tensor)
                if (tensor_prods.size() == 1)
                {
                    bool done = false;
                    int k;
                    for (k = 0; k < mfa_data.dom_dim; k++)
                    {
                        if (input.ndom_pts(k) <= tensor_prods[0].nctrl_pts(k))
                        {
                            done = true;
                            break;
                        }
                    }
                    if (done)
                    {
                        if (verbose)
                            fmt::print(stderr, "\nKnot insertion done after {} iterations; Control points in dim. {} is {} which is the max.\n\n",
                                    iter + 1, k, tensor_prods[0].nctrl_pts(k));
                        break;
                    }
                }
                // check if all levels are done
                if (parent_level >= tmesh.max_level)
                {
                    if (verbose)
                        fmt::print(stderr, "\nKnot insertion done after {} iterations; no new knots added.\n\n", iter + 1);
                    break;
                }
                else                                                // one iteration only is done
                {
                    CheckNewTensors(new_tensors, pad, edge_pad);
                    fmt::print(stderr, "Level {} done, adding {} new tensor(s)\n", parent_level, new_tensors.size());
                    double add_tensors_time = MPI_Wtime();
                    bool add_success = AddNewTensors(new_tensors, tightest_pad);
                    add_tensors_time = MPI_Wtime() - add_tensors_time;
                    fmt::print(stderr, "Solving and adding new tensors time:   {:.3e} s.\n", add_tensors_time);
                    if (add_success)
                        parent_level++;
                    new_tensors.clear();
                }
            }   // iterations

            fmt::print(stderr, "{} iterations.\n", iter + 1);
            fmt::print(stderr, "{} tensor products.\n", tensor_prods.size());

            // debug: print tmesh
//             fmt::print(stderr, "\n----- final T-mesh -----\n\n");
//             tmesh.print(true, true, false, false);
//             fmt::print(stderr, "--------------------------\n\n");

            // debug: check all spans
            // TODO: comment out after code is debugged
            mfa::NewKnots<T> nk(mfa_data, input);
            if (!nk.CheckAllSpans())
                throw MFAError(fmt::format("AdaptiveEncode(): Error: failed checking all spans for input points\n"));

            // debug: check if all tensors are marked done
            // TODO: comment out after code is debugged
            bool all_done = true;
            for (auto i = 0; i < tensor_prods.size(); i++)
            {
                bool first = true;
                auto& t = tensor_prods[i];
                if (!t.done)
                {
                    if (first)
                        fmt::print(stderr,"\n");
                    fmt::print(stderr, "Tensor {} level {} is not marked done.\n", i, t.level);
                    fmt::print(stderr, "This is normal if the number of rounds is capped; otherwise this should not happen.\n");
                    all_done    = false;
                    first       = false;
                }
            }
            if (all_done)
                fmt::print(stderr, "All tensors are marked as done.\n");

            // debug: confirm that all tensors will have at least p control points
            // TODO: comment out once the code is debugged
            for (auto i = 0; i < tensor_prods.size(); i++)
            {
                if (!tmesh.check_num_ctrl_degree(i, 0))
                {
                    auto& t = tensor_prods[i];
                    fmt::print(stderr, "Error: Refine(): tensor {} has fewer than p control points. This should not happen\n", i);
                    fmt::print(stderr, "knot_mins [{}] knot_maxs[{}] level {}\n",
                            fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level);
                    fmt::print(stderr, "\nAddNewTensors(): T-mesh:\n");
                    tmesh.print(true, true, false, false);
                    abort();
                }
            }

            // debug: verify that the local knots stored in all tensors correspond to the global knots
            // TODO: comment out after code is debugged
            if (!tmesh.check_local_knots())
                throw MFAError(fmt::format("AdaptiveEncode(): Error: failed checking local knots of tensors\n"));

            // debug: print tmesh
//             fmt::print(stderr, "\n----- final T-mesh -----\n\n");
//             tmesh.print(false, true, false, false);
//             fmt::print(stderr, "--------------------------\n\n");
        }

    private:

        // print max error, compression factor, and any pother stats at the end of an adaptive iteration
        void PrintAdaptiveStats(
                const ErrorStats<T>&    error_stats)
        {
            // compute compression
            float in_coords = (input.npts) * (input.pt_dim);
            float out_coords = 0.0;
            for (auto i = 0; i < mfa_data.tmesh.tensor_prods.size(); i++)
                out_coords += mfa_data.tmesh.tensor_prods[i].ctrl_pts.rows() *
                    mfa_data.tmesh.tensor_prods[i].ctrl_pts.cols();
            for (auto j = 0; j < mfa_data.tmesh.all_knots.size(); j++)
                out_coords += mfa_data.tmesh.all_knots[j].size();
            float compression = in_coords / out_coords;

            T rms_abs_err = sqrt(error_stats.sum_sq_abs_errs / (input.domain.rows()));
            T rms_norm_err = sqrt(error_stats.sum_sq_norm_errs / (input.domain.rows()));

            fprintf(stderr, "\n----- estimates of current variable of current model -----\n");
            fprintf(stderr, "estimated max_err               = %e\n",  error_stats.max_abs_err);
            fprintf(stderr, "estimated normalized max_err    = %e\n",  error_stats.max_norm_err);
            fprintf(stderr, "estimated RMS error             = %e\n",  rms_abs_err);
            fprintf(stderr, "estimated normalized RMS error  = %e\n",  rms_norm_err);
            fprintf(stderr, "estimated compression ratio     = %.2f\n",  compression);
            fprintf(stderr, "-----------------------------------------------------------\n");
        }

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
            int last   = R.cols() - 1;                                          // column of range value TODO: weighing only the last column does not make much sense in the split model
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);   // one row for each input point
            VectorX<T> denom(N.rows());                                         // rational denomoninator for param of each input point

            for (int k = 0; k < N.rows(); k++)                                  // for all input points
            {
                denom(k) = (N.row(k).cwiseProduct(weights.transpose())).sum();
#ifdef UNCLAMPED_KNOTS
                if (denom(k) == 0.0)
                    denom(k) = 1.0;
#endif
                Rk.row(k) = input.domain.block(co + k * input.g.ds[cur_dim], mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
            }

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         weights(i) / denom.array() *                           // rationalized
                         Rk.col(j).array()).sum();                              // input points
#else                                                                           // don't weigh domain coordinate (only range)
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
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         Rk.col(j).array()).sum();                              // input points
                R(i, last) =
                    (N.col(i).array() *                                         // ith basis functions for input pts
                     weights(i) / denom.array() *                               // rationalized
                     Rk.col(last).array()).sum();                               // input points
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
            int last   = R.cols() - 1;                                          // column of range value TODO: weighing only the last column does not make much sense in the split model
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);   // one row for each input point
            VectorX<T> denom(N.rows());                                         // rational denomoninator for param of each input point

            for (int k = 0; k < N.rows(); k++)
            {
                denom(k) = (N.row(k).cwiseProduct(weights.transpose())).sum();
                Rk.row(k) = in_pts.row(co + k * cs);
            }

#ifdef WEIGH_ALL_DIMS                                                           // weigh all dimensions
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         weights(i) / denom.array() *                           // rationalized
                         Rk.col(j).array()).sum();                              // input points
#else                                                                           // don't weigh domain coordinate (only range)
            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
            {
                // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                for (int j = 0; j < R.cols() - 1; j++)
                    R(i, j) =
                        (N.col(i).array() *                                     // ith basis functions for input pts
                         Rk.col(j).array()).sum();                              // input points
                R(i, last) =
                    (N.col(i).array() *                                         // ith basis functions for input pts
                     weights(i) / denom.array() *                               // rationalized
                     Rk.col(last).array()).sum();                               // input points
            }
#endif

            // debug
            //     cerr << "R:\n" << R << endl;
        }

        // computes right hand side vector for encoding a tensor product in unified-dimensional form
        // NOTE: Does not support weights
        void RHSUnified(
            // const vector<size_t>&   start_idxs,
            // const vector<size_t>&   end_idxs,
            SparseMatrixX<T>&       Nt,
            MatrixX<T>&             R)
        {
            // REQUIRED for TMesh
            // VectorXi ndom_pts(dom_dim);
            // VectorXi dom_starts(dom_dim);
            // for (auto k = 0; k < dom_dim; k++)
            // {
            //     ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
            //     dom_starts(k)   = start_idxs[k];
            // }

            if (R.cols() != mfa_data.dim())
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (cols)" << endl;
            if (R.rows() != input.npts)
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (rows)" << endl;

            VectorX<T> pt_coords(mfa_data.dim());
            for (auto input_it = input.begin(); input_it != input.end(); ++input_it)
            {
                // extract coordinates in dimension min_dim<-->max_dim and place in pt_coords
                input_it.coords(pt_coords, mfa_data.min_dim, mfa_data.max_dim);
                R.row(input_it.idx()) = pt_coords;
            }

            R = Nt * R;
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
                MatrixX<T>&             R)          // (output) residual matrix allocated by caller
        {
            VectorXi ndom_pts(dom_dim);
            VectorXi dom_starts(dom_dim);
            for (auto k = 0; k < dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];
            }

            // fill Rk, the matrix of input points
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);           // one row for each input point
            VolIterator vol_iter(ndom_pts, dom_starts, input.ndom_pts());                 // iterator over input points
            VectorXi ijk(dom_dim);
            while (!vol_iter.done())
            {
                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);
                Rk.row(vol_iter.cur_iter()) = input.domain.block(vol_iter.ijk_idx(ijk), mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
                vol_iter.incr_iter();
            }

            // compute the matrix R (one row for each control point)
            for (int i = 0; i < N.cols(); i++)
                for (int j = 0; j < R.cols(); j++)
                    // using array() for element-wise multiplication, which is what we want (not matrix mult.)
                    R(i, j) =
                        (N.col(i).array() *                 // ith basis functions for input pts
                         Rk.col(j).array()).sum();          // input points

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
            if (mfa_data.p.size() != input.ndom_pts().size())
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
                if (nctrl_pts(i) > input.ndom_pts(i))
                {
                    fprintf(stderr, "Warning: Encode() number of control points (%d) in dimension %ld "
                            "exceeds number of input data points (%d) in dimension %ld.\n", nctrl_pts(i), i, input.ndom_pts(i), i);
                }
            }

            n.resize(mfa_data.p.size());
            m.resize(mfa_data.p.size());
            for (size_t i = 0; i < mfa_data.p.size(); i++)
            {
                n(i)        =  nctrl_pts(i) - 1;
                m(i)        =  input.ndom_pts(i)  - 1;
            }
        }

        // solves for one curve of control points -- no weights (B-Spline encode only)
        // outputs go to specified control points and weights matrix and vector rather than default mfa
        void ctrlCurveBSpline(
            CurveIterator&          in_curve_iter,                  // current curve
            MatrixX<T>&             R,                              // right hand side, allocated by caller
            MatrixX<T>&             Q0,                             // first matrix of input points, allocated by caller
            MatrixX<T>&             Q1,                             // second matrix of input points, allocated by caller
            const MatrixX<T>&       Nt,                             // transpose of collocation matrix
            Eigen::LLT<MatrixX<T>>& llt,                            // LLT factorization of Nt*N
            MatrixX<T>&             P)                              // (output) control points
        {
            int dim = in_curve_iter.curve_dim();

            // copy one curve of input points to right hand side
            // add zero entries for rows that will correspond to deriv constraints
            while (!in_curve_iter.done())
            {
                int curve_idx = in_curve_iter.cur_iter();
                int vol_idx = in_curve_iter.cur_iter_full();    // point index. One for each grid point on this curve
                
                if (dim % 2 == 0)
                    R.row(curve_idx) = Q0.row(vol_idx);
                else
                    R.row(curve_idx) = Q1.row(vol_idx);
                
                in_curve_iter.incr_iter();
            }
            in_curve_iter.reset();

            P = llt.solve(Nt * R);
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
            Q.resize(input.ndom_pts(k), ctrl_pts.cols());
            if (k == 0)
            {
                for (auto i = 0; i < input.ndom_pts(k); i++)
                    Q.row(i) = input.domain.block(co + i * cs, mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
            }
            else if (k % 2)
            {
                for (auto i = 0; i < input.ndom_pts(k); i++)
                    Q.row(i) = temp_ctrl0.row(co + i * cs);
            }
            else
            {
                for (auto i = 0; i < input.ndom_pts(k); i++)
                    Q.row(i) = temp_ctrl1.row(co + i * cs);
            }

            VectorX<T> temp_weights = VectorX<T>::Ones(N.cols());

#ifndef MFA_NO_WEIGHTS

            if (weighted)
                if (k == dom_dim - 1)                               // only during last dimension of separable iteration over dimensions
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
            if (k == dom_dim - 1)
            {
                for (auto i = 0; i < temp_weights.size(); i++)
                    weights(to + i * cs) = temp_weights(i);
            }
        }

        // Copy individual curve of control points to intermediate volumes
        // For use with dimension-separable encode
        void copyCtrl(
            CurveIterator&          outputCurveIter,                // curve in output volume
            const MatrixX<T>&       P,                              // control points to copy
            MatrixX<T>&             Q0,                             // first matrix of output points
            MatrixX<T>&             Q1)                             // second matrix of output points
        {
            MatrixX<T>& output = outputCurveIter.curve_dim() % 2 == 0 ? Q1 : Q0;
            while (!outputCurveIter.done())
            {
                // Copy control point in P to corresponding output row
                output.row(outputCurveIter.cur_iter_full()) = P.row(outputCurveIter.cur_iter());
                
                // Move along the curve
                outputCurveIter.incr_iter();
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
            int ndims = dom_dim;

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
            int ndims = dom_dim;

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
            mfa::Decoder<T> decoder(mfa_data, verbose);
            int pt_dim = tensor.ctrl_pts.cols();            // control point dimensonality
            VectorX<T> cpt(pt_dim);                         // decoded curve point
            int nerr = 0;                                   // number of points with error greater than err_limit
            int span = mfa_data.p[k];                            // current knot span of the domain point being checked
            if (!extents.size())
                extents = VectorX<T>::Ones(input.domain.cols());

            for (auto i = 0; i < input.ndom_pts(k); i++)      // all domain points in the curve
            {
                while (mfa_data.tmesh.all_knots[k][span + 1] < 1.0 && mfa_data.tmesh.all_knots[k][span + 1] <= input.params->param_grid[k][i])
                    span++;

                decoder.CurvePt(k, input.params->param_grid[k][i], ctrl_pts, weights, tensor, cpt);


                // error
                T max_err = 0.0;
                for (auto j = 0; j < mfa_data.max_dim - mfa_data.min_dim + 1; j++)
                {
                    T err = fabs(cpt(j) - input.domain(co + i * input.g.ds[k], mfa_data.min_dim + j)) / extents(mfa_data.min_dim + j);
                    max_err = err > max_err ? err : max_err;
                }

                if (max_err > err_limit &&
                        // don't allow more control points than input points
                        mfa_data.tmesh.tensor_prods.size() == 1 &&
                        tensor.nctrl_pts(k) + err_spans.size() <= input.ndom_pts(k))
                {
                    // debug
//                     fmt::print(stderr, "ErrorCurve(): 1: dim {} i {} max_err {}\n", k, i, max_err);

                    // don't duplicate spans
                    set<int>::iterator it = err_spans.find(span);
                    if (!err_spans.size() || it == err_spans.end())
                    {
                        // ensure there would be a domain point in both halves of the span if it were split
                        bool split_left = false;
                        for (auto j = i; input.params->param_grid[k][j] >= mfa_data.tmesh.all_knots[k][span]; j--)
                            if (input.params->param_grid[k][j] < (mfa_data.tmesh.all_knots[k][span] + mfa_data.tmesh.all_knots[k][span + 1]) / 2.0)
                            {
                                split_left = true;
                                break;
                            }
                        bool split_right = false;
                        for (auto j = i; input.params->param_grid[k][j] < mfa_data.tmesh.all_knots[k][span + 1]; j++)
                            if (input.params->param_grid[k][j] >= (mfa_data.tmesh.all_knots[k][span] + mfa_data.tmesh.all_knots[k][span + 1]) / 2.0)
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

#ifdef MFA_TMESH

        // refines a T-mesh at a given parent level
        // returns true no change in knots; all tensors at the parent level are done
        bool Refine(
                int                         parent_level,                       // level of parent tensors to refine
                int                         child_level,                        // level of children to create
                T                           err_limit,                          // max allowable error
                const VectorX<T>&           extents,                            // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                         pad,                                // padding (per side) for all tensors
                int                         edge_pad,                           // extra padding (per side) for tensor at the global edge
                vector<TensorProduct<T>>&   new_tensors)                        // (output) new tensors scheduled to be added
        {
            // typing shortcuts
            Tmesh<T>&                   tmesh                   = mfa_data.tmesh;
            vector<vector<T>>&          all_knots               = tmesh.all_knots;
            vector<vector<int>>&        all_knot_levels         = tmesh.all_knot_levels;
            vector<vector<ParamIdx>>&   all_knot_param_idxs     = tmesh.all_knot_param_idxs;
            vector<TensorProduct<T>>&   tensor_prods            = tmesh.tensor_prods;
            VectorXi&                   p                       = mfa_data.p;

            vector<vector<KnotIdx>>     inserted_knot_idxs(dom_dim);            // indices in each dim. of inserted knots in full knot vector after insertion
            vector<vector<T>>           inserted_knots(dom_dim);                // knots to be inserted in each dim.
            vector<TensorIdx>           parent_tensor_idxs;                     // tensors having knots inserted

            VectorX<T> myextents = extents.size() ? extents : VectorX<T>::Ones(tensor_prods[0].ctrl_pts.cols());
            ErrorStats<T> error_stats;

            // find new knots
            mfa::NewKnots<T> nk(mfa_data, input);

            // vectors of new_nctrl_pts, new_ctrl_pts, new_weights, one instance for each knot to be inserted
            vector<VectorXi>    new_nctrl_pts;
            vector<MatrixX<T>>  new_ctrl_pts;
            vector<VectorX<T>>  new_weights;

            // timing
            double error_spans_time = MPI_Wtime();

            // check all knots spans for error
            bool done = nk.AllErrorSpans(
                    parent_level,
                    myextents,
                    err_limit,
                    false,
                    new_tensors,
                    parent_tensor_idxs,
                    inserted_knot_idxs,
                    inserted_knots,
                    error_stats);

            // debug
//             fmt::print(stderr, "Refine(): parent_level {} child_level {} n_insertions {} done {}\n",
//                     parent_level, child_level, parent_tensor_idxs.size(), done);

            if (done)                                                           // nothing inserted
                return true;

            int n_insertions = parent_tensor_idxs.size();                       // number of knots to insert
            for (auto j = 0; j < dom_dim; j++)
                assert(inserted_knot_idxs[j].size() == n_insertions &&
                        inserted_knots[j].size() == n_insertions);

            vector<bool> inserted(dom_dim);                                     // whether the current insertion succeeded (in each dim)

            // timing
            error_spans_time    = MPI_Wtime() - error_spans_time;
            double insert_time  = MPI_Wtime();

            // debug
//             fmt::print(stderr, "n_insertions {}\n", n_insertions);

            for (auto i = 0; i < n_insertions; i++)                             // for all knots to be inserted
            {
                // debug: check that parent tensor level matches refinement level
                // TODO: remove after code works
                if (tensor_prods[parent_tensor_idxs[i]].level > parent_level)
                    throw MFAError(fmt::format("Error: Refine(): insertion index {} with parent tensor idx {} "
                            "at level {}, but parent_level is {}. This should not happen.\n",
                            i, parent_tensor_idxs[i], tensor_prods[parent_tensor_idxs[i]].level, parent_level));

                // debug
//                 fmt::print(stderr, "inserted_knot_idxs[{}] = [{}, {}]\n", i, inserted_knot_idxs[0][i], inserted_knot_idxs[1][i]);

                // insert the new knot into tmesh all_knots
                // NB: insert_knot adjusts knot_mins, maxs of existing tensors, meaning
                // no further adjustment to existing tensors should be done here
                int retval;
                bool new_candidate = false;                                     // make new candidate tensor for this knot
                for (auto j = 0; j < dom_dim; j++)
                {
                    inserted[j] = false;
                    retval = tmesh.insert_knot(
                                j,
                                child_level,
                                inserted_knots[j][i],
                                input.params->param_grid,
                                inserted_knot_idxs[j][i]);

                    if (retval == 1 || retval == 2)
                    {
                        inserted[j] = true;
                        new_candidate = true;
                    }
                }

                // it's possible that all components of the knot exist separately but not together
                // in this case check if the full-dimension knot is already part of a candidate tensor before skipping
                if (find(inserted.begin(), inserted.end(), true) == inserted.end())     // knot exists in all dimensions separately
                {
                    vector<KnotIdx> knot(dom_dim);                                      // full-dim. knot
                    for (auto j = 0; j < dom_dim; j++)
                        knot[j] = inserted_knot_idxs[j][i];

                    TensorIdx k;
                    for (k = 0; k < new_tensors.size(); k++)                            // check all candidate tensors
                    {
                        auto& c = new_tensors[k];
                        if (tmesh.in(knot, c.knot_mins, c.knot_maxs))
                            break;
                    }
                    if (k == new_tensors.size())                                        // no candidates contain the knot
                            new_candidate = true;
                }

                if (!new_candidate)
                    continue;

                // make a candidate tensor with the new knot plus the pad
                TensorProduct<T> c(dom_dim);
                c.level                 = child_level;
                c.parent                = parent_tensor_idxs[i];
                c.parent_exists         = true;
                TensorProduct<T>& pt    = tensor_prods[parent_tensor_idxs[i]];             // parent tensor of the candidate tensor
                std::vector<KnotIdx> inserted_knot_idx(dom_dim);
                for (auto j = 0; j < dom_dim; j++)
                    inserted_knot_idx[j] = inserted_knot_idxs[j][i];
                tmesh.make_candidate(inserted_knot_idx, c, pt, pad, edge_pad);

                // adjust knot mins, maxs of tensors to be added so far because of inserted knots
                for (auto tidx = 0; tidx < new_tensors.size(); tidx++)          // for all tensors scheduled to be added so far
                {
                    auto& t = new_tensors[tidx];

                    // adjust previously scheduled tensor knot mins, maxs for new knot insertion
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        if (inserted[j] && inserted_knot_idxs[j][i] <= t.knot_mins[j])
                            t.knot_mins[j]++;
                        if (inserted[j] && inserted_knot_idxs[j][i] <= t.knot_maxs[j])
                            t.knot_maxs[j]++;
                    }
                }

                // debug: check knot mins and maxs of candidate that they don't exceed some level (eg., parent)
                // TODO: remove once stable
                if (!tmesh.check_knot_edge_level(c, tensor_prods[c.parent].level))
                    throw MFAError(fmt::format("Refine() 1: knot mins/maxs exceed parent level\n"));

                // check/adjust candidate knot mins and maxs subset and intersection against tensors to be added so far
                while (1)
                {
                    bool changed_new_tensors = false;                               // new_tensors changed and will need to be (re)checked
                    bool add    = true;                                             // the candidate tensor c should be added to the new tensors

                    for (auto tidx = 0; tidx < new_tensors.size(); tidx++)          // for all tensors scheduled to be added so far
                    {
                        auto& t = new_tensors[tidx];
                        if (t.level < 0)                                            // t was marked as invalid and should be skipped
                            continue;

                        // candidate c is a subset of an already scheduled tensor t
                        // keep t, invalidate c
                        if (tmesh.subset(c.knot_mins, c.knot_maxs, t.knot_mins, t.knot_maxs))
                        {
                            t.level             = child_level;
                            add                 = false;
                            break;
                        }

                        // an already scheduled tensor is a subset of the candidate
                        // keep c, invalidate t
                        else if (tmesh.subset(t.knot_mins, t.knot_maxs, c.knot_mins, c.knot_maxs))
                        {
                            if (t.parent != c.parent)
                            {
                                // TODO: not sure if this should be an error, or if the parent should be reset
                                fmt::print(stderr, "Error: already scheduled tensor is a subset of the candidate tensor, but they have different parents. This should not happen\n");
                                abort();
                                //                         t.parent    = c.parent;
                            }
                            t.level             = -1;
                            add                 = true;
                            changed_new_tensors = true;
                        }

                        // candidate intersects an already scheduled tensor, to within some proximity
                        // merge c into t, update and keep t, invalidate c
                        // pad - 1 allows remainders of size pad, which is what we want
                        else if (tmesh.intersect(c, t, pad - 1))
                        {
                            if (c.parent == t.parent)                           // only merge tensors refined from the same parent
                            {
                                tmesh.merge_tensors(t, c, pad, edge_pad);       // merge c into t, updating the knot mins, maxs of t
                                t.level             = child_level;
                                add                 = false;
                                break;
                            }
                        }

                    }       // for all tensors scheduled to be added so far

                    // schedule the tensor to be added
                    if (add)
                    {
                        new_tensors.push_back(c);

                        // debug: check knot mins and maxs of candidate that they don't exceed some level (eg., parent)
                        // TODO: remove once stable
                        if (!tmesh.check_knot_edge_level(c, tensor_prods[c.parent].level))
                            throw MFAError(fmt::format("Refine() 2: knot mins/maxs exceed parent level\n"));
                    }

                    if (!changed_new_tensors)
                        break;
                }   // while !all_done
            }   // for all knots to be inserted

            // timing
            fmt::print(stderr, "error spans time:       {:.3e} s.\n", error_spans_time);

            return false;
        }

        // final double-check that new tensor products do not intersect each other within some pad amount
        // check all tensors to be added against each other
        void CheckNewTensors(
                vector<TensorProduct<T>>&   new_tensors,                        // new tensors scheduled to be added
                int                         pad,                                // padding for all tensors
                int                         edge_pad)                           // extra padding for tensor at the global edge
        {
            Tmesh<T>& tmesh = mfa_data.tmesh;
            VectorXi& p     = mfa_data.p;

            while (1)
            {
                bool changed_new_tensors = false;                               // new_tensors changed and will need to be (re)checked

                for (auto cidx = 0; cidx < new_tensors.size(); cidx++)          // for all candidate tensors from tensors scheduled to be added so far
                {
                    auto& c = new_tensors[cidx];
                    if (c.level < 0)                                            // c was marked as invalid and should be skipped
                        continue;

                    for (auto tidx = 0; tidx < new_tensors.size(); tidx++)      // for all tensors scheduled to be added so far
                    {
                        auto& t = new_tensors[tidx];
                        if (cidx == tidx || t.level < 0)                        // t was marked as invalid and should be skipped
                            continue;

                        // candidate c is a subset of an already scheduled tensor t
                        // keep t, invalidate c
                        if (tmesh.subset(c.knot_mins, c.knot_maxs, t.knot_mins, t.knot_maxs))
                        {
                            c.level             = -1;
                            break;
                        }

                        // an already scheduled tensor is a subset of the candidate
                        // keep c, invalidate t
                        else if (tmesh.subset(t.knot_mins, t.knot_maxs, c.knot_mins, c.knot_maxs))
                        {
                            if (t.parent != c.parent)
                            {
                                // TODO: not sure if this should be an error, or if the parent should be reset
                                fmt::print(stderr, "Error: already scheduled tensor is a subset of the candidate tensor, but they have different parents. This should not happen\n");
                                abort();
                                //                         t.parent    = c.parent;
                            }
                            t.level             = -1;
                        }

                        // candidate intersects an already scheduled tensor, to within some proximity
                        // merge c into to t, update and keep t, invalidate c
                        // pad - 1 allows remainders of size pad, which is what we want
                        else if (tmesh.intersect(c, t, pad - 1))
                        {
                            if (c.parent == t.parent)                           // only merge tensors refined from the same parent
                            {
                                tmesh.merge_tensors(t, c, pad, edge_pad);       // merge c into t, updating the knot mins, maxs of t
                                c.level                 = -1;
                                changed_new_tensors     = true;
                                break;
                            }
                        }

                    }   // for all tensors tidx
                }   // for all tensors cidx

                if (!changed_new_tensors)
                    break;
            }   // while !all_done
        }

        // appends and encodes a vector of new tensor products
        // returns whether all tensors were successfully added
        bool AddNewTensors(
                vector<TensorProduct<T>>& new_tensors,
                int pad)                                        // required distance between new tensors and existing tensors
        {
            // typing shortcuts
            auto&   tmesh                   = mfa_data.tmesh;
            auto&   tensor_prods            = tmesh.tensor_prods;
            auto&   p                       = mfa_data.p;

            bool retval = true;

            for (auto k = 0; k < new_tensors.size(); k++)       // for all tensors to append
            {
                auto& t = new_tensors[k];

                if (t.level < 0)                                // tensor was removed from the schedule
                    continue;

                // debug
//                 fmt::print(stderr, "AddNewTensors() 1: pt idx {}: parent mins [{}] maxs [{}] level {} t mins [{}] maxs [{}] t.level {}\n",
//                         t.parent, fmt::join(tensor_prods[t.parent].knot_mins, ","), fmt::join(tensor_prods[t.parent].knot_maxs, ","), tensor_prods[t.parent].level,
//                         fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs,  ","), t.level);

                // check one last time intersection of new tensor t and other existing tensors e in new_tensors would result in too small of a tensor
                for (auto j = 0; j < tensor_prods.size(); j++)
                {
                    auto& e = tensor_prods[j];                  // existing tensor

                    // outright intersection that is too near to edge of existing tensor
                    // can't handle this case at this point in the algorithm
                    // need to invalidate the candidate, mark intersected existing tensor as not done
                    // check its level, hopefully no higher up than the parent level
                    // and don't increase the parent level
                    // wait to pick this candidate up in the next iteration

                    // check for intersection between t (tensor to be added) and e (existing tensor)
                    std::vector<KnotIdx> inter_mins(t.knot_mins.size());
                    std::vector<KnotIdx> inter_maxs(t.knot_maxs.size());
                    if (tmesh.intersects(t.knot_mins, t.knot_maxs, e.knot_mins, e.knot_maxs, inter_mins, inter_maxs))
                    {
                        // check intersection amount in each dimension
                        // since the candidate is the deepest level, consider all knot indices using a simple subtraction (not Tmesh::knot_idx_dist()
                        bool found_inter = false;
                        for (auto i = 0; i < dom_dim; i++)
                        {
                            // check t.knot_mins against both sides of e
                            if (t.level > 0 && t.knot_mins[i] > e.knot_mins[i] && t.knot_mins[i] < e.knot_maxs[i])
                            {
                                if (t.knot_mins[i] - e.knot_mins[i] < pad || e.knot_maxs[i] - t.knot_mins[i] < pad)
                                {
                                    found_inter = true;
                                    break;
                                }
                            }
                            // check t.knot_maxs against both sides of e
                            if (t.level > 0 && !found_inter && t.knot_maxs[i] > e.knot_mins[i] && t.knot_maxs[i] < e.knot_maxs[i])
                            {
                                if (t.knot_maxs[i] - e.knot_mins[i] < pad || e.knot_maxs[i] - t.knot_maxs[i] < pad)
                                {
                                    found_inter = true;
                                    break;
                                }
                            }
                        }

                        if (found_inter)
                        {
                            // debug
//                             fmt::print(stderr, "\nAddNewTensors() 3: t: knot_mins [{}] knot_maxs [{}] level {} intersects existing tensor e: knot_mins [{}] knot_maxs [{}] level {} with small distance\n",
//                                     fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level, fmt::join(e.knot_mins, ","), fmt::join(e.knot_maxs, ","), e.level);

                            // invalidate this tensor
                            t.level = -1;
                            retval  = false;
                        }

                        if (t.level < 0)
                            e.done = false;                 // any existing tensor intersecting a tensor that could not be added should be rechecked next time
                    }       // e intersects t

                    // check for nearness to existing tensor without actually intersecting
                    // in this case, merge e and t, enlarging to cover e
                    // pad - 1 allows remainders of size pad to remain unmerged, which is what we want
                    if (!tmesh.intersect(t, e, 0, true, false) && tmesh.intersect(t, e, pad - 1, true, false))
                    {
                        // merge e (existing tensor) into t (tensor to be added)
                        // ie, enlarge t to cover e
                        tmesh.merge_tensors(t, e, -1, 0);

                        // debug
//                         fmt::print(stderr, "AddNewTensors() 4: e was merged into t: possibly expanding t with new knot_mins [{}] knot_maxs [{}] level {}\n\n",
//                                 fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level);
                    }
                }

                if (t.level < 0)
                    continue;

                // debug
//                 fmt::print(stderr, "AddNewTensors() 5: appending t mins [{}] maxs [{}] t.level {}\n",
//                         fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs,  ","), t.level);

                // append the tensor
                int tensor_idx = tmesh.append_tensor(t.knot_mins, t.knot_maxs, t.level);

                // debug
//                 fmt::print(stderr, "\nT-mesh after append\n\n");
//                 mfa_data.tmesh.print(false, true);
//                 mfa_data.tmesh.print();

                // debug: check all spans before solving
                // TODO: comment out once the code is debugged
                mfa::NewKnots<T> nk(mfa_data, input);
                if (!nk.CheckAllSpans())
                    throw MFAError(fmt::format("AddNewTensors(): Error: failed checking all spans for input points\n"));

                // debug: check all knot vs control point quantities
                // TODO: comment out once the code is debugged
                for (auto j = 0; j < tensor_prods.size(); j++)
                    if (!tmesh.check_num_knots_ctrl_pts(j))
                        throw MFAError(fmt::format("AddNewTensors(): number of knots and control points do not agree\n"));

                // debug: confirm that all tensors will have at least p control points
                // TODO: comment out once the code is debugged
                for (auto i = 0; i < tensor_prods.size(); i++)
                {
                    if (!tmesh.check_num_ctrl_degree(i, 0))
                    {
                        auto& t = tensor_prods[i];
                        fmt::print(stderr, "Error: AddNewTensors(): After appending, tensor tidx {} knot_mins [{}] knot_maxs [{}] level {} has fewer than p control points.\n",
                                i, fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level);
                        fmt::print(stderr, "\nAddNewTensors(): T-mesh after appending tensor:\n");
                        tmesh.print(false, false, false, false);
                        abort();
                    }
                }

                // solve for new control points

#ifdef MFA_ENCODE_LOCAL_SEPARABLE

                EncodeTensorLocalSeparable(tensor_idx);

#else

                EncodeTensorLocalUnified(tensor_idx);

#endif
            }   // for all tensors k to append

            return retval;
        }

        // updates and encodes existing tensors if they match any in a vector of new tensor products
        void UpdateExistingTensors(vector<TensorProduct<T>>& new_tensors)
        {
            // typing shortcuts
            auto&   tmesh                   = mfa_data.tmesh;
            auto&   tensor_prods            = tmesh.tensor_prods;
            auto&   dom_dim                 = mfa_data.dom_dim;
            auto&   p                       = mfa_data.p;

            // debug: for checking knot spans for input points
            // TODO: comment out once the code is debugged
            mfa::NewKnots<T> nk(mfa_data, input);

            for (auto k = 0; k < new_tensors.size(); k++)
            {
                auto& t = new_tensors[k];

                if (t.level < 0)                        // tensor was removed from the schedule
                    continue;

                int tensor_idx = tmesh.update_tensor(t.knot_mins, t.knot_maxs, t.level);

                if (tensor_idx < 0)
                    continue;

                // debug
//                 fmt::print(stderr, "\nT-mesh after append\n\n");
//                 mfa_data.tmesh.print(true, true);

                // debug: check all spans before solving
                // TODO: comment out once the code is debugged
                if (!nk.CheckAllSpans())
                    throw MFAError(fmt::format("UpdateExistingTensors(): Error: failed checking all spans for input points\n"));

                // debug: check all knot vs control point quantities
                // TODO: comment out once the code is debugged
                for (auto j = 0; j < tensor_prods.size(); j++)
                    if (!tmesh.check_num_knots_ctrl_pts(j))
                        throw MFAError(fmt::format("UpdateExistingTensors(): number of knots and control points do not agree\n"));

                // debug: confirm that all tensors will have at least p control points
                // TODO: comment out once the code is debugged
                for (auto i = 0; i < tensor_prods.size(); i++)
                {
                    if (!tmesh.check_num_ctrl_degree(i, 0))
                    {
                        auto& t = tensor_prods[k];
                        fmt::print(stderr, "Error: UpdateExistingTensors(): After appending tensor k {} one of the tensors has fewer than p control points. This should not happen\n", k);
                        fmt::print(stderr, "New tensor knot_mins [{}] knot_maxs[{}] level {} parent tensor {}\n",
                                fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level, t.parent);
                        fmt::print(stderr, "\nUpdateExistingTensors(): T-mesh after appending tensor k {}\n", k);
                        tmesh.print(true, true, false, false);
                        abort();
                    }
                }

                // solve for new control points
#ifdef MFA_ENCODE_LOCAL_SEPARABLE
                EncodeTensorLocalSeparable(tensor_idx);
#else
                EncodeTensorLocalUnified(tensor_idx);
#endif
                t.level = -1;                       // remove this tensor from the new tensors
            }
        }

        // constraint control points and corresponding anchors for local solve
        // helper function for EncodeTensorLocalSeparable and EncodeTensorLocalUnified
        // this version checks all tensors, slower than looking at prev/next, but reliably finds all constraints
        void LocalSolveAllConstraints(
                const TensorProduct<T>&     tc,                 // current tensor product being solved
                MatrixX<T>&                 ctrl_pts,           // (output) constraining control points
                vector<vector<KnotIdx>>&    anchors,            // (output) corresponding anchors
                vector<TensorIdx>&          t_idx_anchors)      // (output) tensors containing corresponding anchors
        {
            const Tmesh<T>&         tmesh   = mfa_data.tmesh;
            int                     cols    = tc.ctrl_pts.cols();

            // debug
            bool debug = false;

            // mins, maxs of tc padded by degree p
            vector<KnotIdx> tc_pad_mins(dom_dim);
            vector<KnotIdx> tc_pad_maxs(dom_dim);

            // intersection of tc padded by degree p with tensor being visited
            vector<KnotIdx> intersect_mins(dom_dim);
            vector<KnotIdx> intersect_maxs(dom_dim);

            // get required sizes

            int rows = 0;                                       // number of rows required in ctrl_pts
            VectorXi npts = VectorXi::Zero(dom_dim);

            for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[k];
                if (&t == &tc)
                    continue;
                if (t.level > tc.level)
                    continue;

                // pad mins and maxs of tc
                for (auto i = 0; i < dom_dim; i++)
                {
                    int p = mfa_data.p(i);
                    tmesh.knot_idx_ofst(t, tc.knot_mins[i], -p, i, true, tc_pad_mins[i]);
                    tmesh.knot_idx_ofst(t, tc.knot_maxs[i], p, i, true, tc_pad_maxs[i]);
                }

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < dom_dim; i++)
                    {
                        // compute npts
                        if (mfa_data.p(i) % 2)              // odd degree
                            npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, true);
                        else                                // even degree
                            npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, false);
                    }
                }
                rows += npts.prod();
            }       // for all tensor products

            ctrl_pts.resize(rows, cols);

            anchors.resize(rows);
            t_idx_anchors.resize(rows);

            // get control points and anchors
            int cur_row = 0;
            VectorXi sub_starts(dom_dim);
            VectorXi sub_npts(dom_dim);
            VectorXi all_npts(dom_dim);
            vector<KnotIdx> anchor(dom_dim);           // one anchor
            for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[k];
                if (&t == &tc)
                    continue;
                if (t.level > tc.level)
                    continue;

                // pad mins and maxs of tc
                for (auto i = 0; i < dom_dim; i++)
                {
                    int p = mfa_data.p(i);
                    tmesh.knot_idx_ofst(t, tc.knot_mins[i], -p, i, true, tc_pad_mins[i]);
                    tmesh.knot_idx_ofst(t, tc.knot_maxs[i], p, i, true, tc_pad_maxs[i]);
                }

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < dom_dim; i++)
                    {
                        int p = mfa_data.p(i);
                        // compute sub_starts, sub_npts, all_npts
                        sub_starts(i) = tmesh.knot_idx_dist(t, t.knot_mins[i], intersect_mins[i], i, false);
                        if (t.knot_mins[i] == 0)
                        sub_starts(i) -= (p + 1) / 2;
                        if (mfa_data.p(i) % 2)              // odd degree
                            sub_npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, true);
                        else                                // even degree
                            sub_npts(i) = tmesh.knot_idx_dist(t, intersect_mins[i], intersect_maxs[i], i, false);
                        all_npts(i) = t.nctrl_pts(i);
                    }

                    VolIterator voliter(sub_npts, sub_starts, all_npts);
                    VectorXi ijk(dom_dim);
                    while (!voliter.done())
                    {
                        // skip MFA_NAW control points (used in odd degree cases)
                        if (t.weights(voliter.sub_full_idx(voliter.cur_iter())) != MFA_NAW)
                        {
                            // control point
                            ctrl_pts.row(cur_row) = t.ctrl_pts.row(voliter.sub_full_idx(voliter.cur_iter()));

                            // anchor
                            anchors[cur_row].resize(dom_dim);
                            voliter.idx_ijk(voliter.cur_iter(), ijk);
                            mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);
                            for (auto i = 0; i < dom_dim; i++)
                                anchors[cur_row][i] = anchor[i];
                            t_idx_anchors[cur_row] = k;
                            cur_row++;
                        }
                        voliter.incr_iter();
                    }   // voliter
                }   // intersects
            }   // for all tensor products

            if (cur_row < ctrl_pts.rows())          // not all control points were used because of skipping MFA_NAW
            {
                ctrl_pts.conservativeResize(cur_row, cols);
                anchors.resize(cur_row);
                t_idx_anchors.resize(cur_row);
            }
        }

#endif      // MFA_TMESH

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
            new_knots.resize(dom_dim);
            vector<vector<int>> new_levels(dom_dim);

            for (auto& t : mfa_data.tmesh.tensor_prods)                             // for all tensor products in the tmesh
            {
                // check and assign main quantities
                VectorXi n = t.nctrl_pts - VectorXi::Ones(dom_dim);        // number of control point spans in each domain dim
                VectorXi m = input.ndom_pts()  - VectorXi::Ones(dom_dim);    // number of input data point spans in each domain dim

                // resize control points and weights
                t.ctrl_pts.resize(t.nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
                t.weights = VectorX<T>::Ones(t.ctrl_pts.rows());

                for (size_t k = 0; k < dom_dim; k++)                       // for all domain dimensions
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
                        int span = mfa_data.tmesh.FindSpan(k, input.params->param_grid[k][i], mfa_data.tmesh.tensor_prods[0]);
#ifndef MFA_TMESH       // original version for one tensor product
                        mfa_data.OrigBasisFuns(k, input.params->param_grid[k][i], span, N, i);
#else                   // tmesh version
                        mfa_data.BasisFuns(k, input.params->param_grid[k][i], span, N, i);
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

                    size_t ncurves   = input.domain.rows() / input.ndom_pts(k);     // number of curves in this dimension
                    int nsame_steps  = 0;                                   // number of steps with same number of erroneous points
                    int n_step_sizes = 0;                                   // number of step sizes so far

                    // starting step size over curves
                    size_t s0 = ncurves / 2 > 0 ? ncurves / 2 : 1;

#ifdef MFA_CHECK_ALL_CURVES
                    s0              = 1;
                    max_num_curves  = ncurves;
#endif

                    for (size_t s = s0; s >= 1 && ncurves / s <= max_num_curves; s /= 2)     // for all step sizes over curves
                    {
                        bool new_max_nerr = false;                          // this step size changed the max_nerr

                        for (size_t j = 0; j < ncurves; j++)                // for all the curves in this dimension
                        {
                            // each time the step changes, shift start of s-th curves by one (by subtracting
                            // n_step-sizes below)
                            if (j >= n_step_sizes && (j - n_step_sizes) % s == 0)           // this is one of the s-th curves; compute it
                            {
                                // compute R from input domain points
                                RHS(k, N, R, weights, input.g.co[k][j]);

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
                                size_t nerr = ErrorCurve(k, t, input.g.co[k][j], P, weights, extents, err_spans, err_limit);

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
//                     fprintf(stderr, "\rdimension %ld of %d encoded\n", k + 1, dom_dim);
                }                                                           // domain dimensions

                // debug
//                 for (auto i = 0; i < mfa_data.dom_dim; i++)
//                     fmt::print(stderr, "new_knots in dim {}: [{}]\n", i, fmt::join(new_knots[i], ","));

                // insert the new knots
                mfa::NewKnots<T> nk(mfa_data, input);
                vector<vector<KnotIdx>> unused(dom_dim);
                nk.OrigInsertKnots(new_knots, new_levels, unused);

                // increase number of control points, weights, basis functions
                for (auto k = 0; k < dom_dim; k++)
                {
                    t.nctrl_pts(k) += new_knots[k].size();
                }
                auto tot_nctrl_pts = t.nctrl_pts.prod();
                t.ctrl_pts.resize(tot_nctrl_pts, t.ctrl_pts.cols());
                t.weights =  VectorX<T>::Ones(tot_nctrl_pts);
                mfa_data.tmesh.tensor_knot_idxs(t);
            }                                                               // tensor products

            return(tot_nnew_knots ? 0 : 1);
        }

    public:
        // Writes collocation matrix N to a file in triplet format (row-major order)
        void dumpCollocationMatrix(string name = "")
        {
            string filename = "N_encode" + name + ".txt";

            if (verbose >= 2)
            {
                fmt::print(" * Encoder: Dumping collocation matrix to {}\n", filename);
            }

            if (input.is_structured())
            {
                fmt::print("dumpCollocationMatrix is not yet supported for structured input.\n");
                return;
            }

            std::ofstream os;
            os.open(filename);

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[0];

            // control point strides
            VectorXi cps = VectorXi::Ones(dom_dim); 
            for (int k = 1; k < dom_dim; k++)
            {
                cps(k) = cps(k-1) * t.nctrl_pts(k-1);
            }

            // Prep for evaluating basis functions
            VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
            BasisFunInfo<T>     bfi(q);                                             // buffers for basis fun evaluation
            vector<vector<T>>  lB(dom_dim);                               // stores value of basis funs
            for (int k = 0; k < dom_dim; k++)
            {
                lB[k].resize(q[k]);
            }

            VectorXi ctrl_starts(dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi spans(dom_dim);                                       // current knot span in each dimension

            // Iterate thru every point in subvolume given by tensor
            VectorX<T> param(input.dom_dim);
            for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
            {
                input_it.params(param);

                // Compute basis functions at input point
                for (auto k = 0; k < dom_dim; k++)
                {
                    int p   = mfa_data.p(k);
                    T   u   = param(k);

                    spans[k] = mfa_data.tmesh.FindSpan(k, u);

                    ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

                    mfa_data.FastBasisFuns(k, u, spans[k], lB[k], bfi);
                }

                // Compute matrix value and insert into Nt
                VolIterator ctrl_vol_iter(q);
                while (!ctrl_vol_iter.done())
                {
                    int ctrl_idx_full = 0;
                    T coeff_prod = 1;
                    for (int k = 0; k < dom_dim; k++)
                    {
                        int idx = ctrl_vol_iter.idx_dim(k);
                        ctrl_idx_full += (idx + ctrl_starts(k)) * cps(k);
                        coeff_prod *= lB[k][idx];
                    }

                    if (coeff_prod != 0)
                    {
                        // Rows correspond to data, columns to control points
                        int row = input_it.idx();
                        int col = ctrl_idx_full;
                        fmt::print(os, "{} {} {}\n", row, col, coeff_prod);
                    }

                    ctrl_vol_iter.incr_iter();
                }
            }

            os.close();

            return;
        }
    };          // Encoder class
}           // mfa namespace

#endif
