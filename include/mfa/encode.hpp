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

#include    <vector>
#include    <set>
#include    <iostream>

// temporary utilities for testing
#include    <ctime>
#include    <chrono>
#include    <iomanip>
#include    <fstream>
#include    <sstream>
#include    <mpi.h>     // for MPI_Wtime() only

#ifndef      MFA_NO_WEIGHTS

#include    "coin/ClpSimplex.hpp"
#include    "coin/ClpInterior.hpp"

#endif

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

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

        const MFA<T>&       mfa;                            // the mfa top-level object
        MFA_Data<T>&        mfa_data;                       // the mfa data model
        int                 verbose;                        // output level
        const PointSet<T>&  input;                          // input points
        size_t              max_num_curves;                 // max num. curves per dimension to check in curve version

    public:

        Encoder(
                const MFA<T>&       mfa_,                   // MFA top-level object
                MFA_Data<T>&        mfa_data_,              // MFA data model
                const PointSet<T>&  input_,                 // input points
                int                 verbose_) :             // debug level
            mfa(mfa_),
            mfa_data(mfa_data_),
            verbose(verbose_),
            input(input_),
            max_num_curves(1.0e5)                           // max num. curves to check in one dimension of curve version
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
            if (mfa_data.p.size() != input.ndom_pts.size())
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

            int      ndims  = input.ndom_pts.size();          // number of domain dimensions
            size_t   cs     = 1;                            // stride for input points in curve in cur. dim
            int      pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;// control point dimensonality
            // resize matrices in case number of control points changed
            ctrl_pts.resize(nctrl_pts.prod(), pt_dim);
            weights.resize(ctrl_pts.rows());

            // resize basis function matrices and initialize to 0; will only fill nonzeros later
            for (auto k = 0; k < ndims; k++)
                    mfa_data.N[k] = MatrixX<T>::Zero(input.ndom_pts(k), nctrl_pts(k));

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

            VectorXi ntemp_ctrl = input.ndom_pts;     // current num of temp control pts in each dim

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

                for (int i = 0; i < mfa_data.N[k].rows(); i++)
                {
                    int span = mfa_data.tmesh.FindSpan(k, input.params->param_grid[k][i], nctrl_pts(k));

#ifndef MFA_TMESH   // original version for one tensor product

                    mfa_data.OrigBasisFuns(k, input.params->param_grid[k][i], span, mfa_data.N[k], i);

#else               // tmesh version

                    mfa_data.BasisFuns(k, input.params->param_grid[k][i], span, mfa_data.N[k], i);

#endif
                }

                // debug
//                 fmt::print(stderr, "N[{}]:\n {}\n", k, mfa_data.N[k]);

                // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
                // NtN has semibandwidth < p + 1 nonzero entries across diagonal
                MatrixX<T> NtN  = mfa_data.N[k].transpose() * mfa_data.N[k];

                // debug
//                 cerr << "N[k]:\n" << mfa_data.N[k] << endl;
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
                        MatrixX<T> R(mfa_data.N[k].cols(), pt_dim);
                        // P are the unknown control points and the solution to NtN * P = R
                        // NtN is positive definite -> do not need pivoting
                        // TODO: use a common representation for P and ctrl_pts to avoid copying
                        MatrixX<T> P(mfa_data.N[k].cols(), pt_dim);
                        // compute the one curve of control points
                        CtrlCurve(mfa_data.N[k], NtN, R, P, k, co[j], cs, to[j], temp_ctrl0, temp_ctrl1, -1, ctrl_pts, weights, weighted);
                    }   // curves in this dimension
                }, ap);

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
                                SparseMatrixX<T>&   Nt)     // (output) transpose of collocation matrix
        {
            cerr << "begin matrix construction" << endl;
            clock_t fill_time = clock();

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];

            VectorXi ctrl_starts(mfa_data.dom_dim);                                 // subvolume ctrl pt indices in each dimension
            VectorXi spans(mfa_data.dom_dim);                                       // current knot span in each dimension
            VectorXi nctrl_patch = mfa_data.p + VectorXi::Ones(mfa_data.dom_dim);   // number of nonzero basis functions at a given parameter, in each dimension
            vector<MatrixX<T>>  B(mfa_data.dom_dim);                                // list of 1D basis functions valued at a given parameter (one per dimension)
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                B[k].resize(1, t.nctrl_pts(k));
            }


            // resize matrices in case number of control points changed
            const int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                           // control point dimensonality
            t.ctrl_pts.resize(t.nctrl_pts.prod(), pt_dim);
            t.weights.resize(t.ctrl_pts.rows());
            assert(Nt.rows() == t.nctrl_pts.prod());
            assert(Nt.cols() == input.npts);

            // Reserve space in sparse matrix; don't forget to call makeCompressed() at end!
            int bf_per_pt = (mfa_data.p + VectorXi::Ones(mfa_data.dom_dim)).prod();       // nonzero basis functions per input point
            Nt.reserve(VectorXi::Constant(Nt.cols(), bf_per_pt));

            // Iterate thru every point in subvolume given by tensor
            VectorX<T> param(input.dom_dim);
            for (auto input_it = input.begin(), input_end = input.end(); input_it != input_end; ++input_it)
            {
                input_it.params(param);

                for (auto k = 0; k < mfa_data.dom_dim; k++)
                {
                    int p   = mfa_data.p(k);
                    T   u   = param(k);

                    spans[k] = mfa_data.tmesh.FindSpan(k, u);

                    ctrl_starts(k) = spans[k] - p - t.knot_mins[k];

                    // basis functions
                    vector<T> loc_knots(p + 2);
                    B[k].row(0).setZero();

                    for (auto j = 0; j < p + 1; j++)
                    {
                        for (auto i = 0; i < p + 2; i++)
                            loc_knots[i] = mfa_data.tmesh.all_knots[k][spans[k] - p + j + i];
                        int col = spans[k] - p + j - t.knot_mins[k];
                        if (col >= 0 && col < B[k].cols())
                            B[k](0, col) = mfa_data.OneBasisFun(k, u, loc_knots);
                    }
                }

                // Iterate over all basis functions (ctrl points) which are nonzero at the given input point
                VolIterator ctrl_vol_iter(nctrl_patch, ctrl_starts, t.nctrl_pts);
                while (!ctrl_vol_iter.done())
                {
                    int ctrl_idx_full = ctrl_vol_iter.cur_iter_full();
                    T coeff_prod = 1;
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                    {
                        int idx = ctrl_vol_iter.idx_dim(k);                                 // index in current dim of this control point
                        coeff_prod *= B[k](0,idx);
                    }

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

            fill_time = clock() - fill_time;
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
        void EncodeUnified( TensorIdx   t_idx,                      // tensor product being encoded
                            bool        weighted=true)                   // solve for and use weights 
        {
            // debug
            cerr << "EncodeTensor (Unified Dimensions)" << endl;
            cerr << "NOTE: Only valid for single tensor product!" << endl;
            if (weighted)  // We want weighted encoding to be default behavior eventually. However, not currently implemented.
            {
                cerr << "Warning: NURBS (nonuniform weights) are not implemented for unified-dimensional encoding!" << endl;
            }

            const int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                           // control point dimensonality
            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];

            // REQUIRED for Tmesh
            // Compute total number of points in tensor product
            int tot_dom_pts = 1;
            vector<size_t> start_idxs(mfa_data.dom_dim);
            vector<size_t> end_idxs(mfa_data.dom_dim);
            mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, true, start_idxs, end_idxs);
            for (int k=0; k < mfa_data.dom_dim; k++)
                tot_dom_pts *= end_idxs[k] - start_idxs[k] + 1;

            // Assemble collocation matrix
            SparseMatrixX<T> Nt(t.nctrl_pts.prod() , input.npts);
            CollMatrixUnified(t_idx, /*start_idxs, end_idxs,*/ Nt);

            // Set up linear system
            SparseMatrixX<T> Mat(Nt.rows(), Nt.rows()); // Mat will be the matrix on the LHS

#ifdef MFA_TBB  // TBB version
                // TODO potentially unnecessary deep copies, maybe make N and Nt in col-major form simultaneously?
                // Creating a separate matrix for N makes threading the sparse matrix product easier
                Eigen::SparseMatrix<T, Eigen::ColMajor> NCol = Nt.transpose();

                int ntn_sparsity = (2*mfa_data.p + VectorXi::Ones(mfa_data.dom_dim)).prod();       // nonzero basis functions per input point
                MatProdThreaded(Nt, NCol, Mat, ntn_sparsity);
#else
                Mat = Nt * Nt.transpose();
#endif

// EXPERIMENTAL search for potential infinite ctrl points >>
#if 0
            Mat.prune(1e-5);
            // Check for unconstrained control points (we set to zero later)
            // this can happen if there is no input data within the support of a 
            // tensor basis function
            vector<int> undef_ctrl_pts;
            for (int i = 0; i < Mat.cols(); i++)
            {
                if (Mat.coeff(i,i)==0)
                    undef_ctrl_pts.push_back(i);
            }     
#endif
// << EXPERIMENTAL

            MatrixX<T>  R(Nt.cols(), pt_dim);           // R is the right hand side 
            RHSUnified(/*start_idxs, end_idxs,*/ Nt, R);

            // Solve Linear System
            // Eigen::ConjugateGradient<SparseMatrixX<T>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteLUT<T>>  solver;
            Eigen::ConjugateGradient<SparseMatrixX<T>, Eigen::Lower|Eigen::Upper>  solver;  // Default preconditioner is Jacobi

            // // Optional parameters for solver    
            // solver.setTolerance(1e-5);
            // solver.preconditioner().setDroptol(0.001);
            // solver.preconditioner().setFillfactor(1);

            solver.compute(Mat);
            if (solver.info() != Eigen::Success) 
                cerr << "Matrix decomposition failed in EncodeTensor" << endl;
            else
                cerr << "Sparse matrix factorization successful" << endl;

            t.ctrl_pts = solver.solve(R); 
            if (solver.info() != Eigen::Success)
                cerr << "Least-squares solve failed in EncodeTensor" << endl;
            else
            {
                cerr << "Sparse matrix solve successful" << endl;
                cerr << "  # iterations: " << solver.iterations() << endl;
            }


// EXPERIMENTAL search for potential infinite ctrl points >>
#if 0
            for (auto& idx : undef_ctrl_pts)
            {
                cerr << "idx=" << idx << ", value(s)=" << t.ctrl_pts(idx,0) << endl;
                t.ctrl_pts.row(idx).setZero();
            }

            for (int i = 0; i < t.ctrl_pts.rows(); i++)
            {
                for (int k = 0; k < t.ctrl_pts.cols(); k++)
                {
                    if (fabs(t.ctrl_pts(i,k)) > 1e3)
                    {
                        cerr << "Extremely large control point:" << endl;
                        cerr << "  (i,j) = " << i << " " << k << endl;
                        cerr << "  value = " << t.ctrl_pts(i,k) << endl;
                        cerr << " row i nnzs: " << Mat.col(i).nonZeros() << endl;

                        T* vals = Mat.valuePtr();
                        int* outerIds = Mat.outerIndexPtr();
                        T rowmax = -1;
                        for (int l = outerIds[i]; l < outerIds[i+1]; l++)
                        {
                            if (vals[l] > rowmax) rowmax = vals[l];
                        }
                        cerr << " row i max:  " << rowmax << endl;
                        t.ctrl_pts(i,k) = 0;
                    }
                }
            }
#endif
// << EXPERIMENTAL
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
            VectorX<T> mid_param(mfa_data.dom_dim);

            for (auto i = dim + 1; i < mfa_data.dom_dim; i++)           // only checks dimensions after current dim; earlier dims guranteed to intersect
            {
                mid_param(i) = input.params->param_grid[i][mid_ijk(i)];
                if (mid_param(i) < mfa_data.tmesh.all_knots[i][t.knot_mins[i]] || mid_param(i) > mfa_data.tmesh.all_knots[i][t.knot_maxs[i]])
                {
                    // debug
//                     fmt::print(stderr, "CurveIntersectsTensor(): does not intersect t_idx {} dim {} start_ijk [{}] mid_param({}) = {}\n",
//                             t_idx, dim, start_ijk.transpose(), i, mid_param(i));

                    return false;
                }
            }

            return true;
        }

        // computes curve of free control points in one dimension
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

            if (cons_type != ConsType::MFA_NO_CONSTRAINT)
                ConsCtrlPtCurve(dim, t_idx, start_ijk, nin_pts, cons_type, Ncons, Pcons);

            // timing
            cons_time   += (MPI_Wtime() - t0);

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

            // TODO: understand why the second version of normalization gives a worse answer than the first
            // and why not normalizing at all also gives a good answer, very close to the first version
            // then pick one of these versions and remove the other

#if 1

                {
                    bool error = false;
                    T sum = Nfree.row(i).sum();
                    if (cons_type != ConsType::MFA_NO_CONSTRAINT)
                        sum += Ncons.row(i).sum();
                    if (sum > 0.0)
                    {
                        Nfree.row(i) /= sum;
                        if (cons_type != ConsType::MFA_NO_CONSTRAINT)
                            Ncons.row(i) /= sum;
                    }
//                     else
//                         throw MFAError(fmt::format("ComputeCtrlPtCurve(): row {} has 0.0 row sum for free and constraint basis functions in dim {}",
//                                 i, dim));
//                         fmt::print(stderr, "ComputeCtrlPtCurve(): row {} has 0.0 row sum for free and constraint basis functions in dim {}\n",
//                                 i, dim);
                }

#else

                {
                    T Nfree_sum, Ncons_sum;
                    Ncons_sum = 0.0;
                    T Nfree_sum = Nfree.row(i).sum();
                    if (cons_type != ConsType::MFA_NO_CONSTRAINT)
                        Ncons_sum = Ncons.row(i).sum();
                    if (Nfree_sum + Ncons_sum == 0.0)
                        throw MFAError(fmt::format("ComputeCtrlPtCurve(): row {} has 0.0 row sum for free and constraint basis functions in dim {}",
                                    i, dim));
                    if (Nfree_sum > 0.0)
                        Nfree.row(i) /= Nfree_sum;
                    if (Ncons_sum > 0.0)
                        Ncons.row(i) /= Ncons_sum;
                }

#endif

#ifdef MFA_TBB

            }, ap);

#endif

            norm_time += (MPI_Wtime() - t0);                    // timing

            // R is the right hand side needed for solving N * P = R
            if (Pcons.rows())
                R -= Ncons * Pcons;

            // solve
            t0 = MPI_Wtime();

            // debug: check matrix product sizes
            // TODO: remove once stable
            if (Nfree.rows() != R.rows() || P.rows() != Nfree.cols())
                throw MFAError(fmt::format("ComputeCtrlPtCurve(): Nfree.rows() {} should equal R.rows {} and P.rows() {} should equal Nfree.cols() {}\n",
                        Nfree.rows(), R.rows(), P.rows(), Nfree.cols()));

            P = (Nfree.transpose() * Nfree).ldlt().solve(Nfree.transpose() * R);
            solve_time += (MPI_Wtime() - t0);

            // debug
            MatrixXd::Index max_row, max_col;
            if (P.maxCoeff(&max_row, &max_col) > 300)
            {
                fmt::print(stderr, "ComputeCtrlPtCurve(): very large control points\n");
                fmt::print(stderr, "N row {}:\n {}\n", max_row, Nfree.row(max_row));
                fmt::print(stderr, "NtN row {}:\n {}\n", max_row, (Nfree.transpose() * Nfree).row(max_row));
                fmt::print(stderr, "P row{}:\n {}\n", max_row, P.row(max_row));
            }
        }

        // interpolates curve of free control points in one dimension using Boehm knot insertion
        void InterpCtrlPtCurve(
                int                 dim,                                                // current dimension
                TensorIdx           t_idx,                                              // index of original tensor
                const VectorXi&     start_ijk,                                          // multidim index of first input point for the curve, including constraints
                const VectorXi&     npts,                                               // number of input points (could be control points for input in some dims) including constraints
                MatrixX<T>&         P)                                                  // (output) solution control points, allocated by caller
        {
            // typing shortcuts
            auto& dom_dim       = mfa_data.dom_dim;
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;
            auto& p             = mfa_data.p;
            auto& t             = tensor_prods[t_idx];

            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor
            vector<KnotIdx>     temp_anchor(dom_dim);                                  // temporary anchor
            vector<KnotIdx>     inserted_anchor(dom_dim);                              // inserted control point anchor
            vector<int>         inserted_dims(dom_dim);                                // which dims actually added a knot and ctrl pt

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices in all dims
            vector<T> local_knots(p(dim) + 2);                                         // local knot vector for current dim
            for (auto k = 0; k < dom_dim; k++)
                local_knot_idxs[k].resize(p(k) + 2);

            VectorX<T> param(dom_dim);

            // for start of the curve, for dims prior to current dim, find anchor and param
            // those dims are in control point index space for the current tensor
            for (auto i = 0; i < dim; i++)
            {
                tmesh.knot_idx_ofst(t, t.knot_mins[i], start_ijk(i), i, true, anchor[i]);                     // computes anchor as offset from start of tensor
                param(i)        = tmesh.all_knots[i][anchor[i]];
            }

            // for the start of the curve, for current dim. and higher, find param
            // these dims are in the input point index space
            for (auto i = dim; i < dom_dim; i++)
                param(i)        = input.params->param_grid[i][start_ijk(i)];

            // find tensor product containing the parameters of the start of the curve (may be outside of original tensor)
            bool found          = false;
            TensorIdx found_idx = tmesh.find_tensor(param, t_idx, found);
            auto& found_tensor  = tmesh.tensor_prods[found_idx];
            if (!found)
                throw MFAError(fmt::format("InterpCtrlPtCurve: tensor containing parameter not found. This should not happen\n"));

            // for the start of the curve, find anchor in the found tensor
            // in the current dim, the anchor coordinate will be replaced below by the control point anchor
            for (auto i = 0; i < dom_dim; i++)
            {
                // if param == 0, FindSpan finds the last 0-value knot span, but we may want earlier ones
                if (param(i) == 0.0)
                    anchor[i] = (mfa_data.p(i) + 1) / 2 + start_ijk(i);
                else if (param(i) == 1.0)
                {
                    if (i < dim)
                    {
                        // input points are control points, handle possibly several anchors with value 1
                        anchor[i] = tmesh.all_knots[i].size() - 1 - (mfa_data.p(i) + 1) / 2 - (npts(i) - start_ijk(i) - 1);
                        if (mfa_data.p(i) % 2 == 0)
                            anchor[i]--;
                    }
                    else
                        // input points are original input points, set anchor to first knot with value 1
                        anchor[i] = tmesh.all_knots[i].size() - 1 - mfa_data.p(i);
                }
                else
                    anchor[i] = mfa_data.tmesh.FindSpan(i, param(i), found_tensor);
            }

            // debug
//             fmt::print(stderr, "InterpCtrlPtCurve(): 1: dim {} found_idx {} param [{}] anchor [{}]\n",
//                     dim, found_idx, param.transpose(), fmt::join(anchor, ","));

            for (auto i = 0; i < t.nctrl_pts(dim); i++)                                                 // for all control points in current dim
            {
                // reset anchor and parameter in current dim to anchor of control point
                anchor[dim] = mfa_data.tmesh.ctrl_pt_anchor_dim(dim, t, i);
                param(dim) = tmesh.all_knots[dim][anchor[dim]];

                // find correct tensor in case it needs to be adjusted
                found_idx = mfa_data.tmesh.find_tensor(anchor, found_idx, false, found);
                if (!found)
                    throw MFAError(fmt::format("FreeCtrlPtCurve: tensor containing parameter not found\n"));
                auto& ft = tmesh.tensor_prods[found_idx];

                // debug
//                 fmt::print(stderr, "InterpCtrlPtCurve(): dim {} i {} found_idx {} param [{}] \t\tanchor [{}] start_ijk [{}]\n",
//                         dim, i, found_idx, param.transpose(), fmt::join(anchor, ","), start_ijk.transpose());

                // find control point aligned with curve
                if (tmesh.anchor_matches_param(anchor, param))                                          // control point exists already
                    P.row(i) = ft.ctrl_pts.row(tmesh.anchor_ctrl_pt_idx(ft, anchor));
                else                                                                                    // control point needs to be inserted
                {
                    TensorProduct<T>        new_tensor(ft.knot_mins, ft.knot_maxs, ft.level); // temporary tensor to hold new control points
                    vector<vector<T>>       new_knots;                                                  // temporary new knots after insertion
                    vector<vector<int>>     new_knot_levels;                                            // temporary new knot levels after insertion
                    mfa_data.NewKnotInsertion(
                            param,
                            found_idx,
                            new_tensor.nctrl_pts,
                            new_knots,
                            new_knot_levels,
                            new_tensor.ctrl_pts,
                            new_tensor.weights,
                            inserted_dims);

                    // adjust anchor by inserted dims
                    for (auto j = 0; j < dom_dim; j++)
                    {
                        inserted_anchor[j] = anchor[j] + inserted_dims[j];
                        new_tensor.knot_maxs[j] += inserted_dims[j];
                    }

                    // copy inserted control point into P
                    P.row(i) = new_tensor.ctrl_pts.row(tmesh.anchor_ctrl_pt_idx(new_tensor, inserted_anchor));
                }
            }       // control points
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
                size_t              npts,               // number of input points in current dim, inclusing constraints
                MatrixX<T>&         Nfree)              // (output) matrix of free control points basis functions
        {
            auto&               t = mfa_data.tmesh.tensor_prods[t_idx];
            vector<KnotIdx>     anchor(mfa_data.dom_dim);                                       // control point anchor
            Nfree = MatrixX<T>::Zero(npts, t.nctrl_pts(dim));

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(mfa_data.dom_dim);                          // local knot indices in all dims
            vector<T> local_knots(mfa_data.p(dim) + 2);                                         // local knot vector for current dim
            for (auto k = 0; k < mfa_data.dom_dim; k++)
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);

#if 0

            // TODO: decide whether there is enough parallelism, if so, update TBB for serial version below

// #ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts); // iterator over input points
            VolIterator free_iter(t.nctrl_pts);                         // iterator over free control points

            enumerable_thread_specific<vector<vector<KnotIdx>>> thread_local_knot_idxs(mfa_data.dom_dim);   // local knot idices
            enumerable_thread_specific<vector<vector<T>>>       thread_local_knots(mfa_data.dom_dim);       // local knot idices
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(mfa_data.dom_dim);           // multidim index of domain point
            enumerable_thread_specific<VectorXi>                thread_free_ijk(mfa_data.dom_dim);          // multidim index of control point
            enumerable_thread_specific<vector<KnotIdx>>         thread_anchor(mfa_data.dom_dim);            // anchor of control point
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, dom_iter.tot_iters(), 0, free_iter.tot_iters()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)                                // for control points
                {
                    if (i == r.cols().begin())
                    {
                        for (auto k = 0; k < mfa_data.dom_dim; k++)
                        {
                            thread_local_knot_idxs.local()[k].resize(mfa_data.p(k) + 2);
                            thread_local_knots.local()[k].resize(mfa_data.p(k) + 2);
                        }
                    }

                    free_iter.idx_ijk(i, thread_free_ijk.local());                                      // ijk of domain point
                    mfa_data.tmesh.ctrl_pt_anchor(t, thread_free_ijk.local(), thread_anchor.local());   // anchor of control point

                    // local knot vector
                    mfa_data.tmesh.knot_intersections(thread_anchor.local(), t_idx, true, thread_local_knot_idxs.local());
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                    {
                        for (auto n = 0; n < thread_local_knot_idxs.local()[k].size(); n++)
                            thread_local_knots.local()[k][n] = mfa_data.tmesh.all_knots[k][thread_local_knot_idxs.local()[k][n]];
                    }

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)                            // for input domain points
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                                    // ijk of domain point
                        for (auto k = 0; k < mfa_data.dom_dim; k++)
                        {
                            T u = input.params->param_grid[k][thread_dom_ijk.local()(k)];               // parameter of current input point
                            T B = mfa_data.OneBasisFun(k, u, thread_local_knots.local()[k]);                           // basis function
                            Nfree(j, i) = (k == 0 ? B : Nfree(j, i) * B);
                        }
                    }       // for blocked range rows, ie, input domain points
                }       // for blocked range cols, ie, control points
            }, ap); // parallel for

#else           // serial

            VectorX<T> param(mfa_data.dom_dim);

            // for start of the curve, for dims prior to current dim, find anchor and param
            // those dims are in control point index space for the current tensor
            for (auto i = 0; i < dim; i++)
            {
                mfa_data.tmesh.knot_idx_ofst(t, t.knot_mins[i], start_ijk(i), i, true, anchor[i]);                     // computes anchor as offset from start of tensor
                param(i)    = mfa_data.tmesh.all_knots[i][anchor[i]];
            }

            // for the start of the curve, for current dim. and higher, find param
            // these dims are in the input point index space
            for (auto i = dim; i < mfa_data.dom_dim; i++)
                param(i) = input.params->param_grid[i][start_ijk(i)];

            // find tensor product containing the parameters of the start of the curve (may be outside of original tensor)
            bool found          = false;
            TensorIdx found_idx = mfa_data.tmesh.find_tensor(param, t_idx, found);
            auto& found_tensor  = mfa_data.tmesh.tensor_prods[found_idx];
            if (!found)
                throw MFAError(fmt::format("FreeCtrlPtCurve: tensor containing parameter not found. This should not happen\n"));

            // for the start of the curve, for the current dim. and higher, find anchor
            // these dims are in the input point space
            // in the current dim, the anchor coordinate will be replaced below by the control point anchor
            for (auto i = dim; i < mfa_data.dom_dim; i++)
            {
                // if param == 0, FindSpan finds the last 0-value knot span, but we want the first control point anchor, which is an earlier span
                if (param(i) == 0.0)
                    anchor[i] = (mfa_data.p(i) + 1) / 2;
                else if (param(i) == 1.0)
                    anchor[i] = mfa_data.tmesh.all_knots[i].size() - 2 - (mfa_data.p(i) + 1) / 2;
                else
                    anchor[i] = mfa_data.tmesh.FindSpan(i, param(i), found_tensor);
            }

            // debug
//             fmt::print(stderr, "FreeCtrlPtCurve: dim {} t_idx {} start point anchor [{}]\n", dim, t_idx, fmt::join(anchor, ","));

            for (auto i = 0; i < t.nctrl_pts(dim); i++)                                                 // for all control points in current dim
            {
                // anchor of control point in current dim
                anchor[dim] = mfa_data.tmesh.ctrl_pt_anchor_dim(dim, t, i);

                // local knot vector in currrent dimension
                mfa_data.tmesh.knot_intersections(anchor, t_idx, local_knot_idxs);                  // local knot indices in all dimensions
                for (auto n = 0; n < local_knot_idxs[dim].size(); n++)                                  // local knots in only current dim
                    local_knots[n] = mfa_data.tmesh.all_knots[dim][local_knot_idxs[dim][n]];

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
            vector<KnotIdx>     anchor(mfa_data.dom_dim);                                       // control point anchor
            Nfree = MatrixX<T>::Zero(ndom_pts.prod(), t.ctrl_pts.rows());
            int max_nnz_col = 0;                                                                // max num nonzeros in any column

            // debug
            fmt::print(stderr, "Nfree has {} rows and {} columns\n", Nfree.rows(), Nfree.cols());

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(mfa_data.dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(mfa_data.dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

#ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts); // iterator over input points
            VolIterator free_iter(t.nctrl_pts);                         // iterator over free control points

            enumerable_thread_specific<vector<vector<KnotIdx>>> thread_local_knot_idxs(mfa_data.dom_dim);   // local knot idices
            enumerable_thread_specific<vector<vector<T>>>       thread_local_knots(mfa_data.dom_dim);       // local knot idices
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(mfa_data.dom_dim);           // multidim index of domain point
            enumerable_thread_specific<VectorXi>                thread_free_ijk(mfa_data.dom_dim);          // multidim index of control point
            enumerable_thread_specific<vector<KnotIdx>>         thread_anchor(mfa_data.dom_dim);            // anchor of control point
            enumerable_thread_specific<vector<int>>             thread_nnz_col(Nfree.cols(), 0);            // number of nonzeros in each column
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, dom_iter.tot_iters(), 0, free_iter.tot_iters()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)                                // for control points
                {
                    if (i == r.cols().begin())
                    {
                        for (auto k = 0; k < mfa_data.dom_dim; k++)
                        {
                            thread_local_knot_idxs.local()[k].resize(mfa_data.p(k) + 2);
                            thread_local_knots.local()[k].resize(mfa_data.p(k) + 2);
                        }
                    }

                    free_iter.idx_ijk(i, thread_free_ijk.local());                                      // ijk of domain point
                    mfa_data.tmesh.ctrl_pt_anchor(t, thread_free_ijk.local(), thread_anchor.local());   // anchor of control point

                    // local knot vector
                    mfa_data.tmesh.knot_intersections(thread_anchor.local(), t_idx, thread_local_knot_idxs.local());
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                    {
                        for (auto n = 0; n < thread_local_knot_idxs.local()[k].size(); n++)
                            thread_local_knots.local()[k][n] = mfa_data.tmesh.all_knots[k][thread_local_knot_idxs.local()[k][n]];
                    }

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)                            // for input domain points
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                                    // ijk of domain point
                        for (auto k = 0; k < mfa_data.dom_dim; k++)
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
                VectorXi ijk(mfa_data.dom_dim);                                                 // ijk of current control point
                free_iter.idx_ijk(free_iter.cur_iter(), ijk);

                // anchor of control point
                mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchor, t_idx, local_knot_idxs);
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                int nnz_col = 0;                                                                // num nonzeros in current column

                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts);                     // iterator over input points
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < mfa_data.dom_dim; k++)                                 // for all dims
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
            vector<KnotIdx>     anchor(mfa_data.dom_dim);                                       // control point anchor

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(mfa_data.dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(mfa_data.dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

            // iterator over free control points
            VolIterator free_iter(t.nctrl_pts);
            while (!free_iter.done())
            {
                VectorXi ijk(mfa_data.dom_dim);                                                 // ijk of current control point
                free_iter.idx_ijk(free_iter.cur_iter(), ijk);

                // anchor of control point
                mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);

                // local knot vector
                mfa_data.tmesh.knot_intersections(anchor, t_idx, true, local_knot_idxs);
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts);
                while (!dom_iter.done())
                {
                    T v;                                                                        // basis function value
                    for (auto k = 0; k < mfa_data.dom_dim; k++)                                 // for all dims
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

        // computes curve of constraint control points basis functions in one dimension
        // matrix Ncons of basis functions is m rows x n columns, where
        //  m is number of input points covered by constraints and free control points
        //  n is number of constraint control points for one curve (both left and right constraints)
        // assumes Ncons was allocated by caller to correct size for number of constraints
        // the curve of input points may not be in the tensor of control points, but can be in a neighbor if the
        //  curve is in the constraint region in a dim. orthogonal to the current dim.
        // helper function for EncodeTensorLocalSeparable
        void ConsCtrlPtCurve(
                int                 dim,                // current dimension
                TensorIdx           t_idx,              // index of original tensor of (free) control points
                const VectorXi&     start_ijk,          // multidim index of first input point for the curve, including constraints
                const VectorXi&     npts,               // number of input points (could be control points for input in some dims) including constraints
                ConsType            cons_type,          // none, left, right, both
                MatrixX<T>&         Ncons,              // (output) matrix of constraint control points basis functions
                MatrixX<T>&         Pcons)              // (output) matrix of constraint control points
        {
            if (cons_type == ConsType::MFA_NO_CONSTRAINT)
                return;

            // find the left constraints
            if (cons_type == ConsType::MFA_LEFT_ONLY_CONSTRAINT || cons_type == ConsType::MFA_BOTH_CONSTRAINT)
                PrevConsCtrlPtCurve(dim, t_idx, start_ijk, npts, Ncons, Pcons);

            // find the right constraints
            if (cons_type == ConsType::MFA_RIGHT_ONLY_CONSTRAINT)
                NextConsCtrlPtCurve(dim, t_idx, start_ijk, npts, 0, Ncons, Pcons);
            else if (cons_type == ConsType::MFA_BOTH_CONSTRAINT)
                NextConsCtrlPtCurve(dim, t_idx, start_ijk, npts, mfa_data.p(dim), Ncons, Pcons);
        }

        // computes curve of previous constraint control points basis functions in one dimension
        // matrix Ncons of basis functions is m rows x n columns, where
        //  m is number of input points covered by constraints and free control points
        //  n is number of constraint control points for one curve (both left and right constraints)
        // assumes Ncons was allocated by caller to correct size for number of constraints
        // the curve of input points may not be in the tensor of control points, but can be in a neighbor if the
        //  curve is in the constraint region in a dim. orthogonal to the current dim.
        // helper function for EncodeTensorLocalSeparable
        void PrevConsCtrlPtCurve(
                int                 dim,                // current dimension
                TensorIdx           t_idx,              // index of original tensor of (free) control points
                const VectorXi&     start_ijk,          // multidim index of first input point for the curve, including constraints
                const VectorXi&     npts,               // number of input points (could be control points for input in some dims) including constraints
                MatrixX<T>&         Ncons,              // (output) matrix of constraint control points basis functions
                MatrixX<T>&         Pcons)              // (output) matrix of constraint control points
        {
            // typing shortcuts
            auto& dom_dim       = mfa_data.dom_dim;
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;
            auto& p             = mfa_data.p;
            auto& t             = tensor_prods[t_idx];

            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor
            vector<KnotIdx>     inserted_anchor(dom_dim);                              // inserted control point anchor
            vector<int>         inserted_dims(dom_dim);                                // which dims actually added a knot and ctrl pt

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices in all dims
            vector<T> local_knots(p(dim) + 2);                                         // local knot vector for current dim
            for (auto k = 0; k < dom_dim; k++)
                local_knot_idxs[k].resize(p(k) + 2);

            VectorX<T> param(dom_dim);
            VectorX<T> param_eps(dom_dim);                                              // param + small epsilon
            T eps = 1.0e-6;

            // for start of the curve, for dims prior to current dim, find anchor and param
            // those dims are in control point index space for the current tensor
            for (auto i = 0; i < dim; i++)
            {
                tmesh.knot_idx_ofst(t, t.knot_mins[i], start_ijk(i), i, true, anchor[i]);                     // computes anchor as offset from start of tensor
                param(i)        = tmesh.all_knots[i][anchor[i]];
                param_eps(i)    = anchor[i] == t.knot_mins[i] ? param(i) + eps : param(i);
            }

            // for the start of the curve, for current dim. and higher, find param
            // these dims are in the input point index space
            for (auto i = dim; i < dom_dim; i++)
            {
                param(i)        = input.params->param_grid[i][start_ijk(i)];
                param_eps(i)    = param(i);
            }

            // find tensor product containing the parameters of the start of the curve (may be outside of original tensor)
            bool found          = false;
            TensorIdx found_idx = tmesh.find_tensor(param_eps, t_idx, found);
            auto& found_tensor  = tmesh.tensor_prods[found_idx];
            if (!found)
                throw MFAError(fmt::format("PrevConsCtrlPtCurve: tensor containing parameter not found. This should not happen\n"));

            // for the start of the curve, find anchor in the found tensor
            // in the current dim, the anchor coordinate will be replaced below by the control point anchor
            for (auto i = 0; i < dom_dim; i++)
            {
                // if param == 0, FindSpan finds the last 0-value knot span, but we may want earlier ones
                if (param(i) == 0.0)
                    anchor[i] = (mfa_data.p(i) + 1) / 2 + start_ijk(i);
                else if (param(i) == 1.0)
                {
                    if (i < dim)
                    {
                        // input points are control points, handle possibly several anchors with value 1
                        anchor[i] = tmesh.all_knots[i].size() - 1 - (mfa_data.p(i) + 1) / 2 - (npts(i) - start_ijk(i) - 1);
                        if (mfa_data.p(i) % 2 == 0)
                            anchor[i]--;
                    }
                    else
                        // input points are original input points, set anchor to first knot with value 1
                        anchor[i] = tmesh.all_knots[i].size() - 1 - mfa_data.p(i);
                }
                else
                    anchor[i] = mfa_data.tmesh.FindSpan(i, param(i), found_tensor);
            }

            // debug
//             fmt::print(stderr, "PrevConsCtrlPtCurve(): 1: dim {} found_idx {} param [{}] anchor [{}]\n",
//                     dim, found_idx, param.transpose(), fmt::join(anchor, ","));

            for (auto i = 0; i < p(dim); i++)                                                           // for all constraint control points in current dim
            {
                // reset parameter in current dim to anchor of control point
                param(dim) = tmesh.all_knots[dim][anchor[dim]];

                // debug
                fmt::print(stderr, "PrevConsCtrlPtCurve(): dim {} found_idx {} param [{}] \t\tanchor [{}] start_ijk [{}]\n",
                        dim, found_idx, param.transpose(), fmt::join(anchor, ","), start_ijk.transpose());

                // local knot vector in currrent dimension
                tmesh.knot_intersections(anchor, found_idx, local_knot_idxs);                           // local knot indices in all dimensions
                for (auto n = 0; n < local_knot_idxs[dim].size(); n++)                                  // local knots in only current dim
                    local_knots[n] = tmesh.all_knots[dim][local_knot_idxs[dim][n]];

                // write Ncons
                for (auto j = 0; j < npts(dim); j++)                                                         // for all input points (for this tensor) in current dim
                {
                    T u = input.params->param_grid[dim][start_ijk(dim) + j];                            // parameter of current input point
                    Ncons(j, i) = mfa_data.OneBasisFun(dim, u, local_knots);                            // basis function
                }

                // find constraint control point aligned with curve
                if (tmesh.anchor_matches_param(anchor, param))                                          // control point exists already
                    Pcons.row(i) = found_tensor.ctrl_pts.row(tmesh.anchor_ctrl_pt_idx(found_tensor, anchor));
                else                                                                                    // control point needs to be inserted
                {
                    TensorProduct<T>        new_tensor(found_tensor.knot_mins, found_tensor.knot_maxs, found_tensor.level); // temporary tensor to hold new control points
                    vector<vector<T>>       new_knots;                                                  // temporary new knots after insertion
                    vector<vector<int>>     new_knot_levels;                                            // temporary new knot levels after insertion
                    mfa_data.NewKnotInsertion(
                            param,
                            found_idx,
                            new_tensor.nctrl_pts,
                            new_knots,
                            new_knot_levels,
                            new_tensor.ctrl_pts,
                            new_tensor.weights,
                            inserted_dims);

                    // adjust anchor by inserted dims
                    for (auto j = 0; j < dom_dim; j++)
                        inserted_anchor[j] = anchor[j] + inserted_dims[j];

                    // copy inserted control point into Pcons
                    Pcons.row(i) = new_tensor.ctrl_pts.row(tmesh.anchor_ctrl_pt_idx(new_tensor, inserted_anchor));
                }

                // offset anchor for next constraint
                if (i < p(dim) - 1)
                {
                    if (!tmesh.knot_idx_ofst(found_tensor, anchor[dim], 1, dim, true, anchor[dim]))
                        throw MFAError(fmt::format("PrevConsCtrlPtCurve(): cannot offset anchor for next constraint\n"));
                }
            }       // control points
        }

        // computes curve of next constraint control points basis functions in one dimension
        // matrix Ncons of basis functions is m rows x n columns, where
        //  m is number of input points covered by constraints and free control points
        //  n is number of constraint control points for one curve (both left and right constraints)
        // assumes Ncons was allocated by caller to correct size for number of constraints
        // the curve of input points may not be in the tensor of control points, but can be in a neighbor if the
        //  curve is in the constraint region in a dim. orthogonal to the current dim.
        // helper function for EncodeTensorLocalSeparable
        void NextConsCtrlPtCurve(
                int                 dim,                // current dimension
                TensorIdx           t_idx,              // index of original tensor of (free) control points
                const VectorXi&     start_ijk,          // multidim index of first input point for the curve, including constraints
                const VectorXi&     npts,               // number of input points (could be control points for input in some dims) including constraints
                int                 ofst,               // column-offset to start writing into Ncons (and row-offset for Pcons)
                MatrixX<T>&         Ncons,              // (output) matrix of constraint control points basis functions
                MatrixX<T>&         Pcons)              // (output) matrix of constraint control points
        {
            // typing shortcuts
            auto& dom_dim       = mfa_data.dom_dim;
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;
            auto& p             = mfa_data.p;
            auto& t             = tensor_prods[t_idx];

            vector<KnotIdx>     anchor(dom_dim);                                       // control point anchor
            vector<KnotIdx>     inserted_anchor(dom_dim);                              // inserted control point anchor
            vector<int>         inserted_dims(dom_dim);                                // which dims actually added a knot and ctrl pt

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(dom_dim);                          // local knot indices in all dims
            vector<T> local_knots(p(dim) + 2);                                         // local knot vector for current dim
            for (auto k = 0; k < dom_dim; k++)
                local_knot_idxs[k].resize(p(k) + 2);

            VectorX<T> param(dom_dim);
            VectorX<T> param_eps(dom_dim);                                              // param + small epsilon
            T eps = 1.0e-6;

            // for start of the curve, for dims prior to current dim, find anchor and param
            // those dims are in control point index space for the current tensor
            for (auto i = 0; i < dim; i++)
            {
                tmesh.knot_idx_ofst(t, t.knot_mins[i], start_ijk(i), i, true, anchor[i]);                   // computes anchor as offset from start of tensor
                param(i)        = tmesh.all_knots[i][anchor[i]];
                param_eps(i)    = anchor[i] == t.knot_mins[i] ? param(i) + eps : param(i);
            }

            // for the start of the curve, for current dim., make the param just past the max of the tensor
            param(dim)      = tmesh.all_knots[dim][t.knot_maxs[dim]] + eps;
            param_eps(dim)  = param(dim);

            // for the start of the curve, for dims after current, find param
            // these dims are in the input point index space
            for (auto i = dim + 1; i < dom_dim; i++)
            {
                param(i)        = input.params->param_grid[i][start_ijk(i)];
                param_eps(i)    = param(i);
            }

            // find tensor product containing the parameters of the start of the constraints (may be outside of original tensor)
            bool found          = false;
            TensorIdx found_idx = tmesh.find_tensor(param_eps, t_idx, found);
            auto& found_tensor  = tensor_prods[found_idx];
            if (!found)
                throw MFAError(fmt::format("NextConsCtrlPtCurve: tensor containing parameter not found. This should not happen\n"));

            // for the start of the curve, find anchor in the found tensor
            // in the current dim, the anchor coordinate will be replaced below by the control point anchor
            for (auto i = 0; i < dom_dim; i++)
            {
                // if param == 0, FindSpan finds the last 0-value knot span, but we want the first control point anchor, which is an earlier span
                if (param(i) == 0.0)
                    anchor[i] = (mfa_data.p(i) + 1) / 2 + start_ijk(i);
                else if (param(i) == 1.0)
                {
                    if (i < dim)
                    {
                        // input points are control points, handle possibly several anchors with value 1
                        anchor[i] = tmesh.all_knots[i].size() - 1 - (mfa_data.p(i) + 1) / 2 - (npts(i) - start_ijk(i) - 1);
                        if (mfa_data.p(i) % 2 == 0)
                            anchor[i]--;
                    }
                    else
                        // input points are original input points, set anchor to first knot with value 1
                        anchor[i] = tmesh.all_knots[i].size() - 1 - mfa_data.p(i);
                }
                else
                    anchor[i] = mfa_data.tmesh.FindSpan(i, param(i), found_tensor);
            }

            // in the current dim, if odd degree, starting anchor needs to be one farther
            // because there are p constraints after the first knot, which is at the tensor min. boundary
            // this is not an issue for even degree, where anchors are in the spaces instead of on knot lines
            if (mfa_data.p(dim) % 2 == 1 && !tmesh.knot_idx_ofst(found_tensor, anchor[dim], 1, dim, true, anchor[dim]))
                    throw MFAError(fmt::format("NextConsCtrlPtCurve(): cannot offset anchor for next constraint\n"));

            // debug
//             fmt::print(stderr, "NextConsCtrlPtCurve(): 1: dim {} found_idx {} param [{}] anchor [{}]\n",
//                     dim, found_idx, param.transpose(), fmt::join(anchor, ","));

            for (auto i = 0; i < p(dim); i++)                                                           // for all constraint control points in current dim
            {
                // reset parameter in current dim to anchor of control point
                param(dim) = tmesh.all_knots[dim][anchor[dim]];

                // debug
                fmt::print(stderr, "NextConsCtrlPtCurve(): dim {} found_idx {} param [{}] \t\tanchor [{}] start_ijk [{}]\n",
                        dim, found_idx, param.transpose(), fmt::join(anchor, ","), start_ijk.transpose());

                // local knot vector in currrent dimension
                tmesh.knot_intersections(anchor, found_idx, local_knot_idxs);                           // local knot indices in all dimensions
                for (auto n = 0; n < local_knot_idxs[dim].size(); n++)                                  // local knots in only current dim
                    local_knots[n] = tmesh.all_knots[dim][local_knot_idxs[dim][n]];

                // write Ncons
                for (auto j = 0; j < npts(dim); j++)                                                         // for all input points (for this tensor) in current dim
                {
                    T u = input.params->param_grid[dim][start_ijk(dim) + j];                            // parameter of current input point
                    Ncons(j, ofst + i) = mfa_data.OneBasisFun(dim, u, local_knots);                     // basis function
                }

                // find constraint control point aligned with curve
                if (tmesh.anchor_matches_param(anchor, param))                                          // control point exists already
                    Pcons.row(ofst + i) = found_tensor.ctrl_pts.row(tmesh.anchor_ctrl_pt_idx(found_tensor, anchor));
                else                                                                                    // control point needs to be inserted
                {
                    TensorProduct<T>        new_tensor(found_tensor.knot_mins, found_tensor.knot_maxs, found_tensor.level);     // temporary tensor to hold new control points
                    vector<vector<T>>       new_knots;                                                  // temporary new knots after insertion
                    vector<vector<int>>     new_knot_levels;                                            // temporary new knot levels after insertion
                    mfa_data.NewKnotInsertion(
                            param,
                            found_idx,
                            new_tensor.nctrl_pts,
                            new_knots,
                            new_knot_levels,
                            new_tensor.ctrl_pts,
                            new_tensor.weights,
                            inserted_dims);

                    // adjust anchor by inserted dims
                    for (auto j = 0; j < dom_dim; j++)
                        inserted_anchor[j] = anchor[j] + inserted_dims[j];

                    // copy inserted control point into Pcons
                    Pcons.row(ofst + i) = new_tensor.ctrl_pts.row(tmesh.anchor_ctrl_pt_idx(new_tensor, inserted_anchor));
                }

                // offset anchor for next constraint
                if (i < p(dim) - 1)
                {
                    if (!tmesh.knot_idx_ofst(found_tensor, anchor[dim], 1, dim, true, anchor[dim]))
                        throw MFAError(fmt::format("NextConsCtrlPtCurve(): cannot offset anchor for next constraint\n"));
                }
            }       // control points
        }

        // constraint control points matrix of basis functions
        // helper function for EncodeTensorLocalUnified
        // Ncons needs to be sized correctly by caller
        void ConsCtrlPtMat(VectorXi&                ndom_pts,           // number of relevant input points in each dim
                           VectorXi&                dom_starts,         // starting offsets of relevant input points in each dim
                           vector<vector<KnotIdx>>& anchors,            // anchors of constraint control points                                                    // corresponding anchors
                           vector<TensorIdx>&       t_idx_anchors,      // tensors containing corresponding anchors
                           MatrixX<T>&              Ncons)              // (output) matrix of constraint control points basis functions
        {
            Ncons = MatrixX<T>::Constant(ndom_pts.prod(), Ncons.cols(), -1);         // basis functions, -1 means unassigned so far

            // debug
            fmt::print(stderr, "Ncons has {} rows and {} columns\n", Ncons.rows(), Ncons.cols());

            // local knot vector
            vector<vector<KnotIdx>> local_knot_idxs(mfa_data.dom_dim);                          // local knot indices
            vector<vector<T>> local_knots(mfa_data.dom_dim);                                    // local knot vector for current dim in parameter space
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                local_knot_idxs[k].resize(mfa_data.p(k) + 2);
                local_knots[k].resize(mfa_data.p(k) + 2);
            }

#ifdef MFA_TBB  // TBB

            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts);                         // iterator over input points

            enumerable_thread_specific<vector<vector<KnotIdx>>> thread_local_knot_idxs(mfa_data.dom_dim);   // local knot idices
            enumerable_thread_specific<vector<vector<T>>>       thread_local_knots(mfa_data.dom_dim);       // local knot idices
            enumerable_thread_specific<VectorXi>                thread_dom_ijk(mfa_data.dom_dim);   // multidim index of domain point
            static affinity_partitioner                         ap;
            parallel_for (blocked_range2d<size_t>(0, Ncons.rows(), 0, Ncons.cols()), [&] (blocked_range2d<size_t>& r)
            {
                for (auto i = r.cols().begin(); i < r.cols().end(); i++)
                {
                    if (i == r.cols().begin())
                    {
                        for (auto k = 0; k < mfa_data.dom_dim; k++)
                        {
                            thread_local_knot_idxs.local()[k].resize(mfa_data.p(k) + 2);
                            thread_local_knots.local()[k].resize(mfa_data.p(k) + 2);
                        }
                    }

                    // local knot vector
                    mfa_data.tmesh.knot_intersections(anchors[i], t_idx_anchors[i], thread_local_knot_idxs.local());
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                    {
                        for (auto n = 0; n < thread_local_knot_idxs.local()[k].size(); n++)
                            thread_local_knots.local()[k][n] = mfa_data.tmesh.all_knots[k][thread_local_knot_idxs.local()[k][n]];
                    }

                    for (auto j = r.rows().begin(); j < r.rows().end(); j++)
                    {
                        dom_iter.idx_ijk(j, thread_dom_ijk.local());                        // ijk of domain point
                        for (auto k = 0; k < mfa_data.dom_dim; k++)
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
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    for (auto n = 0; n < local_knot_idxs[k].size(); n++)
                        local_knots[k][n] = mfa_data.tmesh.all_knots[k][local_knot_idxs[k][n]];

                // iterator over input points
                VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts);
                while (!dom_iter.done())
                {
                    for (auto k = 0; k < mfa_data.dom_dim; k++)                                 // for all dims
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
            fmt::print(stderr, "EncodeTensorLocalUnified tidx = {}\n", t_idx);

            // debug
            bool debug = false;

            // timing
            double setup_time   = MPI_Wtime();
            double q_time       = MPI_Wtime();
            fmt::print(stderr, "\nSetting up...\n");

            TensorProduct<T>& t = mfa_data.tmesh.tensor_prods[t_idx];                               // current tensor product
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                                   // control point dimensionality

            // get input domain points covered by the tensor and the constraints
            vector<size_t> start_idxs(mfa_data.dom_dim);
            vector<size_t> end_idxs(mfa_data.dom_dim);
            mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, true, start_idxs, end_idxs);

            // Q matrix of relevant input domain points
            VectorXi ndom_pts(mfa_data.dom_dim);
            VectorXi dom_starts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];                                                    // need Eigen vector from STL vector
            }
            MatrixX<T> Q(ndom_pts.prod(), pt_dim);
            VolIterator dom_iter(ndom_pts, dom_starts, input.ndom_pts);
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
            LocalSolveAllConstraints(t, Pcons, anchors, t_idx_anchors);

            // debug
//             fmt::print(stderr, "Pcons.rows = {} t_idx_anchors.size() = {}\n", Pcons.rows(), t_idx_anchors.size());

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

            // TODO: understand why the second version of normalization gives a worse answer than the first
            // and why not normalizing at all also gives a good answer, very close to the first version
            // then pick one of these versions and remove the other

#if 1

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
                    VectorXi ijk(mfa_data.dom_dim);
                    dom_iter.idx_ijk(i, ijk);
                    cerr << "ijk = " << ijk.transpose() << endl;
                    fmt::print(stderr, "params = [ ");
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                        fmt::print(stderr, "{} ", input.params->param_grid[k][ijk(k)]);
                    fmt::print(stderr, "]\n");
                }
            }

#else

            {
                bool error = false;
                T Nfree_sum, Ncons_sum;
                Nfree_sum = Nfree.row(i).sum();
                if (Pcons.rows())
                    Ncons_sum = Ncons.row(i).sum();
                if (Nfree_sum + Ncons_sum == 0.0)       // either Nfree or Ncons or both have to have a nonzero row sum
                {
                    VectorXi ijk(mfa_data.dom_dim);
                    dom_iter.idx_ijk(i, ijk);
                    cerr << "ijk = " << ijk.transpose() << endl;
                    fmt::print(stderr, "params = [ ");
                    for (auto k = 0; k < mfa_data.dom_dim; k++)
                        fmt::print(stderr, "{} ", input.params->param_grid[k][ijk(k)]);
                    fmt::print(stderr, "]\n");
                    abort();
                }
                if (Nfree_sum > 0.0)
                    Nfree.row(i) /= Nfree_sum;
                if (Ncons_sum > 0.0)
                    Ncons.row(i) /= Ncons_sum;
            }

#endif

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

            // debug: collect some metrics about N
//             size_t nonzeros = (Nfree.array() > 0).count();
//             fmt::print(stderr, "EncodeTensorLocalUnified Nfree matrix: {} rows x {} cols = {} entries of which {} are nonzero ({})\n",
//                     Nfree.rows(), Nfree.cols(), Nfree.rows() * Nfree.cols(), nonzeros, float(nonzeros) / (float)(Nfree.rows() * Nfree.cols()));

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
            cerr << "EncodeTensorLocalLinar(): The relative error is " << relative_error << endl;
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
            // debug
            fmt::print(stderr, "EncodeTensorLocalSeparable tidx = {}\n", t_idx);
//             fmt::print(stderr, "\n Current T-mesh:\n");
//             mfa_data.tmesh.print(true, true);

            double t0 = MPI_Wtime();

            // typing shortcuts
            auto& dom_dim       = mfa_data.dom_dim;
            auto& tmesh         = mfa_data.tmesh;
            auto& tensor_prods  = tmesh.tensor_prods;

            auto& t = tensor_prods[t_idx];                                                          // current tensor product
            int pt_dim = mfa_data.max_dim - mfa_data.min_dim + 1;                                   // control point dimensionality

            // get input domain points covered by the tensor, including constraints
            vector<size_t> start_idxs(dom_dim);                                                     // start of input points including constraints
            vector<size_t> end_idxs(dom_dim);                                                       // end of input points including constraints
            mfa_data.tmesh.domain_pts(t_idx, input.params->param_grid, true, start_idxs, end_idxs); // true: extend to cover constraints

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

            // two matrices of input points for subsequent dimensions after dimension 0, which draws directly from input domain
            // input points cover constraints as well as free control point basis functions
            // (double buffering output control points to input points)
            VectorXi npts = ndom_pts;                                                               // number of output points in current dim = input pts for next dim
            MatrixX<T> Q(npts.prod(), pt_dim);                                                      // first matrix size of input points
            npts(0) = t.nctrl_pts(0);
            MatrixX<T> Q1(npts.prod(), pt_dim);                                                     // second matrix already smaller, size of ctrl pts in first dim

            // debug
//             fmt::print(stderr, "input domain points covered by tensor and constraints:\n");
//             for (auto k = 0; k < dom_dim; k++)
//                 fmt::print(stderr, "param_start[{}] = {} param_end[{}] = {} ", k, input.params->param_grid[k][dom_starts(k)],
//                         k, input.params->param_grid[k][dom_starts(k) + ndom_pts(k) - 1]);
//             fmt::print(stderr, "\n");

            // input and output number of points
            VectorXi nin_pts    = ndom_pts;
            VectorXi nout_pts   = npts;
            VectorXi in_starts  = dom_starts;
            VectorXi in_all_pts = input.ndom_pts;
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
#if 1
                cons_type = ConsType::MFA_NO_CONSTRAINT;
#else

                if (t.knot_mins[dim] == 0 && t.knot_maxs[dim] == tmesh.all_knots[dim].size() - 1)
                    cons_type = ConsType::MFA_NO_CONSTRAINT;
                else if (t.knot_mins[dim] == 0)
                {
                    cons_type   = ConsType::MFA_RIGHT_ONLY_CONSTRAINT;
                    Ncons       = MatrixX<T>::Zero(nin_pts(dim), mfa_data.p(dim));
                    Pcons       = MatrixX<T>::Zero(mfa_data.p(dim), t.ctrl_pts.cols());
                }
                else if (t.knot_maxs[dim] == tmesh.all_knots[dim].size() - 1)
                {
                    cons_type   = ConsType::MFA_LEFT_ONLY_CONSTRAINT;
                    Ncons       = MatrixX<T>::Zero(nin_pts(dim), mfa_data.p(dim));
                    Pcons       = MatrixX<T>::Zero(mfa_data.p(dim), t.ctrl_pts.cols());
                }
                else
                {
                    cons_type   = ConsType::MFA_BOTH_CONSTRAINT;
                    Ncons       = MatrixX<T>::Zero(nin_pts(dim),  2 * mfa_data.p(dim));
                    Pcons       = MatrixX<T>::Zero(2 * mfa_data.p(dim), t.ctrl_pts.cols());
                }
#endif
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
                        ComputeCtrlPtCurve(in_curve_iter, t_idx, dim, R, Q, Q1, Nfree, Ncons, Pcons, P,
                                cons_type, nin_pts, start_ijk, free_time, cons_time, norm_time, solve_time);
                    else
                        InterpCtrlPtCurve(dim, t_idx, start_ijk, nin_pts, P);

                    // copy solution to one curve of output points
                    CurveIterator   out_curve_iter(out_slice_iter);                                     // one curve of the output points in the current dim
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
                }       // for all curves

                // adjust input, output numbers of points for next iteration
                nin_pts(dim) = t.nctrl_pts(dim);
                if (dim < dom_dim - 1)
                    nout_pts(dim + 1) = t.nctrl_pts(dim + 1);
                in_starts   = VectorXi::Zero(dom_dim);                  // subvolume = full volume for later dims
                in_all_pts  = nin_pts;
            }       // for all domain dimensions

            // copy final result back to tensor product
            if (dom_dim % 2 == 0)
                t.ctrl_pts = Q.block(0, 0, t.nctrl_pts.prod(), pt_dim);
            else
                t.ctrl_pts = Q1.block(0, 0, t.nctrl_pts.prod(), pt_dim);

            // timing
            fmt::print(stderr, "EncodeTensorLocalSeparable() time {} s.\n", MPI_Wtime() - t0);
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
            fmt::print(stderr, "Using OrigAdaptiveEncode() w/ 1-d curve knot splitting\n");
#ifdef MFA_CHECK_ALL_CURVES
            fmt::print(stderr, "Checking all curves (slower but more accurate)\n");
#else
            fmt::print(stderr, "Checking a sampling of curves (faster but less accurate)\n");
#endif

            // TODO: use weights for knot insertion
            // for now, weights are only used for final full encode

            // loop until no change in knots
            for (int iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; max iterations reached.\n\n");
                    break;
                }

                if (verbose)
                    fprintf(stderr, "\n--- Iteration %d ---\n", iter);

                // low-d w/ splitting spans in the middle
                bool done = OrigNewKnots_curve(new_knots, err_limit, extents, iter);

                // no new knots to be added
                if (done)
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; no new knots added.\n\n");
                    break;
                }

                // check if the new knots would make the number of control points >= number of input points in any dim
                done = false;
                for (auto k = 0; k < mfa_data.dom_dim; k++)
                    // hard-coded for first tensor
                    if (input.ndom_pts(k) <= mfa_data.tmesh.tensor_prods[0].nctrl_pts(k) + new_knots[k].size())
                    {
                        done = true;
                        break;
                    }
                if (done)
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; control points would outnumber input points.\n\n");
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
                for (auto j = 0; j < mfa_data.dom_dim; j++)
                    t.nctrl_pts[j] = mfa_data.tmesh.all_knots[j].size() - mfa_data.p(j) - 1;
                Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);

                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; max iterations reached.\n\n");

                    // debug
//                     fmt::print(stderr, "\nFinal Tmesh\n\n");
//                     mfa_data.tmesh.print();

                    break;
                }

                if (verbose)
                    fprintf(stderr, "\n--- Iteration %d ---\n", iter);

                // debug
//                 fmt::print(stderr, "\nTmesh at start of iteration\n\n");
//                 mfa_data.tmesh.print();

                // check all knots spans for error
                vector<vector<KnotIdx>>     inserted_knot_idxs(mfa_data.dom_dim);   // indices in each dim. of inserted knots in full knot vector after insertion
                vector<vector<T>>           inserted_knots(mfa_data.dom_dim);       // knots to be inserted in each dim.
                vector<TensorIdx>           parent_tensor_idxs;                     // tensors having knots inserted
                bool done = nk.AllErrorSpans(
                        myextents,
                        err_limit,
                        true,
                        parent_tensor_idxs,
                        inserted_knot_idxs,
                        inserted_knots,
                        error_stats);

                // no new knots to be added
                if (done)
                {
                    if (verbose)
                        fprintf(stderr, "\nDone; no new knots added.\n\n");

                    // debug
//                     fmt::print(stderr, "\nFinal Tmesh\n\n");
//                     mfa_data.tmesh.print();

                    break;
                }

                // insert new knots into knot vectors
                int n_insertions = parent_tensor_idxs.size();                       // number of knots to insert
                for (auto j = 0; j < mfa_data.dom_dim; j++)
                    assert(inserted_knot_idxs[j].size() == n_insertions &&
                            inserted_knots[j].size() == n_insertions);

                vector<bool> inserted(mfa_data.dom_dim);                            // whether the current insertion succeed (in each dim)

                for (auto i = 0; i < n_insertions; i++)                             // for all knots to be inserted
                {
                    // debug
//                     fmt::print(stderr, "Knot insertion {} of {}: ", i, n_insertions);
//                     fmt::print(stderr, "\nTrying to insert knot idx [ ");
//                     for (auto j = 0; j < mfa_data.dom_dim; j++)
//                         fmt::print(stderr, "{} ", inserted_knot_idxs[j][i]);
//                     fmt::print(stderr, "] with value [ ");
//                     for (auto j = 0; j < mfa_data.dom_dim; j++)
//                         fmt::print(stderr, "{} ", inserted_knots[j][i]);
//                     fmt::print(stderr, "]\n");

                    // insert the new knots into tmesh all_knots
                    for (auto j = 0; j < mfa_data.dom_dim; j++)
                    {
                        inserted[j] = false;
                        if (mfa_data.tmesh.insert_knot_at_pos(j,
                                    inserted_knot_idxs[j][i],
                                    0,                                              // all knots at level 0 in this version
                                    inserted_knots[j][i], input.params->param_grid))
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
                T                   err_limit,                  // maximum allowable normalized error
                bool                weighted,                   // solve for and use weights
                const VectorX<T>&   extents,                    // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds = 0)             // optional maximum number of rounds
        {
            int parent_level = 0;                               // parent level currently being refined

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
//             fprintf(stderr, "\n----- initial T-mesh -----\n\n");
//             mfa_data.tmesh.print();
//             fprintf(stderr, "--------------------------\n\n");

            vector<TensorProduct<T>>    new_tensors;                            // newly refined tensors to be added

            // loop until all tensors are done
            int iter;
            for (iter = 0; ; iter++)
            {
                if (max_rounds > 0 && iter >= max_rounds)               // optional cap on number of rounds
                    break;

                if (verbose)
                {
                    fmt::print(stderr, "\n--- Iteration {} ---\n", iter);
                    fmt::print(stderr, "Refining level {} with {} new tensors so far\n", parent_level, new_tensors.size());
                }

                // debug
//                 fmt::print(stderr, "\nTmesh before refinement\n\n");
//                 mfa_data.tmesh.print(true, true, false, false);

                bool done_parent_level = Refine(parent_level, iter + 1, err_limit, extents, new_tensors);

                if (done_parent_level)
                {
                    if (parent_level >= mfa_data.tmesh.max_level)
                    {
                        if (verbose)
                            fprintf(stderr, "\nKnot insertion done after %d iterations; no new knots added.\n\n", iter + 1);
                        break;
                    }
                    else                                                        // one iteration only is done
                    {
                        fmt::print(stderr, "Level {} done, adding {} new tensors\n", parent_level, new_tensors.size());
                        double add_tensors_time = MPI_Wtime();
                        AddNewTensors(new_tensors);
                        add_tensors_time = MPI_Wtime() - add_tensors_time;
                        fmt::print(stderr, "Solving and adding new tensors time:   {} s.\n", add_tensors_time);
                        parent_level++;
                        new_tensors.clear();
                    }
                }
            }   // iterations

            fmt::print(stderr, "{} iterations.\n", iter + 1);
            fmt::print(stderr, "{} tensor products.\n", mfa_data.tmesh.tensor_prods.size());

            // debug: print tmesh
//             fprintf(stderr, "\n----- final T-mesh -----\n\n");
//             mfa_data.tmesh.print(true, true, false, false);
//             fprintf(stderr, "--------------------------\n\n");

            // debug: check all spans
            // TODO: comment out after code is debugged
            mfa::NewKnots<T> nk(mfa_data, input);
            if (!nk.CheckAllSpans())
            {
                fmt::print(stderr, "AdaptiveEncode(): Error: failed checking all spans for input points\n");
                abort();
            }

            // debug: check if all tensors are marked done
            // TODO: comment out after code is debugged
            bool all_done = true;
            for (auto i = 0; i < mfa_data.tmesh.tensor_prods.size(); i++)
            {
                bool first = true;
                auto& t = mfa_data.tmesh.tensor_prods[i];
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
        void RHSUnified(
            // const vector<size_t>&   start_idxs,
            // const vector<size_t>&   end_idxs,
            SparseMatrixX<T>&       Nt,
            MatrixX<T>&             R)
        {
            // REQUIRED for TMesh
            // VectorXi ndom_pts(mfa_data.dom_dim);
            // VectorXi dom_starts(mfa_data.dom_dim);
            // for (auto k = 0; k < mfa_data.dom_dim; k++)
            // {
            //     ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
            //     dom_starts(k)   = start_idxs[k];
            // }

            if (R.cols() != mfa_data.max_dim - mfa_data.min_dim + 1)
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (cols)" << endl;
            if (R.rows() != input.npts)
                cerr << "Error: Incorrect matrix dimensions in RHSUnified (rows)" << endl;

            VectorX<T> pt_coords(input.dom_dim);
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
            VectorXi ndom_pts(mfa_data.dom_dim);
            VectorXi dom_starts(mfa_data.dom_dim);
            for (auto k = 0; k < mfa_data.dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];
            }

            // fill Rk, the matrix of input points
            MatrixX<T> Rk(N.rows(), mfa_data.max_dim - mfa_data.min_dim + 1);           // one row for each input point
            VolIterator vol_iter(ndom_pts, dom_starts, input.ndom_pts);                 // iterator over input points
            VectorXi ijk(mfa_data.dom_dim);
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
            if (mfa_data.p.size() != input.ndom_pts.size())
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

        //         DEPRECATE
//         // solves for one curve of control points
//         void CtrlCurve(
//                 const MatrixX<T>&   N,                  // basis functions for current dimension
//                 const MatrixX<T>&   NtN,                // Nt * N
//                 MatrixX<T>&         R,                  // residual matrix for current dimension and curve
//                 MatrixX<T>&         P,                  // solved points for current dimension and curve
//                 size_t              k,                  // current dimension
//                 size_t              co,                 // starting ofst for reading domain pts
//                 size_t              cs,                 // stride for reading domain points
//                 size_t              to,                 // starting ofst for writing control pts
//                 MatrixX<T>&         temp_ctrl0,         // first temporary control points buffer
//                 MatrixX<T>&         temp_ctrl1,         // second temporary control points buffer
//                 int                 curve_id,           // debugging
//                 TensorProduct<T>&   tensor,             // (output) tensor product containing result
//                 bool                weighted = true)    // solve for and use weights
//         {
//             // solve for weights
//             // TODO: avoid copying into Q by passing temp_ctrl0, temp_ctrl1, co, cs to Weights()
//             // TODO: check that this is right, using co and cs for copying control points and domain points
//             MatrixX<T> Q;
//             Q.resize(input.ndom_pts(k), tensor.ctrl_pts.cols());
//             if (k == 0)
//             {
//                 for (auto i = 0; i < input.ndom_pts(k); i++)
//                     Q.row(i) = input.domain.block(co + i * cs, mfa_data.min_dim, 1, mfa_data.max_dim - mfa_data.min_dim + 1);
//             }
//             else if (k % 2)
//             {
//                 for (auto i = 0; i < input.ndom_pts(k); i++)
//                     Q.row(i) = temp_ctrl0.row(co + i * cs);
//             }
//             else
//             {
//                 for (auto i = 0; i < input.ndom_pts(k); i++)
//                     Q.row(i) = temp_ctrl1.row(co + i * cs);
//             }
// 
//             VectorX<T> weights = VectorX<T>::Ones(N.cols());
// 
// #ifndef MFA_NO_WEIGHTS
// 
//             if (weighted)
//                 if (k == mfa_data.dom_dim - 1)                      // only during last dimension of separable iteration over dimensions
//                     Weights(k, Q, N, NtN, curve_id, weights);   // solve for weights
// 
// #endif
// 
//             // compute R
//             // first dimension reads from domain
//             // subsequent dims alternate reading temp_ctrl0 and temp_ctrl1
//             // even dim reads temp_ctrl1, odd dim reads temp_ctrl0; opposite of writing order
//             // because what was written in the previous dimension is read in the current one
//             if (k == 0)
//                 RHS(k, N, R, weights, co);                 // input points = default domain
//             else if (k % 2)
//                 RHS(k, temp_ctrl0, N, R, weights, co, cs); // input points = temp_ctrl0
//             else
//                 RHS(k, temp_ctrl1, N, R, weights, co, cs); // input points = temp_ctrl1
// 
//             // rationalize NtN, ie, weigh the basis function coefficients
//             MatrixX<T> NtN_rat = NtN;
//             mfa_data.Rationalize(k, weights, N, NtN_rat);
// 
//             // solve for P
// #ifdef WEIGH_ALL_DIMS                                   // weigh all dimensions
//             P = NtN_rat.ldlt().solve(R);
// #else                                                   // don't weigh domain coordinate (only range)
//             // TODO: avoid 2 solves?
//             MatrixX<T> P2(P.rows(), P.cols());
//             P = NtN.ldlt().solve(R);                            // nonrational domain coordinates
//             P2 = NtN_rat.ldlt().solve(R);                       // rational range coordinate
//             for (auto i = 0; i < P.rows(); i++)
//                 P(i, P.cols() - 1) = P2(i, P.cols() - 1);
// #endif
// 
//             // append points from P to control points that will become inputs for next dimension
//             // TODO: any way to avoid this?
//             CopyCtrl(P, k, co, cs, to, tensor, temp_ctrl0, temp_ctrl1);
// 
//             // copy weights of final dimension to mfa
//             if (k == mfa_data.dom_dim - 1)
//             {
//                 for (auto i = 0; i < weights.size(); i++)
//                     tensor.weights(to + i * cs) = weights(i);
//             }
//         }

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
            int ndims = mfa_data.dom_dim;

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
            int ndims = mfa_data.dom_dim;

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

            for (auto i = 0; i < input.ndom_pts[k]; i++)      // all domain points in the curve
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
                vector<TensorProduct<T>>&   new_tensors)                        // (output) new tensors scheduled to be added
        {
            // typing shortcuts
            Tmesh<T>&                   tmesh                   = mfa_data.tmesh;
            vector<vector<T>>&          all_knots               = tmesh.all_knots;
            vector<vector<int>>&        all_knot_levels         = tmesh.all_knot_levels;
            vector<vector<ParamIdx>>&   all_knot_param_idxs     = tmesh.all_knot_param_idxs;
            vector<TensorProduct<T>>&   tensor_prods            = tmesh.tensor_prods;
            int&                        dom_dim                 = mfa_data.dom_dim;
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
                {
                    fmt::print(stderr, "Error: Refine(): insertion index {} with parent tensor idx {} "
                            "at level {}, but parent_level is {}. This should not happen.\n",
                            i, parent_tensor_idxs[i], tensor_prods[parent_tensor_idxs[i]].level, parent_level);
                    abort();
                }

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

                // make a candidate tensor of some size, eg., p+1 or p+2 control points
                TensorProduct<T> c(dom_dim);
                for (auto j = 0; j < dom_dim; j++)
                {
                    // make p + 1 control points in the added tensor
                    // start with p + 1 control points, and after trimming to the parent, and making other adjustments,
                    // hopefully we'll end up with no less than p control points in any tensor
                    c.knot_mins[j] = inserted_knot_idxs[j][i] - p(j) / 2     >= 0                  ? inserted_knot_idxs[j][i] - p(j) / 2     : 0;
                    c.knot_maxs[j] = inserted_knot_idxs[j][i] + p(j) / 2 + 1 < all_knots[j].size() ? inserted_knot_idxs[j][i] + p(j) / 2 + 1 : all_knots[j].size() - 1;

                    // ---- or -----

                    // make p + 2 control points in the added tensor
                    // start with p + 2 control points, and after trimming to the parent, and making other adjustments,
                    // hopefully we'll end up with no less than p + 1 control points in any tensor
//                     c.knot_mins[j] = inserted_knot_idxs[j][i] - p(j) / 2 - 1 >= 0                  ? inserted_knot_idxs[j][i] - p(j) / 2 - 1 : 0;
//                     c.knot_maxs[j] = inserted_knot_idxs[j][i] + p(j) / 2 + 1 < all_knots[j].size() ? inserted_knot_idxs[j][i] + p(j) / 2 + 1 : all_knots[j].size() - 1;
                }
                c.level     = child_level;
                c.parent    = parent_tensor_idxs[i];
                c.parent_exists = true;

                TensorProduct<T>& pt = tensor_prods[parent_tensor_idxs[i]];             // parent tensor of the candidate tensor

                // intersection proximity (assumes same for all dims)
                int pad         = p(0) % 2 == 0 ? p(0) + 1 : p(0);                      // padding for all tensors
                int edge_pad    = (p(0) / 2) + 1;                                       // extra padding for tensor at the global edge

                // constrain candidate to be no larger than parent in any dimension
                // also don't leave parent with a small remainder anywhere
                for (auto j = 0; j < dom_dim; j++)
                {
                    int min_ofst  = (pt.knot_mins[j] == 0) ? pad + edge_pad : pad;
                    int max_ofst  = (pt.knot_maxs[j] == all_knots[j].size() - 1) ? pad + edge_pad : pad;

                    // adjust min edge of candidate
                    while (c.knot_mins[j] > pt.knot_mins[j]  &&
                            tmesh.knot_idx_dist(pt, c.knot_mins[j], pt.knot_maxs[j], j, false) < max_ofst)
                        c.knot_mins[j]--;
                    if (c.knot_mins[j] < pt.knot_mins[j] ||
                            tmesh.knot_idx_dist(pt, pt.knot_mins[j], c.knot_mins[j], j, false) < min_ofst)
                        c.knot_mins[j] = pt.knot_mins[j];

                    // adjust max edge of candidate
                    while (c.knot_maxs[j] < pt.knot_maxs[j]  &&
                            tmesh.knot_idx_dist(pt, pt.knot_mins[j], c.knot_maxs[j], j, false) < min_ofst)
                        c.knot_maxs[j]++;
                    if (c.knot_maxs[j] > pt.knot_maxs[j] ||
                            tmesh.knot_idx_dist(pt, c.knot_maxs[j], pt.knot_maxs[j], j, false) < max_ofst)
                        c.knot_maxs[j] = pt.knot_maxs[j];
                }

                // force knots at candidate tensor bounds to be at level no deeper than previous level
                // however do not allow candidate tensor bounds to be beyond those of parent tensor
                int level = (parent_level ? parent_level - 1 : 0);
                for (auto j = 0; j < dom_dim; j++)
                {
                    while (all_knot_levels[j][c.knot_mins[j]] > level && c.knot_mins[j] > pt.knot_mins[j])
                        c.knot_mins[j]--;
                    while (all_knot_levels[j][c.knot_maxs[j]] > level && c.knot_maxs[j] < pt.knot_maxs[j])
                        c.knot_maxs[j]++;
                }

                // recheck after adjustments that candidate is no larger than parent in any dimension
                // and doesn't leave parent with a small remainder anywhere
                tmesh.constrain_to_parent(c, pad);

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

                // check/adjust candidate knot mins and maxs subset and intersection against tensors to be added so far
                bool add    = true;
                for (auto tidx = 0; tidx < new_tensors.size(); tidx++)          // for all tensors scheduled to be added so far
                {
                    auto& t = new_tensors[tidx];

                    // candidate is a subset of an already scheduled tensor
                    if (tmesh.subset(c.knot_mins, c.knot_maxs, t.knot_mins, t.knot_maxs))
                        add = false;

                    // an already scheduled tensor is a subset of the candidate
                    else if (tmesh.subset(t.knot_mins, t.knot_maxs, c.knot_mins, c.knot_maxs))
                    {
                        t.knot_mins = c.knot_mins;
                        t.knot_maxs = c.knot_maxs;
                        if (t.parent != c.parent)
                        {
                            // TODO: not sure if this should be an error, or if the parent should be reset
                            fmt::print(stderr, "Error: already scheduled tensor is a subset of the candidate tensor, but they have different parents. This should not happen\n");
                            abort();
//                         t.parent    = c.parent;
                        }
                        add         = false;
                    }

#ifndef MFA_TMESH_MERGE_NONE

                        // candidate intersects an already scheduled tensor, to within some proximity
                        // the #defines adjust whether we merge tensors that intersect or only those that are close
#ifdef MFA_TMESH_MERGE_SOME
                        else if (tmesh.intersect(c, t, pad) && !tmesh.intersect(c, t, 0))
#endif
#ifdef MFA_TMESH_MERGE_MAX
                        else if (tmesh.intersect(c, t, pad))
#endif
                    {
                        if (c.parent == t.parent)       // only merge tensors refined from the same parent
                        {
                            tmesh.merge_tensors(t, c, pad);
                            add         = false;
                        }
                    }

#endif

                }       // for all tensors scheduled to be added so far

                // schedule the tensor to be added
                if (add)
                    new_tensors.push_back(c);

            }   // for all knots to be inserted

            // DEPRECATE
            // This code should not be needed
            // Only turn it back on if other consistency checks fail
            // Remove it once the code is stable

//             // check/adjust tensors scheduled to be added against each other
//             // use level = -1 to indicate removing the tensor from the schedule
//             int pad         = p(0) % 2 == 0 ? p(0) + 1 : p(0);                      // padding for all tensors
//             for (auto tidx = 0; tidx < new_tensors.size(); tidx++)          // for all tensors scheduled to be added so far
//             {
//                 auto& t = new_tensors[tidx];
// 
//                 if (t.level < 0)
//                     continue;
// 
//                 for (auto tidx1 = 0; tidx1 < new_tensors.size(); tidx1++)   // for all tensors scheduled to be added so far
//                 {
//                     auto& t1 = new_tensors[tidx1];
// 
//                     if (tidx == tidx1 || t1.level < 0)
//                         continue;
// 
//                     // t is a subset of t1
//                     if (tmesh.subset(t.knot_mins, t.knot_maxs, t1.knot_mins, t1.knot_maxs))
//                     {
//                         fmt::print(stderr, "Warning: tensor to be added is a subset of another tensor to be added. This should not happen\n");
//                         t.level = -1;                   // remove t from the schedule
//                     }
// 
//                     // t is a superset of t1
//                     else if (tmesh.subset(t1.knot_mins, t1.knot_maxs, t.knot_mins, t.knot_maxs))
//                     {
//                         fmt::print(stderr, "Warning: tensor to be added is a superset of another tensor to be added. This should not happen\n");
//                         t1.level = -1;                  // remove t1 from the schedule
//                     }
// 
// #ifndef MFA_TMESH_MERGE_NONE
// 
//                     // candidate intersects an already scheduled tensor, to within some proximity
//                     // the #defines adjust whether we merge tensors that intersect or only those that are close
// #ifdef MFA_TMESH_MERGE_SOME
//                     else if (tmesh.intersect(t1, t, pad) && !tmesh.intersect(t1, t, 0))
// #endif
// #ifdef MFA_TMESH_MERGE_MAX
//                     else if (tmesh.intersect(t1, t, pad))
// #endif
//                     {
//                         if (t.parent == t1.parent)       // only merge tensors refined from the same parent
//                         {
//                             fmt::print(stderr, "Warning: tensor to be added is being merged with another tensor to be added. This should not happen\n");
//                             tmesh.merge_tensors(t, t1, pad);
//                             t1.level = -1;              // remove t1 from the schedule
//                         }
//                     }
// 
// #endif
// 
//                 }   // tidx1
// 
//                 // debug: confirm that tensor to be added will have at least p + 1 control points
//                 // TODO: remove this check once the code is stable
//                 if (t.level >= 0 && !tmesh.check_num_knots_degree(t, 1))     // level >= 0: tensor wasn't marked for removal from the schedule
//                 {
//                     fmt::print(stderr, "Error: Tensor being added is too small. This should not happen\n");
//                     fmt::print(stderr, "Tensor tidx {} knot_mins [{}] knot_maxs[{}] level {} parent {}\n",
//                             tidx, fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level, t.parent);
//                     abort();
//                 }
// 
//             }   // tidx


// DEPRECATE
            // append the tensors
//             double add_tensors_time = MPI_Wtime();
//             AddNewTensors(new_tensors);
//             add_tensors_time = MPI_Wtime() - add_tensors_time;

            // timing
            fmt::print(stderr, "error spans time:       {} s.\n", error_spans_time);

            return false;
        }

        // appends and encodes a vector of new tensor products
        void AddNewTensors(vector<TensorProduct<T>>& new_tensors)
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

                // debug
                fmt::print(stderr, "AddNewTensors(): appending tensor knot_mins [{}] knot_maxs [{}] level {}\n",
                        fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level);

                int tensor_idx = tmesh.append_tensor(t.knot_mins, t.knot_maxs, t.level);

                // debug: check all spans before solving
                // TODO: comment out once the code is debugged
                if (!nk.CheckAllSpans())
                    throw MFAError(fmt::format("AddNewTensors(): Error: failed checking all spans for input points\n"));

                // debug: check all knot vs control point quantities
                // TODO: comment out once the code is debugged
                for (auto j = 0; j < tensor_prods.size(); j++)
                    tmesh.check_num_knots_ctrl_pts(j);

                // debug: confirm that all tensors will have at least p control points
                // TODO: comment out once the code is debugged
                for (auto i = 0; i < tensor_prods.size(); i++)
                {
                    if (!tmesh.check_num_ctrl_degree(i, 0))
                    {
                        auto& t = tensor_prods[k];
                        fmt::print(stderr, "Error: AddNewTensors(): After appending tensor k {} one of the tensors has fewer than p control points. This should not happen\n", k);
                        fmt::print(stderr, "New tensor knot_mins [{}] knot_maxs[{}] level {} parent tensor {}\n",
                                fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","), t.level, t.parent);
                        fmt::print(stderr, "\nAddNewTensors(): T-mesh after appending tensor k {}\n", k);
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
            }
        }

        // constraint control points and corresponding anchors for local solve
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
            vector<KnotIdx> tc_pad_mins(mfa_data.dom_dim);
            vector<KnotIdx> tc_pad_maxs(mfa_data.dom_dim);

            // intersection of tc padded by degree p with tensor being visited
            vector<KnotIdx> intersect_mins(mfa_data.dom_dim);
            vector<KnotIdx> intersect_maxs(mfa_data.dom_dim);

            // get required sizes

            int rows = 0;                                       // number of rows required in ctrl_pts
            VectorXi npts(mfa_data.dom_dim);

            for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[k];
                if (&t == &tc)
                    continue;
                if (t.level > tc.level)
                    continue;

                // pad mins and maxs of tc
                for (auto i = 0; i < mfa_data.dom_dim; i++)
                {
                    int p = mfa_data.p(i);
                    tmesh.knot_idx_ofst(t, tc.knot_mins[i], -p, i, true, tc_pad_mins[i]);
                    tmesh.knot_idx_ofst(t, tc.knot_maxs[i], p, i, true, tc_pad_maxs[i]);
                }

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
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
            VectorXi sub_starts(mfa_data.dom_dim);
            VectorXi sub_npts(mfa_data.dom_dim);
            VectorXi all_npts(mfa_data.dom_dim);
            vector<KnotIdx> anchor(mfa_data.dom_dim);           // one anchor
            for (auto k = 0; k < tmesh.tensor_prods.size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[k];
                if (&t == &tc)
                    continue;
                if (t.level > tc.level)
                    continue;

                // pad mins and maxs of tc
                for (auto i = 0; i < mfa_data.dom_dim; i++)
                {
                    int p = mfa_data.p(i);
                    tmesh.knot_idx_ofst(t, tc.knot_mins[i], -p, i, true, tc_pad_mins[i]);
                    tmesh.knot_idx_ofst(t, tc.knot_maxs[i], p, i, true, tc_pad_maxs[i]);
                }

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
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
                    VectorXi ijk(mfa_data.dom_dim);
                    while (!voliter.done())
                    {
                        // skip MFA_NAW control points (used in odd degree cases)
                        if (t.weights(voliter.sub_full_idx(voliter.cur_iter())) != MFA_NAW)
                        {
                            // control point
                            ctrl_pts.row(cur_row) = t.ctrl_pts.row(voliter.sub_full_idx(voliter.cur_iter()));

                            // anchor
                            anchors[cur_row].resize(mfa_data.dom_dim);
                            voliter.idx_ijk(voliter.cur_iter(), ijk);
                            mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);
                            for (auto i = 0; i < mfa_data.dom_dim; i++)
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

        // constraint control points and corresponding anchors for local solve
        // this version checks previous tensors of current tensor in one dimension
        void LocalSolvePrevConstraintsDim(
                int                         dim,                // current dimension
                const TensorProduct<T>&     tc,                 // current tensor product being solved
                MatrixX<T>&                 ctrl_pts,           // (output) constraining control points
                vector<vector<KnotIdx>>&    anchors,            // (output) corresponding anchors
                vector<TensorIdx>&          t_idx_anchors)      // (output) tensors containing corresponding anchors
        {
            const Tmesh<T>&         tmesh   = mfa_data.tmesh;
            int                     cols    = tc.ctrl_pts.cols();

            // mins, maxs of tc padded by degree p
            vector<KnotIdx> tc_pad_mins(mfa_data.dom_dim);
            vector<KnotIdx> tc_pad_maxs(mfa_data.dom_dim);

            // intersection of tc padded by degree p with tensor being visited
            vector<KnotIdx> intersect_mins(mfa_data.dom_dim);
            vector<KnotIdx> intersect_maxs(mfa_data.dom_dim);

            // get required sizes

            int rows = 0;                                       // number of rows required in ctrl_pts
            VectorXi npts(mfa_data.dom_dim);

            for (auto k = 0; k < tc.prev[dim].size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[tc.prev[dim][k]];

                if (t.level > tc.level)
                    continue;

                // mins, maxs of tc padded by degree p on min side of cur. dim
                tc_pad_maxs = tc.knot_maxs;
                tmesh.knot_idx_ofst(t, tc.knot_mins[dim], -mfa_data.p(dim), dim, true, tc_pad_mins[dim]);

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
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
            VectorXi sub_starts(mfa_data.dom_dim);
            VectorXi sub_npts(mfa_data.dom_dim);
            VectorXi all_npts(mfa_data.dom_dim);
            vector<KnotIdx> anchor(mfa_data.dom_dim);           // one anchor
            for (auto k = 0; k < tc.prev[dim].size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[tc.prev[dim][k]];

                if (t.level > tc.level)
                    continue;

                // mins, maxs of tc padded by degree p on min side of cur. dim
                tc_pad_maxs = tc.knot_maxs;
                tmesh.knot_idx_ofst(t, tc.knot_mins[dim], -mfa_data.p(dim), dim, true, tc_pad_mins[dim]);

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
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
                    VectorXi ijk(mfa_data.dom_dim);
                    while (!voliter.done())
                    {
                        // skip MFA_NAW control points (used in odd degree cases)
                        if (t.weights(voliter.sub_full_idx(voliter.cur_iter())) != MFA_NAW)
                        {
                            // control point
                            ctrl_pts.row(cur_row) = t.ctrl_pts.row(voliter.sub_full_idx(voliter.cur_iter()));

                            // anchor
                            anchors[cur_row].resize(mfa_data.dom_dim);
                            voliter.idx_ijk(voliter.cur_iter(), ijk);
                            mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);
                            for (auto i = 0; i < mfa_data.dom_dim; i++)
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

        // constraint control points and corresponding anchors for local solve
        // this version checks next tensors of current tensor in one dimension
        void LocalSolveNextConstraintsDim(
                int                         dim,                // current dimension
                const TensorProduct<T>&     tc,                 // current tensor product being solved
                MatrixX<T>&                 ctrl_pts,           // (output) constraining control points
                vector<vector<KnotIdx>>&    anchors,            // (output) corresponding anchors
                vector<TensorIdx>&          t_idx_anchors)      // (output) tensors containing corresponding anchors
        {
            const Tmesh<T>&         tmesh   = mfa_data.tmesh;
            int                     cols    = tc.ctrl_pts.cols();

            // mins, maxs of tc padded by degree p
            vector<KnotIdx> tc_pad_mins(mfa_data.dom_dim);
            vector<KnotIdx> tc_pad_maxs(mfa_data.dom_dim);

            // intersection of tc padded by degree p with tensor being visited
            vector<KnotIdx> intersect_mins(mfa_data.dom_dim);
            vector<KnotIdx> intersect_maxs(mfa_data.dom_dim);

            // get required sizes

            int rows = 0;                                       // number of rows required in ctrl_pts
            VectorXi npts(mfa_data.dom_dim);

            for (auto k = 0; k < tc.next[dim].size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[tc.next[dim][k]];

                if (t.level > tc.level)
                    continue;

                // mins, maxs of tc padded by degree p on max side of cur. dim
                tc_pad_mins = tc.knot_mins;
                tmesh.knot_idx_ofst(t, tc.knot_maxs[dim], mfa_data.p(dim), dim, true, tc_pad_maxs[dim]);

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
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
            VectorXi sub_starts(mfa_data.dom_dim);
            VectorXi sub_npts(mfa_data.dom_dim);
            VectorXi all_npts(mfa_data.dom_dim);
            vector<KnotIdx> anchor(mfa_data.dom_dim);           // one anchor
            for (auto k = 0; k < tc.next[dim].size(); k++)
            {
                const TensorProduct<T>& t = tmesh.tensor_prods[tc.next[dim][k]];

                if (t.level > tc.level)
                    continue;

                // mins, maxs of tc padded by degree p on max side of cur. dim
                tc_pad_mins = tc.knot_mins;
                tmesh.knot_idx_ofst(t, tc.knot_maxs[dim], mfa_data.p(dim), dim, true, tc_pad_maxs[dim]);

                // intersect padded bounds with tensor t
                if (tmesh.intersects(tc_pad_mins, tc_pad_maxs, t.knot_mins, t.knot_maxs, intersect_mins, intersect_maxs))
                {
                    for (auto i = 0; i < mfa_data.dom_dim; i++)
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
                    VectorXi ijk(mfa_data.dom_dim);
                    while (!voliter.done())
                    {
                        // skip MFA_NAW control points (used in odd degree cases)
                        if (t.weights(voliter.sub_full_idx(voliter.cur_iter())) != MFA_NAW)
                        {
                            // control point
                            ctrl_pts.row(cur_row) = t.ctrl_pts.row(voliter.sub_full_idx(voliter.cur_iter()));

                            // anchor
                            anchors[cur_row].resize(mfa_data.dom_dim);
                            voliter.idx_ijk(voliter.cur_iter(), ijk);
                            mfa_data.tmesh.ctrl_pt_anchor(t, ijk, anchor);
                            for (auto i = 0; i < mfa_data.dom_dim; i++)
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
            new_knots.resize(mfa_data.dom_dim);
            vector<vector<int>> new_levels(mfa_data.dom_dim);

            for (auto& t : mfa_data.tmesh.tensor_prods)                             // for all tensor products in the tmesh
            {
                // check and assign main quantities
                VectorXi n = t.nctrl_pts - VectorXi::Ones(mfa_data.dom_dim);        // number of control point spans in each domain dim
                VectorXi m = input.ndom_pts  - VectorXi::Ones(mfa_data.dom_dim);    // number of input data point spans in each domain dim

                // resize control points and weights
                t.ctrl_pts.resize(t.nctrl_pts.prod(), mfa_data.max_dim - mfa_data.min_dim + 1);
                t.weights = VectorX<T>::Ones(t.ctrl_pts.rows());

                for (size_t k = 0; k < mfa_data.dom_dim; k++)                       // for all domain dimensions
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
                                // debug
//                                 fmt::print(stderr, "OrigNewKnots_curve(): dim {} checking curve {} out of {} curves\n", k, j, ncurves);

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

                                // debug
//                                 fmt::print(stderr, "OrigNewKnots_curve(): nerr {}\n", nerr);

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
//                     fprintf(stderr, "\rdimension %ld of %d encoded\n", k + 1, mfa_data.dom_dim);
                }                                                           // domain dimensions

                // debug
                for (auto i = 0; i < mfa_data.dom_dim; i++)
                    fmt::print(stderr, "new_knots in dim {}: [{}]\n", i, fmt::join(new_knots[i], ","));

                // insert the new knots
                mfa::NewKnots<T> nk(mfa_data, input);
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
                mfa_data.tmesh.tensor_knot_idxs(t);
            }                                                               // tensor products

            return(tot_nnew_knots ? 0 : 1);
        }
    };          // Encoder class
}           // mfa namespace

#endif
