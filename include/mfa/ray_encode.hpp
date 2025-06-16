//--------------------------------------------------------------
// RayEncoder object. Used to create ray models for line integration
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------

#ifndef _MFA_RAY_ENCODE_HPP
#define _MFA_RAY_ENCODE_HPP

#include    <mfa/mfa_data.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/decode.hpp>
#include    <mfa/new_knots.hpp>

#include    <vector>
#include    <set>
#include    <iostream>
#include    <algorithm>

// temporary utilities for testing
#include    <ctime>
#include    <chrono>
#include    <iomanip>
#include    <fstream>
#include    <sstream>
#include    <mpi.h>     // for MPI_Wtime() only

namespace mfa
{
    template <typename T>                                   // float or double
    class RayEncoder
    {
    private:
        MFA_Data<T>&        mfa_data;                       // the mfa data model
        int                 dom_dim;                        // domain dimension of mfa_data
        int                 verbose;                        // output level
        const PointSet<T>&  input;                          // input points
        SparseMatrixX<T>          coll0_t;       // temporary collocation matrix
        SparseMatrixX<T>          coll1_t;     // temporary differentiated collocation matrix
        Eigen::SimplicialLDLT<SparseMatrixX<T>> all_out_llt;
        Eigen::SimplicialLDLT<SparseMatrixX<T>> all_in_llt;
        vector<int>         t_spans;      // temporary vector to hold spans of each input point
        bool reverse_encode{false};

    public:

        RayEncoder(MFA_Data<T>&        mfa_data_,              // MFA data model
                const PointSet<T>&  input_,                 // input points
                int                 verbose_) :             // debug level
            mfa_data(mfa_data_),
            dom_dim(mfa_data.dom_dim),
            verbose(verbose_),
            input(input_)
        { }

        ~RayEncoder() {}

        void encode(TensorIdx t_idx = 0, bool weighted = false)                  // index of tensor product being encoded
        {
            double t0 = MPI_Wtime();
            reverse_encode = false;

            if (mfa_data.tmesh.tensor_prods.size() != 1)
            {
                cerr << "ERROR: RayEncoder::encode() only implemented for single tensor tmesh" << endl;
                exit(0);
            }

            // Basic definitions
            auto& tmesh         = mfa_data.tmesh;
            auto& t             = tmesh.tensor_prods[t_idx];    // current tensor product
            const int pt_dim    = mfa_data.dim();         // control point dimensionality
            t.weights = VectorX<T>::Ones(t.ctrl_pts.rows());           // linear solve does not solve for weights; set to 1

            // get input domain points covered by the tensor
            vector<size_t> start_idxs(dom_dim);
            vector<size_t> end_idxs(dom_dim);
            tmesh.domain_pts(t_idx, input.params->param_grid, true, 0, start_idxs, end_idxs);

            // Number and offset for points in tensor
            VectorXi ndom_pts(dom_dim);
            VectorXi dom_starts(dom_dim);
            for (int k = 0; k < dom_dim; k++)
            {
                ndom_pts(k)     = end_idxs[k] - start_idxs[k] + 1;
                dom_starts(k)   = start_idxs[k];
            }

            // two matrices of input points for subsequent dimensions after dimension 0, which draws directly from input domain
            // input points cover constraints as well as free control point basis functions
            // (double buffering output control points to input points)
            MatrixX<T> Q0(ndom_pts.prod(), pt_dim);
            MatrixX<T> Q1(ndom_pts.prod(), pt_dim);
            VectorXi nin_pts    = ndom_pts;
            VectorXi nout_pts   = ndom_pts;

            // Fill Q0 buffer from input PointSet.
            // Doing this ahead of time allows us to stop thinking in terms of 
            // "subvolumes" and focus on the tensor points only.
            VolIterator setup_iter(ndom_pts, dom_starts, input.ndom_pts());
            while (!setup_iter.done())
            {
                Q0.row(setup_iter.cur_iter()) = input.domain.block(setup_iter.cur_iter_full(), mfa_data.min_dim, 1, pt_dim);
                setup_iter.incr_iter();
            }

            // Main loop over each dimension
            for (auto dimcount = 0; dimcount < dom_dim; dimcount++)
            {
                int dim = reverse_encode ? dom_dim - dimcount - 1 : dimcount;
                int np = ndom_pts(dim);
                int nc = t.nctrl_pts(dim);
                nout_pts(dim) = nc;

                if (verbose)
                {
                    cerr << "  encoding dimension " << dim << endl;
                }

                // Initialize matrix of basis functions and differentiated basis functions
                // TODO: this does not support tensor subvolumes
                SparseMatrixX<T> Nt(nc, np);
                Eigen::SimplicialLDLT<SparseMatrixX<T>> NtN_llt;
                init_matrix_factors(dim, np, nc, Nt, NtN_llt);
                
                // Declarations
                MatrixX<T>      R(nin_pts(dim), pt_dim);                    // RHS for solving N * P = R
                MatrixX<T>      P(t.nctrl_pts(dim), pt_dim);                // Control points
                VolIterator     in_iter(nin_pts);                           // volume of current input points
                VolIterator     out_iter(nout_pts);                         // volume of current output points
                SliceIterator   in_slice_iter(in_iter, dim);                // slice of input points volume missing current dim
                SliceIterator   out_slice_iter(out_iter, dim);              // slice of output points volume missing current dim
                vector<bool>    prev_curve_in_domain(nin_pts(dim), 0);

                // for all curves in the current dimension
                while (!in_slice_iter.done())
                {
                    CurveIterator   in_curve_iter(in_slice_iter);       // one curve of the input points in the current dim
                    CurveIterator   out_curve_iter(out_slice_iter);     // one curve of the output points in the current dim
                    solve_curve(in_curve_iter, prev_curve_in_domain, R, Q0, Q1, Nt, NtN_llt, P);

                    // copy solution to one curve of output points
                    while (!out_curve_iter.done())
                    {
                        if (dimcount % 2 == 0)
                            Q1.row(out_curve_iter.cur_iter_full()) = P.row(out_curve_iter.cur_iter());
                        else
                            Q0.row(out_curve_iter.cur_iter_full()) = P.row(out_curve_iter.cur_iter());
                        out_curve_iter.incr_iter();
                    }

                    out_slice_iter.incr_iter();
                    in_slice_iter.incr_iter();
                }       // for all curves

                // adjust number of input points for next iteration
                nin_pts(dim) = t.nctrl_pts(dim);
            }       // for all domain dimensions

            // copy final result back to tensor product
            if (dom_dim % 2 == 0)
                t.ctrl_pts = Q0.block(0, 0, t.nctrl_pts.prod(), pt_dim);    // TODO: change to Q0.topRows(t.nctrl_pts.prod());
            else
                t.ctrl_pts = Q1.block(0, 0, t.nctrl_pts.prod(), pt_dim);

            // timing
            fmt::print(stderr, "EncodeSeparableConstrained() time {} s.\n", MPI_Wtime() - t0);
        }

        // Create, fill, and factor matrices that contain evaluated basis functions at each point
        // These matrices are recomputed for each dimension during encoding
        void init_matrix_factors(   int dim, 
                                    int np, 
                                    int nc, 
                                    SparseMatrixX<T>& Nt, 
                                    Eigen::SimplicialLDLT<SparseMatrixX<T>>& NtN_llt)
        {
            const int der = 2;
            VectorXi            q = mfa_data.p + VectorXi::Ones(dom_dim);  // order of basis funs
            BasisFunInfo<T>     bfi(q);                                    // buffers for basis fun evaluation

            coll0_t = SparseMatrixX<T>(nc, np);
            coll1_t = SparseMatrixX<T>(nc, np);
            coll0_t.reserve(VectorXi::Constant(coll0_t.cols(), mfa_data.p(dim)+1));
            coll1_t.reserve(VectorXi::Constant(coll1_t.cols(), mfa_data.p(dim)+1));

            t_spans = vector<int>(np, 0);
            vector<vector<T>> funs(der + 1, vector<T>(mfa_data.p(dim)+1, 0));
            for (int i = 0; i < coll0_t.cols(); i++)
            {
                int span = mfa_data.tmesh.FindSpan(dim, input.params->param_grid[dim][i], nc);
                t_spans[i] = span;
                mfa_data.FastBasisFunsDers(dim, input.params->param_grid[dim][i], span, der, funs, bfi);

                for (int j = 0; j < mfa_data.p(dim)+1; j++)
                {
                    coll0_t.coeffRef(span - mfa_data.p(dim) + j, i) = funs[0][j];
                    coll1_t.coeffRef(span - mfa_data.p(dim) + j, i) = funs[1][j];
                }
            }

            // Compute matrix factorizations
            all_in_llt.compute(coll0_t * coll0_t.transpose());
            all_out_llt.compute(coll1_t * coll1_t.transpose());

            // Fill Nt with dummy values in the correct sparsity pattern,
            // so that analyzePattern() can be called
            Nt.reserve(VectorXi::Constant(Nt.cols(), mfa_data.p(dim)+1));
            compute_curve_mat(dim, vector<bool>(np, true), np, nc, Nt);
            NtN_llt.analyzePattern(Nt*Nt.transpose());

            return;
        }

        bool in_domain(const VectorXi& ijk)
        {
            if (dom_dim == 3) return in_domain2d(ijk);
            else if (dom_dim == 5) return in_domain3d(ijk);
            else throw MFAError("Incorrect dom_dim passed to RayEncoder::in_domain");

            return false;
        }

        // Test if input point with index ijk is in domain
        // TODO does not support tensor subvolumes
        bool in_domain2d(const VectorXi& ijk)
        {
            T u_t     = input.params->param_grid[0][ijk(0)];
            T u_rho   = input.params->param_grid[1][ijk(1)];
            T u_alpha = input.params->param_grid[2][ijk(2)];

            T x = (2*u_rho-1)*cos(u_alpha*M_PI) - (2*u_t-1)*sin(u_alpha*M_PI);
            T y = (2*u_rho-1)*sin(u_alpha*M_PI) + (2*u_t-1)*cos(u_alpha*M_PI);
            
            // limits of the (square) domain in parameter space
            // This comes from the fact that we defined r_lim = 1.5 * max_bound,
            // where max_bound the maximum extent of the physical domain.
            // See Block::create_ray_model()
            // 
            // n.b. For rectangular (non-square) domains, we can define a
            // different bound in each +/- direction
            T bb = 1/1.5;

            if (x > bb || x < -1*bb || y > bb || y < -1*bb)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        bool in_domain3d(const VectorXi& ijk)
        {
            T u_t = input.params->param_grid[0][ijk(0)];
            T u_rho = input.params->param_grid[1][ijk(1)];
            T u_nu = input.params->param_grid[2][ijk(2)];
            T u_theta = input.params->param_grid[3][ijk(3)];
            T u_phi = input.params->param_grid[4][ijk(4)];

            T theta = u_theta * M_PI;
            T phi = u_phi * M_PI;
            T ST = sin(theta);
            T CT = cos(theta);
            T SP = sin(phi);
            T CP = cos(phi);

            T x = (2*u_rho-1)*CT*SP - (2*u_nu-1)*ST + (2*u_t-1)*CT*CP;
            T y = (2*u_rho-1)*ST*SP + (2*u_nu-1)*CT + (2*u_t-1)*ST*CP;
            T z = (2*u_rho-1)*CP - (2*u_t-1)*SP;

            T bb = 1/1.5;

            if (x > bb || x < -1*bb || y > bb || y < -1*bb)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        // computes curve of free control points in one dimension
        void solve_curve(
                CurveIterator&          in_curve_iter,                  // current curve
                vector<bool>&           curve_in_domain,                // previous curve
                MatrixX<T>&             R,                              // right hand side, allocated by caller
                MatrixX<T>&             Q0,                             // first matrix of input points, allocated by caller
                MatrixX<T>&             Q1,                             // second matrix of input points, allocated by caller
                SparseMatrixX<T>&       Nt,
                Eigen::SimplicialLDLT<SparseMatrixX<T>>& NtN_llt,
                MatrixX<T>&             P)                              // (output) control points
        {
            VectorXi cur_ijk;
            bool same_pattern = true;
            bool all_out = true;
            bool all_in = true;
            int dim = in_curve_iter.curve_dim();
            int npts = in_curve_iter.tot_iters();
            int nctrl = coll0_t.rows();

            int dimcount = 0;
            if (!reverse_encode)
                dimcount = dim;
            else
                dimcount = dom_dim - dim - 1;

            // copy one curve of input points to right hand side
            // add zero entries for rows that will correspond to deriv constraints
            while (!in_curve_iter.done())
            {
                int curve_idx = in_curve_iter.cur_iter();
                int vol_idx = in_curve_iter.cur_iter_full();  // point index. One for each grid point on this curve
                cur_ijk = in_curve_iter.cur_ijk();      // ijk coordinates in full dimensional volume
                
                if (dimcount == 0)
                {
                    // If the in_domain pattern of the curve matches the previous curve
                    // we can avoid the expensive computation and factorization of N
                    bool inside = in_domain(cur_ijk);
                    if (inside != curve_in_domain[curve_idx])
                    {
                        same_pattern = false;
                    }
                    if (inside)
                    {
                        all_out = false;
                    }
                    else
                    {
                        all_in = false;
                    }

                    curve_in_domain[curve_idx] = inside; // save for next curve

                    if (inside)
                    {
                        if (dimcount % 2 == 0)
                            R.row(curve_idx) = Q0.row(vol_idx);
                        else
                            R.row(curve_idx) = Q1.row(vol_idx);
                    }
                    else
                    {
                        R.row(curve_idx).setZero();
                    }
                }
                else
                {
                    if (dimcount % 2 == 0)
                        R.row(curve_idx) = Q0.row(vol_idx);
                    else
                        R.row(curve_idx) = Q1.row(vol_idx);
                }
                
                in_curve_iter.incr_iter();
            }
            in_curve_iter.reset();

            // Compute constraint matrix (if necessary) and solve
            double t0 = MPI_Wtime();
            if (dimcount == 0)
            {
                if (all_out)
                {
                    P = all_out_llt.solve(coll1_t * R);
                }
                else if (all_in)
                {
                    P = all_in_llt.solve(coll0_t * R);
                }
                else
                {
                    if (same_pattern == false || in_curve_iter.slice_iter_->cur_iter() == 0)
                    {
                        compute_curve_mat(dim, curve_in_domain, npts, nctrl, Nt);
                        NtN_llt.factorize(Nt * Nt.transpose());
                    }
                    P = NtN_llt.solve(Nt*R);
                } 
            }
            else
            {
                P = all_in_llt.solve(coll0_t * R);
            }
        }

        void compute_curve_mat(
                int                 dim,                // current dimension
                const vector<bool>& curve_in_domain,    // flag if in domain for each point on curve
                int                 npts,               // number of input points in current dim
                int                 nctrl,              // number of control points in current dim
                SparseMatrixX<T>&   Nt)                 // (output) matrix of free control points basis functions
        {
            for (int j = 0; j < npts; j++)
            {
                bool inside = curve_in_domain[j];
                int ctrl_idx = t_spans[j] - mfa_data.p(dim);
                for (int i = 0; i < mfa_data.p(dim) + 1; i++)
                {
                    if (inside)
                        Nt.coeffRef(ctrl_idx + i, j) = coll0_t.coeffRef(ctrl_idx + i, j);
                    else
                        Nt.coeffRef(ctrl_idx + i, j) = coll1_t.coeffRef(ctrl_idx + i, j);
                }
            }

            return;
        }


    }; // RayEncoder
} // namespace mfa

#endif // _MFA_RAY_ENCODE_HPP