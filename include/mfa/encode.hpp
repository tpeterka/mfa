//--------------------------------------------------------------
// encoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _ENCODE_HPP
#define _ENCODE_HPP

#include <mfa/mfa.hpp>

#include <Eigen/Dense>
#include <vector>
#include <set>

#ifndef      MFA_NO_WEIGHTS

#include    "coin/ClpSimplex.hpp"
#include    "coin/ClpInterior.hpp"

#endif

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

namespace mfa
{
    template <typename T>                   // float or double
    class Encoder
    {
    public:

        Encoder(MFA<T>& mfa_);
        ~Encoder() {}
        void Encode(
                bool weighted = true);      // solve for and use weights
        void AdaptiveEncode(
                T    err_limit,             // maximum allowable normalized error
                bool weighted,              // solve for and use weights
                int  max_rounds = 0);       // optional maximum number of rounds

   private:

#ifndef      MFA_NO_WEIGHTS

        bool Weights(
                int         k,              // current dimension
                MatrixX<T>& Q,              // input points
                MatrixX<T>& N,              // basis functions
                MatrixX<T>& NtN,            // N^T * N
                int         curve_id,       // debugging
                VectorX<T>& weights)        // output weights
        {
            bool success;

            // Nt, NtNi
            // TODO: offer option of saving time or space by comuting Nt and NtN each time it is needed?
            MatrixX<T> Nt   = N.transpose();
            MatrixX<T> NtNi = NtN.partialPivLu().inverse();

            int pt_dim = mfa.domain.cols();             // dimensionality of input and control points (domain and range)

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
                            fprintf(stderr, "curve %d: successful linear solve from %d eigenvectors\n", curve_id, i);
                        }
                    }

                    if (success)
                        break;
                }                                               // increasing number of eigenvectors

                if (!success)
                {
                    weights = VectorX<T>::Ones(nweights);
                    fprintf(stderr, "curve %d: linear solver could not find positive weights; setting to 1\n", curve_id);
                }
            }                                                   // else need to expand eigenspace

            return success;
        }

#endif

        // default domain
        void RHS(
                int         cur_dim,  // current dimension
                MatrixX<T>& N,        // matrix of basis function coefficients
                MatrixX<T>& R,        // (output) residual matrix allocated by caller
                VectorX<T>& weights,  // precomputed weights for n + 1 control points on this curve
                int         ko,       // index of starting knot
                int         po,       // index of starting parameter
                int         co);      // index of starting domain pt in current curve

        // new input points
        void RHS(
                int         cur_dim,  // current dimension
                MatrixX<T>& in_pts,   // input points (not the default domain stored in the mfa)
                MatrixX<T>& N,        // matrix of basis function coefficients
                MatrixX<T>& R,        // (output) residual matrix allocated by caller
                VectorX<T>& weights,  // precomputed weights for n + 1 control points on this curve
                int         ko,       // index of starting knot
                int         po,       // index of starting parameter
                int         co,       // index of starting input pt in current curve
                int         cs);      // stride of input pts in current curve

        void Quants(
                VectorXi& n,          // (output) number of control point spans in each dim
                VectorXi& m);         // (output) number of input data point spans in each dim

        void CtrlCurve(
                MatrixX<T>& N,           // basis functions for current dimension
                MatrixX<T>& NtN,         // Nt * N
                MatrixX<T>& R,           // residual matrix for current dimension and curve
                MatrixX<T>& P,           // solved points for current dimension and curve
                size_t      k,           // current dimension
                size_t      co,          // starting ofst for reading domain pts
                size_t      cs,          // stride for reading domain points
                size_t      to,          // starting ofst for writing control pts
                MatrixX<T>& temp_ctrl0,  // first temporary control points buffer
                MatrixX<T>& temp_ctrl1,  // second temporary control points buffer
                int         curve_id,    // debugging
                bool        weighted = true);   // solve for and use weights

        void CopyCtrl(
                MatrixX<T>& P,          // solved points for current dimension and curve
                int         k,          // current dimension
                size_t      co,         // starting offset for reading domain points
                size_t      cs,         // stride for reading domain points
                size_t      to,         // starting offset for writing control points
                MatrixX<T>& temp_ctrl0, // first temporary control points buffer
                MatrixX<T>& temp_ctrl1); // second temporary control points buffer

        void CopyCtrl(
                MatrixX<T>& P,          // solved points for current dimension and curve
                int         k,          // current dimension
                size_t      co,         // starting offset for reading domain points
                MatrixX<T>& temp_ctrl); // temporary control points buffer

        // various versions of ErrorCurve follow

        // this version only returns number of erroneous input domain points
        int ErrorCurve(
                size_t         k,             // current dimension
                size_t         co,            // starting ofst for reading domain pts
                MatrixX<T>&    ctrl_pts,      // control points
                VectorX<T>&    weights,       // weights associated with control points
                T              err_limit);    // max allowable error

        // in addition to returning number of erroneous input domain points
        // this version inserts erroneous spans into a set
        // allowing the same span to be inserted multiple times w/o duplicates
        int ErrorCurve(
                size_t         k,             // current dimension
                size_t         co,            // starting ofst for reading domain pts
                MatrixX<T>&    ctrl_pts,      // control points
                VectorX<T>&    weights,       // weights associated with control points
                set<int>&      err_spans,     // spans with error greater than err_limit
                T              err_limit);    // max allowable error

        // compute new knots to be inserted into a curve
        void ErrorCurve(
                size_t           k,           // current dimension
                size_t           co,          // starting ofst for reading domain pts
                MatrixX<T>&      ctrl_pts,    // control points
                VectorX<T>&      weights,     // weights associated with control points
                VectorXi&        nnew_knots,  // number of new knots
                vector<T>&       new_knots,   // new knots
                T                err_limit);  // max allowable error

        // DEPRERCATED
#if 0

        // in addition to returning number of erroneous input domain points
        // this version inserts erroneous spans into a set
        // allowing the same span to be inserted multiple times w/o duplicates
        // control points are taken from mfa
        int ErrorCurve(
                size_t       k,             // current dimension
                size_t       co,            // starting ofst for reading domain pts
                size_t       to,            // starting ofst for reading control pts
                set<int>&    err_spans,     // spans with error greater than err_limit
                T            err_limit);    // max allowable error

        // error of points decoded from a curve aligned with a curve of control points
        int ErrorCtrlCurve(
                size_t       k,             // current dimension
                size_t       to,            // starting ofst for reading control pts
                set<int>&    err_spans,     // spans with error greater than err_limit
                T            err_limit);    // max allowable error
#endif

        template <typename>
        friend class NewKnots;

        MFA<T>& mfa;                           // the mfa object
    };
}

#endif
