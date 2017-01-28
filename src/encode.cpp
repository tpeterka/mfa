//--------------------------------------------------------------
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

mfa::
Encoder::
Encoder(MFA& mfa_) :
    mfa(mfa_),
    p(mfa_.p),
    ndom_pts(mfa_.ndom_pts),
    nctrl_pts(mfa_.nctrl_pts),
    domain(mfa_.domain),
    params(mfa_.params),
    ctrl_pts(mfa_.ctrl_pts),
    knots(mfa_.knots),
    po(mfa_.po),
    ko(mfa_.ko),
    co(mfa_.co)
{
}

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
    VectorXi n;                 // number of control point spans in each domain dim
    VectorXi m;                 // number of input data point spans in each domain dim
    int      tot_nparams;       // total number of params = sum of ndom_pts over all dimensions
                                // not the total number of data points, which would be the product
    int      tot_nknots;        // total number of knots = sum of number of knots over all dims
    int      tot_nctrl;         // total number of control points
    int      ndims = ndom_pts.size();        // number of domain dimensions

    Quants(n, m, tot_nparams, tot_nknots, tot_nctrl);

    // following are counters for slicing domain and params into curves in different dimensions
    // TODO: some of these are stored in the mfa object
    vector<size_t> pos(ndims, 0);     // starting offset for params in all dims
    vector<size_t> kos(ndims, 0);     // starting offset for knots in all dims
    vector<size_t> dss(ndims, 1);     // strides for domain pts in all dims
    size_t cs = 1;                    // stride for domain points in curve in cur. dim
    size_t dt = 0;                    // starting ofst for reading domain pts in curve in cur. dim.
    for (size_t k = 1; k < ndims; k++)
    {
        dss[k] = dss[k - 1] * ndom_pts(k - 1);
        pos[k] = pos[k - 1] + ndom_pts(k - 1);
        kos[k] = kos[k - 1] + n(k - 1) + p(k - 1) + 2;
    }

    // control points
    ctrl_pts.resize(tot_nctrl, domain.cols());

    // 2 buffers of temporary control points
    // double buffer needed to write output curves of current dim without changing its input pts
    // temporary control points need to begin with size as many as the input domain points
    // except for the first dimension, which can be the correct number of control points
    // because the input domain points are converted to control points one dimension at a time
    // TODO: need to find a more space-efficient way
    size_t tot_ntemp_ctrl = 1;
    for (size_t k = 0; k < ndims; k++)
        tot_ntemp_ctrl *= (k == 0 ? nctrl_pts(k) : ndom_pts(k));
    MatrixXf temp_ctrl0 = MatrixXf::Zero(tot_ntemp_ctrl, domain.cols());
    MatrixXf temp_ctrl1 = MatrixXf::Zero(tot_ntemp_ctrl, domain.cols());

    VectorXi ntemp_ctrl = ndom_pts;               // current num of temp control pts in each dim

    for (size_t k = 0; k < ndims; k++)            // for all domain dimensions
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
            int span = mfa.FindSpan(k, params(pos[k] + i), kos[k]);
            // debug
            // fprintf(stderr, "p(k) %d n(k) %d span %d params(po + i) %.3f\n",
            //         p(k), n(k), span, params(po + i));
            assert(span - kos[k] <= n(k));            // sanity
            mfa.BasisFuns(k, params(pos[k] + i), span, N, 1, n(k) - 1, i - 1, kos[k]);
        }

        // debug
        // cerr << "k " << k << " N:\n" << N << endl;

        // compute the product Nt x N
        // TODO: NtN is going to be very sparse when it is large: switch to sparse representation
        // NtN has semibandwidth < p + 1 nonzero entries across diagonal
        MatrixXf NtN(n(k) - 1, n(k) - 1);
        NtN = N.transpose() * N;

        // debug
        // cerr << "k " << k << " NtN:\n" << NtN << endl;

        // R is the residual matrix needed for solving NtN * P = R
        MatrixXf R(n(k) - 1, domain.cols());

        // P are the unknown interior control points and the solution to NtN * P = R
        // NtN is positive definite -> do not need pivoting
        // TODO: use a common representation for P and ctrl_pts to avoid copying
        MatrixXf P(n(k) - 1, domain.cols());

        // number of curves in this dimension
        size_t ncurves;
        ncurves = 1;
        for (int i = 0; i < ndims; i++)
        {
            if (i < k)
                ncurves *= nctrl_pts(i);
            else if (i > k)
                ncurves *= ndom_pts(i);
            // NB: current dimension contributes no curves, hence no i == k case
        }
        // debug
        // cerr << "k: " << k << " ncurves: " << ncurves << endl;
        // cerr << "ndom_pts:\n" << ndom_pts << endl;
        // cerr << "ntemp_ctrl:\n" << ntemp_ctrl << endl;
        // if (k > 0 && k % 2 == 1) // input to odd dims is temp_ctrl0
        //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
        // if (k > 0 && k % 2 == 0) // input to even dims is temp_ctrl1
        //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;

        size_t co = 0, to = 0;                    // starting ofst for curve & ctrl pts in cur. dim
        size_t coo = 0, too = 0;                  // co and to at start of contiguous sequence
        for (size_t j = 0; j < ncurves; j++)      // for all the curves in this dimension
        {
            CtrlCurve(N, NtN, R, P, n, k, pos, kos, co, cs, to, temp_ctrl0, temp_ctrl1);

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
                to = too + cs * nctrl_pts(k);
                too = to;
            }
        }                                                  // cuves in this dimension

        // debug
        // if (k % 2 == 0 && k < ndims - 1)
        //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
        // else if (k % 2 && k < ndims - 1)
        //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;
        // else
        //     cerr << "ctrl_pts:\n" << ctrl_pts << endl;

        // adjust offsets and strides for next dimension
        ntemp_ctrl(k) = nctrl_pts(k);
        cs *= ntemp_ctrl(k);

        // free R, NtN, and P
        R.resize(0, 0);
        NtN.resize(0, 0);
        P.resize(0, 0);

        // print progress
        fprintf(stderr, "\rdimension %ld of %d encoded", k + 1, ndims);

    }                                                      // domain dimensions

    fprintf(stderr,"\n");

    // debug
    // cerr << "ctrl_pts:\n" << ctrl_pts << endl;
}

// computes R (residual) vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from the
// original input domain points
void
mfa::
Encoder::
Residual(int       cur_dim,             // current dimension
         MatrixXf& N,                   // matrix of basis function coefficients
         MatrixXf& R,                   // (output) residual matrix allocated by caller
         int       ko,                  // optional index of starting knot
         int       po,                  // optional index of starting parameter
         int       co,                  // optional index of starting domain pt in current curve
         int       cs)                  // optional stride of domain pts in current curve
{
    int n      = N.cols() + 1;               // number of control point spans
    int m      = N.rows() + 1;               // number of input data point spans

    // compute the matrix Rk for eq. 9.63 of P&T, p. 411
    MatrixXf Rk(m - 1, domain.cols());       // eigen frees MatrixX when leaving scope
    MatrixXf Nk;                             // basis coefficients for Rk[i]

    // debug
    // cerr << "Residual domain:\n" << domain << endl;

    for (int k = 1; k < m; k++)
    {
        int span = mfa.FindSpan(cur_dim, params(po + k), ko);
        Nk = MatrixXf::Zero(1, n + 1);      // basis coefficients for Rk[i]
        mfa.BasisFuns(cur_dim, params(po + k), span, Nk, 0, n, 0, ko);

        // debug
        // cerr << "Nk:\n" << Nk << endl;

        // debug
        // cerr << "[" << domain.row(co + k * cs) << "] ["
        //      << domain.row(co) << "] ["
        //      << domain.row(co + m * cs) << "]" << endl;

        Rk.row(k - 1) =
            domain.row(co + k * cs) - Nk(0, 0) * domain.row(co) -
            Nk(0, n) * domain.row(co + m * cs);
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

// computes R (residual) vector of P&T eq. 9.63 and 9.67, p. 411-412 for a curve from a
// new set of input points, not the default input domain
void
mfa::
Encoder::
Residual(int       cur_dim,             // current dimension
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
    // cerr << "Residual in_pts:\n" << in_pts << endl;

    for (int k = 1; k < m; k++)
    {
        int span = mfa.FindSpan(cur_dim, params(po + k), ko);
        Nk = MatrixXf::Zero(1, n + 1);      // basis coefficients for Rk[i]
        mfa.BasisFuns(cur_dim, params(po + k), span, Nk, 0, n, 0, ko);

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
Quants(VectorXi& n,                // (output) number of control point spans in each dim
       VectorXi& m,                // (output) number of input data point spans in each dim
       int&      tot_nparams,      // (output) total number params in all dims
       int&      tot_nknots,       // (output) total number of knots in all dims
       int&      tot_nctrl)        // (output) total number of control points in all dims
{
    if (p.size() != ndom_pts.size())
    {
        fprintf(stderr, "Error: Encode() size of p must equal size of ndom_pts\n");
        exit(1);
    }
    for (size_t i = 0; i < p.size(); i++)
    {
        if (nctrl_pts(i) <= p(i))
        {
            fprintf(stderr, "Error: Encode() number of control points in dimension %ld"
                    "must be at least p + 1 for dimension %ld\n", i, i);
            exit(1);
        }
        if (nctrl_pts(i) > ndom_pts(i))
        {
            fprintf(stderr, "Error: Encode() number of control points in dimension %ld "
                    "cannot be greater than number of input data points in dimension %ld\n", i, i);
            exit(1);
        }
    }

    n.resize(p.size());
    m.resize(p.size());
    tot_nparams = 0;
    tot_nknots  = 0;
    tot_nctrl   = 1;
    for (size_t i = 0; i < p.size(); i++)
    {
        n(i)        =  nctrl_pts(i) - 1;
        m(i)        =  ndom_pts(i)  - 1;
        tot_nparams += ndom_pts(i);
        tot_nknots  += (n(i) + p(i) + 2);
        tot_nctrl   *= nctrl_pts(i);
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
    int ndims = ndom_pts.size();             // number of domain dimensions
    int nctrl_pts = n(k) + 1;                // number of control points in current dim

    // if there is only one dim, copy straight to output
    if (ndims == 1)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        ctrl_pts.row(to) = domain.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + ndom_pts(k) - 1);
        ctrl_pts.row(to + n(k) * cs) = domain.row(co + ndom_pts(k) - 1);
    }
    // first dim copied from domain to temp_ctrl0
    else if (k == 0)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        temp_ctrl0.row(to) = domain.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            temp_ctrl0.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + ndom_pts(k) - 1);
        temp_ctrl0.row(to + n(k) * cs) = domain.row(co + ndom_pts(k) - 1);
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
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        temp_ctrl0.row(to + n(k) * cs) = temp_ctrl1.row(co + (ndom_pts(k) - 1) * cs);
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
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        temp_ctrl1.row(to + n(k) * cs) = temp_ctrl0.row(co + (ndom_pts(k) - 1) * cs);
    }
    // final dim if even is copied from temp_ctrl1 to ctrl_pts
    else if (k == ndims - 1 && k % 2 == 0)
    {
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to, co);
        ctrl_pts.row(to) = temp_ctrl1.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        ctrl_pts.row(to + n(k) * cs) = temp_ctrl1.row(co + (ndom_pts(k) - 1) * cs);
    }
    // final dim if odd is copied from temp_ctrl0 to ctrl_pts
    else if (k == ndims - 1 && k % 2 == 1)
    {
        // debug
        // fprintf(stderr, "t_start[%ld] = d[%ld]\n", to, co);
        ctrl_pts.row(to) = temp_ctrl0.row(co);
        for (int i = 1; i < n(k); i++)
        {
            // debug
            // fprintf(stderr, "t[%ld] = p[%d]\n", to + i * cs, i - 1);
            ctrl_pts.row(to + i * cs) = P.row(i - 1);
        }
        // debug
        // fprintf(stderr, "t_end[%ld] = d[%ld]\n", to + n(k) * cs, co + (ndom_pts(k) - 1) * cs);
        ctrl_pts.row(to + n(k) * cs) = temp_ctrl0.row(co + (ndom_pts(k) - 1) * cs);
    }
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
          vector<size_t> pos,   // starting offsets for params in all dims
          vector<size_t> kos,   // starting offsets for knots in all dims
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
        Residual(k, N, R, kos[k], pos[k], co, cs);             // input points = default domain
    else if (k % 2)
        Residual(k, temp_ctrl0, N, R, kos[k], pos[k], co, cs); // input points = temp_ctrl0
    else
        Residual(k, temp_ctrl1, N, R, kos[k], pos[k], co, cs); // input points = temp_ctrl1

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
    // int ndims = ndom_pts.size();
    // cerr << "k " << k << " P:\n" << P << endl;
    // if (ndims == 1)
    //     cerr << "ctrl_pts:\n" << ctrl_pts << endl;
    // else if (k == 0)
    //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
    // else if (k % 2 == 0 && k < ndims - 1)
    //     cerr << "temp_ctrl0:\n" << temp_ctrl0 << endl;
    // else if (k % 2 == 1 && k < ndims - 1)
    //     cerr << "temp_ctrl1:\n" << temp_ctrl1 << endl;
    // else if (k == ndims - 1)
    //     cerr << "ctrl_pts:\n" << ctrl_pts << endl;
}

// signed normal distance from a point to the domain
// uses 2-point finite differences (first order linear) method to compute gradient and normal vector
// approximates gradient from 2 points diagonally opposite each other in all
// domain dimensions (not from 2 points in each dimension)
float
mfa::
Encoder::
NormalDistance(VectorXf& pt,          // point whose distance from domain is desired
               int       idx)         // index of min. corner of cell in the domain
                                      // that will be used to compute partial derivatives
                                      // (linear) search for correct cell will start at this index
{
    // normal vector = [df/dx, df/dy, df/dz, ..., -1]
    // -1 is the last coordinate of the domain points, ie, the range value
    VectorXf normal(domain.cols());
    int      stride  = 1;                    // stride in domain for current dimension
    int      min_idx = 0;                    // index corresponding to min edge in current dim.
    int      max_idx = 1;                    // index corresponding to max edge in current dim.
    int      last    = domain.cols() - 1;    // last coordinate of a domain pt, ie, the range value

    // convert linear idx to multidim. i,j,k... indices in each domain dimension
    VectorXi ijk(p.size());
    mfa.idx2ijk(idx, ijk);

    for (int i = 0; i < p.size(); i++)       // for all domain dimensions
    {
        // at least 2 points needed in each dimension
        // TODO: do something degenerate if not, but probably will never get to this point
        // because there will be insufficient points to encode in the first place
        assert(ndom_pts(i) >= 2);

        // debug
        // fprintf(stderr, "idx=%d ijk(%d)=%d\n", idx, i, ijk(i));

        // adjust idx until pt is in the cell
        while (ijk(i) >= 0 && domain(idx, i) - pt(i) > mfa.eps)
        {
            if (ijk(i) - 1 >= 0)
            {
                idx -= stride;
                ijk(i)--;
            }
            else
                break;
            // debug
            // fprintf(stderr, "idx=%d ijk(%d)=%d\n", idx, i, ijk(i));
        }
        while (ijk(i) + 1 < ndom_pts(i) && pt(i) - domain(idx + stride, i) > mfa.eps)
        {
            if (ijk(i) + 1 < ndom_pts(i))
            {
                idx += stride;
                ijk(i)++;
            }
            else
                break;
            // debug
            // fprintf(stderr, "idx=%d ijk(%d)=%d\n", idx, i, ijk(i));
        }

        // debug
        // fprintf(stderr, "1: ijk(%d)=%d idx=%d\n", i, ijk(i), idx);

        // set two vertices of the cell
        int i0, i1;
        if (ijk(i) + 1 < ndom_pts(i))
        {
            i0 = idx;
            i1 = idx + stride;
        }
        else
        {
            i0 = idx - stride;
            i1 = idx;
        }

        // debug
        // fprintf(stderr, "2: i=%d i0=%d i1=%d\n", i, i0, i1);

        // debug
        float eps = 1.0e-5;                  // floating point roundoff error
        if (!(domain(i0, i) - eps < pt(i) && pt(i) < domain(i1, i) + eps))
        {
            fprintf(stderr, "i=%d i0=%d i1=%d\n", i, i0, i1);
            cerr << "domain(i0):\n" << domain(i0) <<
                "\npt(i):\n" << pt(i) <<
                "\ndomain(i1):\n" << domain(i1) << endl;
        }
        assert(domain(i0, i) - eps < pt(i) && pt(i) < domain(i1, i) + eps);

        normal(i) = (domain(i1, last) - domain(i0, last)) / (domain(i1, i) - domain(i0, i));
        stride *= ndom_pts(i);
    }
    normal(last) = -1;

    // unit normal
    normal /= normal.norm();

    // project distance from (pt - domain(idx)) to unit normal
    VectorXf dom_pt = domain.row(idx);

    // debug
    // fprintf(stderr, "idx=%d\n", idx);
    // cerr << "unit normal\n" << normal << endl;
    // cerr << "point\n" << pt << endl;
    // cerr << "domain point:\n" << dom_pt << endl;
    // cerr << "pt - dom_pt:\n" << pt - dom_pt << endl;
    // fprintf(stderr, "projection = %e\n\n", normal.dot(pt - dom_pt));

    return normal.dot(pt - dom_pt);
}

// compute the gradient of a grid cell
// uses 2-point finite differences (first order linear) and
// approximates gradient from 2 points diagonally opposite each other in all
// domain dimensions (not from 2 points in each dimension)
void
mfa::
Encoder::
Gradient(int       idx,               // index of min. corner of cell in the domain
         VectorXf& grad)              // output gradient
{
    // normal vector = [df/dx, df/dy, df/dz, ... ]
    grad.resize(domain.cols());
    int      stride  = 1;                    // stride in domain for current dimension
    int      last    = domain.cols() - 1;    // last coordinate of a domain pt, ie, the range value

    // convert linear idx to multidim. i,j,k... indices in each domain dimension
    VectorXi ijk(p.size());
    mfa.idx2ijk(idx, ijk);

    for (int i = 0; i < p.size(); i++)       // for all domain dimensions
    {
        // at least 2 points needed in each dimension
        // TODO: do something degenerate if not, but probably will never get to this point
        // because there will be insufficient points to encode in the first place
        assert(ndom_pts(i) >= 2);

        // debug
        // fprintf(stderr, "idx=%d ijk(%d)=%d\n", idx, i, ijk(i));

        // set two vertices of the cell
        int i0, i1;
        if (ijk(i) + 1 < ndom_pts(i))
        {
            i0 = idx;
            i1 = idx + stride;
        }
        else
        {
            i0 = idx - stride;
            i1 = idx;
        }

        // debug
        // fprintf(stderr, "2: i=%d i0=%d i1=%d\n", i, i0, i1);

        grad(i) = (domain(i1, last) - domain(i0, last)) / (domain(i1, i) - domain(i0, i));
        stride *= ndom_pts(i);
    }

    // debug
    // fprintf(stderr, "idx=%d\n", idx);
    // cerr << "gradient\n" << grad << endl;
}

// compute the gradient of the error of a grid cell
// uses 2-point finite differences (first order linear) and
// approximates gradient from 2 points diagonally opposite each other in all
// domain dimensions (not from 2 points in each dimension)
void
mfa::
Encoder::
ErrorGradient(int       idx,               // index of min. corner of cell in the domain
              VectorXf& grad)              // output gradient
{
    // normal vector = [df/dx, df/dy, df/dz, ... ]
    grad.resize(domain.cols());
    int      stride  = 1;                    // stride in domain for current dimension
    int      last    = domain.cols() - 1;    // last coordinate of a domain pt, ie, the range value

    // convert linear idx to multidim. i,j,k... indices in each domain dimension
    VectorXi ijk(p.size());
    mfa.idx2ijk(idx, ijk);

    // compute i0 and i1 1d and ijk0 and ijk1 nd indices for two points in the cell in each dim.
    // even though the caller provided the minimum corner index as idx, it's
    // possible that idx is at the max end of the domain in some dimension
    // in this case we set i1 <- idx and i0 to be one less
    int i0, i1;                             // 1-d indices of min, max corner points
    VectorXi ijk0(p.size());                // n-d ijk index of min corner
    VectorXi ijk1(p.size());                // n-d ijk index of max corner
    for (int i = 0; i < p.size(); i++)      // for all domain dimensions
    {
        // at least 2 points needed in each dimension
        // TODO: do something degenerate if not, but probably will never get to this point
        // because there will be insufficient points to encode in the first place
        assert(ndom_pts(i) >= 2);

        // set two vertices of the cell
        if (ijk(i) + 1 < ndom_pts(i))
        {
            i0      = idx;
            i1      = idx + stride;
            ijk0(i) = ijk(i);
            ijk1(i) = ijk(i) + 1;
        }
        else
        {
            i0      = idx - stride;
            i1      = idx;
            ijk0(i) = ijk(i) - 1;
            ijk1(i) = ijk(i);
        }
    }

    // compute parameters for the vertices of the cell
    VectorXf param0(p.size());
    VectorXf param1(p.size());
    for (int i = 0; i < p.size(); i++)
    {
        param0(i) = ijk0(i) + po[i];
        param1(i) = ijk1(i) + po[i];
    }

    // approximated values for min, max corners
    VectorXf cpt0(ctrl_pts.cols());          // approximated point
    VectorXf cpt1(ctrl_pts.cols());          // approximated point
    mfa::Decoder decoder(mfa);
    decoder.VolPt(param0, cpt0);
    decoder.VolPt(param1, cpt1);
    
    // compute the gradient of the absolute value of the error
    // NB, the error is not normalized by the range of the data
    for (int i = 0; i < p.size(); i++)
    {
        float e0, e1;                       // fabs(error) at min, max corner
        e0 = fabs(cpt0(last) - domain(i0, last));
        e1 = fabs(cpt1(last) - domain(i1, last));
        grad(i) = (e1 - e0) / (domain(i1, i) - domain(i0, i));
    }

    // debug
    // fprintf(stderr, "idx=%d\n", idx);
    // cerr << "gradient\n" << grad << endl;
}
