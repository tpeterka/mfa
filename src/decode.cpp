//--------------------------------------------------------------
// decoder object
// ref: [P&T] Piegl & Tiller, The NURBS Book, 1995
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <mfa/decode.hpp>

#include <Eigen/Dense>

#include <vector>
#include <iostream>

using namespace std;

mfa::
Decoder::
Decoder(MFA& mfa_) :
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
    // ensure that encoding was already done
    if (!p.size()         ||
        !ndom_pts.size()  ||
        !nctrl_pts.size() ||
        !domain.size()    ||
        !params.size()    ||
        !ctrl_pts.size()  ||
        !knots.size())
    {
        fprintf(stderr, "Decoder() error: Attempting to decode before encoding.\n");
        exit(0);
    }
}

// computes approximated points from a given set of domain points and an n-d NURBS volume
// P&T eq. 9.77, p. 424
// this version recomputes parameter values of input points and
// recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
// assumes all vectors have been correctly resized by the caller
void
mfa::
Decoder::
Decode(MatrixXf& approx)                 // pts in approximated volume (1st dim. changes fastest)
{
    vector<size_t> iter(p.size(), 0);    // parameter index (iteration count) in current dim.
    vector<size_t> ofst(p.size(), 0);    // start of current dim in linearized params

    for (size_t i = 0; i < p.size() - 1; i++)
        ofst[i + 1] = ofst[i] + ndom_pts(i);

    // eigen frees following temp vectors when leaving scope
    VectorXf dpt(domain.cols());         // original data point
    VectorXf cpt(ctrl_pts.cols());       // approximated point
    VectorXf d(domain.cols());           // apt - dpt
    VectorXf param(p.size());            // parameters for one point
    for (size_t i = 0; i < domain.rows(); i++)
    {
        // debug
        // cerr << "input point:\n" << domain.row(i) << endl;

        // extract parameter vector for one input point from the linearized vector of all params
        for (size_t j = 0; j < p.size(); j++)
            param(j) = params(iter[j] + ofst[j]);

        // compute approximated point for this parameter vector
        VolPt(param, cpt);

        // update the indices in the linearized vector of all params for next input point
        for (size_t j = 0; j < p.size(); j++)
        {
            if (iter[j] < ndom_pts(j) - 1)
            {
                iter[j]++;
                break;
            }
            else
                iter[j] = 0;
        }

        approx.row(i) = cpt;

        // print progress
        if (i > 0 && domain.rows() >= 100 && i % (domain.rows() / 100) == 0)
            fprintf(stderr, "\r%.0f %% decoded", (float)i / (float)(domain.rows()) * 100);
    }
    fprintf(stderr, "\r100 %% decoded\n");
}

// compute a point from a NURBS curve at a given parameter value
// algorithm 4.1, Piegl & Tiller (P&T) p.124
// this version recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
void
mfa::
Decoder::
CurvePt(int       cur_dim,                     // current dimension
        float     param,                       // parameter value of desired point
        VectorXf& out_pt)                      // (output) point
{
    int n      = (int)ctrl_pts.rows() - 1;     // number of control point spans
    int span   = mfa.FindSpan(cur_dim, param);
    MatrixXf N = MatrixXf::Zero(1, n + 1);     // basis coefficients
    mfa.BasisFuns(cur_dim, param, span, N, 0, n, 0);
    out_pt = VectorXf::Zero(ctrl_pts.cols());  // initializes and resizes

    for (int j = 0; j <= p(cur_dim); j++)
        out_pt += N(0, j + span - p(cur_dim)) * ctrl_pts.row(span - p(cur_dim) + j);

    // debug
    // cerr << "n " << n << " param " << param << " span " << span << " out_pt " << out_pt << endl;
    // cerr << " N " << N << endl;
}

// decode one curve in one dimension (e.g., y direction) from a set of previously computed
// control curves in the prior dimension (e.g., x direction)
// new curve in current dimension (e.g., y) is normal to curves in prior dimension (e.g., x) and
// is located in the prior dimension (e.g., x) at parameter value prev_param
//
// assumes the caller resized the output out_pts to the correct size, which is number of original
// domain points in the current dimension (e.g., y)
//
// currently not used but can be useful for getting a cross-section curve from a surface
// would need to be expanded to get a curve from a higher dimensional space, currently gets 1D curve
// from 2D surface
void
mfa::
Decoder::
DecodeCurve(size_t    cur_dim,    // current dimension
            float     pre_param,  // parameter value in prior dimension of the pts in the curve
            size_t    ko,         // starting offset for knots in current dim
            size_t    cur_cs,     // stride for control points in current dim
            size_t    pre_cs,     // stride for control points in prior dim
            MatrixXf& out_pts)    // output approximated pts for the curve
{
    if (cur_dim == 0 || cur_dim >= p.size())
    {
        fprintf(stderr, "Error in DecodeCurve(): "
                "cur_dim out of range (must be 1 <= cur_dim <= %ld\n",
                p.size() - 1);
        exit(0);
    }

    // get one set of knots in the prior dimension
    size_t nknots = nctrl_pts(cur_dim - 1) + p(cur_dim - 1) + 1;
    VectorXf pre_knots(nknots);
    for (size_t i = 0; i < nknots; i++)
        pre_knots(i) = knots(ko + i);

    // debug
    // cerr << "pre_knots:\n" << pre_knots << endl;

    // control points for one curve in prior dimension
    MatrixXf pre_ctrl_pts(nctrl_pts(cur_dim - 1), domain.cols());
    size_t n = 0;                            // index of current control point

    // for the desired approximated points in the current dim, same number as domain points
    for (size_t j = 0; j < ndom_pts(cur_dim); j++)
    {
        // get one curve of control points in the prior dim
        for (size_t i = 0; i < nctrl_pts(cur_dim - 1); i++)
        {
            pre_ctrl_pts.row(i) = ctrl_pts.row(n);
            n += pre_cs;
        }
        n += (cur_cs - nctrl_pts(cur_dim - 1));

        // debug
        // cerr << "pre_ctrl_pts:\n" << pre_ctrl_pts << endl;

        // get the approximated point
        VectorXf out_pt(domain.cols());
        CurvePt(cur_dim - 1, pre_param, out_pt);
        out_pts.row(j) = out_pt;

        // debug
        // cerr << "out_pt: " << out_pt << endl;
    }

    // debug
    // cerr << "out_pts:\n " << out_pts << endl;
}

// compute a point from a NURBS n-d volume at a given parameter value
// algorithm 4.3, Piegl & Tiller (P&T) p.134
// this version recomputes basis functions rather than taking them as an input
// this version also assumes weights = 1; no division by weight is done
//
// There are two types of dimensionality:
// 1. The dimensionality of the NURBS tensor product (p.size())
// (1D = NURBS curve, 2D = surface, 3D = volumem 4D = hypervolume, etc.)
// 2. The dimensionality of individual control points (ctrl_pts.cols())
// p.size() should be < ctrl_pts.cols()
void
mfa::
Decoder::
VolPt(VectorXf& param,                       // parameter value in each dim. of desired point
      VectorXf& out_pt)                      // (output) point
{
    // debug
    // cerr << "\n\nparam:\n" << param << endl;

    // check dimensionality for sanity
    assert(p.size() < ctrl_pts.cols());

    out_pt = VectorXf::Zero(ctrl_pts.cols());// initializes and resizes
    vector <MatrixXf> N(p.size());           // basis functions in each dim.
    vector<VectorXf>  temp(p.size());        // temporary point in each dim.
    vector<int>       span(p.size());        // span in each dim.
    vector<int>       n(p.size());           // number of control point spans in each dim
    vector<int>       iter(p.size());        // iteration number in each dim.
    int               tot_iters = 1;         // tot. num. iterations in flattened n-d nested loops
    vector<size_t>    ko(p.size(), 0);       // starting offset for knots in current dim
    vector<size_t>    co(p.size());          // starting offset for control points in each dim
    vector<size_t>    ct(p.size());          // relative coordinates of ctrl pt of current iteration
    vector<size_t>    cs(p.size());          // stride for next co in each dim
    VectorXf          ctrl_pt(ctrl_pts.cols()); // one control point

    for (size_t i = 0; i < p.size(); i++)    // for all dims
    {
        temp[i]    = VectorXf::Zero(ctrl_pts.cols());
        iter[i]    = 0;
        tot_iters  *= (p(i) + 1);
        n[i]       = (int)nctrl_pts(i) - 1;
        span[i]    = mfa.FindSpan(i, param(i), ko[i]);
        N[i]       = MatrixXf::Zero(1, n[i] + 1);
        mfa.BasisFuns(i, param(i), span[i], N[i], 0, n[i], 0, ko[i]);
        if (i == 0)
            cs[i] = 1;
        else
            cs[i] = cs[i - 1] * nctrl_pts(i);
        co[i] = span[i] - p(i) - ko[i];
        if (i < p.size() - 1)
            ko[i + 1] = ko[i] + n[i] + p(i) + 2; // n[i]+p(i)+2 =  number of knots in current dim.

        // debug
        // cerr << "N[" << i << "] = " << N[i] << endl;
        // fprintf(stderr, "i=%d co[i]=%d span[i]=%d p(i)=%d ko[i]=%d cs[i]=%d\n",
        //         i, co[i], span[i], p(i), ko[i], cs[i]);
    }

    for (int i = 0; i < tot_iters; i++)      // 1-d flattening all n-d nested loop computations
    {
        // compute coordinates of first control point of curve corresponding to this iteration
        // these are relative to start of the box of control points located at co
        int div = tot_iters;
        int i_temp = i;
        for (int j = p.size() - 1; j >= 0; j--)
        {
            div    /= (p(j) + 1);
            ct[j]  = i_temp / div;
            i_temp -= (ct[j] * div);
        }

        // control point linear order index
        int ctrl_idx = 0;
        int fac = 1;
        for (int j = 0; j < p.size(); j++)
        {
            ctrl_idx += (co[j] + ct[j]) * fac;
            fac *= nctrl_pts(j);
        }

        // always compute the point in the first dimension
        ctrl_pt = ctrl_pts.row(ctrl_idx);
        temp[0] += (N[0])(0, iter[0] + span[0] - p(0)) * ctrl_pt;

        // debug
//         fprintf(stderr, "1: temp[0] += N[0, %d] * ctrl_pt\n", iter[0] + span[0] - p(0));
//         cerr << "span: " << span[0] << " ctrl_pt=\n" << ctrl_pt << endl;
        // cerr << "1: i=" << i << " iter[0]=" << iter[0] << " temp[0]=\n" << temp[0] << endl;
        iter[0]++;

        // for all dimensions except last, check if span is finished
        for (size_t k = 0; k < p.size() - 1; k++)
        {
            if (iter[k] - 1 == p(k))
            {
                // compute point in next higher dimension
                temp[k + 1] += (N[k + 1])(0, iter[k + 1] + span[k + 1] - ko[k + 1] - p(k + 1)) *
                    temp[k];

                // debug
                // fprintf(stderr, "2: temp[%d] += N[%d, %d] * temp[%d]\n",
                //         k + 1, k + 1, iter[k + 1] + span[k + 1] - ko[k + 1] - p(k + 1), k);
                // fprintf(stderr, "3: temp[%d] = 0\n", k);
                // cerr << "2: i=" << i << " k=" << k << " iter[k+1]=" << iter[k+1] <<
                //     " temp[k+1]=\n" << temp[k+1] << endl;

                // reset the computation for the current dimension
                temp[k]    = VectorXf::Zero(ctrl_pts.cols());
                iter[k]    = 0;
                iter[k + 1]++;
            }
        }
    }

    out_pt = temp[p.size() - 1];

    // debug
    // fprintf(stderr, "out_pt = temp[%d]\n", p.size() - 1);
    // cerr << "out_pt:\n" << out_pt << endl;
}
