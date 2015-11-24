#include <mfa/types.hpp>
#include <mfa/encode.hpp>
#include <mfa/decode.hpp>

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

//--------------------------------------------------------------
// a simple example of encoding / decoding some curve data
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

void GenerateConstantData(vector<Pt1d>&  domain,
                          vector<float>& range,
                          int            npts,
                          float          min_x,
                          float          max_x)
{
    domain.resize(npts);
    range.resize(npts);
    float dx = (max_x - min_x) / npts;

    // the simplest constant function
    for (int i = 0; i < npts; i++)
    {
        domain[i].x = min_x + i * dx;
        range[i] = 1.0f;
    }

    // debug
    cerr << domain.size() << " input points:" << endl;
    for (size_t i = 0; i < domain.size(); i++)
        cerr << "(" << domain[i].x << ", " << range[i] << ")";
    cerr << endl;
}

void GenerateSineData(vector<Pt1d>&  domain,
                      vector<float>& range,
                      int            npts,
                      float          min_x,
                      float          max_x)
{
    domain.resize(npts);
    range.resize(npts);
    float dx = (max_x - min_x) / npts;

    // a sine function
    for (int i = 0; i < npts; i++)
    {
        domain[i].x = min_x + i * dx;
        range[i] = sin(domain[i].x);
    }

    // debug
    cerr << domain.size() << " input points:" << endl;
    for (size_t i = 0; i < domain.size(); i++)
        cerr << "(" << domain[i].x << ", " << range[i] << ")";
    cerr << endl;
}

void PrintApprox(vector<Pt2d>   ctrl_pts,
                 vector <float> knots,
                 vector <float> errs,
                 float          max_err)
{
    cerr << ctrl_pts.size() << " control points:" << endl;
    for (size_t i = 0; i < ctrl_pts.size(); i++)
        cerr << ctrl_pts[i] << " ";
    cerr << endl;

    cerr << knots.size() << " knots:" << endl;
    for (size_t i = 0; i < knots.size(); i++)
        cerr << knots[i] << " ";
    cerr << endl;

    // cerr << errs.size() << " errors at input points:" << endl;
    // for (size_t i = 0; i < errs.size(); i++)
    //     cerr << errs[i] << " ";
    // cerr << endl;

    cerr << "max_err = " << max_err << endl;
}

int main(int argc, char** argv)
{
    int           p         = 3;             // degree
    vector<Pt1d>  domain;                    // input data domain
    vector<float> range;                     // input data range

    // generate data

    // constant function
    // int           nin_pts   = 100;            // number of input points
    // int           nctrl_pts = 7;              // number of control (output) points

    // sine function
    int           nin_pts   = 100;            // number of input points
    int           nctrl_pts = 10;             // number of control (output) points

    // constant function
    // GenerateConstantData(domain, range, nin_pts, 0.0, nin_pts - 1.0);

    // sine function
    GenerateSineData(domain, range, nin_pts, 0.0, 2 * M_PI);

    // encode
    vector<Pt2d>  ctrl_pts;                  // control points
    vector<float> knots;                     // knots
    Approx1d(p, nctrl_pts, domain, range, ctrl_pts, knots);

    // compute error
    vector<Pt2d>  approx(domain.size());     // approximated points (for debug and rendering only)
    vector<float> errs(domain.size());       // error at each input point
    float         max_err;                   // max of errs
    int           max_niter  = 10;           // max number of search iterations
    float         err_bound  = 0.1;          // desrired error bound
    int           search_rad = 4;            // search range is +/- this many input parameters

    // use one or the other of the following

    // plain max
    // MaxErr1d(p, domain, range, ctrl_pts, knots, approx, errs, max_err);

    // max norm, should be better but more expensive
    MaxNormErr1d(p,
                 domain,
                 range,
                 ctrl_pts,
                 knots,
                 max_niter,
                 err_bound,
                 search_rad,
                 approx,
                 errs,
                 max_err);

    // print results
    PrintApprox(ctrl_pts, knots, errs, max_err);

    // cleanup
    domain.clear();
    range.clear();
    knots.clear();
    ctrl_pts.clear();
    errs.clear();
}
