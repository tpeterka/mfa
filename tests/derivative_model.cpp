//--------------------------------------------------------------
// Tests that two ways of differentiating an MFA give the same result
// First, the derivative of an MFA is evaluated at a point. 
// Second, a new MFA is created that represents the derivative of the original MFA,
// and this new MFA is evaluated at the same point.
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <set>

#include "opts.h"

#include "parser.hpp"
#include "block.hpp"
#include "example-setup.hpp"

using namespace std;

real_t sinc(real_t x)
{
    if (x == 0.0)
        return 1.0;
    else
        return sin(x) / x;
}

void generate_data(mfa::PointSet<real_t>& input, const VectorX<real_t>& mins, const VectorX<real_t>& maxs)
{
    int dom_dim = input.dom_dim;
    int pt_dim = input.pt_dim;
    int nvars = input.nvars();

    if (dom_dim != 2)
    {
        throw mfa::MFAError("generate_data() only implemented for 2D domain");
    }
    if (pt_dim != dom_dim + nvars)
    {
        throw mfa::MFAError("generate_data() only implemented for pt_dim = dom_dim + nvars");
    }

    VectorX<real_t> p0 = mins;
    VectorX<real_t> p1 = maxs;
    VectorX<real_t> d(dom_dim);
    for (int i = 0; i < dom_dim; i++)
    {
        d(i) = (p1(i) - p0(i)) / (input.ndom_pts(i) - 1);
    }

    const VectorXi& ndom_pts = input.ndom_pts();

    // assign values to the domain (geometry)
    mfa::VolIterator vol_it(ndom_pts);
    while (!vol_it.done())
    {
        int j = vol_it.cur_iter();
        // compute geometry coordinates of domain point
        for (auto i = 0; i < dom_dim; i++)
            input.domain(j, i) = p0(i) + vol_it.idx_dim(i) * d(i);

        vol_it.incr_iter();
    }

    // assign values to the range (science variables)
    real_t val;
    VectorX<real_t> dom_pt(dom_dim);
    for (int j = 0; j < input.domain.rows(); j++)
    {
        input.geom_coords(j, dom_pt);          // fill dom_pt
        for (auto k = 0; k < nvars; k++)        // for all science variables
        {
            int dmin = input.var_min(k);
            int vardim = input.var_dim(k);

            val = sinc(dom_pt(0)) * sinc(dom_pt(1));
            input.domain(j, dmin) = val;
        }  
    }
}

int main(int argc, char** argv)
{
    // Problem setup
    int dom_dim = 2;
    int pt_dim = 3;
    int verbose = 1;
    int degree = 3;
    VectorXi nctrl_pts(dom_dim);
    nctrl_pts << 20, 30;

    // Data setup
    VectorXi npts(dom_dim);
    VectorX<real_t> domain_mins(dom_dim), domain_maxs(dom_dim), extents(dom_dim);
    npts << 100, 150;
    domain_mins << -4 * M_PI, -4 * M_PI;
    domain_maxs << 4 * M_PI, 4 * M_PI;
    extents = domain_maxs - domain_mins;

    // Set up the structure of the MFA
    mfa::MFA<real_t> mfa(dom_dim, verbose);
    mfa.AddGeometry(dom_dim);
    mfa.AddVariable(degree, nctrl_pts, pt_dim - dom_dim);

    // Create the data set for modeling
    mfa::PointSet<real_t> input(dom_dim, mfa.model_dims(), npts.prod(), npts);
    generate_data(input, domain_mins, domain_maxs);
    input.set_domain_params();

    // Encode the data (solve for optimal control points)
    mfa.FixedEncode(input, 0, false, false);

    // Get the derivative model
    int deriv_dim = 0;   // differentiate w.r.t. first domain dimension
    mfa::MFA<real_t>* dmfa = mfa.getDerivativeModel(deriv_dim, extents(deriv_dim));

    // Decode the derivative model and differentiate the original model at the same locations
    int ntest_pts = 6;
    MatrixX<real_t> test_pts(ntest_pts, dom_dim);
    VectorX<real_t> test_pt(dom_dim);
    test_pts << 0.0, 0.0,
                0.1, 0.2,
                1, 0,
                1, 1,
                0.7, 0.34,
                0.5, 0.99;
    VectorX<real_t> deriv1(pt_dim);
    VectorX<real_t> deriv2(pt_dim);
    VectorXi derivs = VectorXi::Zero(dom_dim);
    derivs(deriv_dim) = 1;

    for (int i = 0; i < ntest_pts; i++)
    {
        test_pt = test_pts.row(i);

        mfa.Decode(test_pt, deriv1, derivs);                        // differentiate original model
        deriv1.tail(pt_dim - dom_dim) *= 1.0 / extents(deriv_dim);  // account for scaling of derivative
        dmfa->Decode(test_pt, deriv2);                              // evaluate derivative model    

        if ((deriv1 - deriv2).norm() > 1e-14)
        {
            fmt::print(stderr, "ERROR: Derivative model does not match differentiated original model at test point {}\n", i);
            fmt::print(stderr, "       test_pt = [{}]\n", fmt::join(test_pt, " "));
            fmt::print(stderr, "       original model derivative = [{}]\n", fmt::join(deriv1, " "));
            fmt::print(stderr, "       derivative model evaluation = [{}]\n", fmt::join(deriv2, " "));
            fmt::print(stderr, "       difference = [{}]\n", fmt::join(deriv1 - deriv2, " "));
            exit(1);
        }
    }

    delete dmfa;
}