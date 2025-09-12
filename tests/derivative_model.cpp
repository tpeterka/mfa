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

            for (int i = 0; i < vardim; i++)
            {
                val = (10 * (k+1)) * sinc(dom_pt(0)) * sinc(dom_pt(1)) + i;
                input.domain(j, dmin + i) = val;
            }
            
        }  
    }
}

void testModels(int test_id, const mfa::MFA<real_t>& mfa, const mfa::MFA<real_t>& dmfa, const MatrixX<real_t>& test_pts, const VectorX<real_t>& extents, const VectorXi& derivs)
{
    int ntest_pts = test_pts.rows();
    int dom_dim = mfa.dom_dim;
    int pt_dim = mfa.pt_dim;
    VectorX<real_t> test_pt(dom_dim);
    VectorX<real_t> deriv1(pt_dim);
    VectorX<real_t> deriv2(pt_dim);
    VectorX<real_t> scale_vec(dom_dim);
    for (int i = 0; i < dom_dim; i++)
    {
        scale_vec(i) = 1.0 / pow(extents(i), derivs(i));
    }
    real_t scale = scale_vec.prod();

    for (int i = 0; i < ntest_pts; i++)
    {
        test_pt = test_pts.row(i);

        mfa.Decode(test_pt, deriv1, derivs);                        // differentiate original model
        deriv1.tail(pt_dim - dom_dim) *= scale;  // account for scaling of derivative
        dmfa.Decode(test_pt, deriv2);                              // evaluate derivative model    

        if ((deriv1 - deriv2).norm() > 1e-14)
        {
            fmt::print(stderr, "ERROR: Derivative model does not match differentiated original model. Test ID: {}\n", test_id);
            fmt::print(stderr, "       test_pt = [{}]\n", fmt::join(test_pt, " "));
            fmt::print(stderr, "       original model derivative = [{}]\n", fmt::join(deriv1, " "));
            fmt::print(stderr, "       derivative model evaluation = [{}]\n", fmt::join(deriv2, " "));
            fmt::print(stderr, "       difference = [{}]\n", fmt::join(deriv1 - deriv2, " "));
            exit(1);
        }
        // else
        // {
        //     fmt::print(stderr, "Test ID: {}\n", test_id);
        //     fmt::print(stderr, "       test_pt = [{}]\n", fmt::join(test_pt, " "));
        //     fmt::print(stderr, "       original model derivative = [{}]\n", fmt::join(deriv1, " "));
        //     fmt::print(stderr, "       derivative model evaluation = [{}]\n", fmt::join(deriv2, " "));
        //     fmt::print(stderr, "       difference = [{}]\n", fmt::join(deriv1 - deriv2, " "));
        // }
    }

    fmt::print(stderr, "Test {} passed\n", test_id);

    return;
}

int main(int argc, char** argv)
{
    // Problem setup
    int dom_dim = 2;
    int verbose = 0;
    int degree = 3;
    VectorXi nctrl_pts(dom_dim);
    nctrl_pts << 20, 30;

    // Data setup
    VectorXi npts(dom_dim);
    VectorX<real_t> domain_mins(dom_dim), domain_maxs(dom_dim), extents(dom_dim);
    npts << 100, 150;
    domain_mins << -4 * M_PI, -4 * M_PI;
    domain_maxs << 3 * M_PI, 3 * M_PI;
    extents = domain_maxs - domain_mins;

    // Define test locations
    int ntest_pts = 6;
    MatrixX<real_t> test_pts(ntest_pts, dom_dim);
    VectorX<real_t> test_pt(dom_dim);
    test_pts << 0.0, 0.0,
                0.1, 0.2,
                1, 0,
                1, 1,
                0.7, 0.34,
                0.5, 0.99;

    // --------------------------------------------------------
    // First set of tests: 1 science variable, scalar-valued
    // --------------------------------------------------------
    mfa::MFA<real_t>* mfa = new mfa::MFA<real_t>(dom_dim, verbose);
    mfa->AddGeometry(dom_dim);
    mfa->AddVariable(degree, nctrl_pts, 1);

    // Generate data and encode
    mfa::PointSet<real_t>* input = new mfa::PointSet<real_t>(dom_dim, mfa->model_dims(), npts.prod(), npts);
    generate_data(*input, domain_mins, domain_maxs);
    input->set_domain_params();
    mfa->FixedEncode(*input, 0, false, false);

    // Get the derivative model
    // differentiate w.r.t. first dimension
    VectorXi derivs = VectorXi::Zero(dom_dim);
    derivs(0) = 1;
    mfa::MFA<real_t>* dmfa = mfa->getDerivativeModel(0, extents(0));

    // Test that the two ways of differentiating give the same result
    testModels(1, *mfa, *dmfa, test_pts, extents, derivs);
    delete dmfa;

    // Differentiate w.r.t. second dimension
    derivs = VectorXi::Zero(dom_dim);
    derivs(1) = 1;
    dmfa = mfa->getDerivativeModel(1, extents(1));
    testModels(2, *mfa, *dmfa, test_pts, extents, derivs);
    delete input;
    delete mfa;
    delete dmfa;

    // --------------------------------------------------------
    // Second set of tests: 3 science variables, some of which are vector-valued
    // --------------------------------------------------------
    mfa = new mfa::MFA<real_t>(dom_dim, verbose);
    mfa->AddGeometry(dom_dim);
    mfa->AddVariable(degree, nctrl_pts, 1);
    mfa->AddVariable(degree, nctrl_pts, 2);
    mfa->AddVariable(degree, nctrl_pts, 3);

    input = new mfa::PointSet<real_t>(dom_dim, mfa->model_dims(), npts.prod(), npts); 
    generate_data(*input, domain_mins, domain_maxs);
    input->set_domain_params();
    mfa->FixedEncode(*input, 0, false, false);

    // Differentiate w.r.t. first dimension
    derivs = VectorXi::Zero(dom_dim);
    derivs(0) = 1;
    dmfa = mfa->getDerivativeModel(0, extents(0));
    testModels(3, *mfa, *dmfa, test_pts, extents, derivs);
    delete dmfa;

    // Differentiate w.r.t. second dimension
    derivs = VectorXi::Zero(dom_dim);
    derivs(1) = 1;
    dmfa = mfa->getDerivativeModel(1, extents(1));
    testModels(4, *mfa, *dmfa, test_pts, extents, derivs);
    delete input;
    delete mfa;
    delete dmfa;

    // --------------------------------------------------------
    // Third set of tests: Differentiating multiple times
    // --------------------------------------------------------
    mfa = new mfa::MFA<real_t>(dom_dim, verbose);
    mfa->AddGeometry(dom_dim);
    mfa->AddVariable(degree, nctrl_pts, 2);

    input = new mfa::PointSet<real_t>(dom_dim, mfa->model_dims(), npts.prod(), npts);
    generate_data(*input, domain_mins, domain_maxs);
    input->set_domain_params();
    mfa->FixedEncode(*input, 0, false, false);

    // Second-order derivative w.r.t. first dimension
    derivs = VectorXi::Zero(dom_dim);
    derivs(0) = 2;
    mfa::MFA<real_t>* dmfa1 = mfa->getDerivativeModel(0, extents(0));
    mfa::MFA<real_t>* dmfa2 = dmfa1->getDerivativeModel(0, extents(0));
    testModels(5, *mfa, *dmfa2, test_pts, extents, derivs);
    delete dmfa1;
    delete dmfa2;

    // Second-order derivative w.r.t. second dimension
    derivs = VectorXi::Zero(dom_dim);
    derivs(1) = 2;
    dmfa1 = mfa->getDerivativeModel(1, extents(1));
    dmfa2 = dmfa1->getDerivativeModel(1, extents(1));
    testModels(6, *mfa, *dmfa2, test_pts, extents, derivs);
    delete dmfa1;
    delete dmfa2;

    // Mixed second-order derivative
    derivs = VectorXi::Zero(dom_dim);
    derivs(0) = 1;
    derivs(1) = 1;
    dmfa1 = mfa->getDerivativeModel(0, extents(0));
    dmfa2 = dmfa1->getDerivativeModel(1, extents(1));
    testModels(7, *mfa, *dmfa2, test_pts, extents, derivs);
    delete dmfa1;
    delete dmfa2;

    // Mixed second-order derivative, other order of differentiation
    derivs = VectorXi::Zero(dom_dim);
    derivs(0) = 1;
    derivs(1) = 1;
    dmfa1 = mfa->getDerivativeModel(1, extents(1));
    dmfa2 = dmfa1->getDerivativeModel(0, extents(0));
    testModels(8, *mfa, *dmfa2, test_pts, extents, derivs);
    delete dmfa1;
    delete dmfa2;

    // Differentiating up to degree of the model
    derivs = VectorXi::Zero(dom_dim);
    derivs(1) = 3;
    dmfa1 = mfa->getDerivativeModel(1, extents(1));
    dmfa2 = dmfa1->getDerivativeModel(1, extents(1));
    mfa::MFA<real_t>* dmfa3 = dmfa2->getDerivativeModel(1, extents(1));
    testModels(9, *mfa, *dmfa3, test_pts, extents, derivs);
    delete input;
    delete mfa;
    delete dmfa1;
    delete dmfa2;
    delete dmfa3;
}