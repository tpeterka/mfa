//--------------------------------------------------------------
// Point-set value decoding: numerical reference, fallbacks, and workspace reuse.
// MFA contributors
//--------------------------------------------------------------

#undef NDEBUG                       // Keep Eigen's allocation assertions active in Release test builds.
#define EIGEN_RUNTIME_NO_MALLOC

#include <mfa/mfa.hpp>

#include <cmath>
#include <exception>
#include <future>
#include <iostream>
#include <limits>

bool fail_check(const char* condition, int line)
{
    std::cerr << "ERROR: decode-pointset-test check failed at line " << line << ": " << condition << '\n';
    return false;
}

#define CHECK(condition) \
    do { if (!(condition)) return fail_check(#condition, __LINE__); } while (false)

#define CHECK_MFA_ERROR(expression) \
    do \
    { \
        bool threw_mfa_error = false; \
        try { expression; } \
        catch (const mfa::MFAError&) { threw_mfa_error = true; } \
        catch (...) {} \
        CHECK(threw_mfa_error); \
    } while (false)

template <typename T>
bool check_pointsets()
{
    for (int dom_dim = 1; dom_dim <= 3; dom_dim++)
    for (int components : {1, 2, 4})
    for (int degree = 0; degree <= 3; degree++)
    {
        VectorXi p(dom_dim), nctrl(dom_dim), dims(3);
        for (int d = 0; d < dom_dim; d++)
        {
            p(d) = degree + d % 2;
            nctrl(d) = p(d) + 4;
        }
        dims << dom_dim, components, 1;
        mfa::MFA_Data<T> model(p, nctrl, dom_dim, dom_dim + components - 1);
        auto& tensor = model.tmesh.tensor_prods[0];
        tensor.ctrl_pts.resize(nctrl.prod(), components);
        tensor.weights = VectorX<T>::Ones(nctrl.prod());
        for (int i = 0; i < tensor.ctrl_pts.rows(); i++)
            for (int c = 0; c < components; c++)
                tensor.ctrl_pts(i, c) = T(std::sin(i * .17 + c) + c);
        for (int d = 0; d < dom_dim; d++)
        {
            auto& knots = model.tmesh.all_knots[d];
            for (auto& knot : knots)
                knot *= knot;
            if (p(d) > 0)
                knots[p(d) + 2] = knots[p(d) + 1];
        }

        mfa::PointSet<T> points(dom_dim, dims, 113);
        points.domain.setConstant(T(-7));
        points.params->param_list.resize(points.npts, dom_dim);
        for (int i = 0; i < points.npts; i++)
            for (int d = 0; d < dom_dim; d++)
                points.params->param_list(i, d) = T((i * (17 + d * 6)) % 101) / T(100);
        points.params->param_list.row(0).setZero();
        points.params->param_list.row(1).setOnes();
        // Exercise an interior repeated knot exactly as well as nearby arbitrary points.
        for (int d = 0; d < dom_dim; d++)
            points.params->param_list(2, d) = model.tmesh.all_knots[d][p(d) + 1];
        const MatrixX<T> original_params = points.params->param_list;
        mfa::Decoder<T> decoder(model, 0);
        VectorX<T> param(dom_dim), out(components);
        const T tolerance = T(500) * std::numeric_limits<T>::epsilon();

        for (bool weighted : {false, true})
        {
            if (weighted)
                for (int i = 0; i < tensor.weights.size(); i++)
                    tensor.weights(i) = T(1) + T(i % 5) / T(10);
            for (int request = 0; request < 3; request++)
            {
                VectorXi derivs;
                if (request)
                {
                    derivs = VectorXi::Zero(dom_dim);
                    if (request == 2 && degree > 0)
                        derivs(0) = 1;
                }
                mfa::DecodeInfo<T> info(model, derivs);
                MatrixX<T> reference(points.npts, components);
                for (int i = 0; i < points.npts; i++)
                {
                    points.params->pt_params(i, param);
                    decoder.VolPt(param, out, info, tensor, derivs);
                    reference.row(i) = out.transpose();
                }
                decoder.DecodePointSet(points, dom_dim, dom_dim + components - 1, derivs);
                CHECK((reference - points.domain.middleCols(dom_dim, components)).cwiseAbs().maxCoeff() < tolerance);
                CHECK((points.domain.leftCols(dom_dim).array() == T(-7)).all());
                CHECK((points.domain.rightCols(1).array() == T(-7)).all());
                CHECK((points.params->param_list.array() == original_params.array()).all());
            }
        }

#if defined(MFA_NO_WEIGHTS) && !defined(MFA_TMESH) && !defined(MFA_OVERLAYS)
        mfa::FastDecodeInfo<T> fast_info(decoder);
        CHECK(fast_info.td.empty());
        // The existing public scalar wrapper must remain allocation-free after scratch setup.
        TensorProduct<T> scalar;
        scalar.ctrl_pts = tensor.ctrl_pts.col(0);
        out.resize(1);
        param.setConstant(T(.43));
        Eigen::internal::set_is_malloc_allowed(false);
        decoder.FastVolPt(param, out, fast_info, scalar);
        Eigen::internal::set_is_malloc_allowed(true);
        CHECK(std::isfinite(out(0)));
        fast_info.ResizeDers(1);
        CHECK(fast_info.td.size() == size_t(dom_dim + 1));
        // FastGrad's existing derivative kernel requires positive degree in every axis.
        if (p.minCoeff() > 0)
        {
            VectorX<T> gradient(dom_dim);
            T value;
            decoder.FastGrad(param, fast_info, scalar, gradient, &value);
            CHECK(std::abs(value - out(0)) < tolerance);
            // Unit weights recover the polynomial derivative reference for the scalar column.
            tensor.weights.setOnes();
            for (int d = 0; d < dom_dim; d++)
            {
                VectorXi derivs = VectorXi::Zero(dom_dim);
                derivs(d) = 1;
                VectorX<T> expected(components);
                mfa::DecodeInfo<T> info(model, derivs);
                decoder.VolPt(param, expected, info, tensor, derivs);
                CHECK(std::abs(gradient(d) - expected(0)) < tolerance);
            }
        }
#endif
    }

    return true;
}

bool test_pointset_decoding()
{
    return check_pointsets<float>() && check_pointsets<double>();
}

bool test_high_level_vector_decoding()
{
    VectorXi p = VectorXi::Ones(2), nctrl = VectorXi::Constant(2, 2), dims(3);
    auto* science = new mfa::MFA_Data<double>(p, nctrl, 2, 3);
    mfa::MFA<double> model(2, 0, new mfa::MFA_Data<double>(p, nctrl, 0, 1), {science});
    auto& tensor = science->tmesh.tensor_prods[0];
    tensor.ctrl_pts.resize(4, 2);
    tensor.ctrl_pts << 0, 0, 1, 2, 2, -1, 3, 1;
    tensor.weights = VectorX<double>::Ones(4);
    dims << 2, 2, 1;
    mfa::PointSet<double> points(2, dims, 10003);
    points.domain.setConstant(-7);
    points.params->param_list.resize(points.npts, 2);
    for (int i = 0; i < points.npts; i++)
    {
        points.params->param_list(i, 0) = (i % 97) / 96.;
        points.params->param_list(i, 1) = (i % 103) / 102.;
    }
    for (int repeat = 0; repeat < 2; repeat++)
    {
        model.DecodeVar(0, points);
        for (int i = 0; i < points.npts; i++)
        {
            double x = points.params->param_list(i, 0), y = points.params->param_list(i, 1);
            CHECK(std::abs(points.domain(i, 2) - (x + 2 * y)) < 1e-14);
            CHECK(std::abs(points.domain(i, 3) - (2 * x - y)) < 1e-14);
        }
        CHECK((points.domain.leftCols(2).array() == -7).all());
        CHECK((points.domain.rightCols(1).array() == -7).all());
    }
    mfa::PointSet<double> other(points.params, dims);
    other.domain.setConstant(-7);
    auto concurrent = std::async(std::launch::async, [&] { model.DecodeVar(0, points); });
    model.DecodeVar(0, other);
    concurrent.get();
    CHECK((points.domain.array() == other.domain.array()).all());
    return true;
}

bool test_invalid_pointset_requests()
{
    VectorXi p = VectorXi::Ones(1), nctrl = VectorXi::Constant(1, 2), dims(2);
    dims << 1, 1;
    mfa::MFA_Data<double> model(p, nctrl, 1, 1);
    auto& tensor = model.tmesh.tensor_prods[0];
    tensor.ctrl_pts = MatrixX<double>::Ones(2, 1);
    tensor.weights = VectorX<double>::Ones(2);
    mfa::Decoder<double> decoder(model, 0);
    mfa::PointSet<double> empty(1, dims, 0), point(1, dims, 1);
    try
    {
        decoder.DecodePointSet(empty, 1, 1);
    }
    catch (...)
    {
        return fail_check("decoder.DecodePointSet(empty, 1, 1) does not throw", __LINE__);
    }
    point.params->param_list = MatrixX<double>::Constant(1, 1, .5);
    CHECK_MFA_ERROR(decoder.DecodePointSet(point, -1, -1));
    CHECK_MFA_ERROR(decoder.DecodePointSet(point, 1, 2));
    CHECK_MFA_ERROR(decoder.DecodePointSet(point, 1, 1, VectorXi::Zero(2)));
    CHECK_MFA_ERROR(decoder.DecodePointSet(point, 1, 1, VectorXi::Constant(1, -1)));
    point.params->param_list.resize(0, 1);
    CHECK_MFA_ERROR(decoder.DecodePointSet(point, 1, 1));
    return true;
}

int main()
{
    try
    {
        if (!test_pointset_decoding() || !test_high_level_vector_decoding() || !test_invalid_pointset_requests())
            return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: decode-pointset-test threw an exception: " << error.what() << '\n';
        return 1;
    }
    catch (...)
    {
        std::cerr << "ERROR: decode-pointset-test threw an unknown exception\n";
        return 1;
    }

    return 0;
}
