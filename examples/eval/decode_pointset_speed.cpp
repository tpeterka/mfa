//--------------------------------------------------------------
// Compare unstructured vector point-set decoding with the generic evaluator.
// MFA contributors
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

// Usage: decode_pointset_speed [point count, default 1000000] [TBB thread limit, default 18]
// Setup is excluded; each reported time is the median of three warm evaluations.
int main(int argc, char** argv)
{
    const int npts = argc > 1 ? std::stoi(argv[1]) : 1000000;
    if (npts < 1)
        throw std::invalid_argument("point count must be positive");
#ifdef MFA_TBB
    const int threads = argc > 2 ? std::stoi(argv[2]) : 18;
    if (threads < 1)
        throw std::invalid_argument("thread limit must be positive");
    tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
#endif
#if defined(MFA_TMESH) || defined(MFA_OVERLAYS)
    throw std::invalid_argument("this benchmark requires T-mesh and overlays disabled");
#endif
    VectorXi degree = VectorXi::Constant(3, 2), controls(3), dims(2);
    controls << 20, 20, 40;
    dims << 3, 2;
    mfa::MFA_Data<double> model(degree, controls, 3, 4);
    auto& tensor = model.tmesh.tensor_prods[0];
    tensor.ctrl_pts.resize(controls.prod(), 2);
    tensor.weights = VectorX<double>::Ones(controls.prod());
    for (int i = 0; i < tensor.ctrl_pts.rows(); i++)
    {
        tensor.ctrl_pts(i, 0) = std::sin(i * .01);
        tensor.ctrl_pts(i, 1) = std::cos(i * .013);
    }
    for (auto& knot : model.tmesh.all_knots[2])
        knot *= knot;
    mfa::PointSet<double> points(3, dims, npts);
    points.params->param_list.resize(npts, 3);
    for (int i = 0; i < npts; i++)
        for (int d = 0; d < 3; d++)
            points.params->param_list(i, d) = ((size_t(i) * (d * 62 + 37) + 13 * d) % 99991) / 99991.;

    mfa::Decoder<double> decoder(model, 0);
    auto generic_range = [&](size_t begin, size_t end)
    {
        mfa::DecodeInfo<double> scratch(model, VectorXi());
        VectorX<double> param(3), value(2);
        for (size_t i = begin; i < end; i++)
        {
            points.params->pt_params(i, param);
            decoder.VolPt(param, value, scratch, tensor);
            points.domain.block(i, 3, 1, 2) = value.transpose();
        }
    };
    auto generic = [&]
    {
#ifdef MFA_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, npts, 4096), [&](const tbb::blocked_range<size_t>& r)
        {
            generic_range(r.begin(), r.end());
        });
#else
        generic_range(0, npts);
#endif
    };
    generic();
    const MatrixX<double> reference = points.domain.rightCols(2);
    auto measure = [&](const char* name, auto evaluate)
    {
        evaluate();
        vector<double> times;
        for (int repeat = 0; repeat < 3; repeat++)
        {
            auto start = std::chrono::steady_clock::now();
            evaluate();
            times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(times.begin(), times.end());
        const double error = (reference - points.domain.rightCols(2)).cwiseAbs().maxCoeff();
        if (!std::isfinite(error) || error > 1e-12)
            throw std::runtime_error("decoded values differ from the generic reference");
        fmt::print("{},points={},seconds={},max_absolute_difference={}\n", name, npts, times[1], error);
    };
    measure("generic VolPt ranges", generic);
    measure("DecodePointSet", [&] { decoder.DecodePointSet(points, 3, 4); });
}
