//--------------------------------------------------------------
// Compare unstructured vector point-set decoding with the generic evaluator.
//
// MFA contributors
//--------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include <mfa/mfa.hpp>

void decode_generic_range(
        const mfa::MFA_Data<double>& model,
        mfa::Decoder<double>&       decoder,
        mfa::PointSet<double>&      points,
        size_t                      begin,
        size_t                      end)
{
    const TensorProduct<double>& tensor = model.tmesh.tensor_prods[0];
    VectorXi                     no_derivs;
    mfa::DecodeInfo<double>       decode_info(model, no_derivs);
    VectorX<double>               param(3);
    VectorX<double>               value(2);

    for (size_t i = begin; i < end; i++)
    {
        points.params->pt_params(i, param);
        decoder.VolPt(param, value, decode_info, tensor);
        points.domain.block(i, 3, 1, 2) = value.transpose();
    }
}

void decode_generic(
        const mfa::MFA_Data<double>& model,
        mfa::Decoder<double>&       decoder,
        mfa::PointSet<double>&      points)
{
#ifdef MFA_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, points.npts, 4096),
            [&](const tbb::blocked_range<size_t>& range)
            {
                decode_generic_range(model, decoder, points, range.begin(), range.end());
            });
#else
    decode_generic_range(model, decoder, points, 0, points.npts);
#endif
}

// Usage: decode_pointset_speed [point count, default 1000000] [TBB thread limit, default 18]
// Setup is excluded; each reported time is the median of three warm evaluations.
int main(int argc, char** argv)
{
    // Get command line arguments
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

    // Set up a quadratic model with two science components in a 3D domain
    VectorXi degree = VectorXi::Constant(3, 2);
    VectorXi controls(3);
    VectorXi dims(2);
    controls << 20, 20, 40;
    dims     << 3, 2;

    mfa::MFA_Data<double> model(degree, controls, 3, 4);
    TensorProduct<double>& tensor = model.tmesh.tensor_prods[0];
    tensor.ctrl_pts.resize(controls.prod(), 2);
    tensor.weights = VectorX<double>::Ones(controls.prod());
    for (int i = 0; i < tensor.ctrl_pts.rows(); i++)
    {
        tensor.ctrl_pts(i, 0) = std::sin(i * 0.01);
        tensor.ctrl_pts(i, 1) = std::cos(i * 0.013);
    }
    for (double& knot : model.tmesh.all_knots[2])
        knot *= knot;

    // Generate the same deterministic, unstructured parameters for both methods
    mfa::PointSet<double> points(3, dims, npts);
    points.params->param_list.resize(npts, 3);
    for (int i = 0; i < npts; i++)
    {
        for (int d = 0; d < 3; d++)
            points.params->param_list(i, d) =
                    ((size_t(i) * (d * 62 + 37) + 13 * d) % 99991) / 99991.0;
    }

    // Compute a reference before timing either method
    mfa::Decoder<double> decoder(model, 0);
    decode_generic(model, decoder, points);
    const MatrixX<double> reference = points.domain.rightCols(2);

    // Warm up each method, then report the median of three timed evaluations
    const int   num_repeats = 3;
    const char* names[2]    = {"generic VolPt ranges", "DecodePointSet"};

    for (int method = 0; method < 2; method++)
    {
        double times[num_repeats];
        for (int repeat = -1; repeat < num_repeats; repeat++)
        {
            const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            if (method == 0)
                decode_generic(model, decoder, points);
            else
                decoder.DecodePointSet(points, 3, 4);

            const double elapsed =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (repeat >= 0)
                times[repeat] = elapsed;
        }

        std::sort(times, times + num_repeats);
        const double error = (reference - points.domain.rightCols(2)).cwiseAbs().maxCoeff();
        if (!std::isfinite(error) || error > 1e-12)
            throw std::runtime_error("decoded values differ from the generic reference");

        fmt::print("{},points={},seconds={},max_absolute_difference={}\n", names[method], npts, times[1], error);
    }

    return 0;
}
