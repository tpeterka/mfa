#include    <pybind11/pybind11.h>
namespace py = pybind11;

#include    <mfa/block_base.hpp>

template <typename T>
void init_block_base(py::module& m)
{
    py::class_<ModelInfo> model_info_class(m, "ModelInfo");

    py::class_<Model<T>> model_class(m, "Model");

    py::class_<BlockBase<T>> block_base_class(m, "BlockBase");
}
