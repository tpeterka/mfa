#include    <pybind11/pybind11.h>
namespace py = pybind11;

#include    <../block.hpp>

template <typename T>
void init_block(py::module& m)
{
    py::class_<ModelInfo> model_info_class(m, "ModelInfo");

    py::class_<Model<T>> model_class(m, "Model");

    py::class_<BlockBase<T>> block_base_class(m, "BlockBase");

    py::class_<ModelInfo> domain_args_class(m, "DomainArgs", model_info_classs);

    py::class_<BlockBase<T>> block_class(m, "Block", block_base_class);

    blah blah blah
}

