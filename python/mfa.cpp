#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_block(py::module& m);

PYBIND11_MODULE(mfa, m)
{
    init_block(m);
}

