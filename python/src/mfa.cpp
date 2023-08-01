#include <pybind11/pybind11.h>
namespace py = pybind11;

#if defined(MFA_MPI4PY)
#include "mpi-comm.h"
#endif

#include "mpi-capsule.h"

void init_block(py::module& m);

PYBIND11_MODULE(_mfa, m)
{
    init_block(m);
}

