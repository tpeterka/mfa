#pragma once

#include <type_traits>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <mpi.h>

// MPI_Comm is a pointer in OpenMPI
template<class Comm, typename std::enable_if<std::is_pointer<Comm>::value,bool>::type = true>
py::capsule to_capsule(Comm comm)
{
    return py::capsule(comm);
}

template<class Comm>
typename std::enable_if<std::is_pointer<Comm>::value, Comm>::type
from_capsule(py::capsule c)
{
    return c;
}

// MPI_Comm is an integer in MPICH
template<class Comm, typename std::enable_if<std::is_integral<Comm>::value,bool>::type = true>
py::capsule to_capsule(Comm comm)
{
    intptr_t comm_ = static_cast<intptr_t>(comm);
    void* comm__ = reinterpret_cast<void*>(comm_);
    return py::capsule(comm__);
}

template<class Comm>
typename std::enable_if<std::is_integral<Comm>::value, Comm>::type
from_capsule(py::capsule c)
{
    void* comm_ = c;
    intptr_t comm__ = reinterpret_cast<intptr_t>(comm_);
    Comm comm = static_cast<Comm>(comm__);
    return comm;
}

