#include    <pybind11/pybind11.h>
#include    <pybind11/stl.h>
#include    <pybind11/eigen.h>
#include    <pybind11/functional.h>
namespace py = pybind11;

#include    <mfa/mfa.hpp>
#include    "../../examples/block.hpp"

#if defined(MFA_MPI4PY)
#include "mpi-comm.h"
#endif

#include "mpi-capsule.h"

namespace
{
    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1>
    to_eigen_vector(const std::vector<T>& values)
    {
        Eigen::Matrix<T, Eigen::Dynamic, 1> out(static_cast<Eigen::Index>(values.size()));
        for (size_t i = 0; i < values.size(); ++i)
            out(static_cast<Eigen::Index>(i)) = values[i];
        return out;
    }

    inline Eigen::VectorXi
    to_eigen_vectori(const std::vector<int>& values)
    {
        Eigen::VectorXi out(static_cast<Eigen::Index>(values.size()));
        for (size_t i = 0; i < values.size(); ++i)
            out(static_cast<Eigen::Index>(i)) = values[i];
        return out;
    }

    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    to_eigen_matrix2d(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr)
    {
        py::buffer_info info = arr.request();
        if (info.ndim != 2)
            throw py::value_error("Expected a 2D array");

        const auto rows = static_cast<Eigen::Index>(info.shape[0]);
        const auto cols = static_cast<Eigen::Index>(info.shape[1]);
        const T* data = static_cast<const T*>(info.ptr);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> out(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i)
        {
            for (Eigen::Index j = 0; j < cols; ++j)
                out(i, j) = data[i * cols + j];
        }
        return out;
    }
}

template <typename T>
void init_block(py::module& m, std::string name)
{
    using Bounds    = diy::Bounds<T>;
    using RCLink    = diy::RegularLink<diy::Bounds<T>>;
    using Master    = diy::Master;

    using namespace pybind11::literals;
    using namespace mfa;

    py::class_<PointSet<T>>(m, "PointSet")
        .def(py::init([](int dom_dim,
                         const std::vector<int>& mdims,
                         int npts,
                         const std::vector<int>& ndom_pts)
            {
                if (dom_dim < 0 || npts < 0)
                    throw py::value_error("dom_dim and npts must be non-negative");

                VectorXi mdims_vec = to_eigen_vectori(mdims);
                VectorXi ndom_pts_vec = to_eigen_vectori(ndom_pts);

                return new PointSet<T>(dom_dim, mdims_vec, npts, ndom_pts_vec);
            }),
            "dom_dim"_a, "mdims"_a, "npts"_a, "ndom_pts"_a = std::vector<int>())
        .def("set_bounds",      [](PointSet<T>& ps, const std::vector<T>& mins, const std::vector<T>& maxs)
            {
                VectorX<T> mins_vec = to_eigen_vector(mins);
                VectorX<T> maxs_vec = to_eigen_vector(maxs);
                ps.set_bounds(mins_vec, maxs_vec);
            }, "mins"_a, "maxs"_a)
        .def("mins", (Eigen::VectorX<T> (PointSet<T>::*)() const) &PointSet<T>::mins)
        .def("mins", (T (PointSet<T>::*)(int) const) &PointSet<T>::mins)
        .def("maxs", (Eigen::VectorX<T> (PointSet<T>::*)() const) &PointSet<T>::maxs)
        .def("maxs", (T (PointSet<T>::*)(int) const) &PointSet<T>::maxs)
        .def("set_domain_params", (void (PointSet<T>::*)()) &PointSet<T>::set_domain_params)
        .def("set_domain_params", [](PointSet<T>& ps, const std::vector<T>& domain_mins, const std::vector<T>& domain_maxs)
            {
                VectorX<T> domain_mins_vec = to_eigen_vector(domain_mins);
                VectorX<T> domain_maxs_vec = to_eigen_vector(domain_maxs);
                ps.set_domain_params(domain_mins_vec, domain_maxs_vec);
            }, "domain_mins"_a, "domain_maxs"_a)
        .def("set_grid_params", (void (PointSet<T>::*)()) &PointSet<T>::set_grid_params)
        .def("set_grid_params", [](PointSet<T>& ps, const std::vector<T>& param_mins, const std::vector<T>& param_maxs)
            {
                VectorX<T> param_mins_vec = to_eigen_vector(param_mins);
                VectorX<T> param_maxs_vec = to_eigen_vector(param_maxs);
                ps.set_grid_params(param_mins_vec, param_maxs_vec);
            }, "param_mins"_a, "param_maxs"_a)
        .def("set_curve_params", &PointSet<T>::set_curve_params)
        .def("is_structured", &PointSet<T>::is_structured)
        .def("validate", &PointSet<T>::validate)
        .def("is_same_layout", &PointSet<T>::is_same_layout, "ps"_a, "verbose"_a = 1)
        .def("nvars", &PointSet<T>::nvars)
        .def("geom_dim", &PointSet<T>::geom_dim)
        .def("var_dim", &PointSet<T>::var_dim, "k"_a)
        .def("var_min", &PointSet<T>::var_min, "k"_a)
        .def("var_max", &PointSet<T>::var_max, "k"_a)
        .def("ndom_pts", (VectorXi (PointSet<T>::*)() const) &PointSet<T>::ndom_pts)
        .def("ndom_pts", (int (PointSet<T>::*)(int) const) &PointSet<T>::ndom_pts, "i"_a)
        .def("model_dims", &PointSet<T>::model_dims)
        // We deliberately do not expose the following functions since they cannot be used in a pythonic way
        // without copies. However, these functions are typically called in tight loops where copies would be
        // too expensive. Therefore, we explicitly disallow this behavior. To get similar functionality in
        // python, users should access the domain matrix directly and use numpy slicing to get views into the data.
        // Deliberately unbound functions:
        //      pt_coords
        //      geom_coords
        //      var_coords
        //      pt_params
        .def("abs_diff", &PointSet<T>::abs_diff, "other"_a, "diff"_a)
        .def_readwrite("dom_dim",   &PointSet<T>::dom_dim)
        .def_readwrite("pt_dim",    &PointSet<T>::pt_dim)
        .def_readwrite("npts",      &PointSet<T>::npts)
        .def_readwrite("domain",    &PointSet<T>::domain)
        .def("set_domain", [](PointSet<T>& ps, const py::array_t<T, py::array::c_style | py::array::forcecast>& domain)
            {
                py::buffer_info info = domain.request();
                if (info.ndim != 2)
                {
                    throw py::value_error("Expected a 2D array");
                }
                if (info.shape[0] != static_cast<py::ssize_t>(ps.npts) ||
                    info.shape[1] != static_cast<py::ssize_t>(ps.pt_dim))
                {
                    throw py::value_error("set_domain: domain shape must match PointSet dimensions (npts x pt_dim)");
                }
                ps.domain = to_eigen_matrix2d<T>(domain);
            }
        )
        .def("set_from_params", [](PointSet<T>& ps, const PointSet<T>& other)
            {
                ps.dom_dim = other.dom_dim;
                ps.pt_dim = other.pt_dim;
                ps.npts = other.npts;
                ps.mdims = other.mdims;
                ps.params = other.params;

                ps.domain.resize(ps.npts, ps.pt_dim);

                // Fill dim_mins/maxs
                ps.dim_mins.resize(ps.nvars());
                ps.dim_maxs.resize(ps.nvars());

                if (ps.nvars() > 0)
                {
                    ps.dim_mins[0] = ps.geom_dim();
                    ps.dim_maxs[0] = ps.dim_mins[0] + ps.var_dim(0) - 1;
                }
                for (int k = 1; k < ps.nvars(); k++)
                {
                    ps.dim_mins[k] = ps.dim_maxs[k-1] + 1;
                    ps.dim_maxs[k] = ps.dim_mins[k] + ps.var_dim(k) - 1;
                }

                if (other.params->structured)
                    ps.add_grid(other.params->ndom_pts);
            }
        )
    ;

    py::class_<ModelInfo>(m,"ModelInfo")
        .def(py::init<int, int, int, int>())
        .def(py::init<int, int, std::vector<int>, std::vector<int>>()) 
        .def_readwrite("dom_dim",       &ModelInfo::dom_dim)
        .def_readwrite("var_dim",       &ModelInfo::var_dim)
        .def_readwrite("p",             &ModelInfo::p)
        .def_readwrite("nctrl_pts",     &ModelInfo::nctrl_pts)
    ;

    py::class_<MFAInfo> (m, "MFAInfo")
        .def(py::init<int, int, ModelInfo, ModelInfo>())        
        .def(py::init<int, int>()) 
        .def("nvars", &MFAInfo::nvars)
        .def("geom_dim", &MFAInfo::geom_dim)
        .def("pt_dim", &MFAInfo::pt_dim)
        .def("var_dim", &MFAInfo::var_dim, "k"_a)
        .def("model_dims", &MFAInfo::model_dims)
        .def("splitStrongScaling", &MFAInfo::splitStrongScaling, "divs"_a)
        .def("reset", &MFAInfo::reset)
        .def_readwrite("dom_dim",           &MFAInfo::dom_dim)
        .def_readwrite("verbose",           &MFAInfo::verbose)
        .def_readwrite("geom_model_info",   &MFAInfo::geom_model_info)
        .def_readwrite("var_model_infos",   &MFAInfo::var_model_infos)
        .def_readwrite("weighted",          &MFAInfo::weighted)
        .def_readwrite("local",             &MFAInfo::local)
        .def_readwrite("reg1and2",          &MFAInfo::reg1and2)
        .def_readwrite("regularization",    &MFAInfo::regularization)
        .def("addGeomInfo",                 &MFAInfo::addGeomInfo, "gmi"_a)
        .def("addVarInfo",                  (void (mfa::MFAInfo::*)(mfa::ModelInfo)) &MFAInfo::addVarInfo, "vmi"_a)
        .def("addVarInfo",                  (void (mfa::MFAInfo::*)(std::vector<mfa::ModelInfo>)) &MFAInfo::addVarInfo, "vmis"_a)
    ;

    py::class_<mfa::MFA<T>>(m, "MFA")
        .def(py::init<int, int>())
        .def(py::init<const MFAInfo>())
        .def("nvars",                   &mfa::MFA<T>::nvars)
        .def("geom_dim",                &mfa::MFA<T>::geom_dim)
        .def("var_dim",                 &mfa::MFA<T>::var_dim, "i"_a)
        .def("model_dims",              &mfa::MFA<T>::model_dims)
        .def_readwrite("dom_dim",       &mfa::MFA<T>::dom_dim)
        .def_readwrite("pt_dim",        &mfa::MFA<T>::pt_dim)
        .def("geom",                    &mfa::MFA<T>::geom, py::return_value_policy::reference_internal)
        .def("var",                     &mfa::MFA<T>::var, "i"_a, py::return_value_policy::reference_internal)
        .def("FixedEncode",             &mfa::MFA<T>::FixedEncode, "input"_a, "regularization"_a, "reg1and2"_a, "weighted"_a) //pointset, regularization, bool, bool (reference output)
        .def("FixedEncodeGeom",         &mfa::MFA<T>::FixedEncodeGeom, "input"_a, "weighted"_a)
        .def("FixedEncodeVar",          &mfa::MFA<T>::FixedEncodeVar, "i"_a, "input"_a, "regularization"_a, "reg1and2"_a, "weighted"_a)
        .def("AdaptiveEncode",          &mfa::MFA<T>::AdaptiveEncode, "input"_a, "err_limit"_a, "weighted"_a, "extents"_a, "max_rounds"_a)
        .def("AdaptiveEncodeGeom",      &mfa::MFA<T>::AdaptiveEncodeGeom, "input"_a, "err_limit"_a, "weighted"_a, "extents"_a, "max_rounds"_a)
        .def("AdaptiveEncodeVar",       &mfa::MFA<T>::AdaptiveEncodeVar, "i"_a, "input"_a, "err_limit"_a, "weighted"_a, "extents"_a, "max_rounds"_a)
        .def("RayEncode",               &mfa::MFA<T>::RayEncode, "i"_a, "input"_a)
        .def("Decode", [](const mfa::MFA<T>& model, PointSet<T>& output, const std::vector<int>& derivs)
            {
                model.Decode(output, to_eigen_vectori(derivs));
            }, "output"_a, "derivs"_a = std::vector<int>())
        .def("DecodeGeom", [](const mfa::MFA<T>& model, PointSet<T>& output, const std::vector<int>& derivs)
            {
                model.DecodeGeom(output, to_eigen_vectori(derivs));
            }, "output"_a, "derivs"_a = std::vector<int>())
        .def("DecodeVar", [](const mfa::MFA<T>& model, int i, PointSet<T>& output, const std::vector<int>& derivs)
            {
                model.DecodeVar(i, output, to_eigen_vectori(derivs));
            }, "i"_a, "output"_a, "derivs"_a = std::vector<int>())
        .def("Integrate1D", [](const mfa::MFA<T>& model,
                                int k,
                                int dim,
                                T u0,
                                T u1,
                                const std::vector<T>& params)
            {
                VectorX<T> output(model.var_dim(k));
                model.Integrate1D(k, dim, u0, u1, to_eigen_vector(params), output);
                return output;
            }, "k"_a, "dim"_a, "u0"_a, "u1"_a, "params"_a)
        .def("DefiniteIntegral", [](const mfa::MFA<T>& model,
                                     int k,
                                     const std::vector<T>& a,
                                     const std::vector<T>& b)
            {
                VectorX<T> output(model.var_dim(k));
                model.DefiniteIntegral(k, output, to_eigen_vector(a), to_eigen_vector(b));
                return output;
            }, "k"_a, "a"_a, "b"_a)
        .def("IntegratePointSet", &mfa::MFA<T>::IntegratePointSet, "mfa_data"_a, "output"_a, "int_dim"_a)
        .def("DecodeAtGrid", [](mfa::MFA<T>& model,
                                 const mfa::MFA_Data<T>& mfa_data,
                                 const std::vector<T>& par_min,
                                 const std::vector<T>& par_max,
                                 const std::vector<int>& ndom_pts)
            {
                VectorX<T> par_min_vec = to_eigen_vector(par_min);
                VectorX<T> par_max_vec = to_eigen_vector(par_max);
                VectorXi ndom_pts_vec = to_eigen_vectori(ndom_pts);

                MatrixX<T> result(static_cast<Eigen::Index>(ndom_pts_vec.prod()),
                                  static_cast<Eigen::Index>(mfa_data.dim()));
                model.DecodeAtGrid(mfa_data, par_min_vec, par_max_vec, ndom_pts_vec, result);
                return result;
            }, "mfa_data"_a, "par_min"_a, "par_max"_a, "ndom_pts"_a)
        .def("AbsPointSetError", &mfa::MFA<T>::AbsPointSetError, "base"_a, "error"_a)
        .def("AddGeometry", (void (mfa::MFA<T>::*)(int)) &mfa::MFA<T>::AddGeometry)
        .def("AddGeometry", (void (mfa::MFA<T>::*)(const ModelInfo&)) &mfa::MFA<T>::AddGeometry, "mi"_a)
        .def("AddVariable", [](mfa::MFA<T>& model, const std::vector<int>& degree, const std::vector<int>& nctrl_pts, int dim)
            {
                model.AddVariable(
                    to_eigen_vectori(degree), 
                    to_eigen_vectori(nctrl_pts), 
                    dim
                );
            }, "degree"_a, "nctrl_pts"_a, "dim"_a)
        .def("AddVariable", [](mfa::MFA<T>& model, int degree, const std::vector<int>& nctrl_pts, int dim)
            {
                model.AddVariable(
                    degree, 
                    to_eigen_vectori(nctrl_pts), 
                    dim
                );
            }, "degree"_a, "nctrl_pts"_a, "dim"_a)
        .def("AddVariable", (void (mfa::MFA<T>::*)(const ModelInfo&)) &mfa::MFA<T>::AddVariable, "mi"_a)
        .def("setGeomKnots", &mfa::MFA<T>::setGeomKnots, "knots"_a=std::vector<std::vector<T>>())
        .def("setKnots", (void (mfa::MFA<T>::*)(int, const std::vector<std::vector<T>>&)) &mfa::MFA<T>::setKnots, "i"_a, "knots"_a=std::vector<std::vector<T>>())
        .def("setKnots", (void (mfa::MFA<T>::*)(const std::vector<std::vector<T>>&)) &mfa::MFA<T>::setKnots, "knots"_a=std::vector<std::vector<T>>())
        .def("shiftGeom", [](mfa::MFA<T>& model, const std::vector<T>& shift)
            {
                model.shiftGeom(to_eigen_vector(shift));
            }, "shift"_a)
        .def("shiftVar", [](mfa::MFA<T>& model, int i, const std::vector<T>& shift)
            {
                model.shiftVar(i, to_eigen_vector(shift));
            }, "i"_a, "shift"_a)
        .def("printDetails", (void (mfa::MFA<T>::*)() const) &mfa::MFA<T>::printDetails)
        .def("printDetails", (void (mfa::MFA<T>::*)(int) const) &mfa::MFA<T>::printDetails, "verbose"_a)
        .def("dumpCollocationMatrixEncode", &mfa::MFA<T>::dumpCollocationMatrixEncode, "i"_a, "ps"_a)
        .def("dumpCollocationMatrixDecode", &mfa::MFA<T>::dumpCollocationMatrixDecode, "i"_a, "ps"_a)
    ;

    py::class_<Tmesh<T>>(m, "Tmesh")
        .def(py::init([](int dom_dim, const std::vector<int>& p, int min_dim, int max_dim, size_t ntensor_prods)
            {
                return Tmesh<T>(
                    dom_dim, 
                    to_eigen_vectori(p), 
                    min_dim, 
                    max_dim, 
                    ntensor_prods
                );
            }), "dom_dim"_a, "p"_a, "min_dim"_a, "max_dim"_a, "ntensor_prods"_a = 0)
        .def_readwrite("tensor_prods",  &Tmesh<T>::tensor_prods)
    ;

    py::class_<mfa::MFA_Data<T>>(m, "MFA_Data")
        .def(py::init([](const std::vector<int>& p, const std::vector<int>& nctrl_pts, int min_dim, int max_dim)
            {
                return mfa::MFA_Data<T>(
                    to_eigen_vectori(p), 
                    to_eigen_vectori(nctrl_pts), 
                    min_dim, 
                    max_dim
                );
            }), "p"_a, "nctrl_pts"_a, "min_dim"_a, "max_dim"_a)
        .def_readwrite("dom_dim",       &MFA_Data<T>::dom_dim)
        .def_readwrite("min_dim",       &MFA_Data<T>::min_dim)
        .def_readwrite("max_dim",       &MFA_Data<T>::max_dim)
        .def_readwrite("p",             &MFA_Data<T>::p)
        .def_readwrite("tmesh",         &MFA_Data<T>::tmesh)
    ;

    py::class_<TensorProduct<T>>(m, "TensorProduct")
        .def(py::init<int>())
        .def_readwrite("nctrl_pts",  &TensorProduct<T>::nctrl_pts)
        .def_readwrite("ctrl_pts",   &TensorProduct<T>::ctrl_pts)
    ;

    py::class_<BlockBase<T>>(m, "BlockBase")
        .def(py::init<>())
        .def("init_block",  &BlockBase<T>::init_block)
        .def("setup_MFA",   &BlockBase<T>::setup_MFA)
    ;

    py::class_<DomainArgs>(m, "DomainArgs")
        .def(py::init<int, std::vector<int>>())
        .def("updateModelDims",    &DomainArgs::updateModelDims, "mdims"_a)
        .def_readwrite("starts",        &DomainArgs::starts)
        .def_readwrite("ndom_pts",      &DomainArgs::ndom_pts)
        .def_readwrite("full_dom_pts",  &DomainArgs::full_dom_pts)
        .def_readwrite("tot_ndom_pts",  &DomainArgs::tot_ndom_pts)
        .def_readwrite("min",           &DomainArgs::min)
        .def_readwrite("max",           &DomainArgs::max)
        .def_readwrite("s",             &DomainArgs::s)
        .def_readwrite("r",             &DomainArgs::r)
        .def_readwrite("f",             &DomainArgs::f)
        .def_readwrite("t",             &DomainArgs::t)
        .def_readwrite("n",             &DomainArgs::n)
        .def_readwrite("infile",        &DomainArgs::infile)
        .def_readwrite("infile2",       &DomainArgs::infile2)
        .def_readwrite("multiblock",    &DomainArgs::multiblock)
        .def_readwrite("structured",    &DomainArgs::structured)
        .def_readwrite("rand_seed",     &DomainArgs::rand_seed)
        .def_readwrite("model_dims",    &DomainArgs::model_dims)
    ;

    using namespace py::literals;

    py::class_<Block<T>, BlockBase<T>>(m, "Block")
        .def(py::init<>())
        .def("generate_analytical_data",&Block<T>::generate_analytical_data)
        .def("print_block",             &Block<T>::print_block)
        .def_static("add",                     [](
                                        int                 gid,
                                        const Bounds&       core,
                                        const Bounds&       bounds,
                                        const Bounds&       domain,
                                        const RCLink&       link,
                                        Master&             master,
                                        int                 dom_dim,
                                        int                 pt_dim,
                                        T                   ghost_factor)
            {
                Block<T>*       b   = new Block<T>;
                RCLink*         l   = new RCLink(link);
                master.add(gid, new py::object(py::cast(b)), l);
                b->init_block(gid, core, domain, dom_dim, pt_dim);
            }, "gid"_a, "core"_a, "bounds"_a, "domain"_a, "link"_a, "master"_a, "dom_dim"_a, "pt_dim"_a,
            "ghost_factor"_a = 0.0)
        .def("fixed_encode_block",                  &Block<T>::fixed_encode_block)
        .def("adaptive_encode_block",               &Block<T>::adaptive_encode_block)
        .def("decode_point", [](Block<T>& b,
                                 const diy::Master::ProxyWithLink& cp,
                                 const std::vector<T>& param)
            {
                if (param.size() != static_cast<size_t>(b.dom_dim))
                    throw py::value_error("decode_point: param length does not match dom_dim");

                VectorX<T> cpt(b.pt_dim);
                b.decode_point(cp, to_eigen_vector(param), cpt);
                return cpt;
            }, "cp"_a, "param"_a)
        .def("range_error",                         &Block<T>::range_error)
        .def_static("save", [](const py::object* b, diy::BinaryBuffer* bb)
            {
                if (!b) throw std::runtime_error("Block.save: null block object");
                mfa::save<Block<T>, T>(b->cast<Block<T>*>(), *bb);
            })
        .def_static("load", [](diy::BinaryBuffer* bb)
            {
                std::unique_ptr<Block<T>> b { new Block<T> };
                mfa::load<Block<T>, T>(b.get(), *bb);
                return b;
            })
        ;
}

void init_block(py::module& m)
{
    // NB: real_t is defined in examples/block.hpp
    init_block<real_t>(m, "Block_double");
}
