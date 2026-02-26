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

// C++ object representing a python or numpy array
// array can be 1-d (vector) or 2-d (matrix)
// ref: https://github.com/pybind/pybind11/issues/3126
template<typename T>
struct PyArray
{
    py::buffer_info info;
    T *data;
    size_t size;

    PyArray(py::array_t<T> arr) :
        info { arr.request() },
        data { static_cast<T*>(info.ptr) },
        size { static_cast<size_t>(info.shape[0]) * (info.ndim > 1 ? static_cast<size_t>(info.shape[1]) : 1)}
    {}

    PyArray(const PyArray &) = delete;
    PyArray &operator=(const PyArray &) = delete;
    PyArray(PyArray&&) = default;
};

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

                VectorXi mdims_vec(mdims.size());
                for (size_t i = 0; i < mdims.size(); i++)
                    mdims_vec(i) = mdims[i];

                VectorXi ndom_pts_vec(ndom_pts.size());
                for (size_t i = 0; i < ndom_pts.size(); i++)
                    ndom_pts_vec(i) = ndom_pts[i];

                return new PointSet<T>(dom_dim, mdims_vec, npts, ndom_pts_vec);
            }),
            "dom_dim"_a, "mdims"_a, "npts"_a, "ndom_pts"_a = std::vector<int>())
        .def("set_bounds",      [](PointSet<T>& ps, const std::vector<T>& mins, const std::vector<T>& maxs)
            {
                VectorX<T> mins_vec(mins.size());
                for (size_t i = 0; i < mins.size(); i++)
                    mins_vec(i) = mins[i];

                VectorX<T> maxs_vec(maxs.size());
                for (size_t i = 0; i < maxs.size(); i++)
                    maxs_vec(i) = maxs[i];

                ps.set_bounds(mins_vec, maxs_vec);
            }, "mins"_a, "maxs"_a)
        .def("mins", (Eigen::VectorX<T> (PointSet<T>::*)() const) &PointSet<T>::mins)
        .def("mins", (T (PointSet<T>::*)(int) const) &PointSet<T>::mins)
        .def("maxs", (Eigen::VectorX<T> (PointSet<T>::*)() const) &PointSet<T>::maxs)
        .def("maxs", (T (PointSet<T>::*)(int) const) &PointSet<T>::maxs)
        .def("set_domain_params", (void (PointSet<T>::*)()) &PointSet<T>::set_domain_params)
        .def("set_domain_params", (void (PointSet<T>::*)(const Eigen::VectorX<T>&, const Eigen::VectorX<T>&)) &PointSet<T>::set_domain_params)
        .def("set_grid_params", (void (PointSet<T>::*)()) &PointSet<T>::set_grid_params)
        .def("set_grid_params", (void (PointSet<T>::*)(const VectorX<T>&, const VectorX<T>&)) &PointSet<T>::set_grid_params)
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
        .def("set_domain", [](PointSet<T>& ps, MatrixX<T> domain_)
            {
                ps.domain = domain_;
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
        .def("geom",                    &mfa::MFA<T>::geom, py::return_value_policy::reference)// empty, returns mfa::MFA_DATA)
        .def("var",                     &mfa::MFA<T>::var, py::return_value_policy::reference)    
        .def("FixedEncode",             &mfa::MFA<T>::FixedEncode, "input"_a, "regularization"_a, "reg1and2"_a, "weighted"_a) //pointset, regularization, bool, bool (reference output)
        .def("FixedEncodeGeom",         &mfa::MFA<T>::FixedEncodeGeom, "input"_a, "weighted"_a)
        .def("FixedEncodeVar",          &mfa::MFA<T>::FixedEncodeVar, "i"_a, "input"_a, "regularization"_a, "reg1and2"_a, "weighted"_a)
        .def("AdaptiveEncode",          &mfa::MFA<T>::AdaptiveEncode, "input"_a, "err_limit"_a, "weighted"_a, "extents"_a, "max_rounds"_a)
        .def("AdaptiveEncodeGeom",      &mfa::MFA<T>::AdaptiveEncodeGeom, "input"_a, "err_limit"_a, "weighted"_a, "extents"_a, "max_rounds"_a)
        .def("AdaptiveEncodeVar",       &mfa::MFA<T>::AdaptiveEncodeVar, "i"_a, "input"_a, "err_limit"_a, "weighted"_a, "extents"_a, "max_rounds"_a)
        .def("RayEncode",               &mfa::MFA<T>::RayEncode, "i"_a, "input"_a)
        .def("Decode", (void (mfa::MFA<T>::*)(PointSet<T>&, const Eigen::VectorXi&) const) &mfa::MFA<T>::Decode, "output"_a, "derivs"_a=Eigen::VectorXi())
        .def("DecodeGeom", (void (mfa::MFA<T>::*)(PointSet<T>&, const Eigen::VectorXi&) const) &mfa::MFA<T>::DecodeGeom, "output"_a, "derivs"_a=Eigen::VectorXi())
        .def("DecodeVar", (void (mfa::MFA<T>::*)(int, PointSet<T>&, const Eigen::VectorXi&) const) &mfa::MFA<T>::DecodeVar, "i"_a, "output"_a, "derivs"_a=Eigen::VectorXi())
        .def("Integrate1D", [](const mfa::MFA<T>& model,
                                int k,
                                int dim,
                                T u0,
                                T u1,
                                const VectorX<T>& params)
            {
                VectorX<T> output(model.var_dim(k));
                model.Integrate1D(k, dim, u0, u1, params, output);
                return output;
            }, "k"_a, "dim"_a, "u0"_a, "u1"_a, "params"_a)
        .def("DefiniteIntegral", [](const mfa::MFA<T>& model,
                                     int k,
                                     const VectorX<T>& a,
                                     const VectorX<T>& b)
            {
                VectorX<T> output(model.var_dim(k));
                model.DefiniteIntegral(k, output, a, b);
                return output;
            }, "k"_a, "a"_a, "b"_a)
        .def("IntegratePointSet", &mfa::MFA<T>::IntegratePointSet, "mfa_data"_a, "output"_a, "int_dim"_a)
        .def("DecodeAtGrid", &mfa::MFA<T>::DecodeAtGrid, "mfa_data"_a, "par_min"_a, "par_max"_a, "ndom_pts"_a, "result"_a)
        .def("AbsPointSetError", &mfa::MFA<T>::AbsPointSetError, "base"_a, "error"_a)
        .def("AddGeometry", (void (mfa::MFA<T>::*)(int)) &mfa::MFA<T>::AddGeometry)
        .def("AddGeometry", (void (mfa::MFA<T>::*)(const ModelInfo&)) &mfa::MFA<T>::AddGeometry, "mi"_a)
        .def("AddVariable", (void (mfa::MFA<T>::*)(const Eigen::VectorXi&, const Eigen::VectorXi&, int)) &mfa::MFA<T>::AddVariable)
        .def("AddVariable", (void (mfa::MFA<T>::*)(int, const Eigen::VectorXi&, int)) &mfa::MFA<T>::AddVariable)
        .def("AddVariable", (void (mfa::MFA<T>::*)(const ModelInfo&)) &mfa::MFA<T>::AddVariable, "mi"_a)
        .def("setGeomKnots", &mfa::MFA<T>::setGeomKnots, "knots"_a=std::vector<std::vector<T>>())
        .def("setKnots", (void (mfa::MFA<T>::*)(int, const std::vector<std::vector<T>>&)) &mfa::MFA<T>::setKnots, "i"_a, "knots"_a=std::vector<std::vector<T>>())
        .def("setKnots", (void (mfa::MFA<T>::*)(const std::vector<std::vector<T>>&)) &mfa::MFA<T>::setKnots, "knots"_a=std::vector<std::vector<T>>())
        .def("shiftGeom", &mfa::MFA<T>::shiftGeom, "shift"_a)
        .def("shiftVar", &mfa::MFA<T>::shiftVar, "i"_a, "shift"_a)
        .def("printDetails", (void (mfa::MFA<T>::*)() const) &mfa::MFA<T>::printDetails)
        .def("printDetails", (void (mfa::MFA<T>::*)(int) const) &mfa::MFA<T>::printDetails, "verbose"_a)
        .def("dumpCollocationMatrixEncode", &mfa::MFA<T>::dumpCollocationMatrixEncode, "i"_a, "ps"_a)
        .def("dumpCollocationMatrixDecode", &mfa::MFA<T>::dumpCollocationMatrixDecode, "i"_a, "ps"_a)
    ;

    py::class_<Tmesh<T>>(m, "Tmesh")
        .def(py::init<int, const Eigen::VectorXi, int, int, size_t>())
        .def_readwrite("tensor_prods",  &Tmesh<T>::tensor_prods)
    ;

    py::class_<mfa::MFA_Data<T>>(m, "MFA_Data")
        .def(py::init<const Eigen::VectorXi, Eigen::VectorXi, int, int>())
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
        .def("decode_point",                        &Block<T>::decode_point)
        .def("range_error",                         &Block<T>::range_error)
        .def_static("save",                         &Block<T>::save)
        .def_static("load",                         &Block<T>::load)
        .def("input_data",              [](
                                        const py::object*                   py_b,
                                        const diy::Master::ProxyWithLink&   cp,
                                        const py::array_t<T>&               arr,    // input data
                                        MFAInfo&                            mfa_info,
                                        DomainArgs&                         d_args)
        {
            PyArray<T> pyarray(arr);

            // debug
//             fmt::print(stderr, "PyArray size {}\n", pyarray.size);
//             for (auto i = 0; i < pyarray.size; i++)
//                 fmt::print(stderr, "data[{}] = {}\n", i, pyarray.data[i]);

            Block<T>* b = py_b->cast<Block<T>*>();
            if (!b) throw std::runtime_error("input_data: failed to cast block");
            b->input_1d_data(cp, pyarray.data, mfa_info, d_args);
        })
        ;

    m.def("add_block", [](
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
            // std::cerr << core.min << std::endl;
            std::cerr << "received gid " << gid << std::endl;
            std::cerr << ">> " << core.min.dimension() << std::endl;
            Block<T>*       b   = new Block<T>;
            RCLink*         l   = new RCLink(link);
            master.add(gid, new py::object(py::cast(b)), l);
            b->init_block(gid, core, domain, dom_dim, pt_dim);
        });

    m.def("save_block", [](const py::object* b, diy::BinaryBuffer* bb)
        {
            if (!b) throw std::runtime_error("save_block: null block object");
            mfa::save<Block<T>, T>(b->cast<Block<T>*>(), *bb);
        });

    m.def("load_block", [](diy::BinaryBuffer* bb)
        {
            std::unique_ptr<Block<T>> b { new Block<T> };
            mfa::load<Block<T>, T>(b.get(), *bb);
            return b;
        });
}

void init_block(py::module& m)
{
    // NB: real_t is defined in examples/block.hpp
    init_block<real_t>(m, "Block_double");
}
