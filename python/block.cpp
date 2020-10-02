#include    <pybind11/pybind11.h>
#include    <pybind11/stl.h>
#include    <pybind11/eigen.h>
#include    <pybind11/functional.h>
namespace py = pybind11;

#include    <../examples/block.hpp>

template <typename T>
void init_block(py::module& m, std::string name)
{
    using Bounds    = diy::Bounds<T>;
    using RCLink    = diy::RegularLink<diy::Bounds<T>>;
    using Master    = diy::Master;

    py::class_<ModelInfo> (m, "ModelInfo")
        .def(py::init<int, int>())
        .def_readwrite("dom_dim",           &ModelInfo::dom_dim)
        .def_readwrite("pt_dim",            &ModelInfo::pt_dim)
        .def_readwrite("geom_p",            &ModelInfo::geom_p)
        .def_readwrite("vars_p",            &ModelInfo::vars_p)
        .def_readwrite("ndom_pts",          &ModelInfo::ndom_pts)
        .def_readwrite("geom_nctrl_pts",    &ModelInfo::geom_nctrl_pts)
        .def_readwrite("vars_nctrl_pts",    &ModelInfo::vars_nctrl_pts)
        .def_readwrite("weighted",          &ModelInfo::weighted)
        .def_readwrite("verbose",           &ModelInfo::verbose)
    ;

    py::class_<Model<T>> (m, "Model")
        .def(py::init<>())
    ;

    py::class_<BlockBase<T>> (m, "BlockBase")
        .def(py::init<>())
    ;

    py::class_<DomainArgs, ModelInfo>(m, "DomainArgs")
        .def(py::init<int, int>())
        .def_readwrite("starts",        &DomainArgs::starts)
        .def_readwrite("full_dom_pts",  &DomainArgs::full_dom_pts)
        .def_readwrite("min",           &DomainArgs::min)
        .def_readwrite("max",           &DomainArgs::max)
        .def_readwrite("s",             &DomainArgs::s)
        .def_readwrite("r",             &DomainArgs::r)
        .def_readwrite("f",             &DomainArgs::f)
        .def_readwrite("t",             &DomainArgs::t)
        .def_readwrite("n",             &DomainArgs::n)
        .def_readwrite("infile",        &DomainArgs::infile)
        .def_readwrite("multiblock",    &DomainArgs::multiblock)
    ;

    using namespace py::literals;

    py::class_<Block<T>, BlockBase<T>>(m, "Block")
        .def(py::init<>())
        .def("generate_analytical_data",&Block<T>::generate_analytical_data)
        .def("print_block",             &Block<T>::print_block)
        .def("add",                     [](
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
                b->init_block(core, domain, dom_dim, pt_dim);
            }, "core"_a, "bounds"_a, "domain"_a, "link"_a, "master"_a, "dom_dim"_a, "pt_dim"_a,
            "ghost_factor"_a = 0.0)
        .def("fixed_encode_block",      &Block<T>::fixed_encode_block)
        .def("adaptive_encode_block",   &Block<T>::adaptive_encode_block)
        .def("decode_point",            &Block<T>::decode_point)
        .def("range_error",             &Block<T>::range_error)
        .def("save",                    &Block<T>::save)
        .def("load",                    &Block<T>::load)
        ;

    m.def("save_block", [](const py::object* b, diy::BinaryBuffer* bb)
        {
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
