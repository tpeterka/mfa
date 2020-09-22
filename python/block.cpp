#include    <pybind11/pybind11.h>
#include    <pybind11/stl.h>
#include    <pybind11/eigen.h>
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

    py::class_<Block<T>, BlockBase<T>>(m, "Block")
        .def(py::init<>())
        .def("generate_analytical_data", &Block<T>::generate_analytical_data)
        .def("print_block", &Block<T>::print_block)
        // TODO: added this because mfa.add did not work
//         .def("init",        &Block<T>::init)
        // TODO: folllowing should not be needed, but ContinousBounds are float and Bounds are double
        .def("init",    [](
                        Block<T>&                  b,
                        const diy::ContinuousBounds&    core,
                        const diy::ContinuousBounds&    domain,
                        int                             dom_dim,
                        int                             pt_dim,
                        T                          ghost_factor)
                {
                    Bounds core_(dom_dim);
                    Bounds domain_(dom_dim);
                    // TODO: Bounds and dynamic point do not allow copying between types
                    // ie, core_ = core, or core_min = core.min
                    // need to copy one coordinate at a time, which is annoying
                    for (auto i = 0; i < dom_dim; i++)
                    {
                        core_.min[i]   = core.min[i];
                        core_.max[i]   = core.max[i];
                        domain_.min[i] = domain.min[i];
                        domain_.max[i] = domain.max[i];
                    }
                    b.init(core_, domain_, dom_dim, pt_dim, ghost_factor);
                })
        .def("fixed_encode_block",  &Block<T>::fixed_encode_block)
        .def("range_error",         &Block<T>::range_error)
        .def("create",              &Block<T>::create)
        .def("destroy",             &Block<T>::destroy)
        .def("save",                &Block<T>::save)
        .def("load",                &Block<T>::load)
        ;

        m.def("save_block", [](const py::object* b, diy::BinaryBuffer* bb)
                {
                    mfa::save<Block<T>, T>(b->cast<Block<T>*>(), *bb);
                });

        m.def("load_block", [](diy::BinaryBuffer* bb) -> py::object
                {
                    Block<T> b;
                    mfa::load<Block<T>, T>(&b, *bb);
                    return py::cast(b);
                });
}

void init_block(py::module& m)
{
    // NB: real_t is defined in examples/block.hpp
    init_block<real_t>(m, "Block_double");
}
