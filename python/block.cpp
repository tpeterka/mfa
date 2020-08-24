#include    <pybind11/pybind11.h>
#include    <pybind11/stl.h>
namespace py = pybind11;

#include    <../examples/block.hpp>

void init_block(py::module& m)
{
    using Bounds    = diy::Bounds<real_t>;
    using RCLink    = diy::RegularLink<diy::Bounds<real_t>>;
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

    py::class_<Model<real_t>> (m, "Model")
        .def(py::init<>())
    ;

    py::class_<BlockBase<real_t>> (m, "BlockBase")
        .def(py::init<>())
        // TODO: not sure if these will be needed
//         .def_readwrite("bounds_mins",   &BlockBase<real_t>::bounds_mins)
//         .def_readwrite("bounds_maxs",   &BlockBase<real_t>::bounds_maxs)
//         .def_readwrite("core_mins",     &BlockBase<real_t>::core_mins)
//         .def_readwrite("core_maxs",     &BlockBase<real_t>::core_maxs)
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
        //  TODO: how to bind infile, does not compile
//         .def_readwrite("infile",        &DomainArgs::infile)
        .def_readwrite("multiblock",    &DomainArgs::multiblock)
    ;

    py::class_<Block<real_t>, BlockBase<real_t>>(m, "Block")
        .def(py::init<>())
        .def("generate_analytical_data", &Block<real_t>::generate_analytical_data)
//         .def("generate_analytical_data", [](Block<real_t>& b, const diy::Master::ProxyWithLink& cp, string& fun, DomainArgs& args)
//                 {
//                     b.generate_analytical_data(cp, fun, args);
//                 })
        .def("print_block", &Block<real_t>::print_block)
        // TODO: added this because mfa.add did not work
//         .def("init",        &Block<real_t>::init)
        // TODO: folllowing should not be needed, but ContinousBounds are float and Bounds are double
        .def("init",    [](
                        Block<real_t>&                  b,
                        const diy::ContinuousBounds&    core,
                        const diy::ContinuousBounds&    domain,
                        int                             dom_dim,
                        int                             pt_dim,
                        real_t                          ghost_factor)
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
        .def("fixed_encode_block",  &Block<real_t>::fixed_encode_block)
        .def("range_error",         &Block<real_t>::range_error)
    ;

    // TODO: crashes at runtime
//     m.def("add", [](int             gid,
//                     const Bounds&   core,
//                     const Bounds&   bounds,
//                     const Bounds&   domain,
//                     const RCLink&   link,
//                     Master&         master,
//                     int             dom_dim,
//                     int             pt_dim,
//                     real_t          ghost_factor)
//             {
//                 mfa::add<Block<real_t>, real_t>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);
//             });

}

