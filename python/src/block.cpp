#include    <pybind11/pybind11.h>
#include    <pybind11/stl.h>
#include    <pybind11/eigen.h>
#include    <pybind11/functional.h>
namespace py = pybind11;

#include <vector>

#include    <../examples/block.hpp>
#include    <../examples/domain_args.hpp>
#include    <mfa/tmesh.hpp>
#include    <mfa/mfa_data.hpp>
#include    <mfa/encode.hpp>


template <typename T>
void init_block(py::module& m, std::string name)
{
    using Bounds    = diy::Bounds<T>;
    using RCLink    = diy::RegularLink<diy::Bounds<T>>;
    using Master    = diy::Master;

    using namespace pybind11::literals;
    using namespace mfa;

    py::class_<ModelInfo>(m,"ModelInfo")
        .def(py::init<int, int, int, int>())
        .def(py::init<int, int, std::vector<int>, std::vector<int>>()) // both vectors of ints
        .def_readwrite("dom_dim",       &ModelInfo::dom_dim)
        .def_readwrite("var_dim",       &ModelInfo::var_dim)
        .def_readwrite("p",             &ModelInfo::p)
        .def_readwrite("nctrl_pts",     &ModelInfo::nctrl_pts)
    ;

    py::class_<MFAInfo> (m, "MFAInfo")
        //.def(py::init<int, int, ModelInfo, std::vector<ModelInfo>>())       // vector of model info
        .def(py::init<int, int, ModelInfo, ModelInfo>())        
        .def(py::init<int, int>()) 
        .def_readwrite("dom_dim",           &MFAInfo::dom_dim)
        .def_readwrite("verbose",           &MFAInfo::verbose)
        .def_readwrite("geom_model_info",   &MFAInfo::geom_model_info)
        .def_readwrite("var_model_infos",   &MFAInfo::var_model_infos)
        .def_readwrite("weighted",          &MFAInfo::weighted)
        .def_readwrite("local",             &MFAInfo::local)
        .def_readwrite("reg1and2",          &MFAInfo::reg1and2)
        .def_readwrite("regularization",    &MFAInfo::regularization)
        .def("addGeomInfo",                 &MFAInfo::addGeomInfo, "gmi"_a)
        .def("addVarInfo",                  &MFAInfo::addVarInfo, "vmi"_a)
    ;

// todo what to do about the overloaded functions
// todo what to do about non-python return objects

    py::class_<mfa::MFA<T>>(m, "MFA")
        .def(py::init<int, int>())
        .def(py::init<const MFAInfo>())
        .def_readwrite("dom_dim",       &mfa::MFA<T>::dom_dim)
        .def_readwrite("pt_dim",        &mfa::MFA<T>::pt_dim)
        .def("geom",                    &mfa::MFA<T>::geom, py::return_value_policy::reference)// empty, returns mfa::MFA_DATA)
        .def("var",                     &mfa::MFA<T>::var, py::return_value_policy::reference)    
        .def("FixedEncode",             &mfa::MFA<T>::FixedEncode, "input"_a, "regularization"_a, "regland2"_a, "weighted"_a, "force_unified"_a) //pointset, regularization, bool, bool, bool (reference output)
    ;
     
     // todo overload problems .def("AddGeometry",             &mfa::MFA<T>::AddGeometry, "mi"_a) // ModelInfo, return Void (inplace)
        // todo I think this is a duplicate //.def("AddGeometry",             &mfa::MFA<T>::AddGeometry, "degree"_a, "nctrl_pts"_a, "dim"_a) // VectorXi, VectorXi, int, return void 
        // .def("AddVariable",             &mfa::MFA<T>::AddVariable, "mi"_a) // same
        // .def("AddVariable",             &mfa::MFA<T>::AddVariable, "degree"_a, "nctrl_pts"_a, "dim"_a) // same

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
    //.def("set_knots")

    // .def_readwrite("dom_dim",           &Tmesh<T>::dom_dim)

    py::class_<TensorProduct<T>>(m, "TensorProduct")
        .def(py::init<int>())
        .def_readwrite("nctrl_pts",  &TensorProduct<T>::nctrl_pts)
        .def_readwrite("ctrl_pts",   &TensorProduct<T>::ctrl_pts)
    ;

    py::class_<BlockBase<T>>(m, "BlockBase")
        .def(py::init<>())
        .def("init_block",  &BlockBase<T>::init_block)
    ;

    py::class_<DomainArgs>(m, "DomainArgs")
    //todo vector of ints for intializer 
        .def(py::init<int, std::vector<int>>())
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
        // .def_static("add",                     &Block<T>::add)
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
                std::cerr << core.min.dimension() << std::endl;
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
        .def_static("save",                    &Block<T>::save)
        .def_static("load",                    &Block<T>::load)
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
            b->init_block(core, domain, dom_dim, pt_dim);
        });

    m.def("get_bound", [](const Bounds&  bound)
        {
            std::cerr << "dimension for min bound = " << bound.min.dimension() << std::endl;
            std::cerr << "dimension for max bound = " << bound.max.dimension() << std::endl;
        });

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
