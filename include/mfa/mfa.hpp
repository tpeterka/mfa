//--------------------------------------------------------------
// mfa object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_HPP
#define _MFA_HPP

#define MFA_NAW -1          // an invalid weight, indicating skip this control point

// comment out the following line for unclamped knots (single knot at each end of knot vector)
// clamped knots (repeated at ends) is the default method if no method is specified
// #define UNCLAMPED_KNOTS

// comment out the following line for domain parameterization
// domain parameterization is the default method if no method is specified
// #define CURVE_PARAMS

// low-d knot insertion
// high-d is the default if no method is specified
#define MFA_LOW_D

// check all curves for low-d knot insertion
// default is to sample fewer curves
// #define MFA_CHECK_ALL_CURVES

// comment out the following line for applying weights to only the range dimension
// weighing the range coordinate only is the default if no method is specified
// #define WEIGH_ALL_DIMS

// comment out the following line for original single tensor product version
// #define MFA_TMESH

// comment out the following line to encode local with unified dims
#define MFA_ENCODE_LOCAL_SEPARABLE

// unified dims solve use dense solver
// #define MFA_DENSE

// for debugging, can turn off constraints in local solve
// #define MFA_NO_CONSTRAINTS

#ifdef MFA_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#ifdef MFA_TBB
#define     TBB_SUPPRESS_DEPRECATED_MESSAGES    1
#include    <tbb/tbb.h>
using namespace tbb;
#endif

#include    <mfa/types.hpp>
#include    <mfa/utilities/util.hpp>
#include    <mfa/mfa_data.hpp>
#include    <mfa/pointset.hpp>
#include    <mfa/decode.hpp>
#include    <mfa/encode.hpp>
#include    <mfa/ray_encode.hpp>

namespace mfa
{ 

/*  The ModelInfo struct contains all of the information necessary to set up a MFA_Data object.
    This struct can be used to construct both geometry and science variable MFA_Data's. The 
    "control point dimensionality" (or the number of dimensions in the output space) is given
    by 'var_dim'. Thus, scalar models have var_dim=1, while vector-valued models have var_dim>1.

    To construct a "flat geometry" model, it is enough to use the constructor with signature
    'ModelInfo(dom_dim, var_dim)'. This creates a model with degree 1, two control points per 
    dimension, and no regularization.
*/
struct ModelInfo
{
    int         dom_dim        {0};         // domain dimensionality                                [default 0 (invalid)]
    int         var_dim        {0};         // dimensionality of variable                           [default 0 (invalid)]
    VectorXi    p;                          // degree of MFA in each domain dimension               [default = 1]
    VectorXi    nctrl_pts;                  // number of control points in each domain dimension    [default = p+1]

    // Default constructor. Dimensions = 0, vectors are empty.
    ModelInfo() { }

    // General-purpose constructor
    ModelInfo(int dom_dim_, int var_dim_,
                vector<int> p_, vector<int> nctrl_pts_) :
        dom_dim(dom_dim_),
        var_dim(var_dim_),
        p(Eigen::Map<VectorXi>(&p_[0], p_.size())),
        nctrl_pts(Eigen::Map<VectorXi>(&nctrl_pts_[0], nctrl_pts_.size()))
    {
        validate();
    }

    // Convenience constructor for models that are identical in each dimension
    ModelInfo(int dom_dim_, int var_dim_,
                int p_, int nctrl_pts_) :
        dom_dim(dom_dim_),
        var_dim(var_dim_),
        p(VectorXi::Constant(dom_dim_, p_)),
        nctrl_pts(VectorXi::Constant(dom_dim_, nctrl_pts_))
    { 
        validate();
    }

    ModelInfo(int dom_dim_, int var_dim_, int p_, vector<int> nctrl_pts_) :
        ModelInfo(dom_dim_, var_dim_, vector<int>(dom_dim_, p_), nctrl_pts_)
    { }

    // Convenience constructor for linear model with minimal control points
    ModelInfo(int dom_dim_, int var_dim_) :
        dom_dim(dom_dim_),
        var_dim(var_dim_)
    {
        p.resize(dom_dim);
        nctrl_pts.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            p[i] = 1;
            nctrl_pts[i] = p[i] + 1;
        }

        validate();
    }

    // Convenience constructor for simple geometry
    // Linear (flat) geometry, minimal control points, dom_dim = var_dim
    ModelInfo(int dom_dim_) :
        ModelInfo(dom_dim_, dom_dim_)
    { }

    bool validate()
    {
        if (p.size() != dom_dim || nctrl_pts.size() != dom_dim)
        {
            fmt::print(stderr, "ERROR: Incompatible domain dimension in ModelInfo.\n");
            fmt::print(stderr, "       dom_dim={}, p.size()={}, nctrl_pts.size()={}\n", dom_dim, p.size(), nctrl_pts.size());
            fmt::print(stderr, "Aborting.\n");
            exit(1);
        }
        for (int i = 0; i < dom_dim; i++)
        {
            if (nctrl_pts[i] <= p[i])
            {
                fmt::print(stderr, "ERROR: Too few control points in ModelInfo.\n");
                fmt::print(stderr, "       i={}, p[i]={}, nctrl_pts[i]={}\n", i, p[i], nctrl_pts[i]);
                fmt::print(stderr, "Aborting.\n");
                exit(1);
            }
        }

        return true;
    }
};

/*  The MFAInfo struct contains all of the information necessary to set up a fully general MFA object.
    The main components of MFAInfo are individual ModelInfo's, which describe the setup for an MFA_Data
    object. One ModelInfo is designated for the geometry. An arbitrary number of additional ModelInfo's
    describe the science variables in the MFA. Each variable ModelInfo can be completely different. 
    The only consistency condition is that all ModelInfo's (and MFAInfo) have the same dom_dim.

    The variables 'weighted', 'local', 'reg1and2', and 'regularization' describe settings for how the MFA
    should be encoded. They are not inherent to the MFA per se, we may want to move these settings to the
    Encoder in the future (they also can be set on a per-Model basis in general).
*/
struct MFAInfo
{

    int                 dom_dim;                // Number of dimensions in parameter space
    int                 verbose;                // Verbosity to be used by other operations
    ModelInfo           geom_model_info;        // Description of Geometry model
    vector<ModelInfo>   var_model_infos;        // Descriptions for each Science Variable model

    bool        weighted        {true};         // bool: solve for and use weights                      [default = true]
    bool        local           {false};        // bool: solve locally (with constraints) each round    [default = false]
    bool        reg1and2        {false};        // bool: regularize both 1st and 2nd derivatives        [default = false, 2nd derivs only]
    float       regularization  {0};            // regularization threshold                             [default = 0, no effect]

    // Construct an "empty" MFAInfo with no ModelInfo's (yet)
    // ModelInfo objects can be added with the addGeomInfo() and addVarInfo() methods
    MFAInfo(int dom_dim_, int verbose_) :
        dom_dim(dom_dim_),
        verbose(verbose_)
    { }

    // Constuct an MFAInfo from geometry and variable ModelInfo's all at once.
    MFAInfo(int dom_dim_, int verbose_,
            ModelInfo geom_model_info_,
            vector<ModelInfo> var_model_infos_) :
        dom_dim(dom_dim_),
        verbose(verbose_),
        geom_model_info(geom_model_info_),
        var_model_infos(var_model_infos_)
    {
        bool valid = true;

        if (dom_dim != geom_model_info.dom_dim)
        {
            valid = false;
        }
        for (int k = 0; k < nvars(); k++)
        {
            if (dom_dim != var_model_infos[k].dom_dim)
            {
                valid = false;
            }
        }

        if (!valid)
        {
            fmt::print(stderr, "ERROR: ModelInfos are incompatible with MFAInfo\nAborting.\n");
            exit(1);
        }
    }

    // Convenience constructor for a single science variable
    MFAInfo(int dom_dim_, int verbose_,
            ModelInfo geom_model_info_,
            ModelInfo var_model_info_) :
        MFAInfo(dom_dim_, verbose_, geom_model_info_, vector<ModelInfo>(1,var_model_info_))
    { }

    // Constructor from STL containers
    MFAInfo(int dom_dim_, int verbose_,
            vector<int> model_dims,
            int geom_degree,
            int geom_nctrl,
            int vars_degree,
            vector<int> vars_nctrl) :
        dom_dim(dom_dim_),
        verbose(verbose_)
    {
        int nvars       = model_dims.size() - 1;
        int geom_dim    = model_dims[0];

        // If only one value for vars_nctrl was parsed, assume it applies to all dims
        if (vars_nctrl.size() == 1 & dom_dim > 1)
        {
            vars_nctrl = vector<int>(dom_dim, vars_nctrl[0]);
        }

        // Minimal necessary control points
        if (geom_nctrl == -1) geom_nctrl = geom_degree + 1;
        for (int i = 0; i < vars_nctrl.size(); i++)
        {
            if (vars_nctrl[i] == -1) vars_nctrl[i] = vars_degree + 1;
        }

        ModelInfo geom_info(dom_dim, geom_dim, geom_degree, geom_nctrl);
        addGeomInfo(geom_info);

        for (int k = 0; k < nvars; k++)
        {
            ModelInfo var_info(dom_dim, model_dims[k+1], vars_degree, vars_nctrl);
            addVarInfo(var_info);
        }
    }

    int nvars() const {return var_model_infos.size();}

    int geom_dim() const
    {
        int gd = geom_model_info.var_dim;

        if (gd != 0) return gd;
        else
        {
            fmt::print(stderr, "ERROR: geom_model_info not set when calling geom_dim()\nAborting.\n");
            exit(1);
        }
        
        return 0;
    }

    int pt_dim() const
    {
        int pt_dim = geom_dim();
        for (int k = 0; k < nvars(); k++)
        {
            pt_dim += var_dim(k);
        }

        return pt_dim;
    }

    int var_dim(int k) const
    {
        if (k >= 0 && k < nvars())
        {
            return var_model_infos[k].var_dim;
        }
        else
        {
            fmt::print(stderr, "ERROR: var_dim() index out of range.\nAborting\n");
            exit(1);
        }

        return 0;
    }

    // Return a vector that contains the dimensionality of each model, INCLUDING the geometry model
    // Useful for constructing PointSets
    VectorXi model_dims() const
    {    
            VectorXi mds(nvars() + 1);
            
            mds(0) = geom_dim();
            for (int k = 0; k < nvars(); k++)
            {
                mds(k+1) = var_dim(k);
            }

            return mds;
    }

    // Add a ModelInfo and designate it as geometry
    void addGeomInfo(ModelInfo gmi)
    {
        if (dom_dim == gmi.dom_dim)
        {
            geom_model_info = gmi;
        }
        else
        {
            fmt::print(stderr, "ERROR: Incompatible dom_dim in addGeomInfo\nAborting.\n");
            exit(1);
        }

        return;
    }

    // Add a ModelInfo and designate it as a science variable
    void addVarInfo(ModelInfo vmi)
    {
        if (dom_dim == vmi.dom_dim)
        {
            var_model_infos.push_back(vmi);
        }
        else
        {
            fmt::print(stderr, "ERROR: Incompatible dom_dim in addVarInfo\nAborting.\n");
            exit(1);
        }

        return;
    }

    // Add a vector of ModelInfo's all at once
    void addVarInfo(vector<ModelInfo> vmis)
    {
        for (int k = 0; k < vmis.size(); k++)
        {
            addVarInfo(vmis[k]);
        }

        return;
    }

    // Reduce the number of control points of each model for a strong scaling study
    void splitStrongScaling(vector<int> divs)
    {
        for (int i = 0; i < dom_dim; i++)
        {
            // Divide geometry control points
            int gn = geom_model_info.nctrl_pts(i);
            int gp = geom_model_info.p(i);
            geom_model_info.nctrl_pts(i) = (gn / divs[i] > gp) ? gn / divs[i] : gp + 1;

            // Divide each variable's control points
            for (int k = 0; k < nvars(); k++)
            {
                int vn = var_model_infos[k].nctrl_pts(i);
                int vp = var_model_infos[k].p(i);
                var_model_infos[k].nctrl_pts(i) = (vn / divs[i] > vp) ? vn / divs[i] : vp + 1;
            }
        }
    }

    void reset()
    {
        geom_model_info = ModelInfo();
        var_model_infos.clear();

        return;
    }
};
} // namespace mfa

// forward-declare diy::Serialization so that it can be declared as a friend by MFA
namespace diy
{
    template <typename U>
    struct Serialization;
}

namespace mfa
{
    template <typename T>                           // float or double
    class MFA
    {
        template <typename U> friend struct diy::Serialization;

    public:
        int                 dom_dim{0};            // domain dimensionality
        int                 pt_dim{0};             // full control point dimensionality

    private:        
        unique_ptr<MFA_Data<T>>         geometry;
        vector<unique_ptr<MFA_Data<T>>> vars;
        int                             verbose{0};

        // Recomputes pt_dim so that the MFA stays consistent after each model is added.
        // In theory, pt_dim should always equal the max_dim of the last science variable
        // This method emits a warning if this is not the case.
        int recompute_pt_dim()
        {
            pt_dim = geom_dim();

            for (int i = 0; i < vars.size(); i++)
            {
                if (vars[i])
                    pt_dim += var_dim(i);
                else
                    fmt::print(stderr, "WARNING: Encountered null variable model in MFA::recompute_pt_dim()\n");
            }

            if (vars.size() > 0)
            {
                if (pt_dim != vars.back()->max_dim + 1)
                {
                    fmt::print(stderr, "WARNING: MFA pt_dim is inconsistent\n");
                }
            }

            return pt_dim;
        }
    

    public:
        // Constructs an "empty" MFA that can have Geometry and Variable models added to it
        MFA(int dom_dim_, int verbose_ = 0) :
            dom_dim(dom_dim_),
            verbose(verbose_)
        {
#ifdef EIGEN_OPENMP

            // set openMP threading for Eigen
            Eigen::initParallel();          // strictly not necessary for Eigen 3.3, but a good safety measure
            // Most modern CPUs have 2 hyperthreads per core, and openmp (hence Eigen) uses the number hyperthreads by default.
            // We want an automatic way to set the number of threads to number of physical cores, to prevent oversubscription.
            // So we set the number of Eigen threads to be half the default.
            Eigen::setNbThreads(Eigen::nbThreads()  / 2);
            fmt::print(stderr, "\nEigen is using {} openMP threads.\n\n", Eigen::nbThreads());
#endif
        }

        // Constructs an MFA from a full MFAInfo object. Additional Variable models can still
        // be added after this construction, if desired.
        MFA(const MFAInfo& mi) :
            dom_dim(mi.dom_dim),
            verbose(mi.verbose)
        {
            AddGeometry(mi.geom_model_info);
            for (int k = 0; k < mi.nvars(); k++)
            {
                AddVariable(mi.var_model_infos[k]);
            }

            recompute_pt_dim();

            if (mi.pt_dim() != pt_dim)
            {
                fmt::print(stderr, "ERROR: Incompatible pt_dim in MFA construction\nAborting.\n");
                exit(1);
            }
        }


        // This constructor is intended to be used for loading MFAs in and out of core
        // and can lead to an inconsistent state if the MFA_Data pointers are not set properly
        // 
        // NOTE: this constructor takes ownership of the MFA_Data pointers.
        //       these raw pointers should be set to nullptr after the MFA is constructed 
        //       to avoid double-frees, etc
        MFA(int dom_dim_, int verbose_, MFA_Data<T>* geom_, vector<MFA_Data<T>*> vars_) :
            dom_dim(dom_dim_),
            verbose(verbose_)
        {
            // If vars_ is nonempty, then we should always have a geometry model
            if (geom_ == nullptr && vars_.size() > 0)
                fmt::print(stderr, "WARNING: Constructing MFA with a null geometry model\n");

            // Each entry of vars should typically be non-null
            for (int i = 0; i < vars_.size(); i++)
            {
                if (vars_[i] == nullptr)
                    fmt::print(stderr, "WARNING: null variable model added during MFA construction\n");
            }

            // Set geometry model
            geometry.reset(geom_);

            // Set variable models and pt_dim
            vars.resize(vars_.size());
            for (int i = 0; i < vars.size(); i++)
            {
                vars[i].reset(vars_[i]);
            }

            recompute_pt_dim();
        }

        ~MFA() { }

        // Getter for geom_dim that checks for existence
        int geom_dim() const
        {
            return geom().dim();
        }

        // Getter for variable dimension that checks for existence and bounds
        int var_dim(int i) const
        {
            return var(i).dim();
        }

        // Return a vector that contains the dimensionality of each model, INCLUDING the geometry model
        // Useful for constructing PointSets
        VectorXi model_dims() const
        {
            VectorXi mds(nvars() + 1);
            
            mds(0) = geom_dim();
            for (int k = 0; k < nvars(); k++)
            {
                mds(k+1) = var_dim(k);
            }

            return mds;
        }

        const MFA_Data<T>& geom() const
        {
            if (!geometry)
            {
                fmt::print(stderr, "ERROR: Can't dereference null geometry model\nAborting.\n");
                exit(1);
            }

            return *geometry;
        }

        const MFA_Data<T>& var(int i) const
        {
            if (i < 0 || i >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::var()\n");
                fmt::print(stderr, "       i={}, nvars()={}\n", i, nvars());
                exit(1);
            }
            if (!vars[i])
            {
                fmt::print(stderr, "ERROR: Can't dereference null variable model (index {})\nAborting.\n", i);
                exit(1);
            }

            return *(vars[i]);
        }

        int nvars() const
        {
            return vars.size();
        }

        void AddGeometry(const ModelInfo& mi)
        {
            AddGeometry(mi.p, mi.nctrl_pts, mi.var_dim);
        }

        void AddGeometry(const VectorXi& degree, const VectorXi& nctrl_pts, int dim)
        {
            if (verbose) 
                fmt::print(stderr, "MFA: Adding geometry model\n");

            if (degree.size() != dom_dim || degree.minCoeff() < 0)
            {
                fmt::print(stderr, "ERROR: AddGeometry failed (degree invalid)\n");
                fmt::print(stderr, "       dom_dim={}, degree=[{}]\n", dom_dim, fmt::join(degree, ","));
                exit(1);
            }
            if (nctrl_pts.size() != dom_dim || (nctrl_pts - degree - VectorXi::Ones(dom_dim)).minCoeff() < 0)
            {
                fmt::print(stderr, "ERROR: AddGeometry failed (nctrl_pts invalid)\n");
                fmt::print(stderr, "       dom_dim={}, nctrl_pts=[{}]\n", dom_dim, nctrl_pts);
                exit(1);
            }
            if (dim <= 0)
            {
                fmt::print(stderr, "ERROR: AddGeometry failed (dim invalid): dim={}\n", dim);
                exit(1);
            }

            // set up geometry model
            int min_dim = 0;
            int max_dim = dim - 1;
            geometry.reset(new MFA_Data<T>(degree, nctrl_pts, min_dim, max_dim));

            recompute_pt_dim();
        }

        void AddVariable(const ModelInfo& mi)
        {
            AddVariable(mi.p, mi.nctrl_pts, mi.var_dim);
        }

        void AddVariable(const VectorXi& degree, const VectorXi& nctrl_pts, int dim)
        {
            if (verbose) 
                fmt::print(stderr, "MFA: Adding variable model {}\n", vars.size());

            if (!geometry)
            {
                fmt::print(stderr, "ERROR: Cannot add variable model before adding geometry model\n");
                exit(1);
            }
            if (degree.size() != dom_dim || degree.minCoeff() < 0)
            {
                fmt::print(stderr, "ERROR: AddVariable failed (degree invalid)\n");
                fmt::print(stderr, "       dom_dim={}, degree=[{}]\n", dom_dim, fmt::join(degree, ","));
                exit(1);
            }
            if (nctrl_pts.size() != dom_dim || (nctrl_pts - degree - VectorXi::Ones(dom_dim)).minCoeff() < 0)
            {
                fmt::print(stderr, "ERROR: AddVariable failed (nctrl_pts invalid)\n");
                fmt::print(stderr, "       dom_dim={}, nctrl_pts=[{}]\n", dom_dim, nctrl_pts);
                exit(1);
            }
            if (dim <= 0)
            {
                fmt::print(stderr, "ERROR: AddVariable failed (dim invalid): dim={}\n", dim);
                exit(1);
            }

            // Update vars vector
            int id = vars.size();

            // Compute min/max dim
            int min_dim = 0, max_dim = 0;            
            if (id == 0)
            {
                min_dim = geometry->max_dim + 1;
                max_dim = min_dim + dim - 1;
            }
            else
            {
                min_dim = vars[id-1]->max_dim + 1;
                max_dim = min_dim + dim - 1;
            }

            // Set up variable model
            vars.push_back(nullptr);                                                // Increase the size of the vars vector
            vars[id].reset(new MFA_Data<T>(degree, nctrl_pts, min_dim, max_dim));   // Add model to vars vector

            recompute_pt_dim();
        }

        // Update knot vector for geometry model.
        // This method should rarely be called by the user, except when the 
        // data are sampled from a manifold with complex shape and special knots
        // are needed.
        // The geometry model is set up with uniformly spaced knots during 
        // the construction of the MFA_Data object
        void setGeomKnots(const vector<vector<T>>& knots = vector<vector<T>>())
        {
            if (verbose)
                fmt::print(stderr, "MFA: Setting custom knot distribution for geometry model\n");

            geometry->set_knots(knots);
        }

        // Update knot vector for variable i. 
        // The variable models are set up with uniformly spaced knots during 
        // the construction of the MFA_Data object
        void setKnots(int i, const vector<vector<T>>& knots = vector<vector<T>>())
        {
            if (verbose)
                fmt::print(stderr, "MFA: Setting custom knot distribution for variable model {}\n", i);

            if (i < 0 || i >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::setKnots()\n");
                fmt::print(stderr, "       i={}, nvars()={}\n", i, nvars());
                exit(1);
            }
            vars[i]->set_knots(knots);
        }

        // Sets the same knot vector for all variables
        // Does not affect knots for geometry model, which are uniform in almost all cases
        // NOTE: This method can only work properly if every variable model has the same
        //       number of control points. Otherwise, it will cause an error and exit.
        void setKnots(const vector<vector<T>>& knots = vector<vector<T>>())
        {
            for (int i = 0; i < nvars(); i++)
            {
                setKnots(i, knots);
            }
        }

        // fixed number of control points encode
        void FixedEncodeImpl(
                MFA_Data<T>&        mfa_data,               // mfa data model
                const PointSet<T>&  input,                  // input points
                T                   regularization,
                bool                reg1and2,
                bool                weighted)               // solve for and use weights
        {
            if (weighted)
            {
                fmt::print(stderr, "WARNING: NURBS weights enabled during model encoding. This feature is not actively developed.\n");
                fmt::print(stderr, "         Are you sure this is what you want?\n");
            }

            // fixed encode assumes the tmesh has only one tensor product
            TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];

            t.weights = VectorX<T>::Ones(t.nctrl_pts.prod());
            Encoder<T> encoder(mfa_data, input, verbose);

            if (input.is_structured())
                encoder.Encode(t.nctrl_pts, t.ctrl_pts, t.weights, weighted);
            else
                encoder.EncodeUnified(0, regularization, reg1and2, weighted);  // Assumes only one tensor product
        }

        void RayEncode(
                int i,
                const PointSet<T>& input)
        {
            if (verbose)
                fmt::print(stderr, "MFA: Starting Ray Encoding for variable {}\n", i);
                
            vars[i]->set_param_idxs(input);
            RayEncoder<T> encoder(*vars[i], input, true);
            encoder.encode();
        }

        // adaptive encode
        void AdaptiveEncodeImpl(
                MFA_Data<T>&        mfa_data,               // mfa data model
                const PointSet<T>&  input,                  // input points
                T                   err_limit,              // maximum allowable normalized error
                bool                weighted,               // solve for and use weights (default = true)
                const VectorX<T>&   extents,                // extents in each dimension, for normalizing error (size 0 means do not normalize)
                int                 max_rounds)             // maximum number of rounds
        {
            mfa_data.set_param_idxs(input);                 // needed for the single tensor tmesh even when MFA_TMESH is not defined
#ifndef MFA_TMESH
            Encoder<T> encoder(mfa_data, input, verbose);
            encoder.OrigAdaptiveEncode(err_limit, weighted, extents, max_rounds);
#else
            Encoder<T> encoder(mfa_data, input, verbose);
            encoder.AdaptiveEncode(err_limit, weighted, extents, max_rounds);
#endif
        }

        // Decode geometry model at set of points
        void DecodeGeom(
                PointSet<T>&        output,
                const VectorXi&     derivs = VectorXi()) const
        {
            if (verbose)
                fmt::print(stderr, "MFA: Decoding geometry\n");

            mfa::Decoder<T> decoder(geom(), verbose);
            decoder.DecodePointSet(output, geometry->min_dim, geometry->max_dim, derivs);
        }

        // Decode geometry model at single point
        void DecodeGeom(
                const VectorX<T>&   param,
                VectorX<T>&         out_point,
                const VectorXi&     derivs = VectorXi()) const
        {
            if (out_point.size() != geom_dim())
            {
                fmt::print(stderr, "ERROR: Incorrect output vector dimension in MFA::DecodeGeom()\n");
                fmt::print(stderr, "       out_point.size()={}, geom_dim()={}\n", out_point.size(), geom_dim());
                exit(1);
            }

            Decoder<T> decoder(geom(), 0);        // nb. turning off verbose output when decoding single points
            
            // TODO: hard-coded for one tensor product
            decoder.VolPt(param, out_point, geometry->tmesh.tensor_prods[0], derivs);
        }

        // Decode variable model at set of points
        void DecodeVar(
                int                 i,
                PointSet<T>&        output,
                const VectorXi&     derivs = VectorXi()) const
        {
            if (verbose)
                fmt::print(stderr, "MFA: Decoding science variable {}\n", i);

            if (i < 0 || i >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::DecodeVar()\n");
                fmt::print(stderr, "       i={}, nvars()={}\n", i, nvars());
                exit(1);
            }

            mfa::Decoder<T> decoder(*(vars[i]), verbose);
            decoder.DecodePointSet(output, vars[i]->min_dim, vars[i]->max_dim, derivs);
        }

        // Decode variable model at single point
        void DecodeVar(
                int                 i,
                const VectorX<T>&   param,
                VectorX<T>&         out_point,
                const VectorXi&     derivs = VectorXi()) const
        {
            if (i < 0 || i >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::DecodeVar()\n");
                fmt::print(stderr, "       i={}, nvars()={}\n", i, nvars());
                exit(1);
            }

            if (out_point.size() != var_dim(i))
            {
                fmt::print(stderr, "ERROR: Incorrect output vector dimension in MFA::DecodeVar()\n");
                fmt::print(stderr, "       out_point.size()={}, var_dim(i)={}\n", out_point.size(), var_dim(i));
                exit(1);
            }

            mfa::Decoder<T> decoder(*(vars[i]), 0);     // nb. turning off verbose output when decoding single points

            // TODO: hard-coded for one tensor product
            decoder.VolPt(param, out_point, vars[i]->tmesh.tensor_prods[0], derivs);
        }

        // Decode all models at set of points
        void Decode(
                PointSet<T>&        output,
                const VectorXi&     derivs = VectorXi()) const
        {
            DecodeGeom(output);         // Don't differentiate geometry
            for (int i = 0; i < nvars(); i++)
            {
                DecodeVar(i, output, derivs);
            }
        }

        // Decode all models at single point
        // 
        // Note: 
        // If desired, we can re-rewrite this to avoid the copies from temp_out into out_point by writing decode
        // functions to take MatrixBase<> objects instead of vectors. This allows "block" expressions like
        // out_point.segment(a,b) to be passed to VolPt, and decoding can be done in-place.
        // However, we would need to refactor DecodeVar, DecodeGeom, and potentially all of the VolPt functions
        // to accept MatrixBase<> inputs. This would not be hard, but could be confusing to read; also this function
        // is not intended to be high-performance anyway, so the benefit may be minimal.
        // 
        // See: http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
        void Decode(
                const VectorX<T>&       param,
                Eigen::Ref<VectorX<T>>  out_point,
                const VectorXi&         derivs = VectorXi()) const
        {
            if (out_point.size() != pt_dim)
            {
                fmt::print(stderr, "ERROR: Incorrect output vector dimension in MFA::Decode()\n");
                fmt::print(stderr, "       out_point.size()={}, pt_dim()={}\n", out_point.size(), pt_dim);
                exit(1);
            }

            // We need an lvalue for passing into DecodeGeom and DecodeVar
            VectorX<T> temp_out = out_point.head(geom_dim());

            // Decode geometry
            DecodeGeom(param, temp_out);        // Don't differentiate geometry
            out_point.head(geom_dim()) = temp_out;

            // Decode variables
            for (int i = 0; i < nvars(); i++)
            {
                temp_out.resize(var_dim(i));
                DecodeVar(i, param, temp_out, derivs);

                out_point.segment(vars[i]->min_dim, var_dim(i)) = temp_out;
            }
        }

        // One-dimensional integral in direction 'dim' from u0 to u1
        void Integrate1D(
            int                 k,
            int                 dim,
            T                   u0,
            T                   u1,
            const VectorX<T>    params,
            VectorX<T>&         output) const
        {
            if (k < 0 || k >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::Integrate1D()\n");
                fmt::print(stderr, "       k={}, nvars()={}\n", k, nvars());
                exit(1);
            }

            mfa::Decoder<T> decoder(*(vars[k]), 0);  // no verbose output for single points
            decoder.AxisIntegral(dim, u0, u1, params, output);
        }

        void DefiniteIntegral(
                  int           k,
                  VectorX<T>&   output,
            const VectorX<T>&   a,
            const VectorX<T>&   b) const
        {
            if (k < 0 || k >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::DefiniteIntegral()\n");
                fmt::print(stderr, "       k={}, nvars()={}\n", k, nvars());
                exit(1);
            }

            mfa::Decoder<T> decoder(*(vars[k]), verbose);
            decoder.DefiniteIntegral(a, b, output);
        }

        // Integrates with respect to parameter space
        // Multiply by extent of each domain dimension to obtain integral wrt physical space
        void IntegratePointSet(
                const MFA_Data<T>&  mfa_data,
                PointSet<T>&        output,
                int                 int_dim)
        {
            const TensorProduct<T>&t = mfa_data.tmesh.tensor_prods[0];

            mfa::Decoder<T> decoder(mfa_data, verbose);
            decoder.IntegratePointSet(output, int_dim, t, mfa_data.min_dim, mfa_data.max_dim);
        }


        // decode points on grid in parameter space
        void DecodeAtGrid(  const MFA_Data<T>&      mfa_data,               // mfa_data
                            const VectorX<T>&       par_min,                // lower corner of domain in param space
                            const VectorX<T>&       par_max,                // upper corner of domain in param space
                            const VectorXi&         ndom_pts,              // number of points per direction
                            MatrixX<T>&             result)                 // decoded result
        {
            int verbose = 0;
            Decoder<T> decoder(mfa_data, verbose);
            decoder.DecodeGrid(result, mfa_data.min_dim, mfa_data.max_dim, par_min, par_max, ndom_pts);
        }

        // NOTE: This is very inefficient if called multiple times. Creation of Decoders and
        //       VectorX<T> all take non-negligible time due to dynamic memory allocation
        // compute the error (absolute value of coordinate-wise difference) of the mfa at a domain point
        // error is not normalized by the data range (absolute, not relative error)
        void AbsCoordError(
                const MFA_Data<T>&  mfa_data,               // mfa data model
                const PointSet<T>&  input,
                size_t              idx,                    // index of domain point
                VectorX<T>&         error) const            // (output) absolute value of error at each coordinate
        {
            VectorX<T> param(dom_dim);
            input.pt_params(idx, param);

            // approximated value
            VectorX<T> cpt(mfa_data.dim());          // approximated point
            Decoder<T> decoder(mfa_data, verbose);
            decoder.VolPt(param, cpt, mfa_data.tmesh.tensor_prods[0]);      // TODO: hard-coded for first tensor product

            for (auto i = 0; i < mfa_data.dim(); i++)
                error(i) = fabs(cpt(i) - input.domain(idx, mfa_data.min_dim + i));
        }

        // TODO This can be sped up significantly by NOT calling AbsCoordError.
        // By using AbsCoordError, we construct a Decoder for every single point.
        // Much faster to compute a single Decoder and call VolPt directly
        void AbsPointSetError(
            const   mfa::PointSet<T>& base,
                    mfa::PointSet<T>& error)
        {
            if (!base.is_same_layout(error))
            {
                fmt::print(stderr, "ERROR: PointSets in AbsPointSetError() have incompatible layouts\n");
                exit(1);
            }

#if defined( MFA_SERIAL) || defined(MFA_KOKKOS)   // serial version
            // copy geometric point coordinates
            for (size_t i = 0; i < error.npts; i++)
            {   
                for (auto j = 0; j < geom_dim(); j++)
                {
                    error.domain(i,j) = base.domain(i,j); // copy the geometric location of each point
                } 
            }

            // compute errors for each point and model
            for (size_t i = 0; i < error.npts; i++)
            {
                VectorX<T> err_vec;                             // errors for all coordinates in current model
                for (auto k = 0; k < nvars(); k++)          // for all science models
                {
                    err_vec.resize(var_dim(k));
                    AbsCoordError(*(vars[k]), base, i, err_vec);

                    for (auto j = 0; j < var_dim(k); j++)
                    {
                        error.domain(i, vars[k]->min_dim + j) = err_vec(j);      // error for each science variable
                    }
                }
            }

#endif // MFA_SERIAL || KOKKOS

#ifdef MFA_TBB

            parallel_for (size_t(0), (size_t)error.npts, [&] (size_t i)
                {
                    for (auto j = 0; j < geom_dim(); j++)
                    {
                        error.domain(i,j) = base.domain(i,j); // copy the geometric location of each point
                    }
                });

            parallel_for (size_t(0), (size_t)error.npts, [&] (size_t i)
                {
                VectorX<T> err_vec;                                 // errors for all coordinates in current model
                for (auto k = 0; k < nvars(); k++)              // for all science models
                {
                    err_vec.resize(var_dim(k));
                    AbsCoordError(*(vars[k]), base, i, err_vec);

                    for (auto j = 0; j < var_dim(k); j++)
                    {
                        error.domain(i, vars[k]->min_dim + j) = err_vec(j); // error for each science variable
                    }
                }
                });

#endif // MFA_TBB
        }


        //-------------------------------------//
        //-------Convenience Functions---------//

        // Convenience function for adding linear geometry model
        void AddGeometry(int dim)
        {
            VectorXi degree = VectorXi::Ones(dom_dim);
            VectorXi nctrl_pts = VectorXi::Constant(dom_dim, 2);

            AddGeometry(degree, nctrl_pts, dim);
        }

        // Convenience function for adding models with same degree in each dimension
        void AddVariable(int degree, const VectorXi& nctrl_pts, int dim)
        {
            VectorXi degree_vec = VectorXi::Constant(dom_dim, degree);

            AddVariable(degree_vec, nctrl_pts, dim);
        }

        // Fixed encode geometry model only
        // Useful if input arguments vary between models, or different models are encoded by different threads
        void FixedEncodeGeom(const PointSet<T>& input, bool weighted)
        {
            if (verbose)
                fmt::print(stderr, "MFA: Encoding geometry model (fixed)\n");

            T regularization = 0;
            FixedEncodeImpl(*geometry, input, regularization, false, weighted);
        }

        // Fixed encode single variable model only
        // Useful if input arguments vary between models, or different models are encoded by different threads 
        void FixedEncodeVar(int i, const PointSet<T>& input, T regularization, bool reg1and2, bool weighted)
        {
            if (i < 0 || i >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::FixedEncodeVar()\n");
                fmt::print(stderr, "       i={}, nvars()={}\n", i, nvars());
                exit(1);
            }

            if (verbose)
                fmt::print(stderr, "MFA: Encoding variable model {} (fixed)\n", i);

            FixedEncodeImpl(*(vars[i]), input, regularization, reg1and2, weighted);
        }

        // Fixed encode all models simultaneously
        void FixedEncode(const PointSet<T>& input, T regularization, bool reg1and2, bool weighted)
        {
            FixedEncodeGeom(input, weighted);

            for (int i = 0; i < nvars(); i++)
            {
                FixedEncodeVar(i, input, regularization, reg1and2, weighted);
            }
        }

        // Adaptive encode geometry model only
        // Useful if input arguments vary between models, or different models are encoded by different threads
        void AdaptiveEncodeGeom(const PointSet<T>&  input,
                                T                   err_limit,
                                bool                weighted,
                                const VectorX<T>&   extents,
                                int                 max_rounds)
        {
            if (verbose)
                fmt::print(stderr, "MFA: Encoding geometry model (adaptive)\n");

            AdaptiveEncodeImpl(*geometry, input, err_limit, weighted, extents, max_rounds);
        }

        // Adaptive encode single variable model only
        // Useful if input arguments vary between models, or different models are encoded by different threads
        void AdaptiveEncodeVar( int i,
                                const PointSet<T>&  input,
                                T                   err_limit,
                                bool                weighted,
                                const VectorX<T>&   extents,
                                int                 max_rounds)
        {
            if (i < 0 || i >= nvars())
            {
                fmt::print(stderr, "ERROR: var index out of range in MFA::AdaptiveEncodeVar()\n");
                fmt::print(stderr, "       i={}, nvars()={}\n", i, nvars());
                exit(1);
            }

            if (verbose)
                fmt::print(stderr, "MFA: Encoding variable model {} (adaptive)\n", i);
            
            AdaptiveEncodeImpl(*(vars[i]), input, err_limit, weighted, extents, max_rounds);
        }

        // Adaptive encode all models simultaneously
        void AdaptiveEncode(const PointSet<T>&  input,
                            T                   err_limit,
                            bool                weighted,
                            const VectorX<T>&   extents,
                            int                 max_rounds)
        {
            AdaptiveEncodeGeom(input, err_limit, weighted, extents, max_rounds);

            for (int i = 0; i < nvars(); i++)
            {
                AdaptiveEncodeVar(i, input, err_limit, weighted, extents, max_rounds);
            }
        }

        void shiftModel(MFA_Data<T>& model, const VectorX<T>& shift)
        {
            if (shift.size() != model.dim()) throw MFAError("Dimension mismatch in MFA::shiftModel()");

            for (auto t : model.tmesh.tensor_prods)
            {
                for (int i = 0; i < model.dim(); i++)
                {
                    t.ctrl_pts.col(i).array() += shift(i);
                }
            }
        }

        void shiftGeom(const VectorX<T>& shift)
        {
            fmt::print("MFA: Shifting geometry model\n");

            // todo: check geometry is initialized
            shiftModel(*geometry, shift);
        }

        void shiftVar(int i, const VectorX<T>& shift)
        {
            fmt::print("MFA: Shifting variable model {}\n", i);

            // todo: check variable is initialized
            if (i < 0 || i >= nvars())
            {
                throw MFAError("var index out of range in MFA::shiftVar()");
            }
            if (!vars[i])
            {
                throw MFAError(fmt::format("Attempted to deference null variable model (index {})", i));
            }

            shiftModel(*var[i], shift);
        }

        // Helper function that takes a knot span with or without repeated knots and returns
        // the same distribution with properly pinned knots
        vector<T> pinKnots(const vector<T>& knots, int degree)
        {
            assert(degree >= 0);
            assert(knots.size() > 0);

            int idx = 0;
            int count0 = 0;
            int count1 = 0;
            int extra0 = 0; // number of repeated knots at 0 to add
            int extra1 = 0; // number of repeated knots at 1 to add
            int last = knots.size() - 1; // index of final entry

            // Count number of knots previously pinned at 0
            idx = 0;
            while (idx < knots.size() && knots[idx] == 0)
            {
                count0++;
            }

            // Count number of knots previously pinned at 1
            idx = 0;
            while (idx < knots.size() && knots[last - idx] == 1)
            {
                count1++;
            }

            // Warn if there are already more pinned knots than we need, but continue
            if (count0 > degree + 1)
            {
                fmt::print(stderr, "WARNING: Tried to add pinned knots, but the knot vector already contained more than {} knots at 0", count0);
                extra0 = 0;
            }
            if (count1 > degree + 1)
            {
                fmt::print(stderr, "WARNING: Tried to add pinned knots, but the knot vector already contained more than {} knots at 1", count1);
                extra1 = 0;
            }

            // Create lists of extra knots to add
            extra0 = degree + 1 - count0;
            extra1 = degree + 1 - count1;
            vector<T> addKnots0(extra0, 0.0);
            vector<T> addKnots1(extra1, 1.0);
            
            // Concatenate into final vector with pinned knots
            vector<T> pinnedKnots;
            pinnedKnots.insert(pinnedKnots.end(), addKnots0.begin(), addKnots0.end());
            pinnedKnots.insert(pinnedKnots.end(), knots.begin(), knots.end());
            pinnedKnots.insert(pinnedKnots.end(), addKnots1.begin(), addKnots1.end());

            return pinnedKnots;
        }

        void dumpCollocationMatrixEncode(int i, PointSet<T>* ps)
        {
            Encoder<T> encoder(*vars[i], *ps, verbose);
            encoder.dumpCollocationMatrix();
        }

        void dumpCollocationMatrixDecode(int i, PointSet<T>* ps)
        {
            Decoder<T> decoder(*vars[i], verbose);
            decoder.dumpCollocationMatrix(*ps);
        }

        void printDetails() const
        {
            printDetails(this->verbose);
        }

        void printDetails(int verbose_) const
        {
            fmt::print(stderr, "MFA Details:\n");
            fmt::print(stderr, "  Domain dimension: {}\n", dom_dim);
            fmt::print(stderr, "  Total point dimension: {}\n", pt_dim);
            fmt::print(stderr, "  Number of variable models: {}\n", nvars());
            fmt::print(stderr, "  Geometry model:\n");
            if (geometry)
                geometry->printDetails(verbose_);
            else
                fmt::print(stderr, "    <null>\n");

            for (int i = 0; i < nvars(); i++)
            {
                fmt::print(stderr, "  Variable model {}:\n", i);
                if (vars[i])
                    vars[i]->printDetails(verbose_);
                else
                    fmt::print(stderr, "    <null>\n");
            }
        }
    };      // class MFA
}       // namespace mfa

#endif
