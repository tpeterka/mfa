//--------------------------------------------------------------
// base class for one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include    <mfa/mfa.hpp>

#include    <diy/master.hpp>
#include    <diy/reduce-operations.hpp>
#include    <diy/decomposition.hpp>
#include    <diy/assigner.hpp>
#include    <diy/io/block.hpp>
#include    <diy/pick.hpp>
#include    <diy/fmt/format.h>

#include    <stdio.h>

#include    <Eigen/Dense>

#include    <random>

using namespace std;

using Index = MatrixXf::Index;

template <typename T>
using Bounds = diy::Bounds<T>;
template <typename T>
using RCLink = diy::RegularLink<diy::Bounds<T>>;

struct ModelInfo
{
    ModelInfo(int dom_dim_, int pt_dim_) :
        dom_dim(dom_dim_),
        pt_dim(pt_dim_)
    {
        geom_p.resize(dom_dim);
        vars_p.resize(pt_dim - dom_dim);
        for (auto i = 0; i < vars_p.size(); i++)
            vars_p[i].resize(dom_dim);
        ndom_pts.resize(dom_dim);
        geom_nctrl_pts.resize(dom_dim);
        vars_nctrl_pts.resize(pt_dim - dom_dim);
        for (auto i = 0; i < vars_nctrl_pts.size(); i++)
            vars_nctrl_pts[i].resize(dom_dim);
    }
    virtual ~ModelInfo()                        {}
    int                 dom_dim;                // domain dimensionality
    int                 pt_dim;                 // point dimensionality (> dom_dim)
    vector<int>         geom_p;                 // degree in each dimension of geometry
    vector<vector<int>> vars_p;                 // degree in each dimension of each science variable vars_p[var][dim]
    vector<int>         ndom_pts;               // number of input points in each dimension of domain
    vector<int>         geom_nctrl_pts;         // number of input points in each dimension of geometry
    vector<vector<int>> vars_nctrl_pts;         // number of input pts in each dim of each science variable vars_nctrl_pts[var][dim]
    bool                weighted;               // solve for and use weights (default = true)
    bool                local;                  // solve locally (with constraints) each round (default = false)
    int                 verbose;                // debug level
};

// a solved and stored MFA model (geometry or science variable or both)
template <typename T>
struct Model
{
    int                 min_dim;                // starting coordinate of this model in full-dimensional data
    int                 max_dim;                // ending coordinate of this model in full-dimensional data
    mfa::MFA_Data<T>    *mfa_data;              // MFA model data
};

// block
template <typename T>
struct BlockBase
{
    // dimensionality
    int                 dom_dim;                // dimensionality of domain (geometry)
    int                 pt_dim;                 // dimensionality of full point (geometry + science vars)

    // input data
    MatrixX<T>          domain;                 // input data (1st dim changes fastest)
    VectorX<T>          bounds_mins;            // local domain minimum corner
    VectorX<T>          bounds_maxs;            // local domain maximum corner
    VectorX<T>          core_mins;              // local domain minimum corner w/o ghost
    VectorX<T>          core_maxs;              // local domain maximum corner w/o ghost

    // MFA object
    mfa::MFA<T>         *mfa;

    // MFA models
    Model<T>            geometry;               // geometry MFA
    vector<Model<T>>    vars;                   // science variable MFAs

    // output data
    MatrixX<T>          approx;                 // points in approximated volume

    // errors for each science variable
    vector<T>           max_errs;               // maximum (abs value) distance from input points to curve
    vector<T>           sum_sq_errs;            // sum of squared errors

    // error field for last science variable only
    MatrixX<T>          errs;                   // error field (abs. value, not normalized by data range)

    // fixed number of control points encode block
    void fixed_encode_block(
            const       diy::Master::ProxyWithLink& cp,
            ModelInfo&  info)
    {
        ModelInfo* a = &info;

        VectorXi nctrl_pts(dom_dim);
        VectorXi p(dom_dim);
        for (auto j = 0; j < dom_dim; j++)
        {
            nctrl_pts(j)    = a->geom_nctrl_pts[j];
            p(j)            = a->geom_p[j];
        }

        // encode geometry
        if (a->verbose && cp.master()->communicator().rank() == 0)
            fprintf(stderr, "\nEncoding geometry\n\n");
        geometry.mfa_data = new mfa::MFA_Data<T>(p,
                mfa->ndom_pts(),
                domain,
                mfa->params(),
                nctrl_pts,
                0,
                dom_dim - 1);
        // TODO: consider not weighting the geometry (only science variables), depends on geometry complexity
        mfa->FixedEncode(*geometry.mfa_data, domain, nctrl_pts, a->verbose, a->weighted);

        // encode science variables
        for (auto i = 0; i< vars.size(); i++)
        {
            if (a->verbose && cp.master()->communicator().rank() == 0)
                fprintf(stderr, "\nEncoding science variable %d\n\n", i);

            for (auto j = 0; j < dom_dim; j++)
            {
                p(j)            = a->vars_p[i][j];
                nctrl_pts(j)    = a->vars_nctrl_pts[i][j];
            }

            vars[i].mfa_data = new mfa::MFA_Data<T>(p,
                    mfa->ndom_pts(),
                    domain,
                    mfa->params(),
                    nctrl_pts,
                    dom_dim + i,        // assumes each variable is scalar
                    dom_dim + i);
            mfa->FixedEncode(*(vars[i].mfa_data), domain, nctrl_pts, a->verbose, a->weighted);
        }
    }

    // adaptively encode block to desired error limit
    void adaptive_encode_block(
            const diy::Master::ProxyWithLink& cp,
            T                                 err_limit,
            int                               max_rounds,
            ModelInfo&                        info)
    {
        ModelInfo* a = &info;
        VectorXi nctrl_pts(dom_dim);
        VectorXi p(dom_dim);
        VectorXi ndom_pts(dom_dim);
        VectorX<T> extents = bounds_maxs - bounds_mins;
        for (auto j = 0; j < dom_dim; j++)
        {
            nctrl_pts(j)    = a->geom_nctrl_pts[j];
            ndom_pts(j)     = a->ndom_pts[j];
            p(j)            = a->geom_p[j];
        }

        // encode geometry
        if (a->verbose && cp.master()->communicator().rank() == 0)
            fprintf(stderr, "\nEncoding geometry\n\n");
        geometry.mfa_data = new mfa::MFA_Data<T>(p,
                mfa->ndom_pts(),
                domain,
                mfa->params(),
                nctrl_pts,
                0,
                dom_dim - 1);
        // TODO: consider not weighting the geometry (only science variables), depends on geometry complexity
        mfa->AdaptiveEncode(*geometry.mfa_data, domain, err_limit, a->verbose, a->weighted, a->local, extents, max_rounds);

        // encode science variables
        for (auto i = 0; i< vars.size(); i++)
        {
            if (a->verbose && cp.master()->communicator().rank() == 0)
                fprintf(stderr, "\nEncoding science variable %d\n\n", i);

            for (auto j = 0; j < dom_dim; j++)
            {
                p(j)            = a->vars_p[i][j];
                nctrl_pts(j)    = a->vars_nctrl_pts[i][j];
            }

            vars[i].mfa_data = new mfa::MFA_Data<T>(p,
                    mfa->ndom_pts(),
                    domain,
                    mfa->params(),
                    nctrl_pts,
                    dom_dim + i,        // assumes each variable is scalar
                    dom_dim + i);
            mfa->AdaptiveEncode(*(vars[i].mfa_data), domain, err_limit, a->verbose, a->weighted, a->local, extents, max_rounds);
        }
    }

    // decode entire block
    void decode_block(
            const   diy::Master::ProxyWithLink& cp,
            int                                 verbose,        // debug level
            bool                                saved_basis)    // whether basis functions were saved and can be reused
    {
        approx.resize(domain.rows(), domain.cols());

        // geometry
        fprintf(stderr, "\n--- Decoding geometry ---\n\n");
        mfa->DecodeDomain(*geometry.mfa_data, verbose, approx, 0, dom_dim - 1, saved_basis);

        // science variables
        for (auto i = 0; i < vars.size(); i++)
        {
            fprintf(stderr, "\n--- Decoding science variable %d ---\n\n", i);
            mfa->DecodeDomain(*(vars[i].mfa_data), verbose, approx, dom_dim + i, dom_dim + i, saved_basis);  // assumes each variable is scalar
        }
    }

    // differentiate entire block
    void differentiate_block(
            const diy::Master::ProxyWithLink& cp,
            int                               verbose,  // debug level
            int                               deriv,    // which derivative to take (1 = 1st, 2 = 2nd, ...) in each domain dim.
            int                               partial,  // limit to partial derivative in just this dimension (-1 = no limit)
            int                               var)      // differentiate only this one science variable (0 to nvars -1, -1 = all vars)
    {
        approx.resize(domain.rows(), domain.cols());
        VectorXi derivs(dom_dim);

        for (auto i = 0; i < derivs.size(); i++)
            derivs(i) = deriv;

        // optional limit to one partial derivative
        if (deriv && dom_dim > 1 && partial >= 0)
        {
            for (auto i = 0; i < dom_dim; i++)
            {
                if (i != partial)
                    derivs(i) = 0;
            }
        }

        // science variables
        for (auto i = 0; i < vars.size(); i++)
            if (var < 0 || var == i)
            {
                // TODO: hard-coded for one tensor product
                vars[i].mfa_data = new mfa::MFA_Data<T>(vars[i].mfa_data->p,
                        mfa->ndom_pts(),
                        vars[i].mfa_data->tmesh,
                        dom_dim + i,        // assumes each variable is scalar
                        dom_dim + i);
                mfa->DecodeDomain(*(vars[i].mfa_data), verbose, approx, dom_dim + i, dom_dim + i, false, derivs);  // assumes each variable is scalar
            }

        // the derivative is a vector of same dimensionality as domain
        // derivative needs to be scaled by domain extent because u,v,... are in [0.0, 1.0]
        if (deriv)
        {
            if (dom_dim == 1 || partial >= 0) // TODO: not for mixed partials
            {
                if (dom_dim == 1)
                    partial = 0;
                for (auto j = 0; j < approx.cols(); j++)
                    // scale once for each derivative
                    for (auto i = 0; i < deriv; i++)
                        approx.col(j) /= (bounds_maxs(partial) - bounds_mins(partial));
            }
        }

        // for plotting, set the geometry coordinates to be the same as the input
        if (deriv)
            for (auto i = 0; i < dom_dim; i++)
                approx.col(i) = domain.col(i);
    }

    // compute error field and maximum error in the block
    // uses coordinate-wise difference between values
    void range_error(
            const   diy::Master::ProxyWithLink& cp,
            int     verbose,                                // output level
            bool    decode_block_,                          // decode entire block first
            bool    saved_basis)                            // whether basis functions were saved and can be reused
    {
        errs.resize(domain.rows(), domain.cols());
        errs            = domain;

        if (decode_block_)
            decode_block(cp, verbose, saved_basis);

#ifdef MFA_TBB      // TBB version

        // distance computation
        if (decode_block_)
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    VectorX<T> cpt = approx.row(i);
                    for (auto j = 0; j < domain.cols(); j++)
                    {
                    T err = fabs(cpt(j) - domain(i, j));
                    if (j >= dom_dim)
                    errs(i, j) = err;           // error for each science variable
                    }
                    });
        }
        else
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    VectorX<T> err;                                 // errors for all coordinates in current model
                    for (auto k = 0; k < vars.size() + 1; k++)      // for all models, geometry + science
                    {
                    if (k == 0)                                 // geometry
                    {
                    err.resize(geometry.max_dim - geometry.min_dim);
                    mfa->AbsCoordError(*geometry.mfa_data, domain, i, err, verbose);
                    }
                    else
                    {
                    err.resize(vars[k - 1].max_dim - vars[k - 1].min_dim);
                    mfa->AbsCoordError(*(vars[k - 1].mfa_data), domain, i, err, verbose);
                    }

                    for (auto j = 0; j < err.size(); j++)
                    if (k)                                              // science variables
                    errs(i, vars[k - 1].min_dim + j) = err(j); // error for each science variable
                    }
                    });
        }

#endif              // end TBB version

#ifdef MFA_SERIAL   // serial version

        for (auto i = 0; i < (size_t)domain.rows(); i++)
        {
            if (decode_block_)
            {
                VectorX<T> cpt = approx.row(i);
                for (auto j = 0; j < domain.cols(); j++)
                {
                    T err = fabs(cpt(j) - domain(i, j));
                    if (j >= dom_dim)
                        errs(i, j) = err;           // error for each science variable
                }
            }
            else
            {
                VectorX<T> err;                                 // errors for all coordinates in current model
                for (auto k = 0; k < vars.size() + 1; k++)      // for all models, geometry + science
                {
                    if (k == 0)                                 // geometry
                    {
                        err.resize(geometry.max_dim - geometry.min_dim);
                        mfa->AbsCoordError(*geometry.mfa_data, domain, i, err, verbose);
                    }
                    else
                    {
                        err.resize(vars[k - 1].max_dim - vars[k - 1].min_dim);
                        mfa->AbsCoordError(*(vars[k - 1].mfa_data),domain, i, err, verbose);
                    }

                    for (auto j = 0; j < err.size(); j++)
                    {
                        if (k)                                              // science variables
                            errs(i, vars[k - 1].min_dim + j) = err(j);      // error for each science variable
                    }
                }
            }
        }

#endif              // end serial version

        for (auto j = dom_dim; j < domain.cols(); j++)
            sum_sq_errs[j - dom_dim] = 0.0;
        for (auto i = 0; i < domain.rows(); i++)
        {
            for (auto j = dom_dim; j < domain.cols(); j++)
            {
                sum_sq_errs[j - dom_dim] += (errs(i, j) * errs(i, j));
                if ((i == 0 && j == dom_dim) || errs(i, j) > max_errs[j - dom_dim])
                    max_errs[j - dom_dim] = errs(i, j);
            }
        }
    }

    void print_block(const diy::Master::ProxyWithLink& cp,
            bool                              error)       // error was computed
    {
        fprintf(stderr, "gid = %d\n", cp.gid());
        //         cerr << "domain\n" << domain << endl;

        VectorXi tot_nctrl_pts = VectorXi::Zero(geometry.mfa_data->dom_dim);

        // geometry
        cerr << "\n------- geometry model -------" << endl;
        for (auto j = 0; j < geometry.mfa_data->tmesh.tensor_prods.size(); j++)
            tot_nctrl_pts += geometry.mfa_data->tmesh.tensor_prods[j].nctrl_pts;
        cerr << "# output ctrl pts     = [ " << tot_nctrl_pts.transpose() << " ]" << endl;

        //  debug: print control points and weights
        //         print_ctrl_weights(geometry.mfa_data->tmesh);
        // debug: print knots
        //         print_knots(geometry.mfa_data->tmesh);

        fprintf(stderr, "# output knots        = [ ");
        for (auto j = 0 ; j < geometry.mfa_data->tmesh.all_knots.size(); j++)
        {
            fprintf(stderr, "%ld ", geometry.mfa_data->tmesh.all_knots[j].size());
        }
        fprintf(stderr, "]\n");

        cerr << "-----------------------------" << endl;

        // science variables
        cerr << "\n----- science variable models -----" << endl;
        for (auto i = 0; i < vars.size(); i++)
        {
            T range_extent = domain.col(dom_dim + i).maxCoeff() - domain.col(dom_dim + i).minCoeff();
            cerr << "\n---------- var " << i << " ----------" << endl;
            tot_nctrl_pts = VectorXi::Zero(vars[i].mfa_data->dom_dim);
            for (auto j = 0; j < vars[i].mfa_data->tmesh.tensor_prods.size(); j++)
                tot_nctrl_pts += vars[i].mfa_data->tmesh.tensor_prods[j].nctrl_pts;
            cerr << "# ouput ctrl pts      = [ " << tot_nctrl_pts.transpose() << " ]" << endl;

            //  debug: print control points and weights
            //             print_ctrl_weights(vars[i].mfa_data->tmesh);
            // debug: print knots
            //             print_knots(vars[i].mfa_data->tmesh);


            fprintf(stderr, "# output knots        = [ ");
            for (auto j = 0 ; j < vars[i].mfa_data->tmesh.all_knots.size(); j++)
            {
                fprintf(stderr, "%ld ", vars[i].mfa_data->tmesh.all_knots[j].size());
            }
            fprintf(stderr, "]\n");

            cerr << "-----------------------------" << endl;

            T rms_err = sqrt(sum_sq_errs[i] / (domain.rows()));
            fprintf(stderr, "range extent          = %e\n",  range_extent);
            if (error)
            {
                fprintf(stderr, "max_err               = %e\n",  max_errs[i]);
                fprintf(stderr, "normalized max_err    = %e\n",  max_errs[i] / range_extent);
                fprintf(stderr, "sum of squared errors = %e\n",  sum_sq_errs[i]);
                fprintf(stderr, "RMS error             = %e\n",  rms_err);
                fprintf(stderr, "normalized RMS error  = %e\n",  rms_err / range_extent);
            }
            cerr << "-----------------------------" << endl;
        }
        cerr << "\n-----------------------------------" << endl;

        //  debug: print approximated points
        //         cerr << approx.rows() << " approximated points\n" << approx << endl;
        //         fprintf(stderr, "# input points        = %ld\n", domain.rows());

        fprintf(stderr, "compression ratio     = %.2f\n", compute_compression());
    }

    // compute compression ratio
    float compute_compression()
    {
        // TODO: hard-coded for one tensor product
        float in_coords = domain.rows() * domain.cols();
        float out_coords = geometry.mfa_data->tmesh.tensor_prods[0].ctrl_pts.rows() *
            geometry.mfa_data->tmesh.tensor_prods[0].ctrl_pts.cols();
        for (auto j = 0; j < geometry.mfa_data->tmesh.all_knots.size(); j++)
            out_coords += geometry.mfa_data->tmesh.all_knots[j].size();
        for (auto i = 0; i < vars.size(); i++)
        {
            out_coords += (vars[i].mfa_data->tmesh.tensor_prods[0].ctrl_pts.rows() *
                    vars[i].mfa_data->tmesh.tensor_prods[0].ctrl_pts.cols());
            for (auto j = 0; j < vars[i].mfa_data->tmesh.all_knots.size(); j++)
                out_coords += vars[i].mfa_data->tmesh.all_knots[j].size();
        }
        return(in_coords / out_coords);
    }

    //  debug: print control points and weights in all tensor products of a tmesh
    void print_ctrl_weights(mfa::Tmesh<T>& tmesh)
    {
        for (auto i = 0; i < tmesh.tensor_prods.size(); i++)
        {
            cerr << "tensor_prods[" << i << "]:\n" << endl;
            cerr << tmesh.tensor_prods[i].ctrl_pts.rows() <<
                " final control points\n" << tmesh.tensor_prods[i].ctrl_pts << endl;
            cerr << tmesh.tensor_prods[i].weights.size()  <<
                " final weights\n" << tmesh.tensor_prods[i].weights << endl;
        }
    }

    // debug: print knots in a tmesh
    void print_knots(mfa::Tmesh<T>& tmesh)
    {
        for (auto j = 0 ; j < tmesh.all_knots.size(); j++)
        {
            fprintf(stderr, "%ld knots[%d]: [ ", tmesh.all_knots[j].size(), j);
            for (auto k = 0; k < tmesh.all_knots[j].size(); k++)
                fprintf(stderr, "%.3lf ", tmesh.all_knots[j][k]);
            fprintf(stderr, " ]\n");
        }
        fprintf(stderr, "\n");
    }

    void print_deriv(const diy::Master::ProxyWithLink& cp)
    {
        fprintf(stderr, "gid = %d\n", cp.gid());
        cerr << "domain\n" << domain << endl;
        cerr << approx.rows() << " derivatives\n" << approx << endl;
        fprintf(stderr, "\n");
    }

    // write original and approximated data in raw format
    // only for one block (one file name used, ie, last block will overwrite earlier ones)
    void write_raw(const diy::Master::ProxyWithLink& cp)
    {
        int last = domain.cols() - 1;           // last column in domain points

        // write original points
        ofstream domain_outfile;
        domain_outfile.open("orig.raw", ios::binary);
        vector<T> out_domain(domain.rows());
        for (auto i = 0; i < domain.rows(); i++)
            out_domain[i] = domain(i, last);
        domain_outfile.write((char*)(&out_domain[0]), domain.rows() * sizeof(T));
        domain_outfile.close();

#if 0
        // debug: read back original points
        ifstream domain_infile;
        vector<T> in_domain(domain.rows());
        domain_infile.open("orig.raw", ios::binary);
        domain_infile.read((char*)(&in_domain[0]), domain.rows() * sizeof(T));
        domain_infile.close();
        for (auto i = 0; i < domain.rows(); i++)
            if (in_domain[i] != domain(i, last))
                fprintf(stderr, "Error writing raw data: original data does match writen/read back data\n");
#endif

        // write approximated points
        ofstream approx_outfile;
        approx_outfile.open("approx.raw", ios::binary);
        vector<T> out_approx(approx.rows());
        for (auto i = 0; i < approx.rows(); i++)
            out_approx[i] = approx(i, last);
        approx_outfile.write((char*)(&out_approx[0]), approx.rows() * sizeof(T));
        approx_outfile.close();

#if 0
        // debug: read back original points
        ifstream approx_infile;
        vector<T> in_approx(approx.rows());
        approx_infile.open("approx.raw", ios::binary);
        approx_infile.read((char*)(&in_approx[0]), approx.rows() * sizeof(T));
        approx_infile.close();
        for (auto i = 0; i < approx.rows(); i++)
            if (in_approx[i] != approx(i, last))
                fprintf(stderr, "Error writing raw data: approximated data does match writen/read back data\n");
#endif
    }

    // send decoded ghost points
    // assumes entire block was already decoded
    void send_ghost_pts(const diy::Master::ProxyWithLink&           cp,
            const diy::RegularDecomposer<Bounds<T>>&    decomposer)
    {
        RCLink<T> *l = static_cast<RCLink<T> *>(cp.link());
        map<diy::BlockID, vector<VectorX<T> > > outgoing_pts;
        vector<T>   dom_pt(dom_dim);                    // only domain coords of point, for checking neighbor bounds
        VectorX<T>  full_pt(approx.cols());             // full coordinates of point
        T eps = 1.0e-6;

        // check decoded points whether they fall into neighboring block bounds (including ghost)
        for (auto i = 0; i < (size_t)approx.rows(); i++)
        {
            vector<int> dests;                      // link neighbor targets (not gids)
            auto it = dests.begin();
            insert_iterator<vector<int> > insert_it(dests, it);
            for (auto j = 0; j < dom_dim; j++)
                dom_pt[j] = approx(i, j);
            diy::near(*l, dom_pt, eps, insert_it, decomposer.domain);
            if (dests.size())
                full_pt = approx.row(i);

            // prepare map of pts going to each neighbor
            for (auto j = 0; j < dests.size(); j++)
            {
                diy::BlockID bid = l->target(dests[j]);
                outgoing_pts[bid].push_back(full_pt);
                // debug: print the point
                cerr << "gid " << cp.gid() << " sent " << full_pt.transpose() << " to gid " << bid.gid << endl;
            }
        }

        // enqueue the vectors of points to send to each neighbor block
        for (auto it = outgoing_pts.begin(); it != outgoing_pts.end(); it++)
            for (auto i = 0; i < it->second.size(); i++)
                cp.enqueue(it->first, it->second[i]);
    }

    void recv_ghost_pts(const diy::Master::ProxyWithLink& cp)
    {
        VectorX<T> pt(approx.cols());                   // incoming point

        // gids of incoming neighbors in the link
        std::vector<int> in;
        cp.incoming(in);

        // for all neighbor blocks
        // dequeue data received from this neighbor block in the last exchange
        for (unsigned i = 0; i < in.size(); ++i)
        {
            while (cp.incoming(in[i]))
            {
                cp.dequeue(in[i], pt);
                // debug: print the point
                cerr << "gid " << cp.gid() << " received " << pt.transpose() << endl;
            }
        }
    }

    // ----- t-mesh methods -----

    // initialize t-mesh with some test data
    void init_tmesh(const diy::Master::ProxyWithLink&   cp)
    {
        // geometry mfa
        geometry.mfa = new mfa::MFA<T>(geometry.p,
                mfa->ndom_pts(),
                domain,
                geometry.ctrl_pts,
                geometry.nctrl_pts,
                geometry.weights,
                geometry.knots,
                0,
                dom_dim - 1);

        // science variable mfas
        for (auto i = 0; i< vars.size(); i++)
        {
            vars[i].mfa = new mfa::MFA<T>(vars[i].p,
                    mfa->ndom_pts(),
                    domain,
                    vars[i].ctrl_pts,
                    vars[i].nctrl_pts,
                    vars[i].weights,
                    vars[i].knots,
                    dom_dim + i,        // assumes each variable is scalar
                    dom_dim + i);
        }

        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // initialize all_knots
        tmesh.all_knots.resize(dom_dim);
        tmesh.all_knot_levels.resize(dom_dim);

        for (auto i = 0; i < dom_dim; i++)
        {
            tmesh.all_knots[i].resize(6);       // hard-coded to match diagram
            tmesh.all_knot_levels[i].resize(6);
            for (auto j = 0; j < tmesh.all_knots[i].size(); j++)
            {
                tmesh.all_knots[i][j] = j / static_cast<T>(tmesh.all_knots[i].size() - 1);
                tmesh.all_knot_levels[i][j] = 0;
            }
        }

        // initialize first tensor product
        vector<KnotIdx> knot_mins(dom_dim);
        vector<KnotIdx> knot_maxs(dom_dim);
        for (auto i = 0; i < dom_dim; i++)
        {
            knot_mins[i] = 0;
            knot_maxs[i] = 5;
        }

        tmesh.append_tensor(knot_mins, knot_maxs);
    }

    // refine the t-mesh the first time
    void refine1_tmesh(const diy::Master::ProxyWithLink&   cp)
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // insert new knots into all_knots
        tmesh.insert_knot(0, 2, 1, 0.3);
        tmesh.insert_knot(1, 3, 1, 0.5);

        // insert tensor product
        vector<KnotIdx> knot_mins(dom_dim);
        vector<KnotIdx> knot_maxs(dom_dim);
        assert(dom_dim == 2);           // testing 2d for now
        knot_mins[0] = 0;
        knot_mins[1] = 1;
        knot_maxs[0] = 4;
        knot_maxs[1] = 5;
        tmesh.append_tensor(knot_mins, knot_maxs);
    }

    // refine the t-mesh the second time
    void refine2_tmesh(const diy::Master::ProxyWithLink&   cp)
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // insert new knots into all_knots
        tmesh.insert_knot(0, 4, 2, 0.5);
        tmesh.insert_knot(1, 4, 2, 0.55);

        // insert tensor product
        vector<size_t> knot_mins(dom_dim);
        vector<size_t> knot_maxs(dom_dim);
        assert(dom_dim == 2);           // testing 2d for now
        knot_mins[0] = 2;
        knot_mins[1] = 2;
        knot_maxs[0] = 6;
        knot_maxs[1] = 6;
        tmesh.append_tensor(knot_mins, knot_maxs);
    }

    // print the t-mesh
    void print_tmesh(const diy::Master::ProxyWithLink&      cp)
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        tmesh.print();
    }

    // decode a point in the t-mesh
    void decode_tmesh(const diy::Master::ProxyWithLink&     cp,
            const VectorX<T>&                     param)      // parameters of point to decode
    {
        // pretend this tmesh is for the first science variable
        mfa::Tmesh<T>& tmesh = vars[0].mfa->mfa_data().tmesh;

        // compute range of anchor points for a given point to decode
        vector<vector<size_t>> anchors(dom_dim);                        // anchors affecting the decoding point
        tmesh.anchors(param, anchors);

        // print anchors
        fmt::print(stderr, "for decoding point = [ ");
        for (auto i = 0; i < dom_dim; i++)
            fmt::print(stderr, "{} ", param[i]);
        fmt::print(stderr, "],\n");

        for (auto i = 0; i < dom_dim; i++)
        {
            fmt::print(stderr, "dim {} local anchors = [", i);
            for (auto j = 0; j < anchors[i].size(); j++)
                fmt::print(stderr, "{} ", anchors[i][j]);
            fmt::print(stderr, "]\n");
        }
        fmt::print(stderr, "\n--------------------------\n\n");

        // compute local knot vectors for each anchor in Cartesian product of anchors

        int tot_nanchors = 1;                                               // total number of anchors in flattened space
        for (auto i = 0; i < dom_dim; i++)
            tot_nanchors *= anchors[i].size();
        vector<int> anchor_idx(dom_dim);                                    // current index of anchor in each dim, initialized to 0s

        for (auto j = 0; j < tot_nanchors; j++)
        {
            vector<size_t> anchor(dom_dim);                                 // one anchor from anchors
            for (auto i = 0; i < dom_dim; i++)
                anchor[i] = anchors[i][anchor_idx[i]];
            anchor_idx[0]++;

            // for all dimensions except last, check if anchor_idx is at the end
            for (auto k = 0; k < dom_dim - 1; k++)
            {
                if (anchor_idx[k] == anchors[k].size())
                {
                    anchor_idx[k] = 0;
                    anchor_idx[k + 1]++;
                }
            }

            vector<vector<size_t>> loc_knot_vec(dom_dim);                   // local knot vector
            tmesh.local_knot_vector(anchor, loc_knot_vec);

            // print local knot vectors
            fmt::print(stderr, "for anchor = [ ");
            for (auto i = 0; i < dom_dim; i++)
                fmt::print(stderr, "{} ", anchor[i]);
            fmt::print(stderr, "],\n");

            for (auto i = 0; i < dom_dim; i++)
            {
                fmt::print(stderr, "dim {} local knot vector = [", i);
                for (auto j = 0; j < loc_knot_vec[i].size(); j++)
                    fmt::print(stderr, "{} ", loc_knot_vec[i][j]);
                fmt::print(stderr, "]\n");
            }
            fmt::print(stderr, "\n--------------------------\n\n");

            // TODO: insert any missing knots and control points (port Youssef's knot insertion algorithm)
            // TODO: insert missing knots into tmesh so that knot lines will be intersected

            // TODO: compute basis function in each dimension

            // TODO: locate corresponding control point

            // TODO: multiply basis function by control point and add to current sum
        }

        // TODO: normalize sum of basis functions * control points
    }
};

namespace mfa
{
    template<typename B>                        // B = block object
        void* create()          { return new B; }

    template<typename B>                        // B = block object
        void destroy(void* b)
        {
            B* block = static_cast<B*>(b);
            if (block->mfa)
                delete block->mfa;
            delete block;
        }

    template<typename B, typename T>                // B = block object,  T = float or double
        void add(                                       // add the block to the decomposition
                int                 gid,                // block global id
                const Bounds<T>&    core,               // block bounds without any ghost added
                const Bounds<T>&    bounds,             // block bounds including any ghost region added
                const Bounds<T>&    domain,             // global data bounds
                const RCLink<T>&    link,               // neighborhood
                diy::Master&        master,             // diy master
                int                 dom_dim,            // domain dimensionality
                int                 pt_dim,             // point dimensionality
                T                   ghost_factor = 0.0) // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
        {
            B*              b   = new B;
            RCLink<T>*      l   = new RCLink<T>(link);
            diy::Master&    m   = const_cast<diy::Master&>(master);
            m.add(gid, b, l);

            b->dom_dim = dom_dim;
            b->pt_dim  = pt_dim;

            // NB: using bounds to hold full point dimensionality, but using core to hold only domain dimensionality
            b->bounds_mins.resize(pt_dim);
            b->bounds_maxs.resize(pt_dim);
            b->core_mins.resize(dom_dim);
            b->core_maxs.resize(dom_dim);

            // manually set ghosted block bounds as a factor increase of original core bounds
            for (int i = 0; i < dom_dim; i++)
            {
                T ghost_amount = ghost_factor * (core.max[i] - core.min[i]);
                if (core.min[i] > domain.min[i])
                    b->bounds_mins(i) = core.min[i] - ghost_amount;
                else
                    b->bounds_mins(i)= core.min[i];

                if (core.max[i] < domain.max[i])
                    b->bounds_maxs(i) = core.max[i] + ghost_amount;
                else
                    b->bounds_maxs(i) = core.max[i];
                b->core_mins(i) = core.min[i];
                b->core_maxs(i) = core.max[i];
            }

            b->mfa = NULL;
        }

    template<typename B, typename T>                // B = block object,  T = float or double
        void save(
                const void*        b_,
                diy::BinaryBuffer& bb)
        {
            B* b = (B*)b_;

            // TODO: don't save domain in practice
            diy::save(bb, b->domain);

            // top-level mfa data
            diy::save(bb, b->dom_dim);
            diy::save(bb, b->mfa->ndom_pts());

            diy::save(bb, b->bounds_mins);
            diy::save(bb, b->bounds_maxs);
            diy::save(bb, b->core_mins);
            diy::save(bb, b->core_maxs);

            // geometry
            diy::save(bb, b->geometry.mfa_data->p);
            diy::save(bb, b->geometry.mfa_data->tmesh.tensor_prods.size());
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.nctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.ctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.weights);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.knot_mins);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::save(bb, t.knot_maxs);
            diy::save(bb, b->geometry.mfa_data->tmesh.all_knots);
            diy::save(bb, b->geometry.mfa_data->tmesh.all_knot_levels);

            // science variables
            diy::save(bb, b->vars.size());
            for (auto i = 0; i < b->vars.size(); i++)
            {
                diy::save(bb, b->vars[i].mfa_data->p);
                diy::save(bb, b->vars[i].mfa_data->tmesh.tensor_prods.size());
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.nctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.ctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.weights);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.knot_mins);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::save(bb, t.knot_maxs);
                diy::save(bb, b->vars[i].mfa_data->tmesh.all_knots);
                diy::save(bb, b->vars[i].mfa_data->tmesh.all_knot_levels);
            }

            diy::save(bb, b->approx);
            diy::save(bb, b->errs);
        }

    template<typename B, typename T>                // B = block object, T = float or double
        void load(
                void*              b_,
                diy::BinaryBuffer& bb)
        {
            B* b = (B*)b_;

            // TODO: don't load domain in practice
            diy::load(bb, b->domain);

            // top-level mfa data
            diy::load(bb, b->dom_dim);
            VectorXi ndom_pts(b->dom_dim);
            diy::load(bb, ndom_pts);
            b->mfa = new mfa::MFA<T>(b->dom_dim, ndom_pts, b->domain);

            diy::load(bb, b->bounds_mins);
            diy::load(bb, b->bounds_maxs);
            diy::load(bb, b->core_mins);
            diy::load(bb, b->core_maxs);

            VectorXi    p;                  // degree of the mfa
            size_t      ntensor_prods;      // number of tensor products in the tmesh

            // geometry
            diy::load(bb, p);
            diy::load(bb, ntensor_prods);
            b->geometry.mfa_data = new mfa::MFA_Data<T>(p, ntensor_prods);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.nctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.ctrl_pts);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.weights);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.knot_mins);
            for (TensorProduct<T>& t: b->geometry.mfa_data->tmesh.tensor_prods)
                diy::load(bb, t.knot_maxs);
            diy::load(bb, b->geometry.mfa_data->tmesh.all_knots);
            diy::load(bb, b->geometry.mfa_data->tmesh.all_knot_levels);

            // science variables
            size_t nvars;
            diy::load(bb, nvars);
            b->vars.resize(nvars);
            for (auto i = 0; i < b->vars.size(); i++)
            {
                diy::load(bb, p);
                diy::load(bb, ntensor_prods);
                b->vars[i].mfa_data = new mfa::MFA_Data<T>(p, ntensor_prods);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.nctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.ctrl_pts);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.weights);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.knot_mins);
                for (TensorProduct<T>& t: b->vars[i].mfa_data->tmesh.tensor_prods)
                    diy::load(bb, t.knot_maxs);
                diy::load(bb, b->vars[i].mfa_data->tmesh.all_knots);
                diy::load(bb, b->vars[i].mfa_data->tmesh.all_knot_levels);
            }

            diy::load(bb, b->approx);
            diy::load(bb, b->errs);
        }
}                       // namespace

namespace diy
{
    template <typename T>
        struct Serialization<MatrixX<T>>
        {
            static
                void save(diy::BinaryBuffer& bb, const MatrixX<T>& m)
                {
                    diy::save(bb, m.rows());
                    diy::save(bb, m.cols());
                    for (size_t i = 0; i < m.rows(); ++i)
                        for (size_t j = 0; j < m.cols(); ++j)
                            diy::save(bb, m(i, j));
                }
            static
                void load(diy::BinaryBuffer& bb, MatrixX<T>& m)
                {
                    Index rows, cols;
                    diy::load(bb, rows);
                    diy::load(bb, cols);
                    m.resize(rows, cols);
                    for (size_t i = 0; i < m.rows(); ++i)
                        for (size_t j = 0; j < m.cols(); ++j)
                            diy::load(bb, m(i, j));
                }
        };

    template <typename T>
        struct Serialization<VectorX<T>>
        {
            static
                void save(diy::BinaryBuffer& bb, const VectorX<T>& v)
                {
                    diy::save(bb, v.size());
                    for (size_t i = 0; i < v.size(); ++i)
                        diy::save(bb, v(i));
                }
            static
                void load(diy::BinaryBuffer& bb, VectorX<T>& v)
                {
                    Index size;
                    diy::load(bb, size);
                    v.resize(size);
                    for (size_t i = 0; i < size; ++i)
                        diy::load(bb, v(i));
                }
        };

    template<>
        struct Serialization<VectorXi>
        {
            static
                void save(diy::BinaryBuffer& bb, const VectorXi& v)
                {
                    diy::save(bb, v.size());
                    for (size_t i = 0; i < v.size(); ++i)
                        diy::save(bb, v(i));
                }
            static
                void load(diy::BinaryBuffer& bb, VectorXi& v)
                {
                    Index size;
                    diy::load(bb, size);
                    v.resize(size);
                    for (size_t i = 0; i < size; ++i)
                        diy::load(bb, v.data()[i]);
                }
        };
}                       // namespace
