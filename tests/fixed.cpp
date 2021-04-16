//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points and a
// single block in a split model w/ one model containing geometry and other model science variables
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <mfa/util.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include "opts.h"

#include "block.hpp"

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                        // number of local blocks
    int tot_blocks  = nblocks * world.size();   // number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int    pt_dim       = 3;                    // dimension of input points
    int    dom_dim      = 2;                    // dimension of domain (<= pt_dim)
    int    geom_degree  = 1;                    // degree for geometry (same for all dims)
    int    vars_degree  = 4;                    // degree for science variables (same for all dims)
    int    ndomp        = 100;                  // input number of domain points (same for all dims)
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    int    vars_nctrl   = 11;                   // input number of control points for all science variables (same for all dims)
    string input        = "sine";               // input dataset
    int    weighted     = 1;                    // solve for and use weights (bool 0 or 1)
    real_t rot          = 0.0;                  // rotation angle in degrees
    real_t twist        = 0.0;                  // twist (waviness) of domain (0.0-1.0)
    int    error        = 1;                    // decode all input points and check error (bool 0 or 1)
    string infile;                              // input file name
    int    structured   = 1;                    // input format (structured grid if nonzero)
    bool   help         = false;                // show help
    int    resolutionGrid = 0;

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('r', "rotate",      rot,        " rotation angle of domain in degrees");
    ops >> opts::Option('t', "twist",       twist,      " twist (waviness) of domain (0.0-1.0)");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('u', "resolution",  resolutionGrid,    " resolution for grid test ");
    ops >> opts::Option('x', "structured",  structured, " parse input data from a structured (possibly curvilinear) grid");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // minimal number of geometry control points if not specified
    if (geom_nctrl == -1)
        geom_nctrl = geom_degree + 1;

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "pt_dim = "         << pt_dim       << " dom_dim = "        << dom_dim      <<
        "\ngeom_degree = "  << geom_degree  << " vars_degree = "    << vars_degree  <<
        "\ninput pts = "    << ndomp        << " geom_ctrl pts = "  << geom_nctrl   <<
        "\nvars_ctrl_pts = "<< vars_nctrl   << " input = "          << input        << 
        "\nstructured = "   << structured   << endl;
#ifdef CURVE_PARAMS
    cerr << "parameterization method = curve" << endl;
#else
    cerr << "parameterization method = domain" << endl;
#endif
#ifdef MFA_NO_TBB
    cerr << "TBB: off" << endl;
#else
    cerr << "TBB: on" << endl;
#endif
#ifdef MFA_NO_WEIGHTS
    cerr << "weighted = 0" << endl;
#else
    cerr << "weighted = " << weighted << endl;
#endif
    fprintf(stderr, "-------------------------------------\n\n");

    // initialize DIY
    diy::FileStorage          storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &Block<real_t>::create,
                                     &Block<real_t>::destroy,
                                     &storage,
                                     &Block<real_t>::save,
                                     &Block<real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

    // even though this is a single-block example, we want diy to do a proper decomposition with a link
    // so that everything works downstream (reading file with links, e.g.)
    // therefore, set some dummy global domain bounds and decompose the domain
    Bounds<real_t> dom_bounds(dom_dim);
    for (int i = 0; i < dom_dim; ++i)
    {
        dom_bounds.min[i] = 0.0;
        dom_bounds.max[i] = 1.0;
    }
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // set default args for diy foreach callback functions
    DomainArgs d_args(dom_dim, pt_dim);
    d_args.weighted     = weighted;
    d_args.multiblock   = false;
    d_args.verbose      = 1;
    d_args.structured   = structured;
    for (int i = 0; i < pt_dim - dom_dim; i++)
        d_args.f[i] = 1.0;
    for (int i = 0; i < dom_dim; i++)
    {
        d_args.geom_p[i]    = geom_degree;
        d_args.vars_p[0][i] = vars_degree;      // assuming one science variable, vars_p[0]
        d_args.ndom_pts[i]  = ndomp;
    }

    // initilize input data

    // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
    if (input == "sine")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
            d_args.geom_nctrl_pts[i]    = geom_nctrl;
            d_args.vars_nctrl_pts[0][i] = vars_nctrl;               // assuming one science variable, vars_nctrl_pts[0]
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = i + 1;                        // scaling factor on range
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
            d_args.geom_nctrl_pts[i]    = geom_nctrl;
            d_args.vars_nctrl_pts[0][i] = vars_nctrl;               // assuming one science variable, vars_nctrl_pts[0]
        }
        for (int i = 0; i < pt_dim - dom_dim; i++)      // for all science variables
            d_args.s[i] = 10.0 * (i + 1);                 // scaling factor on range
        d_args.r = rot * M_PI / 180.0;   // domain rotation angle in rads
        d_args.t = twist;                // twist (waviness) of domain
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->generate_analytical_data(cp, input, d_args); });
    }

    // S3D dataset
    if (input == "s3d")
    {
        d_args.ndom_pts[0]          = 704;
        d_args.ndom_pts[1]          = 540;
        d_args.ndom_pts[2]          = 550;
        d_args.geom_nctrl_pts[0]    = geom_nctrl;
        d_args.geom_nctrl_pts[1]    = geom_nctrl;
        d_args.geom_nctrl_pts[2]    = geom_nctrl;
        d_args.vars_nctrl_pts[0][0] = 140;
        d_args.vars_nctrl_pts[0][1] = 108;
        d_args.vars_nctrl_pts[0][2] = 110;
        d_args.infile               = infile;
//         d_args.infile               = "/Users/tpeterka/datasets/flame/6_small.xyz";
        if (dom_dim == 1)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_1d_slice_3d_vector_data(cp, d_args); });
        else if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, d_args); });
        else
        {
            fprintf(stderr, "S3D data only available in 2 or 3d domain\n");
            exit(0);
        }
    }

    // nek5000 dataset
    if (input == "nek")
    {
        for (int i = 0; i < 3; i++)
        {
            d_args.ndom_pts[i]          = 200;
            d_args.geom_nctrl_pts[i]    = geom_nctrl;
            d_args.vars_nctrl_pts[0][i] = vars_nctrl;               // assuming one science variable, vars_nctrl_pts[0]
        }
        d_args.infile = infile;
//         d_args.infile = "/Users/tpeterka/datasets/nek5000/200x200x200/0.xyz";
        if (dom_dim == 2)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_2d_slice_3d_vector_data(cp, d_args); });
        else if (dom_dim == 3)
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                    { b->read_3d_vector_data(cp, d_args); });
        else
        {
            fprintf(stderr, "nek5000 data only available in 2 or 3d domain\n");
            exit(0);
        }
    }

    // compute the MFA

    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, d_args); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    double decode_time = MPI_Wtime();
#ifdef CURVE_PARAMS     // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, 1, true); });
#else                   // range coordinate difference
    bool saved_basis = structured; // TODO: basis functions are currently only saved during encoding of structured data
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, 1, true, true); });
#endif
    decode_time = MPI_Wtime() - decode_time;

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, error); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    if (error)
        fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.out", world, master);

    // check the results of the last (only) science variable
    Block<real_t>* b    = static_cast<Block<real_t>*>(master.block(0));
    real_t range_extent = b->input->domain.col(dom_dim).maxCoeff() - b->input->domain.col(dom_dim).minCoeff();
    real_t err_factor   = 1.0e-3;
    real_t expect_err   = -0.0;
    // for ./fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0
    if (input == "sinc" && dom_dim == 2)
        expect_err   = 4.304489e-4;
    // for ./fixed-test -i s3d -d 2 -m 1 -p 1 -q 3 -w 0
    if (input == "s3d" && dom_dim == 1)
        expect_err   = 6.819451e-2;
    // for ./fixed-test -i s3d -d 3 -m 2 -p 1 -q 3 -w 0
    if (input == "s3d" && dom_dim == 2)
        expect_err   = 2.778071e-1;
    real_t our_err      = b->max_errs[0] / range_extent;    // normalized max_err
    if (fabs(expect_err - our_err) / expect_err > err_factor)
    {
        fprintf(stderr, "our error (%e) and expected error (%e) differ by more than a factor of %e\n", our_err, expect_err, err_factor);
        abort();
    }

    if (resolutionGrid > 0 && input == "sinc" && dom_dim == 2) {
        // do an extra test for decode grid
        VectorXi ndom_pts;
        std::vector<int> counts;
        ndom_pts.resize(2);
        ndom_pts[0] = resolutionGrid;
        ndom_pts[1] = resolutionGrid;
        counts.push_back(resolutionGrid); counts.push_back(resolutionGrid);
        master.foreach( [&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
                    b->decode_core_ures(cp, counts); });
        // now look at some values of the blend matrix

        // b still points to the first block !! Block<real_t>* b    = static_cast<Block<real_t>*>(master.block(0));
        MatrixX<real_t> result = b->blend;

        // evaluate at 0,0 using decodeatpoint
        VectorX<real_t> param(2); // dom dim is 2, initialize with 0
        VectorX<real_t> var_cpt(1);
        // loop over all points in the resulted grid, and compare with the DecodePt
        // we have 2 dimensions, each direction has resolutionGrid points

        mfa::VolIterator vol_it(ndom_pts);
        while (!vol_it.done()) {
            int jj = (int) vol_it.cur_iter();
            for (auto ii = 0; ii < 2; ii++) {
                int ix = vol_it.idx_dim(ii); // index along direction ii in grid
                param[ii] = ix / (resolutionGrid - 1.);
            }
            b->mfa->DecodePt(*(b->vars[0].mfa_data), param, var_cpt);
            // compare with our blend result
            if (fabs(var_cpt(0) - result(jj, 2)) > 1.e-10) {
                fprintf(stderr, " %e != %e , params: %f %f, ix: %d %d \n",
                        var_cpt(0), result(jj, 2), param[0], param[1],
                        vol_it.idx_dim(0), vol_it.idx_dim(1));
                abort();
            }
            vol_it.incr_iter();
        }

    }
}
