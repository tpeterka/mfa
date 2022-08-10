//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points and a
// single block in a split model w/ one model containing geometry and other model science variables
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <set>

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include "opts.h"

#include "block.hpp"
#include "example-setup.hpp"

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
    vector<int> vars_nctrl   = {11};            // input number of control points for all science variables (default same for all dims)
    string input        = "sine";               // input dataset
    int    weighted     = 1;                    // solve for and use weights (bool 0/1)
    real_t rot          = 0.0;                  // rotation angle in degrees
    real_t twist        = 0.0;                  // twist (waviness) of domain (0.0-1.0)
    int    error        = 1;                    // decode all input points and check error (bool 0/1)
    string infile;                              // input file name
    int    structured   = 1;                    // input data format (bool 0/1)
    int    rand_seed    = -1;                   // seed to use for random data generation (-1 == no randomization)
    int    resolutionGrid = 0;
    bool   help         = false;                // show help

    // Define list of test keywords
    set<string> analytical_signals = {"sine", "cosine", "sinc", "psinc1", "psinc2", "psinc3", "ml", "f16", "f17", "f18"};
    set<string> datasets_3d = {"s3d", "nek", "rti", "miranda", "tornado"};

    // Constants for this example
    const bool    adaptive        = false;
    const real_t  noise           = 0;
    const int     ntest           = 0;
    const int     scalar          = 1;
    const int     verbose         = 1;
    const int     reg1and2        = 0;
    const real_t  regularization  = 0;


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
    ops >> opts::Option('c', "error",       error,      " decode entire error field (default=true)");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('u', "resolution",  resolutionGrid,    " resolution for grid test ");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // print input arguments
    echo_args("fixed test", pt_dim, dom_dim, scalar, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                ndomp, ntest, input, infile, analytical_signals, noise, structured, weighted, adaptive, 0, 0);
    
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

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    vector<int> model_dims;
    if (scalar) // Set up (pt_dim - dom_dim) separate scalar variables
    {
        model_dims.assign(pt_dim - dom_dim + 1, 1);
        model_dims[0] = dom_dim;                        // index 0 == geometry
    }
    else    // Set up a single vector-valued variable
    {   
        model_dims = {dom_dim, pt_dim - dom_dim};
    }

    // Create empty info classes
    MFAInfo     mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);

    // set up parameters for examples
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                input, infile, ndomp, structured, rand_seed, rot, twist, noise,
                weighted, reg1and2, regularization, adaptive, verbose, mfa_info, d_args);

    // Create data set for modeling
    if (analytical_signals.count(input) == 1)
    {
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->generate_analytical_data(cp, input, mfa_info, d_args); 
        });
    }
    else if (datasets_3d.count(input) == 1)
    {
        if (dom_dim > 3)
        {
            fprintf(stderr, "\'%s\' data only available with dimension <= 3\n", input);
            exit(0);
        }

        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            if (dom_dim == 1) b->read_1d_slice_3d_vector_data(cp, mfa_info, d_args);
            if (dom_dim == 2) b->read_2d_slice_3d_vector_data(cp, mfa_info, d_args);
            if (dom_dim == 3) b->read_3d_vector_data(cp, mfa_info, d_args);
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        abort();
    }

    // compute the MFA

    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, mfa_info); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    double decode_time = MPI_Wtime();
    if (error)
    {
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#ifdef CURVE_PARAMS     // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, 1, true); });
#else                   // range coordinate difference
    bool saved_basis = structured; // TODO: basis functions are currently only saved during encoding of structured data
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, 1, true, saved_basis); });
#endif
    decode_time = MPI_Wtime() - decode_time;
    }

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, error); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    if (error)
        fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.mfa", world, master);

    // check the results of the last (only) science variable
    Block<real_t>* b    = static_cast<Block<real_t>*>(master.block(0));
    real_t range_extent = b->input->domain.col(dom_dim).maxCoeff() - b->input->domain.col(dom_dim).minCoeff();
    real_t err_factor   = 1.0e-3;
    real_t expect_err   = -0.0;
    // for ./fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0
    if (input == "sinc" && dom_dim == 2 && rand_seed == -1)
        expect_err   = 4.304489e-4;
    // for ./fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0 -x 0 -y 4444
    if (input == "s3d" && dom_dim == 1 && rand_seed == 4444)
        expect_err   = 4.282089e-04;
    // for ./fixed-test -i s3d -d 2 -m 1 -p 1 -q 3 -w 0
    if (input == "s3d" && dom_dim == 1 && rand_seed == -1)
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
            b->mfa->DecodeVar(0, param, var_cpt);
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
