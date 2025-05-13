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

#include "parser.hpp"
#include "block.hpp"
#include "example-setup.hpp"

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD
#ifdef MFA_KOKKOS
    Kokkos::initialize( argc, argv );
#endif

    MFAParser opts;
    bool proceed = opts.parse_input(argc, argv);
    if (!proceed)
    {
        if (world.rank() == 0)
            std::cout << opts.ops;
        return 1;
    }

    int tot_blocks = opts.tot_blocks;
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    int dom_dim = opts.dom_dim;
    int pt_dim = opts.pt_dim;
    string input = opts.input;

    // print input arguments
    opts.echo_mfa_settings("fixed test");
    opts.echo_all_data_settings();
    
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

    // set global domain bounds and decompose
    Bounds<real_t> dom_bounds(dom_dim);
    set_dom_bounds(dom_bounds, opts.input);

    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    vector<int> model_dims;
    if (opts.scalar) // Set up (pt_dim - dom_dim) separate scalar variables
    {
        model_dims.assign(pt_dim - dom_dim + 1, 1);
        model_dims[0] = dom_dim;                        // index 0 == geometry
    }
    else    // Set up a single vector-valued variable
    {   
        model_dims = {dom_dim, pt_dim - dom_dim};
    }

    // set up parameters for examples
    mfa::MFAInfo    mfa_info(dom_dim, opts.verbose);
    DomainArgs      d_args(dom_dim, model_dims);
    opts.setup_args(model_dims, mfa_info, d_args);

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
            fprintf(stderr, "\'%s\' data only available with dimension <= 3\n", input.c_str());
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
    if (opts.error)
    {
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#ifdef CURVE_PARAMS     // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, 1, true); });
#else                   // range coordinate difference
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->range_error(cp, true); });
#endif
    decode_time = MPI_Wtime() - decode_time;
    }

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, opts.error); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    if (opts.error)
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
    if (opts.input == "sinc" && dom_dim == 2 && opts.rand_seed == -1)
        expect_err   = 4.304489e-4;
    // for ./fixed-test -i sinc -d 3 -m 2 -p 1 -q 5 -v 20 -w 0 -x 0 -y 4444
    if (opts.input == "s3d" && dom_dim == 1 && opts.rand_seed == 4444)
        expect_err   = 4.282089e-04;
    // for ./fixed-test -i s3d -d 2 -m 1 -p 1 -q 3 -w 0
    if (opts.input == "s3d" && dom_dim == 1 && opts.rand_seed == -1)
        expect_err   = 6.819451e-2;
    // for ./fixed-test -i s3d -d 3 -m 2 -p 1 -q 3 -w 0
    if (opts.input == "s3d" && dom_dim == 2)
        expect_err   = 2.778071e-1;
    real_t our_err      = b->max_errs[0] / range_extent;    // normalized max_err
    if (fabs(expect_err - our_err) / expect_err > err_factor)
    {
        fprintf(stderr, "our error (%e) and expected error (%e) differ by more than a factor of %e\n", our_err, expect_err, err_factor);
        abort();
    }

    if (opts.decode_grid.size() > 0 && opts.input == "sinc" && dom_dim == 2) {
        if (opts.decode_grid.size() != 2 || opts.decode_grid[0] != opts.decode_grid[1])
        {
            cerr << "Error in decode_grid setup" << endl;
            abort();
        }
        // do an extra test for decode grid
        VectorXi ndom_pts;
        std::vector<int> counts;
        ndom_pts.resize(2);
        ndom_pts[0] = opts.decode_grid[0];
        ndom_pts[1] = opts.decode_grid[1];
        counts.push_back(opts.decode_grid[0]); 
        counts.push_back(opts.decode_grid[1]);
        master.foreach( [&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
                    b->decode_core_ures(cp, counts); });
        // now look at some values of the blend matrix

        // b still points to the first block !! Block<real_t>* b    = static_cast<Block<real_t>*>(master.block(0));
        MatrixX<real_t> result = b->blend->domain;

        // evaluate at 0,0 using decodeatpoint
        VectorX<real_t> param(2); // dom dim is 2, initialize with 0
        VectorX<real_t> var_cpt(1);
        // loop over all points in the resulted grid, and compare with the DecodePt
        // we have 2 dimensions, each direction has decode_grid[i] points

        mfa::VolIterator vol_it(ndom_pts);
        while (!vol_it.done()) {
            int jj = (int) vol_it.cur_iter();
            for (auto ii = 0; ii < 2; ii++) {
                int ix = vol_it.idx_dim(ii); // index along direction ii in grid
                param[ii] = ix / (opts.decode_grid[0] - 1.);
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
#ifdef MFA_KOKKOS
    Kokkos::finalize();
#endif
}
