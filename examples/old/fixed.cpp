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
#include "parser.hpp"
#include "example-setup.hpp"

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD


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
    opts.echo_mfa_settings("fixed example");
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
    set_dom_bounds(dom_bounds, input);
    
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    // Different examples can reset this below
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
    MFAInfo     mfa_info(dom_dim, opts.verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    opts.setup_args(model_dims, mfa_info, d_args);

    // Create data set for modeling. Input keywords are defined in example-setup.hpp
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
        // for testing, hard-code a subset of a 3d domain, 1/2 the size in each dim and centered
        // master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        //         { b->read_3d_subset_3d_vector_data(cp, d_args); });
    }
    else if (datasets_2d.count(input) == 1)
    {
        if (dom_dim != 2)
        {
            fprintf(stderr, "\'%s\' data only available with dimension 2\n", input);
            exit(0);
        }

        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_2d_scalar_data(cp, mfa_info, d_args); 
        });
    }
    else if (datasets_unstructured.count(input) == 1)
    {
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_3d_unstructured_data(cp, mfa_info, d_args); 
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
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
        bool saved_basis = opts.structured; // TODO: basis functions are currently only saved during encoding of structured data
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->range_error(cp, true, saved_basis);
        });
#endif
        decode_time = MPI_Wtime() - decode_time;
    }
    else if (opts.decode_grid.size() == dom_dim)
    {
        fprintf(stderr, "\nDecoding on regular grid of size %s\n", mfa::print_vec(opts.decode_grid).c_str());
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        {
            b->decode_block_grid(cp, opts.decode_grid);
        });
        decode_time = MPI_Wtime() - decode_time;
    }

    // debug: write original and approximated data for reading into z-checker
    // only for one block (one file name used, ie, last block will overwrite earlier ones)
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//             { b->write_raw(cp); });

    // debug: save knot span domains for comparing error with location in knot span
//     master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
//             { b->knot_span_domains(cp); });

    // compute the norms of analytical errors synthetic function w/o noise at different domain points than the input
    if (opts.ntest > 0)
    {
        cerr << "Computing analytical error" << endl;
        int nvars = model_dims.size() - 1;
        vector<real_t> L1(nvars), L2(nvars), Linf(nvars);                                // L-1, 2, infinity norms
        vector<int> grid_size(dom_dim, opts.ntest);
        mfa::PointSet<real_t>* temp_in = nullptr;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->analytical_error_field(cp, grid_size, input, L1, L2, Linf, d_args, temp_in, b->approx, b->errs); });

        // print analytical errors
        for (int i = 0; i < nvars; i++)
        {
            fprintf(stderr, "\n------ Analytical error: Var %i -------\n", i);
            fprintf(stderr, "L-1        norm = %e\n", L1[i]);
            fprintf(stderr, "L-2        norm = %e\n", L2[i]);
            fprintf(stderr, "L-infinity norm = %e\n", Linf[i]);
            fprintf(stderr, "-------------------------------------\n\n");
        }
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
}
