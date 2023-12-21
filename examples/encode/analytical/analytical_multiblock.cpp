//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points and
// multiple blocks with ghost zone overlap
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

    // default number of global blocks is world.size(), unless set by user
    int tot_blocks  = opts.tot_blocks > 1 ? opts.tot_blocks : world.size();             
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    int dom_dim = opts.dom_dim;
    int pt_dim = opts.pt_dim;
    string input = opts.input;
    opts.echo_mfa_settings("fixed multiblock example");
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

    // set global domain bounds
    Bounds<real_t> dom_bounds(dom_dim);
    set_dom_bounds(dom_bounds, input);

    // decompose the domain into blocks
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { Block<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, opts.ghost); });

    vector<int> divs(dom_dim);                          // number of blocks in each dimension
    decomposer.fill_divisions(divs);

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

    // Adjust parameters for strong scaling if needed
    d_args.multiblock   = true;
    if (opts.strong_sc) 
    {
        mfa_info.splitStrongScaling(divs);
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.ndom_pts[i] /= divs[i];
        }
    }

    // Print block layout and scaling info
    if (world.rank() == 0)
    {
        opts.echo_multiblock_settings(mfa_info, d_args, world.size(), tot_blocks, divs, opts.strong_sc, opts.ghost);
    }

    // Generate data
    world.barrier();
    if (analytical_signals.count(input) == 1)
    {
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->generate_analytical_data(cp, input, mfa_info, d_args); 
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
    }

    // compute the MFA
    if (world.rank() == 0)
        fprintf(stderr, "\nStarting fixed encoding...\n\n");
    world.barrier();                     // to synchronize timing
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, mfa_info); });
    world.barrier();                     // to synchronize timing
    encode_time = MPI_Wtime() - encode_time;
    if (world.rank() == 0)
        fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    double decode_time = MPI_Wtime();
    if (opts.error)
    {
        if (world.rank() == 0)
            fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#ifdef CURVE_PARAMS     // normal distance
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->error(cp, 0, true); });
#else                   // range coordinate difference
        bool saved_basis = opts.structured; // TODO: basis functions are currently only saved during encoding of structured data
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->range_error(cp, true, saved_basis); });
#endif
        decode_time = MPI_Wtime() - decode_time;
    }

    // compute the norms of analytical errors synthetic function w/o noise at different domain points than the input
    if (opts.ntest > 0)
    {
        int nvars = model_dims.size() - 1;
        vector<real_t> L1(nvars), L2(nvars), Linf(nvars);                                // L-1, 2, infinity norms
        vector<int> grid_size(dom_dim, opts.ntest);
        mfa::PointSet<real_t>* temp_in = nullptr;
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->analytical_error_field(cp, grid_size, input, L1, L2, Linf, d_args, temp_in, b->approx, b->errs); });

        // print analytical errors
        fmt::print(stderr, "\n------ Analytical error norms -------\n");
        fmt::print(stderr, "L-1        norms[vars] = {:e}\n", fmt::join(L1, ","));
        fmt::print(stderr, "L-2        norms[vars] = {:e}\n", fmt::join(L2, ","));
        fmt::print(stderr, "L-infinity norms[vars] = {:e}\n", fmt::join(Linf, ","));
        fmt::print(stderr, "-------------------------------------\n\n");
    }

    // print block results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, opts.error); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    if (opts.error)
        fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // print overall timing results
    if (world.rank() == 0)
        fprintf(stderr, "\noverall encoding time = %.3lf s.\n", encode_time);

    // save the results in diy format
    diy::io::write_blocks("approx.mfa", world, master);
}
