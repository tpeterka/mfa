/**--------------------------------------------------------------
 * example of encoding / decoding higher dimensional data w/ fixed number of control points and
 * multiple blocks with ghost zone overlap
 * will also blend at the overlapping regions, during decoding
 *  the blending will be computed at new discrete points, not necessarily the same grid as input points
 *  there are 2 rounds of communication between blocks, one to send the requested points to the neighbor blocks
 *  that contribute to the local values, and one to send back the computed values
 *
 *  blending will be done only for the local core, as in the fixed_multiblock example, but will be output
 *  at new array points
 * 
 *  Example is hard-coded for a specific s3d example
 */

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

#include "block.hpp"
#include "opts.h"
#include "example-setup.hpp"

using namespace std;

int main(int argc, char **argv) {
    // initialize MPI
    diy::mpi::environment env(argc, argv); // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    // initialize Kokkos if needed
#ifdef MFA_KOKKOS
    Kokkos::initialize( argc, argv );
#endif

    // default command line arguments
    // int pt_dim              = 4;        // dimension of input points
    int dom_dim             = 3;        // dimension of domain (<= pt_dim)
    int geom_degree         = 1;        // degree for geometry (same for all dims)
    int vars_degree         = 4;        // degree for science variables (same for all dims)
    int geom_nctrl          = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl  = {11};     // input number of control points for all science variables
    string infile           = "";       // input file for s3d data
    int weighted            = 0;        // solve for and use weights (bool 0 or 1)
    int strong_sc           = 0;        // strong scaling (bool 0 or 1, 0 = weak scaling)
    bool write_output       = false;
    int         verbose = 0;
    bool help               = false;    // show help

    int         adaptive        = 0;        // do analytical encode (0/1)
    real_t      e_threshold     = 1e-1;     // error threshold for adaptive fitting
    int         rounds          = 0;

    std::vector<int> overlaps = {2, 2, 2};        // ghosting in 3 directions; default is 2, 2, 2
    vector<int> resolutions = {20, 20, 20};


    int chunk = 3;
    int transpose = 0; // if 1, transpose data
    vector<int> nblocks = {2, 2, 3};
    vector<int> starts = {0, 0, 0};
    vector<int> ends = {549, 539, 703};
    vector<int> shape = {550, 540, 704};
    vector<int> model_dims = {3, 1};

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('m', "dom_dim",     dom_dim,        " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree,    " degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree,    " degree in each dimension of science variables");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl,     " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl,     " number of control points in each dimension of all science variables");
    ops >> opts::Option('f', "data_file",   infile,         " s3d data file location");
    ops >> opts::Option('b', "nblocks",     nblocks,        " number of blocks in each direction");
    ops >> opts::Option('x', "starts", starts, " start of block in each direction");
    ops >> opts::Option('y', "ends", ends, " end of block in second direction");
    ops >> opts::Option('D', "chunk", chunk, " number of values per geometric point");
    ops >> opts::Option('T', "transpose", transpose, " transpose input data ");
    ops >> opts::Option('o', "overlap", overlaps, " overlaps in 3 directions ");
    ops >> opts::Option('u', "resolutions", resolutions, " number of output points in each dimension of domain");
    ops >> opts::Option('t', "strong_sc",   strong_sc,      " strong scaling (1 = strong, 0 = weak)");
    ops >> opts::Option('W', "write", write_output, " write output file");
    ops >> opts::Option('h', "help", help, " show help");
    ops >> opts::Option('z', "adaptive",    adaptive,   " do adaptive encode (0/1)");
    ops >> opts::Option('e', "errorbound",  e_threshold," error threshold for adaptive encoding");
    ops >> opts::Option('z', "rounds",      rounds,     " max number of rounds for adaptive encoding");

    int pt_dim = dom_dim + 1;

    if (!ops.parse(argc, argv) || help) {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    if (starts.size() != 3 || ends.size() != 3 || shape.size() != 3)
    {
        if (world.rank() == 0)
            cerr << "ERROR: Incorrect size of starts, ends, or shape vector. Exiting" << endl;
        
        exit(1);
    }

    // Set up global domain information for DIY decomposition
    const int mem_blocks  = -1;     // Keep all blocks in memory
    const int num_threads = 1;      // Number of threads for DIY (serial for timing purposes)
    int tot_blocks = 1;
    vector<bool> share_face(dom_dim, true);
    vector<bool> wrap(dom_dim, false);
    Bounds<int> dom_bounds(3);
    for (int j = 0; j < 3; j++) {
        tot_blocks *= nblocks[j];
        dom_bounds.min[j] = starts[j];
        dom_bounds.max[j] = ends[j];

        // Must have at least one point per block in each direction
        if (ends[j] - starts[j] + 1 < nblocks[j]) {
            if (world.rank() == 0) {
                cerr << "ERROR: Number of blocks in " << j << " direction is too high. Exiting" << endl;
            }
            exit(1);
        }
    }
    // Warn if MPI ranks are nonuniformly assigned, in case we are timing
    if (tot_blocks % world.size() != 0) {
        if (world.rank() == 0) {
            cerr << "WARNING: Number of blocks not divisible the number of MPI processes." << endl;
        }
    }

    // There are 3 components to every point, so we scale the "shape" of the last dimension by 3
    shape[dom_dim - 1] *= chunk;

    // print input arguments
    if (world.rank() == 0)
    {
        echo_mfa_settings("multiblock blend discrete example", dom_dim, pt_dim, 1, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                            0, 0, adaptive, e_threshold, rounds);
        echo_data_settings("s3d_blend", infile, 0, 0);
    }

    // decide the actual dimension of the problem, looking at the starts and ends
    std::vector<int> mapDim;
    for (int i = 0; i < dom_dim; i++) {
        if (starts[i] < ends[i]) {
            mapDim.push_back(i);
        } else {
            if (world.rank() == 0)
                cerr << " direction " << i << " has no extension\n";
        }
    }
    if (world.rank() == 0) {
        cerr << " actual dimension of domain " << mapDim.size() << endl;
    }

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master master(world, num_threads, mem_blocks, &Block<real_t>::create,
            &Block<real_t>::destroy, &storage, &Block<real_t>::save,
            &Block<real_t>::load);
    diy::ContiguousAssigner assigner(world.size(), tot_blocks);

    // set up parameters for examples
    MFAInfo     mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        "s3d", infile, 0, 1, 0, 0, 0, 0, 0, 0, adaptive, verbose,
                        mfa_info, d_args);

    // Set multiblock options
    d_args.multiblock   = true;
    if (strong_sc) 
    {
        mfa_info.splitStrongScaling(nblocks);
    }

    // Print block layout and scaling info
    if (world.rank() == 0)
    {
        echo_multiblock_settings(mfa_info, d_args, world.size(), tot_blocks, nblocks, strong_sc, overlaps);
    }

    double start_reading = MPI_Wtime();
    Decomposer<int> decomposer(dom_dim, dom_bounds, tot_blocks, share_face, wrap, overlaps, nblocks);
    decomposer.decompose(world.rank(), assigner,
        [&](int gid, const Bounds<int> &core, const Bounds<int> &bounds, const Bounds<int> &domain, const RCLink<int> &link) 
        { Block<real_t>::readfile(gid, dom_dim, pt_dim, core, bounds, link, master, mapDim, infile, shape, chunk, transpose, mfa_info); });

    // compute the MFA
    if (world.rank() == 0)
        fprintf(stderr, "\nStarting fixed encoding...\n\n");
    world.barrier();                     // to synchronize timing
    
    double start_encode = MPI_Wtime();
    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) 
    {
        b->fixed_encode_block(cp, mfa_info);
    });
    world.barrier();                     // to synchronize timing
    double end_encode = MPI_Wtime();

    if (world.rank() == 0)
        fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    if (world.rank() == 0)
        fprintf(stderr, "\nFinal decoding and computing max. error...\n");

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp)
    {
        b->range_error(cp, true, false);
        b->print_brief_block(cp, true);
    });
    world.barrier();
    double end_decode = MPI_Wtime();

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->decode_core_ures(cp, resolutions);
    });
    world.barrier();
    double end_resolution_decode = MPI_Wtime();
    
    if (world.rank() == 0) {

        fprintf(stderr, "decomposing and reading time = %.3lf s.\n", start_encode - start_reading);
        fprintf(stderr, "encoding time                = %.3lf s.\n", end_encode - start_encode);
        fprintf(stderr, "decoding time                = %.3lf s.\n", end_decode - end_encode);
        fprintf(stderr, "decode at resolution         = %.3lf s.\n", end_resolution_decode - end_decode);
    }

    if (overlaps[0] == 0 && overlaps[1] == 0 && overlaps[2] == 0) // no overlap, stop
    {
        if (write_output) {
            diy::io::write_blocks("approx.mfa", world, master);
        }
        return 0;
    }

    // compute the neighbors encroachment
    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp)
    {
        b->compute_neighbor_overlaps(cp);
    });

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp)
    {
        b->decode_patches_discrete(cp, resolutions);
    });
    world.barrier();
    double end_decode_patches = MPI_Wtime();

    // do the actual data transmission, to send the computed values to the requesters
    master.exchange();
    world.barrier();
    double exchange_end = MPI_Wtime();

    // now receive the requested values and do blending
    master.foreach(&Block<real_t>::recv_and_blend);
    world.barrier();
    double recv_blend_end = MPI_Wtime();

    // compute maximum errors over all blocks; (a reduce operation)
    //   merge-based reduction: create the partners that determine how groups are formed
    //   in each round and then execute the reduction
    diy::RegularMergePartners partners(decomposer, 2, true);
    diy::reduce(master, assigner, partners, &max_err_cb);

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) 
    {
        b->print_brief_block(cp, true);
    });
    world.barrier();

    // save the results in diy format
    double write_time;
    if (write_output) {
        diy::io::write_blocks("approx.mfa", world, master);
        world.barrier();
        write_time = MPI_Wtime();
    }
    if (world.rank() == 0) {
        fprintf(stderr, "\n------- Final block results --------\n");
        master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp)
        {
            b->print_block(cp, true);
        }); // only blocks on master

        fprintf(stderr, "decode requests time   = %.3lf s.\n", end_decode_patches - end_resolution_decode);
        fprintf(stderr, "exchange time          = %.3lf s.\n", exchange_end - end_decode_patches);
        fprintf(stderr, "blend time             = %.3lf s.\n", recv_blend_end - exchange_end);
        fprintf(stderr, "blend total            = %.3lf s.\n", recv_blend_end - end_decode);
        if (write_output)
            fprintf(stderr, "write time             = %.3lf s.\n", write_time - recv_blend_end);
    }

    if (world.rank() == 0 && mapDim.size() == 3) {
        Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
        int blockMax = (int) b->max_errs_reduce[1];
        real_t max_red_err = b->max_errs_reduce[0];
        if (blockMax != 5 || fabs(max_red_err - 0.591496) > 1.e-5) {
            std::cout << "expected blockMax == 5 got " << blockMax
                    << " expected max_red_err == 0.591496 got : " << max_red_err
                    << "\n";
            abort();
        }
    }
#ifdef MFA_KOKKOS
    Kokkos::finalize();
#endif
}

