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

    // command line arguments with default values
    int         dom_dim         = 3;                // dimension of domain (<= pt_dim)
    int         vars_degree     = 4;                // degree for science variables (same for all dims)
    vector<int> vars_nctrl      = {11};             // input number of control points for all science variables
    string      infile          = "";               // input file for s3d data
    int         strong_sc       = 0;                // strong scaling (bool 0 or 1, 0 = weak scaling)
    int         adaptive        = 0;                // do analytical encode (0/1)
    real_t      e_threshold     = 1e-1;             // error threshold for adaptive fitting
    int         rounds          = 0;                // number of adaptive fitting rounds
    vector<int> nblocks         = {2, 2, 3};        // number of DIY blocks
    vector<int> starts          = {0, 0, 0};        // starting indices for data read
    vector<int> ends            = {703, 539, 549};  // ending indices for data read (inclusive)
    vector<int> overlaps        = {2, 2, 2};        // number of ghost points in each direction
    vector<int> resolutions     = {20, 20, 20};     // custom grid resolution for decoding
    bool        write_output    = true;             // flag whether to write a .mfa file
    int         verbose         = 0;                // enable verbose output on Block 0
    bool        help            = false;            // show help
    
    // These values do not need to change for this example
    int         geom_degree         = 1;        // degree for geometry (same for all dims)
    int         geom_nctrl          = -1;       // input number of control points for geometry (same for all dims)
    int         weighted            = 0;        // solve for and use weights (bool 0 or 1)

    // These values are known ahead of time for the sample S3D data
    int         vecSize = 3;
    bool        fileOrderC = false;
    vector<int> shape = {704, 540, 550};
    vector<int> model_dims = {3, 1};

    // set up command line arguments
    opts::Options ops;
    ops >> opts::Option('m', "dom_dim",     dom_dim,        " dimension of domain");
    ops >> opts::Option('q', "vars_degree", vars_degree,    " degree in each dimension of science variables");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl,     " number of control points in each dimension of all science variables");
    ops >> opts::Option('f', "infile",      infile,         " s3d data file location");
    ops >> opts::Option('t', "strong_sc",   strong_sc,      " strong scaling (1 = strong, 0 = weak)");
    ops >> opts::Option('z', "adaptive",    adaptive,       " do adaptive encode (0/1)");
    ops >> opts::Option('z', "errorbound",  e_threshold,    " error threshold for adaptive encoding");
    ops >> opts::Option('z', "rounds",      rounds,         " max number of rounds for adaptive encoding");
    ops >> opts::Option('b', "nblocks",     nblocks,        " number of blocks in each direction");
    ops >> opts::Option('x', "starts",      starts,         " start of block in each direction");
    ops >> opts::Option('y', "ends",        ends,           " end of block in second direction");
    ops >> opts::Option('o', "overlap",     overlaps,       " overlaps in 3 directions ");
    ops >> opts::Option('u', "resolutions", resolutions,    " number of output points in each dimension of domain");
    ops >> opts::Option('T', "fileOrderC",  fileOrderC,     " flag for if input data file was written in C ordering or not");
    ops >> opts::Option('W', "write",       write_output,   " write output file");
    ops >> opts::Option('z', "verbose",     verbose,        " allow verbose output");
    ops >> opts::Option('h', "help",        help,           " show help");

    if (!ops.parse(argc, argv) || help) {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }
    int pt_dim = dom_dim + 1;   

    // print input arguments
    if (world.rank() == 0)
    {
        echo_mfa_settings("multiblock blend discrete example", dom_dim, pt_dim, 1, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                            0, 0, adaptive, e_threshold, rounds);
        echo_data_settings("s3d_blend", infile, 0, 0);
    }

    // Set up global domain information for DIY decomposition
    const int mem_blocks  = -1;     // Keep all blocks in memory
    const int num_threads = 1;      // Number of threads for DIY (serial for timing purposes)
    int tot_blocks = 1;
    vector<bool> share_face(3, true);
    vector<bool> wrap(3, false);
    Bounds<int> dom_bounds(3);
    if (starts.size() != 3 || ends.size() != 3 || shape.size() != 3)
    {
        if (world.rank() == 0)
            cerr << "ERROR: Incorrect size of starts, ends, or shape vector. Exiting" << endl;
        
        exit(1);
    }
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

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master master(world, num_threads, mem_blocks, &Block<real_t, int>::create,
            &Block<real_t, int>::destroy, &storage, &Block<real_t, int>::save,
            &Block<real_t, int>::load);
    diy::ContiguousAssigner assigner(world.size(), tot_blocks);

    // Create domain decomposition with DIY
    double start_reading = MPI_Wtime();
    Decomposer<int> decomposer(3, dom_bounds, tot_blocks, share_face, wrap, overlaps, nblocks);
    decomposer.decompose(world.rank(), assigner,
        [&](int gid, const Bounds<int> &core, const Bounds<int> &bounds, const Bounds<int> &domain, const RCLink<int> &link) 
        { 
            Block<real_t, int>::add_int(gid, core, bounds, domain, link, master, dom_dim, pt_dim);
        });

    // set up parameters for examples
    mfa::MFAInfo    mfa_info(dom_dim, verbose);
    DomainArgs      d_args(dom_dim, model_dims);
    d_args.multiblock   = true;
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        "s3d", infile, 0, 1, 0, 0, 0, 0, 0, 0, adaptive, verbose,
                        mfa_info, d_args);

    // Set multiblock options
    if (strong_sc) 
    {
        mfa_info.splitStrongScaling(nblocks);
    }

    // Print block layout and scaling info
    if (world.rank() == 0)
    {
        echo_multiblock_settings(mfa_info, d_args, world.size(), tot_blocks, nblocks, strong_sc, overlaps);
    }

    // Read the data
    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp) 
    {
        b->read_box_data_3d(cp, infile, shape, fileOrderC, vecSize, mfa_info);
    });
    world.barrier();

    // Compute the MFA
    double start_encode = MPI_Wtime();
    if (world.rank() == 0)
        fprintf(stderr, "Starting fixed encoding...\n");

    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp) 
    {
        b->fixed_encode_block(cp, mfa_info);
    });
    world.barrier();
    if (world.rank() == 0)
        fprintf(stderr, "Fixed encoding done.\n");
    double end_encode = MPI_Wtime();

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    if (world.rank() == 0)
        fprintf(stderr, "Final decoding and computing max. error...\n");

    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp)
    {
        b->range_error(cp, true, false);
    });
    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp)
    {
        b->print_brief_block(cp);
    });
    world.barrier();
    double end_decode = MPI_Wtime();

    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp) {
        b->decode_core_ures(cp, resolutions);
    });
    world.barrier();
    double end_resolution_decode = MPI_Wtime();
    
    if (world.rank() == 0)
    {
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
    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp)
    {
        b->compute_neighbor_overlaps(cp);
    });

    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp)
    {
        b->decode_patches(cp, resolutions);
    });
    world.barrier();
    double end_decode_patches = MPI_Wtime();

    // do the actual data transmission, to send the computed values to the requesters
    master.exchange();
    world.barrier();
    double exchange_end = MPI_Wtime();

    // now receive the requested values and do blending
    master.foreach(&Block<real_t, int>::recv_and_blend);
    world.barrier();
    double recv_blend_end = MPI_Wtime();

    // compute maximum errors over all blocks; (a reduce operation)
    //   merge-based reduction: create the partners that determine how groups are formed
    //   in each round and then execute the reduction
    diy::RegularMergePartners partners(decomposer, 2, true);
    diy::reduce(master, assigner, partners, &max_err_cb<int>);

    master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp) 
    {
        b->print_brief_block(cp);
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
        master.foreach([&](Block<real_t, int> *b, const diy::Master::ProxyWithLink &cp)
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
    
#ifdef MFA_KOKKOS
    Kokkos::finalize();
#endif
}

