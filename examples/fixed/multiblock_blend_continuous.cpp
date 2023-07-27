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
    diy::mpi::environment env(argc, argv);  // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;           // equivalent of MPI_COMM_WORLD

    int tot_blocks = world.size();          // default number of global blocks
    int mem_blocks = -1;                    // everything in core for now
    int num_threads = 1;                    // needed in order to do timing

    // default command line arguments
    int pt_dim              = 3;        // dimension of input points
    int dom_dim             = 2;        // dimension of domain (<= pt_dim)
    int geom_degree         = 1;        // degree for geometry (same for all dims)
    int vars_degree         = 4;        // degree for science variables (same for all dims)
    int ndomp               = 100;      // input number of domain points (same for all dims)
    vector<int> resolutions = {120};    // output points resolution
    int geom_nctrl          = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl  = {11};     // input number of control points for all science variables
    string input            = "sine";   // input dataset
    int weighted            = 0;        // solve for and use weights (bool 0 or 1)
    int strong_sc           = 0;        // strong scaling (bool 0 or 1, 0 = weak scaling)
    real_t ghost            = 0.1;      // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    bool write_output       = false;
    int error               = 1;        // decode all input points and check error (bool 0 or 1)
    double noise            = 0;
    bool help               = false;    // show help

    const int structured = 1;
    const int rand_seed = -1;

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,         " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,        " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree,    " degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree,    " degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,          " number of input points in each dimension of domain");
    ops >> opts::Option('r', "resolution",  resolutions,    " number of output points in each dimension of domain");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl,     " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl,   " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,          " input dataset");
    ops >> opts::Option('w', "weights",     weighted,       " solve for and use weights");
    ops >> opts::Option('b', "tot_blocks",  tot_blocks,     " total number of blocks");
    ops >> opts::Option('s', "noise",       noise,          " fraction of noise (0.0 - 1.0)");
    ops >> opts::Option('t', "strong_sc",   strong_sc,      " strong scaling (1 = strong, 0 = weak)");
    ops >> opts::Option('o', "overlap",     ghost,          " relative ghost zone overlap (0.0 - 1.0)");
    ops >> opts::Option('W', "write",       write_output,   " write output file");
    ops >> opts::Option('h', "help",        help,           " show help");

    if (!ops.parse(argc, argv) || help) {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    if (world.rank() == 0)
    {
        echo_mfa_settings("multiblock blend continuous", dom_dim, pt_dim, 1, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                            0, 0, 0, 0, 0);
        echo_data_settings(input, "", ndomp, resolutions);
        echo_data_mod_settings(structured, rand_seed, 0, 0, noise);
    }

    // initialize Kokkos if needed
#ifdef MFA_KOKKOS
    Kokkos::initialize( argc, argv );
#endif

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master master(world, num_threads, mem_blocks, &Block<real_t>::create,
            &Block<real_t>::destroy, &storage, &Block<real_t>::save,
            &Block<real_t>::load);
    diy::ContiguousAssigner assigner(world.size(), tot_blocks);

    // set global domain bounds
    Bounds<real_t> dom_bounds(dom_dim);
    for (int i = 0; i < dom_dim; ++i) {
        dom_bounds.min[i] = -10.0; // easier to debug
        dom_bounds.max[i] = 10.0;
    }

    // decompose the domain into blocks
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(), assigner,
            [&](int gid, const Bounds<real_t> &core,
                    const Bounds<real_t> &bounds, const Bounds<real_t> &domain,
                    const RCLink<real_t> &link) {
                Block<real_t>::add(gid, core, bounds, domain, link, master,
                        dom_dim, pt_dim, ghost);
            });
    vector<int> divs(dom_dim);             // number of blocks in each dimension
    decomposer.fill_divisions(divs);

    int         verbose = 0;
    vector<int> model_dims(pt_dim - dom_dim + 1, 1);    // set each variable to be scalar
    model_dims[0] = dom_dim;                            // dimension of geometry model
    DomainArgs  d_args(dom_dim, model_dims);
    MFAInfo     mfa_info(dom_dim, verbose);

    setup_args( dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        input, "", ndomp, structured, rand_seed, 0, 0, noise, 0, 0, 0, verbose,
                        mfa_info, d_args);

    // Set multiblock options
    d_args.multiblock   = true;
    if (strong_sc) 
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
        echo_multiblock_settings(mfa_info, d_args, world.size(), tot_blocks, divs, strong_sc, ghost);
    }


    // If only one value for resolutions was parsed, assume it applies to all dims
    if (resolutions.size() == 1)
    {
        resolutions = vector<int>(dom_dim, resolutions[0]);
    }


    // initialize input data
    if (input == "sine") {
        for (int i = 0; i < pt_dim - dom_dim; i++)
            d_args.s[i] = i + 1;                      // scaling factor on range
    }
    else if (input == "sinc") {
        for (int i = 0; i < pt_dim - dom_dim; i++)
            d_args.s[i] = 10.0 * (i + 1);             // scaling factor on range
    }
    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) 
    {
        b->generate_analytical_data(cp, input, mfa_info, d_args);
    });

#ifdef MFA_KOKKOS
    Kokkos::Profiling::pushRegion("Encode");
#endif
    double start_encode = MPI_Wtime();

    // compute the MFA
    if (world.rank() == 0)
        fprintf(stderr, "\nStarting fixed encoding...\n\n");
    world.barrier();                     // to synchronize timing

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->fixed_encode_block(cp, mfa_info);
    });
    world.barrier();                     // to synchronize timing
    double end_encode = MPI_Wtime();
    if (world.rank() == 0)
        fprintf(stderr, "\n\nFixed encoding done.\n\n");

#ifdef MFA_KOKKOS
    Kokkos::Profiling::popRegion(); // "Encode"
    Kokkos::Profiling::pushRegion("InitialDecode");
#endif

    if (world.rank() == 0)
        fprintf(stderr, "\nFinal decoding and computing max. error...\n");
#ifdef CURVE_PARAMS     // normal distance
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->error(cp, 0, true); });
#else                   // range coordinate difference
    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->range_error(cp, true, true);
    });
#endif
    world.barrier();
    double end_decode = MPI_Wtime();

#ifdef MFA_KOKKOS
    Kokkos::Profiling::popRegion(); // "InitialDecode"
    Kokkos::Profiling::pushRegion("RefinedDecode");
#endif
    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->decode_core_ures(cp, resolutions);
    });

#ifdef MFA_KOKKOS
    Kokkos::Profiling::popRegion(); // "RefinedDecode"
#endif
    world.barrier();
    double end_resolution_decode = MPI_Wtime();

    // compute the neighbors encroachment
    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->compute_neighbor_overlaps(cp);
    });

    world.barrier();
    double end_compute_overlaps = MPI_Wtime();

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->decode_patches(cp, resolutions);
    });
    // do the actual data transmission, to send the computed values to the requesters
    double decode_patches_end = MPI_Wtime();
    master.exchange();
    world.barrier();
    double exchange_end = MPI_Wtime();
    // now receive the requested values and do blending
    master.foreach(&Block<real_t>::recv_and_blend);
    world.barrier();
    double final_blend_end = MPI_Wtime(); // merge-based reduction: create the partners that determine how groups are formed
    // in each round and then execute the reduction

    bool contiguous = true;
    // partners for merge over regular block grid
    diy::RegularMergePartners partners(decomposer,  // domain decomposition
            2,                                      // radix of k-ary reduction
            contiguous);                            // contiguous = true: distance doubling
                                                    // contiguous = false: distance halving

    // reduction
    diy::reduce(master,               // Master object
            assigner,                 // Assigner object
            partners,                 // RegularMergePartners object
            &max_err_cb);             // merge operator callback function

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->print_brief_block(cp, error);
    });

    // save the results in diy format
    double write_time;
    if (write_output) {
        diy::io::write_blocks("approx.mfa", world, master);
        world.barrier();
        write_time = MPI_Wtime();
    }
    if (world.rank() == 0) {
        // print block results
        fprintf(stderr, "\n------- Final block results --------\n");
        master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) 
        {
            b->print_block(cp, error);
        });

        fprintf(stderr, "encoding time                = %.3lf s.\n", end_encode - start_encode);
        fprintf(stderr, "decoding time                = %.3lf s.\n", end_decode - end_encode);
        fprintf(stderr, "decode at resolution         = %.3lf s.\n", end_resolution_decode - end_decode);
        fprintf(stderr, "decode patches               = %.3lf s.\n", decode_patches_end - end_resolution_decode );
        fprintf(stderr, "exchange time                = %.3lf s.\n", exchange_end - decode_patches_end);
        fprintf(stderr, "blend time                   = %.3lf s.\n", final_blend_end - exchange_end);
        fprintf(stderr, "blend total                  = %.3lf s.\n", final_blend_end - end_decode);
        if (write_output)
            fprintf(stderr, "write time                   = %.3lf s.\n", write_time - final_blend_end);
        fprintf(stderr, "-------------------------------------\n\n");
    }
    // add simple tests for multi_bc test cases
    // test 1: -b 2 -d 2 -m 1 -i sinc -o 0.11
    // tot_blocks == 2, dom_dim == 1
    if (world.size() == 1 && tot_blocks == 2 && dom_dim == 1 && fabs(0.11 - ghost) <1.e-10 ) 
    {
        Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
        real_t max_red_err = b->max_errs_reduce[0];
        if ( fabs(max_red_err - 0.00564167 ) > 1.e-8) 
        {
            std::cout << " expected max_red_err == 0.00564167 got : "
                    << max_red_err << "\n";
            abort();
        }
    }
    // test 2:  -i sinc -d 3 -m 2 -b 2 -o 0.11
    if (world.size() == 1 && tot_blocks == 2 && dom_dim == 2 && fabs(0.11 - ghost) <1.e-10 ) 
    {
        Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
        real_t max_red_err = b->max_errs_reduce[0];
        if ( fabs(max_red_err - 0.163644 ) > 1.e-6) 
        {
            std::cout   << " expected max_red_err == 0.163644 got : "
                    << max_red_err << "\n";
            abort();
        }
    }
    // test3: -i sinc -d 3 -m 2 -b 6
    if (world.size() == 1 && tot_blocks == 6 && dom_dim == 2 && fabs(0.1 - ghost) <1.e-10 ) 
    {
        Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
        real_t max_red_err = b->max_errs_reduce[0];
        if ( fabs(max_red_err - 0.0054730281208712  ) > 1.e-10) 
        {
            std::cout  << std::setprecision(14) << " expected max_red_err == 0.0054730281208712 got : "
                    << max_red_err << "\n";
            abort();
        }
        else
        {
            std::cout << std::setprecision(14) <<  "passed max_red_err == " << max_red_err << "\n";
        }
        Block<real_t> *blendBlock1 = static_cast<Block<real_t>*>(master.block(1));
        MatrixX<real_t> &bl = blendBlock1->blend->domain;
        //std::cout<<bl(0,2) << "\n";
        if  ( (fabs(bl(0, 2) - 0.031224744068286) > 1.e-10 ) ||
                (fabs(bl(1, 2) - 0.022554418713158) > 1.e-10 ) )
        {
            std::cout << std::setprecision(14)
                        <<  "expected blend(0,2) = 0.031224744068286  got "  << bl(0, 2) << "\n";
            std::cout <<  "expected blend(1,2) = 0.022554418713158  got "  << bl(1, 2) << "\n";
            abort();
        }
        else
        {
            std::cout << std::setprecision(14) <<  "passed test blend(0, 2) == "
                    << bl(0, 2) << "  blend->domain(1, 2) == " << bl(1, 2)<< "\n";
        }
    }
#ifdef MFA_KOKKOS
    Kokkos::finalize();
#endif
}

