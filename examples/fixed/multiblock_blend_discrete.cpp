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

// call back for max error call back
// callback function for merge operator, called in each round of the reduction
// one block is the root of the group
// link is the neighborhood of blocks in the group
// root block of the group receives data from other blocks in the group and reduces the data
// nonroot blocks send data to the root
//

int main(int argc, char **argv) {
    // initialize MPI
    diy::mpi::environment env(argc, argv); // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    // initialize Kokkos if needed
#ifdef MFA_KOKKOS
    Kokkos::initialize( argc, argv );
#endif

    // Parallelization 
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int pt_dim              = 3;        // dimension of input points
    int dom_dim             = 2;        // dimension of domain (<= pt_dim)
    int geom_degree         = 1;        // degree for geometry (same for all dims)
    int vars_degree         = 4;        // degree for science variables (same for all dims)
    int geom_nctrl          = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl  = {11};     // input number of control points for all science variables
    string infile = "/media/iulian/ExtraDrive1/MFAData/S3D/6_small.xyz"; // input file for s3d data
    // vector<int> resolutions = {120};    // output points resolution
    int weighted            = 0;        // solve for and use weights (bool 0 or 1)
    int strong_sc           = 0;        // strong scaling (bool 0 or 1, 0 = weak scaling)
    // real_t ghost            = 0.1;      // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    bool write_output       = false;
    // int error               = 1;        // decode all input points and check error (bool 0 or 1)
    // double noise            = 0;
    int         verbose = 0;
    bool help               = false;    // show help

    int         adaptive        = 0;        // do analytical encode (0/1)
    real_t      e_threshold     = 1e-1;     // error threshold for adaptive fitting
    int         rounds          = 0;

    std::vector<int> overlaps = {2, 2, 2};        // ghosting in 3 directions; default is 2, 2, 2
    // int nb[3] = { 2, 2, 3 };                  // nb blocks in x, y, z directions
    vector<int> resolutions = {20, 20, 20};

    // int starts[3] = { 0, 0, 0 };
    // int ends[3] = { 549, 539, 703 }; // maximum possible
    // int shp[3] = { 550, 540, 704 }; // shape of the global block
    int chunk = 3;
    int transpose = 0; // if 1, transpose data


    // default command line arguments
    // int dom_dim         = 3;                    // dimension of domain (<= pt_dim)
    // int geom_degree     = 1;              // degree for geometry (same for all dims)
    // int vars_degree     = 4;            // degree for science variables (same for all dims)
    // int geom_nctrl      = -1;       // input number of control points for geometry (same for all dims)
    // int vars_nctrl      = 11;       // input number of control points for all science variables (same for all dims)
    //real_t ghost        = 0.1;                  // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    // int nb[3] = { 2, 2, 3 };                  // nb blocks in x, y, z directions
    // std::vector<int> overlaps = {2, 2, 2};        // ghosting in 3 directions; default is 2, 2, 2
    // ovs.push_back(2);
    // ovs.push_back(2);
    // ovs.push_back(2);
    // string inputfile = "/media/iulian/ExtraDrive1/MFAData/S3D/6_small.xyz"; // input file for s3d data
    // std::vector<int> resolutions;
    // resolutions.push_back(20);
    // resolutions.push_back(20);
    // resolutions.push_back(20);
    // int starts[3] = { 0, 0, 0 };
    // int ends[3] = { 549, 539, 703 }; // maximum possible
    // int shp[3] = { 550, 540, 704 }; // shape of the global block
    // bool write_output = false;
    // int chunk = 3; // 3 values per point, in this case a velocity
    // bool help;                                // show help
    // double strong_scaling = 0.0; // vary number of control points per direction, based on
    // a fraction of a total number
    // if the number is > 0, then number of control points is based on a fraction of number of input points
    // 0.2 means a compression of 20% per direction !!
    // int error = 1;      // decode all input points and check error (bool 0 or 1)

    // int transpose = 0; // if 1, transpose data

    vector<int> nblocks = {2, 2, 3};
    vector<int> starts = {0, 0, 0};
    vector<int> ends = {549, 539, 703};
    vector<unsigned> shape = {550, 540, 704};
    vector<int> npts = {550, 540, 704};      // number of data points in each dimension 
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
    // ops >> opts::Option('s', "shape", shape, " shape of block in each direction");
    ops >> opts::Option('D', "chunk", chunk, " number of values per geometric point");

    // need to explain better
    ops >> opts::Option('T', "transpose", transpose, " transpose input data ");
    // this does not work as expected yet; use some default values
    ops >> opts::Option('o', "overlap", overlaps, " overlaps in 3 directions ");
    ops >> opts::Option('u', "resolutions", resolutions, " number of output points in each dimension of domain");
    ops >> opts::Option('t', "strong_sc",   strong_sc,      " strong scaling (1 = strong, 0 = weak)");
    ops >> opts::Option('W', "write", write_output, " write output file");
    ops >> opts::Option('h', "help", help, " show help");

    ops >> opts::Option('z', "adaptive",    adaptive,   " do adaptive encode (0/1)");
    ops >> opts::Option('e', "errorbound",  e_threshold," error threshold for adaptive encoding");
    ops >> opts::Option('z', "rounds",      rounds,     " max number of rounds for adaptive encoding");

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
    int tot_blocks = 1;
    vector<bool> share_face(dom_dim, true);
    vector<bool> wrap(dom_dim, false);
    Bounds<int> dom_bounds(3);
    for (int j = 0; j < 3; j++) {
        tot_blocks *= nblocks[j];

        npts[j] = ends[j] - starts[j] + 1;
        shape[j] = ends[j] - starts[j] + 1;

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
    echo_mfa_settings("multiblock blend discrete example", dom_dim, pt_dim, 1, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        0, 0, adaptive, e_threshold, rounds);
    echo_data_settings("s3d_blend", infile, 0, 0);

//     // minimal number of geometry control points if not specified
//     if (geom_nctrl == -1)
//         geom_nctrl = geom_degree + 1;

//     if (world.rank() == 0) {
//         cerr << "\n--------- Input arguments ----------\n";
//         cerr << "Number of MPI tasks: " << world.size()
//                 << "\nNumber of blocks: " << tot_blocks << "\n";
//         cerr << " dom_dim = " << dom_dim << "\ngeom_degree = " << geom_degree
//                 << " vars_degree = " << vars_degree << " tot_blocks = "
//                 << tot_blocks << endl;
//         cerr << " file location: " << inputfile << "\n";
//         cerr << " start block: " << starts[0] << "\t " << starts[1] << "\t"
//                 << starts[2] << endl;
//         cerr << " end block:   " << ends[0] << "\t" << ends[1] << "\t"
//                 << ends[2] << endl;
//         cerr << " divisions : " << nb[0] << ":" << nb[1] << ":" << nb[2]
//                 << endl;
// #ifdef CURVE_PARAMS
//         cerr << "parameterization method = curve" << endl;
// #else
//         cerr << "parameterization method = domain" << endl;
// #endif
// #ifdef MFA_TBB
//     cerr << "threading: TBB" << endl;
// #endif
// #ifdef MFA_KOKKOS
//     cerr << "threading: Kokkos" << endl;
// #endif
// #ifdef MFA_SYCL
//     cerr << "threading: SYCL" << endl;
// #endif
// #ifdef MFA_SERIAL
//     cerr << "threading: serial" << endl;
// #endif
// #ifdef MFA_NO_WEIGHTS
//     cerr << "weighted = 0" << endl;
// #else
//     cerr << "weighted = " << weighted << endl;
// #endif

//         fprintf(stderr, "-------------------------------------\n\n");
//     }

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

    // set global domain bounds of the interested block ; depends on global shape and what kind
    // of problem we solve; for 1d block, we could select any direction, and start1, start2
    // for 2d, we could select a plane and one start point
    // in general, the block of interest has a min and max
    // when min and max are the same in one direction, one dimension is effectively lost
    // the file still needs the whole shape

    // diy::RegularDecomposer<Bounds<int>>::BoolVector share_face;

    // diy::RegularDecomposer<Bounds<int>>::BoolVector wrap;

    // diy::RegularDecomposer<Bounds<int>>::CoordinateVector ghosts;
    // diy::RegularDecomposer<Bounds<int>>::DivisionsVector divs;

    // for (int k = 0; k < dom_dim; k++) {
    //     share_face.push_back(true);
    //     wrap.push_back(false);
    //     ghosts.push_back(ovs[k]);
    //     divs.push_back(nb[k]);
    // }

    // start copy from multiblock req example
    // DomainArgs d_args(dom_dim, dom_dim + 1);

    // set default args for diy foreach callback functions

    // /*d_args.dom_dim      = (int)mapDim.size();
    //  d_args.pt_dim       = d_args.dom_dim+1;*/
    // d_args.weighted = 0;
    // d_args.multiblock = true;
    // d_args.verbose = 0;
    // d_args.r = 0.0;
    // d_args.t = 0.0;
    // d_args.structured   = true; // multiblock not tested for unstructured data yet
    // //d_args.vars_p[0]    = vars_degree; // we know we have exactly one science var
    // //dom_dim = mapDim.size();
    // for (int i = 0; i < dom_dim; i++) {
    //     d_args.geom_p[i] = geom_degree;
    //     d_args.vars_p[0][i] = vars_degree;

    //     //d_args.ndom_pts[i]          = ndomp;
    //     d_args.geom_nctrl_pts[i] = geom_nctrl;
    //     if ((strong_scaling > 0.0001) && (3 == d_args.dom_dim)) {
    //         // in this case the number of control point per direction is decided by a factor
    //         // from the number of input points per direction
    //         // number of input points per direction is interval length per number of blocks per direction
    //         // we also reverse the order, this is why we do 2-i, for dom_dim 3
    //         d_args.vars_nctrl_pts[0][2 - i] = floor(
    //                 ((ends[i] - starts[i] + 1) / nb[i]) * strong_scaling) + 1;
    //     } else {
    //         d_args.vars_nctrl_pts[0][i] = vars_nctrl;
    //     }

    // }
    // if (world.rank() == 0) {
    //     cerr << " number of control points per direction: \n";
    //     for (int k = 0; k < d_args.dom_dim; k++)
    //         cerr << " " << d_args.vars_nctrl_pts[0][k];
    //     cerr << "\n ratio : " << strong_scaling << "\n";

    // }
    // d_args.f[0] = 1.0;


    // set up parameters for examples
    MFAInfo     mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        "s3d_blend", infile, 0, 1, 0, 0, 0, 0, 0, 0, adaptive, verbose,
                        mfa_info, d_args);

    // Set multiblock options
    d_args.multiblock   = true;
    if (strong_sc) 
    {
        mfa_info.splitStrongScaling(nblocks);
        // for (int i = 0; i < dom_dim; i++)
        // {
        //     d_args.ndom_pts[i] /= nblocks[i];
        // }
    }

    // Print block layout and scaling info
    if (world.rank() == 0)
    {
        echo_multiblock_settings(mfa_info, d_args, world.size(), tot_blocks, nblocks, strong_sc, overlaps);
    }

    double start_reading = MPI_Wtime();
    // decompose the domain into blocks
    // diy::decompose(dom_dim, world.rank(), dom_bounds, assigner, 
    //                 [&](int gid, const Bounds<int> &core, const Bounds<int> &bounds, const Bounds<int> &domain, const RCLink<int> &link)
    //                 { Block<real_t>::readfile(gid, core, bounds, link, master, mapDim, infile, shape, chunk, transpose, d_args); },
    //                 share_face,
    //                 wrap,
    //                 overlaps,
    //                 nblocks);
    Decomposer<int> decomposer(dom_dim, dom_bounds, tot_blocks, share_face, wrap, overlaps, nblocks);
    decomposer.decompose(world.rank(), assigner,
        [&](int gid, const Bounds<int> &core, const Bounds<int> &bounds, const Bounds<int> &domain, const RCLink<int> &link) 
        { Block<real_t>::readfile(gid, core, bounds, link, master, mapDim, infile, shape, chunk, transpose, mfa_info); });


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

    if (write_output) {
        diy::io::write_blocks("approx.mfa", world, master);
    }
    if (overlaps[0] == 0 && overlaps[1] == 0 && overlaps[2] == 0) // no overlap, stop
    {
        return 0;
    }

    // compute the neighbors encroachment
    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->compute_neighbor_overlaps(cp);
    });

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
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

    // merge-based reduction: create the partners that determine how groups are formed
    // in each round and then execute the reduction

    int k = 2;                          // the radix of the k-ary reduction tree

    bool contiguous = true;
    // partners for merge over regular block grid
    diy::RegularMergePartners partners(decomposer,  // domain decomposition
            k,           // radix of k-ary reduction
            contiguous); // contiguous = true: distance doubling
                         // contiguous = false: distance halving

    // reduction
    diy::reduce(master,                              // Master object
            assigner,                            // Assigner object
            partners,                            // RegularMergePartners object
            &max_err_cb);                    // merge operator callback function

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
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
        master.foreach(
                [&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
                    b->print_block(cp, true);
                }); // only blocks on master

        fprintf(stderr, "decode requests time         = %.3lf s.\n",
                end_decode_patches - end_resolution_decode);
        fprintf(stderr, " exchange time         = %.3lf s.\n",
                exchange_end - end_decode_patches);
        fprintf(stderr, "blend time                   = %.3lf s.\n",
                recv_blend_end - exchange_end);
        fprintf(stderr, "blend total                  = %.3lf s.\n",
                recv_blend_end - end_decode);
        if (write_output)
            fprintf(stderr, "write time                   = %.3lf s.\n",
                    write_time - recv_blend_end);

    }
#ifdef MFA_KOKKOS
    Kokkos::finalize();
#endif
}

