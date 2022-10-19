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
#include <diy/../../examples/opts.h>

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
    int tot_blocks = world.size();            // default number of global blocks
    int mem_blocks = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int dom_dim = 3;                    // dimension of domain (<= pt_dim)
    int geom_degree = 1;              // degree for geometry (same for all dims)
    int vars_degree = 4;     // degree for science variables (same for all dims)
    int geom_nctrl = -1; // input number of control points for geometry (same for all dims)
    int vars_nctrl = 11; // input number of control points for all science variables (same for all dims)
    //real_t ghost        = 0.1;                  // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    int nb[3] = { 2, 2, 3 };                  // nb blocks in x, y, z directions
    std::vector<int> ovs;        // ghosting in 3 directions; default is 2, 2, 2
    ovs.push_back(2);
    ovs.push_back(2);
    ovs.push_back(2);
    string inputfile = "/media/iulian/ExtraDrive1/MFAData/S3D/6_small.xyz"; // input file for s3d data
    std::vector<int> resolutions;
    resolutions.push_back(20);
    resolutions.push_back(20);
    resolutions.push_back(20);
    int starts[3] = { 0, 0, 0 };
    int ends[3] = { 549, 539, 703 }; // maximum possible
    int shp[3] = { 550, 540, 704 }; // shape of the global block
    bool write_output = false;
    int chunk = 3; // 3 values per point, in this case a velocity
    bool help;                                // show help
    double strong_scaling = 0.0; // vary number of control points per direction, based on
    // a fraction of a total number
    // if the number is > 0, then number of control points is based on a fraction of number of input points
    // 0.2 means a compression of 20% per direction !!
    int error = 1;      // decode all input points and check error (bool 0 or 1)

    int transpose = 0; // if 1, transpose data
    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('m', "dom_dim", dom_dim, " dimension of domain");
    ops >> opts::Option('p', "geom_degree", geom_degree,
                    " degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree,
                    " degree in each dimension of science variables");

    ops >> opts::Option('g', "geom_nctrl", geom_nctrl,
                    " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl", vars_nctrl,
                    " number of control points in each dimension of all science variables");

    ops >> opts::Option('f', "data_file", inputfile, " s3d data file location");

    ops >> opts::Option('x', "nbx", nb[0], " nb of blocks in first direction");
    ops >> opts::Option('y', "nby", nb[1], " nb of blocks in second direction");
    ops >> opts::Option('z', "nbz", nb[2], " nb of blocks in third direction");

    ops >> opts::Option('a', "min_x", starts[0],
                    " start of block in first direction");
    ops >> opts::Option('b', "min_y", starts[1],
                    " start of block in second direction");
    ops >> opts::Option('c', "min_z", starts[2],
                    " start of block in third direction");
    ops >> opts::Option('r', "max_x", ends[0],
                    " end  of block in first direction");
    ops >> opts::Option('s', "max_y", ends[1],
                    " end  of block in second direction");
    ops >> opts::Option('t', "max_z", ends[2],
                    " end  of block in third direction");

    ops >> opts::Option('A', "first_dir", shp[0],
                    " size of block in first direction");
    ops >> opts::Option('B', "second_dir", shp[1],
                    " start of block in second direction");
    ops >> opts::Option('C', "third_dir", shp[2],
                    " start of block in third direction");
    ops >> opts::Option('D', "chunk", chunk,
                    " number of values per geometric point");

    // need to explain better
    ops >> opts::Option('T', "transpose", transpose, " transpose input data ");
    // this does not work as expected yet; use some default values
    ops >> opts::Option('o', "overlap", ovs, " overlaps in 3 directions ");
    ops >> opts::Option('u', "resolutions", resolutions,
                    " number of output points in each dimension of domain");
    ops >> opts::Option('S', "strongScaling", strong_scaling,
                    "fraction of control points per input points per direction");
    ops >> opts::Option('W', "write", write_output, " write output file");
    ops >> opts::Option('h', "help", help, " show help");

    if (!ops.parse(argc, argv) || help) {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    Bounds<int> dom_bounds(3);
    for (int j = 0; j < 3; j++) {
        dom_bounds.min[j] = starts[j];
        dom_bounds.max[j] = ends[j];
        if (ends[j] - starts[j] + 1 < nb[j]) {
            cerr << " number of blocks in " << j << " direction is too high \n";
            return 1;
        }

    }
    // the shape of the file is predetermined
    std::vector<unsigned> shape;
    // shape will be 3x in the last direction, to account for 3 values per point
    // block will still be 704, 540, 550 in size; actually 550 , 540, 704x3 in size
    tot_blocks = 1;
    for (int k = 0; k < dom_dim; k++) {
        shape.push_back(shp[k]);
        tot_blocks *= nb[k];
    }
    shape[dom_dim - 1] *= chunk;

    // minimal number of geometry control points if not specified
    if (geom_nctrl == -1)
        geom_nctrl = geom_degree + 1;

    if (world.rank() == 0) {
        cerr << "\n--------- Input arguments ----------\n";
        cerr << "Number of MPI tasks: " << world.size()
                << "\nNumber of blocks: " << tot_blocks << "\n";
        cerr << " dom_dim = " << dom_dim << "\ngeom_degree = " << geom_degree
                << " vars_degree = " << vars_degree << " tot_blocks = "
                << tot_blocks << endl;
        cerr << " file location: " << inputfile << "\n";
        cerr << " start block: " << starts[0] << "\t " << starts[1] << "\t"
                << starts[2] << endl;
        cerr << " end block:   " << ends[0] << "\t" << ends[1] << "\t"
                << ends[2] << endl;
        cerr << " divisions : " << nb[0] << ":" << nb[1] << ":" << nb[2]
                << endl;
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

        fprintf(stderr, "-------------------------------------\n\n");
    }
    int leftover = tot_blocks % world.size();
    if (0 != leftover) {
        if (world.rank() == 0) {
            cerr
                    << "number of blocks should be divisible by the number of MPI processes.\n";
        }
        exit(1);
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

    // set global domain bounds of the interested block ; depends on global shape and what kind
    // of problem we solve; for 1d block, we could select any direction, and start1, start2
    // for 2d, we could select a plane and one start point
    // in general, the block of interest has a min and max
    // when min and max are the same in one direction, one dimension is effectively lost
    // the file still needs the whole shape

    diy::RegularDecomposer<Bounds<int>>::BoolVector share_face;

    diy::RegularDecomposer<Bounds<int>>::BoolVector wrap;

    diy::RegularDecomposer<Bounds<int>>::CoordinateVector ghosts;
    diy::RegularDecomposer<Bounds<int>>::DivisionsVector divs;

    for (int k = 0; k < dom_dim; k++) {
        share_face.push_back(true);
        wrap.push_back(false);
        ghosts.push_back(ovs[k]);
        divs.push_back(nb[k]);
    }

    // start copy from multiblock req example
    DomainArgs d_args(dom_dim, dom_dim + 1);

    // set default args for diy foreach callback functions

    /*d_args.dom_dim      = (int)mapDim.size();
     d_args.pt_dim       = d_args.dom_dim+1;*/
    d_args.weighted = 0;
    d_args.multiblock = true;
    d_args.verbose = 0;
    d_args.r = 0.0;
    d_args.t = 0.0;
    d_args.structured   = true;
    //d_args.vars_p[0]    = vars_degree; // we know we have exactly one science var
    //dom_dim = mapDim.size();
    for (int i = 0; i < dom_dim; i++) {
        d_args.geom_p[i] = geom_degree;
        d_args.vars_p[0][i] = vars_degree;

        //d_args.ndom_pts[i]          = ndomp;
        d_args.geom_nctrl_pts[i] = geom_nctrl;
        if ((strong_scaling > 0.0001) && (3 == d_args.dom_dim)) {
            // in this case the number of control point per direction is decided by a factor
            // from the number of input points per direction
            // number of input points per direction is interval length per number of blocks per direction
            // we also reverse the order, this is why we do 2-i, for dom_dim 3
            d_args.vars_nctrl_pts[0][2 - i] = floor(
                    ((ends[i] - starts[i] + 1) / nb[i]) * strong_scaling) + 1;
        } else {
            d_args.vars_nctrl_pts[0][i] = vars_nctrl;
        }

    }
    if (world.rank() == 0) {
        cerr << " number of control points per direction: \n";
        for (int k = 0; k < d_args.dom_dim; k++)
            cerr << " " << d_args.vars_nctrl_pts[0][k];
        cerr << "\n ratio : " << strong_scaling << "\n";

    }
    d_args.f[0] = 1.0;

    double start_reading = MPI_Wtime();
    // decompose the domain into blocks
    Decomposer<int> decomposer(dom_dim, dom_bounds, tot_blocks, share_face,
            wrap, ghosts, divs);
    decomposer.decompose(world.rank(), assigner,
            [&](int gid, const Bounds<int> &core, const Bounds<int> &bounds,
                    const Bounds<int> &domain, const RCLink<int> &link) {
                Block<real_t>::readfile(gid, core, bounds, link, master, mapDim,
                        inputfile, shape, chunk, transpose, d_args);
            });

    // compute the MFA
    if (world.rank() == 0)
        fprintf(stderr, "\nStarting fixed encoding...\n\n");
    world.barrier();                     // to synchronize timing
    double start_encode = MPI_Wtime();

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->fixed_encode_block(cp, d_args);
    });
    world.barrier();                     // to synchronize timing
    double end_encode = MPI_Wtime();
    ;
    if (world.rank() == 0)
        fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
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

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->decode_core_ures(cp, resolutions);
    });

    world.barrier();
    double end_resolution_decode = MPI_Wtime();
    if (world.rank() == 0) {
        fprintf(stderr, "decomposing and reading time = %.3lf s.\n",
                start_encode - start_reading);
        fprintf(stderr, "encoding time                = %.3lf s.\n",
                end_encode - start_encode);
        fprintf(stderr, "decoding time                = %.3lf s.\n",
                end_decode - end_encode);
        fprintf(stderr, "decode at resolution         = %.3lf s.\n",
                end_resolution_decode - end_decode);
    }

    if (write_output) {
        diy::io::write_blocks("approx.mfa", world, master);

    }
    if (ovs[0] == 0 && ovs[1] == 0 && ovs[2] == 0) // no overlap, stop
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
    double end_decode_requests = MPI_Wtime();
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
        b->print_brief_block(cp, error);
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
                    b->print_block(cp, error);
                }); // only blocks on master
        fprintf(stderr, "exchange time         = %.3lf s.\n",
                exchange_end - end_resolution_decode);
        fprintf(stderr, "blend time                   = %.3lf s.\n",
                recv_blend_end - exchange_end);
        fprintf(stderr, "blend total                  = %.3lf s.\n",
                recv_blend_end - end_decode);
        if (write_output)
            fprintf(stderr, "write time                   = %.3lf s.\n",
                    write_time - recv_blend_end);
    }
    if (world.rank() == 0 && mapDim.size() == 2 && nb[1] * nb[2] == 6) {
        Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
        int blockMax = (int) b->max_errs_reduce[1];
        real_t max_red_err = b->max_errs_reduce[0];
        if (blockMax != 3 || fabs(max_red_err - 1.0896873215513452) > 1.e-10) {
            std::cout << "expected blockMax == 3 got " << blockMax
                    << " expected max_red_err == 1.0896873215513452 got : "
                    << max_red_err << "\n";
            abort();
        }
        //
        if (world.size() == 1 && 0 == transpose ) {
            Block<real_t> *bmax = static_cast<Block<real_t>*>(master.block(
                    blockMax));
            MatrixX<real_t> &bl = bmax->blend->domain;
            //std::cout<<bl(0,2) << "\n";
            if  ( (fabs(bl(0, 2) - 31.079045650447384) > 1.e-7 ) ||
                  (fabs(bl(1, 2) - 31.175520176541) > 1.e-7 ) )
            {
                std::cout << std::setprecision(14)
                          <<  "expected blend(0,2) = 31.07904565044  got "  << bl(0, 2) << "\n";
                std::cout <<  "expected blend(1,2) = 31.175520176541  got "  << bl(1, 2) << "\n";
                abort();
            }
            else
            {
                std::cout << std::setprecision(14) <<  "passed test blend(0, 2) == "
                        << bl(0, 2) << "  blend(1, 2) == " << bl(1, 2)<< "\n";
            }
        }
        //std::cout << bmax->blend << "\n";
    }
    if (world.rank() == 0 && mapDim.size() == 3 && nb[1] * nb[2] == 6) {
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

