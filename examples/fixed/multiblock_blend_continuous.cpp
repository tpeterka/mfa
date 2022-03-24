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

int main(int argc, char **argv) {
    // initialize MPI
    diy::mpi::environment env(argc, argv); // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int tot_blocks = world.size();            // default number of global blocks
    int mem_blocks = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int pt_dim = 3;                    // dimension of input points
    int dom_dim = 2;                    // dimension of domain (<= pt_dim)
    int geom_degree = 1;              // degree for geometry (same for all dims)
    int vars_degree = 4;     // degree for science variables (same for all dims)
    int ndomp = 100;        // input number of domain points (same for all dims)
    std::vector<int> resolutions;            // output points resolution
    resolutions.push_back(120);
    int geom_nctrl = -1; // input number of control points for geometry (same for all dims)
    std::vector<int> vars_nctrl_v;
    vars_nctrl_v.push_back(11);
    //int vars_nctrl = 11; // input number of control points for all science variables (same for all dims)
    string input = "sine";               // input dataset
    int weighted = 0;                 // solve for and use weights (bool 0 or 1)
    int strong_sc = 0;         // strong scaling (bool 0 or 1, 0 = weak scaling)
    real_t ghost = 0.1; // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    bool write_output = false;
    bool help;                                // show help
    int error = 1;      // decode all input points and check error (bool 0 or 1)
    double noise = 0;

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim", pt_dim, " dimension of points");
    ops >> opts::Option('m', "dom_dim", dom_dim, " dimension of domain");
    ops
            >> opts::Option('p', "geom_degree", geom_degree,
                    " degree in each dimension of geometry");
    ops
            >> opts::Option('q', "vars_degree", vars_degree,
                    " degree in each dimension of science variables");
    ops
            >> opts::Option('n', "ndomp", ndomp,
                    " number of input points in each dimension of domain");
    ops
            >> opts::Option('r', "resolution", resolutions,
                    " number of output points in each dimension of domain");
    ops
            >> opts::Option('g', "geom_nctrl", geom_nctrl,
                    " number of control points in each dimension of geometry");
    ops
            >> opts::Option('v', "vars_nctrl", vars_nctrl_v,
                    " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input", input, " input dataset");
    ops >> opts::Option('w', "weights", weighted, " solve for and use weights");
    ops
            >> opts::Option('b', "tot_blocks", tot_blocks,
                    " total number of blocks");
    ops >> opts::Option('s', "noise", noise, " fraction of noise (0.0 - 1.0)");
    ops
            >> opts::Option('t', "strong_sc", strong_sc,
                    " strong scaling (1 = strong, 0 = weak)");
    ops
            >> opts::Option('o', "overlap", ghost,
                    " relative ghost zone overlap (0.0 - 1.0)");
    ops >> opts::Option('W', "write", write_output, " write output file");
    ops >> opts::Option('h', "help", help, " show help");

    if (!ops.parse(argc, argv) || help) {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // minimal number of geometry control points if not specified
    if (geom_nctrl == -1)
        geom_nctrl = geom_degree + 1;

    // echo args
    if (world.rank() == 0) {
        fprintf(stderr, "\n--------- Input arguments ----------\n");
        cerr << "pt_dim = " << pt_dim << " dom_dim = " << dom_dim
                << "\ngeom_degree = " << geom_degree << " vars_degree = "
                << vars_degree << "\ninput pts = " << ndomp
                << " geom_ctrl pts = " << geom_nctrl << "\nvars_ctrl_pts = "
                << vars_nctrl_v[0] << " input = " << input << " tot_blocks = "
                << tot_blocks << " strong scaling = " << strong_sc
                << " ghost overlap = " << ghost << endl;
#ifdef CURVE_PARAMS
        cerr << "parameterization method = curve" << endl;
#else
        cerr << "parameterization method = domain" << endl;
#endif
#ifdef MFA_TBB
    cerr << "threading: TBB" << endl;
#endif
#ifdef MFA_KOKKOS
    cerr << "threading: Kokkos" << endl;
#endif
#ifdef MFA_SYCL
    cerr << "threading: SYCL" << endl;
#endif
#ifdef MFA_SERIAL
    cerr << "threading: serial" << endl;
#endif
#ifdef MFA_NO_WEIGHTS
    cerr << "weighted = 0" << endl;
#else
    cerr << "weighted = " << weighted << endl;
#endif
        fprintf(stderr, "-------------------------------------\n\n");
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

    DomainArgs d_args(dom_dim, pt_dim);

    // set default args for diy foreach callback functions
    d_args.pt_dim = pt_dim;
    d_args.dom_dim = dom_dim;
    d_args.weighted = weighted;
    d_args.multiblock = true;
    d_args.n = noise;
    d_args.verbose = 0;
    d_args.r = 0.0;
    d_args.t = 0.0;
    d_args.structured   = true; // multiblock not tested for unstructured data yet
    // fill with the last value for nb ctrl points
    if (dom_dim > vars_nctrl_v.size())
    {
        int  sz = vars_nctrl_v.size();
        int fill=vars_nctrl_v[sz-1];
        for (int i=sz; i<dom_dim; i++)
            vars_nctrl_v.push_back(fill);
    }
    // fill with the last value for nb resolution points
    if (dom_dim > resolutions.size())
    {
        int  sz = resolutions.size();
        int fill=resolutions[sz-1];
        for (int i=sz; i<dom_dim; i++)
            resolutions.push_back(fill);
    }
    for (int i = 0; i < dom_dim; i++) {
        d_args.geom_p[i] = geom_degree;
        d_args.vars_p[0][i] = vars_degree;
        if (strong_sc)     // strong scaling, reduced number of points per block
        {
            d_args.ndom_pts[i] = ndomp / divs[i];
            d_args.geom_nctrl_pts[i] =
                    geom_nctrl / divs[i] > geom_degree ?
                            geom_nctrl / divs[i] : geom_degree + 1;
            d_args.vars_nctrl_pts[0][i] = vars_nctrl_v[0] / divs[i];
        } else                  // weak scaling, same number of points per block
        {
            d_args.ndom_pts[i] = ndomp;
            d_args.geom_nctrl_pts[i] = geom_nctrl;
            d_args.vars_nctrl_pts[0][i] = vars_nctrl_v[i]; // should have enough per direction ! otherwise will crash
        }
    }
    for (int i = 0; i < pt_dim - dom_dim; i++)
        d_args.f[i] = 1.0;

    // initialize input data

    // sine function f(x) = sin(x), f(x,y) = sin(x)sin(y), ...
    if (input == "sine") {
        for (int i = 0; i < pt_dim - dom_dim; i++)  // for all science variables
            d_args.s[i] = i + 1;                      // scaling factor on range
        master.foreach(
                [&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
                    b->generate_analytical_data(cp, input, d_args);
                });
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc") {
        for (int i = 0; i < pt_dim - dom_dim; i++)  // for all science variables
            d_args.s[i] = 10.0 * (i + 1);             // scaling factor on range
        master.foreach(
                [&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
                    b->generate_analytical_data(cp, input, d_args);
                });
    }

    double start_encode = MPI_Wtime();
    // compute the MFA
    if (world.rank() == 0)
        fprintf(stderr, "\nStarting fixed encoding...\n\n");
    world.barrier();                     // to synchronize timing

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->fixed_encode_block(cp, d_args);
    });
    world.barrier();                     // to synchronize timing
    double end_encode = MPI_Wtime();
    //encode_time = MPI_Wtime() - encode_time;
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
        b->range_error(cp, 0, true, true);
    });
#endif
    world.barrier();
    double end_decode = MPI_Wtime();

    master.foreach([&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
        b->decode_core_ures(cp, resolutions);
    });

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

    int k = 2;                          // the radix of the k-ary reduction tree

    bool contiguous = true;
    // partners for merge over regular block grid
    diy::RegularMergePartners partners(decomposer,  // domain decomposition
            k,                                      // radix of k-ary reduction
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
        master.foreach(
                [&](Block<real_t> *b, const diy::Master::ProxyWithLink &cp) {
                    b->print_block(cp, error);
                });

        fprintf(stderr, "encoding time                = %.3lf s.\n",
                end_encode - start_encode);
        fprintf(stderr, "decoding time                = %.3lf s.\n",
                end_decode - end_encode);
        fprintf(stderr, "decode at resolution         = %.3lf s.\n",
                end_resolution_decode - end_decode);
        fprintf(stderr, "decode patches         = %.3lf s.\n",
                decode_patches_end - end_resolution_decode );
        fprintf(stderr, "exchange time         = %.3lf s.\n",
                exchange_end - decode_patches_end);
        fprintf(stderr, "blend time                   = %.3lf s.\n",
                final_blend_end - exchange_end);
        fprintf(stderr, "blend total                  = %.3lf s.\n",
                final_blend_end - end_decode);
        if (write_output)
            fprintf(stderr, "write time                   = %.3lf s.\n",
                    write_time - final_blend_end);
        fprintf(stderr, "-------------------------------------\n\n");
    }
    // add simple tests for multi_bc test cases
    // test 1: -b 2 -d 2 -m 1 -i sinc -o 0.11
    // tot_blocks == 2, dom_dim == 1
    if (world.size() == 1 && tot_blocks == 2 && dom_dim == 1 && fabs(0.11 - ghost) <1.e-10 ) {
            Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
            real_t max_red_err = b->max_errs_reduce[0];
            if ( fabs(max_red_err - 0.00564167 ) > 1.e-8) {
                std::cout << " expected max_red_err == 0.00564167 got : "
                        << max_red_err << "\n";
                abort();
            }
    }
    // test 2:  -i sinc -d 3 -m 2 -b 2 -o 0.11
    if (world.size() == 1 && tot_blocks == 2 && dom_dim == 2 && fabs(0.11 - ghost) <1.e-10 ) {
            Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
            real_t max_red_err = b->max_errs_reduce[0];
            if ( fabs(max_red_err - 0.163644 ) > 1.e-6) {
                std::cout   << " expected max_red_err == 0.163644 got : "
                        << max_red_err << "\n";
                abort();
            }
    }
    // test3: -i sinc -d 3 -m 2 -b 6
    if (world.size() == 1 && tot_blocks == 6 && dom_dim == 2 && fabs(0.1 - ghost) <1.e-10 ) {
            Block<real_t> *b = static_cast<Block<real_t>*>(master.block(0));
            real_t max_red_err = b->max_errs_reduce[0];
            if ( fabs(max_red_err - 0.0054730281208712  ) > 1.e-10) {
                std::cout  << std::setprecision(14) << " expected max_red_err == 0.0054730281208712 got : "
                        << max_red_err << "\n";
                abort();
            }
            else
            {
                std::cout << std::setprecision(14) <<  "passed max_red_err == " << max_red_err << "\n";
            }
            Block<real_t> *blendBlock1 = static_cast<Block<real_t>*>(master.block(1));
            MatrixX<real_t> &bl = blendBlock1->blend;
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
                        << bl(0, 2) << "  blend(1, 2) == " << bl(1, 2)<< "\n";
            }
    }
#ifdef MFA_KOKKOS
    Kokkos::finalize();
#endif
}

