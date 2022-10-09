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
#include "nasa_block.hpp"
#include "example-setup.hpp"

using namespace std;

// static const unsigned DIM = 3;
// using SimplePoint = diy::Point<float, DIM>;

// //
// // callback function for redistribute operator, called in each round of the reduction
// //
// void redistribute(Block* b,                                 // local block
//                   const diy::ReduceProxy& srp,              // communication proxy
//                   const diy::RegularSwapPartners& partners) // partners of the current block
// {
//     unsigned      round    = srp.round();                   // current round number

//     // step 1: dequeue
//     // dequeue all the incoming points and add them to this block's vector
//     // could use srp.incoming() instead
//     for (int i = 0; i < srp.in_link().size(); ++i)
//     {
//         int nbr_gid = srp.in_link().target(i).gid;
//         if (nbr_gid == srp.gid())
//             continue;

//         std::vector<Block::Point>    in_points;
//         srp.dequeue(nbr_gid, in_points);
//         fmt::print(stderr, "[{}:{}] Received {} points from [{}]\n",
//                    srp.gid(), round, (int) in_points.size(), nbr_gid);
//         for (size_t j = 0; j < in_points.size(); ++j)
//             b->points.push_back(in_points[j]);
//     }

//     // step 2: sort and enqueue
//     if (srp.out_link().size() == 0)        // final round; nothing needs to be sent
//         return;

//     std::vector< std::vector<Block::Point> > out_points(srp.out_link().size());
//     int group_size = srp.out_link().size();  // number of outbound partners
//     int cur_dim    = partners.dim(round);    // current dimension along which groups are formed
//     // sort points into vectors corresponding to neighbor blocks
//     for (size_t i = 0; i < b->points.size(); ++i) // for all points
//     {
//         auto loc = static_cast<size_t>(floor((b->points[i][cur_dim] - b->box.min[cur_dim]) /
//                                              (b->box.max[cur_dim] - b->box.min[cur_dim]) * group_size));
//         out_points[loc].push_back(b->points[i]);
//     }
//     int pos = -1;
//     // enqueue points to neighbor blocks
//     for (int i = 0; i < group_size; ++i)     // for all neighbors
//     {
//         if (srp.out_link().target(i).gid == srp.gid())
//         {
//             b->points.swap(out_points[i]);
//             pos = i;
//         }
//         else
//         {
//             srp.enqueue(srp.out_link().target(i), out_points[i]);
//             fmt::print(stderr, "[{}] Sent {} points to [{}]\n",
//                        srp.gid(), (int) out_points[i].size(), srp.out_link().target(i).gid);
//         }
//     }

//     // step 3: readjust box boundaries for next round
//     float new_min = b->box.min[cur_dim] + (b->box.max[cur_dim] -
//                                            b->box.min[cur_dim])/group_size*pos;
//     float new_max = b->box.min[cur_dim] + (b->box.max[cur_dim] -
//                                            b->box.min[cur_dim])/group_size*(pos + 1);
//     b->box.min[cur_dim] = new_min;
//     b->box.max[cur_dim] = new_max;
// }

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                        // number of local blocks

    string logname = "nasa-vis.log." + to_string(world.rank());
    ofstream log(logname);

    // default command line arguments
    int         pt_dim          = 4;        // dimension of input points
    int         dom_dim         = 3;        // dimension of domain (<= pt_dim)
    int         scalar          = 1;        // flag for scalar or vector-valued science variables (0 == multiple scalar vars)
    int         geom_degree     = 1;        // degree for geometry (same for all dims)
    int         vars_degree     = 4;        // degree for science variables (same for all dims)
    int         ndomp           = 100;      // input number of domain points (same for all dims)
    int         geom_nctrl      = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl      = {11};     // initial # control points for all science variables (default same for all dims)
    real_t      ghost           = 0.0;      // fraction of block to take as ghost layer
    string      input           = "sinc";   // input dataset
    int         weighted        = 1;        // solve for and use weights (bool 0/1)
    real_t      rot             = 0.0;      // rotation angle in degrees
    real_t      twist           = 0.0;      // twist (waviness) of domain (0.0-1.0)
    real_t      noise           = 0.0;      // fraction of noise
    int         error           = 1;        // decode all input points and check error (bool 0/1)
    string      infile;                     // input file name
    int         structured      = 1;        // input data format (bool 0/1)
    int         rand_seed       = -1;       // seed to use for random data generation (-1 == no randomization)
    real_t      regularization  = 0;        // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int         reg1and2        = 0;        // flag for regularizer: 0 = regularize only 2nd derivs. 1 = regularize 1st and 2nd
    int         verbose         = 1;        // MFA verbosity (0 = no extra output)
    bool        help            = false;    // show help

    int subdomain_id = 0;
    int time_step = 0;
    int time_step_pre = 0;
    string var_name;
    int do_encode = 0;  // false by default
    vector<real_t> domain_min = {-150, -100, -150};
    vector<real_t> domain_max = {200, 100, 100};

    // Constants for this example
    const bool adaptive = false;

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('l', "scalar",      scalar,     " flag for scalar or vector-valued science variables");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('o', "overlap",     ghost,      " relative ghost zone overlap (0.0 - 1.0)");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('c', "error",       error,      " decode entire error field (default=true)");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");

    ops >> opts::Option('z', "id", subdomain_id, "index of subdomain file to read");
    ops >> opts::Option('z', "ts", time_step, "index of time step to read");
    ops >> opts::Option('z', "ts_pre", time_step_pre, "prefix of the directory to search for the given time step");
    ops >> opts::Option('z', "var", var_name, "name of variable to read");
    ops >> opts::Option('z', "num_blocks", nblocks, "number of diy blocks to use");
    ops >> opts::Option('z', "do_encode", do_encode, "flag to run encoding/decoding");
    ops >> opts::Option('z', "domain_min", domain_min, "");
    ops >> opts::Option('z', "domain_max", domain_max, "");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    int tot_blocks  = nblocks * world.size();   // number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    echo_mfa_settings("nasa retropropulsion example", pt_dim, dom_dim, scalar, geom_degree, geom_nctrl, vars_degree, vars_nctrl, regularization, reg1and2, weighted, adaptive, 0, 0, log);
    echo_data_settings(ndomp, 0, input, infile, noise, rot, twist, structured, rand_seed, log);

    // initialize DIY
    diy::FileStorage          storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &NASABlock<real_t>::create,
                                     &NASABlock<real_t>::destroy,
                                     &storage,
                                     &NASABlock<real_t>::save,
                                     &NASABlock<real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

    // set global domain bounds and decompose
    Bounds<real_t> dom_bounds(dom_dim);
    for (int i = 0; i < dom_bounds.min.dimension(); i++)
    {
        dom_bounds.min[i] = domain_min[i];
        dom_bounds.max[i] = domain_max[i];
    }
    

    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { NASABlock<real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });
    vector<int> divs(dom_dim);                          // number of blocks in each dimension
    decomposer.fill_divisions(divs);

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    // Different examples can reset this below
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

    echo_multiblock_settings(mfa_info, d_args, world.size(), tot_blocks, divs, false, ghost, log);

    master.foreach([&](NASABlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
    { 
        b->read_nasa3d_retro(cp, mfa_info, d_args, subdomain_id, time_step, time_step_pre, var_name); 
    });

    // partners for swap over regular block grid
    diy::RegularSwapPartners  partners(decomposer,  // domain decomposition
                                       2,       // radix of k-ary reduction
                                       false);  // contiguous = true: distance doubling
                                                // contiguous = false: distance halving

    diy::reduce(master,                         // Master object
                assigner,                       // Assigner object
                partners,                       // RegularSwapPartners object
                NASABlock<real_t>::redistribute);                 // swap operator callback function
    world.barrier();
    if (world.rank() == 0) cerr << "Done with swap-reduce" << endl;
    world.barrier();

    master.foreach([&](NASABlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
    {
        string out_filename = var_name + "_" + to_string(time_step) + "_out_" + to_string(cp.gid()) + ".txt";
        ofstream os(out_filename);
        for (int i = 0; i < b->points.size(); i++)
        {
            os << b->points[i][0] << " " << b->points[i][1] << " " << b->points[i][2] << " " << b->points[i][3] << "\n";
        }
        os.close();
    });


    double encode_time = 0, decode_time = 0;
    if (do_encode)
    {
        master.foreach([&](NASABlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
        {
            cout << "GID" << cp.gid() << ": Set Input." << endl;
            b->set_input(cp, mfa_info, d_args);
        });
        
        // compute the MFA
        log << "\nStarting fixed encoding...\n\n" << flush;
        encode_time = MPI_Wtime();
        master.foreach([&](NASABlock<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->fixed_encode_block(cp, mfa_info); });
        encode_time = MPI_Wtime() - encode_time;
        log << "\n\nFixed encoding done.\n\n" << flush;

        // debug: compute error field for visualization and max error to verify that it is below the threshold
        decode_time = MPI_Wtime();
        if (error)
        {
            log << "\nFinal decoding and computing max. error...\n" << flush;
            bool saved_basis = structured; // TODO: basis functions are currently only saved during encoding of structured data
            master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { 
                b->range_error(cp, true, saved_basis);
            });
            decode_time = MPI_Wtime() - decode_time;
        }
    }

    // print results
    log << "\n------- Final block results --------\n" << flush;
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, error); });
    log << "encoding time         = " << setprecision(3) << encode_time << " s." << endl;
    if (error)
        log << "decoding time         = " << setprecision(3) << decode_time << " s." << endl;
    log << "-------------------------------------\n\n" << flush;

    // save the results in diy format
    diy::io::write_blocks("approx.mfa", world, master);
}
