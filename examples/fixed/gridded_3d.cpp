//--------------------------------------------------------------
// example of encoding / decoding 3D gridded data sets read from
// a file
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
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
#include "example-setup.hpp"

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    // Constants for this example
    const int tot_blocks  = 1;        // No parallelism
    const int mem_blocks  = -1;       // everything in core for now
    const int num_threads = 1;        // needed in order to do timing
    const int dom_dim     = 3;        // dimension of domain (<= pt_dim)
    const int structured  = 1;        // input data format (bool 0/1)

    // default command line arguments
    int         pt_dim          = 4;        // dimension of input points
    int         scalar          = 1;        // flag for scalar or vector-valued science variables (0 == multiple scalar vars)
    int         geom_degree     = 1;        // degree for geometry (same for all dims)
    int         vars_degree     = 4;        // degree for science variables (same for all dims)
    int         geom_nctrl      = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl      = {11};     // initial # control points for all science variables (default same for all dims)
    string      input           = "";       // input dataset
    string      infile          = "";       // input file name
    int         verbose         = 1;        // MFA verbosity (0 = no extra output)
    int         weighted        = 0;        // Use NURBS weights (0/1)
    int         adaptive        = 0;        // do analytical encode (0/1)
    real_t      e_threshold     = 1e-1;     // error threshold for adaptive encoding
    int         rounds          = 0;        // max number of rounds for adaptive encoding
    bool        help            = false;    // show help

    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('l', "scalar",      scalar,     " flag for scalar or vector-valued science variables");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('z', "adaptive",    adaptive,   " do adaptive encode (0/1)");
    ops >> opts::Option('e', "errorbound",  e_threshold," error threshold for adaptive encoding");
    ops >> opts::Option('z', "rounds",      rounds,     " max number of rounds for adaptive encoding");
    ops >> opts::Option('z', "verbose",     verbose,    " output verbosity (0/1)");
    ops >> opts::Option('h', "help",        help,       " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // print input arguments
    echo_mfa_settings("gridded_3d example", dom_dim, pt_dim, scalar, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        0, 0, adaptive, e_threshold, rounds);
    echo_data_settings(input, infile, 0, 0);
    echo_data_mod_settings(structured, 0, 0, 0, 0);

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
    vector<int> model_dims = {dom_dim, pt_dim - dom_dim};
    if (scalar) // Set up (pt_dim - dom_dim) separate scalar variables
    {
        model_dims.assign(pt_dim - dom_dim + 1, 1);
        model_dims[0] = dom_dim;                        // index 0 == geometry
    }

    // set up parameters for examples
    MFAInfo     mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                input, infile, 0, structured, 0, 0, 0, 0,
                0, 0, adaptive, verbose, mfa_info , d_args);

    // Create data set for modeling. Input keywords are defined in example_signals.hpp
    if (datasets_3d.count(input) == 1)
    {
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->read_3d_vector_data(cp, mfa_info, d_args);
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
    }

    // compute the MFA
    fprintf(stderr, "\nStarting encoding...\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
    { 
        if (!adaptive)
            b->fixed_encode_block(cp, mfa_info);
        else
            b->adaptive_encode_block(cp, e_threshold, rounds, mfa_info);
    });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "Encoding done.\n");

    // decode at original point locations and compute pointwise error
    double decode_time = MPI_Wtime();
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
    { 
        b->range_error(cp, true, false);
    });
    decode_time = MPI_Wtime() - decode_time;

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, true); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("approx.mfa", world, master);
}
