//--------------------------------------------------------------
// example of computing derivatives of an MFA
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

#include "../block.hpp"
#include "../opts.h"

using namespace std;

typedef  diy::RegularDecomposer<Bounds> Decomposer;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    string infile = "approx.out";               // diy input file

    // default command line arguments
    int  deriv     = 1;                         // which derivative to take (1st, 2nd, ...)
    int  partial   = -1;                        // limit derivatives to one partial in this dimension

    // get command line arguments
    using namespace opts;
    Options ops(argc, argv);
    ops >> Option('d', "deriv",   deriv,   " which derivative to take (1 = 1st, 2 = 2nd, ...)");
    ops >> Option('i', "infile",  infile,  " diy input file name");
    ops >> Option('a', "partial", partial, " dimension of 1 partial derivative only");

    if (ops >> Present('h', "help", "show help"))
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        "deriv = "    << deriv << endl;
    fprintf(stderr, "-------------------------------------\n\n");

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master      master(world,
            -1,
            -1,
            &Block<real_t>::create,
            &Block<real_t>::destroy,
            &storage,
            &Block<real_t>::save,
            &Block<real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), -1);   // number of blocks set by read_blocks()

    // read MFA model
    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block<real_t>::load);
    int nblocks = master.size();
    std::cout << nblocks << " blocks read from file "<< infile << "\n";

    // compute derivative
    fprintf(stderr, "\nComputing derivative...\n");
    double decode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->differentiate_block(cp, deriv, partial); });
    decode_time = MPI_Wtime() - decode_time;

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach(&Block<real_t>::print_deriv);
    fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    diy::io::write_blocks("deriv.out", world, master);
}
