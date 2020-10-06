//--------------------------------------------------------------
// example of evaluating and optionally differentiating a single point of an MFA
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

using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    string infile = "approx.out";               // diy input file

    // default command line arguments
    int MAX_DIM     = 10;                       // maximum domain dimensionality (temporary, only for input parameter)
    vector<real_t> param(MAX_DIM);              // point to evaluate, initialized to 0s by default
    int  deriv     = 1;                         // which derivative to take (1st, 2nd, ...)
    bool help;                                  // show help

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('p', "param",   param,   " parameters of point to evaluate");
    ops >> opts::Option('d', "deriv",   deriv,   " which derivative to take (1 = 1st, 2 = 2nd, ...)");
    ops >> opts::Option('f', "infile",  infile,  " diy input file name");
    ops >> opts::Option('h', "help",    help,    " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr << "param (first two dims.) = [ " << param[0] << " " << param[1] << " ]" << endl;
    cerr << "deriv   = "    << deriv << endl;
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
    fprintf(stderr, "-------------------------------------\n\n");

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX");               // used for blocks to be moved out of core
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
    int dom_dim = master.block<Block<real_t>>(0)->dom_dim;  // dimensionality of input domain
    int pt_dim  = master.block<Block<real_t>>(0)->pt_dim;   // dimensionality of output point

    // parameters of input point to evaluate
    VectorX<real_t> in_param(dom_dim);
    for (auto i = 0; i < dom_dim; i++)
        in_param(i) = param[i];
    cerr << "Input point parameters: [ " << in_param.transpose() << " ]" << endl;

    // evaluate point
    VectorX<real_t> out_pt(pt_dim);
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->decode_point(cp, in_param, out_pt); });
    cerr << "Output point: [ " << out_pt.transpose() << " ]" << endl;

    // evaluate total derivative
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->differentiate_point(cp, in_param, deriv, -1, -1, out_pt); });
    cerr << "Total derivative at output point: " << out_pt(pt_dim - 1) << endl;

    // evaluate partial derivatives
    for (auto i = 0; i < dom_dim; i++)
    {
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
                { b->differentiate_point(cp, in_param, deriv, i, -1, out_pt); });
        cerr << "Partial derivative wrt dimension " << i << " at output point: " << out_pt(pt_dim - 1) << endl;
    }
}
