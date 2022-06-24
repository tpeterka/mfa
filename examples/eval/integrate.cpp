//--------------------------------------------------------------
// example of computing a definite integral over an MFA
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

    // default command line arguments
    int MAX_DIM     = 10;                       // maximum domain dimensionality (temporary, only for input parameters)
    vector<real_t> lower_lim(MAX_DIM, 0);       // lower limits of integration in parameter space (zeros by default)
    vector<real_t> upper_lim(MAX_DIM, 1);       // upper limits of integration in parameter space (ones by default)  
    string infile   = "approx.mfa";             // diy input file 
    int  deriv      = 1;                        // which derivative to take (1st, 2nd, ...)
    bool help;                                  // show help

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('a', "lower_lim",   lower_lim,  " lower limits of integration in parameter space");
    ops >> opts::Option('b', "upper_lim",   upper_lim,  " upper limits of integration in parameter space");
    ops >> opts::Option('f', "infile",      infile,     " diy input file name");
    ops >> opts::Option('h', "help",        help,       " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr << "lower_lim (first two dims.) = [ " << lower_lim[0] << " " << lower_lim[1] << " ]" << endl;
    cerr << "upper_lim (first two dims.) = [ " << upper_lim[0] << " " << upper_lim[1] << " ]" << endl;
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
    VectorX<real_t> lim_a(dom_dim);
    VectorX<real_t> lim_b(dom_dim);
    for (auto i = 0; i < dom_dim; i++)
    {
        lim_a(i) = lower_lim[i];
        lim_b(i) = upper_lim[i];
    }
    cerr << "Lower integration limit: [ " << lim_a.transpose() << " ]" << endl;
    cerr << "Upper integration limit: [ " << lim_b.transpose() << " ]" << endl;

    // evaluate point
    VectorX<real_t> int_val(pt_dim - dom_dim);
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->definite_integral(cp, 1, int_val, lim_a, lim_b); });
    cerr << "Integrated value: [ " << int_val.transpose() << " ]" << endl;
}
