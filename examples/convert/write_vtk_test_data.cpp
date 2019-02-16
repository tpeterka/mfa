//--------------------------------------------------------------
// generates test data for analytical functions and writes to VTK format
//
// output precision is float irrespective whether input is float or double
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include    "mfa/mfa.hpp"
#include    <iostream>
#include    <stdio.h>

#include    <diy/master.hpp>
#include    <diy/io/block.hpp>

#include    <diy/../../examples/opts.h>

#include    "writer.hpp"
#include    "block.hpp"

void test_and_write(Block<real_t>*                      b,
                    const diy::Master::ProxyWithLink&   cp,
                    string                              input,
                    DomainArgs&                         args)
{
    int                         nvars;              // number of science variables
    vector<int>                 ntest_pts;          // number of test points in each dim.
    vector<vec3d>               true_pts;           // locations of true points (<= 3d) (may include data value in 2nd or 3rd coord)
    vector<vec3d>               test_pts;           // locations of test points (<= 3d) (may include data value in 2nd or 3rd coord)
    float**                     true_data;          // true data values (4d)
    float**                     test_data;          // test data values (4d)

    DomainArgs* a   = &args;

    nvars = b->vars.size();
    if (!b->dom_dim)
        b->dom_dim =  b->ndom_pts.size();

    // default args for evaluating analytical functions
    for (auto i = 0; i < nvars; i++)
    {
        a->f[i] = 1.0;
        if (input == "sine")
            a->s[i] = i + 1;
        if (input == "sinc")
            a->s[i] = 10.0 * (i + 1);
    }
    a->dom_dim = b->dom_dim;

    // number of test points
    size_t tot_ntest = 1;
    for (auto j = 0; j < b->dom_dim; j++)
    {
        ntest_pts.push_back(a->ndom_pts[j]);
        tot_ntest *= a->ndom_pts[j];
    }

    true_pts.resize(tot_ntest);
    test_pts.resize(tot_ntest);

    // allocate variable data
    true_data = new float*[nvars];
    test_data = new float*[nvars];
    for (size_t j = 0; j < nvars; j++)
    {
        true_data[j] = new float[tot_ntest];
        test_data[j] = new float[tot_ntest];
    }

    // compute the norms of analytical errors synthetic function w/o noise at different domain points than the input
    real_t L1, L2, Linf;                                // L-1, 2, infinity norms
    b->analytical_error(cp, input, L1, L2, Linf, args, false, true, true_pts, true_data, test_pts, test_data);

    // print analytical errors
    fprintf(stderr, "\n------ Analytical error norms -------\n");
    fprintf(stderr, "L-1        norm = %e\n", L1);
    fprintf(stderr, "L-2        norm = %e\n", L2);
    fprintf(stderr, "L-infinity norm = %e\n", Linf);
    fprintf(stderr, "-------------------------------------\n\n");

    // pad dimensions up to 3
    for (auto i = 0; i < 3 - b->dom_dim; i++)
        ntest_pts.push_back(1);

    int* vardims        = new int[nvars];
    char** varnames     = new char*[nvars];
    int* centerings     = new int[nvars];
    float* vars;
    for (int i = 0; i < nvars; i++)
    {
        vardims[i]      = 1;                                // TODO; treating each variable as a scalar (for now)
        varnames[i]     = new char[256];
        centerings[i]   = 1;
        sprintf(varnames[i], "var%d", i);
    }

    // write true points
    write_curvilinear_mesh(
            /* const char *filename */                  "true_points.vtk",
            /* int useBinary */                         0,
            /* int *dims */                             &ntest_pts[0],
            /* float *pts */                            &(true_pts[0].x),
            /* int nvars */                             nvars,
            /* int *vardim */                           vardims,
            /* int *centering */                        centerings,
            /* const char * const *varnames */          varnames,
            /* float **vars */                          true_data);

    // write test points
    write_curvilinear_mesh(
            /* const char *filename */                  "test_points.vtk",
            /* int useBinary */                         0,
            /* int *dims */                             &ntest_pts[0],
            /* float *pts */                            &(test_pts[0].x),
            /* int nvars */                             nvars,
            /* int *vardim */                           vardims,
            /* int *centering */                        centerings,
            /* const char * const *varnames */          varnames,
            /* float **vars */                          test_data);


    delete[] vardims;
    for (int i = 0; i < nvars; i++)
        delete[] varnames[i];
    delete[] varnames;
    delete[] centerings;
    for (int j = 0; j < nvars; j++)
    {
        delete[] true_data[j];
        delete[] test_data[j];
    }
    delete[] true_data;
    delete[] test_data;
}

int main(int argc, char ** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);       // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;                 // equivalent of MPI_COMM_WORLD

    int                         nvars;              // number of science variables (excluding geometry)
    vector<int>                 nraw_pts;           // number of input points in each dim.
    vector<vec3d>               raw_pts;            // input raw data points (<= 3d)
    float**                     raw_data;           // input raw data values (4d)
    vector <int>                geom_nctrl_pts;     // number of control pts in each dim of geometry
    vector < vector <int> >     vars_nctrl_pts;     // number of control pts in each dim. of each science variable
    vector<vec3d>               geom_ctrl_pts;      // control points (<= 3d) in geometry
    vector < vector <vec3d> >   vars_ctrl_pts;      // control points (<= 3d) in science variables
    float**                     vars_ctrl_data;     // control point data values (4d)
    vector<vec3d>               approx_pts;         // aproximated data points (<= 3d)
    float**                     approx_data;        // approximated data values (4d)
    vector<vec3d>               err_pts;            // abs value error field
    string                      input  = "sine";        // input dataset
    int                         ntest  = 0;             // number of input test points in each dim for analytical error tests
    string                      infile = "approx.out";  // diy input file
    bool                        help;                   // show help

    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('f', "infile",      infile,     " diy input file name");
    ops >> opts::Option('a', "ntest",       ntest,      " number of test points in each dimension of domain (for analytical error calculation)");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('h', "help",        help,       " show help");

    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // echo args
    fprintf(stderr, "\n--------- Input arguments ----------\n");
    cerr <<
        " test_points = "    << ntest        << " input = "        << input     << endl;
#ifdef MFA_NO_TBB
    cerr << "TBB: off" << endl;
#else
    cerr << "TBB: on" << endl;
#endif
    fprintf(stderr, "-------------------------------------\n\n");

    if (ntest <= 0)
        exit(0);

    // initialize DIY
    diy::FileStorage storage("./DIY.XXXXXX");     // used for blocks to be moved out of core
    diy::Master      master(world,
            1,
            -1,
            &Block<real_t>::create,
            &Block<real_t>::destroy);
    diy::ContiguousAssigner   assigner(world.size(), -1); // number of blocks set by read_blocks()

    diy::io::read_blocks(infile.c_str(), world, assigner, master, &Block<real_t>::load);
    std::cout << master.size() << " blocks read from file "<< infile << "\n\n";

    // arguments for analytical functions
    DomainArgs d_args;

    if (input == "sine")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
        }
    }

    // sinc function f(x) = sin(x)/x, f(x,y) = sinc(x)sinc(y), ...
    if (input == "sinc")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]               = -4.0 * M_PI;
            d_args.max[i]               = 4.0  * M_PI;
        }
    }

    // f16 function
    if (input == "f16")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]               = -1.0;
            d_args.max[i]               = 1.0;
        }
    }

    // f17 function
    if (input == "f17")
    {
        d_args.min[0] = 80.0;   d_args.max[0] = 100.0;
        d_args.min[1] = 5.0;    d_args.max[1] = 10.0;
        d_args.min[2] = 90.0;   d_args.max[2] = 93.0;
    }

    // f18 function
    if (input == "f18")
    {
        for (int i = 0; i < MAX_DIM; i++)
        {
            d_args.min[i]               = -0.95;
            d_args.max[i]               = 0.95;
        }
    }

    // compute the norms of analytical errors of synthetic function w/o noise at test points
    // and write true points and test points to vtk
    for (int i = 0; i < MAX_DIM; i++)
        d_args.ndom_pts[i] = ntest;

    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { test_and_write(b, cp, input, d_args); });
}
