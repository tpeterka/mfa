//--------------------------------------------------------------
// example of computing line integrals from an encoded MFA
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
#include "example-setup.hpp"


using namespace std;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                        // number of local blocks
    int tot_blocks  = nblocks * world.size();   // number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int    pt_dim       = 3;                    // dimension of input points
    int    dom_dim      = 2;                    // dimension of domain (<= pt_dim)
    int    geom_degree  = 1;                    // degree for geometry (same for all dims)
    int    vars_degree  = 4;                    // degree for science variables (same for all dims)
    int    ndomp        = 100;                  // input number of domain points (same for all dims)
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    vector<int>    vars_nctrl   = {11};                   // input number of control points for all science variables (same for all dims)
    string input        = "sinc";               // input dataset
    int    weighted     = 0;                    // solve for and use weights (bool 0/1)
    string infile;                              // input file name
    int    structured   = 1;                    // input data format (bool 0/1)
    int    rand_seed    = -1;                   // seed to use for random data generation (-1 == no randomization)
    float  regularization = 0;                  // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int    reg1and2     = 0;                       // flag for regularizer: 0 --> regularize only 2nd derivs. 1 --> regularize 1st and 2nd
    bool   help         = false;                // show help

    const int verbose = 1;
    const int scalar = 1;


    // get command line arguments
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
    ops >> opts::Option('f', "infile",      infile,     " input file name");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");

    int n_alpha = 0;
    int n_rho = 0;
    int n_samples = 0;
    int v_alpha = 0;
    int v_rho = 0;
    int v_samples = 0;
    int num_ints = 0;
    ops >> opts::Option('z', "n_alpha", n_alpha, " number of rotational samples for line integration");
    ops >> opts::Option('z', "n_rho", n_rho, " number of samples in offset direction for line integration");
    ops >> opts::Option('z', "n_samples", n_samples, " number of samples along ray for line integration");
    ops >> opts::Option('z', "v_alpha", v_alpha, " number of rotational control points for line integration");
    ops >> opts::Option('z', "v_rho", v_rho, " number of control points in offset direction for line integration");
    ops >> opts::Option('z', "v_samples", v_samples, " number of control points along ray for line integration");
    ops >> opts::Option('z', "num_ints", num_ints, " number of random line integrals to compute");


    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // print input arguments
    echo_mfa_settings("line int example", pt_dim, dom_dim, 1, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        regularization, reg1and2, weighted, false, 0, 0);
    echo_data_settings(ndomp, 0, input, infile, 0, 0, 0, structured, rand_seed);

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
                input, infile, ndomp, structured, rand_seed, 0, 0, 0,
                weighted, reg1and2, regularization, false, verbose, mfa_info, d_args);

    // Create data set for modeling. Input keywords are defined in example-setup.hpp
    if (analytical_signals.count(input) == 1)
    {
        master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->generate_analytical_data(cp, input, mfa_info, d_args); 
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
    }

    // compute the MFA
    fprintf(stderr, "\nStarting fixed encoding...\n\n");
    double encode_time = MPI_Wtime();
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, mfa_info); });
    encode_time = MPI_Wtime() - encode_time;
    fprintf(stderr, "\n\nFixed encoding done.\n\n");

    // debug: compute error field for visualization and max error to verify that it is below the threshold
    double decode_time = MPI_Wtime();
    fprintf(stderr, "\nFinal decoding and computing max. error...\n");
    bool saved_basis = structured; // TODO: basis functions are currently only saved during encoding of structured data
    master.foreach([&](Block<real_t>* b, const diy::Master::ProxyWithLink& cp)
            { 
                b->range_error(cp, true, saved_basis);
                b->print_block(cp, true);

                // Assumes one scalar science variable. Used for relative error metric
                real_t extent = b->input->domain.col(dom_dim).maxCoeff() - b->input->domain.col(dom_dim).minCoeff();

                b->create_ray_model(cp, mfa_info, d_args, 1, n_samples, n_rho, n_alpha, v_samples, v_rho, v_alpha);

                real_t result = 0;
                VectorX<real_t> start_pt(2), end_pt(2);

                // horizontal line where function is identically 0
                // = 0.0
                start_pt(0) = -3*M_PI; start_pt(1) = 0;
                end_pt(0) = 3*M_PI; end_pt(1) = 0;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(-3pi, 0)---(3pi, 0):       " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "error:                      " << result << endl << endl;

                // vertical line
                // = 0.0
                start_pt(0) = M_PI/2; start_pt(1) = -2*M_PI;
                end_pt(0) = M_PI/2; end_pt(1) = 2*M_PI;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(pi/2, -2pi)---(pi/2, 2pi): " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "error:                      " << result << endl << endl;
                
                // horizontal line
                // = 2.0
                start_pt(0) = 0; start_pt(1) = M_PI/2;
                end_pt(0) = M_PI; end_pt(1) = M_PI/2;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(0, pi/2)---(pi, pi/2):     " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "relative error:             " << abs((result-2)/2) << endl << endl;

                // line y=x
                // = 5.75864344326
                start_pt(0) = 0, start_pt(1) = 0;
                end_pt(0) = 8, end_pt(1) = 8;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(0, 0)---(8, 8):            " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "relative error:             " << abs((result-5.75864344326)/5.75864344326) << endl << endl;

                // "arbitrary" line
                // = 1.2198958397433
                start_pt(0) = -2; start_pt(1) = -4;
                end_pt(0) = 3; end_pt(1) = 11;
                result = b->integrate_ray(cp, start_pt, end_pt, 1);
                cerr << "(-2, -4)---(3, 11):         " << result << endl;
                cerr << "actual:                     " << sintest(start_pt, end_pt) << endl;
                cerr << "relative error:             " << abs((result-1.2198958397433)/1.2198958397433) << endl << endl;

                std::vector<real_t> ierrs_abs;
                std::vector<real_t> ierrs_rel;
                std::random_device dev;
                std::mt19937 rng(dev());
                real_t x0, x1, y0, y1;
                real_t ierror_abs=0, ierror_rel=0, actual=0, rms_abs=0, rms_rel=0, avg_abs=0, avg_rel=0, len=0;
                std::uniform_real_distribution<double> dist(0,1); 
                for (int i = 0; i < num_ints; i++)
                {
                    x0 = dist(rng)* 8*M_PI - 4*M_PI;
                    y0 = dist(rng)* 8*M_PI - 4*M_PI;
                    x1 = dist(rng)* 8*M_PI - 4*M_PI;
                    y1 = dist(rng)* 8*M_PI - 4*M_PI;
                    start_pt(0) = x0;
                    start_pt(1) = y0;
                    end_pt(0) = x1;
                    end_pt(1) = y1;
                    len = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
                    result = b->integrate_ray(cp, start_pt, end_pt, 1) / len;   // normalize by segment length
                    actual = sintest(start_pt, end_pt) / len;                        // normalize by segment length
                    ierror_abs = abs(result - actual);
                    ierror_rel = ierror_abs/extent;
                    ierrs_abs.push_back(ierror_abs);
                    ierrs_rel.push_back(ierror_rel);
                    // cerr << "(" << x0 << ", " << y0 << ")---(" << x1 << ", " << y1 << "):  " << endl;
                    // cerr << "  Result: " << setprecision(6) << result << endl;
                    // cerr << "  Actual: " << setprecision(6) << actual << endl;
                    // cerr << "  Length:    " << setprecision(3) << len << endl;
                    // cerr << "  Error (abs):  " << setprecision(6) << ierror_abs << endl;
                    // cerr << "  Error (rel):  " << setprecision(6) << ierror_rel << endl;
                }

                cerr << "\nComputed " << num_ints << " random line integrals." << endl;
                cerr << "  Max error (abs): " << setprecision(6) << *max_element(ierrs_abs.begin(), ierrs_abs.end()) << 
                            "\t" << "Max error (rel): " << *max_element(ierrs_rel.begin(), ierrs_rel.end()) << endl;
                cerr << "  Min error (abs): " << setprecision(6) << *min_element(ierrs_abs.begin(), ierrs_abs.end()) << 
                            "\t" << "Min error (rel): " << *min_element(ierrs_rel.begin(), ierrs_rel.end()) << endl;
                for (int j = 0; j < ierrs_abs.size(); j++)
                {
                    rms_abs += ierrs_abs[j] * ierrs_abs[j];
                    rms_rel += ierrs_rel[j] * ierrs_rel[j];
                    avg_abs += ierrs_abs[j];
                    avg_rel += ierrs_rel[j];
                }
                rms_abs = rms_abs/ierrs_abs.size();
                rms_abs = sqrt(rms_abs);
                rms_rel = rms_rel/ierrs_rel.size();
                rms_rel = sqrt(rms_rel);
                avg_abs = avg_abs/ierrs_abs.size();
                avg_rel = avg_rel/ierrs_rel.size();
                cerr << "  Avg error (abs): " << setprecision(6) << avg_abs << "\t" << "Avg error (rel): " << avg_rel << endl;
                cerr << "  RMS error (abs): " << setprecision(6) << rms_abs << "\t" << "RMS error (rel): " << rms_rel << endl;

                ofstream errfile_abs, errfile_rel;
                errfile_abs.open("li_errors_abs.txt");
                errfile_rel.open("li_errors_rel.txt");
                for (int i = 0; i < ierrs_abs.size(); i++)
                {
                    errfile_abs << ierrs_abs[i] << endl;
                    errfile_rel << ierrs_rel[i] << endl;
                }
                errfile_abs.close();
                errfile_rel.close();

                ofstream segmenterrfile;
                segmenterrfile.open("seg_errors.txt");
                int test_n_alpha = 150;
                int test_n_rho = 150;
                real_t r_lim = b->bounds_maxs(1);   // WARNING TODO: make r_lim query-able in RayMFA class
                for (int i = 0; i < test_n_alpha; i++)
                {
                    for (int j = 0; j < test_n_rho; j++)
                    {
                        real_t alpha = 3.14159265 / (test_n_alpha-1) * i;
                        real_t rho = r_lim*2 / (test_n_rho-1) * j - r_lim;
                        real_t x0, x1, y0, y1;   // end points of full line

                        b->get_box_intersections(alpha, rho, x0, y0, x1, y1, b->box_mins, b->box_maxs);
                        if (x0==0 && y0==0 && x1==0 && y1==0)
                        {
                            segmenterrfile << alpha << " " << rho << " " << " 0 0" << endl;
                        }
                        else
                        {
                            real_t length = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
                            start_pt(0) = x0; 
                            start_pt(1) = y0;
                            end_pt(0) = x1;
                            end_pt(1) = y1;

                            real_t test_result = b->integrate_ray(cp, start_pt, end_pt, 1) / length;   // normalize by segment length
                            real_t test_actual = sintest(start_pt, end_pt) / length;

                            real_t e_abs = abs(test_result - test_actual);
                            real_t e_rel = e_abs/extent;

                            segmenterrfile << alpha << " " << rho << " 0 " << e_rel << endl;
                        }
                        
                    }
                }
                segmenterrfile.close();
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