//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include <stdio.h>

#include <Eigen/Dense>

#define MAX_DIM 8                           // a user limit, not mfa's

using namespace std;

typedef Eigen::MatrixXf                MatrixXf;
typedef Eigen::VectorXf                VectorXf;
typedef MatrixXf::Index                Index;

typedef diy::ContinuousBounds          Bounds;
typedef diy::RegularContinuousLink     RCLink;

// arguments to block foreach functions
struct DomainArgs
{
    int   pt_dim;                            // dimension of points
    int   dom_dim;                           // dimension of domain (<= pt_dim)
    int   p[MAX_DIM];                        // degree in each dimension of domain
    int   ndom_pts[MAX_DIM];                 // number of input points in each dimension of domain
    int   nctrl_pts[MAX_DIM];                // number of input points in each dimension of domain
    float min[MAX_DIM];                      // minimum corner of domain
    float max[MAX_DIM];                      // maximum corner of domain
    float s;                                 // scaling factor or any other usage
};

struct ErrArgs
{
    int   max_niter;                         // max num iterations to search for nearest curve pt
    float err_bound;                         // desired error bound (stop searching if less)
    int   search_rad;                        // number of parameter steps to search path on either
    // side of parameter value of input point
};

// block
struct Block
{
    Block(int point_dim)
        {
            domain_mins.resize(point_dim);
            domain_maxs.resize(point_dim);
        }
    static
    void* create()
        {
            return new Block(2);
        }
    static
    void destroy(void* b)
        {
            delete static_cast<Block*>(b);
        }
    static
    void save(const void* b_, diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;

            diy::save(bb, b->ndom_pts);
            diy::save(bb, b->domain);
            diy::save(bb, b->domain_mins);
            diy::save(bb, b->domain_maxs);
            diy::save(bb, b->p);
            diy::save(bb, b->nctrl_pts);
            diy::save(bb, b->ctrl_pts);
            diy::save(bb, b->knots);
            diy::save(bb, b->approx);
            diy::save(bb, b->errs);
        }
    static
    void load(void* b_, diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;

            diy::load(bb, b->ndom_pts);
            diy::load(bb, b->domain);
            diy::load(bb, b->domain_mins);
            diy::load(bb, b->domain_maxs);
            diy::load(bb, b->p);
            diy::load(bb, b->nctrl_pts);
            diy::load(bb, b->ctrl_pts);
            diy::load(bb, b->knots);
            diy::load(bb, b->approx);
            diy::load(bb, b->errs);
        }
    // f(x,y,z,...) = 1
    void generate_constant_data(const diy::Master::ProxyWithLink& cp,
                                DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);

            // assign values to the domain (geometry)
            int cs = 1;                  // stride of a coordinate in this dim
            for (int i = 0; i < a->dom_dim; i++) // all dimensions in the domain
            {
                float d = (a->max[i] - a->min[i]) / (ndom_pts(i) - 1);
                int k = 0;
                int co = 0;                  // j index of start of a new coordinate value
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    if (a->min[i] + k * d > a->max[i])
                        k = 0;
                    domain(j, i) = a->min[i] + k * d;
                    if (j + 1 - co >= cs)
                    {
                        k++;
                        co = j + 1;
                    }
                }
                cs *= ndom_pts(i);
            }

            // assign values to the range (physics attributes)
            for (int i = a->dom_dim; i < a->pt_dim; i++)
            {
                // the simplest constant function
                for (int j = 0; j < tot_ndom_pts; j++)
                    domain(j, i) = 1.0;
            }

            // extents
            for (int i = 0; i < a->pt_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
        }

    // f(x,y,z,...) = x
    void generate_ramp_data(const diy::Master::ProxyWithLink& cp,
                            DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);

            // assign values to the domain (geometry)
            int cs = 1;                  // stride of a coordinate in this dim
            for (int i = 0; i < a->dom_dim; i++) // all dimensions in the domain
            {
                float d = (a->max[i] - a->min[i]) / (ndom_pts(i) - 1);
                int k = 0;
                int co = 0;                  // j index of start of a new coordinate value
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    if (a->min[i] + k * d > a->max[i])
                        k = 0;
                    domain(j, i) = a->min[i] + k * d;
                    if (j + 1 - co >= cs)
                    {
                        k++;
                        co = j + 1;
                    }
                }
                cs *= ndom_pts(i);
            }

            // assign values to the range (physics attributes)
            for (int i = a->dom_dim; i < a->pt_dim; i++)
            {
                for (int j = 0; j < tot_ndom_pts; j++)
                    domain(j, i) = domain(j, 0);
            }

            // extents
            for (int i = 0; i < a->pt_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
        }

    // f(x,y,z,...) = x^2
    void generate_quadratic_data(const diy::Master::ProxyWithLink& cp,
                                 DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);

            // assign values to the domain (geometry)
            int cs = 1;                  // stride of a coordinate in this dim
            for (int i = 0; i < a->dom_dim; i++) // all dimensions in the domain
            {
                float d = (a->max[i] - a->min[i]) / (ndom_pts(i) - 1);
                int k = 0;
                int co = 0;                  // j index of start of a new coordinate value
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    if (a->min[i] + k * d > a->max[i])
                        k = 0;
                    domain(j, i) = a->min[i] + k * d;
                    if (j + 1 - co >= cs)
                    {
                        k++;
                        co = j + 1;
                    }
                }
                cs *= ndom_pts(i);
            }

            // assign values to the range (physics attributes)
            for (int i = a->dom_dim; i < a->pt_dim; i++)
            {
                for (int j = 0; j < tot_ndom_pts; j++)
                    domain(j, i) = domain(j, 0) * domain(j, 0);
            }

            // extents
            for (int i = 0; i < a->pt_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
        }

    // f(x,y,z,...) = sqrt(x^2 + y^2 + z^2 + ...^2)
    void generate_magnitude_data(const diy::Master::ProxyWithLink& cp,
                                 DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);

            // assign values to the domain (geometry)
            int cs = 1;                  // stride of a coordinate in this dim
            for (int i = 0; i < a->dom_dim; i++) // all dimensions in the domain
            {
                float d = (a->max[i] - a->min[i]) / (ndom_pts(i) - 1);
                int k = 0;
                int co = 0;                  // j index of start of a new coordinate value
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    if (a->min[i] + k * d > a->max[i])
                        k = 0;
                    domain(j, i) = a->min[i] + k * d;
                    if (j + 1 - co >= cs)
                    {
                        k++;
                        co = j + 1;
                    }
                }
                cs *= ndom_pts(i);
            }

            // assign values to the range (physics attributes)
            for (int i = a->dom_dim; i < a->pt_dim; i++)
            {
                // magnitude function
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    VectorXf one_pt = domain.block(j, 0, 1, a->dom_dim).row(0);
                    domain(j, i) = one_pt.norm();
                }
            }

            // extents
            for (int i = 0; i < a->pt_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
            domain_mins(a->pt_dim - 1) = domain(0               , a->pt_dim - 1);
            domain_maxs(a->pt_dim - 1) = domain(tot_ndom_pts - 1, a->pt_dim - 1);
            // cerr << "domain_maxs:\n" << domain_maxs << endl;
        }

    // f(x,y,z,...) = sqrt(r^2 - x^2 - y^2 - z^2 - ...^2)
    void generate_sphere_data(const diy::Master::ProxyWithLink& cp,
                              DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);

            // assign values to the domain (geometry)
            int cs = 1;                  // stride of a coordinate in this dim
            for (int i = 0; i < a->dom_dim; i++) // all dimensions in the domain
            {
                float d = (a->max[i] - a->min[i]) / (ndom_pts(i) - 1);
                int k = 0;
                int co = 0;                  // j index of start of a new coordinate value
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    if (a->min[i] + k * d > a->max[i])
                        k = 0;
                    domain(j, i) = a->min[i] + k * d;
                    if (j + 1 - co >= cs)
                    {
                        k++;
                        co = j + 1;
                    }
                }
                cs *= ndom_pts(i);
            }

            // assign values to the range (physics attributes)
            for (int i = a->dom_dim; i < a->pt_dim; i++)
            {
                // sphere function
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    VectorXf one_pt = domain.block(j, 0, 1, a->dom_dim).row(0);
                    float r = a->s;           // shere radius
                    if (r * r - one_pt.squaredNorm() < 0)
                    {
                        fprintf(stderr, "Error: radius is not large enough for domain points\n");
                        exit(0);
                    }
                    domain(j, i) = sqrt(r * r - one_pt.squaredNorm());
                }
            }

            // extents
            for (int i = 0; i < a->pt_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
            domain_mins(a->pt_dim - 1) = domain(0               , a->pt_dim - 1);
            domain_maxs(a->pt_dim - 1) = domain(tot_ndom_pts - 1, a->pt_dim - 1);
            // cerr << "domain_maxs:\n" << domain_maxs << endl;
        }

    // y = sine(x)/x
    void generate_sinc_data(const diy::Master::ProxyWithLink& cp,
                            DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);
            s = a->s;

            // assign values to the domain (geometry)
            int cs = 1;                           // stride of a coordinate in this dim
            float eps = 1.0e-5;                   // floating point roundoff error
            for (int i = 0; i < a->dom_dim; i++)  // all dimensions in the domain
            {
                float d = (a->max[i] - a->min[i]) / (ndom_pts(i) - 1);
                int k = 0;
                int co = 0;                       // j index of start of a new coordinate value
                for (int j = 0; j < tot_ndom_pts; j++)
                {
                    if (a->min[i] + k * d > a->max[i] + eps)
                        k = 0;
                    domain(j, i) = a->min[i] + k * d;
                    if (j + 1 - co >= cs)
                    {
                        k++;
                        co = j + 1;
                    }
                }
                cs *= ndom_pts(i);
            }

            float min, max;                       // extents of range

            // assign values to the range (physics attributes)
            // f(x,y,z,...) = sine(x)/x * sine(y)/y * sine(z)/z * ...
            for (int j = 0; j < tot_ndom_pts; j++)
            {
                float res = 1.0;                  // product of the sinc functions
                for (int i = 0; i < a->dom_dim; i++)
                {
                    if (domain(j, i) != 0.0)
                        res *= (sin(domain(j, i)) / domain(j, i));
                }
                res *= a->s;

                for (int i = a->dom_dim; i < a->pt_dim; i++)
                    domain(j, i) = res;

                if (j == 0 || res > max)
                    max = res;
                if (j == 0 || res < min)
                    min = res;
            }

            // extents
            for (int i = 0; i < a->dom_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
            domain_mins(a->pt_dim - 1) = min;
            domain_maxs(a->pt_dim - 1) = max;

            cerr << "domain_mins:\n" << domain_mins << endl;
            cerr << "domain_maxs:\n" << domain_maxs << endl;
        }

    // read the flame dataset and take one slice out of the middle of it
    // doubling the resolution because the file I have is a 1/2-resolution downsampled version
    // TODO: only for 1d so far
    void read_1d_file_data_double(const diy::Master::ProxyWithLink& cp,
                           DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  2 * a->ndom_pts[i]; // double resolution
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);

            // rest is hard-coded for 1d
            vector<float> vel(3 * a->ndom_pts[0]);

            // open hard-coded file name, seek to hard-coded start of desired section
            // open hard-coded file name, seek to hard-coded start of desired section
            // file is 704 * 540 * 550 * 3 floats (vx,vy,vz)
            // which is 1/2 the x resolution the simulation
            // the "small" in the file name means it was downsampled by factor of 2 in x
//             FILE *fd = fopen("/Users/tpeterka/datasets/flame/6_small.xyz", "r");
            FILE *fd = fopen("/homes/tpeterka/datasets/flame/6_small.xyz", "r");
            assert(fd);
            fseek(fd, (704 * 540 * 275 + 704 * 270) * 12, SEEK_SET);

            // read all three components of velocity and compute magnitude
            if (!fread(&vel[0], sizeof(float), a->ndom_pts[0] * 3, fd))
            {
                fprintf(stderr, "Error: unable to open file\n");
                exit(0);
            }
            for (size_t i = 0; i < vel.size() / 3; i++)
            {
                domain(2 * i, 1) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                                        vel[3 * i + 1] * vel[3 * i + 1] +
                                        vel[3 * i + 2] * vel[3 * i + 2]);
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }
            // add an interpolated velocity magnitude between each two velocity magnitudes
            for (size_t i = 0; i < (size_t)domain.rows() - 1; i++)
            {
                if (i % 2)
                    domain(i, 1) = (domain(i - 1, 1) + domain(i + 1, 1)) / 2.0;
            }
            domain(domain.rows() - 1, 1) = domain(domain.rows() - 2, 1); // duplicate last value

            // find extent of range
            domain_mins(1) = domain(0, 1);
            domain_maxs(1) = domain(0, 1);
            for (size_t i = 1; i < (size_t)domain.rows(); i++)
            {
                if (domain(i, 1) < domain_mins(1))
                    domain_mins(1) = domain(i, 1);
                if (domain(i, 1) > domain_maxs(1))
                    domain_maxs(1) = domain(i, 1);
            }

            // scale domain to same size as range, from 0 to range_max
            float dx = (domain_maxs(1) - domain_mins(1)) / (domain.rows() - 1);
            for (size_t i = 1; i < (size_t)domain.rows(); i++)
                domain(i, 0) = i * dx;

            // extents
            domain_mins(0) = 0.0;
            domain_maxs(0) = domain_maxs(1) - domain_mins(1);

            // debug
            cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
        }

    // read the flame dataset and take one slice out of the middle of it
    // TODO: only for 1d so far
    void read_1d_file_data(const diy::Master::ProxyWithLink& cp,
                           DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);
            vector<float> vel(3 * tot_ndom_pts);

            // rest is hard-coded for 1d

            // open hard-coded file name, seek to hard-coded start of desired section
            // open hard-coded file name, seek to hard-coded start of desired section
            // file is 704 * 540 * 550 * 3 floats (vx,vy,vz)
//             FILE *fd = fopen("/Users/tpeterka/datasets/flame/6_small.xyz", "r");
            FILE *fd = fopen("/homes/tpeterka/datasets/flame/6_small.xyz", "r");
            assert(fd);
            fseek(fd, (704 * 540 * 275 + 704 * 270) * 12, SEEK_SET);

            // read all three components of velocity and compute magnitude
            if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
            {
                fprintf(stderr, "Error: unable to open file\n");
                exit(0);
            }
            for (size_t i = 0; i < vel.size() / 3; i++)
            {
                domain(i, 1) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                        vel[3 * i + 1] * vel[3 * i + 1] +
                        vel[3 * i + 2] * vel[3 * i + 2]);
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }

            // find extent of range
            for (size_t i = 0; i < (size_t)domain.rows(); i++)
            {
                if (i == 0 || domain(i, 1) < domain_mins(1))
                    domain_mins(1) = domain(i, 1);
                if (i == 0 || domain(i, 1) > domain_maxs(1))
                    domain_maxs(1) = domain(i, 1);
            }

            // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
            int n = 0;
            for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
            {
                domain(n, 0) = i;
                n++;
            }

            // extents
            domain_mins(0) = 0.0;
            domain_maxs(0) = domain(tot_ndom_pts - 1, 0);

            // debug
            cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
        }

    // read the flame dataset and take one plane out of the middle of it
    // TODO: only for 2d so far
    void read_2d_file_data(const diy::Master::ProxyWithLink& cp,
                           DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);
            vector<float> vel(3 * tot_ndom_pts);

            // rest is hard-coded for 2d

            // open hard-coded file name, seek to hard-coded start of desired section
            // which is an x-y plane in the middle of the z range
            // file is 704 * 540 * 550 * 3 floats (vx,vy,vz)
//             FILE *fd = fopen("/Users/tpeterka/datasets/flame/6_small.xyz", "r");
            FILE *fd = fopen("/homes/tpeterka/datasets/flame/6_small.xyz", "r");
            assert(fd);
            // middle plane in z, offset = full x,y range * 1/2 z range
            fseek(fd, (704 * 540 * 275) * 12, SEEK_SET);

            // read all three components of velocity and compute magnitude
            if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
            {
                fprintf(stderr, "Error: unable to open file\n");
                exit(0);
            }
            for (size_t i = 0; i < vel.size() / 3; i++)
            {
                domain(i, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                                    vel[3 * i + 1] * vel[3 * i + 1] +
                                    vel[3 * i + 2] * vel[3 * i + 2]);
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }

            // find extent of range
            for (size_t i = 0; i < (size_t)domain.rows(); i++)
            {
                if (i == 0 || domain(i, 2) < domain_mins(2))
                    domain_mins(2) = domain(i, 2);
                if (i == 0 || domain(i, 2) > domain_maxs(2))
                    domain_maxs(2) = domain(i, 2);
            }

            // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
            int n = 0;
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                {
                    domain(n, 0) = i;
                    domain(n, 1) = j;
                    n++;
                }

            // extents
            domain_mins(0) = 0.0;
            domain_mins(1) = 0.0;
            domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
            domain_maxs(1) = domain(tot_ndom_pts - 1, 1);

            // debug
            cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
        }

    // read the flame dataset
    // TODO: only for 3d so far
    void read_3d_file_data(const diy::Master::ProxyWithLink& cp,
                           DomainArgs&                       args)
        {
            DomainArgs* a = &args;
            int tot_ndom_pts = 1;
            p.resize(a->dom_dim);
            ndom_pts.resize(a->dom_dim);
            nctrl_pts.resize(a->dom_dim);
            domain_mins.resize(a->pt_dim);
            domain_maxs.resize(a->pt_dim);
            for (int i = 0; i < a->dom_dim; i++)
            {
                p(i)         =  a->p[i];
                ndom_pts(i)  =  a->ndom_pts[i];
                nctrl_pts(i) =  a->nctrl_pts[i];
                tot_ndom_pts *= ndom_pts(i);
            }
            domain.resize(tot_ndom_pts, a->pt_dim);

            vector<float> vel(3 * tot_ndom_pts);

            // open hard-coded file name
            // file is 704 * 540 * 550 * 3 floats (vx,vy,vz)
//             FILE *fd = fopen("/Users/tpeterka/datasets/flame/6_small.xyz", "r");
            FILE *fd = fopen("/homes/tpeterka/datasets/flame/6_small.xyz", "r");
            assert(fd);

            // read all three components of velocity and compute magnitude
            if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
            {
                fprintf(stderr, "Error: unable to open file\n");
                exit(0);
            }
            for (size_t i = 0; i < vel.size() / 3; i++)
            {
                domain(i, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                                    vel[3 * i + 1] * vel[3 * i + 1] +
                                    vel[3 * i + 2] * vel[3 * i + 2]);
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }

            // rest is hard-coded for 3d

            // find extent of range
            for (size_t i = 0; i < (size_t)domain.rows(); i++)
            {
                if (i == 0 || domain(i, 3) < domain_mins(3))
                    domain_mins(3) = domain(i, 3);
                if (i == 0 || domain(i, 3) > domain_maxs(3))
                    domain_maxs(3) = domain(i, 3);
            }

            // set domain values (just equal to i, j; ie, dx, dy = 1, 1)
            int n = 0;
            for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++)
                for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++)
                    for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++)
                    {
                        domain(n, 0) = i;
                        domain(n, 1) = j;
                        domain(n, 2) = k;
                        n++;
                    }

            // extents
            domain_mins(0) = 0.0;
            domain_mins(1) = 0.0;
            domain_mins(2) = 0.0;
            domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
            domain_maxs(1) = domain(tot_ndom_pts - 1, 1);
            domain_maxs(2) = domain(tot_ndom_pts - 1, 2);

            // debug
            cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
        }

    void encode_block(const diy::Master::ProxyWithLink& cp)
        {
            mfa = new mfa::MFA(p, ndom_pts, nctrl_pts, domain, ctrl_pts, knots);
            mfa->Encode();
        }

    // re-encode a block with new knots to be inserted
    void reencode_block(const diy::Master::ProxyWithLink& cp,
                        float                             err_limit,
                        bool&                             done)
        {
            done = mfa->Encode(err_limit);
        }

    void decode_block(const diy::Master::ProxyWithLink& cp)
        {
            approx.resize(domain.rows(), domain.cols());
            mfa->Decode(approx);
        }

    // compute maximum error in the block
    void max_error(const diy::Master::ProxyWithLink& cp)
        {
            // normal distance computation
            size_t max_idx;
            for (size_t i = 0; i < (size_t)approx.rows(); i++)
            {
                VectorXf approx_pos = approx.block(i, 0, 1, p.size()).row(0);
                float err = mfa->Error(i);
                if (i == 0 || fabs(err) > fabs(max_err))
                {
                    max_err = err;
                    max_idx = i;
                }
            }

            // normalize max error by size of input data (domain and range)
            float min = domain.minCoeff();
            float max = domain.maxCoeff();
            float range = max - min;

            // debug
            fprintf(stderr, "data range = %.1f\n", range);
            fprintf(stderr, "raw max_error = %e\n", max_err);
            cerr << "position of max error: idx=" << max_idx << "\n" << domain.row(max_idx) << endl;

            max_err /= range;

            fprintf(stderr, "|normalized max_err| = %e\n", fabs(max_err));
        }

    // compute error field (error at every domain point)
    // error is absolute value but not normalized by data range
    void error(const diy::Master::ProxyWithLink& cp)
        {
            errs.resize(domain.rows(), domain.cols());
            errs = domain;
            for (size_t i = 0; i < (size_t)approx.rows(); i++)
                errs(i, errs.cols() - 1) = mfa->Error(i);
        }

    void print_block(const diy::Master::ProxyWithLink& cp)
        {
            fprintf(stderr, "\n--- Final block results ---\n");
            // cerr << "domain\n" << domain << endl;
//             cerr << ctrl_pts.rows() << " control points\n" << ctrl_pts << endl;
//             cerr << knots.size() << " knots\n" << knots << endl;
            // cerr << approx.rows() << " approximated points\n" << approx << endl;
            fprintf(stderr, "|normalized max_err| = %e\n", fabs(max_err));
            fprintf(stderr, "# input points = %ld\n", domain.rows());
            fprintf(stderr, "# output ctrl pts = %ld # output knots = %ld\n",
                    ctrl_pts.rows(), knots.size());
            fprintf(stderr, "compression ratio = %.1f\n",
                    (float)(domain.rows()) / (ctrl_pts.rows() + knots.size() / ctrl_pts.cols()));
        }

    VectorXi ndom_pts;                       // number of domain points in each dimension
    MatrixXf domain;                         // input data (1st dim changes fastest)
    VectorXf domain_mins;                    // local domain minimum corner
    VectorXf domain_maxs;                    // local domain maximum corner
    VectorXi p;                              // degree in each dimension
    VectorXi nctrl_pts;                      // number of control points in each dimension
    MatrixXf ctrl_pts;                       // NURBS control points (1st dim changes fastest)
    VectorXf knots;                          // NURBS knots (1st dim changes fastest)
    MatrixXf approx;                         // points in approximated volume
                                             // (same number as input points, for rendering only)
    float    max_err;                        // maximum (abs value) distance from input points to curve
    MatrixXf errs;                           // error field (abs. value, not normalized by data range)

    float s;                                 // scaling factor on range values (for error checking)
    mfa::MFA *mfa;                           // MFA object
};

namespace diy
{
    template<>
    struct Serialization<MatrixXf>
    {
        static
        void save(diy::BinaryBuffer& bb, const MatrixXf& m)
            {
                diy::save(bb, m.rows());
                diy::save(bb, m.cols());
                diy::save(bb, m.data(), m.rows() * m.cols());
            }
        static
        void load(diy::BinaryBuffer& bb, MatrixXf& m)
            {
                Index rows, cols;
                diy::load(bb, rows);
                diy::load(bb, cols);
                m.resize(rows, cols);
                diy::load(bb, m.data(), rows * cols);
            }
    };
    template<>
    struct Serialization<VectorXf>
    {
        static
        void save(diy::BinaryBuffer& bb, const VectorXf& v)
            {
                diy::save(bb, v.size());
                diy::save(bb, v.data(), v.size());
            }
        static
        void load(diy::BinaryBuffer& bb, VectorXf& v)
            {
                Index size;
                diy::load(bb, size);
                v.resize(size);
                diy::load(bb, v.data(), size);
            }
    };
    template<>
    struct Serialization<VectorXi>
    {
        static
        void save(diy::BinaryBuffer& bb, const VectorXi& v)
            {
                diy::save(bb, v.size());
                diy::save(bb, v.data(), v.size());
            }
        static
        void load(diy::BinaryBuffer& bb, VectorXi& v)
            {
                Index size;
                diy::load(bb, size);
                v.resize(size);
                diy::load(bb, v.data(), size);
            }
    };
}
