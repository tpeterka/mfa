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
    int   starts[MAX_DIM];                   // starting offsets of ndom_pts (optional, usually assumed 0)
    int   ndom_pts[MAX_DIM];                 // number of input points in each dimension of domain
    int   full_dom_pts[MAX_DIM];             // number of points in full domain in case a subset is taken
    int   nctrl_pts[MAX_DIM];                // number of input points in each dimension of domain
    float min[MAX_DIM];                      // minimum corner of domain
    float max[MAX_DIM];                      // maximum corner of domain
    float s;                                 // scaling factor or any other usage
    char infile[256];                        // input filename
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
            diy::save(bb, b->span_mins);
            diy::save(bb, b->span_maxs);
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
            diy::load(bb, b->span_mins);
            diy::load(bb, b->span_maxs);
        }
    // f(x,y,z,...) = 1
    void generate_constant_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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
    void generate_ramp_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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
    void generate_quadratic_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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
    void generate_magnitude_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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
    void generate_sphere_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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

    // y = sine(x)
    void generate_sine_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            p(i)         =  a->p[i];
            ndom_pts(i)  =  a->ndom_pts[i];
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
        // f(x,y,z,...) = sine(x) * sine(y) * sine(z) * ...
        for (int j = 0; j < tot_ndom_pts; j++)
        {
            float res = 1.0;                  // product of the sinc functions
            for (int i = 0; i < a->dom_dim; i++)
                    res *= sin(domain(j, i));
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

        //             cerr << "domain:\n" << domain << endl;
    }

    // y = sine(x)/x
    void generate_sinc_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        p.resize(a->dom_dim);
        ndom_pts.resize(a->dom_dim);
        domain_mins.resize(a->pt_dim);
        domain_maxs.resize(a->pt_dim);
        for (int i = 0; i < a->dom_dim; i++)
        {
            p(i)         =  a->p[i];
            ndom_pts(i)  =  a->ndom_pts[i];
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

        //             cerr << "domain:\n" << domain << endl;
    }

    // read a floating point 3d vector dataset and take one 1-d curve out of the middle of it
    // f = (x, velocity magnitude)
    void read_1d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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

        // open file and seek to a slice in the center
        FILE *fd = fopen(a->infile, "r");
        assert(fd);
        fseek(fd, (a->ndom_pts[0] * a->ndom_pts[1] * a->ndom_pts[2] / 2 + a->ndom_pts[0] * a->ndom_pts[1] / 2) * 12, SEEK_SET);

        // read all three components of velocity and compute magnitude
        if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
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

    // read a floating point 3d vector dataset and take one 2-d surface out of the middle of it
    // f = (x, y, velocity magnitude)
    void read_2d_slice_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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

        // open file and seek to a slice in the center
        FILE *fd = fopen(a->infile, "r");
        assert(fd);
        // middle plane in z, offset = full x,y range * 1/2 z range
        fseek(fd, (a->ndom_pts[0] * a->ndom_pts[1] * a->ndom_pts[2] / 2) * 12, SEEK_SET);

        // read all three components of velocity and compute magnitude
        if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
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

    // read a floating point 3d vector dataset and take one 2d (parallel to x-y plane) subset
    // f = (x, y, velocity magnitude)
    void read_2d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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
        vector<float> vel(a->full_dom_pts[0] * a->full_dom_pts[1] * 3);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // rest is hard-coded for 2d

        // seek to start of desired full x-y plane
        size_t ofst = 0;                                                    // offset to seek to (in bytes)
        ofst += a->starts[2] * a->full_dom_pts[0] * a->full_dom_pts[1];     // z direction
        ofst *= 12;                                                         // 3 components * 4 bytes
        fseek(fd, ofst, SEEK_SET);

        // read all three components of velocity for the entire plane (not just the subset)
        if (!fread(&vel[0], sizeof(float), a->full_dom_pts[0] * a->full_dom_pts[1] * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }

        // compute velocity magnitude only for the points in the subset, dropping the rest
        size_t ijk[2] = {0, 0};                          // i,j,k indices of current point
        size_t n = 0;
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            // is the point in the subset?
            bool keep = true;
            if (ijk[0] < a->starts[0] || ijk[0] >= a->starts[0] + a->ndom_pts[0] ||
                    ijk[1] < a->starts[1] || ijk[1] >= a->starts[1] + a->ndom_pts[1])
                keep = false;

            // debug
            //                 fprintf(stderr, "i=%ld ijk=[%ld %ld] keep=%d\n", i, ijk[0], ijk[1], keep);

            if (keep)
            {
                domain(n, 0) = ijk[0];                  // domain is just i,j
                domain(n, 1) = ijk[1];
                // range (function value) is magnitude of velocity
                domain(n, 2) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                        vel[3 * i + 1] * vel[3 * i + 1] +
                        vel[3 * i + 2] * vel[3 * i + 2]);
                n++;
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }

            // increment ijk
            if (ijk[0] == a->full_dom_pts[0] - 1)
            {
                ijk[0] = 0;
                ijk[1]++;
            }
            else
                ijk[0]++;
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 2) < domain_mins(2))
                domain_mins(2) = domain(i, 2);
            if (i == 0 || domain(i, 2) > domain_maxs(2))
                domain_maxs(2) = domain(i, 2);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        domain_mins(0) = domain(0, 0);
        domain_mins(1) = domain(0, 1);
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 3d vector dataset
    // f = (x, y, z, velocity magnitude)
    void read_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // read all three components of velocity and compute magnitude
        if (!fread(&vel[0], sizeof(float), tot_ndom_pts * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
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

    // read a floating point 3d vector dataset and take a 3d subset out of it
    // f = (x, y, z, velocity magnitude)
    void read_3d_subset_3d_vector_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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
        vector<float> vel(a->full_dom_pts[0] * a->full_dom_pts[1] * a->full_dom_pts[2] * 3);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // rest is hard-coded for 3d

        // read all three components of velocity (not just the subset)
        if (!fread(&vel[0], sizeof(float), a->full_dom_pts[0] * a->full_dom_pts[1] * a->full_dom_pts[2] * 3, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }

        // compute velocity magnitude only for the points in the subset, dropping the rest
        size_t ijk[3] = {0, 0, 0};                          // i,j,k indices of current point
        size_t n = 0;
        for (size_t i = 0; i < vel.size() / 3; i++)
        {
            // is the point in the subset?
            bool keep = true;
            if (ijk[0] < a->starts[0] || ijk[0] >= a->starts[0] + a->ndom_pts[0] ||
                    ijk[1] < a->starts[1] || ijk[1] >= a->starts[1] + a->ndom_pts[1] ||
                    ijk[2] < a->starts[2] || ijk[2] >= a->starts[2] + a->ndom_pts[2])
                keep = false;

            // debug
            //                 fprintf(stderr, "i=%ld ijk=[%ld %ld %ld] keep=%d\n", i, ijk[0], ijk[1], ijk[2], keep);

            if (keep)
            {
                domain(n, 0) = ijk[0];                  // domain is just i,j
                domain(n, 1) = ijk[1];
                domain(n, 2) = ijk[2];
                domain(n, 3) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                        vel[3 * i + 1] * vel[3 * i + 1] +
                        vel[3 * i + 2] * vel[3 * i + 2]);
                n++;
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }

            // increment ijk
            if (ijk[0] == a->full_dom_pts[0] - 1)
            {
                ijk[0] = 0;
                if (ijk[1] == a->full_dom_pts[1] - 1)
                {
                    ijk[1] = 0;
                    ijk[2]++;
                }
                else
                    ijk[1]++;
            }
            else
                ijk[0]++;
        }

        // find extent of range
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (i == 0 || domain(i, 3) < domain_mins(3))
                domain_mins(3) = domain(i, 3);
            if (i == 0 || domain(i, 3) > domain_maxs(3))
                domain_maxs(3) = domain(i, 3);
        }

        // extent of domain is just lower left and upper right corner, which in row-major order
        // is the first point and the last point
        domain_mins(0) = domain(0, 0);
        domain_mins(1) = domain(0, 1);
        domain_mins(2) = domain(0, 2);
        domain_maxs(0) = domain(tot_ndom_pts - 1, 0);
        domain_maxs(1) = domain(tot_ndom_pts - 1, 1);
        domain_maxs(2) = domain(tot_ndom_pts - 1, 2);

        // debug
        cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
    }

    // read a floating point 2d scalar dataset
    // f = (x, y, value)
    void read_2d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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

        vector<float> val(tot_ndom_pts);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // read data values
        if (!fread(&val[0], sizeof(float), tot_ndom_pts, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < val.size(); i++)
            domain(i, 2) = val[i];

        // rest is hard-coded for 3d

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

    // read a floating point 3d scalar dataset
    // f = (x, y, z, value)
    void read_3d_scalar_data(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
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

        vector<float> val(tot_ndom_pts);

        FILE *fd = fopen(a->infile, "r");
        assert(fd);

        // read data values
        if (!fread(&val[0], sizeof(float), tot_ndom_pts, fd))
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }
        for (size_t i = 0; i < val.size(); i++)
            domain(i, 3) = val[i];

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

    // fixed number of control points encode block
    void fixed_encode_block(
            const       diy::Master::ProxyWithLink& cp,
            DomainArgs& args)
    {
        DomainArgs* a = &args;
        nctrl_pts.resize(a->dom_dim);
        for (int i = 0; i < a->dom_dim; i++)
            nctrl_pts(i) =  a->nctrl_pts[i];
        mfa = new mfa::MFA(p, ndom_pts, domain, ctrl_pts, nctrl_pts, knots);
        mfa->FixedEncode(nctrl_pts);
    }

    // adaptively encode block to desired error limit
    void adaptive_encode_block(
            const diy::Master::ProxyWithLink& cp,
            float                             err_limit)
    {
        VectorXi unused;
        mfa = new mfa::MFA(p, ndom_pts, domain, ctrl_pts, unused, knots);
        mfa->AdaptiveEncode(err_limit, nctrl_pts);
    }

    // nonlinear encoding of block to desired error limit
    // only for 1D so far
    void nonlinear_encode_block(
            const   diy::Master::ProxyWithLink& cp,
            float   err_limit)
    {
        // set initial control points here
        // TODO: what if there aren'e enough control points (p + 1 is the minimum needed)?
        float grad;                             // current gradient (finite difference)
        float prev_grad = 0.0;                  // previous gradient (finite difference)
        nctrl_pts.resize(1);
        for (auto i = 0; i < domain.rows(); i++)
        {
            if (i == 0 || i == domain.rows() - 1)
            {
                // first and last control points coincide with domain
                ctrl_pts.conservativeResize(ctrl_pts.rows() + 1, domain.cols());
                ctrl_pts.row(ctrl_pts.rows() - 1) = domain.row(i);
            }
            else
            {
                grad = (domain(i, 1) - domain(i - 1, 1)) / (domain(i, 0) - domain(i - 1, 0));
                // set control point at local min/max (gradient sign change)
                // TODO: checking exactly for 0.0 is not robust
                if ((grad == 0.0) || (grad > 0.0 && prev_grad < 0.0) || (grad < 0.0 && prev_grad > 0.0))
                {
                    ctrl_pts.conservativeResize(ctrl_pts.rows() + 1, domain.cols());
                    ctrl_pts.row(ctrl_pts.rows() - 1) = domain.row(i);
                }
                prev_grad = grad;
            }
        }
        nctrl_pts(0) = ctrl_pts.rows();

        // debug
        cerr << ctrl_pts.rows() << " initial control points:\n" << ctrl_pts << "\n" << endl;

        mfa = new mfa::MFA(p, ndom_pts, domain, ctrl_pts, nctrl_pts, knots);
        mfa->NonlinearEncode(err_limit, nctrl_pts);
    }

    // decode entire block
    void decode_block(const diy::Master::ProxyWithLink& cp)
    {
        approx.resize(domain.rows(), domain.cols());
        mfa->Decode(approx);
    }

    // compute error field and maximum error in the block
    // uses normal distance to the curve, surface, etc.
    void error(
            const   diy::Master::ProxyWithLink& cp,
            bool    decode_block)                            // decode entire block first
    {
        errs.resize(domain.rows(), domain.cols());
        errs = domain;

        if (decode_block)
        {
            approx.resize(domain.rows(), domain.cols());
            mfa->Decode(approx);
        }

#if 1                                               // TBB version

        // distance computation
        size_t max_idx;
        if (decode_block)
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    VectorXf cpt = approx.row(i);
                    errs(i, errs.cols() - 1) = fabs(mfa->NormalDistance(cpt, i));
                    });
        }
        else
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    errs(i, errs.cols() - 1) = mfa->Error(i);
                    });
        }
        sum_sq_err = 0.0;
        for (size_t i = 0; i < domain.rows(); i++)
        {
            sum_sq_err += (errs(i, errs.cols() - 1) * errs(i, errs.cols() - 1));
            if (i == 0 || errs(i, errs.cols() - 1) > max_err)
            {
                max_err = errs(i, errs.cols() - 1);
                max_idx = i;
            }
        }

#else                                               // single thread version

        // distance computation
        size_t max_idx;
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (decode_block)
            {
                VectorXf cpt = approx.row(i);
                errs(i, errs.cols() - 1) = fabs(mfa->NormalDistance(cpt, i));
            }
            else
                errs(i, errs.cols() - 1) = mfa->Error(i);
            if (i == 0 || fabs(errs(i, errs.cols() - 1)) > fabs(max_err))
            {
                max_err = errs(i, errs.cols() - 1);
                max_idx = i;
            }
        }

#endif

        mfa->max_err = max_err;

        // debug
        fprintf(stderr, "data range = %.1f\n", mfa->dom_range);
        fprintf(stderr, "raw max_error = %e\n", max_err);
        cerr << "position of max error: idx=" << max_idx << "\n" << domain.row(max_idx) << endl;
        fprintf(stderr, "|normalized max_err| = %e\n", max_err / mfa->dom_range);
    }

    // compute error field and maximum error in the block
    // uses difference between range values
    void range_error(
            const   diy::Master::ProxyWithLink& cp,
            bool    decode_block)                            // decode entire block first
    {
        errs.resize(domain.rows(), domain.cols());
        errs = domain;

        if (decode_block)
        {
            approx.resize(domain.rows(), domain.cols());
            mfa->Decode(approx);
        }

#if 1                                               // TBB version

        // distance computation
        size_t max_idx;
        int last = errs.cols() - 1;                 // range coordinate
        if (decode_block)
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    VectorXf cpt = approx.row(i);
                    errs(i, last) = fabs(cpt(last) - domain(i, last));
                    });
        }
        else
        {
            parallel_for (size_t(0), (size_t)domain.rows(), [&] (size_t i)
                    {
                    errs(i, last) = mfa->RangeError(i);
                    });
        }
        sum_sq_err = 0.0;
        for (size_t i = 0; i < domain.rows(); i++)
        {
            sum_sq_err += (errs(i, last) * errs(i, last));
            if (i == 0 || errs(i, last) > max_err)
            {
                max_err = errs(i, last);
                max_idx = i;
            }
        }

#else                                               // single thread version

        // distance computation
        size_t max_idx;
        int last = errs.cols() - 1;                 // range coordinate
        for (size_t i = 0; i < (size_t)domain.rows(); i++)
        {
            if (decode_block)
            {
                VectorXf cpt = approx.row(i);
                errs(i, last) = fabs(cpt(last) - domain(i, last));
            }
            else
                errs(i, last) = mfa->RangeError(i);
            if (i == 0 || errs(i, last) > max_err)
            {
                max_err = errs(i, last);
                max_idx = i;
            }
        }

#endif

        mfa->max_err = max_err;

        // debug
        fprintf(stderr, "data range = %.1f\n", mfa->dom_range);
        fprintf(stderr, "raw max_error = %e\n", max_err);
        cerr << "position of max error: idx=" << max_idx << "\n" << domain.row(max_idx) << endl;
        fprintf(stderr, "|normalized max_err| = %e\n", max_err / mfa->dom_range);
    }

    // save knot span domains for later comparison with error field
    void knot_span_domains(const diy::Master::ProxyWithLink& cp)
    {
        mfa->KnotSpanDomains(span_mins, span_maxs);
    }

    void print_block(const diy::Master::ProxyWithLink& cp)
    {
        fprintf(stderr, "\n--- Final block results ---\n");
//         cerr << "domain\n" << domain << endl;
//         cerr << "nctrl_pts:\n" << nctrl_pts << endl;
//         cerr << ctrl_pts.rows() << " final control points\n" << ctrl_pts << endl;
//         cerr << knots.size() << " knots\n" << knots << endl;
//         cerr << approx.rows() << " approximated points\n" << approx << endl;
        fprintf(stderr, "|max_err|             = %e\n", mfa->max_err);
        fprintf(stderr, "|normalized max_err|  = %e\n", mfa->max_err / mfa->dom_range);
        fprintf(stderr, "sum of squared errors = %e\n", sum_sq_err);
        fprintf(stderr, "L2 error              = %e\n", sqrt(sum_sq_err /nctrl_pts.rows()));
        fprintf(stderr, "RMS error             = %e\n", sqrt(sum_sq_err /domain.rows()));
        fprintf(stderr, "# input points = %ld\n", domain.rows());
        fprintf(stderr, "# output ctrl pts = %ld # output knots = %ld\n",
                ctrl_pts.rows(), knots.size());
        fprintf(stderr, "compression ratio = %.2f\n",
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
    VectorXi span_mins;                      // idx of minimum domain points of all knot spans
    VectorXi span_maxs;                      // idx of maximum domain points of all knot spans

    // (same number as input points, for rendering only)
    float    max_err;                        // maximum (abs value) distance from input points to curve
    float    sum_sq_err;                     // sum of squared errors
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
