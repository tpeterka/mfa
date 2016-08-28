//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.h>

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include <stdio.h>

#include <Eigen/Dense>

using namespace std;

typedef Eigen::MatrixXf                MatrixXf;
typedef Eigen::VectorXf                VectorXf;
typedef MatrixXf::Index                Index;

typedef diy::ContinuousBounds          Bounds;
typedef diy::RegularContinuousLink     RCLink;

// arguments to block foreach functions
struct DomainArgs
{
    int   p;                                 // degree
    int   npts;                              // number of input points
    float min_x;                             // minimum x
    float max_x;                             // maximum x
    float y_scale;                           // scaling factor for range
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
    Block(int dim)
        {
            domain_mins.resize(dim);
            domain_maxs.resize(dim);
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

            diy::save(bb, b->domain);
            diy::save(bb, b->domain_mins);
            diy::save(bb, b->domain_maxs);
            diy::save(bb, b->p);
            diy::save(bb, b->ctrl_pts);
            diy::save(bb, b->knots);
            diy::save(bb, b->approx);
            diy::save(bb, b->errs);
            diy::save(bb, b->max_err);
        }
    static
    void load(void* b_, diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;

            diy::load(bb, b->domain);
            diy::load(bb, b->domain_mins);
            diy::load(bb, b->domain_maxs);
            diy::load(bb, b->p);
            diy::load(bb, b->ctrl_pts);
            diy::load(bb, b->knots);
            diy::load(bb, b->approx);
            diy::load(bb, b->errs);
            diy::load(bb, b->max_err);
        }
    void generate_constant_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts, domain_mins.size());
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // the simplest constant function
            for (int i = 0; i < a->npts; i++)
            {
                domain(i, 0) = a->min_x + i * dx;
                domain(i, 1) = a->y_scale;
            }

            // extents
            domain_mins(0) = a->min_x;
            domain_mins(1) = a->y_scale;
            domain_maxs(0) = a->max_x;
            domain_maxs(1) = a->y_scale;
        }

    void generate_circle_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts, domain_mins.size());
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // a circle function
            for (int i = 0; i < a->npts; i++)
            {
                domain(i, 0) = cos(a->min_x + i * dx);
                domain(i, 1) = a->y_scale * sin(a->min_x + i * dx);
            }

            // extents
            domain_mins(0) = 0.0;
            domain_mins(1) = 0.0;
            domain_maxs(0) = 1.0;
            domain_maxs(1) = a->y_scale;
        }

    // y = sine(x)
    void generate_sine_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts, domain_mins.size());
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // a sine function
            for (int i = 0; i < a->npts; i++)
            {
                domain(i, 0) = a->min_x + i * dx;
                domain(i, 1) = a->y_scale * sin(domain(i, 0));
            }

            // extents
            domain_mins(0) = a->min_x;
            domain_mins(1) = -a->y_scale;
            domain_maxs(0) = a->max_x;
            domain_maxs(1) = a->y_scale;
        }

    // y = sine(x)/x
    void generate_sinc_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts, domain_mins.size());
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // sine(x)/x function
            for (int i = 0; i < a->npts; i++)
            {
                domain(i, 0) = a->min_x + i * dx;
                if (domain(i, 0) == 0.0)
                    domain(i, 1) = a->y_scale;
                else
                    domain(i, 1) = a->y_scale * sin(domain(i, 0)) / domain(i, 0);
            }

            // extents
            domain_mins(0) = a->min_x;
            domain_mins(1) = -a->y_scale;
            domain_maxs(0) = a->max_x;
            domain_maxs(1) = a->y_scale;
        }

    // read the flame dataset and take one slice out of the middle of it
    // doubling the resolution because the file I have is a 1/2-resolution downsampled version
    void read_file_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;

            domain.resize(2 * a->npts, domain_mins.size()); // double resolution
            vector<float> vel(3 * a->npts);

            // open hard-coded file name, seek to hard-coded start of desired section
            FILE *fd = fopen("/Users/tpeterka/datasets/flame/6_small.xyz", "r");
            assert(fd);
            fseek(fd, (704 * 540 * 275 + 704 * 270) * 12, SEEK_SET);

            // read all three components of velocity and compute magnitude
            fread(&vel[0], sizeof(float), a->npts * 3, fd);
            for (size_t i = 0; i < vel.size() / 3; i++)
            {
                domain(2 * i, 1) = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                                        vel[3 * i + 1] * vel[3 * i + 1] +
                                        vel[3 * i + 2] * vel[3 * i + 2]);
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }
            // add an interpolated velocity magnitude between each two velocity magnitudes
            for (size_t i = 0; i < domain.rows() - 1; i++)
            {
                if (i % 2)
                    domain(i, 1) = (domain(i - 1, 1) + domain(i + 1, 1)) / 2.0;
            }
            domain(domain.rows() - 1, 1) = domain(domain.rows() - 2, 1); // duplicate last value

            // find extent of range
            domain_mins(1) = domain(0, 1);
            domain_maxs(1) = domain(0, 1);
            for (size_t i = 1; i < domain.rows(); i++)
            {
                if (domain(i, 1) < domain_mins(1))
                    domain_mins(1) = domain(i, 1);
                if (domain(i, 1) > domain_maxs(1))
                    domain_maxs(1) = domain(i, 1);
            }

            // scale domain to same size as range, from 0 to range_max
            float dx = (domain_maxs(1) - domain_mins(1)) / (domain.rows() - 1);
            for (size_t i = 1; i < domain.rows(); i++)
                domain(i, 0) = i * dx;

            // extents
            domain_mins(0) = 0.0;
            domain_maxs(0) = domain_maxs(1) - domain_mins(1);

            // debug
            cerr << "domain extent:\n min\n" << domain_mins << "\nmax\n" << domain_maxs << endl;
        }

    void approx_block(const diy::Master::ProxyWithLink& cp, void* args)
        {
            VectorXi ps(1);                  // p as a vector of one element
            ps(0) = p;
            VectorXi nc(1);                  // ctrl_pts as a vector of one element
            nc(0) = *(int*)args;
            VectorXi nd(1);                  // number of domain points as a vector or one element
            nd(0) = domain.rows();
            Encode(ps, nd, nc, domain, ctrl_pts, knots);
        }

    void max_error(const diy::Master::ProxyWithLink& cp, void* args)
        {
            ErrArgs* a = (ErrArgs*)args;
            approx.resize(domain.rows(), domain.cols());
            errs.resize(domain.rows());

            // use one or the other of the following

            // plain max
            // MaxErr1d(p, domain, range, ctrl_pts, knots, approx, errs, max_err);

            // max norm, should be better than MaxErr1d but more expensive
            MaxNormErr1d(p,
                         domain,
                         ctrl_pts,
                         knots,
                         a->max_niter,
                         a->err_bound,
                         a->search_rad,
                         approx,
                         errs,
                         max_err);
        }

    void print_block(const diy::Master::ProxyWithLink& cp, void*)
        {
            cerr << ctrl_pts.rows() << " control points\n" << ctrl_pts << endl;
            cerr << knots.size() << " knots\n" << knots << endl;
            fprintf(stderr, "max_err = %e\n", max_err);
            fprintf(stderr, "# input points = %ld\n", domain.rows());
            fprintf(stderr, "# output ctrl pts = %ld # output knots = %ld\n",
                    ctrl_pts.rows(), knots.size());
            fprintf(stderr, "compression ratio = %.1f\n",
                    (float)(domain.rows()) / (ctrl_pts.rows() + knots.size() / ctrl_pts.cols()));
        }

    // memory is freed when block goes out of scope, but in case the data in the block needs
    // to be cleared and freed prior to going out of scope, use this function
    // TODO: is this ever needed?
    void reset_block(const diy::Master::ProxyWithLink& cp, void*)
        {
            domain.resize(0, 0);
            knots.resize(0);
            ctrl_pts.resize(0, 0);
            errs.resize(0);
        }

    MatrixXf domain;                         // input data
    VectorXf domain_mins;                    // local domain minimum corner
    VectorXf domain_maxs;                    // local domain maximum corner
    int      p;                              // degree
    MatrixXf ctrl_pts;                       // NURBS control points
    VectorXf knots;                          // NURBS knots
    MatrixXf approx;                         // points on approximated curve
                                             // (same number as input points, for rendering only)
    VectorXf errs;                           // distance from each input point to curve
    float    max_err;                        // maximum distance from input points to curve
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
}
