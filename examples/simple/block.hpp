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

using namespace std;

typedef  diy::ContinuousBounds          Bounds;
typedef  diy::RegularContinuousLink     RCLink;

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
    Block() :
        domain_mins(Pt<float>(2)),
        domain_maxs(Pt<float>(2))
        {
        }
    static
    void* create()
        {
            return new Block;
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
            diy::save(bb, b->dim);
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

            diy::load(bb, b->dim);
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
            dim = 2;
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // the simplest constant function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].resize(dim);
                domain[i][0] = a->min_x + i * dx;
                domain[i][1] = a->y_scale;
            }

            // extents
            domain_mins.resize(dim);
            domain_maxs.resize(dim);
            domain_mins[0] = a->min_x;
            domain_mins[1] = a->y_scale;
            domain_maxs[0] = a->max_x;
            domain_maxs[1] = a->y_scale;
        }

    void generate_circle_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            dim = 2;
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // a circle function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].resize(dim);
                domain[i][0] = cos(a->min_x + i * dx);
                domain[i][1] = a->y_scale * sin(a->min_x + i * dx);
            }

            // extents
            domain_mins.resize(dim);
            domain_maxs.resize(dim);
            domain_mins[0] = 0.0;
            domain_mins[1] = 0.0;
            domain_maxs[0] = 1.0;
            domain_maxs[1] = a->y_scale;
        }

    // y = sine(x)
    void generate_sine_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            dim = 2;
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // a sine function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].resize(dim);
                domain[i][0] = a->min_x + i * dx;
                domain[i][1] = a->y_scale * sin(domain[i][0]);
            }

            // extents
            domain_mins.resize(dim);
            domain_maxs.resize(dim);
            domain_mins[0] = a->min_x;
            domain_mins[1] = -a->y_scale;
            domain_maxs[0] = a->max_x;
            domain_maxs[1] = a->y_scale;
        }

    // y = sine(x)/x
    void generate_sinc_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            dim = 2;
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // sine(x)/x function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].resize(dim);
                domain[i][0] = a->min_x + i * dx;
                if (domain[i][0] == 0.0)
                    domain[i][1] = a->y_scale;
                else
                    domain[i][1] = a->y_scale * sin(domain[i][0]) / domain[i][0];
            }

            // extents
            domain_mins.resize(dim);
            domain_maxs.resize(dim);
            domain_mins[0] = a->min_x;
            domain_mins[1] = -a->y_scale;
            domain_maxs[0] = a->max_x;
            domain_maxs[1] = a->y_scale;
        }

    // read the flame dataset and take one slice out of the middle of it
    // doubling the resolution because the file I have is a 1/2-resolution downsampled version
    void read_file_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            dim = 2;
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;

            domain.resize(2 * a->npts); // double resolution
            vector<float> vel(3 * a->npts);

            for (size_t i = 0; i < domain.size(); i++)
                domain[i].resize(dim);

            // open hard-coded file name, seek to hard-coded start of desired section
            FILE *fd = fopen("/Users/tpeterka/datasets/flame/6_small.xyz", "r");
            assert(fd);
            fseek(fd, (704 * 540 * 275 + 704 * 270) * 12, SEEK_SET);

            // read all three components of velocity and compute magnitude
            fread(&vel[0], sizeof(float), a->npts * 3, fd);
            for (size_t i = 0; i < vel.size() / 3; i++)
            {
                domain[2 * i][1] = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                                        vel[3 * i + 1] * vel[3 * i + 1] +
                                        vel[3 * i + 2] * vel[3 * i + 2]);
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }
            // add an interpolated velocity magnitude between each two velocity magnitudes
            for (size_t i = 0; i < domain.size() - 1; i++)
            {
                if (i % 2)
                    domain[i][1] = (domain[i - 1][1] + domain[i + 1][1]) / 2.0;
            }
            domain[domain.size() - 1][1] = domain[domain.size() - 2][1]; // duplicate last value

            // find extent of range
            domain_mins.resize(dim);
            domain_maxs.resize(dim);

            domain_mins[1] = domain[0][1];
            domain_maxs[1] = domain[0][1];
            for (size_t i = 1; i < domain.size(); i++)
            {
                // fprintf(stderr, "range[%d] = %.3f\n", i, range[i]);
                if (domain[i][1] < domain_mins[1])
                    domain_mins[1] = domain[i][1];
                if (domain[i][1] > domain_maxs[1])
                    domain_maxs[1] = domain[i][1];
            }

            // scale domain to same size as range, from 0 to range_max
            float dx = (domain_maxs[1] - domain_mins[1]) / (domain.size() - 1);
            for (size_t i = 1; i < domain.size(); i++)
                domain[i][0] = i * dx;

            // extents
            domain_mins[0] = 0.0;
            domain_maxs[0] = domain_maxs[1] - domain_mins[1];

            // debug
            fprintf(stderr, "domain [%.3f %.3f] range [%.3f %.3f]\n",
                    domain_mins[0], domain_maxs[0], domain_mins[1], domain_maxs[1]);
        }

    void approx_block(const diy::Master::ProxyWithLink& cp, void* args)
        {
            int nctrl_pts = *(int*)args;
            Approx1d(p, nctrl_pts, dim, domain, ctrl_pts, knots);
        }

    void max_error(const diy::Master::ProxyWithLink& cp, void* args)
        {
            ErrArgs* a = (ErrArgs*)args;
            approx.resize(domain.size(), Pt<float>(dim));
            errs.resize(domain.size());

            // use one or the other of the following

            // plain max
            // MaxErr1d(p, domain, range, ctrl_pts, knots, approx, errs, max_err);

            // max norm, should be better than MaxErr1d but more expensive
            MaxNormErr1d(p,
                         dim,
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
            fprintf(stderr, "%lu control points\n", ctrl_pts.size());
            for (size_t i = 0; i < ctrl_pts.size(); i++)
                cerr << ctrl_pts[i] << " ";
            fprintf(stderr, "\n");

            fprintf(stderr, "%lu knotss\n", knots.size());
            for (size_t i = 0; i < knots.size(); i++)
                fprintf(stderr, "%.3lf ", knots[i]);
            fprintf(stderr, "\n");

            fprintf(stderr, "max_err = %.6lf\n",max_err);
        }

    void reset_block(const diy::Master::ProxyWithLink& cp, void*)
        {
            domain.clear();
            knots.clear();
            ctrl_pts.clear();
            errs.clear();
        }

    vector<Pt <float> > domain;              // input data
    int                 dim;                 // input dimensionality
    Pt<float>           domain_mins;         // local domain minimum corner
    Pt<float>           domain_maxs;         // local domain maximum corner
    int                 p;                   // degree
    vector<Pt <float> > ctrl_pts;            // NURBS control points
    vector <float>      knots;               // NURBS knots
    vector<Pt <float> > approx;              // points on approximated curve
                                             // (same number as input points, for rendering only)
    vector <float>      errs;                // distance from each input point to curve
    float               max_err;             // maximum distance from input points to curve
};
