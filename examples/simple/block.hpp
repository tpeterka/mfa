//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/types.hpp>
#include <mfa/encode.hpp>
#include <mfa/decode.hpp>

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
    Block() {}
    static void* create()
        {
            return new Block;
        }
    static void  destroy(void* b)
        {
            delete static_cast<Block*>(b);
        }
    static void  save(const void* b_, diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;
            diy::save(bb, b->domain);
            diy::save(bb, b->range);
            diy::save(bb, b->domain_dim);
            diy::save(bb, b->range_dim);
            diy::save(bb, b->domain_mins);
            diy::save(bb, b->domain_maxs);
            diy::save(bb, b->range_min);
            diy::save(bb, b->range_max);
            diy::save(bb, b->p);
            diy::save(bb, b->ctrl_pts);
            diy::save(bb, b->knots);
            diy::save(bb, b->approx);
            diy::save(bb, b->errs);
            diy::save(bb, b->max_err);
        }
    static void  load(void* b_, diy::BinaryBuffer& bb)
        {
            Block* b = (Block*)b_;
            diy::load(bb, b->domain);
            diy::load(bb, b->range);
            diy::load(bb, b->domain_dim);
            diy::load(bb, b->range_dim);
            diy::load(bb, b->domain_mins);
            diy::load(bb, b->domain_maxs);
            diy::load(bb, b->range_min);
            diy::load(bb, b->range_max);
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
            domain.resize(a->npts);
            range.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // the simplest constant function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].x = a->min_x + i * dx;
                range[i]    = a->y_scale;
            }

            // extents
            domain_dim = 1;
            range_dim = 1;
            domain_mins.resize(1);
            domain_maxs.resize(1);
            domain_mins[0] = a->min_x;
            domain_maxs[0] = a->max_x;
            range_min = a->y_scale;
            range_max = a->y_scale;

            // debug
            // cerr << domain.size() << " input points:" << endl;
            // for (size_t i = 0; i < domain.size(); i++)
            //     cerr << "(" << domain[i].x << ", " << range[i] << ")";
            // cerr << endl;
        }

    void generate_circle_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts);
            range.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // a circle function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].x = cos(a->min_x + i * dx);
                range[i]    = a->y_scale * sin(a->min_x + i * dx);
            }

            // extents
            domain_dim = 1;
            range_dim = 1;
            domain_mins.resize(1);
            domain_maxs.resize(1);
            domain_mins[0] = 0.0;
            domain_maxs[0] = 1.0;
            range_min = 0.0;
            range_max = a->y_scale;

            // debug
            // cerr << domain.size() << " input points:" << endl;
            // for (size_t i = 0; i < domain.size(); i++)
            //     cerr << "(" << domain[i].x << ", " << range[i] << ")";
            // cerr << endl;
        }

    // y = sine(x)
    void generate_sine_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts);
            range.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // a sine function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].x = a->min_x + i * dx;
                range[i]    = a->y_scale * sin(domain[i].x);
            }

            // extents
            domain_dim = 1;
            range_dim = 1;
            domain_mins.resize(1);
            domain_maxs.resize(1);
            domain_mins[0] = a->min_x;
            domain_maxs[0] = a->max_x;
            range_min = -a->y_scale;
            range_max = a->y_scale;

            // debug
            // cerr << domain.size() << " input points:" << endl;
            // for (size_t i = 0; i < domain.size(); i++)
            //     cerr << "(" << domain[i].x << ", " << range[i] << ")";
            // cerr << endl;
        }

    // y = sine(x)/x
    void generate_sinc_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(a->npts);
            range.resize(a->npts);
            float dx = (a->max_x - a->min_x) / (a->npts - 1);

            // sine(x)/x function
            for (int i = 0; i < a->npts; i++)
            {
                domain[i].x = a->min_x + i * dx;
                if (domain[i].x == 0.0)
                    range[i] = a->y_scale;
                else
                    range[i] = a->y_scale * sin(domain[i].x) / domain[i].x;
            }

            // extents
            domain_dim = 1;
            range_dim = 1;
            domain_mins.resize(1);
            domain_maxs.resize(1);
            domain_mins[0] = a->min_x;
            domain_maxs[0] = a->max_x;
            range_min = -a->y_scale;
            range_max = a->y_scale;

            // debug
            // cerr << domain.size() << " input points:" << endl;
            // for (size_t i = 0; i < domain.size(); i++)
            //     cerr << "(" << domain[i].x << ", " << range[i] << ")";
            // cerr << endl;
        }

    // read the flame dataset and take one slice out of the middle of it
    // doubling the resolution because the file I have is a 1/2-resolution downsampled version
    void read_file_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
            p = a->p;
            domain.resize(2 * a->npts);      // doubling resolution for this test
            range.resize(2 * a->npts);
            vector<float> vel(3 * a->npts);

            // open hard-coded file name, seek to hard-coded start of desired section
            FILE *fd = fopen("/Users/tpeterka/software/mfa/examples/simple/6_small.xyz", "r");
            assert(fd);
            fseek(fd, (704 * 540 * 275 + 704 * 270) * 12, SEEK_SET);

            // read all three components of velocity and compute magnitude
            fread(&vel[0], sizeof(float), a->npts * 3, fd);
            for (size_t i = 0; i < vel.size() / 3; i++)
            {
                range[2 * i] = sqrt(vel[3 * i    ] * vel[3 * i    ] +
                                    vel[3 * i + 1] * vel[3 * i + 1] +
                                    vel[3 * i + 2] * vel[3 * i + 2]);
                // fprintf(stderr, "vel [%.3f %.3f %.3f] mag %.3f\n",
                //         vel[3 * i], vel[3 * i + 1], vel[3 * i + 2], range[i]);
            }
            // add an interpolated velocity magnitude between each two velocity magnitudes
            for (size_t i = 0; i < range.size() - 1; i++)
            {
                if (i % 2)
                    range[i] = (range[i - 1] + range[i + 1]) / 2.0;
            }
            range[range.size() - 1] = range[range.size() - 2]; // duplicate last value

            // find extent of range
            range_min = range[0];
            range_max = range[0];
            for (size_t i = 1; i < range.size(); i++)
            {
                // fprintf(stderr, "range[%d] = %.3f\n", i, range[i]);
                if (range[i] < range_min)
                    range_min = range[i];
                if (range[i] > range_max)
                    range_max = range[i];
            }

            // scale domain to same size as range, from 0 to range_max
            float dx = (range_max - range_min) / (domain.size() - 1);
            for (size_t i = 1; i < domain.size(); i++)
                domain[i].x = i * dx;

            // extents
            domain_dim = 1;
            range_dim = 1;
            domain_mins.resize(1);
            domain_maxs.resize(1);
            domain_mins[0] = 0.0;
            domain_maxs[0] = range_max - range_min;

            // debug
            fprintf(stderr, "domain [%.3f %.3f] range [%.3f %.3f]\n",
                    domain_mins[0], domain_maxs[0], range_min, range_max);
            // for (size_t i = 0; i < domain.size(); i++)
            //     fprintf(stderr, "%d: (%.3f %.3f)\n", i, domain[i].x, range[i]);
        }

    void approx_block(const diy::Master::ProxyWithLink& cp, void* args)
        {
            int nctrl_pts = *(int*)args;
            Approx1d(p, nctrl_pts, domain, range, ctrl_pts, knots);
        }

    void max_error(const diy::Master::ProxyWithLink& cp, void* args)
        {
            ErrArgs* a = (ErrArgs*)args;
            approx.resize(domain.size());
            errs.resize(domain.size());

            // use one or the other of the following

            // plain max
            // MaxErr1d(p, domain, range, ctrl_pts, knots, approx, errs, max_err);

            // max norm, should be better than MaxErr1d but more expensive
            MaxNormErr1d(p,
                         domain,
                         range,
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
            cerr << ctrl_pts.size() << " control points:" << endl;
            for (size_t i = 0; i < ctrl_pts.size(); i++)
                cerr << ctrl_pts[i] << " ";
            cerr << endl;

            cerr << knots.size() << " knots:" << endl;
            for (size_t i = 0; i < knots.size(); i++)
                cerr << knots[i] << " ";
            cerr << endl;

            // cerr << errs.size() << " errors at input points:" << endl;
            // for (size_t i = 0; i < errs.size(); i++)
            //     cerr << errs[i] << " ";
            // cerr << endl;

            cerr << "max_err = " << max_err << endl;
        }

    void reset_block(const diy::Master::ProxyWithLink& cp, void*)
        {
            domain.clear();
            range.clear();
            knots.clear();
            ctrl_pts.clear();
            errs.clear();
        }

    vector<Pt1d>   domain;                   // domain, eg. x coordinates
    vector<float>  range;                    // range, ie, physical attribute such as pressure
    int            domain_dim;               // domain dimensionality
    int            range_dim;                // range dimensionality
    vector<float>  domain_mins;              // local domain minimum corner
    vector<float>  domain_maxs;              // local domain maximum corner
    float          range_min;                // local range minimum value
    float          range_max;                // local range maximum value
    int p;                                   // degree
    vector<Pt2d>   ctrl_pts;                 // NURBS control points
    vector <float> knots;                    // NURBS knots
    vector<Pt2d>   approx;                   // points on approximated curve
    // (same number as input points, for rendering only)
    vector <float> errs;                     // distance from each input point to curve
    float          max_err;                  // maximum distance from input points to curve
};

// //
// // add blocks to a master
// //
// struct AddBlock
// {
//     AddBlock(diy::Master& master_, int p_): master(master_), p(p_)     {}
//     void operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
//                     const RCLink& link) const
//         {
//             Block*  b = new Block();
//             RCLink* l = new RCLink(link);
//             master.add(gid, b, l);
//             b->p = p;
//         }
//     diy::Master& master;                     // master
//     int p;                                   // degree
// };
