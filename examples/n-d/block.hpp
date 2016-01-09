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
            diy::save(bb, b->ctrl_pts);
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
            diy::load(bb, b->max_err);
        }
    void generate_constant_data(const diy::Master::ProxyWithLink& cp, void* args)
        {
            DomainArgs* a = (DomainArgs*)args;
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
                    domain(j, i) = a->s;
            }

            // extents
            for (int i = 0; i < a->pt_dim; i++)
            {
                domain_mins(i) = a->min[i];
                domain_maxs(i) = a->max[i];
            }
        }

    void approx_block(const diy::Master::ProxyWithLink& cp, void* args)
        {
            Approx(p, ndom_pts, nctrl_pts, domain, ctrl_pts, knots);
        }

    // void max_error(const diy::Master::ProxyWithLink& cp, void* args)
    //     {
    //         ErrArgs* a = (ErrArgs*)args;
    //         approx.resize(domain.rows(), domain.cols());
    //         errs.resize(domain.rows());

    //         // use one or the other of the following

    //         // plain max
    //         // MaxErr1d(p, domain, range, ctrl_pts, knots, approx, errs, max_err);

    //         // max norm, should be better than MaxErr1d but more expensive
    //         MaxNormErr1d(p,
    //                      domain,
    //                      ctrl_pts,
    //                      knots,
    //                      a->max_niter,
    //                      a->err_bound,
    //                      a->search_rad,
    //                      approx,
    //                      errs,
    //                      max_err);
    //     }

    void print_block(const diy::Master::ProxyWithLink& cp, void*)
        {
            cerr << ctrl_pts.rows() << " control points\n" << ctrl_pts << endl;
            cerr << knots.size() << " knots\n" << knots << endl;
            fprintf(stderr, "max_err = %.6lf\n", max_err);
        }

    VectorXi ndom_pts;                       // number of domain points in each dimension
    MatrixXf domain;                         // input data
    VectorXf domain_mins;                    // local domain minimum corner
    VectorXf domain_maxs;                    // local domain maximum corner
    VectorXi p;                              // degree
    VectorXi nctrl_pts;                      // number of control points in each dimension
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
