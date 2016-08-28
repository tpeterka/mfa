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
            diy::save(bb, b->nctrl_pts);
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
    // f(x,y,z) = 1
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
    // f(x,y,z,t) = sqrt(x^2 + y^2 + z^2 + t^2)
    void generate_magnitude_data(const diy::Master::ProxyWithLink& cp, void* args)
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
        }

    void approx_block(const diy::Master::ProxyWithLink& cp, void* args)
        {
            // compute MFA from domain points
            Encode(p, ndom_pts, nctrl_pts, domain, ctrl_pts, knots);
        }

    // max error for the magnitude data set
    void mag_max_error(const diy::Master::ProxyWithLink& cp, void* args)
        {
            ErrArgs* a = (ErrArgs*)args;
            approx.resize(domain.rows(), domain.cols());
            errs.resize(domain.rows());

            // Compute domain points from MFA
            Decode(p, ndom_pts, domain, ctrl_pts, nctrl_pts, knots, approx);

            // max error
            for (size_t i = 0; i < approx.rows(); i++)
            {
                VectorXf approx_pos = approx.block(i, 0, 1, p.size()).row(0);
                // approx_mag  = what the magnitude of the position should be (ground truth)
                float approx_mag = approx_pos.norm();
                // approx_val = the approximated value of the MFA
                float approx_val = approx(i, p.size());
                float err = fabs(approx_mag - approx_val);
                if (i == 0 || err > max_err)
                    max_err = err;
            }

            // normalize max error by size of input data (domain and range)
            float min = domain.minCoeff();
            float max = domain.maxCoeff();
            float range = max - min;

            // debug
            fprintf(stderr, "range = %.1f\n", range);
            fprintf(stderr, "raw max_error = %e\n", max_err);

            max_err /= range;
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
