//--------------------------------------------------------------
// new knots inserter object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef _NEW_KNOTS_HPP
#define _NEW_KNOTS_HPP

#include    <mfa/mfa.hpp>
#include    <mfa/decode.hpp>

#include    <Eigen/Dense>
#include    <vector>
#include    <set>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

using namespace std;

namespace mfa
{
    template <typename T>                                   // float or double
    class NewKnots
    {
    public:

        NewKnots(MFA<T>& mfa_) :
            mfa(mfa_),
            max_num_curves(1.0e4)                           // max num. curves to check in one dimension of curve version
        {
        }

        ~NewKnots() {}

        // encodes at full dimensionality and decodes at full dimensionality
        // decodes full-d points in each knot span and adds new knot spans where error > err_limit
        // returns true if done, ie, no knots are inserted
        bool NewKnots_full(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
                T              err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)

        // 1d encoding and 1d decoding
        // adds knots error spans from all curves in all directions (into a set)
        // adds knots in middles of spans that have error higher than the limit
        // returns true if done, ie, no knots are inserted
        bool NewKnots_curve(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
                T              err_limit,                   // max allowable error
                int            iter);                       // iteration number of caller (for debugging)

    private:

        // computes error in knot spans
        // marks the knot spans that are done (error <= max_error in the entire span)
        // assumes caller allocated new_knots to number of spans and nnew_knots to domain dimensions
        // (does no resizing of new_knots and nnew_knots) and zeroed nnew_knots
        // returns true if all done, ie, no new knots inserted
        bool ErrorSpans(
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
                T              err_limit,                   // max. allowed error
                int            iter)                        // iteration number
        {

#ifndef MFA_NO_TBB                                          // TBB version

            Decoder<T> decoder(mfa, 1);

            // initialize all knot spans to not done
            for (auto i = 0; i < mfa.knot_spans.size(); i++)
                mfa.knot_spans[i].done = false;

            // spans that have already been split in this round (to prevent splitting twice)
            vector<bool> split_spans(mfa.knot_spans.size());                // intialized to false by default

            parallel_for(size_t(0), mfa.knot_spans.size(), [&] (size_t i)          // knot spans
                    {
                    if (!mfa.knot_spans[i].done)
                    {
                    size_t nspan_pts = 1;                                   // number of domain points in the span
                    for (auto k = 0; k < mfa.p.size(); k++)
                    nspan_pts *= (mfa.knot_spans[i].max_param_ijk(k) - mfa.knot_spans[i].min_param_ijk(k) + 1);

                    VectorXi p_ijk = mfa.knot_spans[i].min_param_ijk;           // indices of current parameter in the span
                    VectorX<T> param(mfa.p.size());                               // value of current parameter
                    bool span_done = true;                                  // span is done until error > err_limit

                    // TODO:  consider binary search of the points in the span?
                    // (error likely to be higher in the center of the span?)
                    for (auto j = 0; j < nspan_pts; j++)                    // parameters in the span
                    {
                    for (auto k = 0; k < mfa.p.size(); k++)
                    param(k) = mfa.params(mfa.po[k] + p_ijk(k));

                    // approximate the point and measure error
                    size_t idx;
                    mfa.ijk2idx(p_ijk, idx);
                    VectorX<T> cpt(mfa.ctrl_pts.cols());       // approximated point
                    decoder.VolPt(param, cpt);
                    T err = fabs(mfa.NormalDistance(cpt, idx)) / mfa.range_extent;     // normalized by data range

                    // span is not done
                    if (err > err_limit)
                    {
                        span_done = false;
                        break;
                    }

                    // increment param ijk
                    for (auto k = 0; k < mfa.p.size(); k++)                 // dimensions in the parameter
                    {
                        if (p_ijk(k) < mfa.knot_spans[i].max_param_ijk(k))
                        {
                            p_ijk(k)++;
                            break;
                        }
                        else
                            p_ijk(k) = mfa.knot_spans[i].min_param_ijk(k);
                    }                                                   // dimension in parameter
                    }                                                       // parameters in the span

                    if (span_done)
                        mfa.knot_spans[i].done = true;
                    }                                                           // knot span not done
                    });                                                           // knot spans

            // split spans that are not done
            auto norig_spans = mfa.knot_spans.size();
            bool new_knot_found = false;
            for (auto i = 0; i < norig_spans; i++)
            {
                if (!mfa.knot_spans[i].done && !split_spans[i])
                {
                    new_knots.resize(1);
                    nnew_knots = VectorXi::Zero(mfa.p.size());
                    SplitSpan(i, nnew_knots, new_knots, iter, split_spans);
                    if (nnew_knots.sum())
                    {
                        new_knot_found = true;
                        mfa.InsertKnots(nnew_knots, new_knots);
                    }
                }
            }

            // debug
            //     for (auto i = 0; i < mfa.knot_spans.size(); i++)                  // knot spans
            //     {
            //         cerr <<
            //             "span_idx="          << i                           <<
            //             "\nmin_knot_ijk:\n"  << mfa.knot_spans[i].min_knot_ijk  <<
            //             "\nmax_knot_ijk:\n"  << mfa.knot_spans[i].max_knot_ijk  <<
            //             "\nmin_knot:\n"      << mfa.knot_spans[i].min_knot      <<
            //             "\nmax_knot:\n"      << mfa.knot_spans[i].max_knot      <<
            //             "\nmin_param_ijk:\n" << mfa.knot_spans[i].min_param_ijk <<
            //             "\nmax_param_ijk:\n" << mfa.knot_spans[i].max_param_ijk <<
            //             "\nmin_param:\n"     << mfa.knot_spans[i].min_param     <<
            //             "\nmax_param:\n"     << mfa.knot_spans[i].max_param     <<
            //             "\n"                 << endl;
            //     }

            return !nnew_knots.sum();

#else                                                       // single-thread version

            Decoder<T> decoder(mfa, 1);

            // initialize all knot spans to not done
            for (auto i = 0; i < mfa.knot_spans.size(); i++)
                mfa.knot_spans[i].done = false;

            // spans that have already been split in this round (to prevent splitting twice)
            vector<bool> split_spans(mfa.knot_spans.size());                // intialized to false by default

            for (auto i = 0; i < mfa.knot_spans.size(); i++)                // knot spans
            {
                size_t nspan_pts = 1;                                       // number of domain points in the span
                for (auto k = 0; k < mfa.p.size(); k++)
                    nspan_pts *= (mfa.knot_spans[i].max_param_ijk(k) - mfa.knot_spans[i].min_param_ijk(k) + 1);

                VectorXi p_ijk = mfa.knot_spans[i].min_param_ijk;           // indices of current parameter in the span
                VectorX<T> param(mfa.p.size());                               // value of current parameter
                bool span_done = true;                                      // span is done until error > err_limit

                // TODO:  consider binary search of the points in the span?
                // (error likely to be higher in the center of the span?)
                for (size_t j = 0; j < nspan_pts; j++)                      // parameters in the span
                {
                    for (auto k = 0; k < mfa.p.size(); k++)
                        param(k) = mfa.params(mfa.po[k] + p_ijk(k));

                    // approximate the point and measure error
                    size_t idx;
                    mfa.ijk2idx(p_ijk, idx);
                    VectorX<T> cpt(mfa.ctrl_pts.cols());                      // approximated point
                    decoder.VolPt(param, cpt);
                    T err = fabs(mfa.NormalDistance(cpt, idx)) / mfa.range_extent;     // normalized by data range

                    // span is not done
                    if (err > err_limit)
                    {
                        span_done = false;
                        break;
                    }

                    // increment param ijk
                    for (auto k = 0; k < mfa.p.size(); k++)                 // dimensions in the parameter
                    {
                        if (p_ijk(k) < mfa.knot_spans[i].max_param_ijk(k))
                        {
                            p_ijk(k)++;
                            break;
                        }
                        else
                            p_ijk(k) = mfa.knot_spans[i].min_param_ijk(k);
                    }                                                       // dimension in parameter
                }                                                           // parameters in the span

                if (span_done)
                    mfa.knot_spans[i].done = true;
            }                                                               // knot spans

            // split spans that are not done
            auto norig_spans = mfa.knot_spans.size();
            bool new_knot_found = false;
            for (auto i = 0; i < norig_spans; i++)
            {
                if (!mfa.knot_spans[i].done && !split_spans[i])
                {
                    new_knots.resize(1);
                    nnew_knots = VectorXi::Zero(mfa.p.size());
                    SplitSpan(i, nnew_knots, new_knots, iter, split_spans);
                    if (nnew_knots.sum())
                    {
                        new_knot_found = true;
                        mfa.InsertKnots(nnew_knots, new_knots);
                    }
                }
            }

            // debug
            //     fprintf(stderr, "\nspans after splitting:\n-----\n");
            //     for (auto i = 0; i < mfa.knot_spans.size(); i++)                  // knot spans
            //         fprintf(stderr, "i=%d min_knot=[%.3f %.3f] max_knot=[%.3f %.3f]\n", i,
            //                 mfa.knot_spans[i].min_knot(0), mfa.knot_spans[i].min_knot(1),
            //                 mfa.knot_spans[i].max_knot(0), mfa.knot_spans[i].max_knot(1));

            return !new_knot_found;

#endif

        }

        // splits a knot span into two
        // also splits all other spans sharing the same knot values
        void SplitSpan(
                size_t         si,                          // id of span to split
                VectorXi&      nnew_knots,                  // number of new knots in each dim
                vector<T>&     new_knots,                   // new knots (1st dim changes fastest)
                int            iter,                        // iteration number
                vector<bool>&  split_spans)                 // spans that have already been split in this iteration
        {
            // new split dimension based on alternating dimension per span
            // check if span can be split (both halves would have domain points in its range)
            // if not, check other split directions
            int sd = mfa.knot_spans[si].last_split_dim;         // alternating per knot span
            T new_knot;                                     // new knot value in the split dimension
            size_t k;                                           // dimension
            for (k = 0; k < mfa.p.size(); k++)
            {
                sd       = (sd + 1) % mfa.p.size();
                new_knot = (mfa.knot_spans[si].min_knot(sd) + mfa.knot_spans[si].max_knot(sd)) / 2;
                if (mfa.params(mfa.po[sd] + mfa.knot_spans[si].min_param_ijk(sd)) < new_knot &&
                        mfa.params(mfa.po[sd] + mfa.knot_spans[si].max_param_ijk(sd)) > new_knot)
                    break;
            }
            if (k == mfa.p.size())                                  // a split direction could not be found
            {
                mfa.knot_spans[si].done = true;
                split_spans[si]         = true;

                // debug
                fprintf(stderr, "--- SplitSpan(): span %ld could not be split further ---\n", si);

                return;
            }

            // find all spans with the same min_knot_ijk as the span to be split and that are not done yet
            // those will be split too (NB, in the same dimension as the original span to be split)
            bool new_split = false;                             // the new knot was used to actually split a span
            for (auto j = 0; j < split_spans.size(); j++)       // original number of spans in this round
            {
                if (split_spans[j] || mfa.knot_spans[j].min_knot_ijk(sd) != mfa.knot_spans[si].min_knot_ijk(sd))
                    continue;

                // debug
                //      fprintf(stderr, "splitting span %d in sd=%d by new_knot=%.3f\n", j, sd, new_knot);

                new_split = true;

                // copy span to the back
                mfa.knot_spans.push_back(mfa.knot_spans[j]);

                // modify old span
                auto pi = mfa.knot_spans[j].min_param_ijk(sd);          // one coordinate of ijk index into params
                if (mfa.params(mfa.po[sd] + pi) < new_knot)                 // at least one param (domain pt) in the span
                {
                    while (mfa.params(mfa.po[sd] + pi) < new_knot)          // pi - 1 = max_param_ijk(sd) in the modified span
                        pi++;
                    mfa.knot_spans[j].last_split_dim    = sd;
                    mfa.knot_spans[j].max_knot(sd)      = new_knot;
                    mfa.knot_spans[j].max_param_ijk(sd) = pi - 1;
                    mfa.knot_spans[j].max_param(sd)     = mfa.params(mfa.po[sd] + pi - 1);

                    // modify new span
                    mfa.knot_spans.back().last_split_dim     = -1;
                    mfa.knot_spans.back().min_knot(sd)       = new_knot;
                    mfa.knot_spans.back().min_param_ijk(sd)  = pi;
                    mfa.knot_spans.back().min_param(sd)      = mfa.params(mfa.po[sd] + pi);
                    mfa.knot_spans.back().min_knot_ijk(sd)++;

                    split_spans[j] = true;
                }
            }

            if (!new_split)
                return;

            // increment min and max knot ijk for any knots after the inserted one
            for (auto j = 0; j < mfa.knot_spans.size(); j++)
            {
                if (mfa.knot_spans[j].min_knot(sd) > mfa.knot_spans[si].max_knot(sd))
                    mfa.knot_spans[j].min_knot_ijk(sd)++;
                if (mfa.knot_spans[j].max_knot(sd) > mfa.knot_spans[si].max_knot(sd))
                    mfa.knot_spans[j].max_knot_ijk(sd)++;
            }

            // add the new knot to nnew_knots and new_knots (only a single knot inserted at a time)
            new_knots.resize(1);
            new_knots[0]    = new_knot;
            nnew_knots      = VectorXi::Zero(mfa.p.size());
            nnew_knots(sd)  = 1;

            // debug
            //     fprintf(stderr, "inserted new knot value=%.3f dim=%d\n", new_knot, sd);
        }

        size_t  max_num_curves;                             // max num. curves per dimension to check in curve version
        MFA<T>& mfa;                                        // the mfa object
    };
}

#endif
