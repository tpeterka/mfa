//--------------------------------------------------------------
// T-mesh object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _TMESH_HPP
#define _TMESH_HPP

#include    <mfa/types.hpp>
#include    <unordered_map>
#include    <stack>

using KnotIdx   = size_t;
using TensorIdx = size_t;
using ParamIdx  = size_t;
using CtrlIdx   = size_t;

struct NeighborTensor                                   // neighboring tensor product
{
    int         dim;                                    // neighbor is in this dimension from the original tensor
    int         level;                                  // level of neighbor
    TensorIdx   tensor_idx;                             // index in tensor_prods of the neighbor
};

template <typename T>
struct TensorProduct
{
    vector<KnotIdx>             knot_mins;              // indices into all_knots
    vector<KnotIdx>             knot_maxs;              // indices into all_knots
    VectorXi                    nctrl_pts;              // number of control points in each domain dimension
    MatrixX<T>                  ctrl_pts;               // control points in row major order
    VectorX<T>                  weights;                // weights associated with control points
    int                         level;                  // refinement level
    vector<vector<KnotIdx>>     knot_idxs;              // all_knots indices of knots belonging to this tensor [dim][index] (sorted)
                                                        // assumes closed knot intervals, even at the maximum end for even degree
                                                        // (ie, not all knots in this tensor have control points in this tensor)
    bool                        done;                   // no more knots need to be added to this tensor
    TensorIdx                   parent;                 // parent from which this tensor was refined, if parent_exists is true
    vector<TensorIdx>           children;               // children tensors of this parent
    bool                        parent_exists;          // parent exists and the index to it is valid

    TensorProduct() : done(false)   {}
    TensorProduct(int dom_dim) : done(false)
    {
        knot_mins.resize(dom_dim);
        knot_maxs.resize(dom_dim);
        nctrl_pts.resize(dom_dim);
        knot_idxs.resize(dom_dim);
    }
    TensorProduct(vector<KnotIdx>& knot_mins_, vector<KnotIdx>& knot_maxs_, int level_) :
        done(false),
        knot_mins(knot_mins_),
        knot_maxs(knot_maxs_),
        level(level_)
    {
        int dom_dim = knot_mins.size();
        nctrl_pts.resize(dom_dim);
        knot_idxs.resize(dom_dim);
    }
};

namespace mfa
{
    // TODO all_knot_param_idxs assumes that knot value <= params[dim][idx] < next knot value for
    //      each input point. However, this is only true if:
    //       1. The input data is structured
    //       2. Every knot span contains at least one input point
    //       3. The last input point has parameter value equal to 1.0
    //      If these conditions are not satisfied, all_knot_params_idxs will behave unpredictably,
    //      which may cause the Tmesh to fail.
    template <typename T>
    struct Tmesh
    {
        vector<vector<T>>                   all_knots;          // all_knots[dimension][index] (sorted)
        vector<vector<int>>                 all_knot_levels;    // refinement levels of all_knots[dimension][index]
        vector<vector<ParamIdx>>            all_knot_param_idxs;// index of first input point whose parameter is >= knot value in all_knots[dimension][idx] (same layout as all_knots)
                                                                // knot value <= params[dim][idx] < next knot value
        vector<TensorProduct<T>>            tensor_prods;       // all tensor products
        unordered_map<KnotIdx, TensorIdx>   knot_tensor;        // n-d location in index space linearized to 1-d and mapped to deepest-level tensor containing the location
        int                                 dom_dim_;           // domain dimensionality
        VectorXi                            p_;                 // degree in each dimension
        int                                 min_dim_;           // starting coordinate of this model in full-dimensional data
        int                                 max_dim_;           // ending coordinate of this model in full-dimensional data
        int                                 max_level;          // deepest level of refinement

#ifdef MFA_DEBUG_KNOT_INSERTION

        TensorProduct<T>            debug_tensor_prod;  // used for viewing inserted control points

#endif

        Tmesh(int               dom_dim,                // number of domain dimension
              const VectorXi&   p,                      // degree in each dimension
              int               min_dim,                // starting coordinate of this model in full-dimensional data
              int               max_dim,                // ending coordinate of this model in full-dimensional data
              size_t            ntensor_prods =  0) :   // number of tensor products to allocate
                dom_dim_(dom_dim),
                p_(p),
                min_dim_(min_dim),
                max_dim_(max_dim),
                max_level(0)
        {
            all_knots.resize(dom_dim_);
            all_knot_levels.resize(dom_dim_);
            all_knot_param_idxs.resize(dom_dim_);

            if (ntensor_prods)
                tensor_prods.resize(ntensor_prods);
        }

        // ----- tensors ----- //

        // append a tensor product to the vector of tensor_prods
        // returns index of new tensor in the vector of tensor products
        int append_tensor(const vector<KnotIdx>&   knot_mins,       // indices in all_knots of min. corner of tensor to be inserted
                          const vector<KnotIdx>&   knot_maxs,       // indices in all_knots of max. corner
                          int                      level,           // level to assign to new tensor
                          bool                     parent_exists,   // whether parent exists
                          TensorIdx                parent,          // parent if it exists
                          bool                     debug = false)   // print debugging output
        {
            bool vec_grew = false;                  // vector of tensor_prods grew
            bool tensor_inserted = false;           // the desired tensor was already inserted

            // create a new tensor product
            TensorProduct<T> new_tensor(dom_dim_);
            new_tensor.knot_mins     = knot_mins;
            new_tensor.knot_maxs     = knot_maxs;
            new_tensor.parent_exists = parent_exists;
            new_tensor.parent        = parent;

            // initialize control points
            new_tensor.nctrl_pts.resize(dom_dim_);
            size_t tot_nctrl_pts = 1;

            if (!tensor_prods.size())                                   // no existing tensor products; this is the first tensor
            {
                new_tensor.level = level;
                tensor_knot_idxs(new_tensor);
                new_tensor.parent_exists = false;

                // resize control points
                for (auto j = 0; j < dom_dim_; j++)
                {
                    new_tensor.nctrl_pts[j] = all_knots[j].size() - p_(j) - 1;
                    tot_nctrl_pts *= new_tensor.nctrl_pts[j];
                }
                new_tensor.ctrl_pts.resize(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
                new_tensor.weights.resize(tot_nctrl_pts);               // will get initialized to 1 later
            }
            else                                                        // add a new tensor to existing vector of tensor products
            {
                new_tensor.level = level;
                tensor_knot_idxs(new_tensor);

                // resize control points
                for (auto j = 0; j < dom_dim_; j++)
                {
                    // count number of knots in the new tensor in this dimension
                    // inserted tensor is at the deepest level of refinement, ie, all knots in the global knot vector between
                    // min and max knots are in this tensor (don't skip any knots)
                    size_t nknots   = 0;
                    size_t nanchors = 0;
                    for (auto i = knot_mins[j]; i <= knot_maxs[j]; i++)
                        nknots++;
                    if (p_(j) % 2 == 0)         // even degree: anchors are between knot lines
                        nanchors = nknots - 1;
                    else                            // odd degree: anchors are on knot lines
                        nanchors = nknots;
                    if (knot_mins[j] < (p_(j) + 1) / 2)                       // skip up to (p+1)/2 anchors at start of global knots
                        nanchors -= ((p_(j) + 1) / 2 - knot_mins[j]);
                    if (knot_maxs[j] > all_knots[j].size() - (p_(j) + 1) / 2 - 1)     // skip up to p-1 anchors at end of global knots
                        nanchors -= (knot_maxs[j] + (p_(j) + 1) / 2 + 1 - all_knots[j].size());
                    new_tensor.nctrl_pts[j] = nanchors;
                    tot_nctrl_pts *= nanchors;
                }
                new_tensor.ctrl_pts.resize(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
                new_tensor.weights.resize(tot_nctrl_pts);               // will get initialized to 1 later
            }

            // check for intersection of the new tensor with existing tensors
            for (auto j = 0; j < tensor_prods.size(); j++)  // for all tensor products
            {
                // check if new tensor completely covers existing tensor j and set parent to existing tensor
                if (subset(tensor_prods[j].knot_mins, tensor_prods[j].knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
                {
                    new_tensor.parent_exists = true;
                    new_tensor.parent = j;
                    break;
                }
            }

            // initialize the new tensor weights to 1
            new_tensor.weights = VectorX<T>::Ones(new_tensor.weights.size());

            // add the tensor and add the child to the parent, if it exists
            if (new_tensor.parent_exists)
                tensor_prods[new_tensor.parent].children.push_back(tensor_prods.size());
            tensor_prods.push_back(new_tensor);

            // update the knot_tensor hash map for all tensors
            hash_all_tensors();

            return tensor_prods.size() - 1;
        }

        // update knot_tensor hash map for all tensors
        // the same knot can appear in multiple tensors, but the hash map stores the deepest tensor for a knot
        // the depth of the tensor increases as we iterate over tensors, so deeper tensors overwrite shallower ones for the same knot
        void hash_all_tensors()
        {
            knot_tensor.clear();
            VectorXi tensor_nknots(dom_dim_), all_nknots(dom_dim_);
            VectorXi tensor_ijk(dom_dim_);                              // local ijk of knot in tensor
            vector<KnotIdx> all_ijk(dom_dim_);                          // global ijk of knot in all_knots
            vector<KnotIdx> temp_ijk(dom_dim_);                         // temporary knot multidim index
            TensorIdx unused;

            for (auto j = 0; j < tensor_prods.size(); j++)              // for all tensors
            {

                // set up the vol iterators for the knots in the tensor and for all knots
                for (auto i = 0; i < dom_dim_; i++)
                {
                    all_nknots(i) = all_knots[i].size();
                    tensor_nknots(i) = tensor_prods[j].knot_idxs[i].size();
                }
                VolIterator tensor_knots_iter(tensor_nknots);
                VolIterator all_knots_iter(all_nknots);

                while (!tensor_knots_iter.done())                            // for all knots in the tensor
                {
                    bool skip = false;

                    // get the knot indices in the global knot space
                    tensor_knots_iter.idx_ijk(tensor_knots_iter.cur_iter(), tensor_ijk);
                    for (auto i = 0; i < dom_dim_; i++)
                    {
                        all_ijk[i] = tensor_prods[j].knot_idxs[i][tensor_ijk(i)];

                        // if even degree, the knot is at the max edge of an interior tensor, and the knot existed in an ancestor, don't overwrite its hash map entry
                        if (p_(i) % 2 == 0 &&                                                   // even degree
                            tensor_ijk(i) == tensor_prods[j].knot_idxs[i].size() - 1 &&         // knot is at max edge of tensor
                            all_ijk[i] < all_knots[i].size() - 1)                               // tensor is interior, tensor max edge is not global max edge
                        {
                            for (auto k = 0; k < dom_dim_; k++)
                                temp_ijk[k] = tensor_prods[j].knot_idxs[k][tensor_ijk(k)];
                            if (lookup_tensor(temp_ijk, unused))
                            {
                                // debug
//                                 fmt::print(stderr, "for tensor {}, skipping hashing knot [{}] because it is at the max edge and exists in tensor {}\n",
//                                         j, fmt::join(temp_ijk, ","), unused);

                                skip = true;
                                break;
                            }
                        }
                    }

                    // convert the multidim knot indices to a linear 1-d index for hashing
                    if (!skip)
                    {
                        auto idx = all_knots_iter.ijk_idx(all_ijk);
                        knot_tensor[idx] = j;                               // hash the linear knot idx to the tensor
                    }

                    tensor_knots_iter.incr_iter();
                }
            }

        }

        // checks if two tensors intersect to within a padding distance
        // adjacency (to within pad distance) counts as an intersection if adjacency_counts = true (default)
        // subset also counts as intersection
        bool intersect(const TensorProduct<T>&  t1,
                       const TensorProduct<T>&  t2,
                       KnotIdx                  pad = 0,                    // pad distance per side that counts as intersecting
                       bool                     adjacency_counts = true,    // whether exact adjacency qualifies as intersection
                       bool                     corner_pad_counts = true) const  // when pad > 0, whether intersecting at a corner is sufficient
        {
            int k;
            for (k = 0; k < dom_dim_; k++)
            {
                if (adjacency_counts)
                {
                    if (t1.knot_maxs[k] + pad < t2.knot_mins[k] || t2.knot_maxs[k] + pad < t1.knot_mins[k])
                        return false;
                }
                if (!adjacency_counts)
                {
                    if (t1.knot_maxs[k] + pad <= t2.knot_mins[k] || t2.knot_maxs[k] + pad <= t1.knot_mins[k])
                        return false;
                }
            }

            if (pad && !corner_pad_counts)
            {
                for (k = 0; k < dom_dim_; k++)
                {
                    // either of these two tests is true -> intersects within pad on a face (not a corner)
                    if (adjacency_counts && t1.knot_maxs[k] >= t2.knot_mins[k] && t2.knot_maxs[k] >= t1.knot_mins[k])
                            break;
                    if (!adjacency_counts && t1.knot_maxs[k] > t2.knot_mins[k] && t2.knot_maxs[k] > t1.knot_mins[k])
                            break;
                }
                if (k == dom_dim_)  // did not intersect on any faces, only corner
                    return false;
            }

            return true;
        }

        // clamp tensor knot_mins, knot_maxs to parent of tensor, if closer to parent than pad
        void clamp_to_parent(
                TensorProduct<T>&   t,          // tensor to constrain
                int                 pad,        // constrain to parent if tensor is within pad (per side) of parent or greater
                int                 edge_pad)   // extra padding per side for tensor at the global edge
        {
            auto& parent = tensor_prods[t.parent];
            for (auto j = 0; j < dom_dim_; j++)
            {
                int ofst = (t.knot_mins[j] == 0) ? pad + edge_pad : pad;
                if (t.knot_mins[j] < parent.knot_mins[j] ||
                        knot_idx_dist(parent, parent.knot_mins[j], t.knot_mins[j], j, false) < ofst)
                    t.knot_mins[j] = parent.knot_mins[j];
                ofst = (t.knot_maxs[j] == all_knots[j].size() - 1) ? pad + edge_pad : pad;
                if (t.knot_maxs[j] > parent.knot_maxs[j] ||
                        knot_idx_dist(parent, t.knot_maxs[j], parent.knot_maxs[j], j, false) < ofst)
                    t.knot_maxs[j] = parent.knot_maxs[j];
            }
        }

        // merges two tensor product knot mins, maxs, optionally constrained by parent of resulting tensor
        void merge_tensors(
                TensorProduct<T>&   inout,      // one of the input tensors and the output of the merge
                TensorProduct<T>&   in,         // other input tensor
                int                 pad,        // constrain merge to parent of inout if merge is within pad (per side) of parent; -1: don't constrain to parent
                int                 edge_pad)   // extra padding per side for tensor at the global edge
        {
            vector<KnotIdx> merge_mins(dom_dim_);
            vector<KnotIdx> merge_maxs(dom_dim_);
            merge(inout.knot_mins, inout.knot_maxs, in.knot_mins, in.knot_maxs, merge_mins, merge_maxs);
            inout.knot_mins = merge_mins;   // adjust t to the merged extents
            inout.knot_maxs = merge_maxs;

            // don't overshoot the parent or leave it with a small remainder
            if (pad >= 0)
                clamp_to_parent(inout, pad, edge_pad);
        }

        // finds the deepest tensor containing a point in parameter space using a depth-first search
        // assumes that tensors at the same level are disjoint, ie, param cannot be found in a sibling, only a child
        TensorIdx find_tensor(const VectorX<T>&     param) const    // multidim parameter point
        {
            TensorIdx tidx = 0;
            TensorIdx found_idx = 0;                                // candidate tensor containing param, although not necessarily the deepest one yet
            bool found = false;                                     // found a candidate tensor containing param, although not necessarily the deepest one yet
            std::vector<size_t> cidx(tensor_prods.size(), 0);       // index of next child to be traversed, for each tensor product
            while (1)
            {
                // if param is in the tensor, visit children looking for a leaf node
                if (in(param, tensor_prods[tidx]))
                {
                    found       = true;
                    found_idx   = tidx;

                    // leaf node, terminate successfully
                    if (tensor_prods[tidx].children.size() == 0)
                        return found_idx;

                    // sanity check on bounds of cidx[tidx]
                    if (cidx[tidx] >= tensor_prods[tidx].children.size())
                        throw MFAError(fmt::format("find_tensor(): child index out of bounds while looking for param \n{}", param));

                    // descend to next child
                    tidx = tensor_prods[tidx].children[cidx[tidx]];
                    continue;
                }

                // sanity check, param must be in the root, and if iterating over children, parent must exist
                if (tidx == 0 || !tensor_prods[tidx].parent_exists)
                    throw MFAError(fmt::format("find_tensor(): root tensor (tidx 0) does not contain param \n{}\n or parent does not exist while iterating over children", param));

                // backtrack
                tidx = tensor_prods[tidx].parent;
                cidx[tidx]++;


                if (cidx[tidx] >= tensor_prods[tidx].children.size())
                {
                    if (found)
                        return found_idx;
                    else
                        throw MFAError(fmt::format("find_tensor(): ran out of children before finding param \n{}", param));
                }
            }
        }

        // make candidate tensor constrained to another tensor (e.g., parent)
        // candidate can be no larger in any dimension than the other tensor and also doesn't leave the other tensor with a small remainder anywhere
        // candidate tensor knot_mins and knot_maxs will be adjusted accordingly
        void make_candidate(
                std::vector<KnotIdx>    inserted_knot_idx,      // inserted knot
                TensorProduct<T>&       c,                      // candidate tensor being constrained
                const TensorProduct<T>& t,                      // other tensor providing the constraints
                int                     pad,                    // padding per side for interior tensor (not at global edge)
                int                     edge_pad)               // extra padding per side for tensor at the global edge
        {
            // make initial set of knot mins and maxs
            for (auto j = 0; j < dom_dim_; j++)
            {
                // min side
                knot_idx_ofst(t, inserted_knot_idx[j], -(pad / 2 + 1), j, true, c.knot_mins[j]);

                // max side
                if (p_(j) % 2)      // odd degree
                    knot_idx_ofst(t, inserted_knot_idx[j], pad / 2 + 1, j, true, c.knot_maxs[j]);
                else                // even degree
                    knot_idx_ofst(t, inserted_knot_idx[j], pad / 2 + 2, j, true, c.knot_maxs[j]);
            }

            for (auto j = 0; j < dom_dim_; j++)
            {
                int min_ofst  = (t.knot_mins[j] == 0) ? pad + edge_pad : pad;
                int max_ofst  = (t.knot_maxs[j] == all_knots[j].size() - 1) ? pad + edge_pad : pad;

                // adjust min edge of candidate

                // sanity
                if (c.knot_mins[j] > t.knot_maxs[j])
                    throw MFAError(fmt::format("make_candidate: c.knot_mins[{}] {} > t.knot_maxs[{}] {}\n",
                                j, c.knot_mins[j], j, t.knot_maxs[j]));

                // check/adjust min edge of c against max edge of t
                if (knot_idx_dist(t, c.knot_mins[j], t.knot_maxs[j], j, false) < max_ofst)
                    knot_idx_ofst(t, t.knot_maxs[j], -max_ofst, j, false, c.knot_mins[j]);

                // check/adjust min edge of c against min edge of t
                if (knot_idx_dist(t, t.knot_mins[j], c.knot_mins[j], j, false) < min_ofst)
                    c.knot_mins[j] = t.knot_mins[j];

                // adjust max edge of candidate

                // sanity
                if (c.knot_maxs[j] < t.knot_mins[j])
                    throw MFAError(fmt::format("make_candidate: c.knot_maxs[{}] {} < t.knot_mins[{}] {}\n",
                                j, c.knot_maxs[j], j, t.knot_mins[j]));

                // check/adjust max edge of c against min edge of t
                if (knot_idx_dist(t, t.knot_mins[j], c.knot_maxs[j], j, false) < min_ofst)
                    knot_idx_ofst(t, t.knot_mins[j], min_ofst, j, false, c.knot_maxs[j]);

                // check/adjust max edge of c against max edge of t
                if (knot_idx_dist(t, c.knot_maxs[j], t.knot_maxs[j], j, false) < max_ofst)
                    c.knot_maxs[j] = t.knot_maxs[j];
            }
        }

        // determine starting and ending indices of domain input points covered by one tensor product
        // coverage extends to edge of basis functions corresponding to control points in the tensor product
        void domain_pts(TensorIdx               t_idx,              // index of current tensor product
                        vector<vector<T>>&      params,             // params of input points
                        bool                    extend,             // extend input points to cover neighbors (eg., constraints)
                        int                     extra_cons,         // extra constraints beyond normal extension (if extend is true)
                        vector<size_t>&         start_idxs,         // (output) starting idxs of input points
                        vector<size_t>&         end_idxs) const     // (output) ending idxs of input points
        {
            start_idxs.resize(dom_dim_);
            end_idxs.resize(dom_dim_);
            vector<KnotIdx> min_anchor(dom_dim_);                   // anchor for the min. edge basis functions of the new tensor
            vector<KnotIdx> max_anchor(dom_dim_);                   // anchor for the max. edge basis functions of the new tensor
            vector<vector<KnotIdx>> local_knot_idxs;                // local knot vector for an anchor

            const TensorProduct<T>& tc = tensor_prods[t_idx];

            // left edge
            vector<KnotIdx> start_knot_idxs(dom_dim_);
            for (auto k = 0; k < dom_dim_; k++)
                min_anchor[k] = tc.knot_mins[k];
            if (extend)
            {
                // extend by p/2 + 2 knots from the min corner in all dimensions and then take the min corner of that extension
                knot_intersections(min_anchor, local_knot_idxs, 2 + extra_cons);
                for (auto k = 0; k < dom_dim_; k++)
                    start_knot_idxs[k] = local_knot_idxs[k][1]; // both even and odd degree: 1 after front of local knot vector
            }
            else
            {
                for (auto k = 0; k < dom_dim_; k++)
                    start_knot_idxs[k] = min_anchor[k];
            }

            // right edge
            vector<KnotIdx> end_knot_idxs(dom_dim_);
            local_knot_idxs.clear();
            for (auto k = 0; k < dom_dim_; k++)
            {
                if (p_(k) % 2 == 0)
                    max_anchor[k] = tc.knot_maxs[k] - 1;
                else
                    max_anchor[k] = tc.knot_maxs[k];
            }
            if (extend)
            {
                // extend by p/2 + 2 knots from the max corner in all dimensions and then take the max corner of that extension
                knot_intersections(max_anchor, local_knot_idxs, 2 + extra_cons);
                for (auto k = 0; k < dom_dim_; k++)
                    end_knot_idxs[k] = local_knot_idxs[k][local_knot_idxs[k].size() - 3]; // both even and odd degree: 2 before back of local knot vector
            }
            else
            {
                for (auto k = 0; k < dom_dim_; k++)
                {
                    if (tc.knot_maxs[k] == all_knots[k].size() - 1)
                        end_knot_idxs[k] = all_knots[k].size() - 1 - p_(k);
                    else
                        end_knot_idxs[k] = max_anchor[k];
                }
            }

            // input points corresponding to start and end knot values
            for (auto k = 0; k < dom_dim_; k++)
            {
                // start point begins at all_knot_param_idxs[start_knot_idxs]
                start_idxs[k]   = all_knot_param_idxs[k][start_knot_idxs[k]];

                // end points go up to but do not include all_knot_param_ixs[end_knot_idxs + 1]

                // end point within repeated end knots
                if (end_knot_idxs[k] == all_knots[k].size() - 1)
                    end_idxs[k] = all_knot_param_idxs[k][all_knots[k].size() - 1];
                else if (all_knots[k].size() - 1 - end_knot_idxs[k] <= p_(k))
                    end_idxs[k] = all_knot_param_idxs[k][all_knots[k].size() - 1 - p_(k)] - 1;

                // end point before repeated end knots
                else
                {
                    // TODO: following fixes a particular case but not sure if generally correct
                    if (p_(k) % 2 && !extend)
                        end_idxs[k] = all_knot_param_idxs[k][end_knot_idxs[k]] - 1;
                    else
                        end_idxs[k] = all_knot_param_idxs[k][end_knot_idxs[k] + 1] - 1;
                }
            }
        }

        // gets index of deepest level tensor containing point in index space
        // by converting n-d point into 1-d index and looking up value in knot_tensors hash map
        // returns true if the tensor was found
        bool lookup_tensor(const vector<KnotIdx>&   pt,                   // target point in index space
                           TensorIdx&               t_idx) const          // (output) tensor idx if it was found
        {
            // convert multidim point into linear 1-d index
            VectorXi nknots(dom_dim_);
            for (auto i = 0; i < dom_dim_; i++)
                nknots(i) = all_knots[i].size();
            VolIterator vol_iter(nknots);

            // look up tensor in hash map
            try
            {
                auto idx = vol_iter.ijk_idx(pt);
                t_idx = knot_tensor.at(idx);
                return true;
            } catch (const std::out_of_range& e)
            {
                return false;
            }
        }

        // ----- knots ----- //

        // initialize knots
        void init_knots(VectorXi& nctrl_pts)
        {
            for (auto i = 0; i < dom_dim_; i++)
            {
                all_knots[i].resize(nctrl_pts(i) + p_(i) + 1);
                all_knot_levels[i].resize(nctrl_pts(i) + p_(i) + 1);
                all_knot_param_idxs[i].resize(nctrl_pts(i) + p_(i) + 1);
            }

            // initialize first tensor product
            vector<size_t> knot_mins(dom_dim_);
            vector<size_t> knot_maxs(dom_dim_);
            for (auto i = 0; i < dom_dim_; i++)
            {
                knot_mins[i] = 0;
                knot_maxs[i] = all_knots[i].size() - 1;
            }
            int id = append_tensor(knot_mins, knot_maxs, 0, false, 0);
            assert(id == 0);
        }

        // resize the first tensor (used when the knot distribution
        // is manually overriden by user)
        void reinit_knots(VectorXi& nctrl_pts)
        {
            tensor_prods.clear();
            init_knots(nctrl_pts);
        }

        // insert a knot into all_knots at an unknown position
        // checks for duplicates and invalid insertions
        // returns:
        // 0: no change in knots or levels
        // 1: changed level of an existing knot
        // 2: inserted a new knot and level
        int insert_knot(
                int                        dim,                 // current dimension
                int                        level,               // refinement level of inserted knot
                T                          knot,                // knot value to be inserted
                const vector<vector<T>>&   params,              // params of input points
                KnotIdx&                   pos)                 // (output) inserted position
        {
            pos = FindSpan(dim, knot);
            if (knot > all_knots[dim][pos])
                pos++;
            if (level > max_level)
                max_level = level;
            return insert_knot_at_pos(dim, pos, level, knot, params);
        }

        // insert a knot into all_knots at a given position
        // checks for duplicates and invalid insertions
        // adjusts all tensor products knot_mins, knot_maxs accordingly
        // returns:
        // 0: no change in knots or levels
        // 1: changed level of an existing knot
        // 2: inserted a new knot and level
        int insert_knot_at_pos(
                int                        dim,                 // current dimension
                KnotIdx                    pos,                 // new position in all_knots[dim] of inserted knot
                int                        level,               // refinement level of inserted knot
                T                          knot,                // knot value to be inserted
                const vector<vector<T>>&   params)              // params of input points
        {
            // if knot exists already, just update its level
            if (all_knots[dim][pos] == knot)
            {
                // update to highest (most coarse) level
                if (level < all_knot_levels[dim][pos])
                {
                    all_knot_levels[dim][pos] = level;
                    return 1;
                }
                else
                    return 0;
            }

            // check if knot is out of order
            if ( (pos > 0 && all_knots[dim][pos - 1] >= knot) ||
                (pos < all_knots[dim].size() - 1 && all_knots[dim][pos + 1] <= knot) )
            {
                fmt::print(stderr, "Error: insert_knot_at_pos(): attempting to insert a knot out of order\n");
                fmt::print(stderr, "dim {} pos {} knot {} level {}\n", dim, pos, knot, level);
                print_knots();
                abort();
            }

            // insert knot and level
            all_knots[dim].insert(all_knots[dim].begin() + pos, knot);
            all_knot_levels[dim].insert(all_knot_levels[dim].begin() + pos, level);

            // insert param idx
            auto        param_it = params[dim].begin();     // iterator into params (for one dim.)
            // uninitialized values, search entire params
            if (pos > 0 && all_knot_param_idxs[dim][pos - 1] == all_knot_param_idxs[dim][pos])
                param_it = lower_bound(params[dim].begin(), params[dim].end(), all_knots[dim][pos]);
            else if (pos > 0)       // search for the param idx within the bounds of existing knot values
            {
                ParamIdx low    = all_knot_param_idxs[dim][pos - 1];
                ParamIdx high   = all_knot_param_idxs[dim][pos];
                param_it        = lower_bound(params[dim].begin() + low, params[dim].begin() + high, all_knots[dim][pos]);
            }

            ParamIdx param_idx = param_it - params[dim].begin();
            all_knot_param_idxs[dim].insert(all_knot_param_idxs[dim].begin() + pos, param_idx);

            // adjust tensor product knot_mins and knot_maxs and local knot indices
            for (auto& t : tensor_prods)
            {
                if (t.knot_mins[dim] >= pos)
                    t.knot_mins[dim]++;
                if (t.knot_maxs[dim] >= pos)
                    t.knot_maxs[dim]++;
                tensor_knot_idxs(t);
            }

            return 2;
        }

        // convert global knot_idx to local_knot_idx in existing_tensor in current dim.
        KnotIdx global2local_knot_idx(KnotIdx                   knot_idx,
                                      const TensorProduct<T>&   t,
                                      int                       cur_dim,
                                      bool&                     found,                  // (output) whether knot was found in the tensor
                                      bool                      check = true) const     // check that global and local indices refer to same knot
        {
            KnotIdx local_knot_idx  = 0;
            int     cur_level       = t.level;
            KnotIdx min_idx         = t.knot_mins[cur_dim];
            KnotIdx max_idx         = t.knot_maxs[cur_dim];

            if (knot_idx < min_idx || knot_idx > max_idx)
            {
                found = false;
                return 0;
            }

            for (auto i = min_idx; i < knot_idx; i++)
                if (all_knot_levels[cur_dim][i] <= cur_level)
                    local_knot_idx++;
            // if knot_idx is at a deeper level than the tensor (not included), then overcounted by 1
            if (all_knot_levels[cur_dim][knot_idx] > cur_level && local_knot_idx > 0)
                local_knot_idx--;

            // sanity check that global and local indices refer to same knot
            if (check && t.knot_idxs[cur_dim].size() && knot_idx != t.knot_idxs[cur_dim][local_knot_idx])
            {
                fmt::print("Error: global2local_knot_idx(): knot_idx and local_knot_idx index different knots. This should not happen.\n");
                fmt::print(stderr, "cur_dim {} knot_idx {} local_knot_idx {} t.knot_idxs[local_knot_idx] {} (should equal knot_idx {})\n",
                        cur_dim, knot_idx, local_knot_idx, t.knot_idxs[cur_dim][local_knot_idx], knot_idx);
                fmt::print(stderr, "tensor knot mins [{}] knot maxs [{}]\n", fmt::join(t.knot_mins, ","), fmt::join(t.knot_maxs, ","));
                print(true, true, false, false);
                abort();
            }

            found = true;
            return local_knot_idx;
        }

        // given an anchor in index space, find intersecting knot lines in index space
        // in -/+ directions in all dimensions
        void knot_intersections(const vector<KnotIdx>&      anchor,                 // knot indices of anchor for odd degree or
                                                                                    // knot indices of start of rectangle containing anchor for even degree
                                vector<vector<KnotIdx>>&    loc_knots,              // (output) local knot vector in index space
                                int                         extra_p = 0) const      // extra degree in each dim, producing larger local knot vector
        {
            // sanity check that anchor exists in some tensor
            TensorIdx found_tidx;
            if (!lookup_tensor(anchor, found_tidx))
                throw MFAError(fmt::format("knot intersctions(): no tensor contains anchor [{}]", fmt::join(anchor, ",")));

            loc_knots.resize(dom_dim_);
            assert(anchor.size() == dom_dim_);

            for (auto i = 0; i < dom_dim_; i++)
                knot_intersections_dim(anchor, loc_knots[i], i, extra_p);
        }

        // given an anchor in index space, find intersecting knot lines in index space
        // in -/+ directions in one dimension
        void knot_intersections_dim(const vector<KnotIdx>&  anchor,                 // multidim knot indices of anchor for odd degree or
                                                                                    // knot indices of start of rectangle containing anchor for even degree
                                vector<KnotIdx>&            loc_knots,              // (output) local knot vector in index space
                                int                         cur_dim,                // current dimension
                                int                         extra_p = 0) const      // extra degree in each dim, producing larger local knot vector
        {
            // sanity check that anchor exists in some tensor
            TensorIdx found_tidx;
            if (!lookup_tensor(anchor, found_tidx))
                throw MFAError(fmt::format("knot intersctions_dim(): no tensor contains anchor [{}]", fmt::join(anchor, ",")));

            // degree to use
            VectorXi p = p_.array() + extra_p;

            assert(anchor.size() == dom_dim_);

            // walk the t-mesh in current dimension, min. and max. directions outward from the anchor
            // looking for interecting knot lines

            loc_knots.resize(p(cur_dim) + 2);                           // support of basis func. is p+2 knots (p+1 spans) by definition
            int nprev_knots     = (p(cur_dim) + 1) / 2 + 1;             // number of knot intersections before anchor
            int nnext_knots     = p(cur_dim) / 2 + 2;                   // number of knot intersections after anchor
            prev_knot_intersections_dim(anchor, cur_dim, nprev_knots, 0, loc_knots);
            next_knot_intersections_dim(anchor, cur_dim, nnext_knots, nprev_knots - 1, loc_knots);
        }

        // given an anchor in index space, find given number of previous intersecting knot lines in a given dim. in index space
        // writes anchor[cur_dim] at start_pos + nknots - 1
        // assumes caller allocated loc_knots to desired size, which could be larger than nknots because of nonzero start_pos
        void prev_knot_intersections_dim(
                const vector<KnotIdx>&      anchor,                 // multidim knot indices of anchor for odd degree or
                                                                    // knot indices of start of rectangle containing anchor for even degree
                int                         dim,                    // current dimension to intersect
                int                         nknots,                 // number of knot intersections to find, including anchor
                int                         start_pos,              // starting position of writing the knots in loc_knots (often 0, but can write result starting offset from start)
                vector<KnotIdx>&            loc_knots) const        // (output) knot intersections in index space
        {
            KnotIdx         anchor_pos      = start_pos + nknots - 1;   // position of the anchor, which is the end of the knots
            loc_knots[anchor_pos]           = anchor[dim];              // copy the anchor
            vector<KnotIdx> cur_anchor      = anchor;                   // current knot location in the tmesh (index space)

            // from the anchor in the min. direction
            bool found_next = false;
            for (int j = 0; j < nknots - 1; j++)                        // already copied anchor, nknots -1 left
            {
                // find the next knot
                if (cur_anchor[dim] > 0 && (found_next = next_inter(dim, -1, cur_anchor)))         // updates cur_anchor
                    loc_knots[anchor_pos - j - 1] = cur_anchor[dim];    // record the knot
                else                                                    // no more knots in the tmesh
                    loc_knots[anchor_pos - j - 1] = 0;                  // repeat first index as many times as needed
            }
        }

        // given an anchor in index space, find given number of next intersecting knot lines in a given dim. in index space
        // writes anchor[cur_dim] at start_pos
        // assumes caller allocated loc_knots to desired size, which could be larger than nknots because of nonzero start_pos
        void next_knot_intersections_dim(
                const vector<KnotIdx>&      anchor,                 // multidim knot indices of anchor for odd degree or
                                                                    // knot indices of start of rectangle containing anchor for even degree
                int                         dim,                    // current dimension to intersect
                int                         nknots,                 // number of knot intersections to find, including anchor
                int                         start_pos,              // starting position of writing the knots in loc_knots (often 0, but can write result starting offset from start)
                vector<KnotIdx>&            loc_knots) const        // (output) knot intersections in index space
        {
            loc_knots[start_pos]            = anchor[dim];              // copy the anchor
            vector<KnotIdx> cur_anchor      = anchor;                   // current knot location in the tmesh (index space)

            // from the anchor in the max. direction
            bool found_next = false;
            for (int j = 0; j < nknots-1; j++)                          // already copied anchor, nknots - 1 left
            {
                // find the next knot
                if (cur_anchor[dim] < all_knots[dim].size() - 1 && (found_next = next_inter(dim, 1, cur_anchor)))         // updates cur_anchor
                    loc_knots[start_pos + j + 1] = cur_anchor[dim];             // record the knot
                else                                                            // no more knots in the tmesh
                    loc_knots[start_pos + j + 1] = all_knots[dim].size() - 1;   // repeat last index as many times as needed
            }
        }

        // iterates to the next intersection of knot index
        // returns whether the offset target could be found
        bool next_inter(int                      cur_dim,            // current dimension
                        int                      dir,                // direction iterate +/-1
                        vector<KnotIdx>&         target) const       // (input / output) target knot indices, offset by this function
        {
            KnotIdx         temp_target_dim;
            int             ofst;

            if (dir == 1 || dir == -1)
                ofst = dir;
            else
                throw MFAError(fmt::format("next_inter(): dir must be +/- 1\n"));

            // increment the offset until a tensor is found or we run out of index space
            vector<KnotIdx> temp_target = target;
            TensorIdx found_tidx;
            temp_target[cur_dim] += dir;
            while (temp_target[cur_dim] >= 0 && temp_target[cur_dim] < all_knots[cur_dim].size())
            {
                if (lookup_tensor(temp_target, found_tidx))
                {
                    target[cur_dim] = temp_target[cur_dim];
                    return true;
                }
                temp_target[cur_dim] += dir;
            }

            return false;
        }

        // offsets a knot index by some amount within a tensor, skipping over any knots at a deeper level
        // returns whether the full offset was achieved (true) or whether ran out of tensor bounds (false)
        // if the tensor ran out of bounds, computes as much offset as possible, ie, ofst_idx = the min or max bound
        bool knot_idx_ofst(
                const TensorProduct<T>& t,                          // tensor product
                KnotIdx                 orig_idx,                   // starting knot idx
                int                     ofst,                       // offset amount, can be positive or negative
                int                     cur_dim,                    // current dimension
                bool                    edge_check,                 // check for missing control points at global edge
                KnotIdx&                ofst_idx) const             // (output) offset knot idx, can reuse orig_idx if desired
        {
            ofst_idx    = orig_idx;
            int sgn     = (0 < ofst) - (ofst < 0);                  // sgn = 1 for positive ofst, -1 for negative, 0 for zero
            int p       = p_(cur_dim);                              // degree in current dimension
            int pad     = edge_check ? p - 1 : 0;

            // t is completely to the right of orig_idx and we're offsetting left
            // the offsetted point cannot be inside of t
            if (sgn == -1 && t.knot_mins[cur_dim] >= orig_idx)
            {
                ofst_idx = pad;
                return false;
            }

            // t is completely to the left of orig_idx and we're offsetting right
            // the offsetted point cannpt be inside of t
            if (sgn == 1 && t.knot_maxs[cur_dim] <= orig_idx)
            {
                ofst_idx  = all_knots[cur_dim].size() - 1 - pad;
                return false;
            }

            // the offsetted point can be inside of t
            for (auto i = 0; i < abs(ofst); i++)
            {
                while ((long)ofst_idx + sgn >= t.knot_mins[cur_dim]   &&
                        (long)ofst_idx + sgn <= t.knot_maxs[cur_dim]  &&
                        all_knot_levels[cur_dim][ofst_idx + sgn] > t.level)
                    ofst_idx += sgn;

                if (t.knot_mins[cur_dim] == 0 &&
                        (long)ofst_idx + sgn < pad)                                     // missing control points at global min edge
                {
                    ofst_idx = pad;
                    return false;
                }
                if (t.knot_maxs[cur_dim] == all_knots[cur_dim].size() - 1 &&
                        (long)ofst_idx + sgn > all_knots[cur_dim].size() - 1 - pad)   // missing control points at global max edge
                {
                    ofst_idx  = all_knots[cur_dim].size() - 1 - pad;
                    return false;
                }
                if ((long)ofst_idx + sgn < t.knot_mins[cur_dim])
                {
                    ofst_idx = t.knot_mins[cur_dim];
                    return false;
                }
                if ((long)ofst_idx + sgn > t.knot_maxs[cur_dim])
                {
                    ofst_idx = t.knot_maxs[cur_dim];
                    return false;
                }
                ofst_idx += sgn;
            }

            return true;
        }

        // counts number of knot indices between min and max index skipping knots at a deeper level than current tensor
        KnotIdx knot_idx_dist(
                const TensorProduct<T>& t,                          // tensor product
                KnotIdx                 min,                        // min knot idx
                KnotIdx                 max,                        // max knot idx
                int                     cur_dim,                    // current dimension
                bool                    inclusive) const            // whether to include max
        {
            KnotIdx dist    = 0;
            KnotIdx end     = inclusive ? max + 1 : max;
            for (auto idx = min; idx < end; idx++)
            {
                while (idx < end && all_knot_levels[cur_dim][idx] > t.level)
                    idx++;
                if (idx < end)
                    dist++;
            }
            return dist;
        }

        // binary search to find the span in the knots vector containing a given parameter value
        // returns span index i s.t. u is in [ knots[i], knots[i + 1] )
        // NB closed interval at left and open interval at right
        //
        // i will be in the range [p, n], where n = number of control points - 1 because there are
        // p + 1 repeated knots at start and end of knot vector
        // algorithm 2.1, P&T, p. 68
        //
        // CAUTION: can find a span not in the tensor, looks at all knots irrespective of level
        // use only for a t-mesh with one tensor product, where all knots are used in the tensor
        int FindSpan(
                int                     cur_dim,            // current dimension
                T                       u) const            // parameter value
        {
            int nctrl_pts = all_knots[cur_dim].size() - p_(cur_dim) - 1;

            if (u == all_knots[cur_dim][nctrl_pts])
                return nctrl_pts - 1;

            // binary search
            int low = p_(cur_dim);
            int high = nctrl_pts;
            int mid = (low + high) / 2;
            while (u < all_knots[cur_dim][mid] || u >= all_knots[cur_dim][mid + 1])
            {
                if (u < all_knots[cur_dim][mid])
                    high = mid;
                else
                    low = mid;
                mid = (low + high) / 2;
            }

            return mid;
        }

        // binary search to find the span in the knots vector containing a given parameter value
        // returns span index i s.t. u is in [ knots[i], knots[i + 1] )
        // NB closed interval at left and open interval at right
        //
        // i will be in the range [p, n], where n = number of control points - 1 because there are
        // p + 1 repeated knots at start and end of knot vector
        // algorithm 2.1, P&T, p. 68
        //
        // CAUTION: can find a span not in the tensor, looks at all knots irrespective of level
        // use only for a t-mesh with one tensor product, where all knots are used in the tensor
        int FindSpan(
                int                     cur_dim,            // current dimension
                T                       u,                  // parameter value
                int                     nctrl_pts) const    // number of control points in current dim
        {
            if (u == all_knots[cur_dim][nctrl_pts])
                return nctrl_pts - 1;

            // binary search
            int low = p_(cur_dim);
            int high = nctrl_pts;
            int mid = (low + high) / 2;
            while (u < all_knots[cur_dim][mid] || u >= all_knots[cur_dim][mid + 1])
            {
                if (u < all_knots[cur_dim][mid])
                    high = mid;
                else
                    low = mid;
                mid = (low + high) / 2;
            }

            return mid;
        }

        // binary search to find the span in a given tensor for a parameter value
        // returns span index i s.t. u is in [ knots[i], knots[i + 1] ) in the global knots
        // NB closed interval at left and open interval at right
        //
        // prints an error and aborts if u is not in the min,max range of knots in tensor or if levels of u and the span do not match
        int FindSpan(
                int                     cur_dim,            // current dimension
                T                       u,                  // parameter value
                const TensorProduct<T>& tensor) const       // tensor product in tmesh
        {
            if (u < all_knots[cur_dim][tensor.knot_mins[cur_dim]] ||
                    u > all_knots[cur_dim][tensor.knot_maxs[cur_dim]])
            {
                fmt::print(stderr, "FindSpan(): Asking for parameter value outside of the knot min/max of the current tensor. This should not happen.\n");
                fmt::print(stderr, "u {} cur_dim {} knot_mins [{}] knot_maxs [{}]\n",
                        u, cur_dim, fmt::join(tensor.knot_mins, ","), fmt::join(tensor.knot_maxs, ","));
                print_tensor(tensor, true);
                abort();
            }

            int low, high, mid;
            int found = -1;

            if (tensor.knot_mins[cur_dim] == 0)
                low = p_(cur_dim);
            else
                low = 0;
            if (tensor.knot_maxs[cur_dim] == all_knots[cur_dim].size() - 1)
                high = tensor.knot_idxs[cur_dim].size() - p_(cur_dim) - 1;
            else
                high = tensor.knot_idxs[cur_dim].size() - 1;
            mid = (low + high) / 2;

            if (u >= all_knots[cur_dim][tensor.knot_idxs[cur_dim][high]])
                found = high - 1;

            if (found < 0)
            {
                // binary search
                while (u < all_knots[cur_dim][tensor.knot_idxs[cur_dim][mid]] ||
                        u >= all_knots[cur_dim][tensor.knot_idxs[cur_dim][mid + 1]])
                {
                    if (u < all_knots[cur_dim][tensor.knot_idxs[cur_dim][mid]])
                        high = mid;
                    else
                        low = mid;
                    mid = (low + high) / 2;
                }
                found = mid;
            }

            // sanity checks
            // TODO: comment out once code is stable
            if (all_knot_levels[cur_dim][tensor.knot_idxs[cur_dim][found]] > tensor.level)
            {
                fmt::print(stderr, "FindSpan(): level mismatch at found span. This should not happen.\n");
                fmt::print(stderr, "u {} dim {} knot idx {} knot value {} knot level {} tensor level {}\n",
                        u, cur_dim, tensor.knot_idxs[cur_dim][found], all_knots[cur_dim][tensor.knot_idxs[cur_dim][found]],
                        all_knot_levels[cur_dim][tensor.knot_idxs[cur_dim][found]], tensor.level);
                print_tensor(tensor, true);
                abort();
            }
            bool error = false;
            if (u < all_knots[cur_dim][tensor.knot_idxs[cur_dim][found]])
                error = true;
            if (tensor.knot_maxs[cur_dim] == all_knots[cur_dim].size() - 1)                     // tensor is at global max end
            {
                if (u == 1.0 && u > all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                    error = true;
                else if (u < 1.0 && u >= all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                    error = true;
            }
            else                                                                                // tensor is not at global max end
            {
                if (p_(cur_dim) % 2 == 0)                                                        // even degree
                {
                    if (u >= all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                        error = true;
                }
                else                                                                            // odd degree
                {
                    if (tensor.knot_maxs[cur_dim] > tensor.knot_idxs[cur_dim][found + 1] &&     // right edge of found span is inside the max of the tensor
                            u >= all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                        error = true;
                    if (tensor.knot_maxs[cur_dim] == tensor.knot_idxs[cur_dim][found + 1] &&    // right edge of found span is at max of the tensor
                            u > all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]])
                        error = true;
                }
            }
            if (error)
                throw MFAError(fmt::format("FindSpan(): parameter {} in dim not in local span [{}, {}) global span [{}, {}) = knots [{}, {})\n",
                            u, cur_dim, found, found + 1, tensor.knot_idxs[cur_dim][found], tensor.knot_idxs[cur_dim][found + 1],
                            all_knots[cur_dim][tensor.knot_idxs[cur_dim][found]], all_knots[cur_dim][tensor.knot_idxs[cur_dim][found + 1]]));

            return tensor.knot_idxs[cur_dim][found];
        }

        // updates the vectors of knots belonging to this tensor
        // assumes knot_mins and knot_maxs are up to date, only considering knots in that range
        void tensor_knot_idxs(TensorProduct<T>& t)
        {
            t.knot_idxs.resize(dom_dim_);

            for (auto k = 0; k < dom_dim_; k++)
            {
                // walk the knots, copying relevant indices
                t.knot_idxs[k].clear();
                for (auto i = t.knot_mins[k]; i <= t.knot_maxs[k]; i++)
                {
                    if (i == t.knot_mins[k] || i == t.knot_maxs[k] || all_knot_levels[k][i] <= t.level)
                        t.knot_idxs[k].push_back(i);
                }
            }
        }

        // ----- anchors and control points ----- //

        // given a point in parameter space to decode, compute p + 1 anchor points in all dims in knot index space
        // anchors correspond to those basis functions that cover the decoding point
        // anchors are the centers of basis functions and locations of corresponding control points, in knot index space
        // in Bazilevs 2010, knot indices start at 1, but mine start at 0
        // returns index of deepest level tensor containing the parameters of the point to decode
        TensorIdx anchors(const VectorX<T>&          param,             // parameter value in each dim. of desired point
                          vector<vector<KnotIdx>>&   anchors) const     // (output) anchor points in index space
        {
            anchors.resize(dom_dim_);

            // find tensor containing param
            TensorIdx t_idx = find_tensor(param);

            // convert param to span
            vector<KnotIdx> target(dom_dim_);
            for (auto i = 0; i < dom_dim_; i++)
                target[i] = FindSpan(i, param(i), tensor_prods[t_idx]);

            // find local knot vector (p + 2) knot intersections
            vector<vector<KnotIdx>> loc_knots(dom_dim_);
            knot_intersections(target, loc_knots);

            // take correct p + 1 anchors out of the p + 2 found above
            for (auto i = 0; i < dom_dim_; i++)
            {
                anchors[i].resize(p_(i) + 1);
                for (auto j = 0; j < p_(i) + 1; j++)
                {
                    if (p_(i) % 2 == 0)                             // even degree: first p + 1 anchors, skip last one
                        anchors[i][j] = loc_knots[i][j];
                    else                                            // odd degree: last p + 1 anchors, skip first one
                        anchors[i][j] = loc_knots[i][j + 1];
                }
            }

            return t_idx;
        }

        // for a given tensor, return linear index of control point corresponding to given anchor
        // anchor is in global knot index space (includes knots at higher refinement levels than the tensor)
        size_t anchor_ctrl_pt_idx(
                const TensorProduct<T>& t,                          // tensor product
                const vector<KnotIdx>&  anchor,                     // anchor
                bool&                   found,                      // whether anchor was found in tensor
                bool                    check = true) const         // check anchor validity and global/local knot index agreement
        {
            // TODO: remove once stable
            if (check)
            {
                for (auto i = 0; i < dom_dim_; i++)
                {
                    if (anchor[i] < (p_(i) + 1) / 2 || anchor[i] >= all_knots[i].size() - (p_(i) + 1) / 2)
                        throw MFAError(fmt::format("anchor_ctrl_pt_idx(): anchor[{}] = {} must be in [{}, {}]",
                                    i, anchor[i], (p_(i) + 1) / 2, all_knots[i].size() - (p_(i) + 1) / 2 - 1));
                }
            }

            VectorXi ijk = anchor_ctrl_pt_ijk(t, anchor, found, check);    // multidim local index of anchor

            VolIterator vol_iter(t.nctrl_pts);

            return vol_iter.ijk_idx(ijk);
        }

        // for a given tensor, return multidim index of control point corresponding to given anchor
        // anchor is in global knot index space (includes knots at higher refinement levels than the tensor)
        VectorXi anchor_ctrl_pt_ijk(
                const TensorProduct<T>& t,                          // tensor product
                const vector<KnotIdx>&  anchor,                     // anchor
                bool&                   found,                      // (output) whether anchor was found in tensor
                bool                    check = true) const         // check anchor validity and global/local knot index agreement
        {
            // TODO: remove once stable
            if (check)
            {
                for (auto i = 0; i < dom_dim_; i++)
                {
                    if (anchor[i] < (p_(i) + 1) / 2 || anchor[i] >= all_knots[i].size() - (p_(i) + 1) / 2)
                        throw MFAError(fmt::format("anchor_ctrl_pt_ijk(): anchor[{}] = {} must be in [{}, {}]",
                                    i, anchor[i], (p_(i) + 1) / 2, all_knots[i].size() - (p_(i) + 1) / 2 - 1));
                }
            }

            VectorXi ijk(dom_dim_);                                 // multidim local index of anchor
            for (auto i = 0; i < dom_dim_; i++)
            {
                ijk(i) = anchor_ctrl_pt_dim(t, i, anchor[i], found, check);

                // TODO: remove once stable
                if (ijk(i) < 0)
                    throw MFAError(fmt::format("anchor_ctrl_pt_ijk(): for anchor[{}], ijk(dim {}) = {}, which is  < 0", fmt::join(anchor, ","), i, ijk(i)));
            }

            return ijk;
        }

        // for a given tensor, return index of control point in a given dimension corresponding to given anchor
        // anchor is in global knot index space (includes knots at higher refinement levels than the tensor)
        CtrlIdx anchor_ctrl_pt_dim(
                const TensorProduct<T>& t,                          // tensor product
                int                     dim,                        // current dimension
                KnotIdx                 anchor,                     // anchor
                bool&                   found,                      // (output) whether anchor was found in tensor
                bool                    check = true) const         // check anchor validity and global/local knot index agreement
        {
            size_t ctrl_idx;
            ctrl_idx = global2local_knot_idx(anchor, t, dim, found, check);

            // TODO: remove once stable
            if (check)
            {
                if ((t.knot_mins[dim] == 0 && ctrl_idx < (p_(dim) + 1) / 2) ||
                        (t.knot_maxs[dim] == all_knots[dim].size() - 1 && ctrl_idx >= all_knots[dim].size() - (p_(dim) + 1) / 2))
                    throw MFAError(fmt::format("anchor_ctrl_pt_dim(): ctrl_idx out of range: dim {} ctrl_idx {} must be in [{}, {})",
                                dim, ctrl_idx, (p_(dim) + 1) / 2, all_knots[dim].size() - (p_(dim) + 1) / 2));
            }

            if (t.knot_mins[dim] == 0)
                ctrl_idx -= (p_(dim) + 1) / 2;

            // TODO: remove once stable
            if (check && ctrl_idx >= t.nctrl_pts(dim))
                throw MFAError(fmt::format("anchor_ctrl_pt_dim(): ctrl_idx out of range: dim {} ctrl_idx {} must be < {}",
                            dim, ctrl_idx, t.nctrl_pts(dim)));

            return ctrl_idx;
        }

        // for a given tensor, get anchor of control point, given control point multidim index
        // anchor is in global knot index space at the correct level of the tensor
        void ctrl_pt_anchor(const TensorProduct<T>& t,              // tensor product
                            const VectorXi&         ijk,            // multidim index of control point
                            vector<KnotIdx>&        anchor) const   // (output) anchor
        {
            for (auto j = 0; j < dom_dim_; j++)
                anchor[j] = ctrl_pt_anchor_dim(j, t, ijk(j));
        }

        // for a given tensor, return anchor of control point in one dimension, given control point index in one dim
        // anchor is in global knot index space at the correct level of the tensor
        KnotIdx ctrl_pt_anchor_dim(
                int                     dim,                        // dimension
                const TensorProduct<T>& t,                          // tensor product
                int                     idx) const                  // index of control point in current dim
        {
            KnotIdx anchor;

            bool retval = knot_idx_ofst(t, t.knot_mins[dim], idx, dim, false, anchor);
            if (!retval)
            {
                fmt::print(stderr, "ctrl_pt_anchor_dim(): invalid offset result\n");
                abort();
            }

            if (t.knot_mins[dim] == 0)
            {
                retval = knot_idx_ofst(t, anchor, (p_(dim) + 1) / 2, dim, false, anchor);
                if (!retval)
                {
                    fmt::print(stderr, "ctrl_pt_anchor_dim(): invalid offset result\n");
                    abort();
                }
            }

            // ensure anchor isn't at a deeper level, if so, back up to earlier anchor in this tensor
            while (all_knot_levels[dim][anchor] > t.level && anchor > t.knot_mins[dim])
                anchor--;

            return anchor;
        }

        // expands anchors originating at a point in some tensor adjusted for all intersecting tensors
        // result is the extents (first and last) of the original anchors possibly expanded in all directions to cover neigboring tensors
        // returns whether any changes were made
        bool expand_anchors(
                const vector<vector<KnotIdx>>&  orig_anchors,               // original original anchors in all dims
                TensorIdx                       t_idx,                      // original tensor product
                vector<vector<KnotIdx>>&        anchor_extents) const       // (output) possibly expanded first and last anchors in all dims
        {
            bool changed = false;
            auto& t = tensor_prods[t_idx];
            anchor_extents.resize(dom_dim_);                                // output extents (front and back) possibly expanded

            for (auto i = 0; i < dom_dim_; i++)
            {
                anchor_extents[i].resize(2);
                anchor_extents[i][0] = orig_anchors[i].front();
                anchor_extents[i][1] = orig_anchors[i].back();
            }

            vector<KnotIdx> anchor(dom_dim_);                               // one multidim anchor
            for (auto a = 0; a < orig_anchors[0].size(); a++)               // for all multidim original anchors
            {
                for (auto i = 0; i < dom_dim_; i++)
                    anchor[i] = orig_anchors[i][a];

                TensorIdx found_tidx;

                if (!lookup_tensor(anchor, found_tidx))
                    continue;

                auto& t_k = tensor_prods[found_tidx];
                for (auto i = 0; i < dom_dim_; i++)                     // for all dims
                {
                    for (auto j = 0; j < orig_anchors[i].size(); j++)   // for original anchors in current dim
                    {
                        if (orig_anchors[i][j] >= t_k.knot_mins[i] && orig_anchors[i][j] < t_k.knot_maxs[i])
                        {
                            KnotIdx ofst_idx, temp_anchor;

                            // t_k is to the max side of t
                            temp_anchor = orig_anchors[i][j];
                            while(temp_anchor >= t_k.knot_mins[i] && all_knot_levels[i][temp_anchor] > t_k.level)
                            {
                                temp_anchor--;
                                // if we try to offset more than the tensor boundary, knot_idx_ofst clamps the offset to the knot_mins, maxs, which is what we want
                                knot_idx_ofst(t_k, anchor_extents[i][0], -1, i, false, ofst_idx);
                                anchor_extents[i][0] = ofst_idx;
                                changed = true;
                            }

                            // t_k is to the min side of t
                            temp_anchor = orig_anchors[i][j];
                            while(temp_anchor <= t_k.knot_maxs[i] && all_knot_levels[i][temp_anchor] > t_k.level)
                            {
                                temp_anchor++;
                                // if we try to offset more than the tensor boundary, knot_idx_ofst clamps the offset to the knot_mins, maxs, which is what we want
                                knot_idx_ofst(t_k, anchor_extents[i][1], 1, i, false, ofst_idx);
                                anchor_extents[i][1] = ofst_idx;
                                changed = true;
                            }
                        }
                    }
                }
            }
            return changed;
        }

        // ----- utilities ----- //

        // checks if a_mins, maxs intersect b_mins, maxs, with the intersection in c_mins, c_maxs
        // returns whether there is an intersection (larger than edges just touching)
        bool intersects(const vector<KnotIdx>&  a_mins,
                        const vector<KnotIdx>&  a_maxs,
                        const vector<KnotIdx>&  b_mins,
                        const vector<KnotIdx>&  b_maxs,
                        vector<KnotIdx>&        c_mins,
                        vector<KnotIdx>&        c_maxs) const
        {
            // check that sizes are identical
            size_t a_size = a_mins.size();
            if (a_size != a_maxs.size() || a_size != b_mins.size() || a_size != b_maxs.size() ||
                    a_size != c_mins.size() || a_size != c_maxs.size())
            {
                fprintf(stderr, "Error: intersects(): size mismatch\n");
                abort();
            }

            // check intersection cases
            for (auto i = 0; i < a_size; i++)
            {
                // no intersection
                if (a_maxs[i] <= b_mins[i] || b_maxs[i] <= a_mins[i])
                        return false;

                // a is a subset of b
                else if (a_mins[i] >= b_mins[i] && a_maxs[i] <= b_maxs[i])
                {
                    c_mins[i] = a_mins[i];
                    c_maxs[i] = a_maxs[i];
                }

                // b is a subset of a
                else if (b_mins[i] >= a_mins[i] && b_maxs[i] <= a_maxs[i])
                {
                    c_mins[i] = b_mins[i];
                    c_maxs[i] = b_maxs[i];
                }

                // a is to the left of b but overlaps it
                else if (a_maxs[i] > b_mins[i] && a_maxs[i] < b_maxs[i])
                {
                    c_mins[i] = b_mins[i];
                    c_maxs[i] = a_maxs[i];
                }

                // b is to the left of a but overlaps it
                else if (b_maxs[i] > a_mins[i] && b_maxs[i] < a_maxs[i])
                {
                    c_mins[i] = a_mins[i];
                    c_maxs[i] = b_maxs[i];
                }

                else
                {
                    fprintf(stderr, "Error: intersects(): ran out of cases\n");
                    abort();
                }
            }

            return true;
        }

        // checks if a_mins, maxs are a subset of b_mins, maxs
        // identical bounds counts as a subset (does not need to be proper subset)
        bool subset(const vector<KnotIdx>& a_mins,
                    const vector<KnotIdx>& a_maxs,
                    const vector<KnotIdx>& b_mins,
                    const vector<KnotIdx>& b_maxs) const
        {
            // check that sizes are identical
            size_t a_size = a_mins.size();
            if (a_size != a_maxs.size() || a_size != b_mins.size() || a_size != b_maxs.size())
            {
                fprintf(stderr, "Error, size mismatch in subset()\n");
                abort();
            }

            // check subset condition
            for (auto i = 0; i < a_size; i++)
                if (a_mins[i] < b_mins[i] || a_maxs[i] > b_maxs[i])
                        return false;

            return true;
        }

        // forms union of mins and maxs of a and b and stores result in res
        void merge(const vector<KnotIdx>& a_mins,
                   const vector<KnotIdx>& a_maxs,
                   const vector<KnotIdx>& b_mins,
                   const vector<KnotIdx>& b_maxs,
                   vector<KnotIdx>&       res_mins,
                   vector<KnotIdx>&       res_maxs)
        {
            // check that sizes are identical
            size_t a_size = a_mins.size();
            if (a_size != a_maxs.size() || a_size != b_mins.size() || a_size != b_maxs.size())
            {
                fprintf(stderr, "Error, size mismatch in subset()\n");
                abort();
            }

            res_mins.resize(a_size);
            res_maxs.resize(a_size);

            // form union
            for (auto i = 0; i < a_size; i++)
            {
                res_mins[i] = a_mins[i] < b_mins[i] ? a_mins[i] : b_mins[i];
                res_maxs[i] = a_maxs[i] > b_maxs[i] ? a_maxs[i] : b_maxs[i];
            }
        }

        // checks if a point in index space is in [knot_mins, knot_maxs] in all dims
        bool in(const vector<KnotIdx>&  pt,
                const vector<KnotIdx>&  knot_mins,
                const vector<KnotIdx>&  knot_maxs) const
        {
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (pt[i] < knot_mins[i] || pt[i] > knot_maxs[i])
                    return false;
                // for even degree, the max edge of an interior tensor is an open interval
                if (p_(i) % 2 == 0 && pt[i] == knot_maxs[i] && knot_maxs[i] < all_knots[i].size() - 1)
                    return false;
            }
            return true;
        }

        // checks if a point in parameter space is in a tensor product
        bool in(const VectorX<T>&       param,
                const TensorProduct<T>& tensor) const
        {
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (param(i) < all_knots[i][tensor.knot_mins[i]] || param(i) > all_knots[i][tensor.knot_maxs[i]])
                    return false;
                // for even degree, the max edge of an interior tensor is an open interval
                if (param(i) == all_knots[i][tensor.knot_maxs[i]] && tensor.knot_maxs[i] < all_knots[i].size() - 1 && p_(i) % 2 == 0)
                    return false;
            }
            return true;
        }

        // ----- diagnostics ----- //

        // check number of knots belonging to this tensor against the number of control points (for debugging)
        bool check_num_knots_ctrl_pts(TensorIdx tidx)
        {
            auto& t = tensor_prods[tidx];
            for (auto i = 0; i < dom_dim_; i++)
            {
                int nctrl = t.knot_idxs[i].size() - 1;
                if (p_(i) % 2)                                  // odd degree
                    nctrl++;
                if (t.knot_mins[i] == 0)                        // min. edge of global domain
                    nctrl -= (p_(i) + 1) / 2;
                if (t.knot_maxs[i] == all_knots[i].size() - 1)  // max. edge of global domain
                    nctrl -= (p_(i) + 1) / 2;

                if (nctrl != t.nctrl_pts(i))
                {
                    fmt::print(stderr, "Error: check_num_knots_ctrl_pts(): Number of knots and control points in tensor {} in dim. {} do not agree.\n",
                            tidx, i);
                    print_tensor(t, true, false, false);
                    return false;
                }
            }
            return true;
        }

        // check number of knots belonging to this tensor is at least degree + extra
        // returns true if the check passes
        bool check_num_knots_degree(TensorProduct<T>&   t,
                                    int                 extra)
        {
            for (auto k = 0; k < dom_dim_; k++)
            {
                KnotIdx dist = knot_idx_dist(t, t.knot_mins[k], t.knot_maxs[k], k, false);
                if (p_(k) % 2 == 0 && dist < p_(k) + extra || p_(k) % 2 == 1 && dist <  p_(k) + extra - 1)
                    return false;
            }
            return true;
        }

        // check number of control points belonging to this tensor is at least degree + extra
        // returns true if the check passes
        bool check_num_ctrl_degree(TensorIdx            tidx,
                                    int                 extra)
        {
            auto& t = tensor_prods[tidx];
            for (auto j = 0; j < dom_dim_; j++)
            {
                if (t.nctrl_pts(j) < p_(j) + extra)
                    return false;
            }
            return true;
        }

        // check all tensors for minimum size
        // returns true if the check passes
        bool check_min_size(int min_interior,                           // minimum size of interior tensors
                            int min_border) const                       // minimum size of global border tensors
        {
            for (auto i = 0; i < tensor_prods.size(); i++)
            {
                auto& t = tensor_prods[i];

                for (auto j = 0; j < dom_dim_; j++)
                {
                    KnotIdx dist = knot_idx_dist(t, t.knot_mins[j], t.knot_maxs[j], j, true);
                    if (t.knot_mins[j] == 0 || t.knot_maxs[j] == all_knots[j].size() - 1)      // border tensor
                    {
                        if (dist < min_border)
                        {
                            fmt::print(stderr, "check_min_size(): border tensor idx {} cur_dim {} has {} knots which is less than min_border {}\n",
                                    i, j, dist, min_border);
                            return false;
                        }
                    }
                    else                                                                    // interior tensor
                    {
                        if (dist < min_interior)
                        {
                            fmt::print(stderr, "check_min_size(): interior tensor idx {} cur_dim {} has {} knots which is less than min_interior {}\n",
                                    i, j, dist, min_interior);
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        // check all tensors that local knot indices agree with global knot indices
        // returns true if the check passes
        bool check_local_knots() const
        {
            for (auto i = 0; i < tensor_prods.size(); i++)
            {
                auto& t = tensor_prods[i];

                for (auto j = 0; j < dom_dim_; j++)
                {
                    KnotIdx dist = knot_idx_dist(t, t.knot_mins[j], t.knot_maxs[j], j, true);
                    if (dist != t.knot_idxs[j].size())
                    {
                        fmt::print(stderr, "check_local_knots(): tensor idx {} cur_dim {} distance between global knots {} != size of local knots {}\n",
                                i, j, dist, t.knot_idxs[j].size());
                        return false;
                    }

                    auto cur_knot_idx = t.knot_mins[j];
                    for (auto k = 0; k < t.knot_idxs[j].size(); k++)
                    {
                        if (t.knot_idxs[j][k] != cur_knot_idx)
                        {
                            fmt::print(stderr, "check_local_knots(): tensor idx {} cur_dim {} {}th local knot idx {} does not match global knot idx {}\n",
                                    i, j, k, t.knot_idxs[j][k], cur_knot_idx);
                            return false;
                        }

                        if (k < t.knot_idxs[j].size() - 1)
                        {
                            if (!knot_idx_ofst(t, cur_knot_idx, 1, j, false, cur_knot_idx))
                            {
                                fmt::print(stderr, "check_local_knots(): tensor idx {} cur_dim {} k = {} cur_knot_idx {} knot_idx_ofst() failed\n",
                                        i, j, k, cur_knot_idx);
                                return false;
                            }
                        }
                    }
                }
            }

            return true;
        }

        // check edge knots of tensor product to ensure they don't exceed a given level
        // returns: true if ok
        bool check_knot_edge_level(TensorProduct<T>&    t,
                                   int                  level)
        {
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (all_knot_levels[i][t.knot_mins[i]] > level)
                {
                    fmt::print(stderr, "check_knot_edge_level(): knot_mins in dim {} is at level {} which is > allowed level {}\n",
                            i, all_knot_levels[i][t.knot_mins[i]], level);
                    return false;
                }
                if (all_knot_levels[i][t.knot_maxs[i]] > level)
                {
                    fmt::print(stderr, "check_knot_edge_level(): knot_maxs in dim {} is at level {} which is > allowed level {}\n",
                            i, all_knot_levels[i][t.knot_maxs[i]], level);
                    return false;
                }
            }
            return true;
        }

        void print_tensor(
            const TensorProduct<T>&     t,
            bool                        print_knots     = false,
            bool                        print_ctrl_pts  = false,
            bool                        print_weights   = false) const
        {
            fmt::print(stderr, "parent_exists {} parent {} children [{}]\n", t.parent_exists, t.parent, fmt::join(t.children, " "));
            fmt::print(stderr, "knot_mins [{}] knot_maxs [{}]\n", fmt::join(t.knot_mins, " "), fmt::join(t.knot_maxs, " "));
            fmt::print(stderr, "nctrl_pts [{}]\n", fmt::join(t.nctrl_pts, " "));

            fmt::print(stderr, "n_local_knots [ ");
            for (int i = 0; i < dom_dim_; i++)
                fmt::print(stderr, "{} ", t.knot_idxs[i].size());
            fmt::print(stderr, "]\n\n");

            if (print_knots)
            {
                for (int i = 0; i < dom_dim_; i++)
                {
                    fmt::print(stderr, "knots[dim {}]\n", i);
                    for (auto j = 0; j < t.knot_idxs[i].size(); j++)
                    {
                        KnotIdx idx = t.knot_idxs[i][j];
                        fmt::print(stderr, "idx {}: {:.4} (l {}) [p {}]\n",
                                idx, all_knots[i][idx], all_knot_levels[i][idx], all_knot_param_idxs[i][idx]);
                    }
                    fmt::print(stderr, "\n");
                }
            }

            if (print_ctrl_pts)
                cerr << "ctrl_pts:\n" << t.ctrl_pts << endl;

            if (print_weights)
                cerr << "weights:\n" << t.weights << endl;

            if (!print_knots)
                fmt::print(stderr, "\n");
        }

        void print_tensors(
                bool print_knots    = false,
                bool print_ctrl_pts = false,
                bool print_weights  = false) const
        {
            for (auto j = 0; j < tensor_prods.size(); j++)
            {
                const TensorProduct<T>& t = tensor_prods[j];
                if (j == 0)
                    fmt::print(stderr, "-----\n\n");
                fmt::print(stderr, "tensor_prods[{}] level={} done={}\n", j, t.level, t.done);
                print_tensor(t, print_knots, print_ctrl_pts, print_weights);
                fmt::print(stderr, "-----\n\n");
            }
        }

        void print_knots() const
        {
            for (int i = 0; i < dom_dim_; i++)
            {
                fprintf(stderr, "all_knots[dim %d]\n", i);
                for (auto j = 0; j < all_knots[i].size(); j++)
                    fprintf(stderr, "%d: %.4lf (l %d) [p %lu]\n",
                            j, all_knots[i][j], all_knot_levels[i][j], all_knot_param_idxs[i][j]);
                fprintf(stderr, "\n");
            }
        }

        void print(
                bool print_all_knots    = false,
                bool print_local_knots  = false,
                bool print_ctrl_pts     = false,
                bool print_weights      = false) const
        {
            if (print_all_knots)
            {
                print_knots();
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "T-mesh has %lu tensor products\n\n", tensor_prods.size());
            print_tensors(print_local_knots, print_ctrl_pts, print_weights);
            fprintf(stderr, "\n");
        }

        // debug: check that knots are nondecreasing
        // returns true if knots are ordered correctly
        bool check_knots_order()
        {
            for (auto i = 0; i < dom_dim_; i++)
            {
                for (auto j = 0; j < all_knots[i].size() - 1; j++)
                {
                    if (all_knots[i][j] > all_knots[i][j + 1])
                        return false;
                }
            }
            return true;
        }

        void printDetails(int verbose_)
        {
            if (verbose_ >= 1)
            {
                if (tensor_prods.size() == 0)
                {
                    fmt::print("    *Empty control mesh*\n");
                }
                else if (tensor_prods.size() == 1)
                {
                    vector<int> knot_sizes(dom_dim_);
                    transform(all_knots.begin(), all_knots.end(), knot_sizes.begin(), [](auto k){return k.size();});
                    fmt::print(stderr, "    Number of control points: [{}]\n", fmt::join(tensor_prods[0].nctrl_pts, ","));
                    fmt::print(stderr, "    Number of knots: [{}]\n", fmt::join(knot_sizes, " "));
                }
                else
                {
                    vector<int> knot_sizes(dom_dim_);
                    transform(all_knots.begin(), all_knots.end(), knot_sizes.begin(), [](auto k){return k.size();});
                    fmt::print(stderr, "    T-mesh details:\n");
                    fmt::print(stderr, "      Size of all_knots: [{}]\n", fmt::join(knot_sizes, " "));
                    fmt::print(stderr, "      Number of tensor products: {}\n", tensor_prods.size());
                    if (verbose_ >= 2)
                    {
                        for (int i = 0; i < tensor_prods.size(); i++)
                        {
                            fmt::print(stderr, "      Tensor product {}:\n", i);
                            fmt::print(stderr, "        Level: {}\n", tensor_prods[i].level);
                            fmt::print(stderr, "        Number of control points: [{}]\n", fmt::join(tensor_prods[i].nctrl_pts, ","));
                            if (verbose_ >= 3)
                            {
                                fmt::print(stderr, "        Knot mins: [{}]\n", fmt::join(tensor_prods[i].knot_mins, ","));
                                fmt::print(stderr, "        Knot maxs: [{}]\n", fmt::join(tensor_prods[i].knot_maxs, ","));
                                fmt::print(stderr, "        Encoding done: {}\n", tensor_prods[i].done);
                            }
                        }
                    }
                }
            }
        }
    };
}

#endif
