//--------------------------------------------------------------
// T-mesh object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _TMESH_HPP
#define _TMESH_HPP

using namespace std;

using KnotIdx   = size_t;
using TensorIdx = size_t;

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
    vector<vector<TensorIdx>>   next;                   // next[dim][index of next tensor product]
    vector<vector<TensorIdx>>   prev;                   // prev[dim][index of previous tensor product]
    int                         level;                  // refinement level
};

namespace mfa
{
    template <typename T>
    struct Tmesh
    {
        vector<vector<T>>           all_knots;          // all_knots[dimension][index]
        vector<vector<int>>         all_knot_levels;    // refinement levels of all_knots[dimension][index]
        vector<TensorProduct<T>>    tensor_prods;       // all tensor products
        int                         dom_dim_;           // domain dimensionality
        VectorXi                    p_;                 // degree in each dimension
        int                         min_dim_;           // starting coordinate of this model in full-dimensional data
        int                         max_dim_;           // ending coordinate of this model in full-dimensional data

        Tmesh(int               dom_dim,                // number of domain dimension
              const VectorXi&   p,                      // degree in each dimension
              int               min_dim,                // starting coordinate of this model in full-dimensional data
              int               max_dim,                // ending coordinate of this model in full-dimensional data
              size_t            ntensor_prods =  0) :   // number of tensor products to allocate
                dom_dim_(dom_dim),
                p_(p),
                min_dim_(min_dim),
                max_dim_(max_dim)
        {
            all_knots.resize(dom_dim_);
            all_knot_levels.resize(dom_dim_);

            if (ntensor_prods)
                tensor_prods.resize(ntensor_prods);
        }

        // initialize knots
        void init_knots(VectorXi& nctrl_pts)
        {
            for (auto i = 0; i < dom_dim_; i++)
            {
                all_knots[i].resize(nctrl_pts(i) + p_(i) + 1);
                all_knot_levels[i].resize(nctrl_pts(i) + p_(i) + 1);
            }
        }

        // insert a knot into all_knots
        void insert_knot(int        dim,                // dimension of knot vector
                         KnotIdx    pos,                // new position in all_knots[dim] of inserted knot
                         int        level,              // refinement level of inserted knot
                         T          knot)               // knot value to be inserted
        {
            all_knots[dim].insert(all_knots[dim].begin() + pos, knot);
            all_knot_levels[dim].insert(all_knot_levels[dim].begin() + pos, level);

            // adjust tensor product knot_mins and knot_maxs
            for (TensorProduct<T>& t: tensor_prods)
            {
                if (t.knot_mins[dim] >= pos)
                    t.knot_mins[dim]++;
                if (t.knot_maxs[dim] >= pos)
                    t.knot_maxs[dim]++;
            }
        }

        // append a tensor product to the back of tensor_prods
        void append_tensor(const vector<KnotIdx>&   knot_mins,      // indices in all_knots of min. corner of tensor to be inserted
                           const vector<KnotIdx>&   knot_maxs)      // indices in all_knots of max. corner
        {
            bool vec_grew;                          // vector of tensor_prods grew
            bool tensor_inserted = false;           // the desired tensor was already inserted

            // create a new tensor product
            TensorProduct<T> new_tensor;
            new_tensor.next.resize(dom_dim_);
            new_tensor.prev.resize(dom_dim_);
            new_tensor.knot_mins = knot_mins;
            new_tensor.knot_maxs = knot_maxs;

            // initialize control points
            new_tensor.nctrl_pts.resize(dom_dim_);
            size_t tot_nctrl_pts = 1;
            if (!tensor_prods.size())
            {
                new_tensor.level = 0;

                // resize control points
                // level 0 has only one box of control points
                for (auto j = 0; j < dom_dim_; j++)
                {
                    new_tensor.nctrl_pts[j] = all_knots[j].size() - p_[j] - 1;
                    tot_nctrl_pts *= new_tensor.nctrl_pts[j];
                }
                new_tensor.ctrl_pts.resize(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
                new_tensor.weights.resize(tot_nctrl_pts);
            }
            else
            {
                new_tensor.level = tensor_prods.back().level + 1;

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
                    if (p_[j] % 2 == 0)         // even degree: anchors are between knot lines
                        nanchors = nknots - 1;
                    else                            // odd degree: anchors are on knot lines
                        nanchors = nknots;
                    if (knot_mins[j] < p_[j] - 1)                       // skip up to p-1 anchors at start of global knots
                        nanchors -= (p_[j] - 1 - knot_mins[j]);
                    if (knot_maxs[j] > all_knots[j].size() - p_[j])     // skip up to p-1 anchors at end of global knots
                        nanchors -= (knot_maxs[j] + p_[j] - all_knots[j].size());
                    new_tensor.nctrl_pts[j] = nanchors;
                    tot_nctrl_pts *= nanchors;

                    // debug
//                     fprintf(stderr, "appending tensor with %ld knots and %d control points in dimension %d\n", nknots, new_tensor.nctrl_pts[j], j);
                }
                new_tensor.ctrl_pts.resize(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
                new_tensor.weights.resize(tot_nctrl_pts);
            }

            vector<int> split_side(dom_dim_);       // whether min (-1) or max (1) or both (2) sides of
                                                    // new tensor are inside existing tensor (one value for each dim.)

            // check for intersection of the new tensor with existing tensors
            do
            {
                vec_grew = false;           // tensor_prods grew and iterator is invalid
                bool knots_match;           // intersect resulted in a tensor with same knot mins, maxs as tensor to be added

                for (auto j = 0; j < tensor_prods.size(); j++)
                {
                    // debug
//                     fprintf(stderr, "checking for intersection between new tensor and existing tensor idx=%lu\n", j);

                    if (nonempty_intersection(new_tensor, tensor_prods[j], split_side))
                    {
                        // debug
//                         fprintf(stderr, "intersection found between new tensor and existing tensor idx=%lu split_side=[%d %d]\n",
//                                 j, split_side[0], split_side[1]);
//                         fprintf(stderr, "\ntensors before intersection\n\n");
//                         print();

                        if ((vec_grew = intersect(new_tensor, j, split_side, knots_match)) && vec_grew)
                        {
                            if (knots_match)
                                tensor_inserted = true;

                            // debug
                            fprintf(stderr, "\ntensors after intersection\n\n");
                            print();

                            break;  // adding a tensor invalidates iterator, start iteration over
                        }
                    }
                }
            } while (vec_grew);   // keep checking until no more tensors are added

            // the new tensor has either been added already or still needs to be added
            // either way, create reference to new tensor and get its index at the end of vector of tensor prods.
            TensorIdx           new_tensor_idx;
            TensorProduct<T>&   new_tensor_ref = new_tensor;
            if (!tensor_inserted)
            {
                // new tensor will go at the back of the vector
                new_tensor_idx = tensor_prods.size();
                new_tensor_ref = new_tensor;
            }
            else
            {
                // new tensor is already at the back of the vector
                new_tensor_idx = tensor_prods.size() - 1;
                new_tensor_ref = tensor_prods[new_tensor_idx];
            }

            // adjust next and prev pointers for new tensor
            for (int j = 0; j < dom_dim_; j++)
            {
                for (auto k = 0; k < new_tensor_idx; k++)
                {
                    // debug
//                     fprintf(stderr, "final add: cur_dim=%d new_tensor_idx=%lu checking existing_tensor_idx=%lu\n", j, new_tensor_idx, k);

                    TensorProduct<T>& existing_tensor_ref = tensor_prods[k];
                    int adjacent_retval = adjacent(new_tensor_ref, existing_tensor_ref, j);

                    if (adjacent_retval == 1)
                    {
                        new_tensor_ref.next[j].push_back(k);
                        existing_tensor_ref.prev[j].push_back(new_tensor_idx);
                    }
                    else if (adjacent_retval == -1)
                    {
                        new_tensor_ref.prev[j].push_back(k);
                        existing_tensor_ref.next[j].push_back(new_tensor_idx);
                    }
                }
            }

            // add the tensor
            if (!tensor_inserted)
                tensor_prods.push_back(new_tensor);
        }

        // check if nonempty intersection exists in all dimensions between knot_mins, knot_maxs of two tensors
        // assumes new tensor cannot be larger than existing tensor in any dimension (continually refining smaller or equal)
        bool nonempty_intersection(TensorProduct<T>&    new_tensor,         // new tensor product to be added
                                   TensorProduct<T>&    existing_tensor,    // existing tensor product
                                   vector<int>&         split_side)         // (output) whether min (-1) or max (1) of new_tensor is
                                                                            // inside existing tensor (one value for each dim.) if both, picks max (1)
        {
            split_side.clear();
            split_side.resize(dom_dim_);
            bool retval = false;
            for (int j = 0; j < dom_dim_; j++)
            {
                if (new_tensor.knot_mins[j] > existing_tensor.knot_mins[j] && new_tensor.knot_mins[j] < existing_tensor.knot_maxs[j])
                {
//                     // debug
//                     fprintf(stderr, "cur_dim=%d split_side=-1 new min %lu exist min %lu exist max %lu\n",
//                             j, new_tensor.knot_mins[j], existing_tensor.knot_mins[j], existing_tensor.knot_maxs[j]);

                    split_side[j] = -1;
                    retval = true;
                }
                if (new_tensor.knot_maxs[j] > existing_tensor.knot_mins[j] && new_tensor.knot_maxs[j] < existing_tensor.knot_maxs[j])
                {
                    // debug
//                     fprintf(stderr, "cur_dim=%d split_side=1 new max %lu exist min %lu exist max %lu\n",
//                             j, new_tensor.knot_maxs[j], existing_tensor.knot_mins[j], existing_tensor.knot_maxs[j]);

                    split_side[j] = 1;
                    retval = true;
                }
                // if no intersection found in this dimension, in order to continue checking other dimensions,
                // new_tensor must match exactly or be bigger than existing_tensor. Otherwise, no intersection exists.
                if ( !split_side[j] &&
                     (new_tensor.knot_mins[j] > existing_tensor.knot_mins[j] || new_tensor.knot_maxs[j] < existing_tensor.knot_maxs[j]) )
                    return false;
            }

            return retval;
        }

        // intersect in one dimension a new tensor product with an existing tensor product, if the intersection exists
        // returns true if intersection found (and the vector of tensor products grew as a result of the intersection, ie, an existing tensor was split into two)
        // sets knots_match to true if during the course of intersecting, one of the tensors in tensor_prods was added or modified to match the new tensor
        // ie, the caller should not add the tensor later if knots_match
        bool intersect(TensorProduct<T>&    new_tensor,             // new tensor product to be inserted
                       TensorIdx            existing_tensor_idx,    // index in tensor_prods of existing tensor
                       vector<int>&         split_side,             // whether min (-1) or max (1) or both (2) sides of
                                                                    // new tensor are inside existing tensor (one value for each dim.)
                       bool&                knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            knots_match             = false;
            bool    retval          = false;
            KnotIdx split_knot_idx;

            for (int k = 0; k < dom_dim_; k++)      // for all domain dimensions
            {
                if (!split_side[k])
                    continue;

                split_knot_idx                      = (split_side[k] == -1 ? new_tensor.knot_mins[k] : new_tensor.knot_maxs[k]);
                TensorProduct<T>&   existing_tensor = tensor_prods[existing_tensor_idx];
                vector<KnotIdx>     temp_maxs       = existing_tensor.knot_maxs;
                temp_maxs[k]                        = split_knot_idx;

                // split existing_tensor at the knot index knot_idx as long as doing so would not create
                // a tensor that is a subset of new_tensor being inserted
                // existing_tensor is modified to be the min. side of the previous existing_tensor
                // a new max_side_tensor is appended to be the max. side of existing_tensor
                if (!subset(existing_tensor.knot_mins, temp_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
                {
                    retval |= new_max_side(new_tensor, existing_tensor_idx, k, split_knot_idx, knots_match);

                    // if there is a new tensor, return and start checking again for intersections
                    if (retval)
                        return true;
                }
            }
            return retval;
        }

        // split existing tensor product creating extra tensor on maximum side of current dimension
        // returns true if a an extra tensor product was inserted
        bool new_max_side(TensorProduct<T>&     new_tensor,             // new tensor product that started all this
                          TensorIdx             existing_tensor_idx,    // index in tensor_prods of existing tensor
                          int                   cur_dim,                // current dimension to intersect
                          KnotIdx               knot_idx,               // knot index in current dim of split point
                          bool&                 knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            TensorProduct<T>& existing_tensor  = tensor_prods[existing_tensor_idx];

            // intialize a new max_side_tensor for the maximum side of the existing_tensor
            TensorProduct<T> max_side_tensor;
            max_side_tensor.next.resize(dom_dim_);
            max_side_tensor.prev.resize(dom_dim_);
            max_side_tensor.nctrl_pts.resize(dom_dim_);
            max_side_tensor.knot_mins           = existing_tensor.knot_mins;
            max_side_tensor.knot_maxs           = existing_tensor.knot_maxs;
            max_side_tensor.knot_mins[cur_dim]  = knot_idx;
            max_side_tensor.level               = existing_tensor.level;

            existing_tensor.knot_maxs[cur_dim]  = knot_idx;

            TensorIdx max_side_tensor_idx       = tensor_prods.size();                  // index of new tensor to be added

            // check if tensor will be added before adding a next pointer to it
            if (!subset(max_side_tensor.knot_mins, max_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {
                // adjust next and prev pointers for existing_tensor and max_side_tensor in the current dimension
                for (int i = 0; i < existing_tensor.next[cur_dim].size(); i++)
                {
                    if (adjacent(max_side_tensor, tensor_prods[existing_tensor.next[cur_dim][i]], cur_dim))
                    {
                        max_side_tensor.next[cur_dim].push_back(existing_tensor.next[cur_dim][i]);
                        vector<TensorIdx>& prev = tensor_prods[existing_tensor.next[cur_dim][i]].prev[cur_dim];
                        auto it                 = find(prev.begin(), prev.end(), existing_tensor_idx);
                        assert(it != prev.end());
                        size_t k                = distance(prev.begin(), it);
                        prev[k]                 = max_side_tensor_idx;
                    }
                }

                // connect next and prev pointers of existing and new max side tensors only if
                // the new tensor will not completely separate the two
                if (!occluded(new_tensor, existing_tensor, cur_dim))
                {
                    existing_tensor.next[cur_dim].push_back(max_side_tensor_idx);
                    max_side_tensor.prev[cur_dim].push_back(existing_tensor_idx);
                }

                // adjust next and prev pointers for existing_tensor and max_side_tensor in other dimensions
                for (int j = 0; j < dom_dim_; j++)
                {
                    if (j == cur_dim)
                        continue;

                    // next pointer
                    for (int i = 0; i < existing_tensor.next[j].size(); i++)
                    {
                        // debug
                        int retval = adjacent(max_side_tensor, tensor_prods[existing_tensor.next[j][i]], j);

                        // add new next pointers
                        if (adjacent(max_side_tensor, tensor_prods[existing_tensor.next[j][i]], j))
                        {
                            max_side_tensor.next[j].push_back(existing_tensor.next[j][i]);
                            tensor_prods[existing_tensor.next[j][i]].prev[j].push_back(max_side_tensor_idx);

                            // debug
                            assert(retval == 1);
                        }

                    }

                    // prev pointer
                    for (int i = 0; i < existing_tensor.prev[j].size(); i++)
                    {
                        // debug
                        int retval = adjacent(max_side_tensor, tensor_prods[existing_tensor.prev[j][i]], j);

                        // add new prev pointers
                        if (adjacent(max_side_tensor, tensor_prods[existing_tensor.prev[j][i]], j))
                        {
                            max_side_tensor.prev[j].push_back(existing_tensor.prev[j][i]);
                            tensor_prods[existing_tensor.prev[j][i]].next[j].push_back(max_side_tensor_idx);

                            // debug
                            assert(retval == -1);
                        }

                    }
                }

                // convert global knot_idx to local_knot_idx in existing_tensor
                KnotIdx local_knot_idx = global2local_knot_idx(knot_idx, existing_tensor_idx, cur_dim);

                //  split control points between existing and max side tensors
                //  TODO: hard-coded knot_idx - 1 for split knot idx in current tensor (not global knot idx); not a robust solution

//                 // debug
//                 fprintf(stderr, "1: calling split_ctrl_pts existing_tensor_idx=%lu local_knot_idx=%lu\n", existing_tensor_idx, local_knot_idx);

                split_ctrl_pts(existing_tensor_idx, max_side_tensor, cur_dim, local_knot_idx, false);

                // add the new max side tensor
                tensor_prods.push_back(max_side_tensor);

                // delete next and prev pointers of existing tensor that are no longer valid as a result of adding new max side
                delete_old_pointers(existing_tensor_idx);

                // check if the knot mins, maxs of the existing or added tensor match the original new tensor
                if ( (max_side_tensor.knot_mins == new_tensor.knot_mins && max_side_tensor.knot_maxs == new_tensor.knot_maxs) ||
                     (existing_tensor.knot_mins == new_tensor.knot_mins && existing_tensor.knot_maxs == new_tensor.knot_maxs) )
                    knots_match = true;

                return true;
            }
            else
            {
                // convert global knot_idx to local_knot_idx in existing_tensor
                KnotIdx local_knot_idx = global2local_knot_idx(knot_idx, existing_tensor_idx, cur_dim);

//                 // debug
//                 fprintf(stderr, "2: calling split_ctrl_pts existing_tensor_idx=%lu local_knot_idx=%lu\n", existing_tensor_idx, local_knot_idx);

                split_ctrl_pts(existing_tensor_idx, new_tensor, cur_dim, local_knot_idx, true);
            }

            // delete next and prev pointers of existing tensor that are no longer valid as a result of adding new max side
            delete_old_pointers(existing_tensor_idx);

            return false;
        }

        // convert global knot_idx to local_knot_idx in existing_tensor
        KnotIdx global2local_knot_idx(KnotIdx   knot_idx,
                                      TensorIdx existing_tensor_idx,
                                      int       cur_dim)
        {
            KnotIdx local_knot_idx  = 0;
            int     cur_level       = tensor_prods[existing_tensor_idx].level;
            KnotIdx min_idx         = tensor_prods[existing_tensor_idx].knot_mins[cur_dim];
            KnotIdx max_idx         = tensor_prods[existing_tensor_idx].knot_maxs[cur_dim];

            if (knot_idx < min_idx || knot_idx > max_idx)
            {
                fprintf(stderr, "Error: in global2local_knot_idx, knot_idx is not within min, max knot_idx of existing tensor\n");
                abort();
            }

            for (auto i = min_idx; i < knot_idx; i++)
                if (all_knot_levels[cur_dim][i] <= cur_level)
                    local_knot_idx++;

            return local_knot_idx;
        }

        // split control points between existing and max side tensors
        void split_ctrl_pts(TensorIdx            existing_tensor_idx,    // index in tensor_prods of existing tensor
                            TensorProduct<T>&    max_side_tensor,        // new max side tensor
                            int                  cur_dim,                // current dimension to intersect
                            KnotIdx              split_knot_idx,         // local (not global!) knot index in current dim of split point in existing tensor
                            bool                 skip_max_side)          // don't add control points to max_side tensor, only adjust exsiting tensor control points
        {
            TensorProduct<T>& existing_tensor = tensor_prods[existing_tensor_idx];

            // index of min (in new max side) and max (in existing tensor) control points in current dim
            // allowed to be negative in order for the logic below to partition correctly (long long instead of size_t)
            long long min_ctrl_idx, max_ctrl_idx;

            // convert split_knot_idx to ctrl_pt_idx
            min_ctrl_idx = split_knot_idx;
            if (p_[cur_dim] % 2 == 0)                       // even degree
                max_ctrl_idx = split_knot_idx - 1;
            else                                                // odd degree
                max_ctrl_idx = split_knot_idx;

            // if existing tensor starts at global minimum, the first p-1 knots do not have anchors
            if (existing_tensor.knot_mins[cur_dim] == 0)
            {
                min_ctrl_idx -= (p_[cur_dim] - 1);
                max_ctrl_idx -= (p_[cur_dim] - 1);
            }

            // if max_ctrl_idx is past last existing control point, then split is too close to global edge and must be clamped to last control point
            if (max_ctrl_idx >= existing_tensor.nctrl_pts[cur_dim])
                max_ctrl_idx = existing_tensor.nctrl_pts[cur_dim] - 1;

            // debug
//             fprintf(stderr, "splitting ctrl points in dim %d split_knot_idx=%lu max_ctrl_idx=%lu min_ctrl_idx=%lu\n",
//                     cur_dim, split_knot_idx, max_ctrl_idx, min_ctrl_idx);
//             fprintf(stderr, "old existing tensor tot_nctrl_pts=%lu = [%d %d]\n", existing_tensor.ctrl_pts.rows(), existing_tensor.nctrl_pts[0], existing_tensor.nctrl_pts[1]);

            // allocate new control point matrix for existing tensor
            size_t tot_nctrl_pts = 1;
            VectorXi new_exist_nctrl_pts(dom_dim_);
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (i != cur_dim)
                {
                    new_exist_nctrl_pts(i)  = existing_tensor.nctrl_pts(i);
                    tot_nctrl_pts           *= new_exist_nctrl_pts(i);
                }
                else
                {
                    new_exist_nctrl_pts(i)  = max_ctrl_idx + 1;
                    tot_nctrl_pts           *= new_exist_nctrl_pts(i);
                }
            }
            MatrixX<T> new_exist_ctrl_pts(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
            VectorX<T> new_exist_weights(tot_nctrl_pts);

            // allocate new control point matrix for new max side tensor
            if (!skip_max_side)
            {
                tot_nctrl_pts = 1;
                for (auto i = 0; i < dom_dim_; i++)
                {
                    if (i != cur_dim)
                        max_side_tensor.nctrl_pts(i) = existing_tensor.nctrl_pts(i);
                    else
                        max_side_tensor.nctrl_pts(i) = existing_tensor.nctrl_pts(i) - min_ctrl_idx;
                    tot_nctrl_pts *= max_side_tensor.nctrl_pts(i);
                }
                max_side_tensor.ctrl_pts.resize(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
                max_side_tensor.weights.resize(tot_nctrl_pts);
            }

            // split the control points
            vector<int> dim_idx(dom_dim_);                              // current index in each dim, initialized to 0s
            size_t cur_exist_idx    = 0;                                // current index into new_exist_ctrl_pts and weights
            size_t cur_max_side_idx = 0;                                // current index into max_side_tensor.ctrl_pts and weights
            for (auto j = 0; j < existing_tensor.ctrl_pts.rows(); j++)  // flattened loop over all the points in a domain in dimension dom_dim
            {
                // debug
//                 fprintf(stderr, "dim_idx=[%d %d]\n", dim_idx[0], dim_idx[1]);

                // control point goes either to existing or max side tensor depending on index in current dimension
                if (dim_idx[cur_dim] <= max_ctrl_idx)
                {
                    // debug
//                     fprintf(stderr, "moving to new_exist_ctrl_pts[%lu]\n", cur_exist_idx);

                    new_exist_ctrl_pts.row(cur_exist_idx) = existing_tensor.ctrl_pts.row(j);
                    new_exist_weights(cur_exist_idx) = existing_tensor.weights(j);
                    cur_exist_idx++;
                }
                if (dim_idx[cur_dim] >= min_ctrl_idx)
                {
                    if (!skip_max_side)
                    {
                        // debug
//                         fprintf(stderr, "moving to max_side_tensor.ctrl_pts[%lu]\n", cur_max_side_idx);

                        max_side_tensor.ctrl_pts.row(cur_max_side_idx) = existing_tensor.ctrl_pts.row(j);
                        max_side_tensor.weights(cur_max_side_idx) = existing_tensor.weights(j);
                    }
                    cur_max_side_idx++;
                }

                dim_idx[0]++;

                // for all dimensions except last, check for end of the line, part of flattened loop logic
                for (auto k = 0; k < dom_dim_ - 1; k++)
                {
                    if (dim_idx[k] == existing_tensor.nctrl_pts(k))
                    {
                        dim_idx[k] = 0;
                        dim_idx[k + 1]++;
                    }
                }
            }

            // copy new_exist_ctrl_pts and weights to existing_tensor.ctrl_pts and weights, resizes automatically
            existing_tensor.ctrl_pts    = new_exist_ctrl_pts;
            existing_tensor.weights     = new_exist_weights;
            existing_tensor.nctrl_pts   = new_exist_nctrl_pts;

            // debug
//             fprintf(stderr, "new existing tensor tot_nctrl_pts=%lu = [%d %d]\n", existing_tensor.ctrl_pts.rows(), existing_tensor.nctrl_pts[0], existing_tensor.nctrl_pts[1]);
//             if (!skip_max_side)
//                 fprintf(stderr, "max side tensor tot_nctrl_pts=%lu = [%d %d]\n\n", max_side_tensor.ctrl_pts.rows(), max_side_tensor.nctrl_pts[0], max_side_tensor.nctrl_pts[1]);
        }

        // delete pointers that are no longer valid as a result of adding a new max side tensor
        void delete_old_pointers(TensorIdx existing_tensor_idx)         // index in tensor_prods of existing tensor
        {
            TensorProduct<T>& existing_tensor  = tensor_prods[existing_tensor_idx];

            for (int j = 0; j < dom_dim_; j++)
            {
                // next pointer
                size_t valid_size = existing_tensor.next[j].size();     // size excluding invalid entries at back
                for (int i = 0; i < valid_size; i++)
                {
                    if (!adjacent(existing_tensor, tensor_prods[existing_tensor.next[j][i]], j))
                    {
                        // debug
//                         fprintf(stderr, "next tensor %lu is no longer adjacent to existing_tensor %lu\n",
//                                 existing_tensor.next[j][i], existing_tensor_idx);

                        // remove the prev pointer of the next tensor
                        vector<TensorIdx>& prev = tensor_prods[existing_tensor.next[j][i]].prev[j];
                        auto it                 = find(prev.begin(), prev.end(), existing_tensor_idx);
                        if (it != prev.end())                       // it's possible the pointer was removed earlier and won't be found
                        {
                            size_t k                = distance(prev.begin(), it);
                            prev[k] = prev.back();
                            prev.resize(prev.size() - 1);
                        }

                        // remove the next pointer of the existing tensor
                        existing_tensor.next[j][i] = existing_tensor.next[j][valid_size - 1];
                        valid_size--;
                    }
                }
                existing_tensor.next[j].resize(valid_size);         // drop the invalid entries at back

                // prev pointer
                valid_size = existing_tensor.prev[j].size();        // size excluding invalid entries at back
                for (int i = 0; i < valid_size; i++)
                {
                    if (!adjacent(existing_tensor, tensor_prods[existing_tensor.prev[j][i]], j))
                    {
                        // debug
//                         fprintf(stderr, "prev tensor %lu is no longer adjacent to existing_tensor %lu\n",
//                                 existing_tensor.prev[j][i], existing_tensor_idx);

                        // remove the next pointer of the prev tensor
                        vector<TensorIdx>& next = tensor_prods[existing_tensor.prev[j][i]].next[j];
                        auto it                 = find(next.begin(), next.end(), existing_tensor_idx);
                        if (it != next.end())                       // it's possible the pointer was removed earlier and won't be found
                        {
                            size_t k                = distance(next.begin(), it);
                            next[k] = next.back();
                            next.resize(next.size() - 1);
                        }

                        // remove the prev pointer of the existing tensor
                        existing_tensor.prev[j][i] = existing_tensor.prev[j][valid_size - 1];
                        valid_size--;
                    }
                }
                existing_tensor.prev[j].resize(valid_size);         // drop the invalid entries at back
            }
        }

        // check if new_tensor is adjacent to existing_tensor in current dimension
        // returns -1: existing_tensor is adjacent on min side of new_tensor in current dim.
        //          0: not adjacent
        //          1: existing_tensor is adjacent on max side of new_tensor in current dim.
        int adjacent(TensorProduct<T>&      new_tensor,       // new tensor product to be added
                     TensorProduct<T>&      existing_tensor,  // existing tensor product
                     int                    cur_dim)          // current dimension
        {
            int retval = 0;

            // check if adjacency exists in current dim
            if (new_tensor.knot_mins[cur_dim] == existing_tensor.knot_maxs[cur_dim])
                retval = -1;
            else if (new_tensor.knot_maxs[cur_dim] == existing_tensor.knot_mins[cur_dim])
                retval = 1;
            else
            {
                // debug
//                 fprintf(stderr, "adj 1: cur_dim=%d retval=0 new_tensor=[%lu %lu : %lu %lu] existing_tensor=[%lu %lu : %lu %lu]\n",
//                         cur_dim, new_tensor.knot_mins[0], new_tensor.knot_mins[1], new_tensor.knot_maxs[0], new_tensor.knot_maxs[1],
//                         existing_tensor.knot_mins[0], existing_tensor.knot_mins[1], existing_tensor.knot_maxs[0], existing_tensor.knot_maxs[1]);

                return 0;
            }

            // confirm that intersection in at least one other dimension exists
            for (int j = 0; j < dom_dim_; j++)
            {
                if (j == cur_dim)
                    continue;

                // the area touching is zero in some dimension
                // two cases are checked because role of new and existing tensor can be interchanged for adjacency; both need to fail to be nonadjacent
                if ( (new_tensor.knot_mins[j]      < existing_tensor.knot_mins[j] || new_tensor.knot_mins[j]      >= existing_tensor.knot_maxs[j]) &&
                     (existing_tensor.knot_mins[j] < new_tensor.knot_mins[j]      || existing_tensor.knot_mins[j] >= new_tensor.knot_maxs[j]))
                {

                    // debug
//                     fprintf(stderr, "adj 2: cur_dim=%d retval=0 new_tensor=[%lu %lu : %lu %lu] existing_tensor=[%lu %lu : %lu %lu]\n",
//                             cur_dim, new_tensor.knot_mins[0], new_tensor.knot_mins[1], new_tensor.knot_maxs[0], new_tensor.knot_maxs[1],
//                             existing_tensor.knot_mins[0], existing_tensor.knot_mins[1], existing_tensor.knot_maxs[0], existing_tensor.knot_maxs[1]);

                    return 0;
                }
            }

            // debug
//             fprintf(stderr, "adj: cur_dim=%d retval=%d new_tensor=[%lu %lu : %lu %lu] existing_tensor=[%lu %lu : %lu %lu]\n",
//                     cur_dim, retval, new_tensor.knot_mins[0], new_tensor.knot_mins[1], new_tensor.knot_maxs[0], new_tensor.knot_maxs[1],
//                     existing_tensor.knot_mins[0], existing_tensor.knot_mins[1], existing_tensor.knot_maxs[0], existing_tensor.knot_maxs[1]);

            return retval;
        }

        // check if new_tensor completely occludes any neighbor of existing_tensor
        // ie, returns true if the face they share is the full size of existing_tensor
        // assumes they share a face in the cur_dim (does not check)
        int occluded(TensorProduct<T>&      new_tensor,       // new tensor product to be added
                     TensorProduct<T>&      existing_tensor,  // existing tensor product
                     int                    cur_dim)          // current dimension
        {
            // confirm that new_tensor is larger than existing_tensor in every dimension except cur_dim
            for (int j = 0; j < dom_dim_; j++)
            {
                if (j == cur_dim)
                    continue;

                if(new_tensor.knot_mins[j] > existing_tensor.knot_mins[j] ||
                   new_tensor.knot_maxs[j] < existing_tensor.knot_maxs[j])
                    return false;
            }

            // debug
//             fprintf(stderr, "cur_dim=%d return=true new_tensor=[%lu %lu : %lu %lu] existing_tensor=[%lu %lu : %lu %lu]\n",
//                     cur_dim, new_tensor.knot_mins[0], new_tensor.knot_mins[1], new_tensor.knot_maxs[0], new_tensor.knot_maxs[1],
//                     existing_tensor.knot_mins[0], existing_tensor.kot_mins[1], existing_tensor.knot_maxs[0], existing_tensor.knot_maxs[1]);

            return true;
        }

        // checks if a_mins, maxs are a subset of b_mins, maxs
        // identical bounds counts as a subset (does not need to be proper subset)
        bool subset(const vector<KnotIdx>& a_mins,
                    const vector<KnotIdx>& a_maxs,
                    const vector<KnotIdx>& b_mins,
                    const vector<KnotIdx>& b_maxs)
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

            // debug
//             fprintf(stderr, "[%lu %lu : %lu %lu] is a subset of [%lu %lu : %lu %lu]\n",
//                     a_mins[0], a_mins[1], a_maxs[0], a_maxs[1], b_mins[0], b_mins[1], b_maxs[0], b_maxs[1]);

            return true;
        }

        // in
        // checks if a point in index space is in a tensor product
        // in all dimensions except skip_dim (-1 = default, don't skip any dimensions)
        // for dimension i, if degree[i] is even, pt[i] + 0.5 is checked because pt coords are truncated to integers
        bool in(const vector<KnotIdx>&  pt,
                const TensorProduct<T>& tensor,
                int                     skip_dim = -1) const
        {
            for (auto i = 0; i < pt.size(); i++)
            {
                if (i == skip_dim)
                    continue;

                // move pt[i] to center of t-mesh cell if degree is even
                float fp = p_[i] % 2 ? pt[i] : pt[i] + 0.5;

                if (fp < float(tensor.knot_mins[i]) || fp > float(tensor.knot_maxs[i]))
                        return false;
            }

            return true;
        }

        // given a center point in index space, find intersecting knot lines in index space
        // in -/+ directions in all dimensions
        void knot_intersections(const vector<KnotIdx>&      center,                 // knot indices of anchor for odd degree or
                                                                                    // knot indices of start of rectangle containing anchor for even degree
                                const VectorXi&             p,                      // local degree in each dimension
                                vector<vector<KnotIdx>>&    loc_knots,              // (output) local knot vector in index space
                                vector<NeighborTensor>&     neigh_hi_levels) const  // (output) intersected neighbor tensors of higher level than center
        {
            loc_knots.resize(dom_dim_);
            assert(center.size() == dom_dim_);

            // find most refined tensor product containing center and that level of refinement
            // levels monotonically nondecrease; hence, find last tensor product containing the anchor
            TensorIdx max_j = 0;
            for (auto j = 0; j < tensor_prods.size(); j++)
                if (in(center, tensor_prods[j], -1))
                        max_j = j;
            int max_level = tensor_prods[max_j].level;

            // debug
            if (center[0] == 2 && center[1] == 2 && max_j == 2)
            {
                fprintf(stderr, "center = 2,2 and max_j = 2\n");
                in(center, tensor_prods[2], -1);
            }

            // walk the t-mesh in all dimensions, min. and max. directions outward from the center
            // looking for interecting knot lines

            for (auto i = 0; i < dom_dim_; i++)                             // for all dims
            {
                loc_knots[i].resize(p(i) + 2);                              // support of basis func. is p+2 by definition

                KnotIdx start, min, max;
                if (p(i) % 2)                                               // odd degree
                    start = min = max = (p(i) + 1) / 2;
                else                                                        // even degree
                {
                    start = min = p(i) / 2;
                    max         = p(i) / 2 + 1;
                }
                loc_knots[i][start]             = center[i];                // start by center the center of the knot vector
                KnotIdx         cur_knot_idx    = loc_knots[i][start];
                TensorIdx       cur_tensor      = max_j;
                int             cur_level       = max_level;
                vector<KnotIdx> cur             = center;                   // current knot location in the tmesh (index space)

                // from the center in the min. direction
                for (int j = 0; j < min; j++)                               // add 'min' more knots in minimum direction from the center
                {
                    bool done = false;
                    do
                    {
                        if (cur_knot_idx > 0)                               // more knots in the tmesh
                        {
                            cur[i] = cur_knot_idx - 1;

                            // check which is the correct previous tensor
                            // if more than one tensor sharing the target, pick highest level
                            if (cur_knot_idx - 1 < tensor_prods[cur_tensor].knot_mins[i])
                            {
                                // debug
                                if (tensor_prods[cur_tensor].prev[i].size() == 0)
                                    fprintf(stderr, "Error: prev is empty. This should not happen.\n");
                                neighbor_tensors(tensor_prods[cur_tensor].prev[i], i, cur, cur_tensor, cur_level, neigh_hi_levels);
                            }

                            // check if next knot borders a higher level; if so, switch to higher level tensor
                            border_higher_level(cur, cur_tensor, cur_level);

                            // move to next knot
                            if (all_knot_levels[i][cur_knot_idx - 1] > cur_level)
                                cur_knot_idx--;
                            if (cur_knot_idx > 0                                            &&
                                cur_knot_idx - 1 >= tensor_prods[cur_tensor].knot_mins[i]   &&
                                all_knot_levels[i][cur_knot_idx - 1] <= cur_level)
                            {
                                loc_knots[i][start - j - 1] = --cur_knot_idx;
                                done = true;
                            }
                        }
                        else                                                // no more knots in the tmesh
                        {
                            loc_knots[i][start - j - 1] = cur_knot_idx;     // repeat last knot as many times as needed
                            done = true;
                        }
                    } while (!done);
                }

                // reset back to center
                cur_knot_idx    = loc_knots[i][start];
                cur_tensor      = max_j;
                cur_level       = max_level;
                cur             = center;

                // from the center in the max. direction
                for (int j = 0; j < max; j++)                               // add 'max' more knots in maximum direction from the center
                {
                    bool done = false;
                    do
                    {
                        if (cur_knot_idx + 1 < all_knots[i].size())         // more knots in the tmesh
                        {
                            cur[i] = cur_knot_idx + 1;

                            // check which is the correct previous tensor
                            // if more than one tensor sharing the target, pick highest level
                            if (cur_knot_idx + 1 > tensor_prods[cur_tensor].knot_maxs[i])
                            {
                                // debug
                                if (tensor_prods[cur_tensor].next[i].size() == 0)
                                    fprintf(stderr, "Error: next is empty. This should not happen.\n");
                                neighbor_tensors(tensor_prods[cur_tensor].next[i], i, cur, cur_tensor, cur_level, neigh_hi_levels);
                            }

                            // check if next knot borders a higher level; if so, switch to higher level tensor
                            border_higher_level(cur, cur_tensor, cur_level);

                            // move to next knot
                            if (all_knot_levels[i][cur_knot_idx + 1] > cur_level)
                                cur_knot_idx++;
                            if (cur_knot_idx + 1 < all_knots[i].size()                      &&
                                cur_knot_idx + 1 <= tensor_prods[cur_tensor].knot_maxs[i]   &&
                                all_knot_levels[i][cur_knot_idx + 1] <= cur_level)
                            {
                                loc_knots[i][start + j + 1] = ++cur_knot_idx;
                                done = true;
                            }
                        }
                        else                                                // no more knots in the tmesh
                        {
                            loc_knots[i][start + j + 1] = cur_knot_idx;     // repeat last knot as many times as needed
                            done = true;
                        }
                    } while (!done);
                }
            }                                                               // for all dims.
        }

        // check which is the correct previous or next neighbor tensor containing the target in the current dimension
        // if more than one tensor sharing the target, pick highest level
        // updates cur_tensor and cur_level and neigh_hi_levels
        void neighbor_tensors(const vector<TensorIdx>&  prev_next,              // previous or next neighbor tensors
                              int                       cur_dim,                // current dimension
                              const vector<KnotIdx>&    target,                 // target knot indices
                              TensorIdx&                cur_tensor,             // (input / output) highest level neighbor tensor containing the target
                              int&                      cur_level,              // (input / output) level of current tensor
                              vector<NeighborTensor>&   neigh_hi_levels) const  // (input / output) neighbors with higher levels than current tensor
        {
            int         temp_max_level;
            TensorIdx   temp_max_k;
            bool    first_time = true;
            for (auto k = 0; k < prev_next.size(); k++)
            {
                if (in(target, tensor_prods[prev_next[k]]))
                {
                    if (first_time)
                    {
                        temp_max_k      = k;
                        temp_max_level  = tensor_prods[prev_next[k]].level;
                        first_time      = false;
                    }
                    else if (tensor_prods[prev_next[k]].level > temp_max_level)
                    {
                        temp_max_level  = tensor_prods[prev_next[k]].level;
                        temp_max_k      = k;
                    }
                }
            }
            if (first_time)
            {
                fprintf(stderr, "Error: no valid previous tensor for knot vector\n");
                abort();
            }
            cur_tensor = prev_next[temp_max_k];
            if (tensor_prods[cur_tensor].level > cur_level)
            {
                NeighborTensor neighbor = {cur_dim, tensor_prods[cur_tensor].level, prev_next[temp_max_k]};
                neigh_hi_levels.push_back(neighbor);
            }
            cur_level = tensor_prods[cur_tensor].level;
        }

        // check if target knot index borders a higher level; if so, switch to higher level tensor
        void border_higher_level(const vector<KnotIdx>& target,             // target knot indices
                                 TensorIdx&             cur_tensor,         // (input / output) highest level neighbor tensor containing the target
                                 int&                   cur_level) const    // (input / output) level of current tensor
        {
            for (auto k = cur_tensor; k < tensor_prods.size(); k++) // start checking at current tensor because levels are monotonic nondecreasing
            {
                if (in(target, tensor_prods[k], -1) && tensor_prods[k].level > cur_level)
                {
                    cur_tensor  = k;
                    cur_level   = tensor_prods[k].level;
                }
            }
        }

        // given an anchor point in index space, compute local knot vector in all dimensions in index space
        // in Bazilevs 2010, knot indices start at 1, but mine start counting at 0
        void local_knot_vector(const vector<KnotIdx>&       center,                 // knot indices of center of anchors
                               vector<vector<KnotIdx>>&     loc_knot_idxs) const    // (output) local knot vector in index space
        {
            vector<NeighborTensor> unused;
            knot_intersections(center, p_, loc_knot_idxs, unused);
        }

        // given a point in parameter space to decode, compute range of anchor points in index space
        // in Bazilevs 2010, knot indices start at 1, but mine start counting at 0
        // also, Bazilevs anchors start at p - 1 (assuming knot indices start at 0), but my anchors start at 0
        // add (p - 1) to my anchors to get the equivalent of Bazilevs with 0-based knot indices
        void anchors(const VectorX<T>&          param,              // parameter value in each dim. of desired point
                     vector<vector<KnotIdx>>&   anchors) const      // (output) anchor points in index space
        {
            anchors.resize(dom_dim_);
            vector<vector<KnotIdx>> anchor_cands(dom_dim_);         // anchor candidates (possibly more than necessary)

            // convert param to target in index space
            vector<KnotIdx> target(dom_dim_);                       // center anchor in each dim.
            for (auto i = 0; i < dom_dim_; i++)
            {
                    auto it     = upper_bound(all_knots[i].begin() + p_(i), all_knots[i].end() - p_(i) - 1, param(i));
                    target[i]   = it - all_knots[i].begin() - p_(i) - 1;

                    // fill anchors by extending a ray to the max side from the target and intersecting knots
                    // TODO: skip knots not intersected by the ray
                    // ie, write a version of knot_intersections for anchors (don't use knot_intersections for local knot vectors)
                    anchors[i].resize(p_(i) + 1);
                    for (auto j = 0; j < p_(i) + 1; j++)
                        anchors[i][j] = target[i] + j;
            }

            // debug
//             fprintf(stderr, "param=[%.2lf] target=[%lu]\n", param(0), target[0]);

            return;

            // TODO: below is not tested

            vector<NeighborTensor> neigh_hi_levels;                 // neighbor tensors of a higher level than tensor containing target

            // subtract 1 from degree so that same code as for local knot vectors can be reused
            // support for a decoded point is p+1 basis functions, while width of a local knot vector is p+2
            // TODO: this is not right. Don't reuse knot_intersections for local knot vectors; write a new knot_intersections for anchors
//             knot_intersections(target, p_ - VectorXi::Ones(dom_dim_), anchor_cands, neigh_hi_levels);

            if (neigh_hi_levels.size())
            {
                // debug
                fprintf(stderr, "%lu neigh_hi_levels:\n", neigh_hi_levels.size());

                for (auto i = 0; i < neigh_hi_levels.size(); i++)
                {
                    // debug
                    fprintf(stderr, "dim=%d level=%d tensor_idx=%lu\n",
                            neigh_hi_levels[i].dim, neigh_hi_levels[i].level, neigh_hi_levels[i].tensor_idx);

                    // knot intersections at center of neigh_hi_level
                    vector<vector<KnotIdx>> temp_anchors(dom_dim_);
                    vector<NeighborTensor>  unused;
                    vector<KnotIdx>         temp_target(dom_dim_);
                    for (auto j = 0; j < dom_dim_; j++)
                        temp_target[j] = (tensor_prods[neigh_hi_levels[i].tensor_idx].knot_mins[j] +
                                tensor_prods[neigh_hi_levels[i].tensor_idx].knot_maxs[j]) / 2;
                    knot_intersections(temp_target, p_ - VectorXi::Ones(dom_dim_), temp_anchors, unused);

                    // merge knot_intersections of neigh_hi_level with anchor_cands keeping sorted order, growing size of anchor_cands
                    for (auto j = 0; j < dom_dim_; j++)
                    {
                        if (j == neigh_hi_levels[i].dim)
                            continue;

                        // anchor_cands[j] = set union of temp_anchors[j] and anchor_cands[j], maintaining sorted order
                        vector<KnotIdx> union_anchors;
                        set_union(temp_anchors[j].begin(), temp_anchors[j].end(), anchor_cands[j].begin(), anchor_cands[j].end(), back_inserter(union_anchors));
                        anchor_cands[j] = union_anchors;
                    }
                }

                // copy central p+1 anchors from anchor_cands into anchors
                for (auto i = 0; i < dom_dim_; i++)
                {
                    auto it = find(anchor_cands[i].begin(), anchor_cands[i].end(), target[i]);
                    if (it == anchor_cands[i].end())
                    {
                        fprintf(stderr, "Error: target %lu not found in anchor_cands[dim %d]\n", target[i], i);
                        abort();
                    }
                    KnotIdx target_loc = it - anchor_cands[i].begin();           // index of target[i] in anchor_cands[i]
                    KnotIdx start, min, max;                                     // starting index and number of items to copy in min and max direction
                    if (p_(i) % 2)
                    {
                        start = min = p_(i) / 2;
                        max = p_(i) / 2 + 1;
                    }
                    else
                        start = min = max = p_(i) / 2;

                    anchors[i].resize(p_(i) + 1);
                    anchors[i][start] = anchor_cands[i][target_loc];

                    for (int j = 0; j < min; j++)                               // add 'min' more knots in minimum direction from the center
                        anchors[i][start - j - 1] = anchor_cands[i][target_loc - j - 1];
                    for (int j = 0; j < max; j++)                               // add 'max' more knots in maximum direction from the center
                        anchors[i][start + j + 1] = anchor_cands[i][target_loc + j + 1];
                }
            }
            else
            {
                // copy anchor_cands to anchors
                for (auto i = 0; i < dom_dim_; i++)
                    anchors[i] = anchor_cands[i];
            }
        }

        // extract linearized box (tensor) of (p + 1)^d control points and weights at the given anchors
        void ctrl_pt_box(const vector<vector<KnotIdx>>& anchors,            // anchors, starting at 0 rather than p - 1
                         MatrixX<T>&                    ctrl_pts,           // (output) linearized control points
                         VectorX<T>&                    weights) const      // (output) linearized weights
        {
            vector<int> iter(dom_dim_);                                     // iteration number in each dim.

            int tensor_idx = -1;                                            // index of current tensor
            vector<int> start_tensor_idx(dom_dim_, -1);                     // index of starting tensor when dimension changes

            vector<KnotIdx> anchor(dom_dim_);                               // current anchor in all dims
            for (auto j = 0; j < dom_dim_; j++)
                anchor[j] = anchors[j][iter[j]] + p_(j) - 1;                // add p - 1 to anchor to align with indices of all_knots

            // naive linear search for tensor containing first anchor TODO: tree search
            for (auto j = 0; j < tensor_prods.size(); j++)
            {
                if (in(anchor, tensor_prods[j]))
                {
                    tensor_idx = j;
                    for (auto k = 0; k < dom_dim_; k++)
                        start_tensor_idx[k] = j;
                    break;
                }
            }
            if (tensor_idx == -1)
            {
                fprintf(stderr, "Error: anchor not found in tmesh.\n");
                abort();
            }

            // 1-d flattening of the iterations in the box
            for (int i = 0; i < ctrl_pts.rows(); i++)                       // total number of iterations in the box
            {
                // debug: print anchor coords
//                 fprintf(stderr, "iter [ ");
//                 for (auto j = 0; j < dom_dim_; j++)
//                     fprintf(stderr, "%d ", iter[j]);
//                 fprintf(stderr, "]\n");
//
                // debug
//                 fprintf(stderr, "ctrl_pt_box() 1: anchor [ ");
//                 for (auto j = 0; j < dom_dim_; j++)
//                     fprintf(stderr, "%lu ", anchor[j]);
//                 fprintf(stderr, "]\n");

                // locate the control point in the current tensor
                size_t ctrl_idx;                                            // index of desired control point
                VectorXi ijk(dom_dim_);                                     // ijk of desired control point
                for (auto j = 0; j < dom_dim_; j++)
                {
                    // sanity check: anchor values should correspond to knots in this level
                    assert(all_knot_levels[j][anchor[j]] <= tensor_prods[tensor_idx].level);

                    // ijk is wrt to control points in current tensor
                    // ie, subtract (p - 1) that was added into the anchor earlier (so that anchor 0 = ctrl pt 0)
                    ijk(j) = anchor[j] - tensor_prods[tensor_idx].knot_mins[j] - (p_(j) - 1);
                    // skip over knots in finer levels
                    for (auto k = tensor_prods[tensor_idx].knot_mins[j]; k < anchor[j]; k++)
                        if (all_knot_levels[j][k] > tensor_prods[tensor_idx].level)
                            ijk(j)--;
                    assert(ijk(j) >= 0);
                }
                ijk2idx(tensor_prods[tensor_idx].nctrl_pts, ijk, ctrl_idx);

                // debug
//                 if (ctrl_idx >= tensor_prods[tensor_idx].ctrl_pts.rows())
//                     fprintf(stderr, "ctrl_idx %ld is out of bounds, only %ld ctrl points\n", ctrl_idx, tensor_prods[tensor_idx].ctrl_pts.rows());

                // copy the control point
                ctrl_pts.row(i) = tensor_prods[tensor_idx].ctrl_pts.row(ctrl_idx);

                // prepare for next iteration by increasing iteration count in first dimension and checking
                // if iterations in dimensions exceed the size in each dimension, cascading dimensions as necessary

                iter[0]++;

                // for all dimensions, check for last point
                bool reset_iter = false;                                    // reset iteration in some dimension
                for (size_t k = 0; k < dom_dim_; k++)
                {
                    if (k < dom_dim_ - 1 && iter[k] - 1 == p_(k))           // reset iteration for current dim and increment next dim.
                    {
                        reset_iter = true;
                        iter[k] = 0;
                        iter[k + 1]++;
                        for (auto j = 0; j < dom_dim_; j++)
                            anchor[j] = anchors[j][iter[j]] + p_(j) - 1;    // add p - 1 to anchor to align with indices of all_knots

                        // debug
//                         fprintf(stderr, "ctrl_pt_box() 2: anchor [ ");
//                         for (auto j = 0; j < dom_dim_; j++)
//                             fprintf(stderr, "%lu ", anchor[j]);
//                         fprintf(stderr, "]\n");

                        // check for the anchor in the current tensor and in next pointers in next higher dim, starting back at last tensor of current dim
                        tensor_idx = in_and_next(anchor, start_tensor_idx[k + 1], k + 1);
                        if (tensor_idx >= 0)
                        {
                            // TODO: following is untested, need higher dimension example with multiple tensors
                            start_tensor_idx[k + 1] = tensor_idx;           // adjust start tensor of next dim
                            start_tensor_idx[k]     = start_tensor_idx[0];  // reset start tensor of current dim
                        }
                    }
                }

                // normal next iteration in 0th dimension
                if (!reset_iter && iter[0] < p_(0) + 1)                     // check for the anchor in the current tensor and in next pointers in current dim
                {
                    anchor[0] = anchors[0][iter[0]] + p_(0) - 1;            // add p - 1 to anchor to align with indices of all_knots

                    // debug
//                     fprintf(stderr, "ctrl_pt_box() 3: anchor [ ");
//                     for (auto j = 0; j < dom_dim_; j++)
//                         fprintf(stderr, "%lu ", anchor[j]);
//                     fprintf(stderr, "]\n");

                    // check for the anchor in the current tensor and in next pointers in current dim
                    tensor_idx = in_and_next(anchor, tensor_idx, 0);
                }

                assert(tensor_idx >= 0);                                    // sanity: anchor was found in some tensor

                // debug
                fprintf(stderr, "tensor_idx = %d\n", tensor_idx);
            }                                                               // total number of flattened iterations
        }

        // check tensor and next pointers of tensor looking for a tensor containing the point
        // only checks the direct next neighbors, not multiple hops
        // returns index of tensor containing the point, or -1 if not found
        int in_and_next(const vector<KnotIdx>&  pt,                     // target point in index space
                        int                     tensor_idx,             // index of starting tensor for the walk
                        int                     cur_dim) const          // dimension in which to walk
        {
            const TensorProduct<T>& tensor = tensor_prods[tensor_idx];

            // check current tensor
            if (in(pt, tensor))
                return tensor_idx;

            // check nearest neighbor next tensors
            for (auto i = 0; i < tensor.next[cur_dim].size(); ++i)
            {
                if (in(pt, tensor_prods[tensor.next[cur_dim][i]]))
                    return tensor.next[cur_dim][i];
            }
            return -1;
        }

        // convert (i,j,k,...) multidimensional index into linear index into domain
        // number of dimension is the domain dimensionality
        void ijk2idx(
                const VectorXi& ndom_pts,               // number of input points in each dimension
                const VectorXi& ijk,                    // i,j,k,... indices to all dimensions
                size_t&         idx) const              // (output) linear index
        {
            idx           = 0;
            size_t stride = 1;
            for (int i = 0; i < dom_dim_; i++)
            {
                idx += ijk(i) * stride;
                stride *= ndom_pts(i);
            }
        }

        void print() const
        {
            // all_knots
            for (int i = 0; i < dom_dim_; i++)
            {
                fprintf(stderr, "all_knots[dim %d] ", i);
                for (auto j = 0; j < all_knots[i].size(); j++)
                    fprintf(stderr, "%.2lf (l%d) ", all_knots[i][j], all_knot_levels[i][j]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");

            fprintf(stderr, "T-mesh has %lu tensor products\n\n", tensor_prods.size());

            // tensor products
            for (auto j = 0; j < tensor_prods.size(); j++)
            {
                const TensorProduct<T>& t = tensor_prods[j];
                fprintf(stderr, "tensor_prods[%d] level=%d\n", j, t.level);

                fprintf(stderr, "knots [ ");
                for (int i = 0; i < dom_dim_; i++)
                    fprintf(stderr,"%lu ", t.knot_mins[i]);
                fprintf(stderr, "] : ");

                fprintf(stderr, "[ ");
                for (int i = 0; i < dom_dim_; i++)
                    fprintf(stderr,"%lu ", t.knot_maxs[i]);
                fprintf(stderr, "]\n");

                fprintf(stderr, "nctrl_pts [ ");
                for (int i = 0; i < dom_dim_; i++)
                    fprintf(stderr,"%d ", t.nctrl_pts[i]);
                fprintf(stderr, "]\n");

                fprintf(stderr, "next tensors [ ");
                for (int i = 0; i < dom_dim_; i++)
                {
                    fprintf(stderr, "[ ");
                    for (const TensorIdx& n : t.next[i])
                        fprintf(stderr, "%lu ", n);
                    fprintf(stderr, "] ");
                    fprintf(stderr," ");
                }
                fprintf(stderr, "]\n");

                fprintf(stderr, "previous tensors [ ");
                for (int i = 0; i < dom_dim_; i++)
                {
                    fprintf(stderr, "[ ");
                    for (const TensorIdx& n : t.prev[i])
                        fprintf(stderr, "%lu ", n);
                    fprintf(stderr, "] ");
                    fprintf(stderr," ");
                }

                fprintf(stderr, "]\n\n");
            }
            fprintf(stderr, "\n");
        }

    };
}

#endif
