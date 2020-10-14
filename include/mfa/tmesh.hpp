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
using ParamIdx  = size_t;

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
        vector<vector<ParamIdx>>    all_knot_param_idxs;// span in input point parameters containing knot value in all_knots[dimension][idx] (same layout as all_knots)
                                                        // params[dim][idx] <= knot_value < params[dim][idx + 1]
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
            all_knot_param_idxs.resize(dom_dim_);

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
                all_knot_param_idxs[i].resize(nctrl_pts(i) + p_(i) + 1);
            }
        }

        // insert a knot into all_knots
        void insert_knot(int                dim,            // current dimension
                         KnotIdx            pos,            // new position in all_knots[dim] of inserted knot
                         int                level,          // refinement level of inserted knot
                         T                  knot,           // knot value to be inserted
                         vector<vector<T>>& params)         // params of input points
        {
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
            if (param_idx == params[dim].size() - 1 || knot < params[dim][param_idx])
                param_idx--;
            all_knot_param_idxs[dim].insert(all_knot_param_idxs[dim].begin() + pos, param_idx);

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
                new_tensor.weights.resize(tot_nctrl_pts);               // will get initialized to 1 later
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
                new_tensor.weights.resize(tot_nctrl_pts);               // will get initialized to 1 later
            }

            vector<int> split_side(dom_dim_);       // whether min (-1), max (1), or both (2) sides of
                                                    // new tensor split the existing tensor (one value for each dim.)

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
//                             fprintf(stderr, "\ntensors after intersection\n\n");
//                             print();

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

            // initialize the new tensor weights to 1
            // done here, relatively late, because it's possible for it to have changed, eg, set to MFA_NAW earlier
            // this is how we ensure the new tensor has all valid weights
            new_tensor.weights = VectorX<T>::Ones(new_tensor.weights.size());

            // add the tensor
            if (!tensor_inserted)
                tensor_prods.push_back(new_tensor);
        }

        // Append a tensor product to the back of tensor_prods and copy control points and weights into it
        // This version takes a (possibly larger) set of control points and weights to copy into the appended tensor
        // If the input control points and weights are a superset of the tensor, the correct subset of them will be used
        // The number of control points is the number of the input superset
        void append_tensor(const vector<KnotIdx>&   knot_mins,      // indices in all_knots of min. corner of tensor to be inserted
                           const vector<KnotIdx>&   knot_maxs,      // indices in all_knots of max. corner
                           const VectorXi&          new_nctrl_pts,  // number of control points in each dim. for this tensor (possibly superset)
                           const MatrixX<T>&        new_ctrl_pts,   // local control points for this tensor (possibly superset)
                           const VectorX<T>&        new_weights)    // local weights for this tensor (possibly superset)
        {
            // debug
            fprintf(stderr, "*** append_tensor with provided control points ***\n");
            cerr << "new_nctrl_pts: " << new_nctrl_pts.transpose() << endl;

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
            if (!tensor_prods.size())
                new_tensor.level = 0;
            else
                new_tensor.level = tensor_prods.back().level + 1;

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
//                         fprintf(stderr, "intersection found between new tensor and existing tensor idx=%d split_side=[%d %d]\n",
//                                 j, split_side[0], split_side[1]);
//                         fprintf(stderr, "\ntensors before intersection\n\n");
//                         print();

                        if ((vec_grew = intersect(new_tensor, j, split_side, knots_match)) && vec_grew)
                        {
                            if (knots_match)
                                tensor_inserted = true;

                            // debug
//                             fprintf(stderr, "\ntensors after intersection\n\n");
//                             print();

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

            // copy the control points
            // TODO: deal with the case that the tensor was already inserted, check if it's possible to not be at the end
            // following assumes the appended tensor is last
            int tensor_idx = tensor_prods.size() - 1;
            subset_ctrl_pts(new_nctrl_pts, new_ctrl_pts, new_weights, tensor_idx);

        }

        // check if nonempty intersection exists in all dimensions between knot_mins, knot_maxs of two tensors
        // assumes new tensor cannot be larger than existing tensor in any dimension (continually refining smaller or equal)
        bool nonempty_intersection(TensorProduct<T>&    new_tensor,         // new tensor product to be added
                                   TensorProduct<T>&    existing_tensor,    // existing tensor product
                                   vector<int>&         split_side)         // (output) whether min (-1), max (1), or both sides of new_tensor split
                                                                            // existing tensor (one value for each dim.)
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

                    if (!split_side[j])
                        split_side[j] = 1;
                    else
                        split_side[j] = 2;
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

        // intersect a new tensor product with an existing tensor product, if the intersection exists
        // returns true if intersection found (and the vector of tensor products grew as a result of the intersection, ie, an existing tensor was split into two)
        // sets knots_match to true if during the course of intersecting, one of the tensors in tensor_prods was added or modified to match the new tensor
        // ie, the caller should not add the tensor later if knots_match
        bool intersect(TensorProduct<T>&    new_tensor,             // new tensor product to be inserted
                       TensorIdx            existing_tensor_idx,    // index in tensor_prods of existing tensor
                       vector<int>&         split_side,             // whether min (-1), max (1), or both sides (2) of new tensor
                                                                    // split the existing tensor (one value for each dim.)
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
                vector<KnotIdx>     temp_mins       = existing_tensor.knot_mins;
                if (split_side[k] == -1)
                    temp_maxs[k] = split_knot_idx;
                else
                    temp_mins[k] = split_knot_idx;

                // split existing_tensor at the knot index knot_idx as long as doing so would not create
                // a tensor that is a subset of new_tensor being inserted
                // existing_tensor is modified to be the min. side of the previous existing_tensor
                // a new max_side_tensor is appended to be the max. side of existing_tensor
                if (!subset(temp_mins, temp_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
                {
                    // if there is a new tensor, return and start checking again for intersections
                    retval |= new_side(new_tensor, existing_tensor_idx, k, split_knot_idx, split_side[k], knots_match);
                    if (retval)
                        return true;
                }
            }
            return retval;
        }

        // split existing tensor product creating extra tensor on minimum or maximum side of current dimension
        // returns true if a an extra tensor product was inserted
        bool new_side(TensorProduct<T>&     new_tensor,             // new tensor product that started all this
                      TensorIdx             exist_tensor_idx,       // index in tensor_prods of existing tensor
                      int                   cur_dim,                // current dimension to intersect
                      KnotIdx               knot_idx,               // knot index in current dim of split point
                      int                   split_side,             // whether min (-1) or max (1) or both sides (2) of new tensor split the existing tensor
                      bool&                 knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            // convert global knot_idx to local_knot_idx in exist_tensor
            KnotIdx local_knot_idx = global2local_knot_idx(knot_idx, exist_tensor_idx, cur_dim);

            TensorProduct<T>& exist_tensor  = tensor_prods[exist_tensor_idx];

            // intialize a new side_tensor for the minimum or maximum side of the existing tensor
            TensorProduct<T> side_tensor;
            side_tensor.next.resize(dom_dim_);
            side_tensor.prev.resize(dom_dim_);
            side_tensor.nctrl_pts.resize(dom_dim_);
            side_tensor.knot_mins   = exist_tensor.knot_mins;
            side_tensor.knot_maxs   = exist_tensor.knot_maxs;
            if (split_side == -1 || split_side == 2)
            {
                side_tensor.knot_mins[cur_dim]  = knot_idx;
                exist_tensor.knot_maxs[cur_dim] = knot_idx;
            }
            else
            {
                side_tensor.knot_maxs[cur_dim]  = knot_idx;
                exist_tensor.knot_mins[cur_dim] = knot_idx;
            }
            side_tensor.level           = exist_tensor.level;
            TensorIdx side_tensor_idx   = tensor_prods.size();                  // index of new tensor to be added

            // new side tensor will be added
            if (!subset(side_tensor.knot_mins, side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {
                // adjust prev and nex pointers
                adjust_prev_next(exist_tensor_idx, side_tensor, side_tensor_idx, new_tensor, split_side, cur_dim);

                //  split control points between existing and max side tensors

                // debug
//                 fprintf(stderr, "1: calling split_ctrl_pts exist_tensor_idx=%lu cur_dim=%d split_side=%d global knot_idx = %lu local_knot_idx=%lu\n",
//                         exist_tensor_idx, cur_dim, split_side, knot_idx, local_knot_idx);

                if (split_side == -1 || split_side == 2)
                    split_ctrl_pts(exist_tensor_idx, side_tensor, cur_dim, local_knot_idx, split_side, false, true);
                else
                    split_ctrl_pts(exist_tensor_idx, side_tensor, cur_dim, local_knot_idx, split_side, false, false);

                // add the new max side tensor
                tensor_prods.push_back(side_tensor);

                // reset the reference, which could be invalid after the push_back
                TensorProduct<T>& et  = tensor_prods[exist_tensor_idx];

                // delete next and prev pointers of existing tensor that are no longer valid as a result of adding new max side
                delete_old_pointers(exist_tensor_idx);

                // check if the knot mins, maxs of the existing or added tensor match the original new tensor
                if ( (side_tensor.knot_mins == new_tensor.knot_mins && side_tensor.knot_maxs == new_tensor.knot_maxs) ||
                     (et.knot_mins == new_tensor.knot_mins && et.knot_maxs == new_tensor.knot_maxs) )
                    knots_match = true;

                return true;
            }

            // new side tensor will not be added
            else
            {
                // debug
//                 fprintf(stderr, "2: calling split_ctrl_pts exist_tensor_idx=%lu cur_dim=%d split_side=%d global knot_idx = %lu local_knot_idx=%lu\n",
//                         exist_tensor_idx, cur_dim, split_side, knot_idx, local_knot_idx);

                if (split_side == -1 || split_side == 2)
                    split_ctrl_pts(exist_tensor_idx, new_tensor, cur_dim, local_knot_idx, split_side, true, true);
                else
                    split_ctrl_pts(exist_tensor_idx, new_tensor, cur_dim, local_knot_idx, split_side, true, false);

                // delete next and prev pointers of existing tensor that are no longer valid as a result of adding new max side
                delete_old_pointers(exist_tensor_idx);

                return false;
            }
        }

        // adjust prev and next pointers for splitting a tensor into two
        void adjust_prev_next(
                TensorIdx           exist_tensor_idx,       // idx of existing tensor after split
                TensorProduct<T>&   side_tensor,            // new side tensor after split, not update in tensor_prods yet
                                                            // use this reference instead of tensor_prods[side_tensor_idx]
                TensorIdx           side_tensor_idx,        // idx of new side tensor after split
                TensorProduct<T>&   new_tensor,             // new tensor being added that started all this
                int                 split_side,             // whether min (-1) or max (1) or both sides (2) of new tensor split the existing tensor
                int                 cur_dim)                // current dimension to intersect
        {
            TensorProduct<T>&   exist_tensor    = tensor_prods[exist_tensor_idx];
            TensorProduct<T>*   min_tensor;
            TensorProduct<T>*   max_tensor;
            TensorIdx           min_tensor_idx;
            TensorIdx           max_tensor_idx;

            // figure which side is min and which is max
            if (split_side == 1)
            {
                min_tensor      = &side_tensor;
                max_tensor      = &exist_tensor;
                min_tensor_idx  = side_tensor_idx;
                max_tensor_idx  = exist_tensor_idx;
            }
            else
            {
                min_tensor      = &exist_tensor;
                max_tensor      = &side_tensor;
                min_tensor_idx  = exist_tensor_idx;
                max_tensor_idx  = side_tensor_idx;
            }

            // adjust next and prev pointers for min and max tensors in the current dimension
            for (int i = 0; i < min_tensor->next[cur_dim].size(); i++)
            {
                if (adjacent(*max_tensor, tensor_prods[min_tensor->next[cur_dim][i]], cur_dim))
                {
                    max_tensor->next[cur_dim].push_back(min_tensor->next[cur_dim][i]);
                    vector<TensorIdx>& prev = tensor_prods[min_tensor->next[cur_dim][i]].prev[cur_dim];
                    auto it                 = find(prev.begin(), prev.end(), min_tensor_idx);
                    assert(it != prev.end());       // sanity
                    size_t k                = distance(prev.begin(), it);
                    prev[k]                 = max_tensor_idx;
                }
            }
            for (int i = 0; i < max_tensor->prev[cur_dim].size(); i++)
            {
                if (adjacent(*min_tensor, tensor_prods[max_tensor->prev[cur_dim][i]], cur_dim))
                {
                    min_tensor->prev[cur_dim].push_back(max_tensor->prev[cur_dim][i]);
                    vector<TensorIdx>& next = tensor_prods[max_tensor->prev[cur_dim][i]].next[cur_dim];
                    auto it                 = find(next.begin(), next.end(), max_tensor_idx);
                    assert(it != next.end());       // sanity
                    size_t k                = distance(next.begin(), it);
                    next[k]                 = min_tensor_idx;
                }
            }

            // connect next and prev pointers of existing and new side tensors only if
            // the new tensor will not completely separate the two
            if (!occluded(new_tensor, *min_tensor, cur_dim))
            {
                min_tensor->next[cur_dim].push_back(max_tensor_idx);
                max_tensor->prev[cur_dim].push_back(min_tensor_idx);
            }

            // adjust next and prev pointers for min and max tensors in other dimensions
            for (int j = 0; j < dom_dim_; j++)
            {
                if (j == cur_dim)
                    continue;

                // next pointer
                for (int i = 0; i < exist_tensor.next[j].size(); i++)
                {
                    // add new next pointers
                    if (adjacent(side_tensor, tensor_prods[exist_tensor.next[j][i]], j))
                    {
                        side_tensor.next[j].push_back(exist_tensor.next[j][i]);
                        tensor_prods[exist_tensor.next[j][i]].prev[j].push_back(side_tensor_idx);
                    }

                }

                // prev pointer
                for (int i = 0; i < exist_tensor.prev[j].size(); i++)
                {
                    // add new prev pointers
                    if (adjacent(side_tensor, tensor_prods[exist_tensor.prev[j][i]], j))
                    {
                        side_tensor.prev[j].push_back(exist_tensor.prev[j][i]);
                        tensor_prods[exist_tensor.prev[j][i]].next[j].push_back(side_tensor_idx);
                    }

                }
            }
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

        // split control points between existing and new side tensors
        void split_ctrl_pts(TensorIdx            existing_tensor_idx,    // index in tensor_prods of existing tensor
                            TensorProduct<T>&    new_side_tensor,        // new side tensor
                            int                  cur_dim,                // current dimension to intersect
                            KnotIdx              split_knot_idx,         // local (not global!) knot index in current dim of split point in existing tensor
                            int                  split_side,             // whether min (-1) or max (1) or both sides (2) of new tensor split the existing tensor
                            bool                 skip_new_side,          // don't add control points to new_side tensor, only adjust exsiting tensor control points
                            bool                 max_side)               // new side is in the max. direction (false = min. direction)
        {
            TensorProduct<T>& existing_tensor = tensor_prods[existing_tensor_idx];

            // index of min and max in new side or existing tensor (depending on max_side true/false) control points in current dim
            // allowed to be negative in order for the logic below to partition correctly (long long instead of size_t)
            long long min_ctrl_idx; // index of min (max_side = true: in new side; max_side = false: in existing tensor) control points in current dim
            long long max_ctrl_idx; // index of max (max_side = true: in existing tensor; max_side = false: in new side) control points in current dim

            // convert split_knot_idx to ctrl_pt_idx
            min_ctrl_idx = split_knot_idx;
            max_ctrl_idx = split_knot_idx;

            // TODO: unclear why the following is needed, but things break if we don't do this
            if (p_[cur_dim] % 2 == 0)                           // even degree
                max_ctrl_idx--;

            // if tensor starts at global minimum, skip the first (p + 1) / 2 knots
            if ((max_side  && existing_tensor.knot_mins[cur_dim] == 0) ||
                (!max_side && new_side_tensor.knot_mins[cur_dim] == 0))
            {
                min_ctrl_idx -= (p_[cur_dim] + 1) / 2;
                max_ctrl_idx -= (p_[cur_dim] + 1) / 2;
            }

            // if max_ctrl_idx is past last existing control point, then split is too close to global edge and must be clamped to last control point
            // NB there is no equivalent for !max_side because new_side_tensor does not have any numbers of control points assigned yet
            if (max_side && max_ctrl_idx >= existing_tensor.nctrl_pts[cur_dim])
                max_ctrl_idx = existing_tensor.nctrl_pts[cur_dim] - 1;

            // DEPRECATE: there is no equivalent for !max_side because new_side_tensor does not have any numbers of control points assigned yet
//             if (!max_side && min_ctrl_idx >= new_side_tensor.nctrl_pts[cur_dim])
//                 min_ctrl_idx = new_side_tensor.nctrl_pts[cur_dim] - 1;

            // debug
//             fprintf(stderr, "splitting ctrl points in dim %d max_side %d split_knot_idx=%lu max_ctrl_idx=%lld min_ctrl_idx=%lld\n",
//                     cur_dim, max_side, split_knot_idx, max_ctrl_idx, min_ctrl_idx);
//             fprintf(stderr, "old existing tensor tot_nctrl_pts=%lu = [%d]\n", existing_tensor.ctrl_pts.rows(), existing_tensor.nctrl_pts[0]);

            // allocate new control point matrix for existing tensor
            size_t tot_nctrl_pts = 1;
            VectorXi new_exist_nctrl_pts(dom_dim_);
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (i != cur_dim)
                    new_exist_nctrl_pts(i) = existing_tensor.nctrl_pts(i);
                else
                {
                    if (max_side)
                        new_exist_nctrl_pts(i) = max_ctrl_idx + 1;
                    else
                        new_exist_nctrl_pts(i) = existing_tensor.nctrl_pts(i) - min_ctrl_idx;
                }
                tot_nctrl_pts *= new_exist_nctrl_pts(i);
            }
            MatrixX<T> new_exist_ctrl_pts(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
            VectorX<T> new_exist_weights(tot_nctrl_pts);

            // allocate new control point matrix for new side tensor
            if (!skip_new_side)
            {
                tot_nctrl_pts = 1;
                for (auto i = 0; i < dom_dim_; i++)
                {
                    if (i != cur_dim)
                        new_side_tensor.nctrl_pts(i) = existing_tensor.nctrl_pts(i);
                    else
                    {
                        if (max_side)
                            new_side_tensor.nctrl_pts(i) = existing_tensor.nctrl_pts(i) - min_ctrl_idx;
                        else
                            new_side_tensor.nctrl_pts(i) = max_ctrl_idx + 1;
                    }
                    tot_nctrl_pts *= new_side_tensor.nctrl_pts(i);
                }
                new_side_tensor.ctrl_pts.resize(tot_nctrl_pts, max_dim_ - min_dim_ + 1);
                new_side_tensor.weights.resize(tot_nctrl_pts);
            }

            // split the control points
            VolIterator vol_iter(existing_tensor.nctrl_pts);            // for iterating in a flat loop over n dimensions
            size_t      cur_exist_idx    = 0;                           // current index into new_exist_ctrl_pts and weights
            size_t      cur_new_side_idx = 0;                           // current index into new_side_tensor.ctrl_pts and weights
            while (!vol_iter.done())
            {
                // control point goes either to existing or new side tensor depending on index in current dimension

                // control point goes to existing tensor
                if ((max_side && vol_iter.idx_dim(cur_dim) <= max_ctrl_idx) || (!max_side && vol_iter.idx_dim(cur_dim) >= min_ctrl_idx))
                {
                    // debug
//                     fprintf(stderr, "moving to new_exist_ctrl_pts[%lu]\n", cur_exist_idx);

                    new_exist_ctrl_pts.row(cur_exist_idx) = existing_tensor.ctrl_pts.row(vol_iter.cur_iter());
                    new_exist_weights(cur_exist_idx) = existing_tensor.weights(vol_iter.cur_iter());

                    // when degree is odd, set the ctrl point to nan, real value is in the new side tensor
                    // TODO: check for being outside the new tensor bounds in the other dims
                    if (split_side != 2 && p_(cur_dim) % 2 &&
                            ((max_side && vol_iter.idx_dim(cur_dim) == max_ctrl_idx) ||
                             (!max_side && vol_iter.idx_dim(cur_dim) == min_ctrl_idx)))
                        new_exist_weights(cur_exist_idx, 0) = MFA_NAW;

                    cur_exist_idx++;
                }

                // control point goes to new side tensor
                if ((max_side && vol_iter.idx_dim(cur_dim) >= min_ctrl_idx) || (!max_side && vol_iter.idx_dim(cur_dim) <= max_ctrl_idx))
                {
                    if (!skip_new_side)
                    {
                        // debug
//                         fprintf(stderr, "moving to new_side_tensor.ctrl_pts[%lu]\n", cur_new_side_idx);

                        new_side_tensor.ctrl_pts.row(cur_new_side_idx) = existing_tensor.ctrl_pts.row(vol_iter.cur_iter());
                        new_side_tensor.weights(cur_new_side_idx) = existing_tensor.weights(vol_iter.cur_iter());

                        // when degree is odd and new tensor is inside the existing tensor,
                        // set the ctrl point to nan, real value is in the added tensor
                        // TODO: check for being outside the new tensor bounds in the other dims
                        if (split_side == 2 && p_(cur_dim) % 2 &&
                                ((max_side && vol_iter.idx_dim(cur_dim) == min_ctrl_idx) ||
                                 (!max_side && vol_iter.idx_dim(cur_dim) == max_ctrl_idx)))
                            new_side_tensor.weights(cur_new_side_idx, 0) = MFA_NAW;
                    }

                    cur_new_side_idx++;
                }

                vol_iter.incr_iter();                                   // must increment volume iterator at the bottom of the loop
            }

            // copy new_exist_ctrl_pts and weights to existing_tensor.ctrl_pts and weights, resizes automatically
            existing_tensor.ctrl_pts    = new_exist_ctrl_pts;
            existing_tensor.weights     = new_exist_weights;
            existing_tensor.nctrl_pts   = new_exist_nctrl_pts;

            // debug
//             fprintf(stderr, "new existing tensor tot_nctrl_pts=%lu = [%d %d]\n", existing_tensor.ctrl_pts.rows(), existing_tensor.nctrl_pts[0], existing_tensor.nctrl_pts[1]);
//             if (!skip_new_side)
//                 fprintf(stderr, "max side tensor tot_nctrl_pts=%lu = [%d %d]\n\n", new_side_tensor.ctrl_pts.rows(), new_side_tensor.nctrl_pts[0], new_side_tensor.nctrl_pts[1]);
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
                        i--;                                        // keep loop counter the same for next iteration
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
                        i--;                                        // keep loop counter the same for next iteration
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

        // checks if a point in index space is in a tensor product
        // in all dimensions except skip_dim (-1 = default, don't skip any dimensions)
        // if ctrl_pt_anchor == true, for dimension i, if degree[i] is even, pt[i] + 0.5 is checked because pt coords are truncated to integers
        bool in(const vector<KnotIdx>&  pt,
                const TensorProduct<T>& tensor,
                bool                    ctrl_pt_anchor,         // whether pt refers to control point anchor (shifted 1/2 space for even degree)
                int                     skip_dim = -1) const
        {
            for (auto i = 0; i < pt.size(); i++)
            {
                if (i == skip_dim)
                    continue;

                // move pt[i] to center of t-mesh cell if degree is even and pt refers to control point anchor
                float fp;               // floating point version of pt[i]
                if (ctrl_pt_anchor && p_[i] % 2 == 0)
                    fp = pt[i] + 0.5;
                else
                    fp = pt[i];

                if (fp < float(tensor.knot_mins[i]) || fp > float(tensor.knot_maxs[i]))
                        return false;
            }

            return true;
        }

        // checks if a point in index space is in a tensor product in one dimension
        // for dimension i, if degree[i] is even, pt[i] + 0.5 is checked because pt coords are truncated to integers
        // if ctrl_pt_anchor == true, if degree[cur_dim] is even, pt + 0.5 is checked because pt coords are truncated to integers
        bool in_dim(KnotIdx                 pt,
                    const TensorProduct<T>& tensor,
                    bool                    ctrl_pt_anchor,         // whether pt refers to control point anchor (shifted 1/2 space for even degree)
                    int                     cur_dim) const
        {
            // move pt to center of t-mesh cell if degree is even and pt refers to control point anchor
            float fp;               // floating point version of pt
            if (ctrl_pt_anchor && p_[cur_dim] % 2 == 0)
                fp = pt + 0.5;
            else
                fp = pt;
            return(fp >= float(tensor.knot_mins[cur_dim]) && fp <= float(tensor.knot_maxs[cur_dim]));
        }

        // checks if a point in index space is in the neighbors of a tensor product, for one dimension
        void in_neighbors(const vector<KnotIdx>&    pt,                     // anchor point in index space
                          int                       cur_dim,                // current dimension
                          int                       tensor_idx,             // index of current tensor
                          vector<int>&              neighbor_idxs) const    // (output) neighbor tensor idxs containing the point in cur_dim
        {
            const TensorProduct<T>& t = tensor_prods[tensor_idx];

            for (auto i = 0; i < t.prev[cur_dim].size(); i++)
            {
                const TensorProduct<T>&   tp = tensor_prods[t.prev[cur_dim][i]];
                if (pt[cur_dim] >= tp.knot_mins[cur_dim] && pt[cur_dim] <= tp.knot_maxs[cur_dim])
                    neighbor_idxs.push_back(t.prev[cur_dim][i]);
            }

            for (auto i = 0; i < t.next[cur_dim].size(); i++)
            {
                const TensorProduct<T>&   tn = tensor_prods[t.next[cur_dim][i]];
                if (pt[cur_dim] >= tn.knot_mins[cur_dim] && pt[cur_dim] <= tn.knot_maxs[cur_dim])
                    neighbor_idxs.push_back(t.next[cur_dim][i]);
            }
        }

        // given an anchor in index space, find intersecting knot lines in index space
        // in -/+ directions in all dimensions
        void knot_intersections(const vector<KnotIdx>&      anchor,                 // knot indices of anchor for odd degree or
                                                                                    // knot indices of start of rectangle containing anchor for even degree
                                TensorIdx                   t_idx,                  // index of tensor product containing anchor
                                bool                        ctrl_pt_anchor,         // whether pt refers to control point anchor (shifted 1/2 space for even degree)
                                vector<vector<KnotIdx>>&    loc_knots) const        // (output) local knot vector in index space
        {
            loc_knots.resize(dom_dim_);
            assert(anchor.size() == dom_dim_);

            int max_level = tensor_prods[t_idx].level;                      // level of tensor product

            vector<NeighborTensor> unused;

            // walk the t-mesh in all dimensions, min. and max. directions outward from the anchor
            // looking for interecting knot lines

            for (auto i = 0; i < dom_dim_; i++)                             // for all dims
            {
                loc_knots[i].resize(p_(i) + 2);                              // support of basis func. is p+2 by definition

                KnotIdx start, min, max;
                if (p_(i) % 2)                                               // odd degree
                    start = min = max = (p_(i) + 1) / 2;
                else                                                        // even degree
                {
                    start = min = p_(i) / 2;
                    max         = p_(i) / 2 + 1;
                }
                loc_knots[i][start]             = anchor[i];                // start at the anchor
                KnotIdx         cur_knot_idx    = loc_knots[i][start];
                TensorIdx       cur_tensor      = t_idx;
                int             cur_level       = max_level;
                vector<KnotIdx> cur             = anchor;                   // current knot location in the tmesh (index space)

                // from the anchor in the min. direction
                for (int j = 0; j < min; j++)                               // add 'min' more knots in minimum direction from the anchor
                {
                    bool done = false;
                    do
                    {
                        // find previous knot in the current tensor, skipping higher level knots
                        int incr = -1;
                        while ((long)cur_knot_idx + incr >= tensor_prods[cur_tensor].knot_mins[i] &&
                                all_knot_levels[i][cur_knot_idx + incr] > cur_level)
                            incr--;

                        if ((long)cur_knot_idx + incr > 0)                               // more knots in the tmesh
                        {
                            cur[i] = cur_knot_idx + incr;

                            // check which is the correct previous tensor
                            // if more than one tensor sharing the target, pick highest level
                            if (!in_dim(cur_knot_idx + incr, tensor_prods[cur_tensor], true, i))
                            {
                                // debug
                                if (tensor_prods[cur_tensor].prev[i].size() == 0)
                                    fprintf(stderr, "1:\n");
                                assert(tensor_prods[cur_tensor].prev[i].size() > 0);            // sanity
                                neighbor_tensors(tensor_prods[cur_tensor].prev[i], i, cur, ctrl_pt_anchor, cur_tensor, cur_level, unused);
                            }

                            // check if next knot borders a higher level; if so, switch to higher level tensor
                            border_higher_level(cur, ctrl_pt_anchor, cur_tensor, cur_level);

                            // move to next knot
                            if (cur_knot_idx + incr > 0                                         &&
                                cur_knot_idx + incr >= tensor_prods[cur_tensor].knot_mins[i]    &&
                                all_knot_levels[i][cur_knot_idx + incr] <= cur_level)
                            {
                                cur_knot_idx += incr;
                                loc_knots[i][start - j - 1] = cur_knot_idx;
                                done = true;
                            }
                        }
                        else                                                // no more knots in the tmesh
                        {
                            loc_knots[i][start - j - 1] = 0;                // repeat first index as many times as needed
                            done = true;
                        }
                    } while (!done);
                }

                // reset back to anchor
                cur_knot_idx    = loc_knots[i][start];
                cur_tensor      = t_idx;
                cur_level       = max_level;
                cur             = anchor;

                // from the anchor in the max. direction
                for (int j = 0; j < max; j++)                               // add 'max' more knots in maximum direction from the anchor
                {
                    bool done = false;
                    do
                    {
                        // find next knot in the current tensor, skipping higher level knots
                        int incr = 1;
                        while (cur_knot_idx + incr <= tensor_prods[cur_tensor].knot_maxs[i] &&
                                all_knot_levels[i][cur_knot_idx + incr] > cur_level)
                            incr++;

                        if (cur_knot_idx + incr < all_knots[i].size() - 1)      // more knots in the tmesh
                        {
                            cur[i] = cur_knot_idx + incr;

                            // check which is the correct next tensor
                            // if more than one tensor sharing the target, pick highest level
                            if (!in_dim(cur_knot_idx + incr, tensor_prods[cur_tensor], true, i))
                            {
                                assert(tensor_prods[cur_tensor].next[i].size() > 0);        // sanity
                                neighbor_tensors(tensor_prods[cur_tensor].next[i], i, cur, ctrl_pt_anchor, cur_tensor, cur_level, unused);
                            }

                            // check if next knot borders a higher level; if so, switch to higher level tensor
                            border_higher_level(cur, ctrl_pt_anchor, cur_tensor, cur_level);

                            // move to next knot
                            if (cur_knot_idx + incr < all_knots[i].size()                      &&
                                cur_knot_idx + incr <= tensor_prods[cur_tensor].knot_maxs[i]   &&
                                all_knot_levels[i][cur_knot_idx + incr] <= cur_level)
                            {
                                cur_knot_idx += incr;
                                loc_knots[i][start + j + 1] = cur_knot_idx;
                                done = true;
                            }
                        }
                        else                                                // no more knots in the tmesh
                        {
                            loc_knots[i][start + j + 1] = all_knots[i].size() - 1;  // repeat last index as many times as needed
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
                              bool                      ctrl_pt_anchor,         // whether pt refers to control point anchor (shifted 1/2 space for even degree)
                              TensorIdx&                cur_tensor,             // (input / output) highest level neighbor tensor containing the target
                              int&                      cur_level,              // (input / output) level of current tensor
                              vector<NeighborTensor>&   neigh_hi_levels) const  // (input / output) neighbors with higher levels than current tensor
        {
            int         temp_max_level;
            TensorIdx   temp_max_k;
            bool        first_time = true;
            for (auto k = 0; k < prev_next.size(); k++)
            {
                if (in(target, tensor_prods[prev_next[k]], ctrl_pt_anchor))
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
                                 bool                   ctrl_pt_anchor,     // whether pt refers to control point anchor (shifted 1/2 space for even degree)
                                 TensorIdx&             cur_tensor,         // (input / output) highest level neighbor tensor containing the target
                                 int&                   cur_level) const    // (input / output) level of current tensor
        {
            for (auto k = cur_tensor; k < tensor_prods.size(); k++) // start checking at current tensor because levels are monotonic nondecreasing
            {
                if (in(target, tensor_prods[k], ctrl_pt_anchor, -1) && tensor_prods[k].level > cur_level)
                {
                    cur_tensor  = k;
                    cur_level   = tensor_prods[k].level;
                }
            }
        }

        // given a point in parameter space to decode, compute p + 1 anchor points in all dims in knot index space
        // anchors are the centers of basis functions and locations of corresponding control points, in knot index space
        // in Bazilevs 2010, knot indices start at 1, but mine start at 0
        // returns index of tensor containing the parameters of the point to decode
        TensorIdx anchors(const VectorX<T>&          param,             // parameter value in each dim. of desired point
                          bool                       expand,            // whether to expand anchors to one level higher than the desired point
                                                                        // a safety mechanism to ensure anchors cover control points in neighboring tensors
                          vector<vector<KnotIdx>>&   anchors) const     // (output) anchor points in index space
        {
            anchors.resize(dom_dim_);

            // convert param to target (center of p + 1 anchors) in index space
            // the target is being computed on the global knot vector, without regarding the level of
            // refinement of the tensor where the decoded point is, and adjusted afterwards.
            vector<KnotIdx> target(dom_dim_);                       // center anchor in each dim.
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (param(i) < 1.0)
                {
                    auto it     = upper_bound(all_knots[i].begin() + p_(i), all_knots[i].end() - p_(i) - 1, param(i));
                    target[i]   = it - all_knots[i].begin() - 1;
                }
                else
                    target[i] = all_knots[i].size() - (p_(i) + 2);
            }

            // find most refined tensor product containing anchor and that level of refinement
            // levels monotonically nondecrease; hence, find last tensor product containing the anchor
            TensorIdx t_idx = 0;
            for (auto j = 0; j < tensor_prods.size(); j++)
                if (in(target, tensor_prods[j], -1))
                        t_idx = j;
            int max_level = tensor_prods[t_idx].level;

            // adjust target to skip over any knots at a higher refinement level than the target
            for (auto i = 0; i < dom_dim_; i++)
            {
                int skip = 0;
                for (auto j = 0; j <= target[i]; j++)
                {
                    if (all_knot_levels[i][j] > max_level)
                        skip++;
                    else                // reset skip once crossing back into correct level
                        skip = 0;
                }
                target[i] -= skip;
            }

            // find local knot vector (p + 2) knot intersections
            vector<vector<KnotIdx>> loc_knots(dom_dim_);
            knot_intersections(target, t_idx, true, loc_knots);

            // take correct p + 1 anchors out of the p + 2 found above
            for (auto i = 0; i < dom_dim_; i++)
            {
                anchors[i].resize(p_(i) + 1);
                for (auto j = 0; j < p_(i) + 1; j++)
                {
                    if (p_(i) % 2 == 0)                             // even degree
                        anchors[i][j] = loc_knots[i][j];
                    else                                            // odd degree
                        anchors[i][j] = loc_knots[i][j + 1];
                }
            }

            // debug
//             vector<vector<KnotIdx>> old_anchors = anchors;
//             bool changed_anchors = false;

            // expand anchors by one level higher
            if (expand)
            {
                // expand anchors to next level higher than current
                // TODO: assumes that anchors don't cover more than one change of level
                // this may be reasonable assuming tensors are always at least size p control points, but need to verify
                int level = max_level > 0 ? max_level - 1 : max_level;
                for (auto i = 0; i < dom_dim_; i++)
                {
                    int start = p_(i) % 2 == 0 ? p_(i) / 2 : p_(i) / 2 - 1;        // loop index of target
                    // from the target to the left
                    int skip = 0;
                    for (auto j = start; j >= 0; j--)
                    {
                        anchors[i][j] -= skip;
                        skip = 0;
                        while (all_knot_levels[i][anchors[i][j]] > level)
                        {
                            anchors[i][j]--;
                            skip++;
                            // debug
//                             changed_anchors = true;
                        }
                    }
                    // from after the target to the right
                    skip = 0;
                    for (auto j = start + 1; j < p_(i) + 1; j++)
                    {
                        anchors[i][j] += skip;
                        skip = 0;
                        while (all_knot_levels[i][anchors[i][j]] > level)
                        {
                            anchors[i][j]++;
                            skip++;
                            // debug
//                             changed_anchors = true;
                        }
                    }
                }
            }

            // debug
//             if (param(0) > 0.8 && param(0) < 0.9 && param(1) > 0.6 && param(1) < 0.7)
//             {
//                 if (dom_dim_ == 1)
//                     fprintf(stderr, "anchors(): param=[%.2lf] target=[%lu]\n", param(0), target[0]);
//                 if (dom_dim_ == 2)
//                     fprintf(stderr, "anchors(): param=[%.2lf %.2lf] target=[%lu %lu]\n", param(0), param(1), target[0], target[1]);
//             }

            // debug
//             if (changed_anchors && param(0) > 0.7 && param(0) < 0.8 && param(1) > 0.6 && param(1) < 0.7)
//             {
//                 cerr << "changed anchors at param: " << param.transpose() << endl;
//                 fprintf(stderr, "target: [ ");
//                 for (auto i = 0; i < dom_dim_; i++)
//                     fprintf(stderr, "%lu ", target[i]);
//                 fprintf(stderr, " ]\n");
//                 for (auto i = 0; i < dom_dim_; i++)
//                 {
//                     fprintf(stderr, "old_anchors[%d]: [ ", i);
//                     for (auto j = 0; j < old_anchors[i].size(); j++)
//                         fprintf(stderr, "%lu ", old_anchors[i][j]);
//                     fprintf(stderr, "] ");
//                 }
//                 fprintf(stderr, "\n");
//                 for (auto i = 0; i < dom_dim_; i++)
//                 {
//                     fprintf(stderr, "anchors[%d]: [ ", i);
//                     for (auto j = 0; j < anchors[i].size(); j++)
//                         fprintf(stderr, "%lu ", anchors[i][j]);
//                     fprintf(stderr, "] ");
//                 }
//                 fprintf(stderr, "\n");
//             }

            return t_idx;
        }

        // checks if a target point in index space is in the multidim range of anchors
        bool in_anchors(const vector<KnotIdx>&          target,             // target point
                        const vector<vector<KnotIdx>>&  anchors) const
        {
            // assumes anchor indices are sorted from low to high, which should always be true
            for (auto k = 0; k < anchors.size(); k++)
            {
                if (target[k] < anchors[k].front() || target[k] > anchors[k].back())
                    return false;
            }
            return true;
        }

        // scatter global set of control points into their proper tensor products in the tmesh
        void scatter_ctrl_pts(const VectorXi&           nctrl_pts,          // number of control points in each dim.
                              const MatrixX<T>&         ctrl_pts,           // control points
                              const VectorX<T>&         weights)            // weights
        {
            // 2-step algorithm:
            // 1. Count required number of control points required to be allocated in each tensor,
            // recording destination tensor. Skip control points that are extensions of refined tensors.
            // 2. Copy the control points to the tensors using the recorded destination tensor for each control point

            vector<KnotIdx>     anchor(dom_dim_);                           // anchor in each dim. of control point in global tensor
            int                 tensor_idx = 0;                             // index of current tensor
            vector<int>         start_tensor_idx(dom_dim_, 0);              // index of starting tensor when dimension changes
            vector<size_t>      tensor_tot_nctrl_pts(tensor_prods.size());  // number of control points needed to allocate in each tensor
            vector<size_t>      tensor_cur_nctrl_pts(tensor_prods.size());  // current number of control points in each tensor so far
            vector<vector<int>> tensor_idxs(ctrl_pts.rows());               // destination tensors for each control point, empty if skipping point

            VolIterator         vol_iter(nctrl_pts);                        // iterator over control points
            VectorXi            ijk(dom_dim_);                              // multidim index of current control point

            while (!vol_iter.done())
            {
                assert(tensor_idx >= 0);                                    // sanity: anchor was found in some tensor

                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);

                // get anchor of the current control point
                for (auto j = 0; j < dom_dim_; j++)
                    anchor[j] = ijk(j) + (p_(j) + 1) / 2;                  // first control point has anchor floor((p + 1) / 2)

                // debug
//                 cerr << "ctrl_pt(" << vol_iter.cur_iter() << ") ijk [ " << ijk.transpose() << " ] anchor [ " << anchor[0] << " " << anchor[1] <<
//                     " ] value " << ctrl_pts.row(vol_iter.cur_iter()) << " found in tensor_idx " << tensor_idx << endl;

                // find minimum level of all the dimensions of the anchor of the control point
                int anchor_max_level;
                for (auto j = 0; j < dom_dim_; j++)
                {
                    if (j == 0 || all_knot_levels[j][anchor[j]] > anchor_max_level)
                        anchor_max_level = all_knot_levels[j][anchor[j]];
                }

                // set the destination tensor for the point
                if (anchor_max_level <= tensor_prods[tensor_idx].level)
                {
                    tensor_tot_nctrl_pts[tensor_idx]++;
                    tensor_idxs[vol_iter.cur_iter()].push_back(tensor_idx);

                    // debug
//                     cerr << "ctrl_pt(" << vol_iter.cur_iter() << ") ijk [ " << ijk.transpose() << " ] value " <<
//                         ctrl_pts.row(vol_iter.cur_iter()) << " destination tensor " << tensor_idx << endl;
                }

                // if degree is odd, check borders of neighboring tensors
                // because an anchor can lie on a line between tensors, in which case
                // the control point is copied in multiple tensors
                for (auto j = 0; j < dom_dim_; j++)
                {
                    if (p_(j) % 2)              // odd degree
                    {
                        vector<int> neighbor_idxs;
                        in_neighbors(anchor, j, tensor_idx, neighbor_idxs);
                        for (auto k = 0; k < neighbor_idxs.size(); k++)
                        {
                            TensorIdx           neigh_idx       = neighbor_idxs[k];
                            TensorProduct<T>&   neigh_tensor    = tensor_prods[neigh_idx];
                            if (anchor_max_level <= neigh_tensor.level && in(anchor, neigh_tensor, true))
                            {
                                tensor_tot_nctrl_pts[neigh_idx]++;
                                tensor_idxs[vol_iter.cur_iter()].push_back(neigh_idx);

                                // debug
//                                 cerr << "ctrl_pt(" << vol_iter.cur_iter() << ") ijk [ " << ijk.transpose() << " ] value " <<
//                                     ctrl_pts.row(vol_iter.cur_iter()) << " destination tensor " << neigh_idx << endl;
                            }
                        }   // neighbors
                    }   // odd degree
                }   // j < dom_dim_

                anchor[0]++;
                vol_iter.incr_iter();                       // increment vol_iter

                // update tensor_idx and start_tensor_idx, dealing with boundary conditions
                for (auto k = 0; k < dom_dim_; k++)
                {
                    if (k < dom_dim_ - 1 && vol_iter.done(k))
                    {
                        anchor[0] = (p_(0) + 1) / 2;
                        anchor[k + 1]++;

                        // debug
//                         cerr << "2: anchor [ " << anchor[0] << " " << anchor[1] << " ]" << endl;

                        if (ijk(k + 1) < nctrl_pts(k + 1))
                        {
                            // check for the anchor in the current tensor and in next pointers in next higher dim, starting back at last tensor of current dim
                            tensor_idx = in_and_next(anchor, start_tensor_idx[k + 1], k + 1, true);
                            if (tensor_idx >= 0)
                            {
                                // TODO: following is untested, need higher dimension example with multiple tensors
                                start_tensor_idx[k + 1] = tensor_idx;           // adjust start tensor of next dim
                                start_tensor_idx[k]     = start_tensor_idx[0];  // reset start tensor of current dim
                            }
                        }
                    }
                    if (!vol_iter.done(k))
                    {
                        // debug
//                         if (anchor[0] == 4 && anchor[1] == 1)
//                             cerr << "0: anchor [ " << anchor[0] << " " << anchor[1] << " ] " << "tensor_idx = " << tensor_idx << endl;

                        tensor_idx = in_and_next(anchor, tensor_idx, 0, true);

                        // debug
//                         if (anchor[0] == 4 && anchor[1] == 1)
//                             cerr << "1: anchor [ " << anchor[0] << " " << anchor[1] << " ] " << "tensor_idx = " << tensor_idx << endl;
                    }
                }
            }       // vol_iter not done

            // debug: assert that total number of control points in each tensor matches
            // allocated space and product of nctrl_pts in the tensor
            for (int i = 0; i < tensor_prods.size(); i++)
            {
                // debug
                if (tensor_tot_nctrl_pts[i] != tensor_prods[i].ctrl_pts.rows())
                    fprintf(stderr, "1:\n");

                // debug
                assert(tensor_tot_nctrl_pts[i] == tensor_prods[i].ctrl_pts.rows());
                assert(tensor_tot_nctrl_pts[i] == tensor_prods[i].weights.size());
                assert(tensor_tot_nctrl_pts[i] == tensor_prods[i].nctrl_pts.prod());
            }

            // copy control points and weights to tensors
            // their size should be correct already because they were resized elsewhere in append_tensor()
            for (int i = 0; i < ctrl_pts.rows(); i++)
            {
                for (auto j = 0; j < tensor_idxs[i].size(); j++)
                {
                    // don't lose the MFA_NAW label on control points by overwriting it
                    if (tensor_prods[tensor_idxs[i][j]].weights(tensor_cur_nctrl_pts[tensor_idxs[i][j]]) != MFA_NAW)
                    {
                        tensor_prods[tensor_idxs[i][j]].ctrl_pts.row(tensor_cur_nctrl_pts[tensor_idxs[i][j]]) = ctrl_pts.row(i);
                        tensor_prods[tensor_idxs[i][j]].weights(tensor_cur_nctrl_pts[tensor_idxs[i][j]])      = weights(i);
                    }
                    tensor_cur_nctrl_pts[tensor_idxs[i][j]]++;
                }
            }
        }

        // copy subset of control points into a given tensor product in the tmesh
        // assumes that the destination tensor is the most refined, ie, no knots or control points in its interior should be skipped
        // also assumes that source control points are global for the entire domain
        // both of these assumptions are reasonable when being called by append_tensor
        void subset_ctrl_pts(const VectorXi&           nctrl_pts,          // number of control points in each dim.
                             const MatrixX<T>&         ctrl_pts,           // control points
                             const VectorX<T>&         weights,            // weights
                             int                       tensor_idx)         // index of destination tensor
        {
            TensorProduct<T>& t = tensor_prods[tensor_idx];

            // get starting offsets and numbers of control points in the subset
            VectorXi sub_starts(dom_dim_);
            t.nctrl_pts.resize(dom_dim_);
            for (auto i = 0; i < dom_dim_; i++)
            {
                if (p_(i) % 2 == 0)
                {
                    sub_starts(i)   = t.knot_mins[i] - 1;
                    t.nctrl_pts(i)  = t.knot_maxs[i] - t.knot_mins[i];
                }
                else
                {
                    sub_starts(i)   = t.knot_mins[i] - 2;
                    t.nctrl_pts(i)  = t.knot_maxs[i] - t.knot_mins[i] + 1;
                }
            }

            // allocate control points
            t.ctrl_pts.resize(t.nctrl_pts.prod(), tensor_prods[0].ctrl_pts.cols());
            t.weights.resize(t.ctrl_pts.rows());

            // copy control points
            VolIterator vol_iter(t.nctrl_pts, sub_starts, nctrl_pts);
            VectorXi ijk(dom_dim_);
            while (!vol_iter.done())
            {
                vol_iter.idx_ijk(vol_iter.cur_iter(), ijk);
                t.ctrl_pts.row(vol_iter.cur_iter()) = ctrl_pts.row(vol_iter.ijk_idx(ijk));
                t.weights(vol_iter.cur_iter()) = weights(vol_iter.ijk_idx(ijk));
                vol_iter.incr_iter();
            }
        }

        // check tensor and next pointers of tensor looking for a tensor containing the point
        // only checks the direct next neighbors, not multiple hops
        // returns index of tensor containing the point, or -1 if not found
        int in_and_next(const vector<KnotIdx>&  pt,                     // target point in index space
                        int                     tensor_idx,             // index of starting tensor for the walk
                        int                     cur_dim,                // dimension in which to walk
                        bool                    ctrl_pt_anchor) const   // whether pt refers to control point anchor (shifted 1/2 space for even degree)
        {
            const TensorProduct<T>& tensor = tensor_prods[tensor_idx];

            // check current tensor
            if (in(pt, tensor, ctrl_pt_anchor))
                return tensor_idx;

            // check nearest neighbor next tensors
            for (auto i = 0; i < tensor.next[cur_dim].size(); ++i)
            {
                if (in(pt, tensor_prods[tensor.next[cur_dim][i]], ctrl_pt_anchor))
                    return tensor.next[cur_dim][i];
            }
            return -1;
        }

        // search for tensor containing point in index space
        // returns index of tensor containing the point, or -1 if not found
        int search_tensors(const vector<KnotIdx>&   pt,                     // target point in index space
                           const VectorXi&          pad)                    // padding in each dim. between target and extents of tensor, zero size -> unused
        {
            for (auto i = 0; i < tensor_prods.size(); i++)                  // for all existing tensors
            {
                TensorProduct<T>& t = tensor_prods[i];

                int j;
                for (j = 0; j < dom_dim_; j++)
                {
                    // point falls into the mins, maxs of this tensor, within the pad
                    int p = pad.size() == 0 ? 0 : pad(j);
                    if (pt[j] - p < t.knot_mins[j] || pt[j] + p > t.knot_maxs[j])
                        break;
                }

                if (j == dom_dim_)
                    return i;
            }   // for all existing tensors

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

        // determine starting and ending indices of domain input points covered by one tensor product
        // coverage extends to edge of basis functions corresponding to control points in the tensor product
        void domain_pts(TensorIdx               t_idx,              // index of current tensor product
                        bool                    pad,                // pad by degree p in each dimension
                        vector<size_t>&         start_idxs,         // (output) starting idxs of input points
                        vector<size_t>&         end_idxs) const     // (output) ending idxs of input points
        {
            start_idxs.resize(dom_dim_);
            end_idxs.resize(dom_dim_);
            vector<KnotIdx> min_anchor(dom_dim_);                // anchor for the min. edge basis functions of the new tensor
            vector<KnotIdx> max_anchor(dom_dim_);                // anchor for the mas. edge basis functions of the new tensor
            vector<vector<KnotIdx>> local_knot_idxs;                // local knot vector for an anchor

            const TensorProduct<T>& tc = tensor_prods[t_idx];

            // left edge
            for (auto k = 0; k < dom_dim_; k++)
            {
                min_anchor[k] = tc.knot_mins[k];
                // TODO: the right way to pad is to subtract p here and then find new knot intersections
                // but this is hard because we would have to visit neighboring tensors to get the right tensor id
                // for the shifted anchor, so we cheat and subtract p later from the edge of the local knot vector
                // without any regard for actual knot lines crossed in the tmesh at that time
            }
            knot_intersections(min_anchor, t_idx, true, local_knot_idxs);
            vector<KnotIdx> start_knot_idxs(dom_dim_);
            for (auto k = 0; k < dom_dim_; k++)
            {
                start_knot_idxs[k] = local_knot_idxs[k][0];
                // TODO: this is not the right place to pad, see comment above
                // but it's ok for now
                if (pad)
                    // casting to long allows checking for a negative value of an expression using an unsigned size_t
                    start_knot_idxs[k] = (long)start_knot_idxs[k] - p_(k) < 0 ? 0 : start_knot_idxs[k] - p_(k);
            }

            local_knot_idxs.clear();

            // right edge
            for (auto k = 0; k < dom_dim_; k++)
            {
                if (p_(k) % 2 == 0)
                {
                    max_anchor[k] = tc.knot_maxs[k] - 1;
                }
                else
                {
                    max_anchor[k] = tc.knot_maxs[k];
                }
                // TODO: the right way to pad is to add p here and then find new knot intersections
                // but this is hard because we would have to visit neighboring tensors to get the right tensor id
                // for the shifted anchor, so we cheat and add p later from the edge of the local knot vector
                // without any regard for actual knot lines crossed in the tmesh at that time
            }
            knot_intersections(max_anchor, t_idx, true, local_knot_idxs);
            vector<KnotIdx> end_knot_idxs(dom_dim_);
            for (auto k = 0; k < dom_dim_; k++)
            {
                end_knot_idxs[k] = local_knot_idxs[k].back();
                // TODO: this is not the right place to pad, see comment above
                // but it's ok for now
                if (pad)
                    end_knot_idxs[k] = end_knot_idxs[k] + p_(k) >= all_knots[k].size() ? all_knots[k].size() - 1 : end_knot_idxs[k] + p_(k);
            }

            // input points corresponding to start and end knot values
            for (auto k = 0; k < dom_dim_; k++)
            {
                start_idxs[k]   = all_knot_param_idxs[k][start_knot_idxs[k]];
                end_idxs[k]     = all_knot_param_idxs[k][end_knot_idxs[k]];
            }

            // debug
            for (auto k = 0; k < dom_dim_; k++)
            {
                fprintf(stderr, "start_knot_idx[%d] = %lu end_knot_idx[%d] = %lu\n", k, start_knot_idxs[k], k, end_knot_idxs[k]);
                fprintf(stderr, "start_input_pt_idx[%d] = %lu end_input_pt_idx[%d] = %lu\n", k, start_idxs[k], k, end_idxs[k]);
            }
        }

        // for a given tensor, get anchor of control point, given control point multidim index
        // anchor is in global knot index space (includes knots at higher refinement levels than the tensor)
        void ctrl_pt_anchor(const TensorProduct<T>& t,              // tensor product
                            const VectorXi&         ijk,            // multidim index of control point
                            vector<KnotIdx>&        anchor) const   // (output) anchor
        {
            for (auto j = 0; j < dom_dim_; j++)
            {
                anchor[j] = ijk(j) + t.knot_mins[j];                // add knot_mins to get from local (in this tensor) to global (in the t-mesh) anchor
                if (t.knot_mins[j] == 0)
                    anchor[j] += (p_(j) + 1) / 2;           // first control point has anchor floor((p + 1) / 2)
                // check for any knots at a higher level of refinement that would add to the anchor index (anchor is global over all knots)
                for (auto i = t.knot_mins[j]; i <= t.knot_maxs[j]; i++)
                {
                    if (all_knot_levels[j][i] > t.level && anchor[j] >= i)
                        anchor[j]++;
                }
            }
        }

        void print() const
        {
            // all_knots
            for (int i = 0; i < dom_dim_; i++)
            {
                fprintf(stderr, "all_knots[dim %d]\n", i);
                for (auto j = 0; j < all_knots[i].size(); j++)
                    fprintf(stderr, "%.3lf (l %d) [p %lu]\n",
                            all_knots[i][j], all_knot_levels[i][j], all_knot_param_idxs[i][j]);
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

                fprintf(stderr, "]\n");

                cerr << "ctrl_pts:\n" << t.ctrl_pts << endl;
                cerr << "weights:\n" << t.weights << endl;

                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }

    };
}

#endif
