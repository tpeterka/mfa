//--------------------------------------------------------------
// T-mesh object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _TMESH_HPP
#define _TMESH_HPP

#include    <diy/fmt/format.h>

using namespace std;

struct TensorProduct
{
    vector<size_t> knot_mins;                   // indices into all_knots
    vector<size_t> knot_maxs;                   // indices into all_knots
    vector< vector <size_t>> next;              // next[dim][index of next tensor product]
    vector< vector <size_t>> prev;              // prev[dim][index of previous tensor product]
};

namespace mfa
{
    template <typename T>                       // float or double
    struct Tmesh
    {
        vector<vector<T>>       all_knots;      // all_knots[dimension][index]
        vector<TensorProduct>   tensor_prods;   // all tensor products
        int                     dom_dim_;       // domain dimensionality

        Tmesh(int dom_dim) :
            dom_dim_(dom_dim)                       { all_knots.resize(dom_dim_); }

        // insert a knot into all_knots
        void insert_knot(int    dim,                        // dimension of knot vector
                         size_t pos,                        // new position in all_knots[dim] of inserted knot
                         T      knot)                       // knot value to be inserted
        {
            all_knots[dim].insert(all_knots[dim].begin() + pos, knot);

            // adjust tensor product knot_mins and knot_maxs
            for (TensorProduct& t: tensor_prods)
            {
                if (t.knot_mins[dim] >= pos)
                    t.knot_mins[dim]++;
                if (t.knot_maxs[dim] >= pos)
                    t.knot_maxs[dim]++;
            }
        }

        // insert a tensor product into tensor_prods
        void insert_tensor(const vector<size_t>&    knot_mins,      // indices in all_knots of min. corner of tensor to be inserted
                           const vector<size_t>&    knot_maxs)      // indices in all_knots of max. corner
        {
            bool vec_grew;                          // vector of tensor_prods grew
            bool tensor_inserted = false;           // the desired tensor was already inserted

            // create a new tensor product
            TensorProduct new_tensor;
            new_tensor.next.resize(dom_dim_);
            new_tensor.prev.resize(dom_dim_);
            new_tensor.knot_mins = knot_mins;
            new_tensor.knot_maxs = knot_maxs;

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
//                     fmt::print(stderr, "checking for intersection between new tensor and existing tensor idx={}\n", j);

                    if (nonempty_intersection(new_tensor, tensor_prods[j], split_side))
                    {
                        // debug
//                         fmt::print(stderr, "intersection found between new tensor and existing tensor idx={} split_side=[{} {}]\n",
//                                 j, split_side[0], split_side[1]);
//                         fmt::print(stderr, "\ntensors before intersection\n\n");
//                         print();

                        if ((vec_grew = intersect(new_tensor, j, split_side, knots_match)) && vec_grew)
                        {
                            if (knots_match)
                                tensor_inserted = true;

                            // debug
                            fmt::print(stderr, "\ntensors after intersection\n\n");
                            print();

                            break;  // adding a tensor invalidates iterator, start iteration over
                        }
                    }
                }
            } while (vec_grew);   // keep checking until no more tensors are added

            // add the tensor
            // TODO: update next and prev pointers based on intersections found
            if (!tensor_inserted)
                tensor_prods.push_back(new_tensor);
        }

        // check if nonempty intersection exists in all dimensions between knot_mins, knot_maxs of two tensors
        // assumes new tensor cannot be larger than existing tensor in any dimension (continually refining smaller or equal)
        bool nonempty_intersection(TensorProduct&   new_tensor,         // new tensor product to be added
                                   TensorProduct&   existing_tensor,    // existing tensor product
                                   vector<int>&     split_side)         // (output) whether min (-1) or max (1) of new_tensor is
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
//                     fmt::print(stderr, "cur_dim={} split_side=-1 new min {} exist min {} exist max{}\n",
//                             j, new_tensor.knot_mins[j], existing_tensor.knot_mins[j], existing_tensor.knot_maxs[j]);

                    split_side[j] = -1;
                    retval = true;
                }
                if (new_tensor.knot_maxs[j] > existing_tensor.knot_mins[j] && new_tensor.knot_maxs[j] < existing_tensor.knot_maxs[j])
                {
                    // debug
//                     fmt::print(stderr, "cur_dim={} split_side=1 new max {} exist min {} exist max{}\n",
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
        bool intersect(TensorProduct&   new_tensor,             // new tensor product to be inserted
                       int              existing_tensor_idx,    // index in tensor_prods of existing tensor
                       vector<int>&     split_side,             // whether min (-1) or max (1) or both (2) sides of
                                                                // new tensor are inside existing tensor (one value for each dim.)
                       bool&            knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            knots_match                     = false;
            bool retval                     = false;
            size_t split_knot_idx;

            for (int k = 0; k < dom_dim_; k++)      // for all domain dimensions
            {
                if (!split_side[k])
                    continue;

                split_knot_idx                  = (split_side[k] == -1 ? new_tensor.knot_mins[k] : new_tensor.knot_maxs[k]);
                TensorProduct& existing_tensor  = tensor_prods[existing_tensor_idx];
                vector<size_t> temp_maxs        = existing_tensor.knot_maxs;
                temp_maxs[k]                    = split_knot_idx;

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
        bool new_max_side(TensorProduct&      new_tensor,             // new tensor product that started all this
                          int                 existing_tensor_idx,    // index in tensor_prods of existing tensor
                          int                 cur_dim,                // current dimension to intersect
                          size_t              knot_idx,               // knot index in current dim of split point
                          bool&               knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            TensorProduct& existing_tensor  = tensor_prods[existing_tensor_idx];

            // intialize a new max_side_tensor for the maximum side of the existing_tensor
            TensorProduct max_side_tensor;
            max_side_tensor.next.resize(dom_dim_);
            max_side_tensor.prev.resize(dom_dim_);
            max_side_tensor.knot_mins           = existing_tensor.knot_mins;
            max_side_tensor.knot_maxs           = existing_tensor.knot_maxs;
            max_side_tensor.knot_mins[cur_dim]  = knot_idx;

            existing_tensor.knot_maxs[cur_dim]  = knot_idx;

            size_t max_side_tensor_idx          = tensor_prods.size();                  // index of new tensor to be added

            // check if tensor will be added before adding a next pointer to it
            if (!subset(max_side_tensor.knot_mins, max_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {
                // adjust next and prev pointers for existing_tensor and max_side_tensor in the current dimension
                if (existing_tensor.next[cur_dim].size())
                {
                    max_side_tensor.next[cur_dim].push_back(existing_tensor.next[cur_dim].back());  // TODO: is the correct pointer at the back, or does it have to be found?
                    existing_tensor.next[cur_dim].back()    = max_side_tensor_idx;                  // TODO: is the correct pointer at the back, or does it have to be found?
                }
                else
                {
                    existing_tensor.next[cur_dim].push_back(max_side_tensor_idx);
                }
                max_side_tensor.prev[cur_dim].push_back(existing_tensor_idx);

                // adjust next and prev pointers for existing_tensor and max_side_tensor in other dimensions
                for (int j = 0; j < dom_dim_; j++)
                {
                    if (j == cur_dim)
                        continue;
                    for (int i = 0; i < existing_tensor.next[j].size(); i++)
                        if (intersect_all_dims(max_side_tensor, tensor_prods[existing_tensor.next[j][i]], cur_dim))
                        {
                            max_side_tensor.next[j].push_back(existing_tensor.next[j][i]);
                            tensor_prods[existing_tensor.next[j][i]].prev[j].push_back(max_side_tensor_idx);
                        }
                }

                tensor_prods.push_back(max_side_tensor);

                // check if the knot mins, maxs of the existing or added tensor match the original new tensor
                if ( (max_side_tensor.knot_mins == new_tensor.knot_mins && max_side_tensor.knot_maxs == new_tensor.knot_maxs) ||
                     (existing_tensor.knot_mins == new_tensor.knot_mins && existing_tensor.knot_maxs == new_tensor.knot_maxs) )
                    knots_match = true;

                return true;
            }
            return false;
        }

        // DEPRECATED
//         // split existing tensor product creating extra tensor on minimum side of current dimension
//         // returns true if a an extra tensor product was inserted
//         bool new_min_side(TensorProduct&      new_tensor,             // new tensor product that started all this
//                           int                 existing_tensor_idx,    // index in tensor_prods of existing tensor
//                           int                 cur_dim,                // current dimension to intersect
//                           size_t              knot_idx,               // knot index in current dim of split point
//                           bool&               knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
//         {
//             TensorProduct& existing_tensor  = tensor_prods[existing_tensor_idx];
// 
//             // intialize a new min_side_tensor for the minimum side of the existing_tensor
//             TensorProduct min_side_tensor;
//             min_side_tensor.next.resize(dom_dim_);
//             min_side_tensor.prev.resize(dom_dim_);
//             min_side_tensor.knot_mins           = existing_tensor.knot_mins;
//             min_side_tensor.knot_maxs           = existing_tensor.knot_maxs;
//             min_side_tensor.knot_maxs[cur_dim]  = knot_idx;
// 
//             existing_tensor.knot_mins[cur_dim]  = knot_idx;
// 
//             size_t min_side_tensor_idx          = tensor_prods.size();                  // index of new tensor to be added
// 
//             // check if tensor will be added before adding a next pointer to it
//             if (!subset(min_side_tensor.knot_mins, min_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
//             {
// 
//                 // adjust next and prev pointers for existing_tensor and min_side_tensor in the current dimension
//                 if (existing_tensor.prev[cur_dim].size())
//                 {
//                     fmt::print(stderr, "7:\n");
//                     min_side_tensor.prev[cur_dim].push_back(existing_tensor.prev[cur_dim].back());  // TODO: is the correct pointer at the back, or does it have to be found?
//                     existing_tensor.prev[cur_dim].back()    = min_side_tensor_idx;                  // TODO: is the correct pointer at the back, or does it have to be found?
//                 }
//                 else
//                 {
//                     existing_tensor.prev[cur_dim].push_back(min_side_tensor_idx);
//                     fmt::print(stderr, "8: cur_dim={} min_side_tensor_idx={}\n", cur_dim, min_side_tensor_idx);
//                 }
//                 min_side_tensor.next[cur_dim].push_back(existing_tensor_idx);
//                 fmt::print(stderr, "9: cur_dim={} existing_tensor_idx={}\n", cur_dim, existing_tensor_idx);
// 
//                 // adjust next and prev pointers for existing_tensor and min_side_tensor in other dimensions
//                 for (int j = 0; j < dom_dim_; j++)
//                 {
//                     if (j == cur_dim)
//                         continue;
//                     for (int i = 0; i < existing_tensor.next[j].size(); i++)
//                         if (intersect_all_dims(min_side_tensor, tensor_prods[existing_tensor.next[j][i]], cur_dim))
//                         {
//                             fmt::print(stderr, "10:\n");
//                             min_side_tensor.next[j].push_back(existing_tensor.next[j][i]);
//                             tensor_prods[existing_tensor.next[j][i]].prev[j].push_back(min_side_tensor_idx);
//                         }
//                 }
// 
//                 tensor_prods.push_back(min_side_tensor);
// 
//                 // check if the knot mins, maxs of the existing or added tensor match the original new tensor
//                 if ( (min_side_tensor.knot_mins == new_tensor.knot_mins && min_side_tensor.knot_maxs == new_tensor.knot_maxs) ||
//                      (existing_tensor.knot_mins == new_tensor.knot_mins && existing_tensor.knot_maxs == new_tensor.knot_maxs) )
//                     knots_match = true;
// 
//                 return true;
//             }
//             return false;
//         }

        // check if intersection exists in all dimensions between knot_mins, knot_maxs of two tensors
        // skip dimension skip_dim
        bool intersect_all_dims(TensorProduct&          new_tensor,     // new tensor product to be added
                                TensorProduct&          existing_tensor,// existing tensor product
                                int                     skip_dim)       // skip checking in this dimension
        {
            for (int j = 0; j < dom_dim_; j++)
            {
                if (j == skip_dim)
                    continue;

                // there is no intersection in at least one of the other dimensions
                if( (new_tensor.knot_mins[j] < existing_tensor.knot_mins[j] || new_tensor.knot_mins[j] > existing_tensor.knot_maxs[j]) &&
                    (new_tensor.knot_maxs[j] > existing_tensor.knot_maxs[j] || new_tensor.knot_maxs[j] < existing_tensor.knot_mins[j]) )
                    return false;
            }

            // debug
//             fmt::print(stderr, "skip_dim={} new_tensor.knot_mins=[{} {}] new_tensor.knot_maxs=[{} {}]\n",
//                     skip_dim, new_tensor.knot_mins[0], new_tensor.knot_mins[1], new_tensor.knot_maxs[0], new_tensor.knot_maxs[1]);

            return true;
        }

        // checks if a_mins, maxs are a subset of b_mins, maxs
        bool subset(const vector<size_t>& a_mins,
                    const vector<size_t>& a_maxs,
                    const vector<size_t>& b_mins,
                    const vector<size_t>& b_maxs)
        {
            // check that sizes are identical
            size_t a_size = a_mins.size();
            if (a_size != a_maxs.size() || a_size != b_mins.size() || a_size != b_maxs.size())
            {
                fmt::print(stderr, "Error, size mismatch in subset()\n");
                abort();
            }

            // check subset condition
            for (auto i = 0; i < a_size; i++)
                if (a_mins[i] < b_mins[i] || a_maxs[i] > b_maxs[i])
                        return false;

            // debug
//             fmt::print(stderr, "[{} {} : {} {}] is a subset of [{} {} : {} {}]\n",
//                     a_mins[0], a_mins[1], a_maxs[0], a_maxs[1], b_mins[0], b_mins[1], b_maxs[0], b_maxs[1]);

            return true;
        }

        void print() const
        {
            // all_knots
            for (int i = 0; i < dom_dim_; i++)
            {
                fmt::print(stderr, "all_knots[dim {}]: ", i);
                for (const T& k : all_knots[i])
                    fmt::print(stderr, "{} ", k);
                fmt::print(stderr, "\n");
            }
            fmt::print(stderr, "\n");

            fmt::print(stderr, "T-mesh has {} tensor products\n\n", tensor_prods.size());

            // tensor products
            for (auto j = 0; j < tensor_prods.size(); j++)
            {
                const TensorProduct& t = tensor_prods[j];
                fmt::print(stderr, "tensor_prods[{}]:\n", j);

                fmt::print(stderr, "[ ");
                for (int i = 0; i < dom_dim_; i++)
                    fmt::print(stderr,"{} ", t.knot_mins[i]);
                fmt::print(stderr, "] : ");

                fmt::print(stderr, "[ ");
                for (int i = 0; i < dom_dim_; i++)
                    fmt::print(stderr,"{} ", t.knot_maxs[i]);
                fmt::print(stderr, "]\n");

                fmt::print(stderr, "next tensors [ ");
                for (int i = 0; i < dom_dim_; i++)
                {
                    fmt::print(stderr, "[ ");
                    for (const size_t& n : t.next[i])
                        fmt::print(stderr, "{} ", n);
                    fmt::print(stderr, "] ");
                    fmt::print(stderr," ");
                }
                fmt::print(stderr, "]\n");

                fmt::print(stderr, "previous tensors [ ");
                for (int i = 0; i < dom_dim_; i++)
                {
                    fmt::print(stderr, "[ ");
                    for (const size_t& n : t.prev[i])
                        fmt::print(stderr, "{} ", n);
                    fmt::print(stderr, "] ");
                    fmt::print(stderr," ");
                }

                fmt::print(stderr, "]\n\n");
            }
            fmt::print(stderr, "\n");
        }

    };
}

#endif
