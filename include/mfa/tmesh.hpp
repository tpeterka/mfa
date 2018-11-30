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

            // check for intersection of the new tensor with existing tensors
            for (int k = 0; k < dom_dim_; k++)          // for all domain dimensions
            {
                // check knot_mins[cur_dim] and knot_maxs[cur_dim] against the existing tensors
                for (int i = 0; i < 2; i++)         // i = 0: knot_mins; i = 1: knot maxs
                {
                    do
                    {
                        vec_grew = false;
                        for (auto j = 0; j < tensor_prods.size(); j++)
                        {
                            bool knots_match;           // intersect resulted in a tensor with same knot mins, maxs as tensor to be added

                            // check knot_mins[cur_dim]
                            if (i == 0 && (vec_grew = intersect(new_tensor, j, k, knots_match, true)) && vec_grew)
                            {
                                // debug
                                fmt::print(stderr, "1:\n");
                                print();

                                if (knots_match)
                                    tensor_inserted = true;
                                break;  // adding a tensor invalidates iterator, start iteration over
                            }
                            // check knot_maxs[cur_dim]
                            else if (i == 1 && (vec_grew = intersect(new_tensor, j, k, knots_match, false)) && vec_grew)
                            {
                                // debug
                                fmt::print(stderr, "2:\n");
                                print();

                                if (knots_match)
                                    tensor_inserted = true;
                                break;
                            }
                        }
                    } while (vec_grew);   // keep checking until no more tensors are added
                }
            }                                       // for all domain dimensions

            // add the tensor
            // TODO: update next and prev pointers based on intersections found
            if (!tensor_inserted)
                tensor_prods.push_back(new_tensor);
        }

        // intersect in one dimension a new tensor product with an existing tensor product, if the intersection exists
        // returns true if intersection found (and the vector of tensor products grew as a result of the intersection, ie, an existing tensor was split into two)
        // sets knots_match to true if during the course of intersecting, one of the tensors in tensor_prods was added or modified to match the new tensor
        // ie, the caller should not add the tensor later if knots_match
        bool intersect(TensorProduct&           new_tensor,             // new tensor product to be inserted
                       int                      existing_tensor_idx,    // index in tensor_prods of existing tensor
                       int                      cur_dim,                // current dimension to intersect
                       bool&                    knots_match,            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
                       bool                     min)                    // intersect with knot_mins[cur_dim] or knot_maxs[cur_dim]
        {
            knots_match                     = false;
            size_t knot_idx                 = (min ? new_tensor.knot_mins[cur_dim] : new_tensor.knot_maxs[cur_dim]);
            TensorProduct& existing_tensor  = tensor_prods[existing_tensor_idx];


            if (knot_idx > existing_tensor.knot_mins[cur_dim] && knot_idx < existing_tensor.knot_maxs[cur_dim] &&   // there is an intersection in the current dim and
                intersect_all_dims(new_tensor, existing_tensor, cur_dim, true))                                     // there is intersection in at least one other dimension
            {
                vector<size_t> temp_maxs    = existing_tensor.knot_maxs;
                temp_maxs[cur_dim]          = knot_idx;
                size_t new_tensor_idx       = tensor_prods.size();                  // index of new tensor to be added
                vector<size_t> temp_mins    = existing_tensor.knot_mins;
                temp_mins[cur_dim]          = knot_idx;
                // split existing_tensor at the knot index knot_idx
                // existing_tensor is modified to be the min. side of the previous existing_tensor
                // a new max_side_tensor is appended to be the max. side of existing_tensor
                // as long as doing so would not create a tensor that is a subset of
                // knot_mins, knot_maxs (covered by new_tensor being inserted)
                if (!subset(existing_tensor.knot_mins, temp_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
                    return new_max_side(new_tensor, existing_tensor_idx, cur_dim, knots_match);

                // split existing_tensor at the knot index knot_idx
                // existing_tensor is modified to be the max. side of the previous existing_tensor
                // a new min_side_tensor is appended to be the max. side of existing_tensor
                // as long as doing so would not create a tensor that is a subset of
                // knot_mins, knot_maxs (covered by new_tensor being inserted)
                else if (!subset(temp_mins, existing_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
                    return new_min_side(new_tensor, existing_tensor_idx, cur_dim, knots_match);
            }
            return false;
        }

        // split existing tensor product creating extra tensor on minimum side of current dimension
        bool new_min_side(TensorProduct&      new_tensor,             // new tensor product to be inserted
                          int                 existing_tensor_idx,    // index in tensor_prods of existing tensor
                          int                 cur_dim,                // current dimension to intersect
                          bool&               knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {

            // intialize a new max_side_tensor for the max. side of the existing_tensor
            TensorProduct max_side_tensor;
            min_side_tensor.next.resize(dom_dim_);
            min_side_tensor.prev.resize(dom_dim_);
            min_side_tensor.knot_mins            = existing_tensor.knot_mins;
            min_side_tensor.knot_maxs            = existing_tensor.knot_maxs;
            min_side_tensor.knot_mins[cur_dim]   = knot_idx;

            existing_tensor.knot_mins[cur_dim]      = knot_idx;

            // check if tensor will be added before adding a next pointer to it
            if (!subset(min_side_tensor.knot_mins, min_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {

                // adjust next and prev pointers for existing_tensor and min_side_tensor in the current dimension
                if (existing_tensor.prev[cur_dim].size())
                {
                    fmt::print(stderr, "7:\n");
                    min_side_tensor.prev[cur_dim].push_back(existing_tensor.prev[cur_dim].back());   // TODO: is the correct pointer at the back, or does it have to be found?
                    existing_tensor.prev[cur_dim].back()    = new_tensor_idx;                   // TODO: is the correct pointer at the back, or does it have to be found?
                }
                else
                {
                    existing_tensor.prev[cur_dim].push_back(new_tensor_idx);
                    fmt::print(stderr, "8: cur_dim={} new_tensor_idx={}\n", cur_dim, new_tensor_idx);
                }
                min_side_tensor.next[cur_dim].push_back(existing_tensor_idx);
                fmt::print(stderr, "9: cur_dim={} existing_tensor_idx={}\n", cur_dim, existing_tensor_idx);

                // adjust next and prev pointers for existing_tensor and min_side_tensor in other dimensions
                for (int j = 0; j < dom_dim_; j++)
                {
                    if (j == cur_dim)
                        continue;
                    for (int i = 0; i < existing_tensor.next[j].size(); i++)
                    {
                        if (intersect_all_dims(new_tensor, tensor_prods[existing_tensor.next[j][i]], cur_dim, false))
                        {
                            fmt::print(stderr, "10:\n");
                            min_side_tensor.next[j].push_back(existing_tensor.next[j][i]);
                            tensor_prods[existing_tensor.next[j][i]].prev[j].push_back(new_tensor_idx);
                        }
                    }
                }

            }

            // check if the knot mins, maxs of the modified tensor match the original new tensor
            if (existing_tensor.knot_mins == new_tensor.knot_mins && existing_tensor.knot_maxs == new_tensor.knot_maxs)
                knots_match = true;

            // append the tensor as long as doing so would not create a tensor that is a subset of knot_mins, knot_maxs (covered by new tensor being inserted)
            if (!subset(min_side_tensor.knot_mins, min_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {
                tensor_prods.push_back(min_side_tensor);
                // check if the knot mins, maxs of the added tensor match the original new tensor
                if (min_side_tensor.knot_mins == new_tensor.knot_mins && min_side_tensor.knot_maxs == new_tensor.knot_maxs)
                    knots_match = true;
                return true;
            }
            return false;
        }

        // split existing tensor product creating extra tensor on maximum side of current dimension
        bool new_max_side(TensorProduct&      new_tensor,             // new tensor product to be inserted
                          int                 existing_tensor_idx,    // index in tensor_prods of existing tensor
                          int                 cur_dim,                // current dimension to intersect
                          bool&               knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            // intialize a new max_side_tensor for the max. side of the existing_tensor
            TensorProduct max_side_tensor;
            max_side_tensor.next.resize(dom_dim_);
            max_side_tensor.prev.resize(dom_dim_);
            max_side_tensor.knot_mins            = existing_tensor.knot_mins;
            max_side_tensor.knot_maxs            = existing_tensor.knot_maxs;
            max_side_tensor.knot_mins[cur_dim]   = knot_idx;

            existing_tensor.knot_maxs[cur_dim] = knot_idx;

            // check if tensor will be added before adding a next pointer to it
            if (!subset(max_side_tensor.knot_mins, max_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {
                // adjust next and prev pointers for existing_tensor and max_side_tensor in the current dimension
                if (existing_tensor.next[cur_dim].size())
                {
                    fmt::print(stderr, "3:\n");
                    max_side_tensor.next[cur_dim].push_back(existing_tensor.next[cur_dim].back());  // TODO: is the correct pointer at the back, or does it have to be found?
                    existing_tensor.next[cur_dim].back()    = new_tensor_idx;                       // TODO: is the correct pointer at the back, or does it have to be found?
                }
                else
                {
                    existing_tensor.next[cur_dim].push_back(new_tensor_idx);
                    fmt::print(stderr, "4: cur_dim={} new_tensor_idx={}\n", cur_dim, new_tensor_idx);
                }
                max_side_tensor.prev[cur_dim].push_back(existing_tensor_idx);
                fmt::print(stderr, "5: cur_dim={} existing_tensor_idx={}\n", cur_dim, existing_tensor_idx);

                // adjust next and prev pointers for existing_tensor and max_side_tensor in other dimensions
                for (int j = 0; j < dom_dim_; j++)
                {
                    if (j == cur_dim)
                        continue;
                    for (int i = 0; i < existing_tensor.next[j].size(); i++)
                    {
                        if (intersect_all_dims(new_tensor, tensor_prods[existing_tensor.next[j][i]], cur_dim, false))
                        {
                            fmt::print(stderr, "6:\n");
                            max_side_tensor.next[j].push_back(existing_tensor.next[j][i]);
                            tensor_prods[existing_tensor.next[j][i]].prev[j].push_back(new_tensor_idx);
                        }
                    }
                }

            }

            // check if the knot mins, maxs of the modified tensor match the original new tensor
            if (existing_tensor.knot_mins == new_tensor.knot_mins && existing_tensor.knot_maxs == new_tensor.knot_maxs)
                knots_match = true;

            // append the tensor as long as doing so would not create a tensor that is a subset of knot_mins, knot_maxs (covered by new tensor being inserted)
            if (!subset(max_side_tensor.knot_mins, max_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {
                tensor_prods.push_back(max_side_tensor);
                // check if the knot mins, maxs of the added tensor match the original new tensor
                if (max_side_tensor.knot_mins == new_tensor.knot_mins && max_side_tensor.knot_maxs == new_tensor.knot_maxs)
                    knots_match = true;
                return true;
            }
            return false;
        }

        // check if intersection exists in all dimensions between knot_mins, knot_maxs of two tensors
        // skip dimension skip_dim
        bool intersect_all_dims(TensorProduct&          new_tensor,     // new tensor product to be added
                                TensorProduct&          existing_tensor,// existing tensor product
                                int                     skip_dim,       // skip checking in this dimension
                                bool                    equality)       // equality counts as intersecting
        {
            for (int j = 0; j < dom_dim_; j++)
            {
                if (j == skip_dim)
                    continue;

                // there is no intersection in at least one of the other dimensions
                if (       equality &&
                          !(new_tensor.knot_mins[j] >= existing_tensor.knot_mins[j] && new_tensor.knot_mins[j] < existing_tensor.knot_maxs[j]) &&
                          !(new_tensor.knot_maxs[j] <= existing_tensor.knot_maxs[j] && new_tensor.knot_maxs[j] > existing_tensor.knot_mins[j]) )
                    return false;

                else if ( !equality &&
                          !(new_tensor.knot_mins[j] > existing_tensor.knot_mins[j] && new_tensor.knot_mins[j] < existing_tensor.knot_maxs[j]) &&
                          !(new_tensor.knot_maxs[j] < existing_tensor.knot_maxs[j] && new_tensor.knot_maxs[j] > existing_tensor.knot_mins[j]) )
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

                fmt::print(stderr, "knot mins [ ");
                for (int i = 0; i < dom_dim_; i++)
                    fmt::print(stderr,"{} ", t.knot_mins[i]);
                fmt::print(stderr, "]\n");

                fmt::print(stderr, "knot maxs [ ");
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
