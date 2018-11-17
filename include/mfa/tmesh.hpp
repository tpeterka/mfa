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

            // check for intersection of the new tensor with existing tensors
            for (int k = 0; k < dom_dim_; k++)          // for all domain dimensions
            {
                // check knot_mins[cur_dim] and knot_maxs[cur_dim] against the existing tensors
                for (int i = 0; i < 2; i++)         // i = 0: knot_mins; i = 1: knot maxs
                {
                    do
                    {
                        vec_grew = false;
                        for (TensorProduct& t : tensor_prods)
                        {
                            bool knots_match;           // intersect resulted in a tensor with same knot mins, maxs as tensor to be added

                            // check knot_mins[cur_dim]
                            if (i == 0 && (vec_grew = intersect(t, knot_mins, knot_maxs, k, knots_match, true)) && vec_grew)
                            {
                                // debug
//                                 fmt::print(stderr, "1:\n");
//                                 print();

                                if (knots_match)
                                    tensor_inserted = true;
                                break;  // adding a tensor invalidates iterator, start iteration over
                            }
                            // check knot_maxs[cur_dim]
                            else if (i == 1 && (vec_grew = intersect(t, knot_mins, knot_maxs, k, knots_match, false)) && vec_grew)
                            {
                                // debug
//                                 fmt::print(stderr, "2:\n");
//                                 print();

                                if (knots_match)
                                    tensor_inserted = true;
                                break;
                            }
                        }
                    } while (vec_grew);   // keep checking until no more tensors are added
                }
            }                                       // for all domain dimensions

            // add the tensor
            // TODO: update next pointers for next tensor based on intersections found
            if (!tensor_inserted)
            {
                TensorProduct tensor;
                tensor.next.resize(dom_dim_);
                tensor.knot_mins = knot_mins;
                tensor.knot_maxs = knot_maxs;
                tensor_prods.push_back(tensor);
            }
        }

        // intersect in one dimension a tensor product with a new knot index, if the intersection exists
        // returns true if intersection found (and a new tensor has been added)
        // sets knots_match to true if during the course of intersecting, one of the tensors in tensor_prods was added or modified to match the new tensor
        // ie, the caller should not add the tensor later if knots_match
        bool intersect(TensorProduct&           t,              // existing tensor product
                       const vector<size_t>&    knot_mins,      // indices in all_knots of min. corner of new tensor
                       const vector<size_t>&    knot_maxs,      // indices in all_knots of max. corner of new tensor
                       int                      cur_dim,        // current dimension to intersect
                       bool&                    knots_match,    // (output) interection resulted in a tensor whose knot mins, max match new tensor's
                       bool                     min)            // intersect with knot_mins[cur_dim] or knot_maxs[cur_dim]
        {
            knots_match = false;
            size_t knot_idx = (min ? knot_mins[cur_dim] : knot_maxs[cur_dim]);

            // debg
//             if (knot_idx > t.knot_mins[cur_dim] && knot_idx < t.knot_maxs[cur_dim])
//                 fmt::print(stderr, "Looking for intersection in cur_dim={} knot_idx={} knot_mins=[{} {}] knot_maxs=[{} {}] with t.knot_mins=[{} {}] t.knot_maxs=[{} {}]\n",
//                         cur_dim, knot_idx, knot_mins[0], knot_mins[1], knot_maxs[0], knot_maxs[1], t.knot_mins[0], t.knot_mins[1], t.knot_maxs[0], t.knot_maxs[1]);

            if (knot_idx > t.knot_mins[cur_dim] && knot_idx < t.knot_maxs[cur_dim] &&   // there is an intersection in the current dim and
                intersect_dims(t, knot_mins, knot_maxs, cur_dim))                          // there is intersection in at least one other dimension
            {
                // split t at the knot index knot_idx
                // tensor t is modified to be the min. side of the old tensor t
                // a new tensor is appended to be the max. side of the old tensor t

                // intialize a new tensor for the max. side of the old tensor t
                TensorProduct tensor;
                tensor.next.resize(dom_dim_);
                tensor.knot_mins            = t.knot_mins;
                tensor.knot_maxs            = t.knot_maxs;
                tensor.knot_mins[cur_dim]   = knot_idx;

                // modify the old tensor for the min. side as long as doing so would not create a tensor that is a subset of knot_mins, knot_maxs (covered by new tensor being inserted)
                vector<size_t> temp_maxs    = t.knot_maxs;
                temp_maxs[cur_dim]          = knot_idx;
                if (!subset(t.knot_mins, temp_maxs, knot_mins, knot_maxs))
                {
                    t.knot_maxs[cur_dim] = knot_idx;
                    // check if tensor will be added before adding a next pointer to it
                    if (!subset(tensor.knot_mins, tensor.knot_maxs, knot_mins, knot_maxs))
                        t.next[cur_dim].push_back(tensor_prods.size());
                    // check if the knot mins, maxs of the modified tensor match the original new tensor
                    if (t.knot_mins == knot_mins && t.knot_maxs == knot_maxs)
                        knots_match = true;

                    // append the tensor as long as doing so would not create a tensor that is a subset of knot_mins, knot_maxs (covered by new tensor being inserted)
                    if (!subset(tensor.knot_mins, tensor.knot_maxs, knot_mins, knot_maxs))
                    {
                        tensor_prods.push_back(tensor);
                        // check if the knot mins, maxs of the added tensor match the original new tensor
                        if (tensor.knot_mins == knot_mins && tensor.knot_maxs == knot_maxs)
                            knots_match = true;
                        return true;
                    }
                }

                // modify the old tensor for the max. side as long as doing so would not create a tensor that is a subset of knot_mins, knot_maxs (covered by new tensor being inserted)
                else
                {
                    // debug
//                     fmt::print(stderr, "subset: a_mins=[{} {}] a_maxs=[{} {}] b_mins=[{} {}] b_maxs=[{} {}]\n",
//                             t.knot_mins[0], t.knot_mins[1], temp_maxs[0], temp_maxs[1], knot_mins[0], knot_mins[1], knot_maxs[0], knot_maxs[1]);

                    vector<size_t> temp_mins    = t.knot_mins;
                    temp_mins[cur_dim]          = knot_idx;
                    if (!subset(temp_mins, t.knot_maxs, knot_mins, knot_maxs))
                    {
                        t.knot_mins[cur_dim] = knot_idx;
                        if (t.knot_mins == knot_mins && t.knot_maxs == knot_maxs)
                            knots_match = true;
                    }
                }
            }
            return false;
        }

        // check if intersection exists in any dimension between knot_mins, knot_maxs
        // and all other tensors
        // in this routine, equality counts as intersecting
        // skip dimension skip_dim
        bool intersect_dims(TensorProduct&          t,              // existing tensor product
                            const vector<size_t>&   knot_mins,      // indices in all_knots of min. corner of new tensor
                            const vector<size_t>&   knot_maxs,      // indices in all_knots of max. corner of new tensor
                            int                     skip_dim)       // skip checking in this dimension
        {
            for (int j = 0; j < dom_dim_; j++)
            {
                if (j == skip_dim)
                    continue;

                // there is no intersection in at least one of the other dimensions
                if ( !(knot_mins[j] >= t.knot_mins[j] && knot_mins[j] < t.knot_maxs[j]) &&
                     !(knot_maxs[j] <= t.knot_maxs[j] && knot_maxs[j] > t.knot_mins[j]) )
                    return false;
            }

            // debug
//             fmt::print(stderr, "skip_dim={} knot_mins=[{} {}] knot_maxs=[{} {}]\n",
//                     skip_dim, knot_mins[0], knot_mins[1], knot_maxs[0], knot_maxs[1]);

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
                    if (t.next[i].size())
                    {
                        fmt::print(stderr, "[ ");
                        for (const size_t& n : t.next[i])
                            fmt::print(stderr, "{} ", n);
                        fmt::print(stderr, "] ");
                    }
                    fmt::print(stderr," ");
                }
                fmt::print(stderr, "]\n\n");
            }
            fmt::print(stderr, "\n");
        }

    };
}

#endif
