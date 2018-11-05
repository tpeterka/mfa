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

Struct TensorProduct
{
    vector<size_t> knot_mins;                   // indices into all_knots
    vector<size_t> knot_maxs;                   // indices into all_knots
    vector< vector <TensorProduct*> next;       // next[dim][next_tp]
};

namespace mfa
{
    template <typename T>                       // float or double
    class Tmesh
    {
        vector<vector<T>>       all_knots;      // all_knots[dimension][index]
        vector<TensorProduct>   tensor_prods;   // all tensor products

        void insert_tensor(const vector<size_t>&    knot_mins,      // indices in all_knots of min. corner of tensor to be inserted
                           const vector<size_t &    knot_maxs)      // indices in all_knots of max. corner
        {
            // empty tensor_prods: just insert tensor and return
            if (!tensor_prods.size())
            {
                TensorProduct tensor;
                tensor.knot_mins = knot_mins;
                tensor.knot_maxs = knot_maxs;
                tensor.next = NULL;
                tenso_prods.push_back(tensor);
                return;
            }

            // check for intersection of the new tensor with existing tensors
            for (k = 0; k < dom_dims; k++)      // for all domain dimensions
            {
                // check minimum side for intersection with other tensors

                // check maximum side for intersection with other tensors
            }
    };
}

#endif
