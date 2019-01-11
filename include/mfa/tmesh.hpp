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

template <typename T>
struct ControlPoints
{
    VectorXi                    nctrl_pts;          // number of control points in each domain dimension
    vector<MatrixX<T>>          ctrl_pts;           // NURBS control points in each cross-slice (1st dim changes fastest)
    vector<VectorX<T>>          weights;            // weights associated with control points in each cross-slice
};

template <typename T>
struct TensorProduct
{
    vector<size_t>              knot_mins;          // indices into all_knots
    vector<size_t>              knot_maxs;          // indices into all_knots
    ControlPoints<T>            ctrl_pts;           // control points
    vector< vector <size_t>>    next;               // next[dim][index of next tensor product]
    vector< vector <size_t>>    prev;               // prev[dim][index of previous tensor product]
    int                         level;              // refinement level
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
        MFA_Data<T>&                mfa_;               // mfa for this t-mesh

        Tmesh(int dom_dim, MFA_Data<T>& mfa) :
            dom_dim_(dom_dim), mfa_(mfa)                { all_knots.resize(dom_dim_); }

        // insert a knot into all_knots
        void insert_knot(int    dim,                // dimension of knot vector
                         size_t pos,                // new position in all_knots[dim] of inserted knot
                         int    level,              // refinement level of inserted knot
                         T      knot)               // knot value to be inserted
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

        // insert a tensor product into tensor_prods
        void insert_tensor(const vector<size_t>&    knot_mins,      // indices in all_knots of min. corner of tensor to be inserted
                           const vector<size_t>&    knot_maxs)      // indices in all_knots of max. corner
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
            new_tensor.ctrl_pts.nctrl_pts.resize(dom_dim_);
            new_tensor.ctrl_pts.ctrl_pts.resize(dom_dim_);
            new_tensor.ctrl_pts.weights.resize(dom_dim_);
            size_t tot_nctrl_pts = 1;
            if (!tensor_prods.size())
            {
                new_tensor.level = 0;

                // resize control points
                // level 0 has only one box of control points
                for (auto j = 0; j < dom_dim_; j++)
                    tot_nctrl_pts *= all_knots[j].size() - mfa_.p[j] - 1;
                new_tensor.ctrl_pts.ctrl_pts[0].resize(tot_nctrl_pts, mfa_.max_dim - mfa_.min_dim + 1);
                new_tensor.ctrl_pts.weights[0].resize(tot_nctrl_pts, mfa_.max_dim - mfa_.min_dim + 1);

            }
            else
            {
                new_tensor.level = tensor_prods.back().level + 1;

                // resize control points
                // levels > 0 have dom_dim_ boxes of control points, one box for each
                // hyperplane of dimension dom_dim_ - 1 crossing the tensor product
                for (auto i = 0; i < dom_dim_; i++)
                {
                    tot_nctrl_pts = 1;
                    for (auto j = 0; j < dom_dim_; j++)
                    {
                        if (j == i)
                            continue;
                        tot_nctrl_pts *= all_knots[j].size() - mfa_.p[j] - 1;
                    }
                    new_tensor.ctrl_pts.ctrl_pts[i].resize(tot_nctrl_pts, mfa_.max_dim - mfa_.min_dim + 1);
                    new_tensor.ctrl_pts.weights[i].resize(tot_nctrl_pts, mfa_.max_dim - mfa_.min_dim + 1);
                }
            }
            new_tensor.ctrl_pts.nctrl_pts.resize(dom_dim_);

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

            // the new tensor has either been added already or still needs to be added
            // either way, create reference to new tensor and get its index at the end of vector of tensor prods.
            size_t              new_tensor_idx;
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
//                     fmt::print(stderr, "final add: cur_dim={} new_tensor_idx={} checking existing_tensor_idx={}\n", j, new_tensor_idx, k);

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
        bool intersect(TensorProduct<T>&    new_tensor,             // new tensor product to be inserted
                       int                  existing_tensor_idx,    // index in tensor_prods of existing tensor
                       vector<int>&         split_side,             // whether min (-1) or max (1) or both (2) sides of
                                                                    // new tensor are inside existing tensor (one value for each dim.)
                       bool&                knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            knots_match                     = false;
            bool retval                     = false;
            size_t split_knot_idx;

            for (int k = 0; k < dom_dim_; k++)      // for all domain dimensions
            {
                if (!split_side[k])
                    continue;

                split_knot_idx                      = (split_side[k] == -1 ? new_tensor.knot_mins[k] : new_tensor.knot_maxs[k]);
                TensorProduct<T>& existing_tensor   = tensor_prods[existing_tensor_idx];
                vector<size_t> temp_maxs            = existing_tensor.knot_maxs;
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
                          int                   existing_tensor_idx,    // index in tensor_prods of existing tensor
                          int                   cur_dim,                // current dimension to intersect
                          size_t                knot_idx,               // knot index in current dim of split point
                          bool&                 knots_match)            // (output) interection resulted in a tensor whose knot mins, max match new tensor's
        {
            TensorProduct<T>& existing_tensor  = tensor_prods[existing_tensor_idx];

            // intialize a new max_side_tensor for the maximum side of the existing_tensor
            TensorProduct<T> max_side_tensor;
            max_side_tensor.next.resize(dom_dim_);
            max_side_tensor.prev.resize(dom_dim_);
            max_side_tensor.knot_mins           = existing_tensor.knot_mins;
            max_side_tensor.knot_maxs           = existing_tensor.knot_maxs;
            max_side_tensor.knot_mins[cur_dim]  = knot_idx;
            max_side_tensor.level               = existing_tensor.level;

            existing_tensor.knot_maxs[cur_dim]  = knot_idx;

            size_t max_side_tensor_idx          = tensor_prods.size();                  // index of new tensor to be added

            // check if tensor will be added before adding a next pointer to it
            if (!subset(max_side_tensor.knot_mins, max_side_tensor.knot_maxs, new_tensor.knot_mins, new_tensor.knot_maxs))
            {
                // adjust next and prev pointers for existing_tensor and max_side_tensor in the current dimension
                for (int i = 0; i < existing_tensor.next[cur_dim].size(); i++)
                {
                    if (adjacent(max_side_tensor, tensor_prods[existing_tensor.next[cur_dim][i]], cur_dim))
                    {
                        max_side_tensor.next[cur_dim].push_back(existing_tensor.next[cur_dim][i]);
                        vector<size_t>& prev    = tensor_prods[existing_tensor.next[cur_dim][i]].prev[cur_dim];
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

            // delete next and prev pointers of existing tensor that are no longer valid as a result of adding new max side
            delete_old_pointers(existing_tensor_idx);

            return false;
        }

        // delete pointers that are no longer valid as a result of adding a new max side tensor
        void delete_old_pointers(int existing_tensor_idx)               // index in tensor_prods of existing tensor
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
//                         fmt::print(stderr, "next tensor {} is no longer adjacent to existing_tensor {}\n",
//                                 existing_tensor.next[j][i], existing_tensor_idx);

                        // remove the prev pointer of the next tensor
                        vector<size_t>& prev    = tensor_prods[existing_tensor.next[j][i]].prev[j];
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
//                         fmt::print(stderr, "prev tensor {} is no longer adjacent to existing_tensor {}\n",
//                                 existing_tensor.prev[j][i], existing_tensor_idx);

                        // remove the next pointer of the prev tensor
                        vector<size_t>& next    = tensor_prods[existing_tensor.prev[j][i]].next[j];
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
//                 fmt::print(stderr, "adj 1: cur_dim={} retval=0 new_tensor=[{} {} : {} {}] existing_tensor=[{} {} : {} {}]\n",
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
//                     fmt::print(stderr, "adj 2: cur_dim={} retval=0 new_tensor=[{} {} : {} {}] existing_tensor=[{} {} : {} {}]\n",
//                             cur_dim, new_tensor.knot_mins[0], new_tensor.knot_mins[1], new_tensor.knot_maxs[0], new_tensor.knot_maxs[1],
//                             existing_tensor.knot_mins[0], existing_tensor.knot_mins[1], existing_tensor.knot_maxs[0], existing_tensor.knot_maxs[1]);

                    return 0;
                }
            }

            // debug
//             fmt::print(stderr, "adj: cur_dim={} retval={} new_tensor=[{} {} : {} {}] existing_tensor=[{} {} : {} {}]\n",
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
//             fmt::print(stderr, "cur_dim={} return=true new_tensor=[{} {} : {} {}] existing_tensor=[{} {} : {} {}]\n",
//                     cur_dim, new_tensor.knot_mins[0], new_tensor.knot_mins[1], new_tensor.knot_maxs[0], new_tensor.knot_maxs[1],
//                     existing_tensor.knot_mins[0], existing_tensor.kot_mins[1], existing_tensor.knot_maxs[0], existing_tensor.knot_maxs[1]);

            return true;
        }

        // checks if a_mins, maxs are a subset of b_mins, maxs
        // identical bounds counts as a subset (does not need to be proper subset)
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

        // in
        // checks if a point in index space is in a tensor product
        // in all dimensions except skip_dim (-1 = default, don't skip any dimensions)
        // for dimension i, moves pt[i] to the center of the t-mesh cell if degree[i] is even
        bool in(const vector<size_t>&   pt,
                const TensorProduct<T>& tensor,
                int                     skip_dim = -1)
        {
            for (auto i = 0; i < pt.size(); i++)
            {
                if (i == skip_dim)
                    continue;

                // move pt[i] to center of t-mesh cell if degree is even
                float fp = mfa_.p[i] % 2 ? pt[i] : pt[i] + 0.5;

                if (fp < float(tensor.knot_mins[i]) || fp > float(tensor.knot_maxs[i]))
                        return false;
            }

            return true;
        }

        // compute local knot vector in index space
        void local_knot_vector(const vector<size_t>&        anchor,             // knot indices of anchor for odd degree or
                                                                                // knot indices of start of rectangle containing anchor for even degree
                               vector<vector<size_t>>&      loc_knots)          // (output) local knot vector in index space
        {
            loc_knots.resize(dom_dim_);
            assert(anchor.size() == dom_dim_);

            // find most refined tensor product containing anchor and that level of refinement
            // levels monotonically nondecrease; hence, find last tensor product containing the anchor
            size_t max_j = 0;
            for (auto j = 0; j < tensor_prods.size(); j++)
                if (in(anchor, tensor_prods[j], -1))
                        max_j = j;
            size_t max_level = tensor_prods[max_j].level;

            // debug (hard-coded for 2d)
//             fmt::print(stderr, "anchor=[{} {}] cur_tensor={} cur_level={}\n", anchor[0], anchor[1], max_j, max_level);

            for (auto i = 0; i < dom_dim_; i++)
            {
                auto p = mfa_.p[i];                                         // degree of current dim.
                loc_knots[i].resize(p + 2);                                 // support of basis func. is p+2 by definition

                size_t start, min, max;
                if (p % 2)                                                  // odd degree
                {
                    start   = (p + 1) / 2;
                    min     = (p + 1) / 2;
                    max     = (p + 1) / 2;
                }
                else                                                        // even degree
                {
                    start   = p / 2;
                    min     = p / 2;
                    max     = p / 2 + 1;
                }
                loc_knots[i][start] = anchor[i];                            // start by anchoring the center of the knot vector
                size_t cur_knot_idx = loc_knots[i][start];
                size_t cur_tensor   = max_j;
                size_t cur_level    = max_level;

                // debug
//                 fmt::print(stderr, "dim={} (prev) cur_tensor={} cur_level={}\n", i, cur_tensor, cur_level);

                for (int j = 0; j < min; j++)                               // add 'min' more knots in minimum direction from the anchor
                {
                    bool done = false;
                    do
                    {
                        if (cur_knot_idx > 0)
                        {
                            if (cur_knot_idx - 1 < tensor_prods[cur_tensor].knot_mins[i])
                            {
                                // check which is the correct previous tensor
                                vector<size_t>& prev = tensor_prods[cur_tensor].prev[i];
                                size_t k;
                                for (k = 0; k < prev.size(); k++)
                                    if (in(anchor, tensor_prods[prev[k]], i))
                                        break;
                                if (k == prev.size())
                                {
                                    fmt::print(stderr, "Error: no valid previous tensor for knot vector\n");
                                    abort();
                                }
                                cur_tensor  = prev[k];
                                cur_level   = tensor_prods[cur_tensor].level;

                                // debug
//                                 fmt::print(stderr, "dim={} (prev) cur_tensor={} cur_level={}\n", i, cur_tensor, cur_level);
                            }
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
                        else
                        {
                            loc_knots[i][start - j - 1] = cur_knot_idx;         // repeat last knot as many times as needed
                            done = true;
                        }
                    } while (!done);
                }
                cur_knot_idx    = loc_knots[i][start];
                cur_tensor      = max_j;
                cur_level       = max_level;

                // debug
//                 fmt::print(stderr, "dim={} (next) cur_tensor={} cur_level={}\n", i, cur_tensor, cur_level);

                for (int j = 0; j < max; j++)                               // add 'max' more knots in maximum direction from the anchor
                {
                    bool done = false;
                    do
                    {
                        if (cur_knot_idx + 1 < all_knots[i].size())
                        {
                            if (cur_knot_idx + 1 > tensor_prods[cur_tensor].knot_maxs[i])
                            {
                                // check which is the correct next tensor
                                vector<size_t>& next = tensor_prods[cur_tensor].next[i];
                                size_t k;
                                for (k = 0; k < next.size(); k++)
                                    if (in(anchor, tensor_prods[next[k]], i))
                                        break;
                                if (k == next.size())
                                {
                                    fmt::print(stderr, "Error: no valid next tensor for knot vector\n");
                                    abort();
                                }
                                cur_tensor  = next[k];
                                cur_level   = tensor_prods[cur_tensor].level;

                                // debug
//                                 fmt::print(stderr, "dim={} (next) cur_tensor={} cur_level={}\n", i, cur_tensor, cur_level);
                            }
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
                        else
                        {
                            loc_knots[i][start + j + 1] = cur_knot_idx;         // repeat last knot as many times as needed
                            done = true;
                        }
                    } while (!done);
                }
            }
        }

        void print() const
        {
            // all_knots
            for (int i = 0; i < dom_dim_; i++)
            {
                fmt::print(stderr, "all_knots[dim {}]: ", i);
                for (auto j = 0; j < all_knots[i].size(); j++)
                    fmt::print(stderr, "{} (l{}) ", all_knots[i][j], all_knot_levels[i][j]);
                fmt::print(stderr, "\n");
            }
            fmt::print(stderr, "\n");

            fmt::print(stderr, "T-mesh has {} tensor products\n\n", tensor_prods.size());

            // tensor products
            for (auto j = 0; j < tensor_prods.size(); j++)
            {
                const TensorProduct<T>& t = tensor_prods[j];
                fmt::print(stderr, "tensor_prods[{}] level={}:\n", j, t.level);

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
