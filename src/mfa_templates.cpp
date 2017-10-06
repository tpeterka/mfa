template
mfa::
MFA<float>::
MFA(
        VectorXi& p_,             // polynomial degree in each dimension
        VectorXi& ndom_pts_,      // number of input data points in each dim
        MatrixXf& domain_,        // input data points (1st dim changes fastest)
        MatrixXf& ctrl_pts_,      // (output, optional input) control points (1st dim changes fastest)
        VectorXi& nctrl_pts_,     // (output, optional input) number of control points in each dim
        VectorXf& weights_,       // (output, optional input) weights associated with control points
        VectorXf& knots_,         // (output) knots (1st dim changes fastest)
        float     eps_ = 1.0e-6); // minimum difference considered significant

template
mfa::
MFA<float>::
~MFA();

template
void
mfa::
MFA<float>::
Encode();

template
void
mfa::
MFA<float>::
FixedEncode(VectorXi& nctrl_pts_); // (output) number of control points in each dim

template
void
mfa::
MFA<float>::
AdaptiveEncode(
        float     err_limit,        // maximum allowable normalized error
        VectorXi& nctrl_pts_);      // (output) number of control points in each dim

template
void
mfa::
MFA<float>::
NonlinearEncode(
        float     err_limit,        // maximum allowable normalized error
        VectorXi& nctrl_pts_);      // (output) number of control points in each dim

template
void
mfa::
MFA<float>::
Decode(MatrixXf& approx);      // decode points

template
float
mfa::
MFA<float>::
Error(size_t idx);            // index of domain point where to compute error of mfa

template
float
mfa::
MFA<float>::
RangeError(size_t idx);       // index of domain point where to compute error of mfa

template
float
mfa::
MFA<float>::
NormalDistance(
        VectorXf& pt,              // point whose distance from domain is desired
        size_t    cell_idx);       // index of min. corner of cell in the domain

template
float
mfa::
MFA<float>::
CurveDistance(
        int       k,               // current dimension in direction of curve
        VectorXf& pt,              // point whose distance from domain is desired
        size_t    cell_idx);       // index of min. corner of cell in the domain

template
void
mfa::
MFA<float>::
KnotSpanDomains(
        VectorXi& span_mins,        // minimum domain points of all knot spans
        VectorXi& span_maxs);       // maximum domain points of all knot spans

template
int
mfa::
MFA<float>::
FindSpan(
        int   cur_dim,              // current dimension
        float u,                    // parameter value
        int   ko    = 0);           // optional index of starting knot

template
void
mfa::
MFA<float>::
BasisFuns(
        int       cur_dim,          // current dimension
        float     u,                // parameter value
        int       span,             // index of span in the knots vector containing u, relative to ko
        MatrixXf& N,                // matrix of (output) basis function values
        int       start_n,          // starting basis function N_{start_n} to compute
        int       end_n,            // ending basis function N_{end_n} to compute
        int       row);             // starting row index in N of result

template
void
mfa::
MFA<float>::
Params();

template
void
mfa::
MFA<float>::
DomainParams();

template
void
mfa::
MFA<float>::
Knots();

template
void
mfa::
MFA<float>::
UniformKnots();

template
void
mfa::
MFA<float>::
KnotSpanIndex();

template
void
mfa::
MFA<float>::
InsertKnots(
        VectorXi&      nnew_knots,  // number of new knots in each dim
        vector<float>& new_knots);  // new knots (1st dim changes fastest)

template
float
mfa::
MFA<float>::
InterpolateParams(
        int       cur_dim,          // curent dimension
        size_t    po,               // starting offset for params in cur. dim.
        size_t    ds,               // stride for domain pts in cuve in cur. dim.
        float     coord);           // target coordinate

template
void
mfa::
MFA<float>::
idx2ijk(
        size_t     idx,             // linear index
        VectorXi&  ijk);            // i,j,k,... indices in all dimensions

template
void
mfa::
MFA<float>::
ijk2idx(
        VectorXi&  ijk,             // i,j,k,... indices to all dimensions
        size_t&    idx);            // (output) linear index


