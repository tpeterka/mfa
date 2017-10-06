template
mfa::
Encoder<float>::
Encoder(MFA<float>& mfa_);

template
mfa::
Encoder<float>::
~Encoder();

template
void
mfa::
Encoder<float>::
Encode();

template
void
mfa::
Encoder<float>::
AdaptiveEncode(float err_limit);               // maximum allowable normalized error

template
void
mfa::
Encoder<float>::
RHS(
        int       cur_dim,  // current dimension
        MatrixXf& N,        // matrix of basis function coefficients
        MatrixXf& R,        // (output) residual matrix allocated by caller
        int       ko = 0,   // optional index of starting knot
        int       po = 0,   // optional index of starting parameter
        int       co = 0);  // optional index of starting domain pt in current curve

template
void
mfa::
Encoder<float>::
RHS(
        int       cur_dim,  // current dimension
        MatrixXf& in_pts,   // input points (not the default domain stored in the mfa)
        MatrixXf& N,        // matrix of basis function coefficients
        MatrixXf& R,        // (output) residual matrix allocated by caller
        int       ko = 0,   // optional index of starting knot
        int       po = 0,   // optional index of starting parameter
        int       co = 0,   // optional index of starting input pt in current curve
        int       cs = 1);  // optional stride of input pts in current curve

template
void
mfa::
Encoder<float>::
Quants(
        VectorXi& n,          // (output) number of control point spans in each dim
        VectorXi& m);         // (output) number of input data point spans in each dim

template
void
mfa::
Encoder<float>::
CtrlCurve(
        MatrixXf& N,           // basis functions for current dimension
        MatrixXf& NtN,         // N^t * N
        MatrixXf& R,           // residual matrix for current dimension and curve
        MatrixXf& P,           // solved points for current dimension and curve
        VectorXi& n,           // number of control point spans in each dimension
        size_t    k,           // current dimension
        size_t    co,          // starting ofst for reading domain pts
        size_t    cs,          // stride for reading domain points
        size_t    to,          // starting ofst for writing control pts
        MatrixXf& temp_ctrl0,  // first temporary control points buffer
        MatrixXf& temp_ctrl1); // second temporary control points buffer

template
void
mfa::
Encoder<float>::
CopyCtrl(
        MatrixXf& P,          // solved points for current dimension and curve
        VectorXi& n,          // number of control point spans in each dimension
        int       k,          // current dimension
        size_t    co,         // starting offset for reading domain points
        size_t    cs,         // stride for reading domain points
        size_t    to,         // starting offset for writing control points
        MatrixXf& temp_ctrl0, // first temporary control points buffer
        MatrixXf& temp_ctrl1); // second temporary control points buffer

template
void
mfa::
Encoder<float>::
CopyCtrl(
        MatrixXf& P,          // solved points for current dimension and curve
        VectorXi& n,          // number of control point spans in each dimension
        int       k,          // current dimension
        size_t    co,         // starting offset for reading domain points
        MatrixXf& temp_ctrl); // temporary control points buffer

template
int
mfa::
Encoder<float>::
ErrorCurve(
        size_t       k,             // current dimension
        size_t       co,            // starting ofst for reading domain pts
        MatrixXf&    ctrl_pts,      // control points
        VectorXf&    weights,       // weights associated with control points
        float        err_limit);    // max allowable error

template
int
mfa::
Encoder<float>::
ErrorCurve(
        size_t       k,             // current dimension
        size_t       co,            // starting ofst for reading domain pts
        MatrixXf&    ctrl_pts,      // control points
        VectorXf&    weights,       // weights associated with control points
        set<int>&    err_spans,     // spans with error greater than err_limit
        float        err_limit);    // max allowable error

template
void
mfa::
Encoder<float>::
ErrorCurve(
        size_t         k,           // current dimension
        size_t         co,          // starting ofst for reading domain pts
        MatrixXf&      ctrl_pts,    // control points
        VectorXf&      weights,     // weights associated with control points
        VectorXi&      nnew_knots,  // number of new knots
        vector<float>& new_knots,   // new knots
        float          err_limit);  // max allowable error

template
int
mfa::
Encoder<float>::
ErrorCurve(
        size_t       k,             // current dimension
        size_t       co,            // starting ofst for reading domain pts
        size_t       to,            // starting ofst for reading control pts
        set<int>&    err_spans,     // spans with error greater than err_limit
        float        err_limit);    // max allowable error

template
int
mfa::
Encoder<float>::
ErrorCtrlCurve(
        size_t       k,             // current dimension
        size_t       to,            // starting ofst for reading control pts
        set<int>&    err_spans,     // spans with error greater than err_limit
        float        err_limit);    // max allowable error


