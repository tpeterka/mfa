// float version of function declarations

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
Encode(bool weighted);              // solve for and use weights

template
void
mfa::
Encoder<float>::
AdaptiveEncode(
        float err_limit,                // maximum allowable normalized error
        bool  weighted,                 // solve for and use weights
        int   max_rounds);              // (optional) maximum number of rounds

template
void
mfa::
Encoder<float>::
RHS(
        int       cur_dim,  // current dimension
        MatrixXf& N,        // matrix of basis function coefficients
        MatrixXf& R,        // (output) residual matrix allocated by caller
        VectorXf& weights,  // precomputed weights for n + 1 control points on this curve
        int       ko,       // index of starting knot
        int       po,       // index of starting parameter
        int       co);      // index of starting domain pt in current curve

template
void
mfa::
Encoder<float>::
RHS(
        int       cur_dim,  // current dimension
        MatrixXf& in_pts,   // input points (not the default domain stored in the mfa)
        MatrixXf& N,        // matrix of basis function coefficients
        MatrixXf& R,        // (output) residual matrix allocated by caller
        VectorXf& weights,  // precomputed weights for n + 1 control points on this curve
        int       ko,       // index of starting knot
        int       po,       // index of starting parameter
        int       co,       // index of starting input pt in current curve
        int       cs);      // stride of input pts in current curve

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
        MatrixXf& NtN,         // Nt * N
        MatrixXf& R,           // residual matrix for current dimension and curve
        MatrixXf& P,           // solved points for current dimension and curve
        size_t    k,           // current dimension
        size_t    co,          // starting ofst for reading domain pts
        size_t    cs,          // stride for reading domain points
        size_t    to,          // starting ofst for writing control pts
        MatrixXf& temp_ctrl0,  // first temporary control points buffer
        MatrixXf& temp_ctrl1,  // second temporary control points buffer
        int       curve_id,    // debugging
        bool      weighted);   // solve for and use weights

template
void
mfa::
Encoder<float>::
CopyCtrl(
        MatrixXf& P,          // solved points for current dimension and curve
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

// DEPRECATED
#if 0

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

#endif

// double version of function declarations

template
mfa::
Encoder<double>::
Encoder(MFA<double>& mfa_);

template
mfa::
Encoder<double>::
~Encoder();

template
void
mfa::
Encoder<double>::
Encode(bool weighted);              // solve for and use weights

template
void
mfa::
Encoder<double>::
AdaptiveEncode(
        double err_limit,           // maximum allowable normalized error
        bool   weighted,            // solve for and use weights
        int    max_rounds);         // (optional) maximum number of rounds

template
void
mfa::
Encoder<double>::
RHS(
        int       cur_dim,  // current dimension
        MatrixXd& N,        // matrix of basis function coefficients
        MatrixXd& R,        // (output) residual matrix allocated by caller
        VectorXd& weights,  // precomputed weights for n + 1 control points on this curve
        int       ko,       // index of starting knot
        int       po,       // index of starting parameter
        int       co);      // index of starting domain pt in current curve

template
void
mfa::
Encoder<double>::
RHS(
        int       cur_dim,  // current dimension
        MatrixXd& in_pts,   // input points (not the default domain stored in the mfa)
        MatrixXd& N,        // matrix of basis function coefficients
        MatrixXd& R,        // (output) residual matrix allocated by caller
        VectorXd& weights,  // precomputed weights for n + 1 control points on this curve
        int       ko,       // index of starting knot
        int       po,       // index of starting parameter
        int       co,       // index of starting input pt in current curve
        int       cs);      // stride of input pts in current curve

template
void
mfa::
Encoder<double>::
Quants(
        VectorXi& n,          // (output) number of control point spans in each dim
        VectorXi& m);         // (output) number of input data point spans in each dim

template
void
mfa::
Encoder<double>::
CtrlCurve(
        MatrixXd& N,           // basis functions for current dimension
        MatrixXd& NtN,         // Nt * N
        MatrixXd& R,           // residual matrix for current dimension and curve
        MatrixXd& P,           // solved points for current dimension and curve
        size_t    k,           // current dimension
        size_t    co,          // starting ofst for reading domain pts
        size_t    cs,          // stride for reading domain points
        size_t    to,          // starting ofst for writing control pts
        MatrixXd& temp_ctrl0,  // first temporary control points buffer
        MatrixXd& temp_ctrl1,  // second temporary control points buffer
        int       curve_id,    // debugging
        bool      weighted);   // solve for and use weights

template
void
mfa::
Encoder<double>::
CopyCtrl(
        MatrixXd& P,          // solved points for current dimension and curve
        int       k,          // current dimension
        size_t    co,         // starting offset for reading domain points
        size_t    cs,         // stride for reading domain points
        size_t    to,         // starting offset for writing control points
        MatrixXd& temp_ctrl0, // first temporary control points buffer
        MatrixXd& temp_ctrl1); // second temporary control points buffer

template
void
mfa::
Encoder<double>::
CopyCtrl(
        MatrixXd& P,          // solved points for current dimension and curve
        int       k,          // current dimension
        size_t    co,         // starting offset for reading domain points
        MatrixXd& temp_ctrl); // temporary control points buffer

template
int
mfa::
Encoder<double>::
ErrorCurve(
        size_t       k,             // current dimension
        size_t       co,            // starting ofst for reading domain pts
        MatrixXd&    ctrl_pts,      // control points
        VectorXd&    weights,       // weights associated with control points
        double        err_limit);    // max allowable error

template
int
mfa::
Encoder<double>::
ErrorCurve(
        size_t       k,             // current dimension
        size_t       co,            // starting ofst for reading domain pts
        MatrixXd&    ctrl_pts,      // control points
        VectorXd&    weights,       // weights associated with control points
        set<int>&    err_spans,     // spans with error greater than err_limit
        double        err_limit);    // max allowable error

template
void
mfa::
Encoder<double>::
ErrorCurve(
        size_t         k,           // current dimension
        size_t         co,          // starting ofst for reading domain pts
        MatrixXd&      ctrl_pts,    // control points
        VectorXd&      weights,     // weights associated with control points
        VectorXi&      nnew_knots,  // number of new knots
        vector<double>& new_knots,   // new knots
        double          err_limit);  // max allowable error

//DEPRECATED
#if 0

template
int
mfa::
Encoder<double>::
ErrorCurve(
        size_t       k,             // current dimension
        size_t       co,            // starting ofst for reading domain pts
        size_t       to,            // starting ofst for reading control pts
        set<int>&    err_spans,     // spans with error greater than err_limit
        double        err_limit);    // max allowable error

template
int
mfa::
Encoder<double>::
ErrorCtrlCurve(
        size_t       k,             // current dimension
        size_t       to,            // starting ofst for reading control pts
        set<int>&    err_spans,     // spans with error greater than err_limit
        double        err_limit);    // max allowable error

#endif
