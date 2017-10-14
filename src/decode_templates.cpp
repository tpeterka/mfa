// float version of function declarations

template
mfa::
Decoder<float>::
Decoder(MFA<float>& mfa_);

template
mfa::
Decoder<float>::
~Decoder();

template
void
mfa::
Decoder<float>::
Decode(MatrixXf& approx);         // (output) points (1st dim changes fastest)

template
void
mfa::
Decoder<float>::
VolPt(VectorXf& param,            // parameter value in each dim. of desired point
        VectorXf& out_pt);          // (output) point

template
void
mfa::
Decoder<float>::
CurvePt(
        int       cur_dim,              // current dimension
        float     param,                // parameter value of desired point
        size_t    co,                   // offset to start of control points for this curve
        VectorXf& out_pt);               // (output) point

// DEPRECATED
#if 0
template
void
mfa::
Decoder<float>::
CurvePt(
        int       cur_dim,              // current dimension
        float     param,                // parameter value of desired point
        MatrixXf& temp_ctrl,            // temporary control points
        VectorXf& out_pt,               // (output) point
        int       ko = 0);              // starting knot offset
#endif

template
void
mfa::
Decoder<float>::
CurvePt(
        int       cur_dim,              // current dimension
        float     param,                // parameter value of desired point
        MatrixXf& temp_ctrl,            // temporary control points
        VectorXf& temp_weights,         // weights associate with temporary control points
        VectorXf& out_pt,               // (output) point
        int       ko = 0);              // starting knot offset

// double version of function declarations

template
mfa::
Decoder<double>::
Decoder(MFA<double>& mfa_);

template
mfa::
Decoder<double>::
~Decoder();

template
void
mfa::
Decoder<double>::
Decode(MatrixXd& approx);         // (output) points (1st dim changes fastest)

template
void
mfa::
Decoder<double>::
VolPt(VectorXd& param,            // parameter value in each dim. of desired point
        VectorXd& out_pt);          // (output) point

template
void
mfa::
Decoder<double>::
CurvePt(
        int       cur_dim,              // current dimension
        double     param,                // parameter value of desired point
        size_t    co,                   // offset to start of control points for this curve
        VectorXd& out_pt);               // (output) point

// DEPRECATED
#if 0
template
void
mfa::
Decoder<double>::
CurvePt(
        int       cur_dim,              // current dimension
        double     param,                // parameter value of desired point
        MatrixXd& temp_ctrl,            // temporary control points
        VectorXd& out_pt,               // (output) point
        int       ko = 0);              // starting knot offset
#endif

template
void
mfa::
Decoder<double>::
CurvePt(
        int       cur_dim,              // current dimension
        double     param,                // parameter value of desired point
        MatrixXd& temp_ctrl,            // temporary control points
        VectorXd& temp_weights,         // weights associate with temporary control points
        VectorXd& out_pt,               // (output) point
        int       ko = 0);              // starting knot offset

