// float version of function declarations

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

// double version of function declarations

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


