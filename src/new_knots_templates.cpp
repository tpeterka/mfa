// float versions of function declarations

template
mfa::
NewKnots<float>::
NewKnots(MFA<float>& mfa_);

template
mfa::
NewKnots<float>::
~NewKnots();

template
bool
mfa::
NewKnots<float>::
NewKnots_full(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<float>& new_knots,                   // new knots (1st dim changes fastest)
        float          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<float>::
NewKnots_curve1(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<float>& new_knots,                   // new knots (1st dim changes fastest)
        float          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<float>::
NewKnots_curve(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<float>& new_knots,                   // new knots (1st dim changes fastest)
        float          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<float>::
NewKnots_hybrid(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<float>& new_knots,                   // new knots (1st dim changes fastest)
        float          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<float>::
ErrorSpans(
        VectorXi&      nnew_knots,      // number of new knots in each dim
        vector<float>& new_knots,       // new knots (1st dim changes fastest)
        float          err_limit,       // max. allowed error
        int            iter);           // iteration number

template
void
mfa::
NewKnots<float>::
SplitSpan(
        size_t         si,              // id of span to split
        VectorXi&      nnew_knots,      // number of new knots in each dim
        vector<float>& new_knots,       // new knots (1st dim changes fastest)
        int            iter,            // iteration number
        vector<bool>&  split_spans);    // spans that have already been split in this iteration

// double versions of function declarations

template
mfa::
NewKnots<double>::
NewKnots(MFA<double>& mfa_);

template
mfa::
NewKnots<double>::
~NewKnots();

template
bool
mfa::
NewKnots<double>::
NewKnots_full(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<double>& new_knots,                   // new knots (1st dim changes fastest)
        double          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<double>::
NewKnots_curve1(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<double>& new_knots,                   // new knots (1st dim changes fastest)
        double          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<double>::
NewKnots_curve(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<double>& new_knots,                   // new knots (1st dim changes fastest)
        double          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<double>::
NewKnots_hybrid(
        VectorXi&      nnew_knots,                  // number of new knots in each dim
        vector<double>& new_knots,                   // new knots (1st dim changes fastest)
        double          err_limit,                   // max allowable error
        int            iter);                       // iteration number of caller (for debugging)

template
bool
mfa::
NewKnots<double>::
ErrorSpans(
        VectorXi&      nnew_knots,      // number of new knots in each dim
        vector<double>& new_knots,       // new knots (1st dim changes fastest)
        double          err_limit,       // max. allowed error
        int            iter);           // iteration number

template
void
mfa::
NewKnots<double>::
SplitSpan(
        size_t         si,              // id of span to split
        VectorXi&      nnew_knots,      // number of new knots in each dim
        vector<double>& new_knots,       // new knots (1st dim changes fastest)
        int            iter,            // iteration number
        vector<bool>&  split_spans);    // spans that have already been split in this iteration

