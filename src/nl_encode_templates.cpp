// float version of function declarations

template
mfa::
NL_Encoder<float>::
NL_Encoder(MFA<float>& mfa_);

template
mfa::
NL_Encoder<float>::
~NL_Encoder();

template
void
mfa::
NL_Encoder<float>::
Encode(float err_limit);       // maximum allowable normalized error

// double version of function declarations

template
mfa::
NL_Encoder<double>::
NL_Encoder(MFA<double>& mfa_);

template
mfa::
NL_Encoder<double>::
~NL_Encoder();

template
void
mfa::
NL_Encoder<double>::
Encode(double err_limit);       // maximum allowable normalized error
