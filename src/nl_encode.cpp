// --------------------------------------------------------------
// nonlinear encoder object
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <mfa/nl_encode.hpp>
#include <mfa/decode.hpp>
#include <iostream>
#include <cppoptlib/meta.h>

mfa::
NL_Encoder::
NL_Encoder(MFA& mfa_) :
    mfa(mfa_)
{
}

void
mfa::
NL_Encoder::
Encode()
{
    fprintf(stderr, "hello world\n");
}
