//--------------------------------------------------------------
// mfa utilities
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _UTIL_HPP
#define _UTIL_HPP

#include <mfa/utilities/grid.hpp>
#include <mfa/utilities/iterator.hpp>
#include <mfa/utilities/logging.hpp>
#include <mfa/utilities/stats.hpp>

namespace mfa
{
    struct MFAError: public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

}   // namespace mfa

#endif

