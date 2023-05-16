//--------------------------------------------------------------
// Logging/output utilities for MFA
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_LOGGING_HPP
#define _MFA_LOGGING_HPP

#include <string>
#include <iostream>
#include <mfa/types.hpp>

namespace mfa
{
    // Not designed for efficiency, should not be used in large loops
    template<typename T>
    string print_vec(const vector<T>& vec)
    {   
        stringstream ss;
        ss << "{";
        for (int i = 0; i < vec.size() - 1; i++)
        {
            ss << vec[i] << " ";
        }
        ss << vec[vec.size()-1] << "}";

        return ss.str();
    }

    // Not designed for efficiency, should not be used in large loops
    template<typename T>
    string print_vec(const VectorX<T>& vec)
    {
        stringstream ss;
        ss << "{";
        for (int i = 0; i < vec.size() - 1; i++)
        {
            ss << vec(i) << " ";
        }
        ss << vec.tail(1) << "}";

        return ss.str();
    }

    template<typename T>
    void print_bbox(const VectorX<T>& mins, const VectorX<T>& maxs, string label="Bounding")
    {
        if (mins.size() != maxs.size()) 
            fmt::print("{} Box: <invalid box>\n", label);
        
        fmt::print("{} Box:\n", label);
        for (int i = 0; i < mins.size(); i++)
        {
            fmt::print("  Dim {}: [{:< 5.5g}, {:< 5.5g}]\n", i, mins(i), maxs(i));
        } 
    }
}   // namespace mfa

#endif  //_MFA_LOGGING_HPP