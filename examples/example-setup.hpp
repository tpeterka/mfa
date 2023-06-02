//--------------------------------------------------------------
// Helper functions to set up pre-defined examples
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef MFA_EX_SETUP_HPP
#define MFA_EX_SETUP_HPP

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <set>

#include "block.hpp"

using namespace std;

    // Set DIY Bounds for decomposition based on example
    // If the input is not an analytical signal, then dummy bounds are used
    // because we assume the example will be run on a single block
    void set_dom_bounds(Bounds<real_t>& dom_bounds, string input)
    {
        if (input == "sine" || input == "cosine" || input == "sinc" ||
            input == "psinc1" || input == "psinc2" || input == "psinc3")
        {
            for (int i = 0; i < dom_bounds.min.dimension(); i++)
            {
                dom_bounds.min[i] = -4.0 * M_PI;
                dom_bounds.max[i] =  4.0 * M_PI;
            }
        }
        else if (input == "ml")
        {
            for (int i = 0; i < dom_bounds.min.dimension(); i++)
            {
                dom_bounds.min[i] = -1.0;
                dom_bounds.max[i] =  1.0;
            }
        }
        else if (input == "f16")
        {
            dom_bounds.min = {-1, -1};
            dom_bounds.max = { 1,  1};
        }
        else if (input == "f17")
        {
            dom_bounds.min = {80,   5, 90};
            dom_bounds.max = {100, 10, 93}; 
        }
        else if (input == "f18")
        {
            dom_bounds.min = {-0.95, -0.95, -0.95, -0.95};
            dom_bounds.max = { 0.95,  0.95,  0.95,  0.95};
        }
        else if (datasets_3d.count(input) || datasets_2d.count(input) || datasets_unstructured.count(input))
        {
            for (int i = 0; i < dom_bounds.min.dimension(); i++)
            {
                dom_bounds.min[i] = 0.0;
                dom_bounds.max[i] = 1.0;
            }
        }
        else
        {
            cerr << "Unrecognized input in set_dom_bounds(). Exiting." << endl;
            exit(1);
        }

        return;
    }

#endif // MFA_EX_SETUP_HPP