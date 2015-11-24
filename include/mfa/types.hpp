//--------------------------------------------------------------
// type definitions
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _TYPES_HPP
#define _TYPES_HPP

#include <iostream>
#include <cmath>

using namespace std;

// TODO: floats for now; still need to parameterize type
// probably just switch over to eigen types or some other matrix and vector library
// no sense writing this stuff myself
struct Pt1d
{
    Pt1d() {}
    Pt1d(float x_) : x(x_) {}

    float x;

    float operator[](int i)
        {
            if (i == 0)
                return x;
            else
            {
                cerr << "Pt1d out of bounds index" << endl;
                exit(1);
            }
        }
};

struct Pt2d
{
    Pt2d() {}
    Pt2d(float x_, float y_) : x(x_), y(y_) {}

    float x, y;

    float operator[](int i)
        {
            if (i == 0)
                return x;
            else if (i == 1)
                return y;
            else
            {
                cerr << "Pt2d out of bounds index" << endl;
                exit(1);
            }
        }

    friend
    ostream&
    operator<< (ostream &out, Pt2d& p)
        {
            out << "(" << p.x << ", " << p.y << ")";
            return out;
        }

    static
    float dist(Pt2d p1, Pt2d p2)
        {
            return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
        }

};

struct Pt3d
{
    float x, y, z;
};

struct Pt4d
{
    float x, y, z, t;
};

#endif
