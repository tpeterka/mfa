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
#include <vector>

using namespace std;

// DEPRECATED
// struct Pt1d
// {
//     Pt1d() {}
//     Pt1d(float x_) : x(x_) {}

//     float x;

//     float operator[](int i)
//         {
//             if (i == 0)
//                 return x;
//             else
//             {
//                 cerr << "Pt1d out of bounds index" << endl;
//                 exit(1);
//             }
//         }
// };

// struct Pt2d
// {
//     Pt2d() {}
//     Pt2d(float x_, float y_) : x(x_), y(y_) {}

//     float x, y;

//     float operator[](int i)
//         {
//             if (i == 0)
//                 return x;
//             else if (i == 1)
//                 return y;
//             else
//             {
//                 cerr << "Pt2d out of bounds index" << endl;
//                 exit(1);
//             }
//         }

//     friend
//     ostream&
//     operator<< (ostream &out, Pt2d& p)
//         {
//             out << "(" << p.x << ", " << p.y << ")";
//             return out;
//         }

//     static
//     float dist(Pt2d p1, Pt2d p2)
//         {
//             return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
//         }

// };

// struct Pt3d
// {
//     float x, y, z;
// };

// struct Pt4d
// {
//     float x, y, z, t;
// };

// TODO switch to eigen or other matrix library?

template<typename T>
struct Pt : public vector<T>
{
    // Pt() {};
    Pt(T x)
        {
            this->push_back(x);
        }
    Pt(T x, T y)
        {
            this->push_back(x);
            this->push_back(y);
        }
    Pt(T x, T y, T z)
        {
            this->push_back(x);
            this->push_back(y);
            this->push_back(z);
        }
    Pt(T x, T y, T z, T t)
        {
            this->push_back(x);
            this->push_back(y);
            this->push_back(z);
            this->push_back(t);
        }
    Pt(T x, T y, T z, T t, T u)
        {
            this->push_back(x);
            this->push_back(y);
            this->push_back(z);
            this->push_back(t);
            this->push_back(u);
        }
    Pt(T x, T y, T z, T t, T u, T v)
        {
            this->push_back(x);
            this->push_back(y);
            this->push_back(z);
            this->push_back(t);
            this->push_back(u);
            this->push_back(v);
        }
    Pt(T x, T y, T z, T t, T u, T v, T w)
        {
            this->push_back(x);
            this->push_back(y);
            this->push_back(z);
            this->push_back(t);
            this->push_back(u);
            this->push_back(v);
            this->push_back(w);
        }
    // Euclidean distance ||p1, p2||
    static
    float dist(Pt p1, Pt p2)
        {
            float sum_sq = 0.0;              // sum of squares
            for (size_t i = 0; i < p1.size(); i++)
                sum_sq += (p1[i] - p2[i]) * (p1[i] - p2[i]);
            return sqrt(sum_sq);
        }
    // print p
    friend
    ostream&
    operator<<(ostream &out, Pt& p)
        {
            out << "(";
            for (size_t i = 0; i < p.size(); i++)
            {
                out << p[i];
                if (i < p.size() - 1)
                    out << ", ";
            }
            out << ")";
            return out;
        }
    // DEPRECATED
    // component-wise vector assignment p1 = p2
    // should not be necessary, defaults to vector stl vector assignment
    // Pt<T>&
    // operator=(const Pt& rhs)
    //     {
    //         this->assign(rhs.begin(), rhs.end());
    //         return *this;
    //     }
    // component-wise vector addition p1 + p2
    Pt<T>&
    operator+=(const Pt& rhs)
        {
            for (size_t i = 0; i < this->size(); i++)
                (*this)[i] += rhs[i];
            return *this;
        }
    friend
    Pt<T>
    operator+(Pt lhs, const Pt& rhs)
        {
            lhs += rhs;
            return lhs;
        }
    // component-wise vector subtraction p1 - p2
    Pt<T>&
    operator-=(const Pt& rhs)
        {
            for (size_t i = 0; i < this->size(); i++)
                (*this)[i] -= rhs[i];
            return *this;
        }
    friend
    Pt<T>
    operator-(Pt lhs, const Pt& rhs)
        {
            lhs -= rhs;
            return lhs;
        }
    // component-wise scalar multiplication p * f
    Pt<T>&
    operator*=(const int rhs)
        {
            for (size_t i = 0; i < this->size(); i++)
                (*this)[i] *= rhs;
            return *this;
        }
    Pt<T>&
    operator*=(const float rhs)
        {
            for (size_t i = 0; i < this->size(); i++)
                (*this)[i] *= rhs;
            return *this;
        }
    Pt<T>&
    operator*=(const double rhs)
        {
            for (size_t i = 0; i < this->size(); i++)
                (*this)[i] *= rhs;
            return *this;
        }
    friend
    Pt<T>
    operator*(Pt lhs, const int rhs)
        {
            lhs *= rhs;
            return lhs;
        }
    friend
    Pt<T>
    operator*(Pt lhs, const float rhs)
        {
            lhs *= rhs;
            return lhs;
        }
    friend
    Pt<T>
    operator*(Pt lhs, const double rhs)
        {
            lhs *= rhs;
            return lhs;
        }
    friend
    Pt<T>
    operator*(const int rhs, Pt lhs)
        {
            lhs *= rhs;
            return lhs;
        }
    friend
    Pt<T>
    operator*(const float rhs, Pt lhs)
        {
            lhs *= rhs;
            return lhs;
        }
    friend
    Pt<T>
    operator*(const double rhs, Pt lhs)
        {
            lhs *= rhs;
            return lhs;
        }
};

#endif
