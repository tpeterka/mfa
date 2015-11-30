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

// TODO switch to eigen or other matrix library?

template<typename T>
struct Pt : public vector<T>
{
    Pt()         : vector<T>() {}
    Pt(size_t n) : vector<T>(n) {}

    Pt<T>&
    set(T x)
        {
            this->resize(1);
            (*this)[0] = x;
            return *this;
        }

    Pt<T>&
    set(T x, T y)
        {
            this->resize(2);
            (*this)[0] = x;
            (*this)[1] = y;
            return *this;
        }

    Pt<T>&
    set(T x, T y, T z)
        {
            this->resize(3);
            (*this)[0] = x;
            (*this)[1] = y;
            (*this)[2] = z;
            return *this;
        }

    Pt<T>&
    set(T x, T y, T z, T t)
        {
            this->resize(4);
            (*this)[0] = x;
            (*this)[1] = y;
            (*this)[2] = z;
            (*this)[3] = t;
            return *this;
        }

    Pt<T>&
    sett(T x, T y, T z, T t, T u)
        {
            this->resize(5);
            (*this)[0] = x;
            (*this)[1] = y;
            (*this)[2] = z;
            (*this)[3] = t;
            (*this)[4] = u;
            return *this;
        }

    Pt<T>&
    set(T x, T y, T z, T t, T u, T v)
        {
            this->resize(6);
            (*this)[0] = x;
            (*this)[1] = y;
            (*this)[2] = z;
            (*this)[3] = t;
            (*this)[4] = u;
            (*this)[5] = v;
            return *this;
        }

    Pt<T>&
    set(T x, T y, T z, T t, T u, T v, T w)
        {
            this->resize(7);
            (*this)[0] = x;
            (*this)[1] = y;
            (*this)[2] = z;
            (*this)[3] = t;
            (*this)[4] = u;
            (*this)[5] = v;
            (*this)[6] = w;
            return *this;
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
