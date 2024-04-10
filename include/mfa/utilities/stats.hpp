//--------------------------------------------------------------
// Stastics helpers for MFA
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_STATS_HPP
#define _MFA_STATS_HPP

#include <mfa/types.hpp>
#include <mfa/pointset.hpp>

namespace mfa
{
    enum class PrintStyle
    {
        Vert,
        Side
    };

    template <typename T>
    struct Stats
    {
        // Here by "L2" error we mean root mean squared error
        // "L1" is mean error
        // "Linf" is the maximum error
        ArrayXi npts;
        ArrayX<T> extent;
        ArrayX<T> sum;  // sum of errors
        ArrayX<T> ssq;  // sum of squares
        ArrayX<T> mxv;  // max value
        int nvars;

        bool log;       // save all data points (to write to files)
        vector<vector<T>> data;

        T l1_max;
        T l2_max;
        T linf_max;
        T l1_rel_max;
        T l2_rel_max;
        T linf_rel_max;
        int l1_max_var;
        int l2_max_var;
        int linf_max_var;
        int l1_rel_max_var;
        int l2_rel_max_var;
        int linf_rel_max_var;

        PrintStyle style;
        bool initialized{false};

    public:
        Stats(bool log_ = false) :
            nvars(0), log(log_),
            l1_max(0), l2_max(0), linf_max(0),
            l1_rel_max(0), l2_rel_max(0), linf_rel_max(0),
            l1_max_var(-1), l2_max_var(-1), linf_max_var(-1),
            l1_rel_max_var(-1), l2_rel_max_var(-1), linf_rel_max_var(-1),
            style(PrintStyle::Vert)
        { }

        void set_style(PrintStyle option)
        {
            style = option;
        }

        void init(mfa::PointSet<T>* input)
        {
            nvars = input->nvars();
            npts = ArrayXi::Zero(nvars);
            sum = ArrayX<T>::Zero(nvars);
            ssq = ArrayX<T>::Zero(nvars);
            mxv = ArrayX<T>::Zero(nvars);
            data.resize(nvars);

            extent = ArrayX<T>::Zero(nvars);
            for (int k = 0; k < nvars; k++)
            {
                int dim_min = input->var_min(k);
                int var_dim = input->var_dim(k);
                VectorX<T> maxs = input->domain.middleCols(dim_min, var_dim).colwise().maxCoeff();
                VectorX<T> mins = input->domain.middleCols(dim_min, var_dim).colwise().minCoeff();
                extent(k) = (maxs - mins).norm();
            }

            initialized = true;
        }

        T l1(int k) const { return sum[k] / npts[k]; }
        T l2(int k) const { return sqrt(ssq[k] / npts[k]); }
        T linf(int k) const { return mxv[k]; }

        ArrayX<T> l1() const
        {
            ArrayX<T> l1vec = ArrayX<T>::Zero(nvars);
            for (int k = 0; k < nvars; k++)
            {
                l1vec(k) = l1(k);
            }
            return l1vec;
        }

        ArrayX<T> l2() const
        {
            ArrayX<T> l2vec = ArrayX<T>::Zero(nvars);
            for (int k = 0; k < nvars; k++)
            {
                l2vec(k) = l2(k);
            }
            return l2vec;
        }

        ArrayX<T> linf() const
        {
            ArrayX<T> linfvec = ArrayX<T>::Zero(nvars);
            for (int k = 0; k < nvars; k++)
            {
                linfvec(k) = linf(k);
            }
            return linfvec;
        }

        void update(int k, T x)
        {
            npts(k)++;
            sum(k) += x;
            ssq(k) += x*x;
            if (x > mxv(k))
                mxv(k) = x;

            if (log)
                data[k].push_back(x);
        }

        void write_var(int k, string filepattern)
        {
            if (!log)
            {
                fmt::print("Warning: ErrorStats did not save data and will not write data file \"{}_var{}\"\n", filepattern, k);
                return;
            }

            string filename_abs = fmt::format("{}_var{}_abs.txt", filepattern, k);
            string filename_rel = fmt::format("{}_var{}_rel.txt", filepattern, k);
            FILE* absfile = fopen(filename_abs.c_str(), "w");
            FILE* relfile = fopen(filename_rel.c_str(), "w");
            for (int i = 0; i < data[k].size(); i++)
            {
                fmt::print(absfile, "{}\n", data[k][i]);
                fmt::print(relfile, "{}\n", data[k][i] / extent(k));
            }

            fclose(absfile);
            fclose(relfile);

            return;
        }

        void write_all_vars(string filepattern)
        {
            for (int k = 0; k < nvars; k++)
                write_var(k, filepattern);

            return;
        }

        void print_var(int k) const
        {
            if (k < 0 || k >= nvars)
            {
                cerr << "ERROR: Index out of bounds in ErrorStats::print_var()" << endl;
                cerr << "         index = " << k << ", nvars = " << nvars << "." << endl;
                cerr << "Exiting." << endl;
                exit(1);
            }

            if (style == PrintStyle::Vert)
            {
                fmt::print("Range Extent           = {:.4e}\n", extent(k));
                fmt::print("Max Error              = {:.4e}\n", linf(k));
                fmt::print("RMS Error              = {:.4e}\n", l2(k));
                fmt::print("Avg Error              = {:.4e}\n", l1(k));
                fmt::print("Max Error (normalized) = {:.4e}\n", linf(k) / extent(k));
                fmt::print("RMS Error (normalized) = {:.4e}\n", l2(k) / extent(k));
                fmt::print("Avg Error (normalized) = {:.4e}\n", l1(k) / extent(k));
            }
            else if (style == PrintStyle::Side)
            {
                fmt::print("Max Error: {:.4e}\tMax Error (rel): {:.4e}\n", linf(k), linf(k) / extent(k));
                fmt::print("RMS Error: {:.4e}\tRMS Error (rel): {:.4e}\n", l2(k), l2(k) / extent(k));
                fmt::print("Avg Error: {:.4e}\tAvg Error (rel): {:.4e}\n", l1(k), l1(k) / extent(k));
            }
            else
            {
                fmt::print("Error: Unrecognized print style in ErrorStats\n");
            }
        }

        // find max over all science variables
        void find_max_stats()
        {
            l1_max = l1().maxCoeff(&l1_max_var);
            l2_max = l2().maxCoeff(&l2_max_var);
            linf_max = linf().maxCoeff(&linf_max_var);
            l1_rel_max = (l1() / extent).maxCoeff(&l1_rel_max_var);
            l2_rel_max = (l2() / extent).maxCoeff(&l2_rel_max_var);
            linf_rel_max = (linf() / extent).maxCoeff(&linf_rel_max_var);
        }

        void print_max()
        {
            find_max_stats();
            fmt::print("Maximum errors over all science variables:\n");
            fmt::print("Max Error                 (var {}) = {:.4e}\n", linf_max_var, linf_max);
            fmt::print("RMS Error                 (var {}) = {:.4e}\n", l2_max_var, l2_max);
            fmt::print("Avg Error                 (var {}) = {:.4e}\n", l1_max_var, l1_max);
            fmt::print("Max Error (normalized)    (var {}) = {:.4e}\n", linf_rel_max_var, linf_rel_max);
            fmt::print("RMS Error (normalized)    (var {}) = {:.4e}\n", l2_rel_max_var, l2_rel_max);
            fmt::print("Avg Error (normalized)    (var {}) = {:.4e}\n", l1_rel_max_var, l1_rel_max);        
        }
    };

    // error statistics, used for iterative encoding (I think? --david)
    template <typename T>
    struct ErrorStats
    {
        T max_abs_err;          // max of absolute errors (absolute value)
        T max_norm_err;         // max of normalized errors (absolute value)
        T sum_sq_abs_errs;      // sum of squared absolute errors
        T sum_sq_norm_errs;     // sum of squared normalized errors

        ErrorStats()
        {
            max_abs_err         = 0.0;
            max_norm_err        = 0.0;
            sum_sq_abs_errs     = 0.0;
            sum_sq_norm_errs    = 0.0;
        }
        ErrorStats(T max_abs_err_, T max_norm_err_, T sum_sq_abs_errs_, T sum_sq_norm_errs_) :
            max_abs_err(max_abs_err_),
            max_norm_err(max_norm_err_),
            sum_sq_abs_errs(sum_sq_abs_errs_),
            sum_sq_norm_errs(sum_sq_norm_errs_)
        {}
    };
}   // namespace mfa

#endif  // _MFA_STATS_HPP