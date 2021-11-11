//--------------------------------------------------------------
// example of iterating over a volume or subvolume using the VolIterator class
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>
#include <iostream>

int main()
{
    // iterate over a full volume
    fprintf(stderr, "----- full volume test -----\n");
    VectorXi npts(3);
    npts << 2, 3, 4;
    mfa::VolIterator vol_iter(npts);
    VectorXi ijk(3);
    while (!vol_iter.done())
    {
        fprintf(stderr, "cur_iter = %lu\n", vol_iter.cur_iter());
        for (auto i = 0; i < 3; i++)
            ijk(i) = vol_iter.idx_dim(i);
        cerr << "ijk: " << ijk.transpose() << "\t\t full volume idx " << vol_iter.ijk_idx(ijk) << endl;

        VectorXi ijk1(3);
        vol_iter.idx_ijk(vol_iter.cur_iter(), ijk1);
        if (ijk != ijk1)
        {
            cerr << "Error: ijk " << ijk.transpose() << " != ijk1 " << ijk1.transpose() << endl;
            abort();
        }
        vol_iter.incr_iter();
    }
    fprintf(stderr, "--------------------------\n");

    // iterate over a subvolume out of a larger full volume
    fprintf(stderr, "----- subvolume test -----\n");
    VectorXi sub_npts(3);
    VectorXi sub_starts(3);
    VectorXi all_npts(3);
    sub_npts << 2, 3, 4;
    sub_starts << 1, 1, 1;
    all_npts << 3, 4, 5;
    mfa::VolIterator sub_vol_iter(sub_npts, sub_starts, all_npts);
    while (!sub_vol_iter.done())
    {
        fprintf(stderr, "cur_iter = %lu\n", sub_vol_iter.cur_iter());
        for (auto i = 0; i < 3; i++)
            ijk(i) = sub_vol_iter.idx_dim(i);
        cerr << "ijk: " << ijk.transpose() << "\t\t full volume idx " << sub_vol_iter.ijk_idx(ijk) << endl;

        VectorXi ijk1(3);
        sub_vol_iter.idx_ijk(sub_vol_iter.cur_iter(), ijk1);
        if (ijk != ijk1)
        {
            cerr << "Error: ijk " << ijk.transpose() << " != ijk1 " << ijk1.transpose() << endl;
            abort();
        }
        sub_vol_iter.incr_iter();
    }
    fprintf(stderr, "--------------------------\n");

    // iterate over a slice skipping dim 0 of the full volume
    {                                                                   // scoping so that slice_iter can be reused later
        fprintf(stderr, "----- slice test full volume skipping dimension 0 -----\n");
        mfa::SliceIterator slice_iter(vol_iter, 0);
        while (!slice_iter.done())
        {
            fprintf(stderr, "slice cur_iter = %lu\n", slice_iter.cur_iter());
            cerr << "ijk: " << slice_iter.cur_ijk().transpose() << endl;

            // iterate over curves originating from the slice
            mfa::CurveIterator curve_iter(slice_iter);
            while(!curve_iter.done())
            {
                fprintf(stderr, "\tcurve cur_iter = %lu", curve_iter.cur_iter());
                cerr << "\t full volume ijk: " << curve_iter.cur_ijk().transpose();
                cerr << "\t full volume idx " << curve_iter.ijk_idx(curve_iter.cur_ijk()) << endl;
                curve_iter.incr_iter();
            }
            slice_iter.incr_iter();
        }
        fprintf(stderr, "--------------------------\n");
    }

    // iterate over a slice skipping dim 1 of the full volume
    {
        fprintf(stderr, "----- slice test full volume skipping dimension 1 -----\n");
        mfa::SliceIterator slice_iter(vol_iter, 1);
        while (!slice_iter.done())
        {
            fprintf(stderr, "cur_iter = %lu\n", slice_iter.cur_iter());
            cerr << "ijk: " << slice_iter.cur_ijk().transpose() << endl;

            // iterate over curves originating from the slice
            mfa::CurveIterator curve_iter(slice_iter);
            while(!curve_iter.done())
            {
                fprintf(stderr, "\tcurve cur_iter = %lu", curve_iter.cur_iter());
                cerr << "\t full volume ijk: " << curve_iter.cur_ijk().transpose();
                cerr << "\t full volume idx " << curve_iter.ijk_idx(curve_iter.cur_ijk()) << endl;
                curve_iter.incr_iter();
            }
            slice_iter.incr_iter();
        }
        fprintf(stderr, "--------------------------\n");
    }

    // iterate over a slice skipping dim 2 of the full volume
    {
        fprintf(stderr, "----- slice test full volume skipping dimension 2 -----\n");
        mfa::SliceIterator slice_iter(vol_iter, 2);
        while (!slice_iter.done())
        {
            fprintf(stderr, "cur_iter = %lu\n", slice_iter.cur_iter());
            cerr << "ijk: " << slice_iter.cur_ijk().transpose() << endl;

            // iterate over curves originating from the slice
            mfa::CurveIterator curve_iter(slice_iter);
            while(!curve_iter.done())
            {
                fprintf(stderr, "\tcurve cur_iter = %lu", curve_iter.cur_iter());
                cerr << "\t full volume ijk: " << curve_iter.cur_ijk().transpose();
                cerr << "\t full volume idx " << curve_iter.ijk_idx(curve_iter.cur_ijk()) << endl;
                curve_iter.incr_iter();
            }
            slice_iter.incr_iter();
        }
        fprintf(stderr, "--------------------------\n");
    }

    // iterate over a slice skipping dim 0 of the subvolume
    {
        fprintf(stderr, "----- slice test subvolume skipping dimension 0 -----\n");
        mfa::SliceIterator slice_iter(sub_vol_iter, 0);
        while (!slice_iter.done())
        {
            fprintf(stderr, "slice cur_iter = %lu\n", slice_iter.cur_iter());
            cerr << "ijk: " << slice_iter.cur_ijk().transpose() << endl;

            // iterate over curves originating from the slice
            mfa::CurveIterator curve_iter(slice_iter);
            while(!curve_iter.done())
            {
                fprintf(stderr, "\tcurve cur_iter = %lu", curve_iter.cur_iter());
                cerr << "\t full volume ijk: " << curve_iter.cur_ijk().transpose();
                cerr << "\t full volume idx " << curve_iter.ijk_idx(curve_iter.cur_ijk()) << endl;
                curve_iter.incr_iter();
            }
            slice_iter.incr_iter();
        }
        fprintf(stderr, "--------------------------\n");
    }

    // iterate over a slice skipping dim 1 of the subvolume
    {
        fprintf(stderr, "----- slice test subvolume skipping dimension 1 -----\n");
        mfa::SliceIterator slice_iter(sub_vol_iter, 1);
        while (!slice_iter.done())
        {
            fprintf(stderr, "cur_iter = %lu\n", slice_iter.cur_iter());
            cerr << "ijk: " << slice_iter.cur_ijk().transpose() << endl;

            // iterate over curves originating from the slice
            mfa::CurveIterator curve_iter(slice_iter);
            while(!curve_iter.done())
            {
                fprintf(stderr, "\tcurve cur_iter = %lu", curve_iter.cur_iter());
                cerr << "\t full volume ijk: " << curve_iter.cur_ijk().transpose();
                cerr << "\t full volume idx " << curve_iter.ijk_idx(curve_iter.cur_ijk()) << endl;
                curve_iter.incr_iter();
            }
            slice_iter.incr_iter();
        }
        fprintf(stderr, "--------------------------\n");
    }

    // iterate over a slice skipping dim 2 of the subvolume
    {
        fprintf(stderr, "----- slice test subvolume skipping dimension 2 -----\n");
        mfa::SliceIterator slice_iter(sub_vol_iter, 2);
        while (!slice_iter.done())
        {
            fprintf(stderr, "cur_iter = %lu\n", slice_iter.cur_iter());
            cerr << "ijk: " << slice_iter.cur_ijk().transpose() << endl;

            // iterate over curves originating from the slice
            mfa::CurveIterator curve_iter(slice_iter);
            while(!curve_iter.done())
            {
                fprintf(stderr, "\tcurve cur_iter = %lu", curve_iter.cur_iter());
                cerr << "\t full volume ijk: " << curve_iter.cur_ijk().transpose();
                cerr << "\t full volume idx " << curve_iter.ijk_idx(curve_iter.cur_ijk()) << endl;
                curve_iter.incr_iter();
            }
            slice_iter.incr_iter();
        }
        fprintf(stderr, "--------------------------\n");
    }
}
