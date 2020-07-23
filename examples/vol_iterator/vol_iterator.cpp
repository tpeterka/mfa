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
        cerr << "ijk: " << ijk.transpose() << endl;

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
        cerr << "ijk: " << ijk.transpose() << endl;

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
}
