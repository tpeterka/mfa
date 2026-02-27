# example illustrating reading an MFA and evaluating a point from it
# assumes that a 2-d domain was previously modeled and saved as "approx.out"

import diy
import mfa
import numpy as np

# MPI, DIY world and master
w = diy.mpi.MPIComm()  # world
m = diy.Master(w)  # master

# load the results and print them out
print("\n\nLoading blocks and printing them out\n")
a = diy.ContiguousAssigner(w.size, -1)
diy.read_blocks("approx.mfa", a, m, load=mfa.Block.load)
m.foreach(lambda b, cp: b.print_block(cp, False))

# evaluate a point
param = np.array([0.5, 0.5])  # input parameters where to decode the point
pts_by_gid = {}
m.foreach(lambda b, cp: pts_by_gid.update({cp.gid(): b.decode_point(cp, param)}))
pt = pts_by_gid[0] if 0 in pts_by_gid else pts_by_gid[min(pts_by_gid)]
print("\nThe point at [u =", param[0], ", v =", param[1], "] =", pt)
