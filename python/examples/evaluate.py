# example illustrating reading an MFA and evaluating a point from it
# assumes that a 2-d domain was previously modeled and saved as "approx.out"

import diy
import mfa
import numpy as np

# MPI, DIY world and master
w = diy.mpi.MPIComm()           # world
m = diy.Master(w)               # master

# load the results and print them out
print("\n\nLoading blocks and printing them out\n")
a = diy.ContiguousAssigner(w.size, -1)
diy.read_blocks("approx.out", a, m, load = mfa.load_block)
m.foreach(lambda b,cp: b.print_block(cp, False))

# evaluate a point
param   = np.array([0.5, 0.5])          # input parameters where to decode the point
pt      = np.array([0.0, 0.0, 0.0])     # assigning some fake values to define shape and type
m.foreach(lambda b, cp: b.decode_point(cp, param, pt))
print(pt)
