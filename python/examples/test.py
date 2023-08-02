import mfa
import diy
from mpi4py import MPI
import math

# MPI, DIY world and master
w = diy.mpi.MPIComm()           # world
m = diy.Master(w)               # master
nblocks = w.size                # hard-code 1 block per MPI rank

# problem dimensionality: 2D input, 1 1D output variable
dom_dim = 2
var_dim = 1
pt_dim = 3
model_dims = [ dom_dim, var_dim ]

# problem domain
dmin = [-4.0 * math.pi, -4.0 * math.pi]
dmax = [4.0 * math.pi, 4.0 * math.pi]

# basic program arguments
fun             = "sinc"
error           = True
geom_degree     = 1
vars_degree     = 4
ndom_pts        = 100
geom_nctrl_pts  = geom_degree + 1
vars_nctrl_pts  = 11
ghost_factor    = 0.0


# default dataset arguments
d_args = mfa.DomainArgs(dom_dim, model_dims)
d_args.min = dmin
d_args.max = dmax
d_args.ndom_pts = [ndom_pts] * dom_dim
d_args.tot_ndom_pts = ndom_pts**dom_dim

# specify MFA model parameters
mfa_info = mfa.MFAInfo(dom_dim, 1) # verbose=1 all output
geom_model_info = mfa.ModelInfo(dom_dim, dom_dim, geom_degree, geom_nctrl_pts) # dom_dim twice for 2D domain
variable_model_info = mfa.ModelInfo(dom_dim, var_dim, vars_degree, vars_nctrl_pts)
mfa_info.addGeomInfo(geom_model_info)
mfa_info.addVarInfo(variable_model_info)

# def add_block(gid, core, bounds, domain_, link):
#     print(">>", core.min.dimension())
#     mfa.add_block(gid, core, bounds, domain_, link, m, dom_dim, pt_dim, ghost_factor)

print("Pre domain decomposition")
# decompose domain using double precision bounds
domain = diy.DoubleContinuousBounds(dmin, dmax)
d = diy.DoubleContinuousDecomposer(dom_dim, domain, nblocks)
a = diy.ContiguousAssigner(w.size, nblocks)
d.decompose(w.rank, a, lambda gid, core, bounds, domain_, link: mfa.Block.add(gid, core, bounds, domain_, link, m, dom_dim, pt_dim, ghost_factor))
print("---")
print(m.size())
# initialize input data
m.foreach(lambda b, cp: b.generate_analytical_data(cp, fun, mfa_info, d_args))

# compute the MFA
m.foreach(lambda b, cp: b.fixed_encode_block(cp, mfa_info))

# debug: compute error field
if error:
    m.foreach(lambda b, cp: b.range_error(cp, True, True))

# print results
m.foreach(lambda b, cp: b.print_block(cp, True))

# error goals: 10-2 to 10-5

# save the results
print("\n\nSaving blocks\n")
diy.write_blocks("approx.mfa", m, save = mfa.save_block)
# convert to vtk and view in paraview
