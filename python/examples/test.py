import mfa
import diy
from mpi4py import MPI
import math

# default program arguments
fun             = "sinc"
error           = True
dom_dim         = 2
pt_dim          = 3
geom_degree     = 1
vars_degree     = 4
ndom_pts        = 100
geom_nctrl_pts  = geom_degree + 1
vars_nctrl_pts  = 11

var_dim = 1
model_dims = [ dom_dim, var_dim ]

print(f"Type of Model Dims: {type(model_dims)}")

mfa_info = mfa.MFAInfo(dom_dim, 1) # verbose=1 all output
geom_model_info = mfa.ModelInfo(dom_dim, dom_dim, geom_degree, geom_nctrl_pts) # dom_dim twice for 2D domain
variable_model_info = mfa.ModelInfo(dom_dim, var_dim, vars_degree, vars_nctrl_pts)

# build the DomainArgs object
d_args = mfa.DomainArgs(dom_dim, model_dims)

# TODO
# add both modelInfos to the mfainfo
# make DomainArgs (to create data)
# ndom_pts or tot_ndom_pts

# default dataset arguments
# d_args                  = mfa.DomainArgs(dom_dim, pt_dim)
# d_args.weighted         = 0
# d_args.n                = 0.0
# d_args.multiblock       = False
# d_args.verbose          = 1
# # NB, arrays bound to STL vectors must be assigned wholesale, not modified elementwise
# d_args.f                = [1.0, 1.0]
# d_args.geom_p           = [geom_degree, geom_degree]
# d_args.vars_p           = [[vars_degree, vars_degree]]
# d_args.ndom_pts         = [ndom_pts, ndom_pts]
# d_args.geom_nctrl_pts   = [geom_nctrl_pts, geom_nctrl_pts]
# d_args.vars_nctrl_pts   = [[vars_nctrl_pts, vars_nctrl_pts]]
dmin              = [-4.0 * math.pi, -4.0 * math.pi]
dmax              = [4.0 * math.pi, 4.0 * math.pi]
ghost_factor      = 0.0
# d_args.s                = [10.0, 1.0, 1.0]

# MPI, DIY world and master
w = diy.mpi.MPIComm()           # world
m = diy.Master(w)               # master
nblocks = w.size                # hard-code 1 block per MPI rank
print(nblocks)

def add_block(gid, core, bounds, domain_, link):
    print(">>", core.min.dimension())
    mfa.add_block(gid, core, bounds, domain_, link, m, dom_dim, pt_dim, ghost_factor)

print("Pre domain decomposition")
print(dmin, dmax)
# decompose domain using double precision bounds
domain = diy.DoubleContinuousBounds([-4., -4.], [4., 4.])
# domain = diy.DoubleContinuousBounds(dmin, dmax)

print(dom_dim, nblocks)
print(domain)

print("domain")
d = diy.DoubleContinuousDecomposer(dom_dim, domain, nblocks)
print("decomposer")
a = diy.ContiguousAssigner(w.size, nblocks)
gidvec = a.local_gids(w.rank)
print(gidvec)
print("assigner")
# d.decompose(w.rank, a, lambda gid, core, bounds, domain_, link: mfa.Block.add(gid, core, bounds, domain_, link, m, dom_dim, pt_dim, ghost_factor))
d.decompose(w.rank, a, add_block)

print("Generating data")

# initialize input data
m.foreach(lambda b, cp: b.generate_analytical_data(cp, fun, mfa_info, d_args))

print("Encoding data")

# compute the MFA
m.foreach(lambda b, cp: b.fixed_encode_block(cp, d_args))

# debug: compute error field
if error:
    m.foreach(lambda b, cp: b.range_error(cp, 1, True, True))

# print results
m.foreach(lambda b, cp: b.print_block(cp, True))

# error goals: 10-2 to 10-5

# save the results
print("\n\nSaving blocks\n")
diy.write_blocks("approx.out", m, save = mfa.save_block)
# convert to vtk and view in paraview
