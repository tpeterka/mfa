# MFA

Multivariate Functional Approximation (MFA) is a data model based on NURBS high-dimensional tensor products to model scientific data sets.

The key features are:

- Fitting multivariate spatiotemporal geometry together with multivariate attribute data.
- Progressively improving the quality of the approximation by locally adding more detail in
  spatial regions of rapid change (edges, turbulence, discontinuities).
- Guaranteeing quantified error bounds.
- A data model based on the MFA can be parallelized and
    scaled to emerging and future HPC architectures.
- Applicable in analysis of scientific datasets.

# Licensing

MFA is licensed [here](./COPYING).

# Installation

You can either install MFA using [Spack](https://spack.readthedocs.io/en/latest/) (recommended), or manually.

## Installing with Spack

First, install Spack as explained [here](https://spack.readthedocs.io/en/latest/getting_started.html) and add Spack to
your path. Next, clone the MFA repository:

```
git clone https://github.com/tpeterka/mfa
```

Add MFA to your local Spack installation:

```
cd /path/to/mfa
spack repo add .
```

You can confirm that Spack can find MFA:
```
spack info mfa
```

Then install MFA. This could take some time depending on whether you already have a Spack system with MPI
installed. The first time you use Spack, many dependencies need to be satisfied, which by default are installed from
scratch. If you are an experienced Spack user, you can tell Spack to use existing dependencies from
elsewhere in your system.

```
spack install mfa
```

The default installation is single-threaded (serial), but you can specify different threading models like this:

```
spack install mfa thread=tbb
```

TBB is the only threading model currently implemented, but other threading models such as SYCL and Kokkos are being developed.

## Installing manually

Build dependencies

- C++11 compiler
- [MPI](http://www.mpich.org)
- [TBB (optional)](https://www.threadingbuildingblocks.org)
- [COIN-OR CLP (optional)](https://projects.coin-or.org/Clp)

Build mfa

```
git clone https://github.com/tpeterka/mfa
cd mfa
mkdir build
cd build

cmake .. \
-DCMAKE_INSTALL_PREFIX=/path/to/mfa/install \
-DCMAKE_CXX_COMPILER=mpicxx \                                   # optional, set to CC on Cray, default or g++ also works if MPI is correctly found
-Dmfa_thread=tbb \                                              # optional TBB threading, serial (no threading) is the default
-DTBB_INCLUDE_DIR=/path/to/tbb/include \                        # optional, will try to find TBB automatically if mfa_thread=tbb
-DTBB_LIBRARY=/path/to/tbb/library \                            # optional, will try to find TBB automatically if mfa_thread=tbb
-DCLP_INCLUDE_DIRS=/path/to/Clp/include \                       # optional, only needed for weights
-DCLP_LIB=/path/to/Clp/lib/libClp.a (.so, .dylib) \             # optional, only needed for weights
-DCOIN_UTILS_LIB=/path/to/Clp/lib/libCoinUtils.a (.so, .dylib)  # optional, only needed for weights

make
make install
```
# Run examples

Many command-line options are available for the examples. Invoke any example with the
-h flag to see the available choices. Minimal defaults are included below.

```
cd /path/to/mfa/install/examples/fixed

# single block, fixed number of control points
./fixed -w 0

# multiple blocks, fixed number of control points
./fixed_multiblock -b 4 -w 0

# adaptive number of control points (single block only so far)
cd path/to/mfa/install/examples/adaptive
./adaptive -e <error_limit, e.g., 1.0e-3> -w 0

# derivatives of a previously computed MFA (single block only so far)
cd /path/to/mfa/install/examples/differentiate
./differentiate -i </path/to/approx.out>
```

The output file, `approx.out`, contains the output MFA data model. The output
file `deriv.out` contains the derivatives from the last example. Either output
can be converted to VTK format (for visualization) as follows:

```
/path/to/mfa/install/examples/write_vtk -f </path/to/approx.out/or/deriv.out>
```

