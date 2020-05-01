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

MFA is [public domain](./COPYING) software.

# Installation

You can either install MFA using [Spack](https://spack.readthedocs.io/en/latest/) (recommended), or manually.

## Installing with Spack

First, install Spack as explained [here](https://spack.readthedocs.io/en/latest/getting_started.html). Once Spack is
installed and available in your path, clone the mfa repository and add it to your local Spack repositories:

```
git clone https://github.com/tpeterka/mfa
spack repo add mfa
```

You can confirm that Spack can find mfa:
```
spack info mfa
```

Then install mfa. This could take some time depending on whether you already have a Spack system with MPI
installed. The first time you use Spack, many dependencies need to be satisfied, which by default are installed from
scratch. If you are an experienced Spack user, you can tell Spack to use existing dependencies from
elsewhere on your system.

```
spack install mfa
```

## Installing manually

Build dependencies

- C++11 compiler
- [diy](https://github.com/diatomic/diy)
- [eigen](http://eigen.tuxfamily.org)
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
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=/path/to/mfa/install \
-DDIY_INCLUDE_DIRS=/path/to/diy/include \
-DEIGEN_INCLUDE_DIRS=/path/to/eigen3 \
-DTBB_INCLUDE_DIR=/path/to/tbb/include \                        # optional
-DTBB_LIBRARY=/path/to/tbb/library \                            # optional
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

