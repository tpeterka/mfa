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

Build dependencies

- C++11 compiler
- [diy](https://github.com/diatomic/diy)
- [eigen](http://eigen.tuxfamily.org)
- [MPI](http://www.mpich.org)
- [TBB](https://www.threadingbuildingblocks.org)
- [COIN-OR CLP](https://projects.coin-or.org/Clp)

Build mfa

```
git clone https://bitbucket.org/tpeterka1/mfa
cd mfa
mkdir build
cd build

cmake .. \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_INSTALL_PREFIX=/path/to/mfa/install \
-DDIY_INCLUDE_DIRS=/path/to/diy/include \
-DEIGEN_INCLUDE_DIRS=/path/to/eigen3 \
-DTBB_INCLUDE_DIR=/path/to/tbb/include \
-DTBB_LIBRARY=/path/to/tbb/library \
-DCLP_INCLUDE_DIRS=/path/to/Clp/include \
-DCLP_LIB=/path/to/Clp/lib/libClp.a # or libClp.so or libClp.dylib

make
make install
```
# Run example

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
# single block
/path/to/mfa/install/examples/write_vtk </path/to/approx.out/or/deriv.out>

# multiple blocks
/path/to/mfa/install/examples/write_vtk_multiblock </path/to/approx.out>
```

