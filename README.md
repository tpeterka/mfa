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
-DCLP_INCLUDE_DIRS=/path/to/Clp/include \
-DCLP_LIB=/path/to/Clp/lib/libClp.a # or libClp.so or libClp.dylib

make
make install
```
# Run example
(currently only serial)

```
cd path/to/mfa/install/examples/adaptive
./adaptive -e <error_limit, e.g., 1.0e-3>
```

The output file, `approx.out`, contains the output mfa data model.
