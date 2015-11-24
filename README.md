# MFA

Multivariate Functional Approximation (MFA) is a data model based on NURBS high-dimensional tensor products to model scientific data sets.

The key features are:

- **Data reduction.**  The MFA is defined by a much smaller set of control data than original raw
data. Moreover, each data dimension is fitted with a different set of control data, so that the
compressibility matches the information content of each attribute. For example, if pressure
varies rapidly, but temperature is relatively constant, then for the same error bound there can
be fewer control points for temperature than for pressure.

- **Extreme scalability.**  The space savings of the MFA comes at a computational cost, and a
  high-dimensional MFA was not possible earlier because data were small and flops were
  expensive.  But this research is driven by extreme-scale computing that is now limited by
  data movement. We will minimize the cost of encoding the MFA by using approximate and
  adaptive methods that are parallelized over processes, threads, and vector units. We will
  control locality over all levels of a deep memory/storage hierarchy through the careful
  mapping of the local support inherent in the MFA to the data movement characteristics of DOE
  extreme-scale machines.

- **Scientific applications.**  The MFA is designed to interface with N-body, structured,
unstructured, and adaptively-refined in situ computations and post hoc data. Many analysis and
visualization operations are possible directly from the MFA without ever resampling discrete
data again. For example, an MFA is invariant to linear and affine transformations, meaning that
visual analytic techniques that change the frame of reference---such as vortex detection---can
be applied to the MFA instead of the raw data.

- **Functional meaning.**  Functions uncover hidden meaning. The underlying physics is governed by
equations, but raw discrete data mask the underlying behavior just as a table of numbers hides
the trends visible in a plot of a simple equation y = f(x).  A functional form enhances
understanding of the behavior because derivatives, trends, integrals, correlations, and
simplifications have analytical solutions.

# Licensing

MFA is [public domain](./COPYING) software.

# Installation

Build dependencies

- [diy2](https://github.com/diatomic/diy2)
- [eigen](http://eigen.tuxfamily.org)
- [MPI](http://www.mpich.org)

Build mfa

```
git clone https://bitbucket.org/tpeterka1/mfa

cmake .. \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_INSTALL_PREFIX=/path/to/mfa/install \
-DDIY_INCLUDE_DIRS=/path/to/diy2/include \
-DEIGEN_INCLUDE_DIRS=/path/to/eigen-3.2.5 \

make
make install
```
# Run example
(currently only serial)

```
cd path/to/mfa/install/examples/simple
./1d
```

The output file, `approx.out` contains the output mfa data model.
