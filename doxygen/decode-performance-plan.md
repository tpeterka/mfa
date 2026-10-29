# Point-set decode performance plan

## Scope

Keep the public `MFA::Decode`, `DecodeVar`, and `DecodeGeom` interfaces and
model representation unchanged. Prioritize value-only point-set decoding with
`MFA_TMESH` and `MFA_OVERLAYS` disabled, domain/custom parameterization, and
unweighted splines. Reuse the existing fast evaluator's contractions; do not
introduce a new numerical kernel, SIMD scheme, control-point ordering, or
application-specific ocean-grid algorithm.

This document records the original investigation and implementation sequence.
The application baseline, unstructured fast value path, and worker workspace
changes are implemented; see [validation results](decode-performance-results.md).
Structured/grid cleanup and derivative evaluation improvements remain follow-ups.

## Observed application path

The MFA Remapping chat **Run Gulf Stream remap** and its saved run logs show:

| Controls | Fit (s) | Source decode (s) | Target decode (s) | Application total (s) |
| --- | ---: | ---: | ---: | ---: |
| 20x20x40 | 5.031 | 5.899 | 167.752 | 184.151 |
| 30x30x40 | 12.147 | 4.565 | 170.929 | 193.352 |
| 40x40x40 | 26.741 | 4.576 | 170.608 | 206.852 |
| 60x40x40 | 44.580 | 6.477 | 193.746 | 251.656 |

These are recorded observations, not new full-application timings. Target
decoding accounts for 77–91% of application time. The run uses degree 2,
TBB with an 18-thread limit, and 34,134,100 target points.

Crucially, temperature and salinity are components of **one vector-valued
science model**, not two scalar MFA models. `makeMFAInfo` constructs model
dimensions `{3, variableNames.size()}`, and the run prints `{3, 2}`.
`forwardMFA.DecodeVar(0, output)` reaches the TBB point loop. Both source
and target point sets use unstructured
parameter lists, despite the target mesh having regular horizontal topology.
The vertical spline knots are nonuniform.

There is also a build-configuration mismatch. The saved executable's Spack
build log identifies installed MFA dependency
`mfa-master-pnvw3eu2jpahwv3ndmqqx4ul7y6zq6jt`; its `mfa.hpp` defines
`MFA_OVERLAYS`. That selects `VolPt_tmesh`, with anchor searches, local-knot
construction, and allocation inside the control-point loop, even for a single
tensor. The working checkout disables overlays and selects reusable `VolPt`.
Neither route currently selects `FastVolPt` for these point sets. The saved
application was compiled with `-O2 -g -DNDEBUG`, `MFA_TBB`, and
`MFA_NO_WEIGHTS`; this is not an unoptimized debug-build issue.

The remapper already copies geometry separately and decodes science variables
only. Skipping geometry in `MFA::Decode` therefore cannot accelerate this run.
Constructing a `Decoder` happens once per model call, not once per point on
this path.

Relevant application files:

- `climate-remap/src/remap_loop.cpp`: `makeMFAInfo`, `weighted = 0`.
- `climate-remap/src/remap_domain.hpp`: unstructured point sets and custom parameters.
- `climate-remap/src/remap_models.hpp`: forward/source/reverse decode calls.
- `climate-remap/CMakeLists.txt`: `MFA_NO_WEIGHTS`, serial/TBB selection.
- `climate-remap/runs/gulfstream-20x20x40-degree2-reg1-20261001/summary/runtime_breakdown.csv`.

Unweighted fitting is requested by `weighted = false/0`; the stored weights
are **one**, not zero. `MFA_NO_WEIGHTS` disables weight-fitting support, and
the current fast routines require this macro. Arbitrary externally supplied
weights must not silently be ignored merely because the macro is present.

## Measured evidence

A temporary benchmark used the current headers, optimized Apple Clang
(`-O3 -DNDEBUG -std=c++17`), double precision, a degree-2 20x20x40 control
lattice, two output components, and deterministic unstructured parameters.
Setup was excluded. Reported times are medians of three warm runs.

| Configuration | Points | Existing DecodePointSet | Existing FastVolPt per component | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Serial | 100,000 | 58.15 ms | 6.19 ms | 9.4x |
| TBB, limit 1 | 1,000,000 | 583.96 ms | 62.42 ms | 9.4x |
| TBB, limit 18 | 1,000,000 | 129.46 ms | 6.48 ms | 20.0x |

Compiling the same synthetic model/parameters against the saved application's
installed overlay-enabled MFA headers gave 20.817 s at one thread and 5.068 s
at an 18-thread limit for one million points. The latter is about 39x slower
than the current single-tensor baseline. Its throughput is consistent with
the recorded 167.752 s for 34.1 million target points, although this is not a
full-application reproduction. Report this configuration effect separately
from the fast-path improvement. The overlay comparison used `-O3` too; the
saved application itself used `-O2`.
An additional 10,000-point comparison of the installed overlay path and the
current single-tensor path found a maximum absolute difference of
`7.78e-16` across 20,000 output components. This validates the synthetic
comparison only; full remap and nonuniform-knot comparisons remain required.

The adapter copied scalar control columns once before timing and called the
unchanged scalar fast kernel twice per point, including repeated basis
evaluation. It is an investigation aid, not the proposed production interface.
TBB adapter ranges used grain size 4096 and range-local scratch; the baseline
uses the existing affinity-partitioned loop. Thus the comparison includes
scheduling/workspace differences, not only evaluator selection. Maximum
absolute output difference was `7.78e-16`.

Counting C++ `operator new` calls in a separate serial pass found approximately
ten allocations per point in the existing path and zero in the fast loop
after setup. This excludes Eigen allocations that use `malloc` directly.
Nine calls come from three temporary vectors per `OrigBasisFuns` call across
three dimensions; another comes from the control-support `VolIterator`.
Additional Eigen support-iterator storage is also allocated per point.

Bypassing the unstructured `PtIterator` while retaining `VolPt` took 58.24 ms,
essentially the same as the serial baseline. Its by-value prefix increment is
undesirable, especially for structured iterators, but is not the main measured
unstructured bottleneck. Do not center the first patch on iterator cleanup.

These timings use synthetic points, uniform knots, and synthetic coefficients,
with overlays disabled. They demonstrate headroom within the requested
single-tensor configuration; they do not predict the Gulf Stream speedup or
measure the configuration change from its installed overlay evaluator.
Temporary benchmark sources and outputs are under `/tmp/mfa-decode-investigation/`.

## Implementation sequence

### 0. Establish a matching application baseline

Record compiler flags and the actual included MFA headers. Rebuild the
application with T-mesh and overlays disabled before attributing performance
to decoder refactoring. Compare this configuration-only run against the saved
overlay run for both timings and numerical output, then use it as the baseline
for the fast-path patch. This separates removal of overlay machinery from
improvements to the single-tensor decoder. Verify that the same single-tensor
model and supplied custom knot vectors are used in both cases.

### 1. Add the value-only path for unstructured scalar AND vector models

Make algorithm selection once per `DecodePointSet` call. The initial fast
eligibility predicate is:

- Single tensor representation: neither T-mesh nor overlays enabled and one tensor.
- `MFA_NO_WEIGHTS` build with unit stored weights; verify the weight invariant
  once per call, outside the point loop, and fall back for nonunit weights.
- Empty or correctly sized all-zero derivative request.
- Unstructured parameters, covering the actual remapping workload immediately.

The conservative unit-weight check is linear in control count, not point
count. A future runtime-unit-weight path in builds without `MFA_NO_WEIGHTS`
would also need to revise the fast routines' current macro checks; it is not
obtained merely by adding a dispatch condition.

Reuse `FastVolPt` directly for scalar models. For vector models, factor its
existing basis/span preparation and scalar contraction into private helpers:

1. Compute spans, local basis values, and support start once for a point.
2. Apply the existing scalar contraction to each control-point column using
   a borrowed contiguous column pointer/view.
3. Write each scalar result directly to the matching output column.

Keep `FastVolPt`'s existing public signature as a wrapper around the same
helpers. This is an interface/scratch refactor of the existing arithmetic,
not a new contraction algorithm. Do not copy whole models or control-point
columns, build scalar PointSets, or gather vector-valued control rows inside
the loop. The prototype's scalar-column copies are unnecessary in production.

Suggested internal boundaries, with names adapted to the surrounding style:

- `can_decode_values_fast`: checks representation, weights, derivative request,
  and parameter layout once per call.
- `decode_point_set_values`: selects serial/TBB scheduling and owns workspace
  lifetime; a shared `decode_value_range` performs the point traversal.
- `prepare_fast_point`: fills spans, compact basis values, and support start
  using `FastDecodeInfo` and `BasisFunInfo`.
- `contract_fast_component`: runs the existing contraction against a borrowed
  control-point column and returns one scalar.

Keep generic decoding behind the existing fallback helpers initially. Share
range traversal where it makes the code clearer, without introducing an
evaluator class hierarchy or requiring changes to application callers.

### 2. Share traversal and minimize workspace

Retain serial/TBB scheduling boundaries, with a common internal range worker
and an evaluator-specific workspace. Select scalar/vector/legacy behavior
outside the hot loop; avoid virtual dispatch or `std::function` per point.

The intended control flow is: validate the call; select representation and
parameter layout; select value/derivative evaluation; allocate only that
workspace; run serial or TBB ranges. Put these decisions in named internal
helpers rather than interleaving representation, threading, and derivative
conditionals inside the point loop.

- Allocate parameter and contraction scratch per worker or coarse range.
- Resolve output-column pointers once; disjoint point ranges write disjoint
  elements of each column.
- Read the three parameter values into reusable storage; this small copy is
  acceptable. Passing an Eigen row expression to a `VectorX` argument can
  create a temporary and is not automatically an improvement.
- Allocate `FastDecodeInfo::td`, derivative tables, and pointer aliases only
  when derivatives are requested. The current constructor eagerly allocates
  gradient intermediate storage even for values.
- Review copy/move ownership before storing derivative scratch in TBB TLS:
  `FastDecodeInfo::M` owns raw alias arrays, so default copies are unsafe once
  initialized. Delete copying or provide correct ownership semantics.
- Use a TBB initializer factory capturing model/decoder references rather
  than forwarding the full model as a stored constructor argument. The
  installed oneTBB `construct_by_args` decays and copies its arguments.
- Remove unused `ijk` vectors in unstructured loops; keep progress/logging
  outside per-point checks wherever the reporting contract permits.
- Tune range size only after the allocation-free path exists. Compare one
  and multiple thread counts to distinguish useful parallelism from allocator
  contention and scheduler overhead.

Keep existing high-level wrappers. Persistent decoder caches would introduce
invalidation and thread-safety obligations and are unnecessary for this first
batch-decoding improvement.

### 3. Simplify structured/grid setup separately

The current serial scalar `DecodeGrid` precomputes dense basis matrices and
constructs both `DecodeInfo` and `FastDecodeInfo`, then calls `FastVolPt`,
which recomputes bases and does not use the saved matrices. Select the path
before allocating its workspace; eliminate unused precomputation for that
scalar fast route. Keep vector saved-basis decoding as a measured alternative
rather than assuming per-point evaluation is best for every structured grid.

Eventually let explicit `DecodeGrid` generate uniform parameter arrays and
delegate to a structured evaluator that consumes per-axis coordinates.
`DecodePointSet` should consume its supplied coordinates. Cache only local
`p+1` basis values and spans for structured axes if benchmarks justify it;
do not build dense point-by-control tables for the unstructured Gulf Stream
target. At its size, even 3x3 local double basis values plus three int spans
would consume about 2.9 GB for a whole-target cache.

There is an existing correctness discrepancy: serial/Kokkos structured
`DecodePointSet` with empty derivatives reconstructs a uniform grid, while
TBB and all-zero derivative requests use actual coordinates. Preserve those
legacy routes in the initial unstructured performance patch. Correct
nonuniform structured behavior in an explicitly identified follow-up with
dedicated tests; do not hide a sampling change inside dispatch cleanup.

### 4. Remove derivative-path allocation overhead

Retain the generic path for arbitrary derivative orders and vector outputs.
Move request validation and fixed workspace setup outside point loops. Reuse
the control-support iterator and its coordinate storage rather than constructing
it for every point. Supply reusable basis/derivative scratch to existing
evaluation routines instead of allocating `ndu`, `a`, `left`, and `right`
on every derivative evaluation.

The existing `BasisFunInfo` and `FastBasisFunsDers` are useful building blocks,
but preserve the existing contraction and derivative-order semantics. Cover
mixed partials and orders exceeding the spline degree. Record derivative
workspace order accurately (`ResizeDers` currently shadows the `nders` member).

Do not automatically route a request for one partial derivative through
`FastGrad`: it computes all first partials. Benchmark that tradeoff first.
`FastGrad` is appropriate when a caller actually needs a scalar gradient,
optionally together with its value; current point-set APIs request a single
multi-index derivative. Weighted derivative limitations remain separate work.

## Validation and acceptance

1. Keep the current generic evaluator as a numerical reference during rollout.
   Compare scalar and multi-component outputs, float/double, 1D–3D, anisotropic
   degrees/control counts, nonuniform knots, boundaries/repeated knots, empty
   inputs, singleton axes, empty/zero derivatives, mixed partials, and invalid
   derivative/output layouts. Use explicit analytic expectations as well.
2. Verify untouched destination columns, component order, parameter ownership,
   repeated calls, and concurrent decoding with independent workspaces.
3. Test serial and TBB fast paths and weighted/T-mesh/overlay fallback builds.
   Preserve existing high-level geometry-versus-science derivative behavior.
4. Require zero allocations proportional to point count in the supported
   value-only inner loop after worker/range setup. Count both C++ allocations
   and Eigen/malloc allocations in validation.
5. Benchmark the actual packed two-component model, not just scalar examples.
   Retain small reproducible point-set benchmarks and report setup separately
   from evaluation, multiple repetitions, thread counts, and peak memory.
6. Rebuild/install the remapper through its managed Spack workflow. Rebuild
   consuming applications because MFA is header based; verify include paths,
   defines, and executable provenance. Run from a dedicated output directory.
7. Reproduce the same Gulf Stream 20x20x40 and 60x40x40 cases with the same
   inputs and thread limit. Time source decode, target decode, fit, and total
   separately. Compare every output field against the existing saved remaps
   with justified floating-point tolerances, including finiteness, field
   ordering, ranges, and reconstruction metrics. Bitwise identity is not
   expected when changing accumulation order.

The first implementation milestone is a measured improvement for unstructured
two-component value decoding with no public API change and no point-count-
proportional allocations. Structured semantics and derivative improvements are
follow-up milestones, not prerequisites for accelerating this workflow.
