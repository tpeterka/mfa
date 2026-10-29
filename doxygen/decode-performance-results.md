# Point-set value decoding: implementation and validation

The public MFA API is unchanged. Unstructured point sets with one tensor,
unit stored weights, `MFA_NO_WEIGHTS`, and empty/all-zero derivatives now use
the existing fast contractions for every output component. Spans and local
basis values are prepared once per point. Control/output columns are borrowed;
there are no scalar-model copies, control-row gathers, or point-local scratch
allocations. Serial and TBB use the same range worker.

`FastVolPt` retains its scalar interface and shares these internal helpers.
`FastDecodeInfo` allocates gradient intermediates when `ResizeDers` is called,
and disallows unsafe copies of its owning pointer aliases. TBB workspace
factories capture references instead of copying the complete model into a
constructor argument pack. The generic derivative/weighted evaluators and
existing structured parameter routing are retained.

The previously prepared decoder extraction and five encoder guard fixes are
included because the single-tensor build depends on them. The checkout's
overlay-disabled configuration is retained. The bundled Catch2 debugger trap
also has an ARM macOS branch, needed to compile the new tests on Apple Silicon.

## Full Gulf Stream validation

The managed `mfa-remap-env` Spack environment was activated with the application's
`load-env.sh` and rebuilt with `spack install` for both configurations. The
registered MFA develop checkout had the same base revision as this checkout,
`7e68adda08b57a6ae3f44ea7a73fb74c4f09c7eb`; the changed MFA headers were
synchronized there for each build. Installed executables were used. No remapper
source or input data changed.

Both phases used the original run-manifest arguments: degree 2, regularization
1, two packed science components, custom vertical knots, the same projected
Gulf Stream input, and `MFA_NUM_THREADS=18`. The baseline includes the earlier
extraction/guard fixes and disabled overlays, without the new fast path.
Application builds used `RelWithDebInfo` (`-O2 -g -DNDEBUG`). Times below are
single complete application runs, not repeated benchmark medians.

| Controls | Saved overlay target decode | Single-tensor baseline target decode | New target decode | Improvement over baseline |
| --- | ---: | ---: | ---: | ---: |
| 20x20x40 | 167.752 s | 4.904 s | 0.769 s | 6.38x |
| 60x40x40 | 193.746 s | 4.152 s | 0.850 s | 4.88x |

Source reconstruction decoding changed from 0.178 to 0.024 s and from 0.158 to
0.030 s, respectively. Encoding was stable: 4.961 vs 4.947 s and 44.245 vs
44.185 s. Initialization plus forward remap/output changed from 15.721 to
10.341 s and from 53.103 to 49.505 s. Target decoding no longer dominates;
encoding dominates the larger case.

The installed `compare_forward` checked both target fields at all 34,134,100
samples per field in each case, including finiteness and field ranges. Maximum
absolute differences from each single-tensor baseline were:

| Controls | Temperature | Salinity |
| --- | ---: | ---: |
| 20x20x40 | 1.07e-14 | 4.27e-14 |
| 60x40x40 | 1.43e-14 | 3.56e-14 |

Configuration-only baseline comparisons against the saved overlay outputs
were within 5.69e-14. Source reconstruction metric CSVs each had 324 matching
rows; sample counts and out-of-range counts/fractions were unchanged. Maximum
RMSE changes were 4.42e-15 and 5.93e-16. Small rounding differences are expected
because unit-weight denominator arithmetic is omitted in the fast path.

Run logs, command manifests with executable hashes, comparison CSVs, sanitizer
logs, and allocation instrumentation are in `/tmp/mfa-pointset-validation/`.
Original saved results remain in the remapping project's
`runs/gulfstream-20x20x40-degree2-reg1-20261001/` directory.

## Regression and allocation checks

- New `decode-pointset-test`: 42,129 assertions, passing in serial and TBB,
  and under AddressSanitizer/UndefinedBehaviorSanitizer. Tests cover float and
  double, domain dimensions 1–3, 1/2/4 output components, degrees 0–4 across
  anisotropic axes, nonuniform/repeated knots, boundaries, empty/zero derivative
  requests, first-derivative and nonunit-weight fallbacks, untouched columns,
  parameter preservation, empty/invalid input, repeated/concurrent calls, and
  the existing scalar `FastVolPt`/positive-degree `FastGrad` interfaces.
- Existing fixed sinc structured/grid/unstructured/random tests and
  `derivative-model-test`: all five pass in each serial/TBB build.
- Separate T-mesh and overlay configurations pass constant vector point-set
  fallback smoke tests. Kokkos was not built in this environment.
- Darwin allocation interposition counted malloc/calloc/realloc calls around
  complete serial point-set calls, including Eigen allocations: **21 calls**
  for 1,000, 100,000, and 1,000,000 points. Allocation count is independent of
  point count. The scalar fast kernel also passes Eigen's no-malloc assertion
  after workspace construction.

## Reproducible local benchmark

Build the `decode_pointset_speed` example with the desired serial/TBB backend,
overlays/T-mesh disabled, and weights disabled. For example:

```sh
cmake --build build --target decode_pointset_speed
build/examples/eval/decode_pointset_speed 1000000 18
```

It uses a 20x20x40 degree-2 lattice, two output components, nonuniform vertical
knots, and deterministic unstructured points. Setup is excluded; each evaluator
is warmed and timed three times, reporting the median and maximum difference
against generic `VolPt` ranges. Generic scratch is range-local; fast scratch
is worker-local, so the comparison includes traversal/workspace differences.
Optimized Apple Clang measurements (`-O3 -DNDEBUG`) were:

| Backend | Generic VolPt ranges | New DecodePointSet | Ratio |
| --- | ---: | ---: | ---: |
| Serial | 656.48 ms | 49.98 ms | 13.1x |
| TBB, limit 1 | 681.77 ms | 54.89 ms | 12.4x |
| TBB, limit 18 | 115.38 ms | 12.18 ms | 9.5x |

Maximum absolute output difference was 7.78e-16. Synthetic speedups should
not be substituted for the full-application measurements above.

## Deferred work

Derivative kernel/workspace refactoring and structured-grid coordinate
semantics are not part of this change. Testing also exposed an existing
`FastGrad` failure for degree-zero splines; review that case during the
derivative follow-up. The current tests exercise its positive-degree contract.
Runtime unit weights in builds without `MFA_NO_WEIGHTS` continue to use the
generic path. No whole-point-set basis cache or persistent model cache was added.
