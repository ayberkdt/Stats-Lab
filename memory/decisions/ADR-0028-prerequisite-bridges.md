# ADR-0028 — Prerequisite bridges with executable evidence

- Status: Accepted
- Date: 2026-10-11

## Context

The ATHENS prerequisite rows (10.1-10.7) were Partial or Planned. The library
already had probability, statistics, dense linear algebra, and optimization
code, but no guide tied them to the prerequisite outcomes, nothing implemented
Gaussian conditioning on a partition, change of variables, or the delta
method, and there was no route for readers coming from R.

## Decision

1. `linalg` gains `determinant` and `log_abs_determinant` from a
   partial-pivot LU factorization [EQ-LINALG-008]. Only an exactly zero pivot
   column is singular, unlike `lu_solve`'s relative threshold, so uniformly
   small non-singular matrices keep their determinant.
2. `MultivariateNormal` gains `marginal_subset(indices)` [EQ-DIST-MVN-006]
   and `conditional(indices, values)` [EQ-DIST-MVN-007]. The conditional uses
   `W = L^-1 Sigma_ba` with `Sigma_bb = L L'`, so its covariance
   `Sigma_aa - W'W` is symmetric by construction and no inverse is formed.
   Free components keep ascending index order.
3. Change of variables (`transformed_density`, `TransformedDensity`
   [EQ-MVN-JAC-001]) and the delta method (`delta_method`,
   `DeltaMethodResult` [EQ-MVN-DELTA-001]) live in a new module
   `probstat.multivariate.transforms` (layer 2). They need
   `optimization.central_jacobian` (layer 1), which rules out `numerics`
   (layer 0), and they are statements about random vectors, which rules out
   `optimization`.
4. Five guides under `docs/prerequisites/` follow one contract: every
   ```` ```python ```` block is immediately followed by a ```` ```text ````
   block with its exact output, and `tests/test_prerequisite_guides.py`
   re-executes them in order. Snippets print with limited precision, or use
   only `+ - * /`, so their output does not depend on platform `libm`
   rounding.
5. The R bridge is documentation plus tests, not bindings. R was not run;
   R defaults were read from the `stats` R sources and quoted values were
   computed independently (mpmath, SciPy, closed forms).
   `tests/test_r_bridge_examples.py` re-checks them with standard-library-only
   computations where possible. The coverage owner for ATHENS-10.7 is renamed
   from `bindings.r` to `docs.prerequisites.r-bridge` to describe what exists.

## Consequences

- ATHENS-10.1, 10.2, 10.3, 10.5, 10.6, and 10.7 are Implemented with code,
  guide, and test evidence.
- A guide edit that changes printed numbers fails the test suite until the
  documented output is regenerated, so guides cannot silently drift.
- `delta_method` is first order only; the guide and tests show the neglected
  curvature bias explicitly.

## Rejected alternatives

- R bindings (`rpy2` or an R package wrapping `probstat`): would add runtime
  dependencies and an R installation to CI for a teaching bridge.
- Doctest-style `>>>` snippets: harder to read as prose and awkward for
  multi-line output; the python/text block pair keeps guides readable.
- Returning the conditional law in caller-specified order: ascending order is
  unambiguous and matches how `marginal_subset` can then reorder if needed.
