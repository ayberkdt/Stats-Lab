# Session: Finite Random Variables and Joint Laws

- Date: 2026-10-09
- Target release: `0.3.0`

## Completed

- Added immutable finite random variables with complete read-only mappings.
- Added induced PMFs, CDFs, preimages, predicate events, transformations, and
  structural versus positive support.
- Added expectation, raw and central moments, variance, and standard deviation.
- Added joint PMFs, both marginals, conditional PMFs, independence testing,
  covariance, and correlation.
- Added categorical/numeric domain separation and exact shared-measure ownership.
- Added PMF and joint-law visualizations, equations, conceptual documentation,
  roadmap state, changelog entry, and ADR-0003.

## Mathematical boundaries

- The finite source sigma-algebra remains the full power set.
- Mapped values must be hashable; numeric operations additionally require finite
  real values.
- Joint variables must share measure identity, not merely equal weights.
- Structural support can include zero-mass values; positive support cannot.
- Floating high-order moments can overflow or lose precision; arbitrary-
  precision backends remain future work.

## Verification

- `python -m unittest discover -s tests -v`: 41 tests passed.
- `ruff check .`: all checks passed.
- `python -m compileall -q src tests examples`: passed.
- Headless figure generation: six PNG files produced; the new PMF and joint-PMF
  figures were visually inspected, and heatmap annotation contrast was repaired.
- A `probstat_library-0.3.0-py3-none-any.whl` wheel was built with isolated
  intermediate tooling, installed into a clean temporary venv, imported, and
  smoke-tested at version `0.3.0`.
- `mypy` remains unavailable in the normal host environment; no type-check
  result is claimed.
- Final tests are repeated from the D-drive repository after deployment.

## Next

Implement finite PGFs, MGFs, and characteristic functions with documented
domains and numerical limitations, then design the common distribution protocol.

