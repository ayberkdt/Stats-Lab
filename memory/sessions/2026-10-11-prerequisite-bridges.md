# Session — 2026-10-11 — PRE-1 prerequisite bridges

## Delivered

- `linalg.determinant`, `linalg.log_abs_determinant` (partial-pivot LU,
  [EQ-LINALG-008]).
- `MultivariateNormal.marginal_subset` and `MultivariateNormal.conditional`
  ([EQ-DIST-MVN-006], [EQ-DIST-MVN-007]).
- New module `probstat.multivariate.transforms`: `transformed_density`,
  `TransformedDensity` ([EQ-MVN-JAC-001]), `delta_method`,
  `DeltaMethodResult` ([EQ-MVN-DELTA-001]).
- Guides in `docs/prerequisites/`: linear algebra, optimization,
  multivariable calculus and conditioning, probability and statistics
  refresher, R bridge. A "Prerequisites" section in `docs/learning-path.md`.
- Five equation records and five bibliography entries (Strang 2016,
  Magnus-Neudecker 2019, Box-Muller 1958, Genz-Bretz 2009, R Core Team).
- ADR-0028. ATHENS-10.1, 10.2, 10.3, 10.5, 10.6, 10.7 now `Implemented`;
  10.7 owner renamed to `docs.prerequisites.r-bridge`.

## Verification

- `tests/test_prerequisite_bridges.py` (19 tests), `tests/test_prerequisite_guides.py`
  (every guide snippet re-executed and its output compared),
  `tests/test_r_bridge_examples.py` (quoted R values plus independent
  standard-library checks): about 3 s together.
- Full suite: 467 passed. `ruff check .` and `mypy src/probstat` are clean.

## Open items and next safe step

- R was not available; R defaults were read from the `stats` R sources
  (development trunk) and quoted values were computed with mpmath/SciPy and
  closed forms, not taken from an R session.
- Chapter/section locators for Magnus-Neudecker (Chapters 5 and 9),
  Casella-Berger (Sections 2.1, 4.3, 4.6, 5.5.4), and Strang (Chapters 5, 6)
  were cited from memory and not checked against the books in this session.
- Next: a multivariate `pmvnorm` analogue (Genz's method) beyond two
  dimensions, and multiple-predictor regression for the `lm` row.
