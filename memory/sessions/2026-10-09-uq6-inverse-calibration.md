# Session — 2026-10-09 — UQ-6 inverse problems and Bayesian calibration

## Delivered

- Added `probstat.inference.regression` with intercept/slope ordinary least
  squares, residuals, SSE/SST, unbiased residual variance, `R^2`, and prediction.
- Added `probstat.inference.bayesian` with Gaussian likelihood, general
  linear-Gaussian conditioning, seeded posterior sampling, joint predictive
  distributions, variance-component separation, and predictive checks.
- Added `probstat.inverse` observation models, residual/RMSE behavior,
  identifiability diagnostics, calibration orchestration, immutable calibration
  inputs, and explicit prediction-target vocabulary.
- Added `probstat.validation.AdditiveGaussianDiscrepancy` with required
  description and applicability domain.
- Added a cited foundation guide, fifteen equation records, six bibliography
  entries, six regression tests, and ADR-0013.
- Added a seeded Bayesian-calibration lab and generated
  `examples/generated/bayesian-calibration-prediction.png`.
- Updated architecture, learning path, README, changelog, roadmap, memory,
  curriculum ledger, package version, and citation metadata to `0.13.0`.

## Numerical laboratory result

- OLS intercept/slope: `(1.37911, 0.76758)`.
- Posterior mean: `(1.3782529979186293, 0.7675856290195453)`.
- Design Gram condition number: `63.17465`.
- Seeded posterior-predictive tail diagnostic: `0.71050` with 2,000 draws.
- The generated figure visually separates parameter uncertainty from the wider
  future-observation band and marks the calibration-domain boundary at `x=7`.

## Verification

- Ruff: clean across source, tests, and examples.
- Strict mypy: clean across 54 package source files.
- Unit tests: 139 passed, including architecture, citation, equation,
  curriculum, analytic-conditioning, identifiability, and reproducibility
  contracts.
- Laboratory: completed and generated a visually inspected PNG.
- Packaging: isolated `0.13.0` wheel built successfully and passed a true
  `python -I` zip-import smoke test for version, inverse, and validation APIs.
- Wheel SHA-256:
  `DED6FACF4D8D43291C94974035F1827000AD06560848204078E8F9702B9CEDB8`.

## Coverage change

All `ATHENS-7.1` through `ATHENS-7.15` rows and the `ATHENS-9.5` Bayesian
calibration lab now have implemented evidence. The canonical ledger is now 74
implemented, 17 partial, and 19 planned topics.

## Next safe step

Implement UQ-7 state-space and data-assimilation foundations. Begin with one
immutable linear-Gaussian state-space contract and a Kalman predict/update
kernel, then add smoothing, nonlinear/ensemble/particle variants without
mixing their approximation assumptions.
