# Session: UQ-1 Modeling and Statistical Fitting

- Date: 2026-10-09
- Release: 0.8.0

## Delivered

- Added Lognormal and Weibull distribution families with stable tails,
  quantiles, moments, and reproducible sampling.
- Added a dependency-free positive-definite multivariate Normal with Cholesky
  density evaluation and sampling.
- Added Normal/Lognormal MLE, one-sample KS and AD diagnostics, z/approximate
  mean intervals, and Wilson binomial intervals.
- Added a fitted-CDF plot, reproducible lognormal-fitting laboratory, complete
  equation records, sources, and numerical tests.

## Explicit limitations

- Singular Gaussian measures are rejected; no jitter is silently added.
- The KS p-value is not valid after fitting on the same sample without
  correction or bootstrap.
- Generic Anderson-Darling reports no universal p-value because calibration is
  family- and estimation-dependent.
- Unknown-scale mean intervals are labeled normal approximations, not exact
  Student-t intervals.

## Next step

Finish UQ-1 with general multivariate protocols, Weibull MLE, Student-t
intervals, fitted-null bootstrap calibration, stream splitting, and a real-data
case study. Then implement UQ-2 quadrature and Monte Carlo estimator contracts.
