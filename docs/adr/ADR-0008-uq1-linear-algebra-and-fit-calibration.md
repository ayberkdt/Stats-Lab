# ADR-0008: UQ-1 Linear Algebra and Fit Calibration Boundaries

- Status: Accepted
- Date: 2026-10-09

## Context

Multivariate Gaussian modeling needs matrix factorization, while statistical
fitting needs estimators, confidence procedures, and goodness-of-fit tests.
Adding a large numerical runtime dependency at this stage would alter the
repository's dependency policy. Treating every fitted-null test like a
fully-specified-null test would also produce statistically invalid p-values.

## Decision

The first multivariate Normal implementation uses a small, cited Cholesky
factorization and forward solve in `probstat.multivariate`. It accepts only
symmetric positive-definite covariance matrices. It does not add jitter,
pseudoinvert singular matrices, or pretend a singular Gaussian has an ordinary
ambient density.

`probstat.inference` owns estimators, confidence procedures, and
goodness-of-fit diagnostics. It may depend on distributions and foundations but
not on UQ workflows or visualization. Generic one-sample KS exposes an
asymptotic fully-specified-null p-value with a warning in its API contract.
Generic Anderson-Darling exposes the statistic without a universal p-value,
because valid calibration depends on the null family and parameter-estimation
procedure.

Unknown-scale mean intervals are labeled normal approximations until a cited
Student-t distribution and quantile are available. Wilson score intervals are
used for binomial probabilities rather than the unstable uncorrected Wald
interval.

## Consequences

- Base installation remains dependency free.
- Covariance errors fail explicitly rather than being silently regularized.
- Larger dense problems will eventually require a numerical backend ADR and
  reference comparisons.
- Family-specific fitted-null corrections and parametric bootstrap remain
  required before the goodness-of-fit curriculum row can be complete.
- Presentation remains above inference; fitted-CDF plots cannot define or
  modify mathematical test results.
