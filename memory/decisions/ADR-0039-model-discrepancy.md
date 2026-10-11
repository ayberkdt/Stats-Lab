# ADR-0039 — Kennedy-O'Hagan model discrepancy and heteroscedastic noise

- Status: Accepted
- Date: 2026-10-12

## Context

`inverse.CalibrationProblem` treats model inadequacy as an independent scalar
variance (`NoiseModel.discrepancy_variance`). `validation.AdditiveGaussianDiscrepancy`
only records that assumption. The roadmap lists "Heteroscedastic noise,
correlated discrepancy processes" as open. Gaussian-process regression lives
in `surrogates` (layer 5). `inverse` is layer 4, so it cannot import
`surrogates`; `surrogates` itself imports `propagation` (layer 4).

## Decision

1. **New module `inverse.discrepancy` (layer 4).** It imports `linalg`,
   `optimization`, `inference` (MCMC, Laplace, HMC/NUTS), `modeling`,
   `statistics`, and `foundations`, plus `inverse.nonlinear.Experiment`. It
   implements its own small kernel (`discrepancy_kernel`, squared exponential
   and Matern 5/2) instead of importing `surrogates.kernel_value`. That
   import would point upward, and the duplicated part is two closed-form
   expressions.
2. **A separate problem class, not an extension of `CalibrationProblem`.**
   `DiscrepancyCalibration` handles one observed quantity and reuses
   `Experiment` and `ProbabilisticModel`.
   - Extending `CalibrationProblem` would have changed its parameter layout
     and its multi-quantity likelihood, which `workflow.UQStudy` depends on.
   - With `discrepancy=None` (or a fixed zero standard deviation) and known
     noise, it reproduces `CalibrationProblem` exactly: log posterior, MAP,
     Laplace covariance, and seeded chains. It uses the same Nataf sampling
     coordinates, the same stream children, and the same chain names. A test
     enforces this.
3. **Marginal likelihood by Cholesky; diagonal fast path.** `delta` is
   integrated out analytically.
   - Without an active discrepancy the covariance is diagonal and the `O(n)`
     sum is used. This keeps heteroscedastic fits with hundreds of points
     fast.
   - A failed factorization returns `-inf`. No jitter is added silently;
     positive noise keeps the matrix positive definite in exact arithmetic.
4. **Parameterization.**
   - Every hyperparameter is either fixed or given a prior, like
     `NoiseModel`.
   - Free `sigma_delta` and length scales are sampled on the log scale, with
     the Jacobian in the prior. Priors are on the standard deviation,
     matching `NoiseModel` and `AdditiveGaussianDiscrepancy`.
   - Log-linear noise coefficients `(a, b)` are unconstrained and need no
     transform.
5. **Point estimation via BFGS.** `fit(objective="posterior"|"likelihood")`
   uses the inverse finite-difference Hessian as covariance, and `None` when
   the Hessian is not positive definite. `fit` exposes MLE as well as MAP
   because the heteroscedastic noise model is commonly estimated by maximum
   likelihood (Harvey 1976).
6. **Predictions keep the simulator and the corrected process apart.**
   - `predict` stores, per draw, `eta_s`, the conditional discrepancy mean
     and variance, and the noise variance.
   - Corrected intervals invert the exact Gaussian-mixture CDF by bisection,
     so they are deterministic and need no seed.
   - Simulator-only intervals are type-7 sample quantiles, the same as
     `inverse.predict`.
7. **The identifiability report shows symptoms, not proof.** It reports
   `theta`-hyperparameter posterior correlations and posterior/prior
   standard-deviation ratios, with a warning that cites Brynjarsdottir and
   O'Hagan (2014). The lab shows that a confounded discrepancy can stay
   below the correlation threshold, and the documentation says so.

## Consequences

- Users can calibrate a known-wrong simulator and see how much of each
  prediction is discrepancy. Known per-observation noise and log-linear
  noise are available with or without a discrepancy.
- Not provided: an emulator GP for expensive simulators, a scale factor
  `rho`, a non-zero discrepancy mean, multi-output discrepancy, joint
  predictive draws, and analytic gradients. HMC/NUTS use central
  differences.
- Cost is `O(n^3)` per likelihood evaluation in pure Python, which suits
  tens of field observations.
- New references: `REF-HIGDON-ETAL-2004` and `REF-HARVEY-1976`, both checked
  against Crossref. New equations: `EQ-KOH-*` (seven records).
- `README.md`, `CHANGELOG.md`, `memory/ROADMAP.md`, and
  `docs/architecture.md` are left for the integrating change.
