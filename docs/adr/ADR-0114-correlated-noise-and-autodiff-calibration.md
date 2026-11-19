# ADR-0114 — Correlated measurement errors and reverse-mode gradients in calibration

- Status: Accepted
- Date: 2026-11-19
- Extends: ADR-0017 (nonlinear calibration and MCMC), ADR-0065 and ADR-0069
  (automatic differentiation), ADR-0090 (differentiable distributions), and
  ADR-0111 (pCN and SMC in calibration)

## Context

`CalibrationProblem` assumed independent measurement errors. Repeated
measurements of one quantity along a time axis, a load ramp, or a traverse
often share drift, warm-up, or alignment errors, so neighbouring errors are
correlated. Treating them as independent overstates the information in the
data: on the straight-line problem of the tests (13 points, exponential
correlation of length 0.6 over `x`, noise 0.3, seed 5) the independent
posterior standard deviations of intercept and slope are `0.156` and
`0.088`, against the exact `0.254` and `0.132`, and the means move by 0.35
and 0.48 exact standard deviations. The foundations guide listed
correlated measurement-error models as open.

Gradient-based samplers in calibration (`method="hmc"` or `"nuts"`) took
central-difference gradients of `standard_log_posterior`: `2n + 1` posterior
evaluations, each one forward-model run per experiment, per leapfrog step.
Reverse-mode automatic differentiation (ADR-0069) gives the gradient from
one recording, but the calibration log-densities could not be traced: they
validate with float-only helpers, and the Nataf map of the prior calls
float-only quantile functions (ADR-0090, Decision 4).

## Decision

1. **`ErrorCorrelation` on `NoiseModel`.** A new field
   `NoiseModel(..., correlation=None)` declares, per observed quantity, that
   its errors are jointly Gaussian across the experiments that observe it,
   with covariance `Sigma_q = sigma_q^2 M_q + sigma_delta_q^2 I`. `M_q` is
   either a declared symmetric positive-definite matrix
   (`ErrorCorrelation(matrix=M)`, rows in experiment order; a fully known
   covariance is `matrix=C` with `standard_deviation=1.0`) or a stationary
   kernel over a declared numeric condition: exponential, Matern 3/2 or
   Matern 5/2 ([EQ-CAL-CORR-001]) with a known `length` or a `length_prior`.
   `sigma_q` keeps its meaning (known or uncertain) and the discrepancy
   variance acts as a nugget. The kernels are new code rather than
   `discrepancy_kernel`, which offers the squared exponential and Matern 5/2
   on vector inputs with a variance factor; measurement errors need the
   rougher exponential and Matern 3/2 and a unit-variance correlation on one
   condition. A default `correlation=None` keeps every existing object and
   number: the likelihood, least-squares, and prediction code take the
   unchanged independent path when no quantity is correlated (tested with
   exact equality).
2. **Gaussian only, explicit errors.** A Student-t law with correlation
   raises `ValueError` when the `NoiseModel` is built: a multivariate t
   would couple all errors through one scale and is a different model. A
   declared matrix that is not symmetric or not positive definite, a matrix
   of the wrong size, a missing condition, a condition that names a
   parameter, and repeated condition values without a nugget raise when the
   problem is built; a known kernel matrix that is numerically singular (a
   length long compared with the spacing) raises with that diagnosis. A
   sampled length whose covariance is not numerically positive definite
   gives log-likelihood `-inf`, the convention of `DiscrepancyCalibration`
   for the same situation, so a chain is not aborted by one proposal.
3. **Cholesky likelihood and whitening.** The correlated term is
   `-0.5 (n log 2 pi + log det Sigma + r' Sigma^-1 r)` through
   `linalg.cholesky` and `solve_lower` ([EQ-CAL-CORR-002]); for a known `M`
   without nugget the factor of `M` is computed once and the evaluation
   costs one triangular solve. `fit_least_squares`, `identifiability`, and
   the new `whitened_residuals` use `e = L^-1 r` ([EQ-CAL-CORR-003]), so
   least squares is generalized least squares and the Fisher matrix is
   `J' Sigma^-1 J`. An unknown scale of a correlated quantity is updated
   from the generalized residual sum of squares; least squares refuses an
   uncertain length and an unknown correlated scale with a nugget (neither
   has a closed-form update) and points to the posterior methods.
4. **Lengths are noise parameters.** An uncertain length is sampled as
   `log ell` after the log noise scales; it appears as `log_length[q]` in
   `vector_names`, `length[q]` in `noise_names` and the physical chains, and
   `w[length[q]]` in the fully standardized coordinates of pCN and SMC,
   where it is mapped through its prior quantile function like a noise
   scale (ADR-0111). `noise_names` therefore lists lengths too; it is
   unchanged for problems without uncertain lengths.
5. **Prediction.** For a kernel correlation, `predict(include_noise=True)`
   draws one joint error vector per posterior draw across the prediction
   conditions from the same kernel, as a new measurement series independent
   of the calibration series (it is not conditioned on the calibration
   residuals, which would require the cross-covariance between old and new
   errors, i.e. the assumption that the new measurements continue the same
   series). A declared matrix does not define errors at new conditions, so
   noise draws for it raise unless `noise_standard_deviations` gives an
   independent scale. The random stream is consumed differently only when a
   correlated quantity is requested with noise.
6. **`gradient="central" | "reverse"`.** `laplace_posterior` and
   `sample_posterior(method="hmc" | "nuts")` take the option; the default
   `"central"` makes exactly the earlier calls. `"reverse"` uses private
   traceable twins of `standard_log_posterior` and `log_posterior` that
   accept `numerics.ReverseVariable` coordinates and return the float
   functions' values for floats. The forward model is called with variables
   and must be written with operators and `numerics.admath`. The prior map
   `theta = T^-1(u)` takes its value from the float Nataf map and its first
   derivative from the inverse-function rule `phi(z_i) / f_i(x_i) L_ik`,
   attached by `x_0 + (g - g_0)` as in [EQ-DIST-AD-001]
   ([EQ-CAL-AD-001]); in physical coordinates a Gaussian copula's normal
   scores use the reciprocal rule. This avoids differentiating quantile
   functions and works for every continuous prior family. The rule is exact
   to first order only, so a forward-over-reverse Hessian through it raises
   `AutodiffUnsupportedError`; `laplace_approximation` then takes the
   central-difference Hessian and records the reason. With an independent
   prior the Laplace Hessian is forward over reverse.
7. **No silent fallback.** Discrete prior inputs, non-Gaussian copulas
   (Rosenblatt map), a noise or length prior without a differentiable
   `log_pdf`, and a forward model that converts variables to floats raise
   `AutodiffUnsupportedError` with the reason before any sampling starts.
   The public methods `standard_log_posterior_and_gradient` and
   `log_posterior_and_gradient` expose the gradients.
8. **`DiscrepancyCalibration` keeps central differences.** Its likelihood
   assembles the Gaussian-process covariance and the log-linear noise with
   float-only `linalg` and `math` code; a traceable twin would duplicate
   that assembly and the hyperparameter priors. It is recorded as follow-up
   work rather than added here.

## Consequences

- Correlated measurement errors are part of the calibration model, with the
  same API for known and uncertain scales and a sampled correlation length.
- Existing results are unchanged: the defaults (`correlation=None`,
  `gradient="central"`) take the earlier code paths, and the full test suite
  passes without changing an existing test.
- With `gradient="reverse"` NUTS on the exponential-decay calibration of the
  guide (21 experiments, known noise; 2 chains, 200 + 250 iterations,
  seed 21) used 105,861 forward-model runs against 542,850 with central
  differences, the factor `2n + 1 = 5` of a two-parameter problem; the
  posterior means agree within their Monte Carlo errors. Seeded HMC/NUTS
  trajectories differ between the two options because the gradients differ
  at rounding level.
- The correlation length and a discrepancy process compete for the same
  structure in the residuals, and under infill the scale and length of a
  Matern correlation are not separately identifiable (H. Zhang, *Journal of the American Statistical Association* 99, 250-261, 2004, doi:10.1198/016214504000000241);
  the guide documents both caveats.

## Rejected alternatives

- **A separate `CorrelatedNoise` class next to `NoiseModel`.** It would
  duplicate the known/uncertain scale and nugget logic and force a second
  type check in every consumer; one optional field keeps one noise
  declaration per quantity.
- **Differentiable quantile functions.** Writing every quantile with
  `admath` would touch every distribution family and its accuracy-critical
  tail code; the inverse-function rule needs only the float quantile and
  the density.
- **Falling back to central differences when tracing fails.** It hides a
  ten-fold cost change behind an option that promises the opposite; the
  inference engines' `"auto-reverse"` already provides that behaviour for
  users who want it.
- **Conditioning prediction noise on the calibration residuals.** It
  presumes that new measurements continue the calibration series, which
  the declaration does not state.
- **Returning `-inf` for a known non-positive-definite matrix.** A declared
  structure that cannot be factorized is an input error, not a region of
  zero likelihood.

## Verification

`tests/test_correlated_noise_autodiff_calibration.py` (23 tests) checks the
kernels against their closed forms; exact equality of the independent
likelihood and standard-space posterior with the earlier formula; that a
declared identity matrix and a kernel whose correlations underflow
reproduce the independent likelihood, posterior, and least-squares estimate
within `1e-12` relative; the correlated likelihood and whitened residuals
against an explicit-inverse multivariate normal for every kernel with and
without nugget; on the correlated straight line, the Laplace mean,
covariance, and log normalizer (both gradient options) against the
conjugate posterior and evidence computed by Gaussian conditioning, least
squares against the GLS estimate and covariance, the Fisher spectrum
against `H' Sigma^-1 H`, adaptive Metropolis and reverse-gradient NUTS
within 4 MCSE, and the SMC evidence within 4 standard errors; a sampled
length whose MCMC mean matches the exact quadrature posterior within
4 MCSE (true length 1.0, exact posterior mean 0.623, sd 0.176, seed 2026);
pCN and SMC layouts with an uncertain length; joint prediction noise with
the kernel correlation; reverse gradients against central differences
within `1e-6` relative for known, unknown, Student-t, correlated with
sampled length, and Gaussian-copula problems in both coordinate systems;
NUTS with reverse gradients against central differences within 4 MCSE with
fewer than a third of the forward-model runs; and the unsupported cases.

## References

[REF-SEBER-WILD-2003, Chapter 6], [REF-RASMUSSEN-WILLIAMS-2006,
Section 4.2], [REF-BRYNJARSDOTTIR-OHAGAN-2014],
[REF-GRIEWANK-WALTHER-2008, Chapter 3], [REF-LIU-DERKIUREGHIAN-1986],
[REF-CASELLA-BERGER-2002, Section 2.1].
