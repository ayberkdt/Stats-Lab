# ADR-0013 — Linear-Gaussian calibration boundaries

- Status: Accepted
- Date: 2026-10-09

## Context

UQ-6 needs a complete inverse-problem learning chain without pretending that a
small dependency-free library already provides production nonlinear Bayesian
calibration. The implementation must distinguish measurement noise, model
discrepancy, parameter uncertainty, latent predictions, and future
observations. It must also expose identifiability failure before returning
apparently precise parameter estimates.

## Decision

1. `probstat.inference.regression` owns ordinary least squares and descriptive
   fit diagnostics.
2. `probstat.inference.bayesian` owns a general finite-dimensional conjugate
   linear-Gaussian update, posterior sampling, posterior prediction, and
   predictive checking.
3. `probstat.inverse` owns observation-model semantics, calibration
   orchestration, design identifiability, and the calibration/prediction split.
4. `probstat.validation` owns discrepancy records and scientific applicability
   scope. Lower inverse/inference layers accept scalar discrepancy variance and
   therefore never import upward from validation.
5. The conjugate update requires a proper positive-definite Gaussian prior and
   known positive homoscedastic observation variance. It rejects invalid shapes
   and singular priors rather than adding hidden jitter.
6. Predictive covariance reports parameter, observation, and discrepancy
   components separately. Whether observation noise is included is an explicit
   call-site choice.
7. Identifiability is checked through exact rank and Gram-matrix spectral
   conditioning for the intercept/slope calibration design.
8. Posterior and predictive draws use call-local seeds. Predictive checks are
   diagnostics, never declarations that the model is validated.

## Consequences

The UQ-6 path is reproducible, auditable, and usable without runtime numerical
dependencies. The general Gaussian update supports more than the two-parameter
teaching lab, while the lab remains interpretable. Cholesky routines are still
duplicated between multivariate probability and Bayesian inference; a later
foundational linear-algebra module may consolidate them only after public
contracts and numerical tests are designed.

Unknown noise variance, nonlinear forward models, heavy-tailed likelihoods,
MCMC diagnostics, correlated discrepancy processes, and optimal experimental
design remain explicitly planned. The analytic posterior must not be used as
if it covered those cases.

## Rejected alternatives

- Treating least squares as the posterior: this hides the prior and uncertainty
  assumptions.
- Combining every uncertainty source into one unnamed variance: this destroys
  the calibration-versus-prediction interpretation.
- Automatically regularizing singular systems: this can create an apparently
  identified result without a declared prior or scientific justification.
- Importing the validation package into inference: this reverses the intended
  architecture and risks a future dependency cycle.
