# ADR-0017 — Nonlinear calibration and MCMC

- Status: Accepted
- Date: 2026-10-11
- Lifts: the scope limits stated in ADR-0013 (unknown noise variance,
  nonlinear forward models, heavy-tailed likelihoods, MCMC diagnostics)

## Context

ADR-0013 deliberately restricted calibration to a linear response, a Gaussian
prior, and known homoscedastic noise. The central `ProbabilisticModel` layer
now gives every workflow one joint input law, a Nataf transformation, and
dependency-free optimizers. That makes general backward modeling possible
without new dependencies. It must still respect the layer graph and stay
auditable.

## Decision

1. `probstat.inference.mcmc` (layer 2) owns generic samplers and diagnostics.
   It needs only a log-density callable, so it never imports `modeling`. It
   provides random-walk Metropolis, adaptive Metropolis, split and
   rank-normalized R-hat, Geyer initial-monotone ESS, MCSE, posterior
   summaries, and a BFGS + finite-difference Laplace approximation.
2. Adaptive Metropolis adapts **only during warm-up**, then freezes its
   proposal covariance. Retained draws come from one time-homogeneous
   Metropolis kernel, so the standard Markov-chain theory and diagnostics
   apply without relying on adaptive-MCMC ergodicity results.
3. Chains are seeded from `RandomStream.spawn(chains, label="mcmc-chain")`.
   Each chain adapts only from its own history, so chains stay independent
   and R-hat stays meaningful.
4. `probstat.inference.bayesian` adds the Normal-inverse-gamma conjugate
   update for linear regression with unknown variance. It includes Student-t
   marginals and predictives, with the Student-t CDF built on the existing
   incomplete-beta routine.
5. `probstat.inverse.nonlinear` (layer 4) owns `CalibrationProblem`. The
   model's `JointInput` is the prior. Experiments carry condition mappings
   that override `model.fixed`. Each `NoiseModel` declares either a known
   sigma or a prior on an unknown sigma, an optional Student-t likelihood,
   and an optional discrepancy variance given as a float. The inverse layer
   cannot import `validation`.
6. Unknown noise scales are parameterized as `log sigma`, with the Jacobian
   term written explicitly in the prior.
7. Posterior sampling runs in the prior's Nataf standard-normal space. The
   push-forward identity `pi(T^-1(u)) |det dT^-1/du| = phi(u)` removes the
   need to code a prior Jacobian. A test checks the identity numerically
   against the physical-space posterior.
8. Inside MCMC, the forward model is called through a lean output check:
   mapping type, presence, and finiteness. Full `evaluate_model` parameter
   validation on every call made pure-Python sampling about four times
   slower. Parameters are validated once per proposal when `theta` is
   materialized.
9. Identifiability is diagnosed from the eigen-decomposition of the
   prior-scaled Gauss-Newton information `D J'J D`. The diagnostic reports
   rank, condition number, and weakly identified parameter combinations.
10. `inverse.predict` pushes posterior draws through the model under new
    conditions. It returns latent (parameter-only) and observation (with
    noise) samples separately; noise is added only on request.

## Consequences

- Linear-Gaussian calibration can now go through either API. The Laplace
  result of the nonlinear API reproduces `condition_linear_gaussian` to six
  decimals, which ties the two together.
- Random-walk MCMC in pure Python is suitable for teaching and for models
  with up to roughly ten to twenty parameters and cheap forward models. It is
  not a replacement for gradient-based samplers.
- `tests/test_architecture.py` moves `visualization` from layer 6 to layer 9.
  This reserves room for future layers above validation without renumbering
  again.

## Rejected alternatives

- Perpetual adaptation: valid under extra conditions, but it complicates
  diagnostics and reproducibility arguments.
- Sampling in physical space with reflection or rejection at prior bounds:
  this wastes proposals and conditions badly for skewed priors.
- Importing `validation.AdditiveGaussianDiscrepancy` into `inverse`: this
  would be an upward import. A float variance with the same semantics is
  accepted instead.
- Silently clipping or regularizing a non-positive-definite Laplace Hessian:
  `laplace_approximation` raises. `sample_posterior` falls back to an
  isotropic start only for initialization and records `laplace=None`.

## Not covered

HMC/NUTS, SMC, correlated or heteroscedastic noise models, Gaussian-process
discrepancy, hierarchical models, model evidence beyond the Laplace estimate,
and optimal experimental design remain planned.
