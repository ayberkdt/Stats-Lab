# ADR-0074: Normalizing flows and energy-based models

- Status: Accepted
- Date: 2026-11-11

## Context

Phase 6 of the roadmap lists normalizing flows, energy-based models,
diffusion and score systems, and neural stochastic differential equations.
Score-based diffusion and neural SDEs belong to `processes`; this record
covers flows and energy models. Both are density models fitted to data:
a flow pushes a standard normal through an invertible map and has an exact
density by the change of variables (Papamakarios et al. 2021); an energy
model `p ∝ exp(-E)` has an intractable normalizing constant and is fitted
by score matching (Hyvärinen 2005; Vincent 2011) and sampled by Langevin
dynamics (Roberts and Tweedie 1996).

The shared building blocks already exist: `numerics.MLP` (flat parameters,
evaluated through `admath`), reverse-mode differentiation
(`numerics.reverse_value_and_gradient`, forward-over-reverse Hessians), and
`optimization.adam_minimize`. Constraints: standard library only, strict
mypy, scalar Python arithmetic, tests in seconds, and no Langevin sampler
anywhere in the library.

## Decision

1. **New package `probstat.generative` at layer 3, label `experimental`.**
   It imports only `foundations`, `numerics`, and `optimization`; layer 3
   leaves room for later use by `inverse` or `surrogates` without an
   upward import.
2. **One layer protocol, two directions.** Every layer maps base to data
   (`forward`) and data to base (`inverse`) and returns the log-determinant
   of the map it applies; `NormalizingFlow` composes them, keeps one flat
   parameter vector, and evaluates `log N(T^{-1}(x)) + log |det J_{T^{-1}}|`
   on floats or tape variables (EQ-GEN-FLOW-001).
3. **Layers.** `PlanarFlow` and `RadialFlow` with the reparameterizations of
   Rezende and Mohamed (Appendix A): `w'u_hat = m(w'u) > -1` and
   `beta = -alpha + softplus > -alpha`, `alpha = softplus > 0`
   (EQ-GEN-FLOW-002, -004). `AffineCoupling` with an `MLP` conditioner
   whose output layer starts at zero (identity start) and log-scale
   `s = c tanh(raw / c)`, `c = 2` by default (EQ-GEN-FLOW-005). `ActNorm`
   with data-dependent initialization from the data side
   (EQ-GEN-FLOW-006). `Permutation` for alternation. `real_nvp_flow` builds
   couplings, reversals, and a final actnorm.
4. **Inverses.** Coupling, actnorm, permutation, and radial layers invert in
   closed form; the radial radius is the positive root of a quadratic,
   computed without cancellation. The planar inverse solves a strictly
   increasing scalar equation by safeguarded Newton in floats, then takes
   one Newton step on the (taped) parameters, which carries the
   implicit-function derivative of the root (EQ-GEN-FLOW-003). Planar
   layers are therefore trainable by maximum likelihood like the others.
5. **`fit_flow`** minimizes the mini-batch negative mean log-likelihood with
   `adam_minimize` (batches without replacement from the iteration's child
   stream), reports the loss trace and the training and optional held-out
   mean log-likelihood with standard errors (EQ-GEN-FLOW-007).
6. **Energy `E = |L'(x - mu)|^2 / 2 + N_theta(x)`** with a log-diagonal
   Cholesky factor and an optional `MLP` (tanh, sigmoid, or softplus units)
   whose output layer starts at zero (EQ-GEN-EBM-001). The quadratic part
   makes the density normalizable and, without a network, gives a Gaussian
   whose score-matching estimate has a closed form (EQ-GEN-EBM-003).
7. **Implicit score matching needs third derivatives.** The parameter
   gradient of `|grad E|^2 / 2 - Laplacian E` differentiates second input
   derivatives, and the reverse-mode tape does not nest. The input gradient
   and Laplacian are therefore propagated forward by a private
   `LaplacianJet` whose components are tape variables of the parameters
   (EQ-GEN-EBM-005); `MLP.forward` runs on it unchanged through the
   `admath` operand hook. One reverse sweep then gives the exact parameter
   gradient, at about `d + 2` network evaluations per point. The
   forward-over-reverse Hessian (`reverse_value_gradient_and_hessian`)
   evaluates the same objective independently
   (`score_matching_objective`, `EnergyModel.energy_laplacian`) and serves as
   the oracle of the jet. The quadratic part enters the jet in closed form
   (`A v`, `trace A`), with `A = L L'` formed once per mini-batch.
8. **Denoising score matching** with Gaussian corruption uses the same jet
   in gradient-only mode (EQ-GEN-EBM-004); the minimizer is the score of the
   smoothed data, which the documentation states explicitly.
9. **Langevin sampling.** `sample_langevin` implements ULA and MALA with
   the step `x + (h/2) grad log pi + sqrt(h) xi` (EQ-GEN-LANG-001, -002) for
   any `admath` log-density, differentiated by reverse mode; MALA rejects
   proposals outside the support, ULA raises there.
10. **Diagnostics.** `quadrature_normalizing_constant` (adaptive
    Gauss-Kronrod, nested in 2-D, shifted by the grid maximum),
    `importance_normalizing_constant` (log-space weights, relative standard
    error, weight ESS), `mean_log_likelihood`, and `kl_divergence_estimate`
    (EQ-GEN-DIAG-001 to -003).

## Consequences

- Twenty-four public names join the new package: the layers `PlanarFlow`,
  `RadialFlow`, `AffineCoupling`, `ActNorm`, `Permutation`, the protocol
  `FlowLayer`, `NormalizingFlow`, `real_nvp_flow`, `fit_flow`, `FlowFit`;
  `EnergyModel`, `EnergyFit`, `fit_score_matching`,
  `fit_denoising_score_matching`, `score_matching_objective`,
  `denoising_score_matching_objective`; `sample_langevin`,
  `LangevinResult`; `quadrature_normalizing_constant`,
  `importance_normalizing_constant`, `NormalizingConstant`,
  `mean_log_likelihood`, `kl_divergence_estimate`, `SampleMean`. The API
  reference must be regenerated when the change is released.
- Equations EQ-GEN-FLOW-001 to -007, EQ-GEN-EBM-001 to -005,
  EQ-GEN-LANG-001 and -002, EQ-GEN-DIAG-001 to -003, and seven
  bibliography entries are added.
- `generative._jet` subclasses the private `_TapeOperand` hook of
  `numerics.autodiff`. The hook is the documented extension point for
  operands that evaluate `admath`; a change to it must keep the jet tests
  passing.
- Cost is scalar Python: one reverse-mode loss and gradient over 64 points
  through two couplings of width 8 takes about 20 ms (300 iterations in
  5.6 s). Nested 2-D quadrature needs 20 000 to 50 000 density
  evaluations (1 to 4 s). The methods are validated for `d <= 2`
  and a few hundred points; score matching scales as `d` network
  evaluations per point.
- A flow fit can overfit small data; the held-out likelihood is reported for
  that reason. Score matching does not maximize likelihood, so its held-out
  likelihood can fall as training continues (observed on the banana
  target: `-2.33` after 150 and `-2.45` after 200 iterations).

## Rejected alternatives

- Nested reverse mode for score matching: the tape rejects variables of
  different tapes by design (ADR-0069), and making it reentrant would change
  a stable package for one consumer.
- Finite-difference parameter gradients of the score-matching objective:
  `2 P` objective evaluations per step (`P = 70` for a 2-D energy of width
  16) with a step-size error.
- Hutchinson trace estimation: unbiased but noisy; the exact trace is
  affordable in the target dimensions.
- Planar and radial layers in the inverse direction only (no density at
  data points): would exclude them from maximum-likelihood fitting, which
  the Newton-step inverse makes available at the cost of one root solve.
- An unbounded coupling log-scale: a single step can then overflow
  `exp(s)`; the tanh bound costs nothing in the tested cases.

## Verification

Measured with the seeded cases of `tests/test_generative_flows_energy.py`
(21 tests with 61 subtests, about 40 s):

- Log-determinants of planar, radial, coupling, actnorm, and permutation
  layers in 1-D and 2-D against central-difference Jacobian determinants:
  worst difference `3.1e-10`. Round trips `inverse(forward(z))`: worst
  error `8.9e-16`, with forward and inverse log-determinants cancelling to
  the same order.
- Reverse-mode gradient of `log p` through two planar (Newton-step) layers,
  a radial layer, and an actnorm against parameter finite differences:
  agreement to six decimals (`2e-9` in a separate run).
- Normalization: a 1-D planar-radial-planar-actnorm flow integrates to
  `log Z = -4.8e-9` over the real line (519 evaluations); a 2-D
  coupling-reversal-radial-actnorm flow to `log Z = -4.9e-8` on
  `[-11, 11]^2` (37 386 evaluations).
- Gaussian data (`n = 300`): a two-coupling flow of width 8 after 200
  iterations has training mean log-likelihood `-2.746` against `-2.754` for
  the Gaussian maximum-likelihood fit; 4000 flow draws have mean
  `(1.18, -0.39)` and covariance `[[2.17, 1.26], [1.26, 1.14]]` against
  data moments `(1.11, -0.46)` and `[[2.27, 1.22], [1.22, 1.03]]`.
- Banana target (`n = 300` train, 300 held out): held-out mean
  log-likelihood `-1.872 ± 0.072` for the flow, `-2.863 ± 0.204` for one
  Gaussian, `-1.792` for the true density; Monte Carlo
  `KL(truth || model)` `0.081 ± 0.020` for the flow and `1.072 ± 0.162` for
  the Gaussian.
- Score matching, quadratic energy, `n = 120`, full batch: precision error
  `4.9e-6` against the inverse `1/n` sample covariance, center error
  `1.1e-6`, objective `-1.83561779` against the closed form `-trace(M)/2`.
- Denoising score matching, `s = 0.3`, 400 iterations with tail averaging:
  maximum precision error `0.049` against `(S + s^2 I)^{-1}` (largest entry
  `1.98`) and `0.63` against `S^{-1}`.
- Score-matched MLP energy (width 8, 150 iterations) on the banana target,
  normalized by quadrature (`log Z` relative error `6.4e-5`): held-out
  mean log-likelihood `-2.335 ± 0.177`, KL estimate `0.54 ± 0.14`, both
  better than the single Gaussian.
- Jet against forward-over-reverse: value and gradient to 12 decimals,
  Laplacian to 10, for tanh, softplus, and sigmoid networks.
- Normalizing constant of a Gaussian energy: quadrature error `6.5e-14`
  (21 756 evaluations); importance sampling with 4000 draws error `4.2e-4`
  with reported relative error `8.0e-3` and weight ESS `3185`.
- MALA on `N(0, [[1, 0.8], [0.8, 1]])`, `h = 0.3`, 6000 draws: acceptance
  `0.857`; `E[xy] = 0.891` with Monte Carlo standard error `0.056`. On
  `N(0, 1)` with `h = 0.5`: ULA variance `1.080 ± 0.033` against its exact
  stationary value `1 / (1 - h/4) = 1.143`, MALA `0.979 ± 0.028` against 1.
- Equal seeds reproduce flow fits, loss traces, and Langevin chains;
  invalid layers, shapes, data, and settings raise.
- Contract tests (documentation, architecture, stability), ruff, and strict
  mypy pass.
