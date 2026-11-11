# ADR-0075 — Score-based diffusion and neural SDEs

- Status: Accepted
- Date: 2026-11-11

## Context

The Phase 6 roadmap item lists diffusion/score systems and neural stochastic
differential equations. `probstat.processes` already had scalar
Euler-Maruyama and Milstein schemes (ADR-0023). The shared neural blocks
were also in place: `numerics.MLP` with flat parameters evaluated through
`admath`, reverse-mode gradients (`numerics.reverse_value_and_gradient`,
ADR-0069), and `optimization.adam_minimize`. Normalizing flows and energy
models belong to a separate `probstat.generative` package. The two topics
here are stochastic-process methods and stay in layer 2.

Score-based generative modeling (Song et al. 2021) defines a forward SDE
that turns data into a Gaussian prior. Generation integrates Anderson's
reverse-time SDE, or the probability-flow ODE, with a learned score
`grad log p_t`. A neural SDE puts network coefficients into an Ito equation
and fits them to observed paths. Two constraints apply: the standard
library only, and scalar Python arithmetic. Small dimensions and small
networks are therefore the design point.

## Decision

1. **Two new modules in `processes`.** `score_diffusion` holds the forward
   SDEs, the exact Gaussian score, score matching, and the samplers.
   `neural_sde` holds the neural SDE, its pseudo-likelihood fit, and
   simulation. They import `foundations`, `numerics` (MLP, `admath`,
   reverse and forward differentiation), `optimization` (Adam), and the
   existing `processes.sde` and `processes.continuous`. Neither module adds
   a dependency direction or a new MLP or optimizer.
2. **Forward SDEs** are frozen dataclasses that satisfy a `ForwardSDE`
   protocol: drift, diffusion, kernel mean coefficient `m(t)`, kernel
   variance `v(t)`, prior variance, default end time, and corrector scale.
   - VP: linear `beta`, defaults `0.1` and `20`, end time `1e-3`.
   - VE: geometric `sigma`, defaults `0.01` and `50`, end time `1e-5`.
     The VE kernel variance is the exact solution `sigma(t)^2 -
     sigma_min^2`, not the source's `sigma(t)^2`.
3. **Exact-score path.** `GaussianScore` gives the score of diffused
   diagonal-Gaussian data in closed form. Every sampler takes any callable
   `score(x, t)`, so the samplers are tested independently of training.
4. **Score network parameterization.** The network input is
   `(c(t) x, 2 tau(t) - 1)`, with `c = 1 / sqrt(m^2 + v)` and the noise
   level `tau = sqrt(v / (m^2 + v))`. The output is divided by
   `sqrt(v(t))`. With the default weighting `lambda(t) = v(t)`, the
   denoising objective becomes the DDPM noise-prediction loss. A custom
   `lambda` is accepted. Conditioning on `tau` instead of `t` was measured:
   at `t = 0.02` the relative score error fell from 1.96 to 1.46 (same
   data, seed, and budget).
5. **Samplers.**
   - Reverse-time Euler-Maruyama.
   - Predictor-corrector: the same predictor plus Langevin moves with
     `eps = 2 alpha (r ||z|| / ||s||)^2`. The norms are averaged over the
     batch, which the source allows. `alpha = 1 - beta(t) h` for VP, and
     the sampler raises when `alpha <= 0`.
   - Probability-flow ODE: classical RK4 or Heun in either time direction,
     so the same function encodes and decodes.
   - Each sampler returns a `DiffusionSample` with its score-evaluation
     count.
6. **Neural SDE.**
   - Autonomous with diagonal noise.
   - The drift MLP acts on inputs standardized by the data mean and
     spread.
   - The diffusion is either a constant `exp(rho)` or
     `b_min + softplus(MLP)`.
   - The fit maximizes the Euler-Maruyama Gaussian pseudo-likelihood over
     mini-batches of transitions. The diffusion starts at the
     quadratic-variation level.
   - One-dimensional fits convert to `ScalarSDE` and run through
     `solve_sde`: Euler-Maruyama, or Milstein with a forward-mode diffusion
     derivative. Higher dimensions use the componentwise Euler-Maruyama
     step.

## Consequences

- Twenty public names join `processes`:
  - constants `ODE_METHODS` and `DIFFUSION_MODELS`;
  - types `ForwardSDE`, `VariancePreservingSDE`, `VarianceExplodingSDE`,
    `GaussianScore`, `ScoreNetwork`, `TrainedScore`, `DiffusionSample`,
    `NeuralSDE`, `NeuralSDEFit`, `VectorSamplePath`;
  - functions `perturbation_moments`, `perturb`, `train_score_network`,
    `reverse_sde_sample`, `predictor_corrector_sample`,
    `probability_flow_map`, `probability_flow_sample`, `fit_neural_sde`.

  The API reference must be regenerated when the change is released.
- Nine equations are added: EQ-PROC-DIFF-001 to EQ-PROC-DIFF-007 and
  EQ-PROC-NSDE-001 to EQ-PROC-NSDE-002. Six bibliography entries are added.
- Costs:
  - Score training costs about 7 ms per iteration for a width-16 network
    and a batch of 32 in one dimension.
  - The neural-SDE fit costs about 15 ms per iteration for width 8 and a
    batch of 100.
  - Larger networks, dimensions, or data sets are out of scope.
- The small-noise score is learned poorly from a few hundred points. The
  weighting gives it little weight in the loss, and the empirical data law
  differs from the population law at small noise. Samples from a trained
  score are too concentrated (variance 0.206 against the data's 0.260). The
  exact-score path shows the samplers themselves are not the cause.
- The pseudo-likelihood has an `O(h)` bias. For OU with `theta h = 0.02`,
  the bias in the drift slope is about 1 %. Sparse observations would need
  a bridge or an exact likelihood, which is not provided.
- Time-dependent neural drifts are not supported; the neural SDE is
  autonomous.

## Rejected alternatives

- **Writing the samplers on top of `ScalarSDE`.** The score samplers work
  on vectors, on batches (the corrector step size uses batch norms), and
  in reversed time. A vector Euler-Maruyama step is simpler than adapting
  the scalar interface.
- **Time conditioning on `2t - 1`.** It was the first choice; the noise
  level `tau` gave lower small-`t` error, see Decision 4.
- **A separate generative package for the score models.** The forward and
  reverse SDEs are stochastic processes and reuse `processes.sde`. The
  generative package owns density models (flows, energy models) that are
  not SDEs.
- **Fitting the neural SDE by backpropagation through simulated paths.**
  It needs path-wise gradients of the solver and a distance between path
  laws. The Euler pseudo-likelihood has a closed form, is
  differentiable on the tape, and is the standard estimator for densely
  observed diffusions.

## Verification

Measured with the seeded cases of `tests/test_score_diffusion_neural_sde.py`
(17 tests, about 22 s):

- Kernels:
  - VP mean coefficient and variance agree with `exp(-B/2)` and
    `1 - exp(-B)` to 14 decimals.
  - VE variance agrees with `sigma(t)^2 - sigma_min^2`.
  - Both kernels satisfy the moment ODEs `m' = (f/x) m` and
    `v' = 2 (f/x) v + g^2` to `1e-6` by central differences.
  - 2,000 `perturb` draws match the kernel moments within 3.5 standard
    errors.
- The closed-form Gaussian score matches a finite-difference gradient of
  the marginal log-density to six decimals in two dimensions.
- Exact-score samplers, data `N(1, 0.25)`, 400 points; deviations are in
  standard errors:

  | SDE | sampler | mean (z) | variance (z) |
  |---|---|---|---|
  | VP | reverse SDE | 1.0198 (0.79) | 0.2464 (-0.20) |
  | VP | predictor-corrector | 1.0145 (0.58) | 0.2630 (0.74) |
  | VP | probability flow | 1.0279 (1.11) | 0.2563 (0.36) |
  | VE | reverse SDE | 1.0093 (0.37) | 0.2304 (-1.11) |
  | VE | predictor-corrector | 0.9983 (-0.07) | 0.2700 (1.13) |
  | VE | probability flow | 1.0212 (0.85) | 0.2562 (0.35) |

  The test threshold is 3.5 standard errors.
- Two-dimensional VP flow, 300 points, target means `(1, -0.5)` and
  variances `(0.25, 1)`:
  - means `(0.961, -0.433)`;
  - variances `(0.228, 0.892)`;
  - every value lies within 1.4 standard errors.
- Probability-flow map, four prior points:
  - repeated calls are identical;
  - RK4 (60 steps) matches the exact affine map to `1.8e-7`;
  - Heun (200 steps) matches it to `8.3e-5`;
  - the encode-decode round trip is accurate to `4.4e-8`.
- Denoising score matching, 300 draws of `N(1, 0.25)`, VP, width 16, 800
  iterations, seed 2:
  - relative grid error 0.070, 0.091, and 0.029 at `t = 0.1, 0.3, 0.7`;
  - the untrained network's errors are 1.28, 1.16, and 1.20;
  - the test requires an error below 0.25 and below one fifth of the
    untrained error;
  - mean loss 0.363 over the first 20 iterations and 0.189 over the last
    100;
  - at `t = 0.02` the error is 1.46, which is not tested.
- Neural SDE on 40 exact OU paths (`theta = 1`, `mu = 0`, `sigma = 0.5`,
  `h = 0.02`, 250 steps), width-8 drift, 400 iterations:

  | quantity | fitted | linear Euler least squares | quadratic variation | truth | tolerance |
  |---|---|---|---|---|---|
  | drift slope on `[-0.6, 0.6]` | -0.969 | -0.996 | | -1 | 0.25 of truth, 0.1 of least squares |
  | drift intercept | 0.029 | | | 0 | 0.1 |
  | diffusion | 0.491 | | 0.494 | 0.5 | 0.025 |

  The fitted drift stays within 0.056 of `-x` on the grid (tolerance 0.15).
  `simulate` reproduces `milstein` on the converted `ScalarSDE` exactly.
- An MLP diffusion's forward-mode derivative matches a central difference
  to six decimals.
- Equal seeds reproduce samplers, training, and fits.
- Twenty-three invalid-input cases raise `ValueError`.
- The documentation and architecture contract tests, the existing
  stochastic-process tests, ruff, and strict mypy pass.

## Unverified details

The equation, section, appendix, and algorithm numbers cited for Song et
al. (2021) and the equation numbers of Ho, Jain and Abbeel (2020) were read
from the arXiv HTML renderings. The bibliographic records of Anderson
(1982), Vincent (2011), Hairer-Nørsett-Wanner (1993), and Iacus (2008) were
checked against Crossref. Two locators were not checked against the books
themselves: Section II.1 of Hairer-Nørsett-Wanner for the Runge-Kutta and
Heun methods, and the pseudo-likelihood material in Chapter 3 of Iacus
("Parametric Estimation", chapter title verified).
