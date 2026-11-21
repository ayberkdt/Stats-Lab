# ADR-0129 — Flow matching, flow-matching posteriors, and neural ratio estimation

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0074 (normalizing flows and energy-based models) and the
  Gaussian amortized posterior of `inference.amortized`

## Context

The roadmap listed "amortized inference with non-Gaussian conditional
density estimators" as open. `inference.fit_amortized_gaussian_posterior`
fits a diagonal Gaussian `q(theta | x)`: it is exact for conjugate normal
models and silently wrong when the posterior is skewed or multimodal. The
flows of `generative.flows` are unconditional and are trained by maximum
likelihood through an exact Jacobian, which ties the architecture to
invertible layers.

Two simulation-based methods avoid both restrictions and need only blocks
the library already has (a small MLP, Adam, a fixed-step ODE solver,
random-walk Metropolis):

- **Flow matching** (Lipman et al. 2023) regresses a velocity field on
  conditional straight-line paths; with the observation as a network input
  it becomes an amortized posterior estimator (Wildberger et al. 2023).
- **Neural ratio estimation** (Hermans, Begy and Louppe 2020) trains a
  classifier whose logit is the log likelihood-to-evidence ratio.

Hermans et al. (2022) show that such estimators can be overconfident, so
the package has to make checking them easy and say so.

## Decision

1. **Three modules in `generative`** (layer 3; imports `foundations`,
   `numerics`, `optimization`, and `inference`, all lower layers):
   `flow_matching.py` (`conditional_flow_path`, `GaussianMarginalVelocity`,
   `integrate_flow`, `flow_log_density`, `VelocityNetwork`,
   `flow_matching_loss_and_gradient`, `fit_flow_matching`,
   `FlowMatchingFit`), `posterior_estimation.py`
   (`fit_flow_matching_posterior`, `FlowMatchingPosterior`), and
   `ratio_estimation.py` (`fit_ratio_estimator`, `RatioEstimator`,
   `GridPosterior`, `ratio_loss_and_gradient`).
2. **Path and target** ([EQ-GEN-FM-001]):
   `x_t = (1 - (1 - s) t) x_0 + t x_1`, `u = x_1 - (1 - s) x_0`, derived by
   differentiating the path. `s = sigma_min` defaults to 0 and must lie in
   `[0, 1)`; the model then targets the standardized data convolved with
   `N(0, s^2 I)`, which the documentation states.
3. **Oracle independent of training** ([EQ-GEN-FM-003]): for Gaussian data
   the marginal field is `m + c(t) (x - t m)` with
   `c = (t sd^2 - (1 - s) a) / (a^2 + t^2 sd^2)`, `a = 1 - (1 - s) t`,
   derived as the linear regression of `u` on `x_t`.
   `GaussianMarginalVelocity` is public so that users can test their own
   solver settings against it.
4. **Sampling** uses `numerics.integrate_fixed_step` (Euler, Heun, or RK4;
   RK4 with 20 steps by default). No adaptive solver is added.
5. **Log-density** ([EQ-GEN-FM-004]) integrates
   `d log p / dt = -div v` backward from the query point together with the
   state. The divergence is exact: for the networks by propagating the
   input tangents through the layers, for user fields by forward-mode
   duals; central differences are available on request and a divergence
   function can be supplied. No stochastic trace estimator is used (the
   dimension is one or two).
6. **Gradients by float backpropagation** ([EQ-GEN-FM-005], private module
   `generative/_backprop.py`) instead of
   `numerics.reverse_value_and_gradient`. For a `(3, 16, 16, 1)` network
   and a batch of 32 the tape needs 18 ms per gradient and backpropagation
   1.9 ms; the tests compare the two to `1e-12`. The parameter layout is
   that of `numerics.MLP`, the hidden activation is fixed to `tanh`, and
   the optimizer is `optimization.adam_minimize` with tail averaging, as
   in the other trained models of the library.
7. **Posterior estimation API** parallel to
   `fit_amortized_gaussian_posterior`: `prior_sampler(rng)`,
   `simulator(theta, rng)`, `simulations`, `hidden_layers`, `iterations`,
   `batch_size`, `learning_rate`, `seed`; the result stores the training
   moments used for standardization, the loss trace, and `simulations`.
8. **Ratio estimation**: the network outputs the logit; contrast pairs are
   built inside each mini-batch by a random cyclic shift of `theta`, so no
   contrast pair coincides with a joint pair. `posterior_on_grid` (scalar
   parameter) reports `ratio_normalization = int p(theta) r(theta, x) d
   theta`, which is one for the exact ratio; `sample_posterior` calls
   `inference.random_walk_metropolis` and returns its `MCMCResult` with
   the usual convergence warnings.
9. **Honest convergence reporting.** Adam has no stopping test. The
   results report the loss trace, the mean loss over the first and last
   tenth of the iterations, and `loss_decreased`, documented as a sanity
   check only; the flow-matching loss has a positive minimum.

## Consequences

Measured over training seeds 0 to 5 at the settings of the tests (pure
Python, one core):

- **Exact Gaussian field**: RK4 with 20 steps maps `z` to
  `m + sd z` within `1e-7` and returns the normal log-density within
  `6e-7`.
- **1-D Gaussian** `N(1.5, 0.7^2)`, 400 points, one hidden layer of 16,
  600 iterations (0.7 s): error of the mean of 400 draws from -0.093 to
  0.034, of the standard deviation from 0.003 to 0.051; largest
  log-density error at five points within 1.4 sd from 0.03 to 0.13.
- **1-D mixture** `0.3 N(-2, 0.5^2) + 0.7 N(2, 0.5^2)`, 500 points, two
  layers of 16, 1200 iterations (2.9 s): mass of the left mode minus the
  data fraction from -0.004 to 0.068; mass in `|x| < 1` from 0.030 to
  0.074 (exact 0.023); mode standard deviations 0.49 to 0.71 against 0.5,
  so the modes are somewhat too wide.
- **Flow-matching posterior, linear-Gaussian** (`theta ~ N(0, 1)`,
  `x = 2 theta + N(0, 1)`, 1000 simulations, 1200 iterations, 2.5 s):
  posterior-mean error at `x = -3, 0, 2, 4` at most 0.046 except 0.096
  once at `x = 4`; standard-deviation error from -0.061 to 0.064 against
  0.447; log-density error up to 0.14 to 0.42 at 1.3 sd from the mean.
  Simulation-based calibration (200 replicates, 19 draws, 20 bins) gave
  p-values 0.42, 0.083, 0.076, 0.066, 0.99, 0.89: no rejection at 5 %, but
  three of six below 0.1. An independent audit repeated the study with 2000
  replicates and two calibration seeds per fit: the p-values were `2e-3`
  and `2e-5` (training seed 0), 0.20 and 0.45 (seed 1), `2e-5` and `1e-7`
  (seed 2), 0.20 and 0.40 (seed 3), 0.03 and 0.45 (seed 4), 0.13 and 0.29
  (seed 5), and the variance of the ranks was above its uniform value in
  eleven of the twelve runs. Measured directly against the exact posterior over 300 observations
  drawn from the prior predictive, the width of the central 50 % interval
  was 0.91, 1.00, 0.87, 0.95, 0.96, 0.94, 0.95, 0.93, 0.91, 1.02 times the
  exact width for training seeds 0 to 9, and the nominal 90 % interval
  covered 86.0 % to 90.5 % (mean 88.7 %). The estimator is miscalibrated
  (overconfident by about 6 % in width on average, up to 13 %) at these
  settings; the number of ODE steps does not change the figures (8 or 40).
  With 4000 simulations and 5000 iterations the width ratio was 0.99 and
  the 90 % coverage 88.7 % and 89.3 % (training seeds 0 and 2).
  The calibration test of the suite (200 replicates, `p > 0.01`, training
  seed 1, one of the two seeds whose width is within 1 % of the exact one)
  cannot detect an error of this size.
- **Bimodal posterior** (`theta ~ U(-2, 2)`, `x = theta^2 + N(0, 0.2^2)`,
  `x_obs = 1`): the flow puts mass 0.43 to 0.53 on the negative mode,
  mode means within 0.08 of `-1` and `+1`, mass in `|theta| < 0.5` from
  0.015 to 0.065; the mode standard deviations are 0.13 to 0.24 against
  about 0.1 exactly, so the estimate is under-confident there. The
  Gaussian estimator returns mean between -0.10 and 0.04 and standard
  deviation 0.96 to 0.99.
- **Ratio estimator, linear-Gaussian** (1000 simulations, 1000 iterations,
  3.8 s): on a grid within two posterior standard deviations at
  `x = -2, 0, 1, 3` the mean absolute logit error is 0.04 to 0.27 and the
  correlation with the exact log-ratio 0.93 to 0.998; grid posterior mean
  error at most 0.078, standard-deviation error from -0.026 to 0.051,
  `ratio_normalization` from 0.90 to 1.15.
- The methods are validated in one dimension (two for the exact-field
  density and the gradient checks). Large-scale training is out of scope.

## Rejected alternatives

- **Minibatch optimal-transport pairing** (Tong et al. 2024). It would
  straighten the trajectories, but independent pairing already meets the
  tolerances here; it is left out and not cited.
- **Training through the tape of `numerics.reverse_autodiff`.** General,
  but about ten times slower, which would have pushed the test file past
  its time budget.
- **A stochastic (Hutchinson) trace estimator** for the divergence. It
  only pays in high dimension and would add noise to a density that can be
  computed exactly here.
- **An adaptive ODE solver.** `numerics` has only fixed-step integration
  over a fixed horizon; the documentation tells users to compare step
  counts.
- **Putting the estimators in `inference`.** They need the velocity
  network and ODE code of `generative`, and `inference` (layer 2) may not
  import layer 3.
- **Sequential (multi-round) posterior or ratio estimation.** It changes
  the proposal and needs a correction that is not derived here.

## Verification

`tests/test_flow_matching_sbi.py` (28 tests, about 22 s) checks the path
and its velocity against their definitions and a finite difference in
time; the Gaussian field against a Monte Carlo regression of `u` on `x_t`;
the flow map, transported moments, and log-density of the exact field with
all three divergence modes and in two dimensions; the first-order
convergence of Euler; backpropagation against the reverse-mode tape for
both losses; the network divergence against duals and central differences;
the trained Gaussian, mixture, linear-Gaussian posterior (moments,
normalization of the density, simulation-based calibration, and rejection
of a deliberately halved spread), bimodal posterior against the Gaussian
estimator, and ratio estimator (logit, grid posterior, MCMC against the
grid); reproducibility; and input validation. Tolerances for trained
networks are about twice the largest error in the seed study above.

## References

[REF-LIPMAN-ETAL-2023], [REF-WILDBERGER-ETAL-2023],
[REF-HERMANS-BEGY-LOUPPE-2020], [REF-HERMANS-ETAL-2022],
[REF-TALTS-ETAL-2018], [REF-PAPAMAKARIOS-MURRAY-2016], [REF-KINGMA-BA-2015].
