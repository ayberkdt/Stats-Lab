# EM, Mixtures, Hierarchical Models, and Posterior Predictive Checks

This guide covers four modules in `probstat.inference`:

- `em` — a generic EM driver and EM for censored normal data;
- `mixtures` — Gaussian mixture models;
- `hierarchical` — the eight-schools normal model and beta-binomial
  empirical Bayes;
- `predictive_checks` — posterior predictive p-values and PIT values.

Every number below was printed by `examples/hierarchical_lab.py`, by the
tests `tests/test_em_and_mixtures.py` and `tests/test_hierarchical_and_ppc.py`,
or by short scripts run in the same session with the seeds stated. They are
the library's own outputs. None is copied from the literature.

## 1. The EM driver (EQ-EM-001)

`expectation_maximization(initial, e_step, m_step)` alternates two steps:

- `e_step(theta) -> (statistics, objective)`, which returns the expected
  complete-data statistics and the observed-data objective `l(theta)`;
- `m_step(statistics) -> theta`.

It stops when the objective changes by at most
`tolerance * (1 + |l|)`.

An exact EM step cannot decrease `l` (Dempster, Laird and Rubin 1977). The
driver therefore checks every step. A decrease larger than
`decrease_slack * (1 + |l_t|)` sets `monotone = False`. It then warns
(`EMMonotonicityWarning`), raises, or only records the decrease, depending on
`on_decrease`. In practice a decrease means the E-step or M-step is coded
wrong. `tests/test_em_and_mixtures.py` checks this with a deliberately broken
M-step.

EM is guaranteed to reach a stationary point under Wu's (1983) conditions.
It is not guaranteed to reach the global maximum. Use restarts when the
likelihood is multimodal.

## 2. Gaussian mixtures (EQ-GMM-001..004)

`fit_gaussian_mixture(data, K, covariance_type="full" | "diag")` accepts
scalars (univariate) or equal-length sequences.

**Covariance floor.** The unpenalized mixture likelihood is unbounded: a
component that collapses onto a single point drives it to infinity. The fit
therefore maximizes a penalized likelihood:

`l_pen = l - 1/2 sum_k tr(Lambda S_k^-1)`, with
`Lambda = diag(c * var_j)` and `c = regularization` (default `1e-6`).

The exact M-step is then `S_k = (W_k + Lambda) / N_k`. `Lambda` is fixed for
the whole run, so EM is still exactly monotone for `l_pen`. AIC and BIC use
the unpenalized `l`.

A floor added *after* dividing by `N_k`, as some libraries do, changes the
objective at every iteration. With that rule the monotonicity check would no
longer be valid.

**Initialization.** Starting values come from k-means++ `D^2` seeding (Arthur
and Vassilvitskii 2007), driven by the `RandomStream` passed as `seed`,
followed by up to 20 Lloyd iterations. `restarts` runs that many
independently seeded EM fits and keeps the one with the largest `l_pen`.

**Canonical ordering.** Components are sorted by mean vector
(lexicographically), then by weight. Two fits of the same data with
different seeds are therefore directly comparable. The test
`test_canonical_order_removes_label_switching` gives means equal to four
decimals.

**Choosing K.** `select_gaussian_mixture` fits each candidate `K` and
returns the fit with the smallest BIC (or AIC). Lab output, `n = 400` from a
three-component bivariate mixture (seed 11, `tolerance = 1e-7`, at most 150
iterations):

| K | log L | params | BIC | iterations | monotone |
|---|---|---|---|---|---|
| 1 | -1816.59 | 5 | 3663.14 | 1 | True |
| 2 | -1560.71 | 11 | 3187.32 | 3 | True |
| 3 | -1428.77 | 17 | **2959.40** | 7 | True |
| 4 | -1425.88 | 23 | 2989.56 | 150 | True |
| 5 | -1418.96 | 29 | 3011.68 | 150 | True |

The over-fitted `K = 4, 5` runs hit the iteration cap without converging.
EM is slow when components overlap or are redundant. Their log-likelihoods
are still enough to rank them by BIC.

With all 600 points, recovery is within 0.05 in the weights, 0.15 in the
means, and 0.25 in the covariance entries
(`test_recovers_seeded_parameters`).

## 3. EM for censored normal data (EQ-EM-CENS-001..002)

We implemented censored normal data rather than missing-at-random
multivariate normal data, for three reasons:

1. **Use in UQ.** Detection limits, gauge saturation and type-I censored
   life tests are common, and dropping or substituting the censored values
   is a common error.
2. **Reuse.** The E-step needs the truncated-normal mean and variance, which
   `distributions.TruncatedNormal` already computes stably, including deep
   tails.
3. **Testability.** The observed-data log-likelihood can be computed exactly
   (`censored_normal_log_likelihood`). The monotonicity check and an
   independent direct maximization are therefore true tests of the code.

`fit_censored_normal(intervals)` takes `(lower, upper)` pairs:

- equal ends are an exact value;
- `-inf` marks left censoring and `inf` right censoring;
- two finite ends mark interval censoring.

Lab output: 300 draws from `Normal(10, 2)`, censored below 8 and above 13
(seed 5); 79 values are censored.

| method | mean | sd |
|---|---|---|
| drop censored values | 10.248 | 1.287 |
| substitute the limit | 10.056 | 1.648 |
| EM (censored likelihood) | 9.918 | 2.143 |

EM converged in 14 iterations, monotonically, to `log L = -568.8272`.
Nelder-Mead applied directly to the observed log-likelihood agrees with EM to
better than `1e-5` in both parameters (the test). Both naive methods
understate the spread badly.

## 4. Hierarchical normal model — eight schools (EQ-HIER-001..004)

The model is

- `y_j ~ N(theta_j, s_j^2)`, with known `s_j`;
- `theta_j ~ N(mu, tau^2)`;
- by default, flat priors on `mu` and on `tau >= 0`.

The data are the classic `y = (28, 8, -3, 7, -1, 1, 18, 12)` and
`s = (15, 10, 16, 11, 9, 11, 10, 18)`. The values were checked in this
session against the PINTS toy-model source, which cites BDA3.

**Exact route.** `hierarchical_normal_posterior` integrates `theta` and `mu`
analytically. The marginal posterior `p(tau | y)` (EQ-HIER-002) is then
evaluated on 2001 points, uniform in `u = log(1 + tau / min s)`, and
integrated with Simpson weights in `u`. The upper end is doubled until the
density, including its Jacobian, falls below `1e-14` of its peak; here that
is `tau = 6752`. Before integrating, the code checks the tail slope. A
uniform `tau` prior with only two groups is rejected as improper, because
the tail decays like `tau^-(J-1)`.

Moments of `mu`, `theta_j` and `B_j` follow from the total-expectation
formulas of EQ-HIER-001. Exact joint draws use inverse-CDF sampling of
`tau`, then the normal conditionals.

Two independent checks support the formula:

- **Closed-form evidence.** With a `N(2, 7^2)` prior on `mu`, the gridded
  `log p(tau | y)` differs from the closed-form multivariate-normal evidence
  `log N(y | m_0 1, diag(s^2 + tau^2) + s_0^2 11')` by a constant. The
  spread of that constant over the grid was `2.1e-14`.
- **Brute-force quadrature.** A 2-D quadrature over `(mu, tau)` (step 0.05 in
  `tau` up to 200, step 0.1 in `mu`) gave `E[tau | y] = 6.592`. The grid gives
  6.575. The left-endpoint rule omits about `h p(0) / 2` of the mass, which
  accounts for the 0.015 difference.

**NUTS route.** `sample_hierarchical_normal` runs the existing
`no_u_turn_sampler` on the non-centered parameterization
`theta_j = mu + tau z_j`, `tau = exp(eta)` (EQ-HIER-003). It uses an
analytic gradient and `target_acceptance = 0.9`.

Lab output: 4 chains, 500 warm-up and 1000 kept draws each, seed 2026. NUTS
took 7.4 s in that run (5.3 to 5.4 s in later runs), with 1 divergence in 4000 draws.

| quantity | grid mean | grid sd | NUTS mean | NUTS sd | MCSE | z | E[B_j] |
|---|---|---|---|---|---|---|---|
| mu | 7.932 | 5.178 | 8.257 | 5.101 | 0.156 | 2.08 | |
| tau | 6.575 | 5.650 | 6.561 | 5.447 | 0.168 | -0.09 | |
| theta_1 | 11.400 | 8.341 | 11.652 | 8.421 | 0.212 | 1.19 | 0.825 |
| theta_2 | 7.895 | 6.275 | 8.039 | 6.125 | 0.121 | 1.19 | 0.723 |
| theta_3 | 6.131 | 7.765 | 6.247 | 7.867 | 0.153 | 0.76 | 0.839 |
| theta_4 | 7.645 | 6.546 | 7.845 | 6.606 | 0.129 | 1.55 | 0.749 |
| theta_5 | 5.126 | 6.357 | 5.203 | 6.427 | 0.133 | 0.58 | 0.693 |
| theta_6 | 6.139 | 6.710 | 6.132 | 6.672 | 0.130 | -0.05 | 0.749 |
| theta_7 | 10.667 | 6.786 | 10.823 | 6.745 | 0.148 | 1.05 | 0.723 |
| theta_8 | 8.457 | 7.888 | 8.602 | 8.041 | 0.171 | 0.85 | 0.862 |

Here `z = (NUTS mean - grid mean) / MCSE`. The `theta` errors share the sign
of the `mu` error because every `theta_j` is correlated with `mu`.

The grid quantiles of `tau` are:

| quantile | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|
| `tau` | 0.49 | 2.47 | 5.24 | 9.12 | 17.18 |

The same comparison over six seeds (same settings) gave these z-scores:

| seed | mu | tau | theta_1 | divergences |
|---|---|---|---|---|
| 1 | -1.53 | -0.36 | -0.87 | 0 |
| 2 | -1.06 | 1.36 | 0.30 | 0 |
| 3 | -1.11 | -1.75 | -0.24 | 0 |
| 4 | -0.76 | -0.05 | -1.42 | 0 |
| 5 | -0.54 | 0.01 | 0.40 | 0 |
| 2026 | 2.08 | -0.09 | 1.19 | 1 |

The test requires `|z| < 4` for `mu`, `tau` and `theta_1`, and the medians of
`tau` to agree within 0.6.

**Read the `tau` MCSE with care.** Under a uniform prior the posterior of
`tau` has a polynomial tail, roughly `tau^-7` here. The MCSE of its mean
therefore converges slowly and is itself noisy.

**Shrinkage.** `B_j = s_j^2 / (s_j^2 + tau^2)` is the weight that
`E[theta_j | tau, y]` puts on the pooled mean. Its posterior mean `E[B_j | y]`
ranges from 0.693 (school 5, `s = 9`) to 0.862 (school 8, `s = 18`). Noisier
schools are pulled harder toward `E[mu | y] = 7.93`. School 1's estimate of
28 becomes a posterior mean of 11.40.

**Priors.** `half_cauchy_log_prior(A)` gives the half-Cauchy prior recommended
by Gelman (2006). With `A = 5` it lowers both `E[tau | y]` and the 95 %
quantile (test).

## 5. Beta-binomial empirical Bayes (EQ-HIER-BB-001)

`fit_beta_binomial(successes, trials)` maximizes the beta-binomial marginal
likelihood over `(logit m, log kappa)` using BFGS. It reports:

- the plug-in posterior means `(y_j + alpha) / (n_j + kappa)`;
- the shrinkage weights `kappa / (n_j + kappa)`.

Lab output: 60 groups from `Beta(4, 12)` with 20 to 80 trials each
(seed 17). The fit gives `alpha = 3.839`, `beta = 11.485`, prior mean 0.2505
(true value 0.25), and `kappa = 15.32`. Group 0 moves from 22/53 = 0.415 to
0.378, with shrinkage 0.224.

The plug-in posterior ignores the uncertainty in `(alpha, beta)`. If the
counts are not over-dispersed, the maximum lies at `kappa -> inf`.

## 6. Posterior predictive checks (EQ-PPC-001..002)

`posterior_predictive_pvalues(draws, simulate, observed, statistics=...,
discrepancies=...)` accepts three kinds of `draws`:

- an `MCMCResult`;
- a `HamiltonianResult`;
- a plain sequence of parameter vectors (treated as independent draws).

For each draw it simulates one replicated dataset and reports, for every
test quantity:

- the upper and lower tail probabilities;
- the two-sided value `min(1, 2 min(upper, lower))`;
- a Monte Carlo SE.

For MCMC draws the SE uses the effective sample size of the indicator chain.
For independent draws it uses `sqrt(p (1 - p) / S)`. `discrepancies` take
`(y, theta)` and give the realized discrepancies of Gelman, Meng and Stern
(1996). `max_draws` thins all chains by a common stride.

The existing `posterior_predictive_check` (linear-Gaussian predictive law,
Mahalanobis discrepancy) keeps its name and signature. The generic function
has a different name to avoid breaking it.

**Calibration.** The test model is a normal model with known `sigma = 1` and
a flat prior on `mu`. It uses `n = 20` observations, 200 posterior draws and
120 seeded datasets.

| test quantity | data | mean p | share p <= 0.1 | KS p-value vs U(0,1) |
|---|---|---|---|---|
| sample variance | true model | 0.502 | 0.10 | 0.975 |
| sample mean | true model | 0.501 (sd 0.037) | 0.00 | 2e-19 |
| sample variance | data sd = 2 | median 0.000, max 0.02 (20 datasets) | | |

The sample variance is ancillary for `mu`, so its p-value is uniform. The
sample mean is fitted directly by the posterior, so its p-value piles up at
0.5. This is the known conservativeness of posterior predictive p-values. A
mid-range p-value from a statistic the model fits directly is no evidence of
good fit.

For the eight schools, the checks on the 4000 NUTS draws (seed 4) gave
p-values (MCSE) of:

| test quantity | p-value (MCSE) |
|---|---|
| `max` | 0.495 (0.009) |
| `min` | 0.168 (0.006) |
| `mean` | 0.467 (0.009) |
| `sd` | 0.789 (0.007) |
| realized `chi2 = sum ((y_j - theta_j)/s_j)^2` | 0.693 (0.008) |

None of these flags a misfit.

**PIT.** `pit_values(observed, replicates)` gives `Pr(y_rep,i <= y_i)`.
Ties are counted at one half, or randomized with `randomize_ties=True`.
`pit_uniformity` runs a KS test against `U(0, 1)`. This simple version uses
replicates conditioned on all of the data. It is not leave-one-out, so its
values are pulled toward 0.5 and the KS p-value is descriptive only.

The eight-schools PIT values were 0.83, 0.49, 0.30, 0.48, 0.27, 0.33, 0.72
and 0.56.

## 7. Limitations

- EM convergence is linear and can be very slow for overlapping or
  redundant mixture components. The driver has no acceleration (SQUAREM,
  Aitken) and computes no standard errors (Louis' method or a numerical
  Hessian).
- The mixture code is pure Python and costs `O(n K d^2)` per iteration. It
  is meant for moderate `n` and `d`.
- The censored-normal EM is univariate and has no covariates (no Tobit
  regression). EM for missing-at-random multivariate normal data is not
  implemented.
- The exact hierarchical route assumes known `s_j` and the normal-normal
  structure. Other likelihoods need the NUTS route with a user-written
  log-density.
- Empirical Bayes plugs in point estimates. It is not a full Bayesian
  treatment of `(alpha, beta)`.
- PIT values are not leave-one-out.
