# ADR-0040 — EM, Gaussian mixtures, hierarchical models, and predictive checks

- Status: Accepted
- Date: 2026-10-12
- Closes (roadmap): the EM part of "Likelihood, score, Fisher information,
  MLE/MAP, EM"; the posterior predictive check part of "Bayesian
  priors/posteriors, conjugacy, posterior predictive checks"; and the
  hierarchical-model part of "Linear/nonlinear regression, GLMs,
  regularization, hierarchical models"

## Context

The library had samplers (Metropolis, HMC, NUTS, ADVI), optimizers, and a
posterior predictive check limited to linear-Gaussian predictive laws. It had
no EM algorithm, no mixture models, no hierarchical models, and no generic
predictive check that works with sampler output.

## Decision

1. **Four new layer-2 modules** in `probstat.inference`:
   - `em`;
   - `mixtures`;
   - `hierarchical`;
   - `predictive_checks`.

   They import only from `foundations`, `linalg`, `optimization` and
   `distributions` (all layers 0 and 1), and from sibling `inference`
   modules: `fitting.information_criteria`, `hamiltonian`, `mcmc` and
   `goodness_of_fit`. The layer graph is unchanged and acyclic.
2. **Generic EM driver.** `e_step(theta) -> (statistics, objective)` and
   `m_step(statistics) -> theta`. The driver checks monotonicity at every
   step. A decrease beyond `slack (1 + |l|)` is recorded, and by default
   reported as an `EMMonotonicityWarning`; it can instead raise. The driver
   stops on a relative objective change.
3. **Mixture regularization is a fixed-penalty MAP.** The covariance floor
   enters as `-(1/2) sum_k tr(Lambda S_k^-1)`, with
   `Lambda = c diag(var_j)` fixed for the run. The M-step
   `S_k = (W_k + Lambda) / N_k` is therefore exact, and EM stays monotone for
   the penalized objective.

   The alternative rule, `S_k = W_k / N_k + c I`, adds a floor after
   dividing by `N_k`. That changes the implied objective at every
   iteration, so it was rejected.

   AIC and BIC use the unpenalized log-likelihood.
4. **Initialization and ordering.**
   - Initialization is k-means++ `D^2` seeding from a `RandomStream`
     (restart `r` uses child `("gmm-restart", r)`), followed by at most 20
     Lloyd iterations.
   - Components are returned sorted by mean vector, then weight, so fits
     are free of label switching.
5. **Censored normal data rather than missing-at-random multivariate
   normal.**
   - The E-step reuses the stable truncated-normal moments.
   - The observed log-likelihood is exact, so the monotonicity check and a
     direct Nelder-Mead cross-check are true tests.
   - Detection limits and type-I censoring are common in UQ data.

   Missing-at-random multivariate normal EM stays open.
6. **Exact hierarchical posterior on a transformed grid.** `p(tau | y)` is
   evaluated on a grid uniform in `u = log(1 + tau / min s)`, integrated
   with Simpson weights, and sampled by a piecewise-linear inverse CDF in
   `u`.
   - The upper end is doubled until the density falls below `1e-14` of its
     peak.
   - A tail-slope test, which requires a log-log slope below `-1.2`, rejects
     posteriors that are apparently improper.

   A normal prior on `mu` is supported in closed form. The `tau` prior is
   any log-density callable.
7. **NUTS uses the non-centered parameterization** with an analytic gradient
   and `target_acceptance = 0.9`. Raw draws are mapped to
   `(mu, tau, theta)` with `MCMCResult.transformed`, so every existing
   diagnostic still applies.
8. **Generic predictive check is a new function,
   `posterior_predictive_pvalues`.** The existing
   `posterior_predictive_check(PosteriorPredictive, ...)` keeps its
   signature.
   - The new function accepts `MCMCResult`, `HamiltonianResult`, or plain
     draws.
   - Data-only statistics and realized discrepancies are separate mappings,
     so their types are explicit.
   - For MCMC draws the MCSE uses the indicator chains' ESS. For plain draws
     it uses the binomial formula, and it falls back to that formula when
     the ESS is undefined.
9. **PIT is the simple, non-LOO version.** Its documentation states that
   it is pulled toward 0.5 and that its KS p-value is descriptive.

## Evidence

All values come from runs in this session:

| Area | Result |
|---|---|
| GMM selection | BIC picks K = 3 on seeded data: 2959.40, against 3187.32 for K = 2 and 2989.56 for K = 4. |
| GMM traces | Monotone in every run. |
| Censored EM | Matches direct maximization to better than `1e-5`. |
| Eight schools | Grid and NUTS posterior means agree within 4 MCSE for `mu`, `tau` and `theta_1` over six seeds; the largest `|z|` is 2.08. |
| Grid formula | Matches the closed-form evidence to `2.1e-14`. |
| Variance-statistic ppp under the true model | Uniform: KS p-value 0.975, mean 0.502. |
| Mean-statistic ppp | Concentrated at 0.5 (sd 0.037). |
| Misspecified data | ppp near 0. |

## Consequences

- EM convergence is linear and has no acceleration. No standard errors
  (Louis' method) are produced.
- Mixture fitting is pure Python, `O(n K d^2)` per iteration.
- Beta-binomial empirical Bayes plugs in point estimates.
- `Betancourt-Girolami` is cited as the 2013 arXiv preprint. The book-chapter
  venue was not verified.
