# ADR-0128 — Competing risks, restricted-mean contrasts, online change points, and forecast reconciliation

- Status: Accepted
- Date: 2026-11-21
- Extends: the survival package (Kaplan-Meier, Nelson-Aalen, log-rank, AFT,
  Cox) and the time-series package (ARIMA, GARCH, VAR, unit roots, CUSUM,
  PELT)

## Context

Four gaps showed up in practice.

- `inference.survival` handled one event type only. With several causes of
  failure the common shortcut, the complement of a Kaplan-Meier curve that
  censors the competing events, overstates the incidence of a cause.
- The restricted mean survival time existed as a method of
  `KaplanMeierEstimate` but there was no two-sample contrast, although the
  difference and ratio of restricted means are the usual effect measures
  when hazards are not proportional.
- Change-point detection (`cusum_test`, `pelt`) was offline: it needs the
  whole series. Monitoring needs a method that updates with each
  observation and reports its uncertainty.
- Forecasts of related series (total, regions, products) made one by one do
  not add up, and the library had no reconciliation step.

## Decision

1. **`inference/survival/competing_risks.py`**: `cumulative_incidence`
   (one cause) and `cumulative_incidence_curves` (all observed causes on a
   shared grid) return the frozen `CumulativeIncidenceEstimate`. Causes are
   non-negative integers and `0` marks censoring. The estimator is
   Aalen-Johansen, `F_k(t) = sum_{t_j <= t} S(t_j-) d_kj / n_j`
   [EQ-SURV-CIF-001], evaluated at every distinct all-cause event time.
2. **Variance of the cumulative incidence** [EQ-SURV-CIF-002], derived
   here. Write `p_kj = d_kj / n_j`, `p_j = d_j / n_j`,
   `S_(j-1) = prod_(i<j) (1 - p_i)` and `F = sum_j S_(j-1) p_kj`. Given the
   risk sets, the counts `(d_kj, d_j - d_kj, n_j - d_j)` are multinomial and
   uncorrelated across `j` (the argument that yields Greenwood's formula),
   so `Var p_kj = p_kj (1 - p_kj) / n_j`, `Var p_j = p_j (1 - p_j) / n_j`,
   `Cov(p_kj, p_j) = p_kj (1 - p_j) / n_j`. `F(t)` depends on `p_kj`
   directly with derivative `S_(j-1)` and on `p_j` through the later
   survival factors with derivative `-(F(t) - F(t_j)) / (1 - p_j)`. The
   delta method gives
   `sum_j [(F(t) - F(t_j))^2 d_j / (n_j (n_j - d_j)) + S_(j-1)^2 d_kj (n_j -
   d_kj) / n_j^3 - 2 (F(t) - F(t_j)) S_(j-1) d_kj / n_j^2]`.
   Two exact reductions are tested: `F (1 - F) / n` without censoring and
   Greenwood's variance with a single cause. Intervals are `F +/- z se`,
   clipped to `[0, 1]`. No bootstrap is offered because the formula checked
   out against replicate simulation.
3. **`inference/survival/restricted_mean.py`**: `restricted_mean_survival`
   is the function form of the existing method (one implementation, no
   duplicate formula). `compare_restricted_means` returns
   `RestrictedMeanComparison` with the difference (`se = sqrt(V_1 + V_2)`),
   the ratio (log-scale delta method), their normal intervals, and the
   two-sided test of equality [EQ-SURV-RMST-002]. `tau` is a required
   argument; no default horizon is chosen from the data.
4. **`timeseries/online_changepoint.py`**: `bayesian_online_changepoint`
   implements the run-length recursion of Adams and MacKay with a constant
   hazard [EQ-TS-BOCPD-001] in log space. Models are selected by the type
   of the prior: `NormalInverseGammaPrior` (Gaussian, unknown mean and
   variance, Student-t predictive) or `GammaPoissonPrior` (counts,
   negative-binomial predictive). Both priors expose `log_predictive` and
   the closed-form `log_marginal_likelihood` of a segment, computed from
   batch statistics so that it is an independent check of the recursion.
5. **Semantics of the run length.** The recursion as published implies the
   model "every boundary between consecutive observations is a change with
   probability `H`", with `r_t` counting the observations up to `t` in the
   segment that also generates `x_(t+1)`. Consequently
   `P(r_t = 0 | x_1:t) = H` for every `t`. The library documents and tests
   this instead of presenting `P(r_t = 0)` as a change probability, and
   provides `segment_start_probability(delay)`, the filtering probability
   that a segment began at an index judged after `delay` of its
   observations. `map_change_points()` is labelled a heuristic.
6. **Truncation.** `max_run_length` drops longer runs after each step. The
   result states `truncated`, the largest `discarded_mass`, and that
   `log_evidence` is then a lower bound.
7. **`timeseries/reconciliation.py`**: `reconcile_forecasts` with
   `method="ols" | "wls" | "mint" | "bottom_up"` computes
   `G = (S' W^-1 S)^-1 S' W^-1` through a Cholesky factor of `W` and of
   `S' W^-1 S` [EQ-TS-RECON-002], or the bottom-level selection matrix.
   `hierarchy_summing_matrix` builds `S` from a parent-to-children mapping
   (trees and grouped structures; cycles and double counting are rejected).
   Weights are explicit: `variances` only with `"wls"`, `covariance` only
   with `"mint"`.
8. **Shrinkage intensity** [EQ-TS-RECON-003]. `shrink_covariance` returns
   `lambda diag(C) + (1 - lambda) C`. For fixed `lambda` the risk
   `R(lambda) = sum_(i != j) E[((1 - lambda) r_ij - rho_ij)^2]` has
   derivative `-2 sum E[r_ij ((1 - lambda) r_ij - rho_ij)]`; with
   `E r_ij = rho_ij` it vanishes at
   `1 - lambda* = sum rho_ij^2 / sum E r_ij^2`, i.e.
   `lambda* = sum Var(r_ij) / sum E[r_ij^2]`. The default is the plug-in of
   this ratio on the correlation scale, with `Var(r_ij)` estimated by the
   sampling variance of the mean of the products of standardized residuals.
   It is documented as a heuristic estimate; a fixed `intensity` can be
   supplied instead.

## Consequences

- Cumulative incidence, exponential latent times with rates 0.6 and 0.9,
  exponential censoring with rate 0.5, `n = 120`, `t = 1`, 600 replicates
  (seed 1978): truth 0.3107, mean estimate 0.3132 (standard error 0.0020);
  replicate variance 0.00230 against a mean variance estimate of 0.00221;
  coverage of the 95% interval 0.943. The naive `1 - KM` averaged 0.454
  (it estimates `1 - exp(-0.6) = 0.451`).
- Restricted mean, exponential rate 0.8, `tau = 2`, uniform censoring on
  `(1, 4)`, `n = 100`, 600 replicates: truth 0.9976, mean 1.0012 (standard
  error 0.0030); replicate variance 0.00545 against a mean estimate of
  0.00491 (the Greenwood plug-in is about 10% low at this sample size);
  coverage 0.935.
- Run-length recursion: the evidence agrees with enumeration over all 512
  segmentations of ten Gaussian observations and all 128 of eight counts to
  `1e-10` in the log; with `H = 0` it equals the conjugate marginal
  likelihood. A shift of three standard deviations at index 150 of 300
  (seed 2007, `H = 0.01`) moves the most probable run length from 150 to 5
  within five observations; the exact recursion used 45 150 predictive
  evaluations and took about 0.1 s.
- Reconciliation on an eight-series hierarchy with a random positive
  definite `W`: `tr(S G W G' S')` is 32.7 for MinT, 50.3 for WLS, 54.3 for
  OLS, 167.0 for bottom-up, and 78.2 for the unreconciled base forecasts;
  simulated squared errors agree with these traces.
- Shrinkage intensity, eight series with correlation `0.3^|i-j|`, 12
  residual vectors, 300 replicates: the derived optimum is 0.784, the best
  fixed value on a grid 0.8, the mean plug-in 0.783; the correlation risk
  falls from 4.95 (sample covariance) to 1.28 (plug-in) against 1.10 at the
  best fixed value. With six series, correlation `0.6^|i-j|` and 20 vectors
  the plug-in averages 0.30 against a derived optimum of 0.18 and barely
  improves on the sample covariance (1.10 against 1.13); it is a noisy
  estimate when little shrinkage is needed.
- The exact recursion stores `T (T + 3) / 2` probabilities; long series
  need `max_run_length`.

## Rejected alternatives

- **Reporting `P(r_t = 0 | x_1:t)` as the change-point probability.** It is
  identically `H` under the published recursion.
- **A bootstrap interval for the cumulative incidence.** Not needed once
  the closed-form variance agreed with replication; it would add a seed
  argument and cost.
- **A hazard function `H(r)` that depends on the run length.** Adams and
  MacKay allow it; the constant hazard is the case with a simple exact
  enumeration oracle, and it is the one implemented.
- **Merging the mass of truncated run lengths into the longest retained
  run.** It keeps the evidence but attaches wrong sufficient statistics to
  that mass; dropping it gives a clean lower bound.
- **Estimating `W` inside `reconcile_forecasts` from residuals.** It would
  hide the estimate that decides whether MinT helps; the covariance is an
  explicit argument.
- **Top-down and middle-out reconciliation.** They need historical
  proportions and do not satisfy `G S = I` in general.
- **Gray's test and Fine-Gray regression, Royston-Parmar or pseudo-value
  methods for restricted means.** No source could be checked for them in
  this work package.

## Verification

`tests/test_survival_extensions.py` (14 tests, about 1 s): a hand example,
`sum_k F_k = 1 - S` with ties and censoring, the empirical sub-distribution
and binomial variance without censoring, the reduction to Kaplan-Meier and
Greenwood with one cause, the inequality against the naive estimate, the
exponential truth with replicate variance and coverage, the uncensored
restricted mean and its variance, the exponential restricted mean with
replicate variance, the contrast arithmetic, level and coverage of the
two-sample comparison, and validation.

`tests/test_timeseries_online_and_reconciliation.py` (31 tests, about 2 s):
conjugate evidences against one- and two-dimensional quadrature, the
recursion against enumeration of all segmentations (final and prefix
evidences), `H = 0` and `H = 1`, `P(r_t = 0) = H`, detection of a Gaussian
mean shift and a Poisson rate shift, truncation as a lower bound, summing
matrices for trees and grouped structures, coherence, invariance of
coherent input, idempotence and `G S = I`, OLS against Householder least
squares, the trace ordering for three random covariances and against 20
random unbiased competitors, a simulation with known error covariance,
the shrinkage intensity against the risk-optimal fixed intensity, and
validation.

## References

[REF-AALEN-JOHANSEN-1978], [REF-GREENWOOD-1926],
[REF-KLEIN-MOESCHBERGER-2003], [REF-ADAMS-MACKAY-2007], [REF-MURPHY-2007],
[REF-GELMAN-ETAL-2013], [REF-WICKRAMASURIYA-ATHANASOPOULOS-HYNDMAN-2019].
