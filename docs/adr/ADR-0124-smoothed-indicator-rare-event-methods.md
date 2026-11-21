# ADR-0124 — Smoothed-indicator rare-event methods and BUS

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0106 (subset-simulation kernels) and the cross-entropy
  importance sampling of `reliability.adaptive`

## Context

The adaptive rare-event methods of `reliability` approach the failure set
through nested events `G <= b_j` placed at sample quantiles: subset
simulation, and cross-entropy importance sampling with a Gaussian proposal.
Two properties limit them. Each level uses only the fraction of its sample
below the threshold, and the cross-entropy method has one Gaussian, which
cannot follow several failure regions.

Papaioannou, Papadimitriou and Straub (2016) and Papaioannou, Geyer and
Straub (2019) replace the indicator by the smooth function
`Phi(-G / sigma)` with a decreasing `sigma`. Straub and Papaioannou (2015)
turn Bayesian updating into a rare-event problem, so that the same
machinery yields posterior draws and the model evidence. The library had
neither.

None of the three papers could be read during this work (no network
access). The formulas below were therefore derived and checked against
exact oracles; where a setting is a choice of this library and not known to
be the source's, the text says so.

## Decision

1. **Three new modules in `reliability`** (layer 5, no new layer edges):
   `improved_cross_entropy.py` (`improved_cross_entropy`,
   `smoothing_scale`, `ImprovedCrossEntropyResult`),
   `sequential_importance.py` (`sequential_importance_sampling`,
   `SequentialImportanceResult`), and `bayesian_updating.py`
   (`bus_limit_state`, `bus_subset_simulation`, `BUSResult`). All work in
   independent standard-normal space on a `ModelLimitState` and count model
   evaluations through the model's counter.
2. **Smooth targets** [EQ-REL-ICE-001]: `h_k` proportional to
   `Phi(-G / sigma_k) phi_n`, with `h_0 = phi_n`. `log Phi` is evaluated
   with `Normal.log_cdf` beyond the range of `erfc`.
3. **Scale search** [EQ-REL-ICE-002]: `smoothing_scale` returns the first
   `sigma` below the previous one at which the weights
   `Phi(-G_i / sigma) b_i` have the target c.o.v. (default `1.5`), by a
   geometric downward scan (factor 0.7) and bisection on `log sigma`. The
   c.o.v. is not guaranteed monotone in `sigma`, hence the scan. The two
   cases without a crossing return a documented end point.
4. **Improved cross entropy**: weighted maximum likelihood with all `N`
   samples [EQ-REL-ICE-003], one Gaussian with full covariance or a Gaussian
   mixture by weighted EM started from weighted k-means++ seeds; components
   below a proportion of `1e-6` are dropped. Stop when the c.o.v. of
   `I(G <= 0) / Phi(-G / sigma_k)` over the current sample is at most the
   target [EQ-REL-ICE-004]; the estimate uses that last sample
   [EQ-REL-ICE-005], so no separate final stage is needed and the cost is
   `N (levels + 1)`. The von Mises-Fisher-Nakagami mixture is not
   implemented.
5. **Covariance floor (library choice)**: eigenvalues of every fitted
   covariance are raised to `0.55` by default, as in
   `cross_entropy_importance_sampling`, because `phi_n / q` has infinite
   variance below `1/2` [EQ-REL-CE-003]. `covariance_floor=None` gives the
   unguarded fit and sets `weights_heavy_tailed`.
6. **Sequential importance sampling** [EQ-REL-SIS-001]: with `eta_0 =
   phi_n` and `P_0 = 1`, the first incremental weight is
   `Phi(-G / sigma_1)`; dividing by `Phi(-G / inf) = 1/2` would double the
   estimate. Levels are populated by multinomial resampling and
   conditional-sampling (pCN) chains with acceptance probability
   `min(1, Phi(-G(v)/sigma) / Phi(-G(u)/sigma))` [EQ-REL-SIS-002], whose
   spread is adapted by the rule of adaptive conditional sampling
   [EQ-REL-SUBSET-005]. A new chain function is needed because the
   subset-simulation chains accept by a threshold; the proposal and the
   adaptation are the same.
7. **Chain layout (library choice)** [EQ-REL-SIS-003]: 20 chains of 50
   states per 1000 samples, 10 burn-in steps, adaptation every 2 chains.
   See the measurements below for the reason.
8. **C.o.v. of the sequential estimate** [EQ-REL-SIS-004]: batch means over
   chains, summed over the factors. It is documented as a heuristic.
9. **BUS** [EQ-REL-BUS-001]: `bus_limit_state` builds a
   `ProbabilisticModel` on `d + 1` standard normals with the limit state
   `log Phi(u_p) - log c - log L(theta(u))`, the logarithm of the event
   `p <= c L(theta)`; the prior is a `JointInput` (or `None` for a
   standard-normal prior) and the likelihood is given as a log-likelihood.
   `bus_subset_simulation` repeats the level loop of `subset_simulation`
   with its own kernels `_cs_level` and `_acs_level` on the same random
   stream, because `subset_simulation` does not return its states. With
   the same seed and options the two give the same probability, thresholds,
   and call count (tested). An optional further level at threshold 0 yields
   `N` posterior states. Every evaluated point is checked for
   `log c + log L <= 0` [EQ-REL-BUS-003].

## Consequences

Measured with `N = 1000` per level, defaults, seeds 0-19 (20 runs each).
"mean" is the mean estimate over the exact value, "c.o.v." the standard
deviation of the 20 estimates over the exact value, "reported" the mean
reported c.o.v., "calls" the mean number of model evaluations, and "work"
is `c.o.v. sqrt(calls)` (smaller is better). Linear states are
`G = beta - sum(u) / sqrt(d)`; the parallel system is
`max(2.5 - u1, 2 - 0.6 u1 - 0.8 u2)` with exact value
`Phi_2(-2.5, -2; 0.6) = 2.1940e-3`; the four-branch reference is
`2.2245e-3`. Subset simulation uses `sampler="acs"` and the default
`"mma"`.

| Case | Method | mean | c.o.v. | reported | calls | work |
| --- | --- | --- | --- | --- | --- | --- |
| linear, `beta = 3`, `d = 2` | iCE Gaussian | 0.988 | 0.037 | 0.048 | 4000 | 2.3 |
| | SIS | 1.033 | 0.256 | 0.224 | 4540 | 17.3 |
| | subset (acs) | 0.994 | 0.248 | 0.220 | 2890 | 13.3 |
| | subset (mma) | 0.997 | 0.229 | 0.247 | 2725 | 11.9 |
| linear, `beta = 3`, `d = 20` | iCE Gaussian | 0.995 | 0.089 | 0.081 | 6250 | 7.0 |
| | SIS | 1.040 | 0.274 | 0.231 | 4540 | 18.5 |
| | subset (acs) | 0.983 | 0.231 | 0.222 | 2935 | 12.5 |
| | subset (mma) | 1.013 | 0.314 | 0.225 | 2980 | 17.1 |
| linear, `beta = 4.5`, `d = 2` | iCE Gaussian | 1.017 | 0.077 | 0.061 | 5700 | 5.8 |
| | SIS | 0.932 | 0.347 | 0.359 | 6900 | 28.8 |
| | subset (acs) | 0.982 | 0.405 | 0.335 | 5500 | 30.0 |
| | subset (mma) | 1.097 | 0.635 | 0.393 | 4930 | 44.6 |
| linear, `beta = 4.5`, `d = 20` | iCE Gaussian | 0.974 | 0.088 | 0.100 | 12250 | 9.8 |
| | SIS | 1.027 | 0.493 | 0.364 | 6900 | 41.0 |
| | subset (acs) | 1.002 | 0.323 | 0.333 | 5500 | 23.9 |
| | subset (mma) | 1.011 | 0.443 | 0.378 | 5545 | 33.0 |
| parallel system | iCE Gaussian | 1.021 | 0.044 | 0.050 | 4000 | 2.8 |
| | iCE mixture (4) | 0.988 | 0.043 | 0.050 | 4000 | 2.7 |
| | SIS | 1.040 | 0.327 | 0.218 | 4540 | 22.0 |
| | subset (acs) | 1.010 | 0.218 | 0.206 | 2800 | 11.5 |
| | subset (mma) | 1.048 | 0.390 | 0.241 | 2614 | 19.9 |
| four-branch | iCE Gaussian | 0.964 | 0.107 | 0.147 | 51000 | 24.1 |
| | iCE mixture (4) | 0.998 | 0.065 | 0.053 | 4000 | 4.1 |
| | SIS | 0.970 | 0.187 | 0.224 | 4540 | 12.6 |
| | subset (acs) | 1.017 | 0.179 | 0.209 | 2800 | 9.5 |
| | subset (mma) | 0.965 | 0.235 | 0.235 | 2594 | 11.9 |

- **Improved cross entropy** has 1.8 to 5.8 times less work than subset
  simulation with adaptive conditional sampling on these cases (the work
  column squared is the cost for a fixed c.o.v., so the saving in calls is
  3 to 33 fold), provided the family fits the failure set. In 20 dimensions the single Gaussian has 230
  parameters and needs more levels (12250 calls at `beta = 4.5`).
- **A single Gaussian on the four-branch function** did not meet the
  stopping rule in any of the 20 runs within 50 levels (51000 calls) and is
  the most expensive entry of the table; the mixture with four components
  converged in every run in three levels.
- **Reported against replicate c.o.v.** For improved cross entropy the
  ratio of replicate to mean reported c.o.v. was 0.73 to 1.27 over the eight
  rows, and the largest `|estimate - exact|` was 3.33 reported standard
  errors over 160 runs. For sequential importance sampling the ratio was
  0.84 to 1.50 and the largest `|log(estimate / exact)|` was 3.4 reported
  c.o.v. over 120 runs.
- **Sequential importance sampling is not better than subset simulation
  here.** Its work is 0.96 to 1.9 times that of subset simulation with
  adaptive conditional sampling. With the layout first tried (100 chains of
  10 states, no burn-in) it was much worse: at `beta = 4.5`, `d = 20` the
  replicate c.o.v. over 30 seeds was 1.61 with 5680 calls against a
  reported 0.26, and single runs were off by a factor of 8. The cause is
  that a level sample which lags behind its target biases the next ratios
  in the same direction; the ratio errors of one run had the same sign over
  consecutive levels (checked against the exact
  `P_k = Phi(-beta / sqrt(1 + sigma_k^2))` of the linear state). Long
  chains with burn-in remove most of it, which is why the defaults differ
  from many short chains.
- **Unguarded covariance** (`covariance_floor=None`, 20 seeds): replicate
  c.o.v. 0.034, 0.064, 0.040 on the first three linear cases (lower than
  with the floor) but 0.262 at `beta = 4.5`, `d = 20` with one of 20 runs
  unconverged after 50 levels; with the mixture on the parallel system the
  replicate c.o.v. was 1.56 times the reported one.
- **BUS, conjugate normal-normal problem** (prior `N(1, 2^2)`, five
  observations with noise 0.5; exact posterior mean 3.0543, variance
  0.04938, log evidence -4.5163, acceptance probability 0.0651 with
  `c = 1 / sup L`), seeds 0-4, `N = 1000`: posterior means 3.041 to 3.077,
  sample variances 0.041 to 0.061, log evidence -4.619 to -4.373 with a
  reported c.o.v. of 0.10 to 0.11, 2400 likelihood calls per run (1900 for
  the evidence, 500 for the final level). With `log c` raised by 2 the
  check counted 390 violating points and the sample variance was 0.098.

## Audit addendum (independent re-measurement, 2026-11-21)

- **Sources.** The papers themselves could still not be opened. The
  reference implementations of the authors' group were read instead
  (`iCE_GM.py` and `SIS_GM.py` in the ERA-Software "Overview" repository on
  GitHub). They agree with this ADR on: the scale chosen so that the
  c.o.v. of `Phi(-G / sigma) w` equals the target, searched below the
  previous scale; the stopping rule (c.o.v. of `I(G <= 0) / Phi(-G / sigma)`
  over the current sample at most the target); the estimate from the last
  sample, `mean(w I)`; in sequential importance sampling the first weight
  `Phi(-G / sigma_1)` without a factor, later weights as ratios,
  `S_k = mean(w)`, and the final estimate
  `prod(S_k) mean(I / Phi(-G / sigma_L))`. Neither file contains a
  covariance floor, a c.o.v. of the sequential estimate, or the chain
  layout used here; these remain choices of this library. The default
  target 1.5 was not found as a default in those files (it is an argument
  there). The BUS constant and evidence formula were not checked against a
  source text.
- **Improved cross entropy, mean of estimate over exact** (linear states,
  `N = 1000`, floor 0.55, seeds 1000 onwards): `beta = 3`, `d = 2`: 1.004
  (100 runs, Monte Carlo standard error 0.005); `beta = 4.5`, `d = 2`:
  0.999 (100 runs, 0.006); `beta = 5.5`, `d = 2`: 0.997 (100 runs, 0.006);
  `beta = 4.5`, `d = 20`: 0.981 (60 runs, 0.013); `beta = 5.5`, `d = 20`:
  0.997 (50 runs, 0.015; 4 of the 50 runs did not converge within 50
  levels and the mean cost was 22660 calls). Replicate over mean reported
  c.o.v.: 0.94, 1.00, 0.95, 1.01, 0.82. The fraction of runs further than
  two reported standard errors from the exact value was 3 to 5 % in two
  dimensions and 12 % at `beta = 4.5`, `d = 20`.
- **The covariance floor changes efficiency, not the expectation**: the
  weights use the exact density of the floored proposal, and without the
  floor the means were 0.995 (`beta = 4.5`, `d = 2`, 100 runs) and 1.004
  (`d = 20`, 30 runs). In two dimensions the floor costs efficiency
  (replicate c.o.v. 0.061 against 0.037 without it).
- **Stopping-rule bias.** With small samples the estimate is biased
  upwards: 1.023 (standard error 0.007) at `N = 100` and 1.037 (0.015) at
  `N = 30` for `beta = 3`, `d = 2`; at `N = 30` the replicate c.o.v. was
  1.3 times the reported one and 4.5 % of the runs were more than three
  reported standard errors away.
- **Sequential importance sampling, 60 runs per case, defaults**:
  replicate over mean reported c.o.v. 1.44 (`beta = 3`, `d = 2`), 1.29
  (`beta = 4.5`, `d = 20`), 1.30 (`beta = 5.5`, `d = 2`), 1.37
  (`beta = 3`, `d = 50`); 10 to 17 % of the estimates were further than
  two reported c.o.v. from the exact value on the log scale; the mean of
  estimate over exact was 0.96 to 1.03 (standard errors 0.04 to 0.08).
  The heuristic c.o.v. understated the scatter in every case tried.
- **BUS, conjugate problem, 60 seeds**: evidence mean over exact 1.012
  with `c = 1 / sup L` and 1.033 with `c` smaller by `e^3`; the standard
  deviation of the log evidence was 0.089 and 0.339 against reported
  c.o.v. of 0.107 and 0.263. `c` too large by factors `e^0.05`, `e^0.3`
  and `e^2` was flagged in 60 of 60 runs each (the check sees it as soon
  as a point near the likelihood maximum is evaluated; a violation in a
  region that is never visited cannot be seen). The 1000 returned draws
  had 47 % distinct values, and the posterior mean scattered between runs
  like that of about 250 independent draws (140 with the smaller `c`).

## Rejected alternatives

- **The von Mises-Fisher-Nakagami mixture.** Excluded from this work
  package; it needs the normalizing constants and the approximate
  maximum-likelihood updates of two further families, none of which could
  be checked against the source.
- **Running the smooth levels as subset simulation in an augmented space.**
  `Phi(-G / sigma) = P(G - sigma xi <= 0)` for an extra standard normal
  `xi`, so `h_k` is the marginal of a conditional normal law and the
  existing threshold-acceptance chains would apply unchanged. The
  Metropolis ratio of [EQ-REL-SIS-002] integrates `xi` out analytically and
  was preferred.
- **Returning the states from `subset_simulation`.** It would change a
  public result type used by other packages; the BUS driver repeats the
  loop with the shared kernels and a test pins the equality.
- **Passing the likelihood itself to BUS.** Products of many densities
  underflow; a log-likelihood and `log c` keep the limit state finite.
- **An automatic or adaptive choice of `c`.** Not part of the cited paper
  as far as could be confirmed; the constant is a required argument and the
  run reports whether it was violated.
- **A separate final importance-sampling stage for improved cross
  entropy.** The stopping sample is independent of the proposal that
  generated it. The decision to stop does depend on that sample, which
  leaves a small bias for small `N` (see the audit addendum); a fresh
  final sample would remove it at the cost of `N` more calls.

## Verification

`tests/test_rare_event_extensions.py` (33 tests, about 12 s) checks the
scale search (attained c.o.v., decreasing scales, end points), weighted
moments against repeated points, weighted EM on two clusters, improved
cross entropy on linear states (`beta` 3 and 4.5, `d` 2 and 20), the
parallel system against `bivariate_normal_cdf`, the four-branch function
(mixture within four standard errors on five seeds, single Gaussian
unconverged with a larger c.o.v. at more than twice the cost), reported
against replicate c.o.v., the unguarded-covariance flag, the invariance of
the smooth-target chain (mean from Stein's identity), the intermediate
constants of sequential importance sampling against
`Phi(-beta / sqrt(1 + sigma^2))` (which fixes the level-0 factor), its
estimates on all six cases, cost accounting, the BUS limit state against
the acceptance event, `c Z` by crude Monte Carlo, the conjugate posterior
and evidence, exact agreement with `subset_simulation`, the constant check,
and input validation.

## References

[REF-PAPAIOANNOU-GEYER-STRAUB-2019],
[REF-PAPAIOANNOU-PAPADIMITRIOU-STRAUB-2016], [REF-STRAUB-PAPAIOANNOU-2015],
[REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015], [REF-AU-BECK-2001],
[REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-DEMPSTER-LAIRD-RUBIN-1977],
[REF-ARTHUR-VASSILVITSKII-2007], [REF-GEYER-1992].
