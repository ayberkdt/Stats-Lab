# ADR-0054: GP hyperparameter regularization and a CE-IS coverage check

- Status: Accepted (proposed during the 2026-10-12 session)
- Date: 2026-10-12

## Context

The benchmark suite (ADR-0049) left two open issues:

1. `surrogates.fit_gaussian_process` maximizes the likelihood over the box
   `|theta| <= 25` only. On smooth limit states it ends at near-singular
   hyperparameters. On the parabola `G = 3 + u1^2/4 - u2` the length scales
   were 1e3-2e5, signal/noise about 1e18, and the predictive variance
   cancelled to exactly 0 off the design. `ak_mcs` then stopped right after
   its 12-point design with `min U` = 114-2320 (10 seeds, N = 2000), and on
   the linear benchmark with `min U` = 1.7e4-2.6e4. Further investigation
   found the opposite failure on the four-branch function: a white-noise
   optimum (length scales 0.0017 for a smallest design gap of 0.33). The
   likelihood is flat there, so warm-started refits never left it.
2. `cross_entropy_importance_sampling` returned 1.19e-3 for an exact
   4.46e-3 on the four-branch system (|z| about 17, 0/2 coverage in the
   quick benchmark) with no warning. A single Gaussian proposal follows one
   of the four failure regions, and the weight diagnostics of ADR-0041 cannot
   see mass the proposal never sampled.

## Decision

1. **`GaussianProcessRegularization`** (frozen dataclass, opt-in through
   `fit_gaussian_process(..., regularization=)` and
   `fit_gaussian_process_to_model`) [EQ-SUR-GP-006]:
   - `log ell_i in [log(a h_i), log(b E_i)]`, where `h_i` is the smallest
     positive gap between coordinates of input `i` and `E_i` its extent.
     Defaults: `a = 0.1` (the task suggested "min spacing / 10") and
     `b = 20` (within the suggested 10-100 x extent).
   - `sigma_f^2 in [1e-3, 1e3]` in standardized output units.
   - A minimum relative nugget `tau = 1e-8`: the diagonal is
     `sigma_f^2 (1 + tau) + nugget [+ sigma_n^2]`, so the condition number of
     the correlation part is at most `1 + n / tau` and signal/noise is at most
     `1 / tau`. This is a fraction of the signal, not an automatic increase
     after a failed Cholesky factorization (still rejected, ADR-0019).
   - An optional log-normal prior `log ell_i ~ N(log E_i, s^2)` (MAP fit).
     It is off by default: the bounds and `tau` alone fixed every observed
     failure, and a prior adds a scale choice that users would have to
     justify.
   - The box is searched with `theta = l + (u - l) sin^2(phi / 2)`. An
     optimum on a bound is then the stationary point `phi = 0` or `pi` and
     BFGS converges at its usual rate. A logistic map was tried first: it
     sends `phi` to infinity with a vanishing gradient and took 200
     iterations and 5454 evaluations (5.8 s) on one start. On this path the
     BFGS gradient tolerance is `1e-5` instead of `1e-6`: on one at-bound
     start rounding kept the gradient norm at 2.8e-6 for 200 iterations.
   - Every fit (regularized or not) reports `log_parameter_bounds` and
     `at_bound` (parameters within 0.01 log units of a bound, e.g.
     `"length_scale[1]:upper"`). The new
     `standardized_noise_variance` property gives the diagonal exactly as
     factorized; `ak_mcs`'s incremental posterior uses it.
   - New public names: `GaussianProcessRegularization`,
     `gaussian_process_log_parameter_bounds`; new keyword
     `relative_nugget` of `gaussian_process_log_marginal_likelihood`
     (default 0). New dataclass fields have defaults, so positional
     construction is unchanged.
2. **Defaults.** `fit_gaussian_process` keeps `regularization=None`: the
   unregularized fit, bit-for-bit unchanged (the legacy code path is
   separate, and a test checks that the default equals `regularization=None`).
   A trial with the regularized default changed 2 of 79 tests in the GP,
   conformal, AK-MCS, visualization and workflow test files. Both are
   documented behaviours that would change meaning, not accuracy failures:
   exact interpolation (variance at training points 2.9e-8 > 1e-8, from
   `tau`) and the conformal "vanishing variance needs an offset" path, which
   no longer occurs. Regression users also get a good predictive *mean* from
   the near-singular optimum (parabola: 1.070 vs true 1.0625). The defect only
   bites where the *variance* drives decisions. So regularization is opt-in
   for regression, and the default of `reliability.ak_mcs`
   (`regularization=AKMCS_GP_REGULARIZATION`) where it does. `None` restores
   the old AK-MCS.
3. **CE-IS coverage check** [EQ-REL-CE-004]. The first adaptation level is
   drawn from the starting density before any adaptation. The share `f` of
   its elite points with `u . mu / ||mu|| <= 0` (behind the origin as seen
   from the final proposal mean) is reported as `uncovered_elite_fraction`.
   When `f > CE_MAX_UNCOVERED_ELITE_FRACTION = 0.1`, a `UserWarning` names
   `find_design_points` + `mixture_importance_sampling` and
   `subset_simulation`, and `multiple_failure_regions_suspected` is set. The
   check is skipped (`nan`) when `||mu|| < CE_MIN_MEAN_NORM = 1`. It costs no
   model calls. Alternatives considered: comparing the final failure-sample
   spread with the proposal covariance (the failure samples come from the
   proposal itself, so they cannot reveal unsampled regions), and a
   multi-start or mixture refit (extra model calls on every run).

## Evidence (all numbers from runs in this session)

AK-MCS, quick-benchmark configuration (N = 2000, `target_cov = 1`):

| problem, seeds | fit | calls (total) | misclassified (total) | `min U` range |
|---|---|---|---|---|
| parabola, 1-10 | unregularized | 120 | 0 | 114-2320 |
| parabola, 1-10 | regularized | 124 | 0 | 2.23-9.99 |
| four-branch, 1-10 | unregularized | 552 | 19 (seeds 5, 6, 9) | 2.08-2.92 |
| four-branch, 1-10 | regularized | 264 | 0 | 2.15-3.03 |

On four-branch seed 5 the unregularized run made 198 additions (50 s), gave
`P_f = 1.0e-3` against 4.5e-3 on the same population, and misclassified 7
points. Seed 9 made 128 additions and predicted no failure (11 misclassified,
not converged). Regularized: 29 and 30 calls, 0 misclassified. Parabola with
N = 10^4 growing to 5 x 10^4 (`target_cov = 0.1`, seeds 1-5): unregularized
runs added no point and misclassified 1, 0, 1, 0, 2 points with `min U` up to
639. Regularized runs added 3-7 points and misclassified none. In the lab
configuration (four-branch, N = 10^4, seeds 1-5) seeds 1-4 are identical.
Seed 5 needs 41 calls instead of 48 and misclassifies 1 point instead of 0
(`min U = 2.19`). With `squared_exponential`, seed 1 went from 212 calls / 24
misclassified to 34 / 0, and seed 2 from 37 / 0 to 35 / 3.

CE-IS, mean-shift default, N = 500, seeds 1-10. On four-branch, 7 runs
converged on one region (estimates 0.95e-3 to 1.76e-3, z = -7 to -32) and all
7 warn with `f` = 0.48-0.58. In the other 3 the mean stalled near the origin,
the check was skipped, and the failure-ESS warning fired. With the
benchmark's seed streams all 10 warn. On linear-2d, linear-10d, the parabola,
and lognormal, `f` = 0.00 in 39/40 runs and 0.02 in one, with no warning.
With `adapt_covariance=True` the four-branch proposal stays wide around the
origin, is unbiased (|z| <= 1.2), and the check is skipped.

Quick benchmark before/after: every estimate, error, |z|, coverage and
evaluation count is unchanged. The CE-IS four-branch row now carries the
coverage warning. AK-MCS notes now show `min U`: linear-2d 5.09/12.8
(unregularized 1.7e4/2.6e4), parabola 3.2/7.8 (1.2e3/2.3e3). The parabolic
AK-MCS row still shows |z| > 1000 and 1/2 coverage. Seed 2's population of
2000 contains no failure point (same-population crude MC is also 0), so that
row is limited by population size, not by the surrogate.

## Consequences

- AK-MCS results change for runs whose unregularized fit was near-singular
  or white-noise. The ADR-0049 regression test of the incremental posterior
  now passes `regularization=None` to keep testing the near-singular case.
- The bound factors and `tau` are computational choices, documented as such.
  `[REF-ANDRIANAKIS-CHALLENOR-2012]` motivates a nugget that controls
  conditioning and interpolation. It does not supply these values.
- Limitations: the lower bound uses the smallest gap, which shrinks as AK-MCS
  clusters points near the limit state, so it guards mainly the first fit.
  Warm starts then keep the length scales sensible, but a later fit could in
  principle reach a short-scale optimum. The CE check needs a first-level
  elite point in each region, uses the mean direction only (not the
  covariance), and is weakened by `initial_mean`. A stalled CE run with
  `||mu|| < 1` is caught only by the ESS warning. Non-convergence after
  `max_levels` is still reported only through `converged`.
