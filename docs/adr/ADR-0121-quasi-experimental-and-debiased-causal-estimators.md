# ADR-0121 — Quasi-experimental and debiased causal estimators

- Status: Accepted
- Date: 2026-11-21
- Extends: the causal-inference package (`causal.effects`,
  `causal.sensitivity`)

## Context

`probstat.causal` estimated the average effect of a binary treatment under
conditional exchangeability with linear outcome models and a logistic
propensity score, and measured sensitivity to unmeasured confounding only
on the risk-ratio scale (E-values). Its guide listed "flexible learners and
cross-fitting are not provided" as a limitation. Four common situations
were not covered:

- the confounding function is nonlinear and has to be learned, which
  biases plug-in estimators unless the score is orthogonal and the
  nuisances are cross-fitted;
- the analysis is a linear regression and the question is how strong an
  omitted confounder would have to be, in units a regression user can
  judge (partial R-squared);
- identification comes from the design rather than from measured
  covariates: a policy adopted by some units at different times
  (difference-in-differences), by a single unit (synthetic control), or an
  instrument;
- under staggered adoption the usual two-way fixed-effects regression does
  not estimate an average of the group-time effects.

## Decision

1. **Five new modules in `causal`** (layer 3), importing only
   `foundations`, `linalg`, `distributions` and `inference`:
   `omitted_variable_bias.py`, `double_ml.py`,
   `difference_in_differences.py`, `synthetic_control.py`,
   `instrumental.py`. The layer map is unchanged.
2. **Omitted-variable bias** ([EQ-CAUSAL-OVB-001] to
   [EQ-CAUSAL-OVB-005]). `partial_r2`, `omitted_variable_bias`,
   `robustness_value`, `sensitivity_statistics`, `adjust_for_confounder`.
   Every formula was re-derived from the least-squares algebra before use:
   `bias^2 = gamma^2 delta^2`, `RSS_long = RSS_short (1 - R2_Y)` and
   `SS(D | X, Z) = SS(D | X)(1 - R2_D)` give the bias, the adjusted
   standard error (with the factor `sqrt(df / (df - 1))`) and the adjusted
   t. The robustness value for significance is the smallest bound on both
   partial R-squared values that can bring the adjusted t to the critical
   value. For `f_q <= 1 / f*` (`f_q = q |t| / sqrt(df)`,
   `f* = t* / sqrt(df - 1)`) this is the equal-strength root of the same
   quadratic with `f_{q,alpha} = f_q - f*`; for `f_q > 1 / f*` it is
   `(f_q^2 - f*^2) / (1 + f_q^2)` (added by the audit, see below). The sign
   of the bias is a user assumption (`reduce`).
3. **Double ML** ([EQ-CAUSAL-DML-001] to [EQ-CAUSAL-DML-003]).
   `dml_partially_linear` implements the partialling-out score with pooled
   residuals (one estimate from all folds) and the sandwich variance. A
   learner is a plain callable `fit(rows, targets) -> predict`; the
   built-ins wrap `fit_linear_model`, `fit_ridge`, `fit_glm` and a locally
   written exhaustive k-nearest-neighbour average ([EQ-CAUSAL-KNN-001]).
   Folds are a seeded random permutation dealt round robin (child stream
   `"folds"`). `folds=1` means no cross-fitting and is reported in
   `cross_fitted`. `dml_average_treatment_effect` applies the same
   cross-fitting to the AIPW score and returns the existing
   `EffectEstimate`; it does not call `estimate_aipw`, which fits its own
   parametric models, but equals it for `folds=1` with the default
   learners. Predicted propensities are clipped to `[clip, 1 - clip]` with
   a `PositivityWarning`.
4. **Difference-in-differences** ([EQ-CAUSAL-DID-001] to
   [EQ-CAUSAL-DID-005]). `did_two_by_two` works on unit-level changes.
   `group_time_effects` takes a unit-by-period matrix (which enforces
   balance) and per-unit adoption periods, and estimates the unconditional
   `ATT(g, t)`, `t >= g`, with base period `g - 1` and never-treated or
   not-yet-treated comparison units. Standard errors come from influence
   functions kept on the result, so that `event_study()` and `overall()`
   can add the delta-method term for the estimated group-size weights.
   The "overall" aggregate is defined as the group-size-weighted mean of
   all estimated post-treatment cells and documented as such. Cells
   without comparison units are listed in `skipped_cells`.
5. **Synthetic control** ([EQ-CAUSAL-SC-001] to [EQ-CAUSAL-SC-004]).
   The simplex-constrained quadratic program is solved by projected
   gradient with the exact sort-based simplex projection and step
   `1 / (2 lambda_max(X0'VX0))`; every 20 iterations the quadratic is
   minimized exactly on the current support (a KKT linear system) and the
   result kept if it is feasible and lowers the duality gap. Since the
   audit the iteration runs on donor predictors centred at the treated
   unit (an equivalent problem with a smaller `lambda_max`). The stopping
   rule is the duality gap `g'w - min_j g_j <= tol * s`, `s` the largest
   of `x1'Vx1` and the `X0_j'VX0_j`, which is a
   certificate (`f(w) - f* <= gap`), and the result reports it together
   with `converged` and `iterations`. `V` is user-supplied. Placebo
   inference ranks the post/pre RMSPE ratio; by default the treated unit
   joins the donor pool of each placebo run (`treated_in_placebo_pools`).
6. **Instrumental variables** ([EQ-CAUSAL-IV-001] to
   [EQ-CAUSAL-IV-003]). `two_stage_least_squares` with HC0/HC1 sandwich
   covariance and first-stage F statistics. The textbook source is the
   already registered [REF-HERNAN-ROBINS-2020, Chapter 16]; the sandwich
   is White's estimator applied to the second-stage regressors
   ([REF-WHITE-1980]), verified against the closed form of the exactly
   identified case. No new citation was added for this method.

## Consequences

Measured in `tests/test_causal_extensions.py` (seeds fixed):

- **Omitted-variable bias.** With a simulated confounder (`n = 120`) the
  formula bias and the observed change of the coefficient are both
  0.3064467 and agree to `2e-16`; the adjusted standard error and t
  statistic agree with the long regression to `4e-15`. A confounder with
  both partial R-squared equal to `RV_1 = 0.684` moves the estimate to
  `-2e-16`; one at `RV_{1,0.05} = 0.629` leaves `t = 1.98081`, the
  critical value on 115 degrees of freedom.
- **Double ML.** `folds=1` with linear learners equals the least-squares
  coefficient and its HC0 standard error to `1e-12`. Over 120 replicates
  (`n = 300`, nonlinear nuisances, 15 nearest neighbours, 4 folds, truth
  0.5): mean estimate 0.515, Monte Carlo SD 0.062, mean standard error
  0.057, coverage of the 95 % interval 0.917; the estimator that does not
  residualize the treatment averages 0.253. The 0.015 bias is the
  smoothing bias of the nearest-neighbour learner; the interval is
  asymptotic and does not account for it.
- **Difference-in-differences.** In a noise-free staggered panel every
  `ATT(g, t)` is exact to `1e-12`, the overall average is 3.0 and the
  static two-way fixed-effects coefficient is 1.875. Over 150 replicates
  with random group membership the ratio of the mean influence-function
  standard error to the Monte Carlo SD is 1.09 for a cell, 1.03 for the
  exposure-1 event-study average and 0.96 for the overall average.
- **Synthetic control.** A duplicated donor and an exact convex
  combination are recovered to `1e-7` in the weights; random problems with
  8 donors and 6 predictors converge in 20 to 60 iterations with a duality
  gap below `3e-15`; a problem with 20 donors and 5 predictors (non-unique
  weights) needs 1200 iterations. One iteration on a 10-donor problem is
  reported as `converged = False` with gap 1.69.
- **Instrumental variables.** Coefficients and robust covariance match the
  exactly identified closed forms to `1e-9`. Over 100 replicates
  (`n = 200`, two instruments, heteroskedastic errors, truth 1.5): mean
  1.487, coverage 0.93, mean robust standard error 0.216 against a Monte
  Carlo SD of 0.212; least squares is above 2.0 in every replicate
  checked.
- The whole test file (45 tests after the audit) runs in about 10 s.
- `EffectEstimate.method` gains the values `"aipw-crossfit"` and
  `"aipw-learners"`.

## Audit addendum (independent re-measurement, 2026-11-21)

Own simulations with other seeds and more replicates; scripts were not
added to the repository.

- **Robustness value for significance.** A grid search over
  `(R2_Y, R2_D)` gives the smallest bound that removes significance. It
  equals the equal-strength root when `f_q <= 1 / f*` and
  `(f_q^2 - f*^2) / (1 + f_q^2)` otherwise; the first implementation
  returned the equal-strength root everywhere and overstated the value in
  the second regime (0.6634 instead of 0.6590 at `t = 6`, `df = 10`;
  0.9831 instead of 0.9639 at `t = 20`, `df = 5`). The rule agrees with the
  `robustness_value` function of the authors' `sensemakr` package, whose
  source was read; the bias, the adjusted standard error with
  `sqrt(df / (df - 1))` and the partial R-squared agree with it as well.
- **Double ML, 1000 replicates** of the same design (`n = 300`, 15
  neighbours, 4 folds): mean 0.5075 (bias 3.8 Monte Carlo standard
  errors), mean standard error 0.92 of the Monte Carlo standard deviation,
  coverage of the 95 % interval 0.929 (binomial standard error 0.008).
  With 5 neighbours coverage is 0.898 (standard-error ratio 0.85), with
  `n = 1000` and 30 neighbours 0.923. With three irrelevant covariates
  added (four in total) the nearest-neighbour bias is 0.127 and coverage
  0.41. `n = 301` (folds of unequal size) behaves like `n = 300`. The
  interval under-covers in every nearest-neighbour configuration tried.
- **Cross-fitted AIPW** (`n = 203`, 5 folds, true effect 1): with the
  default linear and logistic learners coverage was 0.928 (moderate
  confounding) and 0.926 (strong confounding, standard-error ratio 0.82)
  over 500 replicates; with 15-nearest-neighbour learners for outcome and
  propensity it was 0.957 and 0.800 over 300 replicates, with mean
  estimates 0.918 and 0.847 (neighbour averages of the indicator reach 0
  and 1 and are clipped to weights of 100).
- **Group-time effects, 1000 replicates per design**, random group
  membership with shares 0.5, 0.25, 0.15 and 0.10 (never treated), six
  periods, heterogeneous dynamic effects: at `n = 400` the ratio of the
  mean standard error to the Monte Carlo standard deviation is 0.96 to
  0.98 for cells, event-study and overall averages and coverage is 0.93 to
  0.95, so the influence function for the estimated shares is right. At
  `n = 120` (about 12 never treated) the ratio is 0.90 to 1.02 and
  coverage 0.91 to 0.95; with 5 never-treated units among 100 (fixed
  membership) the ratio is 0.85 to 0.94 and coverage 0.86 to 0.91; at
  `n = 60` with shares 0.55, 0.25, 0.12, 0.08 the ratio is 0.71 to 0.83
  and coverage 0.81 to 0.88. A group or comparison set of one unit used to
  report a standard error that left that arm out (0.0 for one treated and
  one comparison unit); it is now `nan`.
- **Synthetic control.** With predictors at a common level of `10^4`
  (30 donors, 12 pre-periods, donor paths within 1 % of the treated path)
  the first implementation stopped at 20000 iterations with
  `converged = False` and an objective of 3329; centred at the treated
  unit the same problem converges in 189 iterations with objective
  `1.2e-6`. Predictors on scales from `10^-3` to `10^4` and a
  Hilbert-matrix-like donor pool still do not converge in 20000
  iterations and are reported as such. A treated unit with all predictors
  zero could never be declared converged (threshold `tol * x1'Vx1 = 0`);
  the threshold now also uses the donors' scale. Under random assignment
  of the intervention among 8 similar units the placebo p-value was at
  most 1/8 in 10.7 % of 150 simulations (nominal 12.5 %).
- **Instrumental variables.** Coefficients, HC0/HC1 covariance, classical
  standard errors and first-stage F agree with matrix formulas for two
  endogenous regressors, three instruments and two exogenous controls,
  with and without intercept, to `1e-13`. Coverage with two endogenous
  regressors (`n = 300`, 600 replicates) is 0.935 and 0.933. With both
  regressors driven by one instrument the per-regressor F statistics
  (median smallest 98) do not flag the weak identification; this is now
  stated in the docstring.

## Rejected alternatives

- **Averaging per-fold estimates (DML1).** The pooled estimator uses one
  denominator and is less sensitive to small folds; only it is provided.
- **Calling `estimate_aipw` inside the cross-fitted estimator.** It has no
  entry point for externally supplied predictions; adding one would have
  changed a stable public function for no gain.
- **The equal-strength robustness value for significance alone.** This
  was the first implementation. It is the formula printed in the paper,
  but for `f_q > 1 / f*` a confounder with a smaller `R2_Y` than `R2_D`,
  both below that value, removes significance; the audit replaced it by
  the minimum over both strengths.
- **Bootstrap standard errors for the group-time effects.** The influence
  function gives the same first-order answer at no extra cost and lets
  aggregates be formed after the fact; a multiplier bootstrap for
  simultaneous bands is left for later.
- **Two-way fixed-effects and "stacked" regressions as estimators.** TWFE
  is shown in the tests only as the thing not to do.
- **A data-driven `V` for synthetic control.** The nested optimization is
  non-convex and its result depends on the optimizer; `V` stays an input.
- **An active-set QP solver.** Projected gradient with an exact support
  solve and a duality-gap certificate is shorter and its convergence is
  reported rather than assumed.
- **Weak-instrument-robust (Anderson-Rubin) intervals and
  over-identification tests.** No registered source covers them.

## Verification

`tests/test_causal_extensions.py` (42 tests): the omitted-variable
identities against long and short regressions fitted with
`fit_linear_model` (both directions of bias), robustness values reaching
their targets, Frisch-Waugh-Lovell for double ML, fold balance and
reproducibility, learner checks by hand, a seeded coverage simulation
against the naive plug-in, the cross-fitted AIPW against `estimate_aipw`,
the four-means difference-in-differences formula, exact recovery of
constructed `ATT(g, t)` under both comparison rules, hand-computed
aggregation weights, the two-way fixed-effects counterexample, Monte Carlo
checks of the influence-function standard errors, simplex-projection KKT
conditions, duplicated-donor and convex-combination recovery, KKT
conditions of the weights, a brute-force grid with three donors, placebo
ranks against manual reruns, the Wald ratio, the exactly identified
closed forms for coefficients and robust covariance, a coverage
simulation with confounding, and input validation throughout. The guide
example in `docs/foundations/causal-inference.md` is executed by
`tests/test_documentation_examples.py`.

## References

[REF-CHERNOZHUKOV-ET-AL-2018], [REF-CALLAWAY-SANTANNA-2021],
[REF-ROTH-ET-AL-2023], [REF-CINELLI-HAZLETT-2020],
[REF-ABADIE-DIAMOND-HAINMUELLER-2010], [REF-HERNAN-ROBINS-2020],
[REF-WHITE-1980], [REF-NOCEDAL-WRIGHT-2006].
