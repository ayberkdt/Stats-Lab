# ADR-0120 — Conformal prediction beyond exchangeability, risk control, and p-values

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0033 (survival and conformal) and the adaptive conformal
  inference module (`conformal.adaptive`)

## Context

`probstat.conformal` covered the exchangeable core: the finite-sample
quantile, split conformal with absolute or normalized scores, conformalized
quantile regression, jackknife+, surrogate calibration, coverage reports,
and adaptive conformal inference for sequences. Its guide listed weighted
conformal and CV+ as not implemented. Six questions that users of those
tools ask next had no answer in the library:

- the test inputs follow a different law from the calibration inputs;
- the data drift, and recent points should count more;
- the quantity to control is an expected loss, not a miss probability;
- the task is to flag outliers among many test points;
- coverage is wanted per subgroup, not on average;
- `n` refits are too many, or a sequence needs a threshold that settles.

Each has a published method with a precise guarantee, and each guarantee is
of a different kind (marginal under a known weight function, marginal minus
an uncomputable gap, expected loss, FDR marginal over calibration,
group-conditional, `1 - 2 alpha` minus a slack, long-run frequency). The
main risk of adding them is blurring those differences.

## Decision

1. **Six new modules in `conformal`** (layer 6), none of which changes an
   existing function:
   `weighted.py`, `risk_control.py`, `pvalues.py`, `grouped.py`, `cross.py`,
   `online.py`. They reuse `quantile.conformal_rank`,
   `quantile.conformal_quantile`, `regression.as_predictor`, the calibration
   validators, and `PredictionInterval`. `pvalues.py` imports
   `inference.adjust_p_values` (layer 2), which the layer map allows.
2. **One weighted quantile for two methods** ([EQ-CONF-006],
   [EQ-CONF-007]). `weighted_conformal_quantile(scores, weights,
   test_weight, alpha)` returns the `1 - alpha` quantile of the weighted
   score law with the test mass at `+inf`. Covariate-shift conformal calls
   it with `w(X_i)` and `w(x)`; fixed-weight nonexchangeable conformal calls
   it with weights in `[0, 1]` and test weight one. Both return a
   `WeightedConformalRegressor`.
3. **Exact rational comparison.** The cumulative weights are summed as the
   binary rationals the floats are and compared with
   `(1 - alpha)(W + w_test)`, `alpha` read from its decimal text as in
   [EQ-CONF-001]. Unit weights therefore reproduce
   `conformal_quantile` exactly, not to rounding. Prefix sums are stored, so
   a test point costs one bisection.
4. **The coverage gap is documented, not estimated.**
   `nonexchangeable_coverage_gap(weights, bounds)` evaluates
   `sum_i w~_i d_i` for user-supplied bounds `d_i` on the total-variation
   distances. Its docstring, the registry record, and the guide state that
   the distances cannot be computed from data.
5. **Conformal risk control on a grid** ([EQ-CONF-008]). The input is a
   loss table `losses[i][j] = L_i(lambda_j)` on a strictly increasing grid.
   The function returns the smallest grid value with
   `(sum_i L_i + B)/(n + 1) <= alpha`, compared exactly. Monotonicity in
   `lambda` and the bound `B` are checked row by row and a violation raises.
   If no grid value qualifies, the largest is returned with
   `feasible=False`.
6. **Conformal p-values** ([EQ-CONF-009]) count ties as `>=`.
   `conformal_outlier_detection` applies the existing Benjamini-Hochberg
   routine ([EQ-CONF-010]) and reports the resolution `1/(n + 1)`.
7. **Mondrian calibration** ([EQ-CONF-011]) stores one `MondrianGroup`
   (size, rank, quantile, guaranteed coverage, scores) per label. A group
   too small for a finite quantile, and a label never seen, both give the
   whole line; `infinite_groups` lists the former.
8. **CV+** ([EQ-CONF-012]) keeps `K` fold predictors and evaluates each once
   per test point. Folds are contiguous balanced blocks by default and
   shuffled from the child stream `"cv-plus-folds"` when a seed is given.
   The docstrings state the guarantee as `1 - 2 alpha - sqrt(2/n)` for
   every `K` (Theorem 4 of the source). An audit confirmed this form from
   excerpts of the paper returned by a web search; the sharper slack in
   `n` and `K` of the same theorem could not be read and is not printed.
9. **Online quantile tracking** ([EQ-CONF-013]) with
   `eta_t = eta t^(-decay)`, `decay` in `[0, 1]`. The reported bound
   ([EQ-CONF-014]) is `(b + eta_1)/(eta_T T)`. It was derived here for
   positive non-increasing steps (invariant interval for `q_t`, then
   summation by parts), and for a fixed step it is the `(b + eta)/(eta T)`
   bound of the quantile tracker. The bound needs a declared score bound
   `b`; a score above it sets `bound_valid=False` and the bound to `inf`.

## Consequences

Measured in `tests/test_conformal_extensions.py` (seeds in the tests):

- Covariate shift `N(0,1) -> N(1,1)`, `n = 200`, `alpha = 0.1`, 300
  replicates: weighted coverage `0.9116` (se `0.0045`); unweighted
  `0.7557` (se `0.0057`).
- Fixed weights `0.97^(60-i)` on i.i.d. data: coverage `0.9165` (se
  `0.0042`). After a noise changepoint, weights `0.9^(100-i)` cover
  `0.962` against `0.693` unweighted.
- Conformal risk control of a false-negative rate, `n = 40`,
  `alpha = 0.1`: mean test risk `0.0697` (se `0.0032`). The miscoverage
  loss returns the split conformal quantile for all 25 `(n, alpha)` pairs
  tried.
- Conformal p-values are exactly uniform over all `(n + 1)!` orderings for
  `n = 1, 3, 5`. Benjamini-Hochberg FDR with 80% inliers: `0.0698` (se
  `0.0042`), target `0.08`.
- Mondrian, two groups of 49: coverage `0.902` and `0.902`; marginal split
  conformal gives `1.000` and `0.822`.
- CV+ with `K = n` equals jackknife+ to the last bit; `K = 5`, `n = 20`
  covers `0.908` (se `0.006`).
- Quantile tracking, 20000 uniform scores: fixed step, miss frequency
  `0.10005` (bound `0.00015`) with `q_t` ranging over `[0.25, 1.40]`;
  `decay = 0.6`, frequency `0.10035` (bound `0.0571`) with `q_t` in
  `[0.878, 0.919]` around the true `0.9`.
- Exact arithmetic costs time: building a weighted regressor is `O(n)`
  rational additions (about 2 ms for `n = 200`).
- The package gains 19 public names.

## Rejected alternatives

- **Estimating the likelihood ratio inside the library.** A fitted ratio
  voids the exact guarantee, and the choice of estimator is a modelling
  decision. The function takes `w` and says what happens if it is wrong.
- **Floating-point comparison with a tolerance for the weighted quantile.**
  It would make the `w = 1` reduction approximate and could move the
  quantile by one order statistic at ties of mass.
- **Enforcing monotone losses by a running maximum.** The source mentions
  it as a device, but applying it silently would change the loss the user
  asked to control.
- **Randomized conformal p-values.** They are exactly uniform with ties but
  make results depend on a seed; the deterministic conservative form is
  enough for FDR control.
- **Raising for an unseen Mondrian group.** The whole line is the valid
  answer with no calibration data, and it is the same rule as a small
  group.
- **Returning a `JackknifePlusRegressor` from CV+.** It would evaluate each
  fold model `n/K` times per test point.
- **Printing the sharper CV+ slack formula, or a stochastic convergence
  statement for decaying steps.** Neither could be checked against the
  source in this environment; only what was derived or verified by an
  oracle is stated.
- **The integrator and scorecaster of conformal PID control.** Outside this
  package's scope; only the quantile tracker is implemented.

## Verification

`tests/test_conformal_extensions.py` (37 tests, about 3 s) checks the
reductions to split conformal and jackknife+, the weighted quantile against
an exact brute-force scan, coverage under covariate shift and for fixed
weights, the risk-control oracle and a false-negative-rate example, exact
uniformity of p-values by enumeration, FDR control, group-conditional
coverage and the infinite-interval rule, CV+ by hand, the telescoping
identity and bound for fixed and decaying steps on random and adversarial
sequences, and input validation for every entry point.

## References

[REF-TIBSHIRANI-ETAL-2019], [REF-BARBER-ETAL-2023],
[REF-ANGELOPOULOS-ETAL-2024], [REF-BATES-ETAL-2023],
[REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-BARBER-ETAL-2021],
[REF-ANGELOPOULOS-CANDES-TIBSHIRANI-2023],
[REF-ANGELOPOULOS-BARBER-BATES-2024], [REF-BENJAMINI-HOCHBERG-1995].
