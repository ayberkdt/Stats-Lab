# ADR-0033 — Survival analysis and conformal prediction

- Status: Accepted
- Date: 2026-10-12

## Context

The roadmap lists survival analysis with censored data (Phase 5) and
conformal prediction (Phase 6) as open items. The library is pure Python with
no runtime dependencies. Seeds go through `RandomStream`, and the layer map
is enforced by `tests/test_architecture.py`.

- Survival estimators need only the risk-set counts, the chi-square and
  normal tails in `inference.sampling_distributions`, and the BFGS solver in
  `optimization`.
- Conformal prediction wraps *any* predictor. That includes the layer-5
  surrogates and the layer-3 `ProbabilisticModel` used for calibration runs.

## Decision

1. **Survival is a subpackage `inference.survival` (layer 2).** It has four
   modules: `data`, `nonparametric`, `parametric`, and `cox`. It imports only
   layers 0-1 plus sibling modules in `inference`, through relative imports.
   It is *not* re-exported from `probstat.inference` to keep that namespace
   from growing by 17 names; users import `probstat.inference.survival`.
2. **One censored-data record.** `SurvivalData(times, events)` accepts
   booleans or 0/1 indicators. `risk_table()` applies the standard tie
   convention: a subject censored at `t` is at risk for events at `t`. Every
   estimator consumes this table, so the tie rule lives in one place.
3. **Kaplan-Meier conventions.**
   - Greenwood variance with log-log pointwise intervals.
   - When `S` reaches 0, the variance is 0 and the interval is `(0, 0)`.
   - Quantile intervals invert the pointwise bands and are `inf` when a band
     never reaches `1 - p`.
   - The RMST is rejected for `tau` beyond follow-up while `S > 0`.
4. **Parametric models in AFT form only.** The model is
   `log T = x' beta + sigma W`. `W` follows the extreme-value law for the
   Weibull (and the exponential, with `sigma = 1`) or the normal law for the
   lognormal. BFGS works on `(beta, log sigma)` with an analytic gradient.
   Standard errors come from a central-difference Jacobian of that gradient,
   symmetrized with tolerance `1e-5`.
   - Proportional-hazards Weibull coefficients are not reported. They are
     `-beta / sigma`, and the documentation says so.
   - Gamma and log-logistic AFT families are left out.
5. **Cox by Newton-Raphson.** The iteration starts at `beta = 0` and halves
   the step until the likelihood does not decrease.
   - Breslow and Efron ties share one accumulation loop. Efron is the
     default.
   - Risk-set sums are accumulated from the largest time down: `O(n p^2)`
     per evaluation.
   - A monotone likelihood is reported as `converged=False`, not as an
     exception.
   - The Breslow baseline hazard is included so that `survival(t, x)` works.
6. **Conformal prediction is a new package `probstat.conformal` at layer 6.**
   It imports `surrogates` (5), `modeling` (3), `inference` (2),
   `foundations`, and `linalg`. Nothing imports it.
7. **Exact conformal index.** `k = ceil((1 - alpha)(n + 1))` is computed from
   `Fraction(str(alpha))`, so decimal levels such as `0.3` cannot be pushed
   across an integer by binary rounding. `k = n + 1` gives `+inf`, not an
   error.
8. **Exchangeability is protected by the API.**
   - `conformalize_surrogate` draws calibration inputs by plain Monte Carlo
     only, never LHS or Sobol, from its own stream child
     `"conformal-calibration"`.
   - Normalized GP scores need an explicit `scale_offset` whenever the GP
     standard deviation vanishes. The library never silently floors the
     denominator, because a data-chosen floor would break the guarantee and
     a tiny absolute floor gave intervals of width `~1e7` in testing.

## Verification

- The Kaplan-Meier, Greenwood, Nelson-Aalen, and RMST values are recomputed
  by hand formulas in the tests.
- Log-rank statistics are checked against explicit hypergeometric loops,
  with 2 and 3 groups.
- The Cox fit is checked against an independent scalar Newton iteration. The
  Efron term is recomputed term by term. The Cox score test equals the
  log-rank statistic on untied data.
- A seeded censored Weibull is recovered within 3 SE; the actual errors are
  within 1 SE.
- Split conformal, normalized, and CQR mean coverage over 500 seeded
  replicates lie within 1.3 standard errors of `k/(n+1) = 10/11`. They are
  about 20 standard errors above the naive-index value `9/11`.

## Consequences

- Conformal intervals give marginal coverage only. The guide states the
  `Beta(k, n + 1 - k)` spread of coverage for a fixed calibration set.
- Citations for the new bibliography entries were written from the author's
  knowledge of the literature. They were not re-read against the publications
  in the session that added them. The section numbers inside Klein and
  Moeschberger are cited at chapter level only for that reason.

## Rejected alternatives

- Putting survival in a new top-level package: it is inference (estimation
  and testing) and needs nothing above layer 1.
- Putting conformal prediction in `inference`: the surrogate and
  probabilistic-model wrappers would be upward imports.
- A full-conformal (refit for every candidate `y`) implementation: it is
  exact but costs one refit per grid value. Jackknife+ covers the no-holdout
  use case.
- Reusing `inference.fit_distribution` for censored fits: its likelihood has
  no survival-function term, and adding one would complicate the generic
  family contract.
