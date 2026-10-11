# ADR-0042 — Proper scoring rules, forecast calibration, and shift tests

- Status: Accepted
- Date: 2026-10-12

## Context

The roadmap item "Conformal prediction, distribution shift, uncertainty
calibration" had only its conformal part. The library had no CRPS, no
interval or quantile scores, no PIT histogram for distribution objects or
ensembles, no Brier decomposition, and no two-sample shift test. The
one-sample KS test (`inference.goodness_of_fit`), the Clopper-Pearson and
Wilson intervals (`inference.confidence`), Pearson's chi-square test
(`inference.hypothesis_tests`), and the area metric and u-pooling
(`validation.metrics`) already existed. `conformal` is at layer 6 and
`validation` at layer 5.

## Decision

1. **Three new modules in `validation` (layer 5).**
   - `scoring.py` holds the proper scores.
   - `calibration.py` holds PIT, histogram, coverage, reliability, and
     sharpness diagnostics.
   - `shift.py` holds energy distance, MMD, and Wasserstein distances.
   - The modules import only layers 0 to 2: `foundations`, `distributions`,
     `statistics`, and `inference`. They could also sit at layer 2.
   - They belong to `validation` because they compare predictions with
     observations, which is that package's job. `validation` already hosts
     PIT-based u-pooling and the area metric. Placing them at layer 5
     still lets `conformal` (layer 6), `workflow`, and `timeseries` use
     them.
2. **The two-sample KS test goes into `inference.goodness_of_fit`.**
   It sits next to the one-sample test and reuses its limiting-law tail.
   - The exact p-value counts lattice paths with exact integers. It
     compares only at ends of tied blocks, so with ties it is exact for the
     permutation distribution conditional on the tie pattern.
   - The statistic is kept as the integer `max |i n - j m|`, so the "at
     least as extreme" comparison needs no floating-point tolerance.
   - Exact counting is used up to `m n <= 250000` (about 0.1 s at
     300 x 400). Beyond that, the limiting law is used without a
     small-sample correction.
3. **CRPS for distribution objects: closed forms plus a generic fallback.**
   - `crps_distribution` uses closed forms for five families. Each is
     checked against Simpson integration of the definition and against the
     fallback.
   - The fallback integrates the quantile-score representation, split at
     `F(y)`, with a private tanh-sinh rule (`validation._quadrature`).
     Quantile integrands are unbounded at `u -> 0, 1`. The existing
     `numerics` rules (trapezoid, Simpson) assume a finite smooth integrand
     and would lose accuracy there.
   - The integrand receives `1 - u` computed without cancellation, so
     upper-tail quantiles come from `isf`.
4. **Fair ensemble scores are opt-in (`fair=True`).** The default is the
   ordinary empirical-CDF CRPS, which is what `crps_ensemble` literally
   computes. The fair version is what users usually want when comparing
   ensembles of different sizes. The tests check its unbiasedness and the
   size of the ordinary score's bias.
5. **The Brier decomposition always returns the two within-bin terms.**
   - The classical `REL - RES + UNC` identity is exact only when each bin
     holds one forecast value.
   - With the Stephenson-Coelho-Jolliffe terms, `identity_residual` is
     rounding-level for any binning.
   - The default `bins=None` groups by distinct forecast value, where the
     extra terms vanish.
6. **Ensemble PIT is randomized over the rank (`(r + V(s+1))/(m+1)`).**
   It is exactly uniform under exchangeability, ties included, so the same
   chi-square and KS uniformity checks apply to ensembles and to
   distribution objects.
7. **Permutation tests compute one block sum per relabeling.**
   - With off-diagonal row sums, the cross and second-sample sums follow
     from the first-sample block. Each relabeling then costs `O(m^2)`
     instead of `O(N^2)`.
   - For 1-D energy tests, the sorted pooled sample gives `O(N)` per
     relabeling.
   - The p-value is `(1 + b)/(1 + B)`, as in `inference.permutation_test`.
   - The MMD bandwidth is computed once from the pooled sample, which is
     invariant under relabeling, so the test keeps its exact size.
8. **Light integration with the existing assessment.**
   - `ExperimentAssessment` gains a computed `crps` property (no new
     constructor field, so `workflow` and all existing call sites are
     unchanged).
   - `ValidationAssessment.summary()` prints it per experiment.
   - The verdict criteria are unchanged. A CRPS has no
     dimensionless pass/fail threshold, so adding it to the verdict would
     have meant inventing one.

## Consequences

- Users can score ensembles, quantile forecasts, distribution objects, and
  event probabilities with proper scores, and diagnose calibration and
  sharpness from one set of PIT values.
- Users can test calibration data against deployment data before trusting
  an exchangeability assumption.
- Not provided:
  - randomized PIT for discrete distribution objects (needs `F(y-)`);
  - the Hersbach reliability/potential decomposition of the CRPS;
  - k-sample Anderson-Darling;
  - MMD bandwidth selection or other kernels;
  - multivariate Wasserstein distances;
  - full Bröcker-Smith resampling of forecast values in the reliability
    bands.
- Independence of cases is assumed by every p-value and binomial interval.
  The foundation document states this.
- New references (all checked online for authors, venue, volume, and pages):
  `REF-GNEITING-RAFTERY-2007`, `REF-MATHESON-WINKLER-1976`,
  `REF-HERSBACH-2000`, `REF-FERRO-2014`,
  `REF-GNEITING-RAFTERY-WESTVELD-GOLDMAN-2005`, `REF-BARAN-LERCH-2015`,
  `REF-JORDAN-KRUGER-LERCH-2019`, `REF-DAWID-SEBASTIANI-1999`,
  `REF-GNEITING-STANBERRY-ETAL-2008`, `REF-SCHEUERER-HAMILL-2015`,
  `REF-BRACHER-RAY-GNEITING-REICH-2021`, `REF-MURPHY-1973`,
  `REF-STEPHENSON-COELHO-JOLLIFFE-2008`, `REF-BROCKER-SMITH-2007`,
  `REF-GNEITING-BALABDAOUI-RAFTERY-2007`, `REF-HAMILL-2001`,
  `REF-CZADO-GNEITING-HELD-2009`, `REF-SZEKELY-RIZZO-2013`,
  `REF-GRETTON-ETAL-2012`, `REF-PANARETOS-ZEMEL-2019`, `REF-HODGES-1958`,
  `REF-TAKAHASI-MORI-1974`.
- New equations: `EQ-VAL-SCORE-*`, `EQ-VAL-CAL-*`, `EQ-VAL-SHIFT-*`,
  `EQ-VAL-QUAD-DE-001`, and `EQ-INF-GOF-KS2-*`.
- The closed-form CRPS expressions for the log-normal, logistic,
  exponential, and uniform laws were checked numerically, not against the
  printed papers.
- `README.md`, `CHANGELOG.md`, `memory/ROADMAP.md`, and
  `docs/architecture.md` are left for the integrating change.
