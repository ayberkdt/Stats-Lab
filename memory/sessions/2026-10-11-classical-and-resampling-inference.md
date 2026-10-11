# Session — 2026-10-11 — Classical and resampling inference (Phase 4)

## Delivered

- `probstat.inference` (layer 2) gains these modules:
  - `sampling_distributions`: t, chi-square, F, and exact noncentral-t
    tails.
  - `hypothesis_tests`: one-sample, paired, pooled, and Welch t tests,
    TOST, chi-square goodness of fit and independence (with a small
    expected-count warning), Fisher exact, the exact binomial test with a
    Clopper-Pearson interval, and one-way ANOVA.
  - `rank_tests`: Mann-Whitney U and Wilcoxon signed-rank, each with an
    exact branch and a tie-corrected normal branch.
  - `resampling`: bootstrap, jackknife, five bootstrap interval methods
    including BCa, and an exact or Monte Carlo permutation test.
  - `power`: Cohen's d, Hedges' g, z and t power, and required sample size.
  - `multiple_testing`: Bonferroni, Holm, BH, and BY.
- `confidence.py` gains `student_t_mean_confidence_interval` and
  `clopper_pearson_interval`. The old z interval is unchanged.
- Supporting records:
  - 29 equation records and 21 bibliography entries;
  - the guide `docs/foundations/hypothesis-testing-and-bootstrap.md`;
  - ADR-0021;
  - ATHENS-2.12 marked `Implemented`.

## Verification

All of the following come from `tests/test_classical_resampling_inference.py`
(36 tests).

- **Sleep data.** Welch `t = -1.8608`, `df = 17.776`, `p = 0.07939`; paired
  `t = -4.0621`, `p = 0.002833`. These match R's documented output.
- **Count tests.** Mendel `X^2 = 0.470`, party table `X^2 = 30.07`, Fisher
  tea `34/70`.
- **Exact oracles.** Binomial, Mann-Whitney, Wilcoxon, and permutation
  p-values equal brute-force enumeration.
- **Power.** `t_test_power(0.5, 64, design="two-sample") = 0.8014596`, the
  same as R's `power.t.test`. Required sizes are 64, 34, and 63 (z).
- **Coverage.** At `n = 6` with seed 97, the t interval covers 0.953 and the
  z-with-`s` interval covers 0.891.
- **Full suite.** `pytest` 338 passed; `ruff` and `mypy --strict` are clean.

## Next safe step

Two Phase 4 items remain open:

- likelihood and Fisher-information tooling;
- regression-based tests (F tests for nested linear models), which can
  reuse `f_sf`.
