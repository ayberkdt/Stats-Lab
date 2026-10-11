# ADR-0021 — Classical and resampling inference

- Status: Accepted
- Date: 2026-10-11
- Scope: ROADMAP Phase 4 items "Confidence intervals, bootstrap, permutation
  methods" and "Hypothesis tests, power, effect sizes, multiple testing,
  equivalence tests"

## Context

`probstat.inference` had point estimators, KS/AD statistics, Wilson and
z intervals, Bayesian updating, and MCMC, but no frequentist tests,
resampling intervals, or power analysis. The estimated-sigma branch of
`normal_mean_confidence_interval` was openly labelled a large-sample
approximation. The library has no runtime dependencies, so every reference
distribution has to come from the existing special functions.

## Decision

1. Everything lives in `probstat.inference` (layer 2). New modules:
   - `sampling_distributions`: t, chi-square, F, and noncentral-t tails;
   - `hypothesis_tests`: t tests, TOST, chi-square, Fisher, binomial, ANOVA;
   - `rank_tests`: Mann-Whitney, Wilcoxon;
   - `resampling`: bootstrap, jackknife, intervals, permutation;
   - `power`: effect sizes, power, sample size;
   - `multiple_testing`;
   - private `_hypothesis`: alternatives, p-value tails, cached t critical
     values.
   They import only `foundations`, `statistics`, `optimization`,
   `distributions._special`, and sibling inference modules. That is the
   existing pattern in `bayesian.py`, and the graph stays acyclic.
2. Upper tails are evaluated directly. They come from `Q(a, x)` and `I_x(a, b)`,
   never as `1 - CDF`, so tiny p-values keep relative accuracy.
3. t-test power uses the **exact** noncentral t law. It is computed by
   Simpson quadrature of `E[Phi(t S - delta)]` over the chi scale variable.
   It is not a normal approximation and not a simulation, and it is accurate
   to about `1e-11` against an independent reference. Sample size is the
   smallest `n` whose exact power meets the target, found by unit steps from
   the closed-form z start.
4. Results are frozen dataclasses with statistic, p-value, df, alternative,
   method label, effect size, and an interval where relevant. Docstrings
   state that a p-value is a null-tail probability, not the probability that
   the null is true. t-test results always carry a **two-sided** interval at
   the requested confidence, even for one-sided alternatives. This keeps
   `ConfidenceInterval` finite and avoids silently changing its meaning.
5. Exactness policy:
   - Rank tests use generating-function exact laws when there are no ties
     and `max(m, n) <= 50` (`method="auto"`). Otherwise they use a
     tie-corrected normal approximation with continuity correction.
     Requesting `method="exact"` with ties raises an error.
   - The permutation test enumerates all relabelings up to `max_exact`.
     Beyond that it uses Monte Carlo with `(b + 1)/(M + 1)`.
   - Two-sided binomial and Fisher p-values use the minimum-likelihood rule
     with relative tolerance `1e-7`.
6. Chi-square tests warn with `SmallExpectedCountWarning` (Cochran's rule,
   expected `< 5`). They also record `small_expected_counts`, and they do not
   switch to an exact test silently.
7. All resampling uses `RandomStream` children (`"bootstrap"`,
   `"permutation"`), so one integer seed reproduces every replicate across
   interval methods. BCa computes `z_0` with ties counted as one half and
   takes its acceleration from the jackknife. It raises an error rather than
   returning an interval when `z_0` is infinite or `a` is undefined.
8. The old `normal_mean_confidence_interval` is unchanged. The exact
   interval is the new `student_t_mean_confidence_interval`. The library also
   gains `clopper_pearson_interval`.

## Consequences

ATHENS-2.12 (confidence intervals and sampling concepts) now has these exact
or resampling interval procedures, each with coverage evidence:

- z intervals;
- t intervals;
- Wilson intervals;
- Clopper-Pearson intervals;
- percentile, basic, normal, studentized, and BCa bootstrap intervals.

Tests check the results against several sources:

- published values (sleep data, Mendel, tea tasting);
- hand formulas;
- brute-force enumeration oracles;
- seeded simulations.

Pure-Python enumeration bounds the exact branches. Power covers only mean
tests with equal group sizes.

## Rejected alternatives

- **Normal approximation of t power.** It is inaccurate for small `n`, which
  is exactly where power planning matters.
- **Simulated power.** It is noisy, slow, and needs a seed in a design
  function.
- **Silent fallback from chi-square to Fisher** for small expected counts.
  It hides a change of method. The library warns instead and leaves the
  choice to the user.
- **A separate `testing` package.** Tests, intervals, and estimators share
  validation and reference distributions. The architecture already assigns
  "tests" to `inference`.
