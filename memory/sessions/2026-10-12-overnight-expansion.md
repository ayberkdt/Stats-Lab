# Session 2026-10-12 — Overnight expansion (0.15.0)

## Goal

Continue the integration round: implement the remaining review
recommendations so analyses run from one study object, keep the figures on
the designed theme, and keep developing until 06:00.

## How the work was organised

- Parallel work streams in separate git worktrees, each told to fast-forward
  to `main` first, merged with `--no-ff`; the append-only registry,
  bibliography, and generated API reference were resolved by keeping both
  sides, then checked for duplicate `REF-`/`EQ-` keys.
- Independent numerical reviewers (read-only, numpy/scipy/mpmath oracles)
  after each wave; every confirmed defect got a regression test that fails
  before and passes after the fix.
- A benchmark suite (`benchmarks/`, ADR-0049) acts as a standing reviewer:
  it found the negative-ESS and near-singular AK-MCS defects.

## What changed (ADR-0034 to ADR-0057)

- New capabilities: causal inference, experimental design, extremes and
  information theory, Kennedy-O'Hagan discrepancy, hierarchical models /
  EM / mixtures / predictive checks, adaptive reliability (multiple design
  points, AK-MCS), proper scoring and forecast calibration, optimal transport
  and information geometry, missing data, contingency tables, kernel density,
  adaptive and infinite-range quadrature, limit theorems and estimator
  theory, streaming statistics, a probabilistic-programming interface,
  regularized Gaussian-process fits.
- `UQStudy` stages: `extremes`, `information_sensitivity`, `design`,
  `design_points`, `active_reliability`, `calibrate_with_discrepancy`,
  `propagate_until`, `verify`; `validate` records CRPS, PIT, coverage, and
  verified numerical uncertainty; the HTML
  report draws a figure for every stage.
- Figures for every new capability; the graphite accent moved from orange
  `#E8804A` to indigo `#9BA9F0` after the dataviz validator showed it sat
  9.5 OKLab dE from the amber model hue (floor 15).
- Theme re-checked against sunumatik's `chart-aesthetics.md` at 05:30: dash
  grammar, reference width ratio, marker rims, and band opacity matched
  except nested inner bands (0.28, now capped at 0.18 everywhere);
  Fritsch-Carlson smoothing and a single fading gradient area were added.
- Tooling: generated API reference with a staleness test; API stability
  labels with a deprecation decorator (ADR-0048); benchmark suite in CI.

## Review rounds and what they found

- Rounds 3-6: Cox baseline overflow, ADVI false convergence, near-singular
  copula correlation, GEV profile crash, entropy cancellation, discrete
  divergence underflow, censored EM cancellation, infinite moments reported
  as finite, tanh-sinh truncation (3.7% error on heavy-tailed Wasserstein).
- Rounds 7-10: coverage-key collisions, docstring dedent, contingency
  residual cancellation; `integrate(auto)` returning 3e-198 for 1.77,
  Gauss-Kronrod overflow reported as converged, double-exponential truncation
  ignored, binomial pmf precision (Loader saddle point), ESS improved-estimate
  term; PPL coordinate labels, CE-IS multi-region flag; blind Gauss-Kronrod
  zero reported as converged, discrete laws without `isf`, `run_until`
  converging on an all-zero stream.

## Mistakes worth remembering

- A ruff error was once committed because `ruff -q | tail -1` hid it; use
  `--output-format concise`.
- Test-design errors of mine (zero AK-MCS misclassification assumed;
  population sd where Silverman needs the sample sd; a residual "largest
  cell" test that depended on rounding) were caught by the tests.
- Running `ruff format` on whole directories reformats other agents' files
  and creates merge noise; format only the files you changed.
- Heredocs with `\\n` inside Python string literals are fragile in Git Bash;
  write patch scripts to files or use the Edit tool.
- A disagreement rule that was too strict (`integrate(auto)`) broke an
  executed guide snippet; the guides' snippet test is a useful tripwire.

## Verification evidence

Recorded in each merge commit message. Final state at 05:40:
`1178 passed, 2351 subtests passed`, ruff clean, strict mypy clean on 220
files, all 31 example labs run, benchmark quick mode 188 runs with 0
unexpected failures, UX auditor 0 HATA / 0 UYARI on the complete study
report (no horizontal scroll at 375 px, 85% of the first screen is primary
content). Eleven numerical review rounds in total (changelog numbering); the last
two found mainly convergence-flag honesty issues and a missing minimum
sample size, all fixed. Nothing was pushed.
