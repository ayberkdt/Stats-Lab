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

## What changed (ADR-0034 to ADR-0048 and later)

- New packages and modules: causal inference, experimental design, extremes
  and information theory, Kennedy-O'Hagan discrepancy, hierarchical models /
  EM / mixtures / predictive checks, adaptive reliability (multiple design
  points, AK-MCS), proper scoring and forecast calibration, optimal transport
  and information geometry, missing data, contingency tables, kernel density.
- `UQStudy` stages: `extremes`, `information_sensitivity`, `design`,
  `design_points`, `active_reliability`, `calibrate_with_discrepancy`;
  `validate` records CRPS, PIT, and coverage; the HTML report draws figures
  for every stage.
- Figures for every new capability; the graphite accent moved from orange
  `#E8804A` to indigo `#9BA9F0` after the dataviz validator showed it sat
  9.5 OKLab dE from the amber model hue (floor 15).
- Tooling: generated API reference with a staleness test; API stability
  labels with a deprecation decorator (ADR-0048).

## Review rounds and what they found

- Round 3: Cox baseline overflow, unmapped study stages, near-singular copula
  correlation, ADVI false convergence.
- Round 4: GEV profile crash on heavy tails, gamma/beta entropy cancellation,
  discrete divergence underflow, greedy design negative increments,
  non-symmetric covariances accepted; causal E-value overflow.
- Round 5: censored-normal EM cancellation far from zero, infinite
  hierarchical moments reported as finite, AK-MCS surrogate/probability
  mismatch, predictive-check thinning cap, KDE lower-tail cdf.
- Round 6: tanh-sinh truncation at 1e-15 (up to 3.7% error on heavy-tailed
  Wasserstein distances), Renyi/alpha overflow, energy-distance offset,
  Fisher-Rao cancellation.

## Mistakes worth remembering

- A ruff error was once committed because `ruff -q | tail -1` hid it; use
  `--output-format concise`.
- Two test-design errors of mine (zero misclassification assumed for
  AK-MCS; population sd used where Silverman needs the sample sd) were
  caught by the tests themselves before commit.
- Running `ruff format` on whole directories reformats other agents' files
  and creates merge noise; format only the files you changed.

## Verification evidence

Recorded in each merge commit message; the latest full run before this note:
`1007 passed, 1078 subtests passed`, ruff clean, strict mypy clean on 207
files, every example lab runs (1-34 s each).
