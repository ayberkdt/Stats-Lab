# Session: UQ-3 advanced Monte Carlo

Date: 2026-10-09

## Delivered

- Antithetic unit-uniform estimation with independent-pair uncertainty and
  two-evaluations-per-pair accounting.
- Fixed or sample-fitted control variates with raw/adjusted variance diagnostics.
- Conditional-expectation Monte Carlo with the model boundary left explicit.
- Equal-width stratified sampling with design-based variance and confidence
  intervals.
- Ordinary importance sampling with density validation, weight ESS, and maximum
  normalized-weight diagnostics.
- Lower-bounded Neyman allocation and cost-normalized efficiency comparison.
- A strategy-comparison plot, an English conceptual guide, and a seeded lab
  targeting the same expectation under a common evaluation budget.
- Eight registered equations, a primary Monte Carlo reference, ADR-0010, and
  regression tests.

## Design boundary

No method silently converts dependent observations into IID samples. Sample
size, model evaluations, output/contribution variance, estimator standard
error, and weight degeneracy remain distinct quantities.

## Verification

- Ruff passed across source, tests, and examples.
- Strict mypy passed across 43 source files.
- All 119 unit tests passed, including architecture, documentation, equation,
  bibliography, and 110-row curriculum-ledger contracts.
- Bytecode compilation and the seeded advanced laboratory completed.
- The generated efficiency chart was visually inspected for readable labels,
  baseline parity, logarithmic scale, and unclipped content.
- The `0.10.0` wheel built in an isolated environment and passed an isolated
  zip-import smoke test.
- The verified UQ-2 and UQ-3 release candidate was deployed to the authoritative
  D-drive repository and the full verification suite was repeated there before
  committing.

## Next safe step

Implement UQ-4 reliability from a named limit-state contract through direct
failure estimation, then add FORM and directional sampling behind explicit
optimization and distribution-transformation boundaries.
