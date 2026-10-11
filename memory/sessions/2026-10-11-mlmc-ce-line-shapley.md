# Session — 2026-10-11 — MLMC, CE-IS, line sampling, Shapley effects

## Delivered

- `monte_carlo.multilevel`: `multilevel_monte_carlo` (fixed allocation),
  `adaptive_multilevel_monte_carlo` (Giles' algorithm), `optimal_level_allocation`,
  `convergence_rates`, `MultilevelResult`, and `LevelStatistics`.
- `reliability.adaptive`: `cross_entropy_importance_sampling` (Gaussian
  proposal; adapts the mean, with optional covariance adaptation) and
  `line_sampling` (FORM `alpha`, Brent on each line). Both are opt-in keys
  `"ce"` and `"line"` of `compare_reliability_methods`. `METHODS` is
  unchanged; the new `AVAILABLE_METHODS` holds all seven keys.
- `sensitivity.shapley`: `shapley_effects` (Song-Nelson-Staum, exact subsets
  or random permutations, Gaussian-copula conditional sampling),
  `exact_shapley_values`, and `linear_gaussian_shapley`.
- Ten new `EQ-*` records, nine bibliography entries, sections appended to
  the advanced Monte Carlo, advanced reliability, and sensitivity guides, and
  ADR-0029.

## Verification

- `tests/test_mlmc_ce_line_shapley.py`: 19 tests, about 5 s.
- Numbers in the guides come from the scratch script `doc_numbers.py`, which
  is not committed and uses the test helpers with the seeds stated in the
  text.
- The 2-D linear Gaussian Shapley closed form was derived by hand and checked
  against subset enumeration on four parameter sets (largest difference
  1.9e-16).

## Open items and next safe step

- CE covariance adaptation is unreliable on planar limit states (it stalled
  in 3 of 5 seeds). Gaussian-mixture CE (Kurtz-Song) is not implemented.
- Page ranges and volume/issue numbers of the new references come from
  memory. They were not checked against publisher pages in this session.
- Next: multiple design points, derivative-based global sensitivity measures,
  and MLMC for random-field PDE hierarchies.
