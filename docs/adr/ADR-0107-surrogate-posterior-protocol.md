# ADR-0107: Surrogate posterior protocol for active-learning reliability

- Status: Accepted
- Date: 2026-11-18

## Context

`reliability.ak_mcs` was written around one surrogate and one rule: a
constant-mean Gaussian process, updated over the Monte Carlo population by
exact rank-one Cholesky steps (ADR-0049), and the learning function
`U = |mu| / sigma` with the stop `min U >= 2`. The incremental update and the
learning rule lived in one private class, so neither could be reused. Planned
methods need both pieces in other combinations: a subset-simulation variant
(AK-SS) feeds a new candidate pool at every level, PC-Kriging has a
polynomial trend that the rank-one update of the constant-mean GP does not
cover, and a deep ensemble has no incremental update at all. The literature
also uses other learning functions (Bichon's `EFF`, the entropy `H`) and a
stopping rule on the failure-probability bounds instead of a pointwise
threshold.

## Decision

1. **Gaussian predictive law as the only contract.** Active learning reads a
   predictive mean and variance per candidate and treats
   `G(x) ~ N(mu, sigma^2)`. `surrogates.ProbabilisticSurrogate` is the
   structural protocol (`predict` and `predict_with_variance(x) -> (mean,
   variance)`); `GaussianProcessRegression` satisfies it unchanged.
2. **Refit hook.** `surrogates.SurrogateFactory` is a callable
   `(points, values, previous) -> ProbabilisticSurrogate`, called on the
   initial design and after every added point, with the surrogate it replaces
   so it can warm-start or keep hyperparameters. `GaussianProcessFactory`
   implements it with a warm-started likelihood fit per call. Both live in
   `surrogates`, which still imports nothing from `reliability` (ADR-0041).
3. **Population posterior protocol.** `reliability.PopulationPosterior`
   exposes `population`, `means`, `variances`, `training_points` and the
   updates `add_training`, `add_population`, and `replace_population`.
   `GaussianProcessPopulationPosterior` is the former private incremental
   class, unchanged in arithmetic (bit-identical to the batch factorization,
   ADR-0049) and now able to swap its candidate pool while keeping its
   factor. `SurrogatePopulationPosterior` serves any factory by refitting and
   re-predicting the whole population.
4. **Learning functions and stopping rules as plain functions.**
   `reliability.learning` holds `learning_value(name, mean, std)`,
   `select_learning_point`, `stopping_satisfied` with the published defaults
   (`min U >= 2`, `max EFF <= 0.001`, `max H <= 0.5`), and
   `probability_bounds` / `bounds_satisfied` for the bound rule
   `(P_f^+ - P_f^-) / P_f^0 <= tolerance` with `mu -/+ 1.96 sigma`. They do
   not know where the candidates came from.
5. **Defaults unchanged.** `ak_mcs` keeps the incremental GP path, `U`, the
   threshold 2, and the LHS design by default; `learning_threshold=None`
   means the function's default, so callers passing `2.0` with `U` get the
   same run. Seeded default runs reproduce the earlier probabilities, call
   counts, and learning histories bit for bit (checked on seven
   configurations, including the four-branch numbers in
   `docs/foundations/adaptive-reliability.md`).
6. **Initial design.** `ak_mcs` adds `design="lhs-maximin"` (the sampling
   module's Morris-Mitchell maximin LHS, called from `reliability`; the
   shared `propagation.make_design` is unchanged) and `extra_points` in
   standard space, for example design points from `find_design_points`.

## Consequences

- AK-SS can build one posterior and call `replace_population` per level;
  PC-Kriging and ensembles plug in through a factory without touching the
  learning loop. A trend-aware incremental update can later implement the
  same protocol.
- The generic path costs one refit and `N` predictions per added point, so it
  is much slower than the incremental GP (about 7 s against 1.5 s for the
  four-branch function with `N = 2000`).
- `AKMCSResult.surrogate` is typed as `ProbabilisticSurrogate`; code that
  needs GP attributes must check the type. The result now carries the
  training data (`training_points`, `training_values`), the learning
  function, the stopping rule, and the final `P_f` bounds; the classification
  plot reads the training points from the result.
- `EFF` and `H` are not scale-free (`EFF` is in units of `G`, `H` contains
  `ln sigma`), so their published thresholds assume a limit state of order
  one; `U` and the bound rule are scale-free.
