# 2026-10-12 — Extreme value analysis and information theory

## Done

- `distributions.GeneralizedPareto` (`mu`, `sigma`, `xi`): full API with a
  stable `xi -> 0` exponential limit and an `entropy` property.
- `risk.extremes`: `block_maxima`, `fit_gev` / `GEVFit` (delta-method and
  profile-likelihood return-level intervals), `gev_return_level(_gradient)`,
  `fit_gpd` (MLE via `fit_distribution` + PWM/L-moments),
  `sample_l_moments`, `mean_residual_life`, `threshold_stability`,
  `exceedance_rate` / `ExceedanceRate`, `decluster_runs`, `fit_pot` /
  `PeaksOverThresholdFit`, `pot_return_level`.
- New layer-2 package `probstat.information`: discrete entropy, joint and
  conditional entropy, mutual information, KL, cross-entropy, Jensen-Shannon;
  closed-form differential entropies (normal, MVN, uniform, exponential,
  gamma, beta, lognormal, GEV/Gumbel, GPD) with a type dispatcher; Gaussian
  KL (uni/multivariate); Kozachenko-Leonenko and KSG estimators.
- Docs: `docs/foundations/extreme-value-analysis.md`,
  `docs/foundations/information-theory.md`; ADR-0036; 30 equation records
  and 8 references appended.

## Verification

`PYTHONPATH=src python -m pytest -q`: 712 passed, 978 subtests. The two new
test files take about 10 s. `ruff check .` clean; `mypy src/probstat` clean.

## Next safe step

Profile intervals for POT return levels, the Hosking-Wallis PWM covariance
(verify the coefficients first), non-stationary GEV/GPD, and a protocol-level
`entropy` capability.
