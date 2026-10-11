# Session — 2026-10-11 — Continuous distribution catalog

## Delivered

- `probstat.distributions` adds the following families, each with the full
  contract (`support`, `log_pdf`, `pdf`, `cdf`, `sf`, `quantile`, `isf`,
  `mean`, `variance`, seeded `sample`):
  - `Gamma` (shape-rate, `from_scale`), `Beta`, `StudentT` (location-scale),
    `ChiSquare`, and `FisherF`;
  - `Logistic`, `Gumbel` (`kind="max" | "min"`), and `Triangular`;
  - `TruncatedNormal` (finite or one-sided) and `GeneralizedExtremeValue`.
- Internal `distributions._continuous` module:
  - a log-tail quantile solver with transformed coordinates and a local
    Brent iteration;
  - log normal tails with a Mills-ratio continued fraction and short-interval
    masses;
  - Marsaglia-Tsang gamma variates in log form.
- `_special.regularized_beta_pair` takes an explicit `1 - x`, which keeps
  Student-t and F tails accurate as `nu -> inf`.
- `numerics.digamma`, `numerics.trigamma`, and `numerics.log_minus_digamma`.
- `inference.fit_gamma_mle` and `inference.fit_weibull_mle`, both by
  profile-equation Newton iteration with a bracket safeguard.
- Equation records `EQ-DIST-INV-001`, `GAM`, `BETA`, `STT`, `CHI2`, `F`,
  `LOGIS`, `GUM`, `TRI`, `TN`, `GEV`, `EQ-NUM-DIGAMMA-001`, and
  `EQ-INF-MLE-GAM/WEIB-001`. Five references (JKB 1994 and 1995,
  Marsaglia-Tsang 2000, Coles 2001, Kotz-van Dorp 2004), a guide in
  `docs/foundations/continuous-distribution-catalog.md`, and ADR-0020.

## Verification

- `tests/test_continuous_distribution_catalog.py`: 31 tests, about 8 s.
  Covers quadrature mass, far-tail round trips to `1e-12`, closed-form and
  table values, sample moments, invalid parameters, Nataf round trips up to
  `|u| = 7.5`, MLE recovery and score equations, and digamma identities.
- Full suite, ruff, and mypy: see the commit report.

## Known limits

- Near a finite support end, probability resolution is limited by the float
  spacing of `x`.
- Gamma and beta prefactors lose relative accuracy for shapes far above
  `1e4`, because they are formed from `lgamma` differences.
- `StudentT` with `nu <= 1` cannot be a `RandomInput`, because its mean is
  undefined and the property raises.

## Next safe step

- Temme uniform asymptotics for large-shape incomplete gamma and beta.
- Rejection samplers for truncated normals.
- GEV and Gumbel maximum-likelihood fitting.
- README, CHANGELOG, and ROADMAP entries (coordinator-owned).
