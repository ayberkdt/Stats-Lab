# 2026-10-12 — Multiple regression, GLMs, and regularization

## Done

- `linalg.qr`: `householder_qr` and `HouseholderQR` (solve, `r_inverse`,
  `gram_inverse`, `hat_diagonal`, rank flags). Equation: EQ-LINALG-009.
- `inference.linear_models`: `design_matrix`/`DesignMatrix`/`ColumnEncoding`,
  `fit_linear_model` (OLS and WLS), `LinearModelResult`, `LinearPrediction`,
  `variance_inflation_factors`, `RankDeficientDesignError`. It returns
  intervals, diagnostics, and HC0-HC3 robust covariances.
- `inference.glm`: `fit_glm` by IRLS for Gaussian, binomial (logit, probit),
  Poisson, and Gamma (inverse, log). Separation is certified and handled by
  `SeparationError`/`SeparationWarning`; non-convergence warns with
  `GLMConvergenceWarning`.
- `inference.regularization`: `fit_ridge`, `fit_elastic_net`,
  `elastic_net_path`, `cross_validate_elastic_net`, `kfold_assignment`.
- New equation IDs: EQ-LM-*, EQ-GLM-*, and EQ-PEN-*. New bibliography
  entries: 13. Also added the guide
  `docs/foundations/regression-and-glms.md` and ADR-0030.

## Verification

`tests/test_linear_models_and_glms.py`: 31 tests, about 3 s. Full-suite,
ruff, and mypy results are recorded in the commit hand-off.

## Next safe step

- Penalized GLMs (logistic lasso by IRLS plus coordinate descent).
- Profile-likelihood intervals.
- A Poisson MLE existence check.
- A regression lab in `examples/`.
- Linking the new guide from `docs/learning-path.md`.
