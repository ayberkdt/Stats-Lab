# ADR-0030 — Multiple regression, GLMs, and regularization

- Status: Accepted
- Date: 2026-10-12
- Scope: ROADMAP Phase 5, "Linear/nonlinear regression, GLMs, regularization"
  (the linear, GLM, and penalized parts)

## Context

`inference.regression` only fitted a simple intercept-and-slope line.
`inference.bayesian` already had conjugate linear-Gaussian and
Normal-inverse-gamma updates, but there was no frequentist multiple
regression: no coefficient inference, case diagnostics, robust errors, GLMs,
or penalized fits. `linalg` had Cholesky, LU, and Jacobi but no QR. The
library has no runtime dependencies.

## Decision

1. **Householder QR in `linalg` (layer 0).** `linalg.householder_qr` returns a
   frozen `HouseholderQR` with the thin `Q`, `R`, column norms, and the
   indices of numerically dependent columns. A column `j` is dependent when
   `|R_jj| <= 1e-10 ||a_j||`.
   - There is no column pivoting. Without pivoting, the flagged index is the
     column that depends on earlier ones, so the error can name it.
   - `solve`, `r_inverse`, `gram_inverse` (`R^-1 R^-T`), and `hat_diagonal`
     (`sum_j Q_ij^2`) are methods. Each refuses to run on a rank-deficient
     factor.
2. **`inference.linear_models` (layer 2).**
   - `design_matrix` builds the model matrix: intercept, numeric columns, and
     treatment-coded factors. Reference levels are configurable.
     `DesignMatrix.transform` re-encodes new data.
   - `fit_linear_model` computes OLS and WLS (WLS is OLS on `sqrt(w)`-scaled
     rows). It returns a frozen `LinearModelResult` with:
     - t inference, R², adjusted R², and the F test;
     - the Gaussian log-likelihood, AIC, and BIC with `p + 1` parameters;
     - leverage, internally and externally studentized residuals, and Cook's
       distance;
     - `interval(kind="confidence" | "prediction")`;
     - `robust_covariance("HC0" | "HC1" | "HC2" | "HC3")`;
     - variance inflation factors.
   - An intercept is recognized as any column identically equal to one.
     Without one, R² and F use the uncentered total sum of squares.
   - Rank deficiency raises `RankDeficientDesignError` naming the columns.
3. **`inference.glm`.** IRLS (Fisher scoring) with step halving.
   - Families and links: Gaussian (identity), binomial (logit, probit;
     proportions with trials as prior weights), Poisson (log), and Gamma
     (inverse, log). Offsets are supported.
   - Logit and probit predictors are clipped to `[-30, 30]` and `[-8, 8]` when
     inverted, so the weights stay positive.
   - Convergence uses the relative coefficient step `< 1e-10`. A
     deviance-only rule was rejected: it stopped probit Fisher scoring with
     score residuals near `2e-7`, and it declared separated fits converged.
   - Dispersion is fixed at 1 for binomial and Poisson. For Gaussian and Gamma
     it is the Pearson estimate `X^2/(n-p)`, and Wald tests then use
     `t_(n-p)` instead of the normal law.
   - AIC counts the dispersion as a parameter when it is estimated. The Gamma
     log-likelihood uses the plug-in `phi = D / sum w`.
4. **Separation is checked, not guessed from the size of the coefficients.**
   After IRLS, the last coefficient step `d` is tested against the
   Albert-Anderson condition: `x_i'd >= 0` when `y_i = 1`, `x_i'd <= 0` when
   `y_i = 0`, and `x_i'd = 0` when `0 < y_i < 1`. The inequalities are checked
   to a relative tolerance of `1e-6`, and only if `d` still moves some
   `x_i'd` by more than `1e-4`.
   - If the condition holds, the log-likelihood is non-decreasing along `d`
     and the MLE does not exist.
   - The default is to raise `SeparationError`. With
     `on_separation="warn"`, the fit emits `SeparationWarning` and is returned
     with `separation_detected` and the perfectly predicted indices.
5. **`inference.regularization`.**
   - **Shared setup.** Predictors are centred and scaled to unit population
     SD (optional). The intercept is not penalized, and coefficients are
     reported on the original scale.
   - **Ridge.** `fit_ridge` uses the closed form on the augmented system
     `[X; sqrt(lambda) I]` with the ESL penalty `RSS + lambda ||b||^2`. It
     also returns the effective degrees of freedom `||X R^-1||_F^2`. A zero
     penalty on a rank-deficient design raises.
   - **Lasso and elastic net.** `fit_elastic_net` and `elastic_net_path` use
     the glmnet objective and cyclic coordinate descent. The path is
     warm-started and log-spaced down from `lambda_max`.
   - **Cross-validation.** `cross_validate_elastic_net` reports `lambda_min`
     and `lambda_1se`. Folds come from a seeded permutation through
     `RandomStream(seed).child("cv-folds")`, and each training part is
     re-standardized on its own.
   - The two penalty scalings are documented together: the elastic net at
     `alpha = 0` equals ridge with penalty `n lambda`.
6. Dependencies stay downward: `inference` imports `linalg`, `foundations`,
   and its own `bayesian` (t quantile), `fitting` (information criteria), and
   `sampling_distributions`. `glm` and `regularization` reuse private helpers
   from `linear_models` in the same package.

## Consequences

- One QR code path serves OLS, WLS, every IRLS step, ridge, and VIF.
- Accuracy follows `kappa(X)` rather than `kappa(X)^2`. Cost is
  `O(n p^2)` per factorization in pure Python, which is fine at teaching and
  moderate sizes.
- A user cannot get silently regularized or pseudo-inverse estimates. Either
  the design is full rank, or an explicit ridge penalty is requested.
- Separated binomial data fail loudly. In warn mode the coefficients are
  documented as meaningless.
- Not covered: column pivoting, rank-revealing pseudo-solutions, penalized
  GLMs (logistic lasso), profile-likelihood intervals, Poisson MLE existence
  checks, and weighted penalized fits.

## Verification

`tests/test_linear_models_and_glms.py` has 31 tests and runs in about 3 s.

- Exact `Fraction` normal equations reproduce the coefficients, SEs, t, F,
  R², adjusted R², leverage, both studentized residuals, Cook's distance,
  HC0, HC3, and both interval widths. The WLS normal equations and
  replicated-row equivalence are checked the same way.
- Anscombe I gives 3.00 / 0.500 / R² 0.67.
- A seeded simulation checks 95% prediction and confidence interval
  coverage.
- An explicit Newton solve reproduces the logistic coefficients and SEs.
  The probit, Gamma-log, and Gamma-inverse fits satisfy their score
  equations.
- Poisson factor model: group-mean MLE, deviance, null deviance, Pearson
  X², AIC, and the `sum(y - mu) = 0` identity. The offset model returns the
  group rates.
- The Gaussian GLM equals WLS.
- Complete and quasi-complete separation raise for both links. Warn mode
  flags observations (0, 1, 4, 5).
- Ridge at `lambda = 0` equals OLS. Ridge matches
  `(X'X + lambda I)^-1 X'y`, and its effective df matches the eigenvalues.
  The elastic net at `alpha = 0` equals ridge at `n lambda`.
- The lasso and elastic-net solutions satisfy the KKT conditions. The path
  starts at `lambda_max`.
- CV folds are balanced and reproducible, and CV keeps the true predictors.
- Rank-deficient designs raise in OLS, GLM, and zero-penalty ridge.

## References

[REF-HOUSEHOLDER-1958], [REF-GOLUB-VANLOAN-2013],
[REF-MONTGOMERY-PECK-VINING-2021], [REF-COOK-1977], [REF-WHITE-1980],
[REF-MACKINNON-WHITE-1985], [REF-NELDER-WEDDERBURN-1972],
[REF-MCCULLAGH-NELDER-1989], [REF-ALBERT-ANDERSON-1984],
[REF-HOERL-KENNARD-1970], [REF-TIBSHIRANI-1996], [REF-ZOU-HASTIE-2005],
[REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009],
[REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010], [REF-ANSCOMBE-1973].
