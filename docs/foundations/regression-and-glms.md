# Regression, Generalized Linear Models, and Regularization

`probstat.inference` fits multiple linear regression (OLS and WLS),
generalized linear models by IRLS, and ridge, lasso, and elastic-net models
with K-fold cross-validation. Every solve goes through the Householder QR in
`probstat.linalg` (EQ-LINALG-009); the normal equations are never formed.
Results are frozen dataclasses. Failures raise errors instead of being
patched silently: a rank-deficient design raises `RankDeficientDesignError`,
and binomial separation raises `SeparationError`.

The numbers below were produced by running the library on the data shown.
The test file `tests/test_linear_models_and_glms.py` checks the same
quantities against independent computations: exact rational arithmetic, a
hand-written Newton iteration, closed-form maximum-likelihood estimates, and
explicit KKT checks.

## 1. Design matrices

`design_matrix(data, categorical=[...])` builds the model matrix from a
mapping of named columns:

- With an intercept, a factor uses treatment coding (EQ-LM-DESIGN-001). The
  first level in sorted order is the reference unless `reference=` names
  another one.
- Without an intercept, the first factor keeps every level (cell-means
  coding).
- `DesignMatrix.transform(new_data)` encodes new rows the same way. It
  rejects levels that were not seen when the design was built.

```python
from probstat.inference import design_matrix, fit_linear_model

design = design_matrix(
    {"dose": [1, 2, 3, 1, 2, 3], "site": ["b", "a", "c", "a", "c", "b"]},
    categorical=["site"],
)
design.column_names   # ('(Intercept)', 'dose', 'site[b]', 'site[c]')
```

## 2. Multiple linear regression

`fit_linear_model(design, y, weights=None)` factors `W^(1/2) X = QR` and
solves `R beta = Q'W^(1/2) y` (EQ-LM-OLS-001). The worked example uses eight
observations, an intercept, and two predictors `a` and `b`:

| a | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| b | 2 | 1 | 5 | 3 | 6 | 4 | 8 | 7 |
| y | 3.1 | 4.0 | 8.2 | 6.9 | 11.4 | 9.8 | 14.1 | 15.3 |

| Term | Estimate | SE | t | p (t, 5 df) |
|---|---|---|---|---|
| (Intercept) | 0.661039 | 0.432365 | 1.529 | 0.187 |
| a | 0.994805 | 0.149589 | 6.650 | 0.00116 |
| b | 0.880519 | 0.149589 | 5.886 | 0.00201 |

Fit summary (EQ-LM-INF-001, EQ-LM-FIT-001, EQ-LM-LL-001):

| Quantity | Value |
|---|---|
| Residual standard error | 0.535881 on 5 df |
| R² | 0.989510 |
| Adjusted R² | 0.985314 |
| F | 235.83 on (2, 5) df, p = 1.13e-5 |
| AIC (with k = p + 1 = 4) | 16.9615 |

The test suite solves `X'X beta = X'y` exactly with `fractions.Fraction`. The
coefficients, RSS, R², adjusted R², and F agree to 12 decimal places, and F
to 9.

**Intervals at `x0 = (1, 4.5, 3.5)` (EQ-LM-PRED-001).**

| Interval | Mean | Bounds |
|---|---|---|
| 95% for the mean | 8.2195 | [7.5989, 8.8400] |
| 95% for a new observation | 8.2195 | [6.7086, 9.7303] |

The prediction interval adds `sigma^2/w0` to the variance of the mean. A
seeded simulation (400 replications, quadratic mean, n = 15) checks that both
intervals cover close to 95%. The tolerance is 3.5 binomial standard errors.

**Diagnostics.**

- Leverages (EQ-LM-DIAG-001): 0.430, 0.430, 0.417, 0.222, 0.222, 0.417,
  0.430, 0.430. They sum to `p = 3`.
- Externally studentized residuals (EQ-LM-DIAG-002) range from -1.618 to
  1.393.
- The largest Cook's distance (EQ-LM-DIAG-003) is 0.498, at observation 7.

**Robust standard errors (EQ-LM-HC-001).**

| Estimator | (Intercept) | a | b |
|---|---|---|---|
| HC0 | 0.3353 | 0.0981 | 0.1115 |
| HC3 | 0.5778 | 0.1654 | 0.1846 |

The test computes `(X'X)^-1 X' diag(omega) X (X'X)^-1` exactly and matches
it to 11 decimal places.

**Collinearity (EQ-LM-VIF-001).** `a` and `b` are correlated with
`r^2 = 0.694`, so both VIFs equal `1/(1 - r^2) = 3.273`.

**Anscombe's first data set** [REF-ANSCOMBE-1973]. The fit gives
intercept 3.00009, slope 0.500091, R² 0.666542, and slope p-value 0.00217.
This matches the rounded values usually quoted (3.00, 0.500, 0.67) and the
exact rational least-squares solution.

**Weighted least squares.** With integer weights, WLS equals OLS on rows
replicated that many times. The test checks this identity and also the exact
rational weighted normal equations.

## 3. Generalized linear models

`fit_glm(design, y, family=..., link=..., weights=..., offset=...)` runs
Fisher scoring as IRLS (EQ-GLM-IRLS-001) with step halving.

- Convergence is declared when `max_j |delta beta_j|/(|beta_j| + 0.1)` falls
  below `tolerance` (default `1e-10`).
- A deviance-change criterion alone is not used. It stops linearly
  convergent Fisher scoring with coefficient errors near `sqrt(tolerance)`:
  an earlier draft left probit score equations at `2e-7`.

Supported families and links (EQ-GLM-FAM-001):

| Family | Links | Dispersion | Wald reference |
|---|---|---|---|
| gaussian | identity | `X^2/(n-p)` | `t_(n-p)` |
| binomial (y = proportion, weights = trials) | logit, probit | 1 | normal |
| poisson | log | 1 | normal |
| gamma | inverse, log | `X^2/(n-p)` | `t_(n-p)` |

Each result reports:

- the deviance and null deviance (EQ-GLM-DEV-001);
- the Pearson chi-square and dispersion (EQ-GLM-PEARSON-001);
- Wald standard errors (EQ-GLM-WALD-001);
- the log-likelihood, AIC, and BIC (EQ-GLM-AIC-001);
- deviance and Pearson residuals;
- the iteration count and the `converged` flag.

### Logistic and probit regression

The example data are `x = 0.5, 1.0, ..., 5.0` with
`y = 0 0 1 0 1 0 1 1 1 1`.

| Link | Intercept (SE) | Slope (SE) | Deviance | AIC | Iterations |
|---|---|---|---|---|---|
| logit | -2.92651 (2.06009) | 1.32442 (0.80019) | 8.62024 | 12.62024 | 6 |
| probit | -1.80698 (1.15349) | 0.81493 (0.43756) | 8.48344 | 12.48344 | 12 |

The null deviance is `-2(6 log 0.6 + 4 log 0.4) = 13.4602`. The test checks
the logit fit against an explicit two-parameter Newton iteration written in
the test, using 2x2 matrix inverses. Coefficients agree to 9 decimal places
and standard errors to 8. For the probit fit, the test checks that the score
equations are zero at the solution.

Grouped binomial data, with the trials passed as `weights`, give the same
coefficients and standard errors as the expanded Bernoulli rows.

### Poisson regression and deviance identities

Counts in three groups: a = (2, 3, 1, 4), b = (6, 8, 5, 7), c = (0, 1, 2, 1).
With a factor-only design the MLE is the vector of group means (2.5, 6.5,
1.0). The coefficients are `log 2.5 = 0.916291`, `log(6.5/2.5) = 0.955511`,
and `log(1/2.5) = -0.916291`.

| Quantity | Value |
|---|---|
| Deviance `2 sum y log(y/mu)` | 5.67698 |
| Null deviance | 25.0187 |
| Pearson X² | 4.76923 |
| AIC | 44.0457 |

The canonical log link with an intercept forces `sum(y - mu) = 0`, so the
`-(y - mu)` term drops out of the deviance. With an offset `log(exposure)`,
each group's fitted rate equals its total count divided by its total
exposure.

### Gamma regression

For `x = 0, ..., 8` and `y = 2, 3, 6, 7, 8, 9, 10, 12, 15`:

| Link | Coefficients | Dispersion | AIC |
|---|---|---|---|
| log | (1.06700, 0.21874) | 0.050011 | 37.031 |
| inverse | (0.253579, -0.024556) | 0.082092 | 43.341 |

The AIC uses the plug-in dispersion `D / sum(w)`. This is an approximation;
it is not the exact shape MLE.

### Separation

Logit and probit MLEs do not exist when some direction `d` separates the
classes (EQ-GLM-SEP-001) [REF-ALBERT-ANDERSON-1984].

- **Detection.** The library checks the separation inequalities on the last
  IRLS step, but only when that step still moves the linear predictor by more
  than `1e-4`. Converged regular fits take far smaller steps.
- **Example.** With `x = 1, 2, 3, 3, 4, 5` and `y = 0 0 0 1 1 1`, the data
  are quasi-completely separated: the tie at `x = 3` holds both classes.
- **Default behavior.** `fit_glm(..., family="binomial")` raises
  `SeparationError`.
- **Warn mode.** `on_separation="warn"` returns the diverged fit with
  `separation_detected=True` and `separated_observations=(0, 1, 4, 5)`. Its
  coefficients have drifted to about (-215, 72) after 100 iterations, so
  neither they nor their standard errors mean anything.

## 4. Ridge, lasso, and elastic net

Predictors are centred and scaled to unit population standard deviation
(EQ-PEN-STD-001). The intercept is not penalized, and coefficients are
reported on the original scale. Each method uses the penalty scaling of its
standard source:

| Model | Objective | Solver |
|---|---|---|
| `fit_ridge` | `||y_c - X~b||^2 + lambda ||b||^2` | augmented QR (EQ-PEN-RIDGE-001) |
| `fit_elastic_net` | `(1/2n)||y_c - X~b||^2 + lambda((1-a)/2 ||b||^2 + a ||b||_1)` | cyclic coordinate descent (EQ-PEN-EN-001) |

As a result, the elastic net with `alpha = 0` and penalty `lambda` equals
ridge with penalty `n lambda`. The test checks this to 8 decimal places.

The examples use seeded data: n = 50, six predictors with pairwise
correlation 0.36, true slopes (2, -1.5, 0, 0, 1, 0), and unit noise.

**Ridge.** Effective degrees of freedom `sum d_j^2/(d_j^2 + lambda)`:

| lambda | 0 | 1 | 10 | 100 |
|---|---|---|---|---|
| df | 6.000 | 5.797 | 4.534 | 1.702 |

At `lambda = 0` ridge reproduces OLS. The test checks this to 11 decimal
places, and checks the effective df against the eigenvalues of `X~'X~`.

**Lasso and elastic net.**

- `lambda_max = 1.93321`. This is the smallest penalty with an all-zero
  solution (EQ-PEN-KKT-001).
- Lasso at `lambda = 0.15`: slopes (1.622, -1.067, 0, 0.243, 0.713, 0), so
  four predictors are active.
- Elastic net at `alpha = 0.5`, `lambda = 0.2`: five predictors are active.
- The tests check the KKT conditions directly. On the active set,
  `x~_j'r/n - lambda(1-alpha) b_j = lambda alpha sign(b_j)`. Off it, the
  same gradient is at most `lambda alpha` in absolute value.

**Cross-validation.** `cross_validate_elastic_net(..., folds=5, seed=11)`
uses a 15-point grid. Folds come from a seeded permutation
(`RandomStream(seed).child("cv-folds")`), and their sizes differ by at most
one (EQ-PEN-CV-001).

| Quantity | Value |
|---|---|
| `lambda_min` | 0.0611 (CV MSE 1.450) |
| `lambda_1se` | 0.1640 |
| Active set at both | x0, x1, x3, x4 |

The truly non-zero x0, x1, and x4 are always kept. The correlated null
predictor x3 also enters, which is a known lasso behavior with correlated
designs and n = 50.

## 5. Limitations

- No column pivoting. A dependent column is reported by name rather than
  dropped.
- No regression weights for the penalized models. The binomial AIC uses
  `lgamma` for non-integer `trials * y`.
- Separation is certified only for binomial families. Poisson MLE
  non-existence (zero counts on a face of the design) shows up only as
  non-convergence.
- Wald intervals only. Profile-likelihood intervals for GLMs are not
  implemented.
- Pure-Python cost is `O(n p^2)` per QR. Coordinate descent costs `O(n p)`
  per sweep. Both are aimed at teaching-sized and moderate problems.
