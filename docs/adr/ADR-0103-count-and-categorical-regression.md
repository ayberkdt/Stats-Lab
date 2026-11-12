# ADR-0103: Count and categorical regression models

- Status: Accepted
- Date: 2026-11-12

## Context

`inference.fit_glm` (ADR-0030) fits Poisson regression by IRLS with
model-based Wald errors only, and `inference.count_models` fits the Poisson
and negative binomial laws to identically distributed counts without
covariates. Applied count and categorical analyses also need:

1. negative binomial regression with the size estimated by maximum
   likelihood, exposures, robust standard errors for Poisson regression,
   and a test of equidispersion;
2. models for excess zeros (zero-inflated and hurdle), together with a
   clear statement of what the Vuong statistic can and cannot decide;
3. regression for ordered and unordered categorical responses, predicted
   probabilities, and marginal effects on the probability scale with
   standard errors.

Constraints: standard library only, strict mypy, documented and cited
definitions, layer 2 (`inference`) imports only `linalg`, `optimization`,
and lower layers, and new code goes into new modules.

## Decision

1. **Shared machinery in `inference._ml_tools` (private).** A damped Newton
   maximizer: each step solves `(-H + lambda I) d = g` with `lambda = 0`
   when `-H` is positive definite and the smallest working multiple
   otherwise, then backtracks to the Armijo condition; the Hessian is
   analytic or the symmetrized central-difference Jacobian of the analytic
   gradient (`optimization.central_jacobian`). Also the inverse observed
   information, Wald tables, and `MarginalEffects` with the delta-method
   covariance `G V G'`, `G` by central differences of the analytic effect
   map (EQ-REG-AME-001).
2. **`inference.count_regression`.** `fit_poisson_regression` reuses
   `fit_glm` and adds exposures (`log t` offsets, EQ-CNT-REG-001) and the
   HC0 sandwich `A^-1 B A^-1` (EQ-CNT-REG-002, REF-WHITE-1982).
   `fit_negative_binomial_regression` maximizes the NB2 likelihood jointly in
   `(beta, log theta)` with the analytic Hessian (EQ-CNT-REG-003), starting
   from the Poisson fit and a moment estimate of `alpha`. The coefficient
   covariance is the inverse expected information `(X'WX)^-1`,
   `W = mu/(1 + alpha mu)`, which is block-diagonal against `theta`;
   `theta` has its observed-information standard error. When
   `sum[(y - mu)^2 - y] <= 0` at the Poisson fit (the leading term of the
   `theta` score at infinity) or Newton drives `theta` past `1e12`, the fit
   reports the Poisson boundary (`theta = inf`, `at_boundary`), as
   `fit_negative_binomial_counts` does. `poisson_overdispersion_test` is
   the auxiliary least-squares test of Cameron and Trivedi (1990)
   (EQ-CNT-REG-004), NB2 or NB1 form.
3. **`inference.zero_count_models`.** `fit_zero_inflated` maximizes the
   zero-inflated Poisson or NB2 likelihood directly (EQ-CNT-ZI-001,
   REF-LAMBERT-1992) with analytic gradients and a central-difference
   Hessian; the start is the plain count regression, a logit intercept at
   the excess-zero share, and the plain NB2 size. `fit_hurdle` uses the
   factorization of the hurdle likelihood (EQ-CNT-HURDLE-001,
   REF-MULLAHY-1986): `fit_glm` for the logit of `1{y > 0}` and Newton steps
   for the zero-truncated count part; the covariance is block diagonal. The
   hurdle zero coefficients model `P(y > 0)`, the zero-inflated ones the
   structural-zero probability; both conventions are stated in the
   docstrings and the guide. `vuong_test` computes Vuong's statistic with
   optional AIC/BIC corrections (EQ-CNT-VUONG-001); its docstring, the
   registry, and the guide state that a count model and its zero-inflated
   extension are not strictly non-nested, so the statistic is not a test for
   zero inflation (REF-WILSON-2015). The function does not refuse such a
   comparison, because the library cannot tell which comparison a pair of
   likelihood vectors represents.
4. **`inference.categorical_regression`.** `fit_ordinal_logistic` fits
   `logit P(Y <= k) = theta_k - x' beta` (MASS `polr` sign convention,
   EQ-CAT-ORD-001, REF-MCCULLAGH-1980) by Newton-Raphson with the analytic
   Hessian, keeping the thresholds ordered by step halving; a constant
   design column is rejected because the thresholds absorb the intercept.
   `brant_test` (EQ-CAT-BRANT-001, REF-BRANT-1990) fits a binary logit for
   each cut and forms the Wald test of equal slopes with the cross-cut
   covariance `(X'W_kk X)^-1 X'W_kl X (X'W_ll X)^-1`, omnibus and per
   column. `fit_multinomial_logit` fits the baseline-category logit
   (EQ-CAT-MNL-001) by Newton-Raphson with the analytic Hessian and an
   optional reference level. Both results give predicted probabilities,
   odds or relative-risk ratios, and derivative or discrete average
   marginal effects. A fitted probability of an observed category above
   `1 - 1e-10` raises `SeparationError`: Newton's decrement can become
   negligible along a separating direction, so a convergence flag alone
   would report diverged estimates as converged.

## Consequences

- Seventeen public names in `inference`: `fit_poisson_regression`,
  `fit_negative_binomial_regression`, `poisson_overdispersion_test`,
  `fit_zero_inflated`, `fit_hurdle`, `vuong_test`, `fit_ordinal_logistic`,
  `brant_test`, `fit_multinomial_logit`, and the records
  `CountRegressionResult`, `OverdispersionTest`, `ZeroCountResult`,
  `VuongTest`, `OrdinalRegressionResult`, `BrantTest`,
  `MultinomialLogitResult`, `MarginalEffects`.
- Eleven equations (EQ-CNT-REG-001 to 004, EQ-CNT-ZI-001,
  EQ-CNT-HURDLE-001, EQ-CNT-VUONG-001, EQ-CAT-ORD-001, EQ-CAT-BRANT-001,
  EQ-CAT-MNL-001, EQ-REG-AME-001) and nine sources, all verified against
  Crossref records.
- New guide `docs/foundations/count-and-categorical-regression.md`, step S16
  of the Statistics track.
- The README equation and public-name counts and `docs/api-reference.md`
  must be regenerated by the integration step.
- Limitations: no cluster-robust or HC1-HC3 variants for count models; no
  generalized ordinal (partial proportional odds) model; Brant's test uses
  the separate binary fits, so it fails when one of them is separated; the
  zero-inflated and hurdle Hessians are central differences (relative
  error near `1e-7`), adequate for standard errors but not for
  second-order corrections.

## Verification

`tests/test_count_and_categorical_regression.py` (29 tests). No R or
statsmodels output is quoted: neither is installed on the build machine,
and no public-domain data set with verified published `MASS::glm.nb`,
`pscl::zeroinfl`/`hurdle`, `MASS::polr`, or `nnet::multinom` output was
available offline. The oracles are:

| Method | Oracle |
|---|---|
| NB2 intercept-only | `fit_negative_binomial_counts` (size, its standard error, log-likelihood) and the closed form `se = sqrt((1 + mu/theta)/(n mu))` |
| NB2 with a covariate | finite-difference score of a likelihood coded in the test vanishes; seeded simulation (n = 1500) within four standard errors for `beta` and `alpha` |
| NB2 boundary | under-dispersed data return the Poisson coefficients and `theta = inf` |
| Poisson exposure | equals `offset = log t`; saturated factor rates `sum y / sum t` |
| Robust errors | saturated two-group design: `sum (y - y_bar)^2 / (n y_bar)^2` per group |
| Overdispersion test | no-intercept OLS recomputed in the test (NB2 and NB1); size at 5% over 200 seeded Poisson replicates within four binomial standard errors |
| ZIP, Poisson hurdle | intercept-only closed form `lambda/(1 - e^-lambda) =` mean positive count, `pi = 1 - y_bar/lambda`, hurdle logit `log(n_+/n_0)` with `se = sqrt(1/(n p (1-p)))`; equal maximized likelihoods; ZIP covariance equals an independently coded inverse Hessian |
| ZINB, NB hurdle | intercept-only maxima agree; seeded simulation (n = 1500) within four standard errors including `log theta` |
| Hurdle count part | finite-difference score of an independently coded truncated likelihood vanishes; binary part equals `fit_glm` |
| Vuong | the defining formula with each correction |
| Ordinal | thresholds-only closed form `theta_k = logit F_k`, `Cov = 1/(n (1 - F_k) F_l)`; two categories equal `fit_glm` logistic regression (slopes, `theta = -intercept`, standard errors); seeded simulation within four standard errors; derivative AMEs equal central differences of predicted probabilities and sum to zero |
| Brant | one binary covariate and three categories: the explicit log-odds-ratio Wald form; a seeded non-proportional design is rejected |
| Multinomial | saturated two-group design: log ratios of counts, `sqrt(sum 1/n)` errors, discrete AMEs equal differences of proportions with the binomial delta-method errors; two categories equal logistic regression; reference change flips signs |
| Separation | perfectly separated data raise `SeparationError` in both categorical fits |
