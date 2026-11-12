# ADR-0100: Linear and generalized linear mixed models

- Status: Accepted
- Date: 2026-11-12

## Context

`probstat.inference` had the eight-schools hierarchical normal model with
known standard errors, beta-binomial empirical Bayes, and GLMs, but no
general mixed model: no way to fit grouped data with an unknown residual
variance, random slopes, nested or crossed factors, or non-normal responses
with random effects. Mixed models are a core tool of applied statistics, and
the reference implementations are lme4 (Bates et al. 2015) and lmerTest
(Kuznetsova et al. 2017).

Constraints: standard library only, strict mypy, every definition
documented and cited, dense linear algebra from `probstat.linalg`,
optimizers from `probstat.optimization`, and the GLM families of
`inference.glm`.

## Decision

1. **New module `inference.mixed_models`** with `random_effect`
   (`RandomEffectTerm`), `fit_linear_mixed_model`,
   `fit_generalized_linear_mixed_model`, `glmm_deviance`,
   `mixed_model_likelihood_ratio_test`, and the result records
   `LinearMixedModelResult`, `GeneralizedLinearMixedModelResult`,
   `VarianceComponent`, `RandomEffectEstimates`, `ContrastTest`, and
   `MixedModelLikelihoodRatioTest`, all exported from `probstat.inference`.
2. **lme4 formulation.** `b = Lambda_theta u` with `u ~ N(0, sigma^2 I)` and
   one lower-triangular block `T_k` per level [EQ-LMM-001]; for fixed
   `theta` one penalized least-squares solve through `L L' = Lambda' Z' Z
   Lambda + I` and the Schur complement `R_X' R_X` gives `beta`, `u`, and
   `sigma` [EQ-LMM-002], leaving the profiled ML and REML deviances
   [EQ-LMM-003, EQ-LMM-004]. `Z` is stored as sparse rows; `Lambda' Z' Z
   Lambda` is formed block by block and factored by an envelope Cholesky, so
   a single grouping factor costs `O(l r^3)` per evaluation instead of
   `O(q^3)`. A general sparse Cholesky with fill-reducing ordering is not
   needed at the intended scale (hundreds of observations, tens of levels).
3. **Optimization in unconstrained `theta`.** lme4 bounds the diagonal of
   `T_k` at zero. Negating a column of `T_k` leaves `T_k T_k'` unchanged, so
   the deviance can be minimized by BFGS (default) or Nelder-Mead over all
   of `R^m` and the result mapped to a non-negative diagonal. The deviance is
   then even in each column, which makes `theta_ii = 0` a stationary point
   that is either a boundary minimum or a saddle between two mirror-image
   optima; the first implementation stopped on such saddles (from the start
   `theta = 1`, BFGS landed on `theta = 0` with an exactly zero central
   difference) and reported boundary fits for 25 of 60 seeded one-way data
   sets whose ANOVA solution was positive. The fit now evaluates trial points
   `theta_ii = 0.01, 0.1, 1` next to every zero diagonal and restarts the
   optimizer from a point that lowers the deviance (at most three restarts).
4. **Reported quantities.** `Cov(beta) = sigma^2 (R_X' R_X)^-1`, BLUPs
   `Lambda u` with lme4's conditional covariances `sigma^2 Lambda (L L')^-1
   Lambda'` [EQ-LMM-005], marginal and conditional fitted values and
   residuals, the log-likelihood (the REML criterion for REML fits), AIC and
   BIC with `k = p + len(theta) + 1`, and a `singular` flag (a diagonal of
   `T_k` below `1e-4`, lme4's rule). An empty term list fits the
   fixed-effects model, the null of a variance-component test.
5. **Satterthwaite degrees of freedom are implemented** (lmerTest's method):
   the Hessian of the deviance in `psi = (theta, sigma)` and the Jacobian of
   `Cov(beta)(psi)` by central differences, `nu = 2 V^2 / (g' (2 H^-1) g)`
   for each coefficient and for any contrast (`test_contrast`)
   [EQ-LMM-006]. When the Hessian is not positive definite the df are `nan`
   and the p-values fall back to the normal law; `denominator_df=
   "asymptotic"` requests z tests. Kenward-Roger is not implemented.
6. **Likelihood-ratio tests** [EQ-LMM-007] return the chi-square p-value,
   the 50:50 `chi^2_(df-1)`/`chi^2_df` boundary mixture (Self and Liang
   1987; Stram and Lee 1994; valid when the full model adds one random
   effect with its covariances), and an optional parametric bootstrap that
   simulates from the reduced fit and refits both models (seeded through
   `RandomStream.spawn`). REML fits with different fixed-effect designs are
   rejected.
7. **GLMMs** for binomial (logit, proportions with trials as weights) and
   Poisson (log) responses. The Laplace deviance uses penalized IRLS for the
   conditional mode with the canonical-link Hessian [EQ-GLMM-001] and
   supports any terms; adaptive Gauss-Hermite quadrature [EQ-GLMM-002]
   requires one scalar random effect per group, where the integral
   factorizes over groups. Nodes are Golub-Welsch eigenvalues refined by
   Newton steps with Christoffel weights (a private rule in the module:
   `modeling.gauss_hermite` lies in a higher layer). `theta` and `beta` are
   optimized jointly from the GLM fit; `Cov(beta)` is the `beta` block of
   twice the inverse numerical Hessian, with Wald z tests.
8. **Data.** The sleepstudy data (Belenky et al. 2003, distributed with
   lme4) are not bundled: lme4 is GPL-licensed while this library is BSD
   3-clause, and the published estimates could not be reproduced locally
   (no R installation). The oracles below use closed forms on balanced
   designs and simulated data.
9. **Guide** `docs/foundations/mixed-models.md` (learning-path step S16) and
   the registry records EQ-LMM-001 to EQ-LMM-008, EQ-GLMM-001, and
   EQ-GLMM-002.

## Consequences

- Grouped data with intercepts, correlated slopes, nested and crossed
  factors are fitted with the same estimates as lme4's `lmer` where closed
  forms exist, and Satterthwaite tests match lmerTest's construction.
- GLMM likelihoods are approximations; the guide shows the Laplace error and
  the convergence of adaptive quadrature. Random slopes in GLMMs use the
  Laplace approximation only.
- Dense storage of `Lambda' Z' W Z Lambda` is `O(q^2)` memory; very large
  numbers of levels need a sparse factorization, which is out of scope.
- No formula interface: designs come from `design_matrix` or raw rows, and
  terms from `random_effect`.

## Verification

`tests/test_mixed_models.py` (24 tests, about 12 s):

- Balanced one-way layout (9 groups of 6): REML `sigma^2 = MSE` and
  `sigma_b^2 = (MSA - MSE)/m`, ML `sigma_b^2 = (SSA/a - MSE)/m`, standard
  errors, ML and REML log-likelihoods, AIC, BIC, BLUPs, and conditional
  variances in closed form [EQ-LMM-008] to relative `1e-6` or better;
  Satterthwaite df `a - 1` to `1e-4`.
- Boundary: for 60 seeded data sets, `singular` is true exactly when
  `MSA <= MSE` (REML) or `SSA/a <= MSE` (ML), and boundary fits pool the sums
  of squares as in [EQ-LMM-008]. This is the regression test for the saddle
  of decision 3; without the restart 25 of the 60 cases fail.
- Balanced two-level nesting (7 x 3 x 4): REML equals the ANOVA estimators;
  df of the intercept `a - 1`.
- Balanced random coefficients (12 subjects, 6 common time points): ML and
  REML `Sigma_b = S - MSE (X'X)^-1` with divisor `a` or `a - 1`, the BLUPs
  `Sigma_b Omega^-1 (b_i - beta)`, and Satterthwaite df `a - 1` for both
  coefficients and the contrast `(1, 1)`.
- Unbalanced crossed design with a random slope: the maximized ML and REML
  log-likelihoods and GLS `beta` agree with a dense `N(X beta, V)`
  likelihood to 7 decimals; perturbing any variance parameter lowers the
  dense likelihood.
- `hierarchical_normal_posterior` with `s_j^2 = sigma^2/m`: at the REML
  `tau`, `mu | tau, y` is `N(beta, Var beta)` to `1e-7`, `E[theta_j]` is
  `beta + b_j`, and the gridded posterior mode of `tau` is within one grid
  step of the REML estimate.
- Bootstrap likelihood ratio: the fraction of null statistics equal to zero
  matches `P(F(a - 1, a(m - 1)) <= a/(a - 1))` within four standard errors
  (300 replicates).
- Coverage: 95% Satterthwaite intervals for intercept and slope in an
  unbalanced design cover within 3.5 Monte Carlo standard errors
  (300 replicates).
- Gauss-Hermite rules integrate `x^(2k) exp(-x^2)` exactly (relative
  `1e-12`) up to order 30.
- GLMM: adaptive quadrature converges to adaptive Gauss-Kronrod integration
  of every group integral (error below `1e-8` at 25 nodes, decreasing from
  1, 3, 9 nodes); one node equals the penalized IRLS Laplace deviance to 9
  decimals; `theta = 0` reproduces the binomial GLM log-likelihood with
  trials as weights; Laplace and quadrature fits agree within 0.2 standard
  errors; Poisson random intercepts and slopes recover the simulated fixed
  effects within four standard errors.
