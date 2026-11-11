# ADR-0047 — Missing-data methods: EM, data augmentation, Rubin's rules

- Status: Accepted
- Date: 2026-10-12
- Closes (roadmap): "missing-data policies remain" (multivariate normal part)

## Context

ADR-0040 added a generic EM driver and EM for censored normal data, and
explicitly left missing-at-random multivariate normal EM open. The library
had no missingness representation, no MCAR test, no multiple imputation and
no pooling rules.

## Decision

1. **One new layer-2 module, `probstat.inference.missing`.**
   - It imports from `foundations`, `linalg`, `distributions` and
     `optimization` (layers 0 and 1).
   - It imports `multivariate.MultivariateNormal`, from layer 2. That module
     does not import `inference`.
   - It also uses the sibling modules `em`, `bayesian.student_t_quantile`
     and `sampling_distributions.chi_square_sf`.

   The layer graph is unchanged and acyclic.
2. **Representation.** Plain rows with `None` or `nan` mark missing entries.
   There is no mask object. Rows are grouped by pattern once, and every
   pattern shares one Cholesky factor of `Sigma_oo` per iteration.
3. **EM reuses the generic driver.**
   - The E-step returns the conditionally expected rows and the summed Schur
     complements.
   - The M-step uses the same centred `fsum` formula as
     `fit_multivariate_normal_mle`, so complete data reproduce the MLE bit for
     bit.
   - The objective is the exact observed-data log-likelihood, so the
     monotonicity check is a real test.
4. **Standard errors from the full observed information.**
   - The Hessian is taken by central differences, in standardized
     coordinates, of the observed-data log-likelihood in `(mu, vech Sigma)`.
     It is then inverted, and the `mu` block is reported.
   - Rejected: the `mu`-block-only or expected information. It ignores the
     `mu`-`Sigma` cross information, which is non-zero under MAR.
   - Rejected for now: Louis's formula. It needs per-pattern fourth moments
     and would add code for no gain at this dimension.
5. **Proper imputation by data augmentation.**
   - The I-step draws from the conditional normal law.
   - The P-step draws from the normal-inverse-Wishart posterior. The default
     is the improper prior `|Sigma|^-(p+1)/2`, so `IW(A, n - 1)`.
   - Inverse-Wishart draws use the Bartlett decomposition on chi-square and
     normal variates from the call-local generator.
   - Each of the `m` imputations is an independent chain on
     `seed.child("imputation", k)`, started at the EM estimate. This keeps
     results reproducible and independent of `m`: imputation `k` is the same
     whether 5 or 20 are requested.
   - Rejected: imputing at the EM estimate (`burn_in = 0`). It is improper,
     because the between-imputation variance omits parameter uncertainty.
     It raises.
6. **Rubin's rules are generic.**
   - `pool_estimates` takes any scalar estimates and variances.
   - It reports `T`, `r`, `lambda`, and `gamma = ((nu + 1) lambda + 2)/(nu + 3)`
     with the reported `nu`.
   - It uses the Barnard-Rubin degrees of freedom when the complete-data
     degrees of freedom are given, and Rubin's `nu_old` otherwise.
7. **Biased baselines are kept and labelled.** `complete_cases` and
   `mean_impute` exist for comparison only. Their docstrings, the lab and the
   guide say they are biased under MAR, and the tests demonstrate it.

## Consequences

- `tests/test_missing_data.py` (17 tests, about 16 s) covers:
  - monotone EM and EM equal to the MLE without missing values;
  - recovery under MAR and complete-case bias;
  - Little's test under MCAR and MAR;
  - the inverse-Wishart mean;
  - the Rubin hand example in exact fractions;
  - MI, EM and complete-case coverage;
  - determinism.
- `examples/missing_data_lab.py` runs in about 43 s.
- Open:
  - fully conditional specification (chained equations);
  - categorical and mixed data;
  - MNAR models and sensitivity analysis;
  - multiparameter pooling;
  - automatic convergence diagnostics for data augmentation.
