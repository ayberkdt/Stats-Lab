# Missing Data: EM, Multiple Imputation, and Rubin's Rules

This guide covers `probstat.inference.missing`:

- representing incomplete rows and summarizing their patterns;
- Little's test of missing completely at random (MCAR);
- EM for the multivariate normal under missing at random (MAR), with
  observed-information standard errors;
- multiple imputation by data augmentation, with a Bartlett inverse-Wishart
  sampler;
- Rubin's combining rules with Barnard-Rubin degrees of freedom;
- two baselines that are biased under MAR: complete cases and mean
  imputation.

Every number below was printed by `examples/missing_data_lab.py` or by the
tests in `tests/test_missing_data.py`, with the seeds stated. They are the
library's own outputs. None is copied from the literature.

## 1. Mechanisms and what the methods assume

Write `R` for the missingness indicators. Rubin's taxonomy:

- **MCAR** — `R` is independent of the data;
- **MAR** — `R` depends only on observed values;
- **MNAR** — `R` depends on the missing values themselves.

EM and data augmentation maximize or sample the observed-data likelihood
([EQ-MISS-001]). This is valid ("ignorable") under MAR when the parameters of
the missingness mechanism are distinct from `(mu, Sigma)`. No test on the
observed data can tell MAR from MNAR. Little's test probes only the stronger
MCAR condition.

The lab's data: `(x0, x1, x2) ~ N(mu, Sigma)` with `mu = (1, 2, -1)`. `x0` is
always observed. `x1` is missing with probability
`logistic(-0.5 + 2 (x0 - 1))`, so large `x0` (and therefore large `x1`,
correlation 0.49) tends to be missing. That is MAR given `x0`. `x2` is
missing completely at random with probability 0.2.

## 2. Representation and patterns

Rows are equal-length sequences; a missing entry is `None` or `math.nan`.
Infinite values and booleans are rejected.

`missing_pattern(rows)` returns:

- the per-variable missing fraction;
- the distinct patterns, sorted by decreasing number of observed variables,
  with their counts;
- the complete-case count.

Lab, `n = 400`, seed 31:

| pattern (`o` observed) | rows |
|---|---|
| `ooo` | 165 |
| `oo.` | 55 |
| `o.o` | 143 |
| `o..` | 37 |

Missing fractions are `(0.000, 0.450, 0.230)`. Only 165 of 400 rows are
complete.

## 3. Little's MCAR test (EQ-MISS-004)

`little_mcar_test(rows)` compares each pattern's observed-variable means with
the EM estimate:

`d^2 = sum_j n_j (ybar_j - mu_o)' Sigma_oo^-1 (ybar_j - mu_o)`.

The test uses the maximum-likelihood `Sigma` and refers `d^2` to chi-square
on `sum_j p_j - p` degrees of freedom.

- Lab: `d^2 = 142.25` on 5 df, `p = 5.96e-29`. MCAR is rejected, as it should
  be, since `x1` depends on `x0`.
- Test, MAR, `n = 300`, seed 9: `d^2 = 120.74` on 5 df, `p = 2.2e-24`.
- Test, MCAR, 20 replications with `n = 150` (seeds 500..519): mean p-value
  0.552, and 2 of 20 below 0.05.

A large p-value does not establish MCAR, and the chi-square reference
assumes normality.

## 4. EM for the multivariate normal (EQ-MISS-002)

`fit_incomplete_multivariate_normal(rows)` works pattern by pattern.

**E-step.** With observed block `o` and missing block `m`, one Cholesky
factor of `Sigma_oo` gives

- `B = Sigma_mo Sigma_oo^-1`;
- the conditional mean `mu_m + B (x_o - mu_o)`;
- the Schur complement `Sigma_mm - B Sigma_om`, the conditional covariance.

**M-step.** `mu` is the mean of the completed rows. `Sigma` is their
divisor-`n` covariance plus the summed conditional covariances. Leaving out
that last term is the classic mistake: it would underestimate `Sigma` just as
single regression imputation does.

The generic `expectation_maximization` driver checks that the observed-data
log-likelihood never decreases.

Evidence:

- **Monotone.** Test, seed 3, `n = 300`: 44 iterations, from `-985.34` to
  `-947.30`, with no decrease.
- **Exact without missing values.** With complete data the EM estimate is
  bit-for-bit equal to `fit_multivariate_normal_mle` (`assertEqual` on both
  `mu` and `Sigma`). The M-step uses the same centred `fsum` formula.
- **Recovers the truth under MAR.** Test, seed 5, `n = 600`, 285 complete
  cases:

  | | `mu_0` | `mu_1` | `mu_2` |
  |---|---|---|---|
  | truth | 1.000 | 2.000 | -1.000 |
  | EM | 0.970 | 1.985 | -1.020 |
  | EM standard error | 0.040 | 0.067 | 0.039 |
  | complete cases | 0.515 | 1.755 | -1.143 |

  Complete cases shift every mean, because rows with large `x0` are
  under-represented. EM uses the `x0`-`x1` relation to correct the shift.

### Standard errors (EQ-MISS-003)

The covariance of `mu_hat` is the `mu` block of the inverse observed
information. That information matrix is the negative Hessian of
[EQ-MISS-001] in all parameters `(mu, vech Sigma)`. It is computed by central
differences in standardized coordinates, so one step size fits every scale.

The full inverse keeps the `mu`-`Sigma` cross information. That term is
non-zero under MAR. Inverting only the `mu` block of the expected information
can give wrong standard errors when the data are MAR but not MCAR.

With complete data the standard errors equal `sqrt(Sigma_jj / n)` to a
relative `1e-6` (test). For 9 parameters (`p = 3`) the Hessian costs about 160
log-likelihood evaluations. Pass `standard_errors=False` to skip it.

## 5. Multiple imputation by data augmentation (EQ-MISS-005, EQ-MISS-006)

`impute_multivariate_normal(rows, m, burn_in=50, seed=...)` alternates two
steps:

- **I-step** — draw every missing block from `N(mu_m + B(x_o - mu_o),
  Sigma_mm.o)`;
- **P-step** — draw `(mu, Sigma)` from the normal-inverse-Wishart posterior
  of the completed data.

The default prior is `p(mu, Sigma) ∝ |Sigma|^-(p+1)/2`, which gives
`Sigma | Y ~ IW(A, n - 1)` and `mu | Sigma, Y ~ N(ybar, Sigma / n)`. A proper
`NormalInverseWishartPrior(mean, kappa, scale, degrees_of_freedom)` can be
passed instead.

Each imputation `k` runs its own chain, on the child stream
`seed.child("imputation", k)`, started at the EM estimate. After `burn_in`
cycles its final I-step becomes data set `k`. The parameters are drawn rather
than fixed at the MLE, so the spread between imputations reflects parameter
uncertainty. That makes the imputations proper in Rubin's sense.

`sample_inverse_wishart(Psi, nu, size, seed=...)` uses the Bartlett
decomposition. `A` is lower triangular, with `A_ii^2 ~ chi^2(nu - i + 1)` and
standard normal entries below the diagonal. With `Psi = C C'` the draw is
`Sigma = C A^-T A^-1 C'`. Test: 8000 draws with `Psi = [[2, 0.5], [0.5, 1]]`
and `nu = 8` match `E[Sigma] = Psi / (nu - p - 1)` entrywise within four Monte
Carlo standard errors.

Determinism: the same integer seed and the equal `RandomStream` give
identical data sets. A different seed gives different ones. Observed entries
are never changed (test).

## 6. Rubin's rules (EQ-MISS-007, EQ-MISS-008)

`pool_estimates(estimates, variances, complete_data_df=None)` pools any
scalar estimate given its completed-data variance, for example a regression
coefficient and its squared standard error from `fit_linear_model`. It
returns:

- the pooled estimate and the within (`Ubar`), between (`B`) and total
  (`T = Ubar + (1 + 1/m) B`) variances;
- the relative increase in variance `r`;
- `lambda = (1 + 1/m) B / T`;
- the fraction of missing information `gamma`;
- the degrees of freedom and a t interval.

Without `complete_data_df` the degrees of freedom are Rubin's
`(m - 1)/lambda^2`. With it they are the Barnard-Rubin value, which never
exceeds the complete-data degrees of freedom.

Hand example (test), estimates `(10, 12, 11, 13, 9)` and variances
`(3.5, 4.5, 4, 3, 5)`:

| quantity | value |
|---|---|
| `Qbar` | 11 |
| `Ubar` | 4 |
| `B` | 2.5 |
| `T` | 7 |
| `r` | 0.75 |
| `lambda` | 3/7 |
| `nu_old` | 196/9 = 21.78 |
| `gamma` | 741/1561 = 0.4747 |
| `nu_obs` (with `nu_com = 20`) | 240/23 |
| Barnard-Rubin `nu` | 11760/1667 = 7.0546 |

Lab, slope of `x0` on `x1` (true 0.400), `m = 20`, seed 32: pooled 0.440, SE
0.041, 95% CI 0.358 to 0.522. Here `Ubar = 0.00115`, `B = 0.00054`,
`T = 0.00172`, and the Barnard-Rubin df is 105.9 (complete-data df 398).

## 7. Comparison on one sample (lab, seed 31)

Variable `x1`: true mean 2.000, variance 1.500, corr(`x0`, `x1`) 0.490.

| analysis | rows | mean | SE | var | corr |
|---|---|---|---|---|---|
| full data before deletion | 400 | 2.001 | 0.064 | 1.638 | 0.477 |
| complete cases (biased) | 165 | 1.764 | 0.096 | 1.518 | 0.525 |
| mean imputation (biased) | 400 | 1.723 | 0.046 | 0.848 | 0.314 |
| EM | 400 | 2.111 | 0.093 | 1.687 | 0.564 |
| MI, `m = 20` | 400 | 2.114 | 0.080 | 1.685 | 0.571 |

Mean imputation halves the variance (0.848 against 1.638 before deletion). It
attenuates the correlation, and it reports the smallest standard error of
all, smaller even than the full data. In this one sample EM and MI land about
1.2 standard errors above the truth. That is sampling noise, as the repeated
samples in section 8 show.

EM converged in 41 iterations, monotone. The MI pooling diagnostics were
`r = 0.519`, `lambda = 0.342`, FMI 0.355, and df 100.2.

The test `test_mean_imputation_shrinks_variance` (seed 21, 42.2% missing)
gives a mean-imputed variance of 0.778 against the EM estimate 1.446 (true
1.5).

## 8. Coverage over repeated samples

Nominal 95% intervals for `E[x1]`:

| interval | lab: 120 samples, `n = 80`, `m = 10`, burn-in 20 | mean error | test: 100 samples, bivariate, `n = 60`, `m = 10`, burn-in 10 |
|---|---|---|---|
| complete cases (t) | 0.692 | -0.308 | 0.59 |
| mean imputation (t) | 0.275 | -0.302 | — |
| EM Wald | 0.933 | 0.021 | 0.95 |
| MI with Rubin's rules | 0.933 | 0.018 | 0.92 |

The bivariate test design has `y` missing with probability
`logistic(-0.3 + 1.5 x)`: on average 44.6% missing and a fraction of missing
information of 0.48. With 100 replications the binomial standard error of a
coverage near 0.95 is about 0.022, so 0.92 and 0.93 are consistent with
nominal coverage. The test accepts MI coverage in `[0.89, 0.995]`.

## 9. Limitations

- Only the multivariate normal model. There is no chained-equations (FCS)
  imputation, no categorical or mixed data, and no MNAR selection or
  pattern-mixture models. Sensitivity analysis for MNAR is not provided.
- EM convergence is linear, with a rate set by the fraction of missing
  information. Heavy missingness needs many iterations. No acceleration
  (SQUAREM, PX-EM) is implemented.
- Data-augmentation chains start at the EM estimate. Convergence is not
  diagnosed automatically. The tests use a burn-in of 10 and the lab uses 20
  or 50. Heavier missingness needs more.
- Little's test uses the ML covariance and the asymptotic chi-square
  reference. It is not robust to non-normality.
- Rubin's rules are implemented for scalar estimands only. Multiparameter
  (Wald or likelihood-ratio) pooling is not included.
- The observed-information standard errors use a numerical Hessian. The
  number of log-likelihood evaluations grows like `p^4`, and each evaluation
  costs `O(n p^3)`, so skip them for large `p`.
