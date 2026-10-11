# Uncertainty Modeling and Statistical Fitting

## Choose a variable before choosing a distribution

An uncertainty model begins with the physical quantity, units, admissible
support, data-generation process, and intended quantity of interest. A familiar
distribution name is not a substitute for those decisions.

- A Normal law permits every real value and is appropriate only when that
  support is defensible or negligible impossible-tail mass is documented.
- A Uniform law encodes constant density between known finite bounds; it does
  not mean “we know nothing.”
- A Lognormal law is strictly positive and uses parameters of `log(X)`, not the
  mean and standard deviation of `X`.
- A Weibull law is non-negative; its shape controls decreasing, constant, or
  increasing hazard behavior.

All four required ATHENS families now expose PDF/log-PDF, CDF, direct survival,
quantile, moments, and seeded sampling where defined. Parameterization is part
of the public contract and is recorded in `docs/equation-registry.md`.

## Multivariate uncertainty

For `X ~ N_d(mu, Sigma)`, the covariance matrix must be symmetric positive
definite for an ordinary density in `d` dimensions. `MultivariateNormal`
validates that condition by Cholesky factorization, evaluates Mahalanobis
distance without forming `Sigma^-1`, exposes univariate marginals and
correlations, and samples through `X = mu + LZ`.

Positive semidefinite but singular covariance is intentionally rejected. Such
a Gaussian law lives on a lower-dimensional affine subspace and needs a
different measure/density contract; silently adding jitter would alter the
model.

## Estimation is not uncertainty elimination

`fit_normal_mle` and `fit_lognormal_mle` implement closed-form maximum
likelihood estimates. For the normal scale, MLE uses divisor `n`, whereas the
usual unbiased sample-variance estimator uses `n-1`. An estimated parameter is
still uncertain; reporting only the fitted number discards that uncertainty.

`normal_mean_confidence_interval` distinguishes a known-scale z interval from
a large-sample approximation that inserts the sample standard deviation. The
latter is not presented as an exact Student-t interval. `wilson_score_interval`
provides a boundary-respecting interval for binomial probabilities, useful for
later reliability calculations.

## Goodness of fit

The one-sample Kolmogorov-Smirnov statistic measures the largest empirical-CDF
gap. The Anderson-Darling statistic weights tail discrepancies more strongly.
The implementation uses direct survival probabilities when a model provides
them, reducing upper-tail cancellation.

Both tests require careful calibration:

- The generic KS p-value is asymptotic and assumes a fully specified continuous
  null distribution.
- Fitting null parameters from the tested sample changes the KS null law.
- Anderson-Darling critical values and p-values depend on the null family and
  on whether parameters were estimated.

Therefore the generic AD result reports the statistic without inventing a
universal p-value. When parameters are estimated from the tested data, use
`bootstrap_goodness_of_fit` (next section), which calibrates both statistics by
a parametric bootstrap with refitting.

## Generic fitting, calibrated goodness of fit, and model selection

### One likelihood engine for every family

`fit_distribution(values, family)` fits any `DistributionFamily`: a
constructor, parameter kinds, a starting rule, and an optional closed form.
Parameters are optimized in unconstrained coordinates (`log` for positive
parameters, data-spread units for locations, a logistic map for a bounded
parameter) by BFGS on the negative log-likelihood built from `log_pdf`
(`[EQ-INF-MLE-GEN-001]`), with a Nelder-Mead fallback. Standard errors come
from the observed information: the central-difference Hessian in the
unconstrained coordinates, mapped back by the delta method
(`[EQ-INF-MLE-GEN-002]`). Each `FitResult` carries estimates, standard errors,
covariance, log-likelihood, AIC, AICc, BIC, and a convergence flag.

Ready-made families: Normal, LogNormal, Exponential, Gamma, Weibull (closed
form or profile likelihood), Gumbel, GEV, Logistic, location-scale Student t,
Beta on `(0, 1)` (numerical), and Uniform (closed form, non-regular: its
standard errors are reported as `nan`). The Student-t `nu` is bounded by
`1000`; on light-tailed data the estimate sits at the bound, its standard
error is `nan`, and the message says so.

Cross-checks (`tests/test_generic_fitting_model_selection.py`): starting the
optimizer away from the MLE, it reproduces the closed-form Normal,
LogNormal, and Exponential estimates and the profile Gamma/Weibull estimates
to `2e-6` relative. The exponential-rate standard error equals
`rate/sqrt(n)`, the normal ones equal `sigma/sqrt(n)` and `sigma/sqrt(2n)`,
and the gamma ones equal the inverse Fisher information.

### The Lilliefors/Stephens problem

The KS and AD reference laws assume the null model is fully specified. A
model fitted to the same data sits closer to it than the true law does, so
the ordinary p-values are too large. `bootstrap_goodness_of_fit` draws
`B` replicates from the fitted law, **refits every replicate with the same
estimator**, and reports `p = (1 + #{T* >= T})/(1 + B_ok)` for KS and AD
(`[EQ-INF-GOF-PB-001]`), together with the number of successful and failed
refits. The smallest attainable p-value is `1/(B+1)`. With `B = 19` nothing
can be rejected at `alpha = 0.05`, and `select_distribution` warns about this.

One seeded check: 40 normal samples (`n = 30`, `B = 49`, root seed 11),
each with a fitted normal null. The bootstrap KS p-values averaged `0.53`,
and `12.5 %` fell below `0.1`. The naive KS p-values against the same fitted
laws averaged `0.78`, which is far too conservative. An exponential sample of
`n = 100` fitted as normal gives bootstrap p-values of `0.01` for both KS and
AD (the minimum with `B = 99`).

`qq_points` and `pp_points` return `(theoretical, empirical)` pairs with
Hazen positions `(i - 1/2)/n` (`[EQ-INF-GOF-QQ-001]`) for plotting.

### Ranking candidates

`select_distribution(sample, candidates)` fits every admissible candidate and
ranks it by AIC (or AICc/BIC). It reports Akaike and BIC weights
`exp(-Delta/2)/sum exp(-Delta/2)` (`[EQ-INF-IC-004]`) and, when
`n_bootstrap > 0`, a calibrated KS/AD p-value per candidate. Candidates whose
domain excludes the data are listed as failures rather than raising an error.
A `ModelSelectionWarning` is emitted when the best model is still rejected.
Weights are relative to the candidate set; they never certify adequacy.

`examples/model_selection_lab.py` prints this table for 250 draws from
`Weibull(1.6, 40)` (seed 2026, `B = 99`):

| Rank | Family | AIC | dAIC | Akaike weight | KS p | AD p |
|---|---|---|---|---|---|---|
| 1 | Weibull | 2207.0 | 0.0 | 0.870 | 0.23 | 0.30 |
| 2 | Gumbel | 2211.4 | 4.5 | 0.094 | 0.57 | 0.27 |
| 3 | Gamma | 2213.3 | 6.3 | 0.036 | 0.01 | 0.04 |
| 4 | Normal | 2251.8 | 44.8 | 0.000 | 0.01 | 0.01 |
| 5 | LogNormal | 2267.5 | 60.6 | 0.000 | 0.01 | 0.01 |
| 6 | Exponential | 2300.8 | 93.9 | 0.000 | 0.01 | 0.01 |

The true family wins. The fitted shape is `1.732 (SE 0.085)`, so the truth
`1.6` lies about 1.6 SE away, and the fitted scale is `40.93 (SE 1.57)`. The table
also shows why ranking and testing answer different questions. The
runner-up Gumbel is not rejected by either test: 250 points cannot rule out
a Gumbel shape. Information criteria still prefer the Weibull by
`dAIC = 4.5`. Gamma has a reasonable AIC but is rejected by both tests at `0.05`.

### Joint inputs from data

`modeling.fit_joint_input(columns)` fits each named column, either with a
user-specified family or by `select_distribution`, and estimates a Gaussian
copula from ranks. The `"spearman"` option maps Spearman correlations by
`rho = 2 sin(pi r_s / 6)` (`[EQ-MOD-COPULA-FIT-001]`). The
`"normal-scores"` option correlates van der Waerden scores of the rank
pseudo-observations `R/(n+1)` (`[EQ-MOD-COPULA-FIT-002]`). The result is a
ready `JointInput`. Because the copula is estimated from ranks, it does not
depend on the marginal fits. A pairwise matrix that is not positive definite
is repaired by eigenvalue clipping, and the repair is flagged. With 800
draws of a three-input Weibull/LogNormal/Gumbel law and latent correlations
`(0.6, -0.3, 0.2)`, both estimators recover every entry within `0.06`.
`inference.fit_multivariate_normal_mle` gives the multivariate normal MLE
(divisor `n`).

Limitations: AIC/BIC for the non-regular Uniform are descriptive only. The
bootstrap with numerical refits costs `B` optimizations per candidate (about
0.02-0.1 s each in pure Python). The Gaussian copula cannot represent tail
dependence.

## Reproducible laboratory

Run `python examples/statistical_fitting_lab.py` after installing visualization
support (and `python examples/model_selection_lab.py` for the ranking above). It generates seeded lognormal data, fits log-space parameters, reports
KS/AD diagnostics and an approximate log-mean interval, and saves an empirical
versus fitted CDF figure. The output labels the fitted-null KS limitation rather
than hiding it.

## Sources

- [REF-NIST-DISTRIBUTIONS].
- [REF-NIST-GOF].
- [REF-CASELLA-BERGER-2002, Chapters 7 and 9].
- [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3].
- [REF-WILSON-1927].
- [REF-AKAIKE-1974], [REF-SCHWARZ-1978], [REF-HURVICH-TSAI-1989].
- [REF-BURNHAM-ANDERSON-2002, Chapter 2].
- [REF-LILLIEFORS-1967], [REF-STEPHENS-1974].
- [REF-DAVISON-HINKLEY-1997, Chapter 4].
- [REF-WILK-GNANADESIKAN-1968].
- [REF-GENEST-GHOUDI-RIVEST-1995], [REF-KRUSKAL-1958].
