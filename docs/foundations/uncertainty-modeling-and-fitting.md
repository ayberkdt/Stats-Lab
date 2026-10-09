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
universal p-value. Family-specific corrections and parametric bootstrap remain
explicit roadmap work.

## Reproducible laboratory

Run `python examples/statistical_fitting_lab.py` after installing visualization
support. It generates seeded lognormal data, fits log-space parameters, reports
KS/AD diagnostics and an approximate log-mean interval, and saves an empirical
versus fitted CDF figure. The output labels the fitted-null KS limitation rather
than hiding it.

## Sources

- [REF-NIST-DISTRIBUTIONS].
- [REF-NIST-GOF].
- [REF-CASELLA-BERGER-2002, Chapters 7 and 9].
- [REF-MARDIA-KENT-BIBBY-1979, Chapters 1-3].
- [REF-WILSON-1927].
