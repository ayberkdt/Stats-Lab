# ADR-0140 — Covariance realism, Gaussian-mixture splitting, and the probability content of a ball

- Status: Accepted
- Date: 2026-11-24
- Extends: ADR-0132 (batch evaluation contract), ADR-0048 (API stability)

## Context

A code that propagates a state covariance (orbit determination, trajectory
estimation) raises three questions the library could not answer. Is the
filter's covariance consistent with its actual errors? What is the state
distribution after a long nonlinear propagation, when it is no longer
Gaussian? What is the probability that a two-dimensional Gaussian miss vector
falls inside a disc (the short-encounter collision probability)?

## Decision

1. **Consistency in `probstat.filtering.consistency`.** NEES per step and its
   Monte-Carlo average with chi-square bounds (`nees_test`), NIS per step and
   its time average (`nis_test`), helpers that read a `KalmanFilterResult`
   (`nees_from_filter`, `nees_test_from_filters`, `nis_test_from_filter`;
   the result stores innovations and innovation covariances, with `None` for a
   missing observation), an innovation whiteness test, and
   `covariance_realism` for (error, covariance) pairs. Verdicts are
   `"consistent"`, `"optimistic"` (covariance too small, statistic above the
   upper bound) and `"pessimistic"`. The per-step NEES bounds are exact; the
   verdict compares the time mean of the averages with the same interval,
   which is conservative because the variance of a mean of identically
   distributed steps does not exceed the per-step variance. Whiteness is
   judged by a Ljung-Box-weighted portmanteau statistic on the standardised
   innovations; counting lags outside a per-lag bound was tried and had too
   little power against one large lag-1 correlation. Covariance realism
   reports both the Kolmogorov-Smirnov and the Anderson-Darling p-value of
   `probstat.inference` against `chi2(n)` (fully specified null) and the
   scale factor `mean(d^2)/n` with a chi-square interval that assumes
   independent pairs.
2. **Splitting in `probstat.filtering.gaussian_mixture`.** The univariate
   library approximates `N(0, 1)` by an odd number of equally spaced
   components of equal standard deviation `s`. For a chosen `s` the weights
   minimise the integrated squared density difference (closed-form Gaussian
   product integrals) with BFGS from `probstat.optimization`; the spacing is
   fixed by `h^2 sum_i w_i i^2 = 1 - s^2` and the mean is zero by symmetry, so
   mean and variance are exact for every candidate (the unknowns are softmax
   logits, so positivity and the sum are also exact). `s` cannot be free: the
   L2 distance is zero for one component with `s = 1`, so `s` is a design
   parameter, with defaults 0.7, 0.5, 0.4 for 3, 5, 7 components (relative
   L2 error 1.7 %, 2.1 %, 1.3 %). Solutions for the defaults are stored as
   constants and re-derived in the tests. No published table is used.
   `split_gaussian` splits along any direction (default principal
   eigenvector) in whitened coordinates and preserves the mean and covariance
   exactly. The fourth moment and the density are not matched; the L2 error
   is stored in the record.
3. **Propagation.** `propagate_gaussian_mixture` splits, applies the existing
   `unscented_transform` to every component and returns a
   `GaussianMixtureState` with moment-matched mean and covariance, density and
   seeded sampling (the density and sampler reuse
   `probstat.inference.GaussianMixture`). Recursive splitting is driven by the
   *relative curvature index*, the largest norm of
   `(g(m+s_i)+g(m-s_i))/2 - g(m)` over the sigma axes divided by
   `sqrt(tr C_y)`: zero for affine maps, free of extra evaluations, and a
   design choice of this library (the cited entropy criterion of DeMars et
   al. was not implemented because the paper was not read). A refined
   component is split along its sigma axis of largest curvature. The
   procedure is written as a generator under ADR-0132
   (`propagate_gaussian_mixture_procedure`): one request per refinement level
   with the `2n + 1` sigma points of all pending components; the pointwise and
   batch wrappers share it. The unscented moments are computed by replaying the
   returned outputs through `unscented_transform`, so the sigma-point formulas
   exist once. The tests verify that for a quadratic map the mixture is closer
   to the exact moments than a single transform; this is a property of the test
   map, not a general guarantee.
4. **Ball probability in `probstat.multivariate.ball_probability`.**
   - `gaussian_disc_probability`: rotation to principal axes, outer
     quadrature (`u = R sin(theta)` removes the edge singularity, ranges cut at
     the mode and at the chord crossing, window `a1 +/- 40 s1`), inner chord in
     closed form with `erfc` for tail accuracy. Error: adaptive Gauss-Kronrod
     estimate, labelled not rigorous.
   - `isotropic_gaussian_ball_probability` and `noncentral_chi_square_cdf`:
     Poisson mixture of central chi-square values, bound from the neglected
     Poisson tails (incomplete gamma). Rigorous.
   - `gaussian_ball_probability` (any `n`): Ruben's chi-square mixture with
     `beta = lambda_min`. Non-negativity and unit sum of the weights are proven
     in the docstring and registry from the Laplace transform (every factor is
     a power series with non-negative coefficients), so truncation is bounded
     by `(1 - sum a_k) F_(n+2K)(R^2/beta)`. Weights are rescaled during the
     recursion to survive an offset of tens of sigma. The number of terms grows
     with the condition number and the Mahalanobis offset; at `max_terms` the
     result says `converged=False`. Imhof's numerical inversion was not added.
   - `small_disc_probability`: `pi R^2 f(0)` with the next term
     `(pi R^4/8) Laplacian f(0)` as the stated error (an estimate).
   - `maximum_disc_probability`: scan of `log c` followed by golden-section
     refinement, for a mean outside the disc; validated against the isotropic
     closed form `R^2/(e d^2)` and a dense scan. Uniqueness of the maximum for
     a strongly anisotropic covariance is not proven.

## Sources

The bibliographic details of DeMars, Bishop and Jah (2013) and of Ruben (1962)
were confirmed from repository and Project Euclid records, and those of
Horwood et al. (2014) from a citing page; no paper, handbook or standard was
read in full. Bar-Shalom et al. (2001), Sheil and O'Muircheartaigh (1977),
Foster and Estes (1992), Alfano (2005), Chan (2008), Huber et al. (2008),
Sorenson and Alspach (1971) and Vittaldev and Russell (2016) are cited from
memory or without confirmation, and the registry entries say so. Every formula
was derived and is checked by an independent oracle (closed forms, R
`qchisq`/`pchisq`/`qnorm` pins, brute-force quadrature, seeded Monte Carlo).
The NASA CARA handbook was not opened; no factor or threshold from it is used.

## Consequences

- A user with a filter can answer the consistency question from its result
  object or from simulated runs, and gets a scale-factor estimate for
  inflating the covariance.
- The ball and disc functions are domain-neutral; the conjunction reading is
  in the docstrings. Anisotropic cases with very large condition numbers are
  best served by the quadrature for `n = 2`.
- The mixture record accepts positive-semidefinite component covariances;
  density and sampling need positive definite ones.
- New public names: 15 in `probstat.filtering`, 8 in `probstat.multivariate`.

## Left out

Entropy-driven splitting (DeMars), multidirectional splitting (Vittaldev and
Russell), non-equal-variance libraries with a fourth-moment constraint,
Imhof inversion, mixture-to-mixture collision probability, and any threshold
or policy of a handbook.

## Verification

`tests/test_filter_consistency.py`, `tests/test_gaussian_mixture_splitting.py`,
`tests/test_ball_probability.py`.
