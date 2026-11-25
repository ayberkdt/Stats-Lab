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
   reports the Kolmogorov-Smirnov, the Cramer-von Mises and the
   Anderson-Darling p-value of `probstat.inference` against `chi2(n)` (fully
   specified null; see the amendment below for the Cramer-von Mises test) and the
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
     Poisson mixture of central chi-square values; the Poisson weights start
     at the mode (no cancellation) and recurse outward, summed with
     `math.fsum`. The error estimate is the neglected Poisson tail mass
     (incomplete gamma) plus an empirical rounding allowance, so it is not
     labelled rigorous. An earlier version computed the weights as
     `exp(-mu + j log mu - lgamma(j+1))`, whose cancellation made the true
     error exceed the reported bound (2.6e-12 against 9.7e-15 at
     noncentrality 1e5). The tests compare against 50-digit references.
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

Opened and read in the source verification pass (everything else below was
confirmed bibliographically at most):

- NASA Spacecraft Conjunction Assessment and Collision Avoidance Best
  Practices Handbook, NASA/SP-20230002470 Rev 1 (February 2023), Appendices I
  and M.4, and Appendix N (update of 20 March 2024) [REF-NASA-CARA-HANDBOOK].
  - *Covariance realism* (Appendix I): normalised errors, squared Mahalanobis
    distance against a chi-square law with three degrees of freedom for the
    position, mean as the first-moment check (expected value 3), and
    empirical-distribution-function tests. Agreement with
    `covariance_realism` (distances `e' P^-1 e` against `chi2(n)`, `n = 3`
    for a position). Differences: (1) the handbook recommends the
    Cramer-von Mises statistic for realism assessment because
    Anderson-Darling is more sensitive to outliers; `covariance_realism`
    defaults to Anderson-Darling on the argument that tails matter for
    collision probability, which is the opposite emphasis, and offers the
    handbook's Cramer-von Mises test as `test="cramer-von-mises"` (first
    version: not offered; see the amendment below); (2) the handbook reads a
    single scale factor from the first moment and names the square root of
    the normalised mean as that factor (its wording does not say whether it
    scales standard deviations or the covariance); this library's
    `s = mean(d^2) / n` is a factor on the covariance and the docstring now
    says that its square root is the factor on standard deviations; (3) the handbook gives no confidence interval for the factor
    and recommends resampling to a standard size of about 50 to keep the
    quadratic statistic from being overwhelmed at large samples; the interval
    here assumes independent pairs and nothing is resampled; (4) the handbook
    warns that points of one ephemeris pair are highly correlated (use one
    comparison point per propagation state) and that successive ephemerides
    should share less than half of their generating data, which the
    independence assumption of the interval needs; the docstring now repeats
    that advice. Thresholds (5, 2 and 1 percent p-values) are
    not encoded.
  - *Disc integral* (Appendix N, Eq. N-1): the density of the miss vector in
    the conjunction plane integrated over the hard-body disc, attributed to
    Foster and Estes; this is `gaussian_disc_probability`. Its Eq. N-2, the
    one-dimensional error-function form attributed to Alfano (2005a), is an
    independent oracle: the tests evaluate it by Simpson's rule and agree
    with the rotated quadrature to seven digits on four geometries
    (`test_matches_the_published_one_dimensional_form`).
  - *Probability dilution and maximum probability* (Section M.4): the term
    and the maximum probability over a manipulated covariance are attributed
    to Alfano (2005b), not to Alfano (2005a) as the first version of this
    record and of the references had it; corrected (new key
    REF-ALFANO-2005B for `maximum_disc_probability`).
- Zanetti, DeMars, Tuggle, Michaelson and Gupta, AAS preprint
  [REF-ZANETTI-ETAL-PREPRINT]: Table 3 is a published three-component library
  (Kullback-Leibler objective, mean and variance constraints) whose offset
  follows from this library's variance condition to nine digits; at its
  `sigma = 0.7844` the L2 optimum computed here has a relative L2 error of
  0.54 percent against 1.53 percent for the printed weights (and, at the
  default `sigma = 0.7`, 1.70 percent: the error depends on `sigma`, which is
  a design parameter). Table 2 reproduces the three-component library of
  DeMars, Bishop and Jah (2013): weights 0.2252, 0.5496, 0.2252, offsets plus
  or minus 1.0575, standard deviation 0.6716; as printed its second moment is
  0.9548, so it does not preserve the variance exactly, and its relative L2
  error is 1.48 percent. For the same `sigma` this library gives weights
  0.2130, 0.5741, 0.2130 with spacing 1.1353 and a 2.37 percent error, the
  difference being that the variance condition is imposed. The library is not
  replaced: the printed ones are not wrong for their purpose, but they are
  not moment-preserving, which `split_gaussian` relies on. The preprint also
  describes DeMars et al. (2013) as splitting along the direction of maximum
  uncertainty under a differential-entropy criterion, which agrees with the
  description of what was not implemented here.
- Huber, Bailey, Durrant-Whyte and Hanebeck (2008) [REF-HUBER-ETAL-2008],
  Table I: a four-component equal-variance library (weights 0.1274, 0.3726,
  0.3726, 0.1274, offsets plus or minus 0.4497 and plus or minus 1.4131,
  standard deviation 0.5175) taken from Hanebeck et al. (2003); second moment
  as printed 0.9273, relative L2 error 2.14 percent. Even component counts
  and unequal spacing are outside this library's family, so no direct
  comparison of weights is possible.
- Confirmed bibliographically only, not opened: DeMars, Bishop and Jah (2013),
  Vittaldev and Russell (2016) (the reference list of the preprint names a
  different 2016 paper by the same authors, in Computer Modeling in
  Engineering and Sciences), Horwood et al. (2014), Ruben (1962), Sheil and
  O'Muircheartaigh (1977), Foster and Estes (1992) (details from Appendix N),
  Alfano (2005a, 2005b) (details and roles from the handbook), Chan (2008)
  (the handbook says he reduces the integral to a series solution; the
  small-disc and Rice-law forms here are derived and no longer attributed to
  him), Sorenson and Alspach (1971) (details from the preprint's reference
  list), Bar-Shalom et al. (2001). Every formula was derived and is checked by
  an independent oracle (closed forms, R `qchisq`/`pchisq`/`qnorm` pins,
  brute-force quadrature, seeded Monte Carlo, and now the handbook's Eq. N-2).

## Consequences

- A user with a filter can answer the consistency question from its result
  object or from simulated runs, and gets a scale-factor estimate for
  inflating the covariance.
- The ball and disc functions are domain-neutral; the conjunction reading is
  in the docstrings. Anisotropic cases with very large condition numbers are
  best served by the quadrature for `n = 2`.
- The mixture record accepts positive-semidefinite component covariances;
  density and sampling need positive definite ones.
- New public names (functions and result records counted together, original
  decision only): 22 in `probstat.filtering` (15 functions and 7 result
  records) and 8 in `probstat.multivariate`. The Cramer-von Mises amendment
  below adds its names to `probstat.inference`.

## Amendment: the Cramer-von Mises goodness-of-fit test (2026-11-25)

The first version offered Kolmogorov-Smirnov and Anderson-Darling only,
because `probstat.inference.cramer_von_mises_statistic` carried no p-value.
`probstat.inference.cramer_von_mises_test(values, distribution)` now returns a
`CramerVonMisesResult` (statistic, sample size, p-value, method label), with
the call shape and record of `anderson_darling_test`, for a **fully specified**
continuous null (no estimated parameter; the docstring and the registry say so,
as for the neighbours).

- *Limit law* (`cramer_von_mises_asymptotic_cdf`, `..._sf`). The Anderson and
  Darling (1952) series in the modified Bessel function `K_(1/4)` for
  `x < 0.6`; the library has no Bessel function, so `K_nu` is evaluated by the
  trapezoidal rule on `int_0^inf exp(-z cosh t) cosh(nu t) dt` (relative
  agreement `1e-15` with mpmath's `besselk`). For `x >= 0.6` the tail is
  computed from its own alternating series of line integrals, written with an
  endpoint-regularising substitution so that the trapezoidal rule converges
  geometrically; that form keeps relative accuracy of `1e-14` to `x = 30`,
  where `1 - V` would have lost it. The two forms agree to `1e-15` where both
  are resolved, and the pinned values and the 0.95 and 0.99 points (0.461361,
  0.743459, matching the published 0.46136 and 0.74346) come from a 60-digit
  mpmath evaluation of the series, independently of this code. The mean `1/6`
  and second moment `1/20` of the law are reproduced by integrating the tail.
- *Finite n* (`cramer_von_mises_cdf`, `..._sf`). `n = 1` is exact
  (`2 sqrt(x - 1/12)`). For `n >= 2` the approximation `V + psi_1/n` of
  Csorgo and Faraway (1996, equations 1.8 and 1.10), error `O(n^-2)`. The
  paper was not opened; the formula was read from SciPy 1.14.1's
  `scipy.stats._hypotests` (which cites it), reimplemented here with log-gamma
  weights and a Bessel evaluation to `1e-18`, and SciPy is used only to pin
  values in the tests (agreement `1e-14` on the statistic and the pinned
  p-values within SciPy's own `1e-7`; the tests never import SciPy). The
  accuracy for small `n`, which the paper's value was not available to state,
  was measured by seeded simulation (three million samples per `n`): the
  upper-tail probabilities for `p <= 0.5` are within `0.01` at `n = 2`,
  `0.004` at `n = 3`, `0.003` at `n = 4` and `1e-3` from `n = 5`; at `n = 2`
  the far tail (`p = 1e-3`) is off by a factor of about nine. The docstrings
  carry these figures, and the tests re-measure them with a smaller seeded
  simulation judged on binomial standard errors plus the stated accuracy.
  The p-value is computed as `(1 - V) - psi_1/n` so the tail keeps its
  absolute accuracy; tails below `1e-12` are not resolved.
- *Realism.* `REALISM_TESTS` gains `"cramer-von-mises"`; the result record
  gains `cvm_p_value` (appended with a default, so existing positional
  construction still works) next to `ks_p_value` and `ad_p_value`; all three
  p-values are always computed. The default stays `"anderson-darling"`: the
  handbook's preference rests on outlier sensitivity, a judgement about the
  user's data rather than about the statistic, and changing the default would
  change the verdict of every existing caller. The docstring states the
  difference and how to select the handbook's choice. The handbook's 5, 2 and
  1 percent thresholds and its resampling to a standard sample size remain
  unencoded. Size and power are verified in `tests/test_filter_consistency.py`
  on the test's own scale: over 40 data sets of 100 pairs the rejection count
  at 5 percent under a correct covariance lies within five binomial standard
  errors of 2, and all 40 data sets with the covariance wrong by a factor of 2
  are rejected.
- *Left out:* the two-sample Cramer-von Mises test, exact finite-n tables
  (the paper's exact law was not opened), and a composite-null calibration
  (use `normality_tests` or `bootstrap_goodness_of_fit`).

## Left out

Entropy-driven splitting (DeMars), multidirectional splitting (Vittaldev and
Russell), non-equal-variance libraries with a fourth-moment constraint,
Imhof inversion, mixture-to-mixture collision probability, and any threshold
or policy of a handbook.

## Verification

`tests/test_filter_consistency.py`, `tests/test_gaussian_mixture_splitting.py`,
`tests/test_ball_probability.py`, `tests/test_cramer_von_mises_test.py`.
