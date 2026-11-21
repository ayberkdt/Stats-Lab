# ADR-0119 — E-values, betting confidence sequences, universal inference, and prediction-powered inference

- Status: Accepted
- Date: 2026-11-21
- Extends: ADR-0101 (classical tests) and the multiple-testing and
  confidence-interval modules of `probstat.inference`

## Context

Every test and interval of `probstat.inference` assumed a sample size fixed
in advance. Monitoring a p-value or an interval while data arrive and
stopping at the first rejection inflates the error rate without bound, and
the library offered nothing that stays valid under such optional stopping.
Three further gaps were related:

- FDR control under arbitrary dependence existed only through
  Benjamini-Yekutieli and its harmonic penalty;
- composite nulls in irregular models (mixtures, boundary parameters) had no
  test with a finite-sample guarantee, because the likelihood-ratio tests in
  the library rely on chi-square limits;
- combining a small labelled sample with predictions of a machine-learning
  model on a large unlabelled sample had no estimator with a stated
  variance.

All four are covered by a small body of recent theory built on e-values and
test martingales, plus prediction-powered inference, which is classical
control-variate algebra with a new use.

## Decision

1. **Two new modules in `probstat.inference`** (layer 2, standard library
   only): `anytime_valid.py` and `prediction_powered.py`. They import
   `foundations` and the package's own `_hypothesis` and `confidence`
   helpers; the layer map is unchanged.
2. **E-values** ([EQ-INF-EVAL-001] to [EQ-INF-EVAL-003]):
   `calibrate_p_value` (the power calibrator `kappa p^(kappa - 1)` and, by
   default, its mixture over `kappa`, evaluated in a cancellation-free form
   with a series near `p = 1`), `e_to_p_value` (`min(1, 1/e)`), and
   `merge_e_values` (weighted mean; product for independent or sequential
   e-values). Weights that do not sum to one are rejected, not normalized.
   These return plain floats, like the tail-probability helpers of the
   package.
3. **e-BH** ([EQ-INF-EVAL-004]): `e_benjamini_hochberg` returns
   `EBenjaminiHochbergResult` with the rejection set, `k*`, and the e-value
   threshold `K / (alpha k*)`.
4. **Betting** ([EQ-INF-CS-001] to [EQ-INF-CS-004]): `betting_mean_test`
   returns the whole capital path (`BettingTestResult`: log capital, bets,
   final e-value, running maximum, anytime-valid p-value, first crossing
   time). One function covers the two-sided simple null, the one-sided
   composite nulls `mean <= m0` and `mean >= m0`, and general bounds
   `[lower, upper]`. `paired_betting_test` is the same test on differences
   in `[-B, B]`.
   - Bets: `"agrapa"` (default for tests), `"plug-in"`, `"fixed"`, always
     truncated to `c/m` and `c/(1 - m)` with `c = 0.5` by default, so every
     factor is at least `1 - c` and the log capital is finite.
   - Two-sided capital: one signed martingale for aGRAPA, the hedged
     `max(theta K^+, (1 - theta) K^-)` for the other rules.
   - Rejection uses `K >= 1/alpha` everywhere (Markov and Ville give the
     bound for the closed event).
5. **Confidence sequence** ([EQ-INF-CS-002]):
   `betting_confidence_sequence` evaluates the capital on a grid of
   `grid_size + 1` candidate means and applies the running intersection by
   default. For the plug-in and fixed rules the hedged capital is monotone
   in `m` on each side, so the reported endpoints are the outermost
   *rejected* grid points: the reported interval contains the exact set and
   the guarantee is kept without any appeal to grid resolution. For aGRAPA
   the set is not an interval in general; the hull of surviving grid points
   widened by one cell is reported and documented as a grid approximation.
   An empty set is reported through `first_empty_time` and NaN endpoints.
6. **Universal inference** ([EQ-INF-EVAL-005]):
   `split_likelihood_ratio_test` takes a joint log-likelihood and two
   fitting callables, splits by a seeded permutation (`shuffle=False` uses
   the given order), and optionally cross-fits. It works on the log scale
   and returns `SplitLikelihoodRatioResult` with the e-value, the p-value,
   and the number of likelihood evaluations and fits.
7. **Prediction-powered mean** ([EQ-INF-PPI-001] to [EQ-INF-PPI-003]):
   `prediction_powered_variance` and `prediction_powered_weight` are the
   population formulas; `prediction_powered_mean` returns
   `PredictionPoweredMean` with the estimate, standard error, a
   `ConfidenceInterval`, the weight, and the classical estimate. The tuned
   weight is the exact minimizer of the estimated variance
   `s^2(Y - w f)/n + w^2 s^2(F)/N`, so the tuned standard error is never
   above that of any fixed weight on the same data. The argument is called
   `weight` because `lambda` is reserved in Python.

## Consequences

- With `e_k = 1/p_k`, e-BH and `adjust_p_values(method="bh")` gave the same
  rejection set in 400 of 400 random families (1 to 25 hypotheses, four
  levels). For discrete null e-values the enumerated FDR was 0.01, 0.1667,
  and 0.1667 against the bound `alpha K_0 / K = 0.3333`; with identical
  null e-values equal to `1/alpha` with probability `alpha` the bound is
  attained.
- Exact enumeration over all `2^10` Bernoulli(`m`) sequences gives
  `E[K_10] = 1` to `4e-16` for the martingale capitals; the hedged maximum
  has expectation 0.818 in the tested case.
- Bernoulli(0.3), `alpha = 0.05`, 1000 replicates of 1000 observations
  (seed 2024): the capital at the true mean ever reached 20 in 4.2% (aGRAPA)
  and 3.4% (plug-in) of the runs; a Wilson interval recomputed after every
  observation from the tenth on excluded the truth at some time in 45.1%.
  With 2000 observations the figures are 4.2%, 3.4%, and 49%.
- One Bernoulli(0.3) run of 1000 observations: plug-in sequence
  `(0.190, 0.456)`, `(0.242, 0.398)`, `(0.242, 0.358)` after 100, 500, and
  1000 observations; aGRAPA `(0.216, 0.430)`, `(0.254, 0.394)`,
  `(0.254, 0.332)`; single Wilson intervals `(0.193, 0.364)`,
  `(0.275, 0.356)`, `(0.253, 0.309)`. Time-uniform coverage costs width.
  1000 observations on the default grid of 500 cells take about 0.15 s.
- Split likelihood ratio, `N(theta, 1)`, `H0: theta <= 0`, `n = 40`,
  cross-fit, `alpha = 0.1`: 1 rejection in 400 at `theta = 0` and 391 in 400
  at `theta = 0.8`. The test is valid but conservative at the boundary.
- Prediction-powered mean, bivariate normal `(Y, f)` with correlation 0.8,
  `n = 20`, `N = 60`, 4000 replicates: empirical variances 0.1996, 0.1165,
  0.1091, 0.2257, 0.3989 against the formula's 0.2000, 0.1175, 0.1100,
  0.2255, 0.3980 for weights 0, 0.5, 1, 1.7, -0.6. Coverage of the 90%
  interval over 2000 replicates (`n = 100`, `N = 400`): 0.894 (tuned),
  0.898 (weight 1), 0.890 (classical); mean variance ratio 0.491 against
  the predicted `1 - rho^2/(1 + n/N) = 0.488`.
- `probstat.inference` gains 16 public names.

## Rejected alternatives

- **Treating `1/p` as an e-value.** It reproduces BH algebraically but
  `E[1/P]` is infinite for a uniform p-value; only calibrated values carry
  the guarantees, and the guide says so.
- **Bisection for each endpoint of the confidence sequence at every time.**
  It costs `O(n^2 log(1/eps))`; the grid costs `O(n grid_size)` and, with
  outward rounding, keeps the guarantee.
- **Reporting the hull of surviving grid points for the hedged capital.**
  It can exclude a true mean lying between a rejected and a surviving grid
  point; taking the outermost rejected points cannot.
- **Clipping the tuned prediction-powered weight to `[0, 1]`.** It would
  discard the gain from anti-correlated predictions and break the property
  that the tuned estimated variance is minimal.
- **A pooled estimate of `Var(f)` in the tuned weight.** It is slightly
  more efficient but no longer the exact minimizer of the reported variance
  estimate.
- **Prediction-powered linear regression and quantiles.** The sandwich
  variance of the regression case and the non-smooth quantile case need
  their own derivations and simulations; they were left out of this work
  package rather than added without that verification.
- **Mixture (non-predictable) betting strategies and the empirical-Bernstein
  confidence sequence.** Useful, but each needs its own boundary formula
  checked against the source, which was not accessible while this was
  written.

## Verification

`tests/test_anytime_valid_inference.py` (44 tests, about 7 s) checks the
calibrators against quadrature over `p` and over `kappa`, the exact Markov
bound for calibrated uniforms, merging under dependence by enumeration,
e-BH against `adjust_p_values` and an enumerated FDR, the Bernoulli closed
form of the capital, unit expectation by enumeration for three bet rules,
supermartingale behaviour under a composite null, predictability of the
bets, Ville's bound and the Wilson contrast by seeded Monte Carlo, the
agreement of the confidence sequence with pointwise tests at every grid
point, nesting, grid refinement, affine equivariance, the empty-set flag,
the Gaussian closed form of the split likelihood ratio, its unit
conditional expectation by Gauss-Hermite quadrature in one and two
dimensions, the cross-fit average, level and power for a composite null,
reproducibility, and input validation.
`tests/test_prediction_powered_inference.py` (11 tests, about 5 s) checks
the variance formula and unbiasedness against simulation, the optimal
weight against a grid and the closed-form minimum, weight zero against
`normal_mean_confidence_interval`, weight one by hand, minimality of the
tuned standard error, convergence of the tuned weight, coverage, degenerate
predictions, and input validation.

The bet rules of [EQ-INF-CS-003] were written from the form of the
strategies in [REF-WAUDBY-SMITH-RAMDAS-2024] without access to the paper
(publisher and preprint hosts were unreachable); their constants are not
confirmed against the printed formulas. This does not affect validity,
which is verified by enumeration for each rule, and the registry record
states it.

## References

[REF-VOVK-WANG-2021], [REF-WANG-RAMDAS-2022],
[REF-WAUDBY-SMITH-RAMDAS-2024], [REF-HOWARD-ETAL-2021],
[REF-RAMDAS-ETAL-2023], [REF-WASSERMAN-RAMDAS-BALAKRISHNAN-2020],
[REF-ANGELOPOULOS-ETAL-2023], [REF-ANGELOPOULOS-DUCHI-ZRNIC-2023].
