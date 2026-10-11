# Scoring, Calibrating, and Shift-Testing Probabilistic Forecasts

A probabilistic forecast states a whole distribution `F` for an outcome `y`.
Three questions decide whether it is any good:

1. **How good is it overall?** A proper scoring rule answers this with one
   number per case.
2. **Is it calibrated?** Do outcomes behave like draws from the forecasts?
   PIT histograms, interval coverage, and reliability diagrams answer this.
3. **Do the conditions still hold?** If the deployment data come from a
   different distribution than the data used to build or calibrate the
   forecast, past calibration says little about the future. Two-sample
   tests answer this.

The tools live in `probstat.validation.scoring`,
`probstat.validation.calibration`, and `probstat.validation.shift`. The
two-sample Kolmogorov-Smirnov test sits next to the one-sample test in
`probstat.inference`. Every number below was printed by
`examples/forecast_calibration_lab.py` or computed in
`tests/test_forecast_calibration.py`.

## 1. Proper scoring rules

A score `S(F, y)` is negatively oriented: smaller is better. It is *proper*
if the expected score under the true law `G` is minimized by reporting
`F = G`. A forecaster then has no reason to report anything other than
their actual belief.

| Score | Function | Needs | Notes |
|---|---|---|---|
| CRPS (EQ-VAL-SCORE-CRPS-001) | `crps_ensemble`, `crps_normal`, ..., `crps_distribution` | CDF or ensemble | In the unit of `y`; equals the absolute error for a point forecast |
| Log score (EQ-VAL-SCORE-LOG-001) | `log_score` | Density or mass | Infinite where the forecast gives zero density |
| Dawid-Sebastiani (EQ-VAL-SCORE-DSS-001) | `dawid_sebastiani_score` | Mean and variance | Proper only with respect to the first two moments |
| Energy (EQ-VAL-SCORE-ES-001) | `energy_score` | Multivariate ensemble | Equals the ensemble CRPS in one dimension |
| Variogram (EQ-VAL-SCORE-VS-001) | `variogram_score` | Multivariate ensemble | Targets dependence; not strictly proper |
| Interval / WIS (EQ-VAL-SCORE-IS-001, -WIS-001) | `interval_score`, `weighted_interval_score` | Quantiles | WIS is twice the mean pinball loss over its levels |
| Pinball (EQ-VAL-SCORE-QS-001) | `pinball_loss` | One quantile | `2 int_0^1 QS_tau dtau = CRPS` |
| Brier (EQ-VAL-SCORE-BS-001) | `brier_score`, `brier_decomposition` | Event probability | Murphy decomposition with within-bin terms |

### The CRPS of an ensemble

For an ensemble `x_1..x_m`, `crps_ensemble` uses the kernel form
`(1/m) sum |x_i - y| - (1/(2m^2)) sum_ij |x_i - x_j|`. It computes the
double sum from the sorted members in `O(m log m)`. Members are shifted by
`y` first, so a common offset of `1e9` changes the result by less than
`1e-6` (asserted).

In the tests, small integer ensembles (with ties, and with `y` outside the
ensemble) match the exact rational integral of `(F_m(x) - 1{x >= y})^2` to
13 decimal places.

The ordinary ensemble CRPS is biased upward by `E|X - X'|/(2m)`.
`crps_ensemble(..., fair=True)` uses `2m(m - 1)` instead and is unbiased.
With 40 ensembles of 400 draws from `Normal(1, 2)` scored at `y = 0.4`:

| Quantity | Value |
|---|---|
| Closed-form `crps_normal(0.4, 1, 2)` | 0.538666 |
| Mean fair ensemble CRPS (40 replicates) | 0.538897, SE 0.005312 |
| Mean plain minus fair | 0.002813 |
| Predicted bias `(2 sigma/sqrt(pi))/(2m)` | 0.002821 |

### Closed forms and quadrature

`crps_distribution` uses closed forms for `Normal`, `LogNormal`, `Logistic`,
`Exponential`, and `Uniform`. For any other law with `cdf` and `quantile`,
it uses `crps_quadrature`, which integrates the quantile-score
representation (EQ-VAL-SCORE-CRPS-009) with tanh-sinh quadrature, split at
`F(y)`.

The tests check each closed form two ways: against composite Simpson
integration of the definition (split at `y` and at support kinks), and
against the quadrature path. Largest absolute differences over the tested
observations:

| Family | vs Simpson on the definition | vs quantile quadrature |
|---|---|---|
| Normal(1, 2) | 8.5e-13 | 8.9e-16 |
| LogNormal(0.2, 0.5) | 3.9e-09 | 2.7e-15 |
| Logistic(1, 0.7) | 6.5e-11 | 8.9e-16 |
| Exponential(1.5) | 1.5e-09 | 1.3e-15 |
| Uniform(-1, 2) | 0 | 1.6e-15 |

The Simpson column is limited by the Simpson rule itself (4000 panels per
piece and a finite range), not by the closed forms. Weibull and gamma
forecasts, which have no closed form here, agree with Simpson to `1e-8`.

## 2. Calibration and sharpness

`pit_values` returns `u = F(y)` for distribution objects. For ensembles,
`ensemble_pit` returns the randomized rank `(r + V(s + 1))/(m + 1)`, where
`r` counts members below `y`, `s` counts ties, and `V ~ U(0, 1)`. If `y` is
exchangeable with the members, `u` is exactly uniform, ties included. The
plain value `r/m` is never uniform for a finite ensemble. A test with
integer ensembles full of ties passes both the chi-square and the KS
uniformity check.

`pit_histogram` bins the values, runs Pearson's chi-square test against
equal counts, and reports two shape indices:

- `edge_ratio` is the outer-bin share relative to uniform. Above 1 means a
  U shape (forecasts too narrow); below 1 means a hump (forecasts too wide).
- `mean` is the mean bin centre. Values away from 0.5 indicate bias.

`pit_coverage` reads the coverage of central intervals at several levels
from the same PIT values, and attaches a Clopper-Pearson or Wilson interval
to each rate. On 1000 seeded cases with `y ~ N(m_i, 1)`:

| Forecast | PIT chi-square p | Edge ratio | Coverage at 50 / 80 / 90 / 95 % |
|---|---|---|---|
| `N(m_i, 1)` (calibrated) | 0.589 | 0.92 | 0.494 / 0.817 / 0.910 / 0.956 |
| `N(m_i, 0.5)` (overconfident) | 3.0e-148 | 2.68 | 0.252 / 0.464 / 0.583 / 0.671 |

Calibration is necessary but cheap: a climatological forecast can be
calibrated. `sharpness_summary` therefore reports mean and median
central-interval widths and the mean predictive standard deviation. The
goal is the sharpest forecast that is still calibrated.

## 3. The laboratory: four forecasters

`examples/forecast_calibration_lab.py` draws 2000 cases with
`Y_i ~ N(m_i, s_i)`, `m_i ~ U(-2, 2)`, `s_i ~ U(0.5, 1.5)` (seed 20261012).
It runs in 7 to 9 s and prints four numbered sections and a closing
sentence.

| Forecaster | CRPS | LogS | DSS | PIT p | Edge | Cov 50 | Cov 90 | Width 90 |
|---|---|---|---|---|---|---|---|---|
| ideal `N(m_i, s_i)` | 0.5664 | 1.3744 | 0.9110 | 0.942 | 1.02 | 0.494 | 0.897 | 3.287 |
| overconfident `N(m_i, s_i/2)` | 0.6137 | 2.1867 | 2.5356 | 4.4e-283 | 2.62 | 0.255 | 0.585 | 1.644 |
| climatology (one normal) | 0.8819 | 1.8541 | 1.8703 | 7.5e-05 | 1.00 | 0.458 | 0.910 | 5.083 |
| 20-member ensemble of the ideal law | 0.5958 | — | 1.0273 | 0.0246 | 1.09 | 0.500 | 0.900 | 2.844 |

How to read the table:

- **All three scores rank the ideal forecast first.** The log score and DSS
  punish the overconfident forecast far more than the CRPS does, because
  they grow without bound for outcomes in the tails.
- **Climatology** has the right overall spread (edge ratio 1.00, 90 %
  coverage 0.910). Its PIT is still not uniform (`p = 7.5e-5`): the marginal
  law of `Y` is a mixture of normals and not itself normal. It is also by
  far the least sharp forecast.
- **The ensemble's plain CRPS (0.5958) is worse than the ideal law's
  (0.5664)** only because of the finite-ensemble bias. Its fair CRPS is
  0.5677. The ensemble's PIT `p = 0.0246` is a chance result. Across 30
  other seeds of the same setting, the chi-square p-values ranged from 0.034
  to 0.947 and looked uniform.

For the ideal law, the weighted interval score from the 50/80/95 % intervals
and the median is 0.4301, against a CRPS of 0.5664. Three intervals
approximate the CRPS only roughly.

### Binary event `Y > 0`

`brier_decomposition(..., bins=10)` reports the five terms. The identity
`BS = REL - RES + UNC + WBV - WBC` closes to rounding (largest residual
`5.6e-17`):

| Forecaster | BS | REL | RES | UNC | WBV | WBC |
|---|---|---|---|---|---|---|
| ideal | 0.1313 | 0.0001 | 0.1181 | 0.2500 | 0.0009 | 0.0016 |
| overconfident | 0.1423 | 0.0096 | 0.1153 | 0.2500 | 0.0007 | 0.0027 |
| climatology | 0.2500 | 0.0000 | 0.0000 | 0.2500 | 0.0000 | 0.0000 |

The within-bin terms (EQ-VAL-SCORE-BSDEC-002, Stephenson et al. 2008) are
needed whenever a bin holds different forecast values. On 2000 raw
forecasts in 10 bins, the classical three-term sum misses the score by more
than `1e-4`; with the two extra terms the residual is below `1e-15`. If
every bin holds a single forecast value, both extra terms are zero and the
three-term identity is exact (asserted to `1e-15`).

`reliability_diagram(..., consistency_level=0.95)` attaches a band to each
bin. The band shows where the observed frequency would fall if the
forecasts were reliable; it is computed by redrawing outcomes as
`Bernoulli(f_i)` for the forecasts in that bin. With 5 bins, the ideal
forecast stays inside every band. The overconfident forecast leaves the
band in bins 0, 1, 3, and 4. For example, in `[0, 0.2)` its mean forecast is
0.034 but the observed frequency is 0.122, against a band of
`[0.022, 0.048]`.

## 4. Distribution shift

| Test | Function | Statistic | Null calibration |
|---|---|---|---|
| Two-sample KS | `inference.kolmogorov_smirnov_two_sample_test` | `max|F_m - G_n|` | Exact lattice-path count (conditional on ties) for `mn <= 250000`, else limiting law |
| Energy distance | `energy_distance_test` | `(mn/(m+n)) E` | Permutations, `(1 + b)/(1 + B)` |
| Kernel MMD | `mmd_test` | Unbiased `MMD^2`, Gaussian kernel, median bandwidth | Permutations |
| Wasserstein | `wasserstein_distance`, `distribution_wasserstein_distance` | `W_p` | Distance only, no test |

The exact KS p-value counts lattice paths with exact Python integers. On
small samples with ties, it matches brute-force enumeration of all
relabelings to 15 decimal places. Two fully separated samples of size 3
give exactly `2/C(6, 3) = 0.1`.

Permutation tests are valid for any number of relabelings. Under the null
(60 replicates, `m = n = 20`, 99 permutations), the energy-test p-values had
mean 0.541 and KS uniformity `p = 0.43`; the MMD p-values (40 replicates)
had mean 0.557 and KS `p = 0.15`. With a unit mean shift and `n = 50` per
sample, the energy, MMD, and KS p-values were 0.010, 0.005, and 0.0058.

In the laboratory, 300 calibration outcomes are compared with 300
deployment outcomes. The marginal standard deviation of the calibration
outcomes is 1.608.

| Scenario | KS p | Energy p | MMD p | W1 | W2 |
|---|---|---|---|---|---|
| no shift | 0.988 | 0.93 | 0.94 | 0.102 | 0.159 |
| +0.4 | 0.293 | 0.068 | 0.155 | 0.290 | 0.320 |
| +0.8 | 1.22e-08 | 0.002 | 0.005 | 0.959 | 0.971 |

A shift of a quarter of a marginal standard deviation goes **undetected** at
`n = 300` by all three tests. A non-significant shift test is therefore not
evidence that no shift occurred. The Wasserstein distances report the
magnitude of the change in the unit of `y`, which is often more useful than
a p-value.

`distribution_wasserstein_distance` integrates `|Q_1(u) - Q_2(u)|^p` over
`u` with tanh-sinh quadrature. It reproduces `W_2(N(0,1), N(1,2)) = sqrt 2`
to `7e-14` and `W_2^2(U(0,1), Exp(1)) = 5/6` to `2e-12`. For `p = 1`,
`|.|` has a kink wherever the two quantile functions cross, so convergence
slows. `W_1(N(0,1), N(1,2))` agrees with the closed form `E|1 + Z|` to
`1.5e-7`, after 6333 integrand evaluations.

## 5. Limitations

- The chi-square PIT test, the binomial coverage intervals, and the
  permutation tests all assume **independent cases** (exchangeable under the
  null). Serially dependent forecasts, such as overlapping lead times, make
  them anti-conservative.
- A uniform PIT histogram does not prove calibration conditional on
  covariates. Conditional biases can cancel (Hamill 2001).
- `pit_values` assumes continuous forecast laws. Randomization is
  implemented only for ensembles; discrete laws would need `F(y-)`.
- Reliability bands keep each bin's forecasts fixed. This is a
  simplification of Bröcker-Smith consistency resampling, which also
  resamples the forecast values.
- The MMD uses a single Gaussian kernel with the median-heuristic bandwidth.
  Power depends on that choice, and no bandwidth optimization is attempted.
- Permutation tests on vectors cost `O(m^2)` per relabeling after one
  `O(N^2)` distance matrix. This suits samples of a few hundred points.
