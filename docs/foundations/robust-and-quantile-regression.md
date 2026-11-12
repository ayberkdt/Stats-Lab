# Robust Estimation and Quantile Regression

The mean and least squares answer the question "what fits best on average
in squared error?". One gross error can move that answer anywhere. This
guide covers two ways out, both in `probstat.inference`:

- **Robust M-estimation** keeps the target (the centre of the bulk of the
  data) but bounds the influence of each observation.
- **Quantile regression** changes the target: it models a chosen quantile
  of the response, such as the median or the 90th percentile, which is
  robust to outlying responses by construction and describes how the whole
  conditional distribution moves.

The numbers below are computed by the snippets, which run in
`tests/test_documentation_examples.py`; the oracles behind the methods are
in `tests/test_robust_estimation.py` and `tests/test_quantile_regression.py`
(decision record ADR-0101).

## 1. Which method when

| Situation | Method | Function |
|---|---|---|
| Location of a sample with a few wild values | Huber M-estimate (MAD scale) | `robust_location` |
| Same, gross errors should be ignored entirely | Tukey biweight | `robust_location(psi="bisquare")` |
| Regression with outlying responses, ordinary design | Huber regression by IRLS | `fit_robust_linear_model` |
| Same, a few responses are far off | Biweight regression (robust start) | `fit_robust_linear_model(psi="bisquare")` |
| One predictor, outliers in `x` as well as `y` | Theil-Sen slope | `theil_sen_slope` |
| The median, or another quantile, is the quantity of interest | Linear quantile regression | `fit_quantile_regression` |
| Spread of the response changes with the predictors | Several regression quantiles | `fit_quantile_regression(quantile=tau)` |
| Normal errors, no outliers | Least squares | `fit_linear_model` |

Two properties pull in different directions:

- **Breakdown point**: the largest fraction of arbitrarily bad observations
  an estimator can absorb before it can be moved arbitrarily far. The mean
  and least squares have breakdown point `1/n`; the median has `1/2`.
- **Efficiency**: the variance at the normal law relative to the best
  estimator. The median has efficiency `2/pi = 0.64`; the Huber and
  biweight constants below are chosen for 95%.

## 2. Score functions (EQ-INF-ROB-001)

An M-estimator solves `sum_i psi(r_i / s) = 0` for standardized residuals.

| Family | `psi(u)` | Default constant | Property |
|---|---|---|---|
| Huber | `max(-k, min(k, u))` | `k = 1.345` (`HUBER_TUNING`) | monotone, unique root |
| Tukey biweight | `u (1 - (u/c)^2)^2` for `abs(u) < c`, else 0 | `c = 4.685` (`BISQUARE_TUNING`) | redescending, weight 0 beyond `c` |

The scale `s` is the median absolute deviation times `1.4826`
(EQ-STAT-MAD-001), which has breakdown point one half. Both estimators are
computed by iteratively reweighted least squares (IRLS) with weights
`w(u) = psi(u)/u`.

## 3. Robust location (EQ-INF-ROB-002)

Ten readings, one of them a transcription error:

```python
import statistics
from probstat.inference import robust_location

sample = (2.1, 2.4, 2.2, 2.3, 2.0, 9.5, 2.25, 2.15, 1.95, 2.35)
statistics.fmean(sample)                 # 2.92, pulled up by one value
statistics.median(sample)                # 2.225

huber = robust_location(sample)
huber.estimate                           # 2.21875
huber.scale                              # 0.1853, the normalized MAD
huber.standard_error                     # 0.0657, sandwich
huber.weights[5]                         # 0.034, the outlier is down-weighted
huber.confidence_interval.lower          # 2.070
huber.confidence_interval.upper          # 2.367

biweight = robust_location(sample, psi="bisquare")
biweight.estimate                        # 2.1909
biweight.weights[5]                      # 0.0, the outlier is rejected
```

The standard error is the empirical sandwich
`s sqrt(n/(n-1) sum psi^2) / sum psi'`, a large-sample value that treats
the MAD as known. In the test suite, 400 seeded normal samples of size 40
give 95% intervals whose coverage lies within three binomial standard
errors of 0.95.

## 4. Robust regression (EQ-INF-ROB-003)

Brownlee's stack-loss data (21 days of a plant oxidizing ammonia;
[REF-R-DATASETS-STACKLOSS]) are the classic test bed: a few days do not
follow the pattern of the others.

```python
from probstat.inference import fit_linear_model, fit_robust_linear_model

air = (80, 80, 75, 62, 62, 62, 62, 62, 58, 58, 58, 58, 58, 58, 50, 50, 50, 50,
       50, 56, 70)
water = (27, 27, 25, 24, 22, 23, 24, 24, 23, 18, 18, 17, 18, 19, 18, 18, 19,
         19, 20, 20, 20)
acid = (89, 88, 90, 87, 87, 87, 93, 93, 87, 80, 89, 88, 82, 93, 89, 86, 72,
        79, 80, 82, 91)
loss = (42, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9,
        15, 15)
design = [(1.0, a, w, c) for a, w, c in zip(air, water, acid)]
names = ("(Intercept)", "air", "water", "acid")

ols = fit_linear_model(design, loss, column_names=names)
ols.coefficients[2]                      # 1.2953

huber = fit_robust_linear_model(design, loss, column_names=names)
huber.coefficient("air")                 # 0.8294
huber.coefficient("water")               # 0.9261
huber.scale                              # 2.441
huber.standard_errors[2]                 # 0.3784
huber.start                              # 'ls'

biweight = fit_robust_linear_model(design, loss, psi="bisquare",
                                   column_names=names)
biweight.coefficient("water")            # 0.6507
biweight.start                           # 'lad', a robust start
biweight.weights[20]                     # 0.0022, day 21 is rejected
biweight.weights[3]                      # 0.336
```

The Huber fit starts from least squares because its objective is convex;
the biweight objective is not, so its root depends on the start, and the
default is the median (least absolute deviations) regression of section 6.
The water-temperature coefficient drops from 1.30 (least squares) to 0.93
(Huber) and 0.65 (biweight) as the influence of the unusual days is
reduced. The coefficient intervals use the sandwich covariance
`s^2 n/(n-p) A^-1 B A^-1` with `A = sum psi'(u_i) x_i x_i'` and
`B = sum psi(u_i)^2 x_i x_i'`.

## 5. Theil-Sen slope (EQ-INF-ROB-004)

Regression M-estimators bound the influence of a residual, not of a design
point: one observation far out in `x` can still pull the line to itself
(breakdown point `1/n` for leverage). For a single predictor the median of
all pairwise slopes resists both. A pair of good points gives a correct
slope, and good pairs are a fraction `(1 - eps)^2` of all pairs when a
fraction `eps` of the points is bad, so the median stays bounded while
`eps < 1 - 1/sqrt(2)`, about 29%.

```python
from probstat.inference import (
    fit_robust_linear_model, fit_simple_linear_regression, theil_sen_slope,
)

x = [float(i) for i in range(15)] + [100.0] * 5
y = [3.0 * v + 1.0 + (0.1 if i % 2 else -0.1) for i, v in enumerate(x[:15])]
y += [-1000.0] * 5                       # a quarter of the points, far away

fit_simple_linear_regression(x, y).slope                        # -10.869
fit_robust_linear_model([(1.0, v) for v in x], y).coefficients[1]  # -10.874
theil_sen_slope(x, y).slope                                     # 2.96
```

`theil_sen_slope` also returns Sen's distribution-free interval, which
inverts Kendall's test of the slope and corrects its variance for ties in
`x`. Here it is very wide (`(-10.4, 3.0)`) because a quarter of the pairwise
slopes involve the planted points; the point estimate is robust, the
interval honestly reports the contamination.

## 6. Quantile regression (EQ-INF-QR-001, EQ-INF-QR-002)

The `tau`-th regression quantile minimizes the check loss
`sum_i rho_tau(y_i - x_i' beta)`, `rho_tau(r) = r (tau - 1{r < 0})`.
`tau = 0.5` is least absolute deviations. The problem is a linear program
with an optimal vertex that interpolates `p` observations; the solver moves
between such elemental bases in the manner of Barrodale and Roberts and
certifies optimality at the end, including at degenerate vertices.

```python
from probstat.inference import fit_quantile_regression

median = fit_quantile_regression(design, loss, column_names=names)
median.coefficient("air")                # 0.8319
median.coefficient("water")              # 0.5739
median.objective                         # 21.0406
median.basis                             # (1, 7, 15, 17)
sum(r < 0 for r in median.residuals)     # 9
sum(r == 0 for r in median.residuals)    # 4, the interpolated days

upper = fit_quantile_regression(design, loss, quantile=0.75, column_names=names)
upper.coefficient("water")               # 0.9828
lower = fit_quantile_regression(design, loss, quantile=0.25, column_names=names)
lower.coefficient("water")               # 1.0

boot = fit_quantile_regression(design, loss, column_names=names,
                               interval="bootstrap", replicates=200, seed=11)
boot.confidence_intervals[1].lower       # 0.375
boot.confidence_intervals[1].upper       # 1.273
boot.replicates                          # 200
```

At a solution with an intercept, the number of negative residuals is at
most `n tau` and the number of non-positive ones at least `n tau`: here 9
negative and 4 zero residuals bracket `21 x 0.5 = 10.5`. The bootstrap
resamples the `(x_i, y_i)` pairs, so it remains valid when the spread of
the response depends on the predictors; the interval is a percentile
interval and needs a few hundred replicates.

## Limitations

- Standard errors of M-estimators are large-sample sandwich values that
  ignore the variability of the MAD scale; for very small `n` they are
  optimistic.
- M-regression is not high-breakdown against leverage points; MM- and
  S-estimators (with breakdown point one half in regression) are not
  implemented. Theil-Sen covers the single-predictor case.
- The quantile-regression solver is a dense pure-Python simplex walk; it is
  exact and handles a few thousand observations with a handful of
  predictors. Rank-inversion intervals are not implemented; intervals come
  from the pairs bootstrap.
- Regression quantiles fitted separately for several `tau` can cross at
  extreme design points.

## Common pitfalls

- **Breakdown point versus efficiency.** The Huber estimate keeps 95%
  efficiency at the normal law, but in regression one bad design point
  breaks it (section 5); use `theil_sen_slope` or inspect leverage first.
- **Start of the biweight.** `psi="bisquare"` has several roots;
  `fit_robust_linear_model` starts from median regression
  (`start="auto"` resolves to `"lad"`), and `start="ls"` can land on a
  different root when outliers dominate least squares.
- **A zero scale.** If more than half of the observations coincide (or
  more than half of the residuals vanish), the MAD is zero and both
  functions raise `ValueError`; pass `scale=` to `robust_location` when the
  scale is known.
- **Non-unique quantile solutions.** When `n tau` is an integer the set of
  minimizers can be an interval (the median of an even sample), and one
  optimal vertex is returned; compare objectives, not coefficients, across
  solvers.
- **Bootstrap seeds.** `fit_quantile_regression(interval="bootstrap")` is
  reproducible only with an explicit `seed`.

## Sources

[REF-HUBER-1964], [REF-HUBER-1981], [REF-MARONNA-MARTIN-YOHAI-2006],
[REF-BEATON-TUKEY-1974], [REF-HAMPEL-1974], [REF-SEN-1968],
[REF-KENDALL-1938], [REF-KOENKER-BASSETT-1978], [REF-KOENKER-2005],
[REF-BARRODALE-ROBERTS-1973], [REF-EFRON-TIBSHIRANI-1993],
[REF-HOLLANDER-WOLFE-CHICKEN-2014], [REF-R-DATASETS-STACKLOSS].
