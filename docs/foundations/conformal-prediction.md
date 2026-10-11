# Conformal Prediction Intervals

`probstat.conformal` turns any point predictor into prediction intervals with
a finite-sample coverage guarantee. The predictor can be a callable, a fitted
PCE or GP surrogate, or a refitting procedure. The guarantee assumes nothing
about the model and needs only that the data are exchangeable. Every number
below was computed with the library and is asserted, or recomputed, in
`tests/test_conformal_prediction.py`.

## 0. What is guaranteed, and what is not

Let `C(x)` be the interval built from `n` calibration points. For a new pair
`(X, Y)` exchangeable with them,

```text
1 - alpha  <=  k / (n + 1)  <=  P(Y in C(X)),    k = ceil((1 - alpha)(n + 1)).
```

When the scores are almost surely distinct, the probability equals `k/(n+1)`,
which is less than `1 - alpha + 1/(n+1)` (EQ-CONF-005).

The probability is **marginal**. It averages over the calibration set *and*
the test point. Three consequences:

- It does **not** promise `1 - alpha` coverage at every `x`. A test region
  where the model is poor can be under-covered while another is over-covered.
- It does **not** promise `1 - alpha` coverage for one fixed calibration set.
  For a single calibration set, the coverage of later points varies around
  `k/(n+1)`; it follows a `Beta(k, n + 1 - k)` law because it equals the law
  probability below the `k`-th order statistic of `n` uniforms.
- A bad model still gets *valid* intervals, but they are wide.

Exchangeability is the key assumption. Distribution shift, time ordering, or
a calibration set reused for fitting all break it.

## 1. The finite-sample quantile (EQ-CONF-001)

`conformal_rank(n, alpha)` returns `k = ceil((1 - alpha)(n + 1))`.
`conformal_quantile(scores, alpha)` returns the `k`-th smallest score, or
`+inf` when `k = n + 1`. This is the empirical `(1 - alpha)(n+1)/n` quantile.

The index is computed in exact rational arithmetic from the decimal text of
`alpha`. Binary rounding therefore cannot move `k`; `(n, alpha) = (9, 0.3)`
gives exactly `k = 7`.

| `n` | `alpha` | `k` | `k/(n+1)` | Naive `ceil((1-alpha) n)` |
|---|---|---|---|---|
| 9 | 0.1 | 9 | 0.900 | 9 |
| 10 | 0.1 | 10 | 0.909 | 9 (would give 0.818) |
| 19 | 0.1 | 18 | 0.900 | 18 |
| 99 | 0.05 | 95 | 0.950 | 95 |
| 5 | 0.1 | 6 > n | 1 (infinite intervals) | 5 |

The `n = 10` row shows why the `(n+1)/n` correction matters. The seeded
replicates below land on `0.909`, not on `0.818`.

## 2. Split conformal regression (EQ-CONF-002)

`split_conformal_regression(predictor, x_cal, y_cal, alpha=0.1)` scores the
held-out residuals `|y - mu(x)|` and returns `mu(x) +/- q_hat`. Passing
`scale=s` switches to normalized (locally weighted) scores
`|y - mu(x)| / s(x)`, and the interval becomes `mu(x) +/- q_hat s(x)`. The
guarantee is unchanged, and the width now follows `s(x)`.

The predictor and `s` must not have seen the calibration data.

Seeded check (`tests/test_conformal_prediction.py`):

- data: `x ~ U(0, 3)`, `y = sin 2x + 0.3x + N(0, (0.2 + 0.2x)^2)`;
- predictor: a deliberately wrong line `0.1 + 0.3x`;
- design: `n = 10`, `alpha = 0.1`, 500 calibration/test replicates with 40
  test points each;
- target: `k/(n+1) = 10/11 = 0.9091`.

| Method | Mean coverage | Monte Carlo se |
|---|---|---|
| absolute residual | 0.9035 | 0.0044 |
| normalized, `s(x) = 0.2 + 0.2x` | 0.9099 | 0.0044 |
| CQR (Section 3) | 0.9050 | 0.0043 |

All three are within 1.3 standard errors of `10/11`. The naive index would
give `0.818`; the absolute-residual mean is about 20 standard errors above
that value.

## 3. Conformalized quantile regression (EQ-CONF-003)

`conformalized_quantile_regression(q_lo, q_hi, x_cal, y_cal)` takes any lower
and upper quantile predictors. It scores `max(q_lo(x) - y, y - q_hi(x))` and
returns `[q_lo(x) - q_hat, q_hi(x) + q_hat]`.

The bands keep their adaptive shape, and the shift repairs their calibration.
In the test, bands only half as wide as the noise end with mean coverage
`0.9050` (target `0.9091`). A negative `q_hat` narrows bands that were too
wide. Fitting the quantile predictors themselves (for example by pinball-loss
regression) is outside this module.

## 4. Jackknife+ (EQ-CONF-004)

`jackknife_plus_regression(fit, x_train, y_train)` refits the model `n` times,
leaving out one point each time, and uses the leave-one-out residuals. The
interval at `x` takes:

- as lower limit, the `floor(alpha(n+1))`-th smallest `mu_{-i}(x) - R_i`;
- as upper limit, the `ceil((1-alpha)(n+1))`-th smallest `mu_{-i}(x) + R_i`.

No data are held out. The guarantee is `1 - 2 alpha` for any symmetric
fitting rule. Coverage is usually near `1 - alpha`.

Seeded check: least-squares lines, `n = 20`, `alpha = 0.1`, 150 replicates of
20 test points.

- mean coverage: `0.908`;
- mean width: `2.51`;
- bound: `0.8`; asserted window: `[0.85, 0.97]`.

## 5. Error bars for a surrogate of a probabilistic model

`conformalize_surrogate(model, surrogate, alpha=0.1, size=n, seed=...)`
quantifies how far an emulator's prediction can be from the true model QoI:

1. It draws `n` independent inputs from `model.inputs`, using stream child
   `"conformal-calibration"`.
2. It runs the true model there; this costs `n` evaluations, which are
   counted in `model.evaluations`.
3. It calibrates split conformal intervals around `surrogate.predict`.

The calibration inputs are plain Monte Carlo draws by design. LHS or Sobol
points are not exchangeable with a random test input, so the guarantee would
not apply. The inputs must also be independent of the surrogate's training
design.

| Surrogate | Calibration | Interval | Coverage on 2000 fresh inputs |
|---|---|---|---|
| Degree-1 PCE (30-run LHS; relative LOO error 0.178) of `x1 + 0.5 x2 + 0.3 x1^2` | 99 runs | `+/- 0.459` | 0.851 |
| 20-point GP of `sin x1 + 0.5 x1 sin x2` | 99 runs | `+/- 0.153` (absolute scores) | 0.937 |
| Same GP | 99 runs | `+/- q sd(x)` (normalized) | 0.933 |

The guarantee for the PCE row is `k/(n+1) = 0.9`. The observed `0.851` is
about 1.6 standard deviations below 0.9 under the `Beta(90, 10)` law of
fixed-calibration coverage (standard deviation about 0.03). That is the
calibration-set variability described in Section 0, not a failure.

For the GP rows, normalized scores divide residuals by the GP predictive
standard deviation. The median width drops from `0.306` to `0.050`, while the
intervals widen far from the data: at `(3, -3)` the interval is `2.53` wide.

**Pathology found while testing.** A GP fitted to the smooth quadratic chose
very long length scales. Its predictive variance then cancelled to exactly 0
at some calibration inputs. A normalized score is meaningless there, so
`conformalize_surrogate` raises and asks for a `scale_offset`. That positive
constant is added to the standard deviation and must be chosen before seeing
the calibration data.

## 6. Coverage checks

`empirical_coverage(intervals, y_test, nominal=0.9)` reports:

- the covered count and rate;
- an exact Clopper-Pearson interval for the rate;
- mean and median widths;
- whether the nominal level is inside that interval.

Use it on a test set disjoint from calibration. Repeat it within subgroups to
look for conditional under-coverage.

## Limitations

- Only regression with scalar responses is covered. Classification sets,
  multi-output, time series, and covariate-shift (weighted) conformal are not
  implemented.
- Split conformal spends data on calibration. Jackknife+ spends `n` refits
  instead. CV+ and full conformal are not implemented.
- Normalized scores are only as useful as `s(x)`. A GP standard deviation can
  vanish numerically (Section 5).
- Coverage is marginal. Nothing here certifies coverage at a particular input
  or for a particular calibration draw.

## Sources

[REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018],
[REF-ROMANO-PATTERSON-CANDES-2019], [REF-BARBER-ETAL-2021],
[REF-ANGELOPOULOS-BATES-2023], [REF-CLOPPER-PEARSON-1934],
[REF-RASMUSSEN-WILLIAMS-2006], [REF-SUDRET-2008].
