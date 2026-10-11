# Extreme Value Analysis

`probstat.risk.extremes` (re-exported from `probstat.risk`) estimates the far
tail of a series: how high the 100-year flood, load, or loss is, and how
uncertain that number is. The generalized Pareto law it needs lives in
`probstat.distributions.GeneralizedPareto`. Every number below was computed
with the library in this repository; the closed forms are re-derived in
`tests/test_extreme_value_analysis.py`.

There are two routes to the tail:

| Route | Data used | Tail model | Main call |
|---|---|---|---|
| Block maxima | one maximum per block (e.g. per year) | GEV | `fit_gev(block_maxima(series, b))` |
| Peaks over threshold (POT) | every excess over a high threshold `u` | GPD + exceedance rate | `fit_pot(series, u, observations_per_year=...)` |

POT uses more of the data, but it needs a threshold choice (section 4) and,
for dependent series, declustering (section 6).

## 1. The generalized Pareto law (EQ-DIST-GPD-001..003)

If `X` is in the domain of attraction of a GEV law, the excess `X - u` given
`X > u` is approximately generalized Pareto for high `u`
(Pickands 1975; Balkema and de Haan 1974):

```text
S(x) = (1 + xi (x - mu)/sigma)^(-1/xi),   x >= mu
```

- `xi > 0`: heavy (Pareto-type) tail.
- `xi = 0`: exponential law with mean `sigma`.
- `xi < 0`: finite upper end `mu - sigma/xi`; `xi = -1` is uniform.

The sign convention matches `GeneralizedExtremeValue` (`xi > 0` is heavy).
Every formula is written with `log1p(xi z)/xi` and `expm1(xi L)/xi`, so the
`xi -> 0` limit is reached without a special case: `GeneralizedPareto(0, 2.5,
1e-12)` agrees with `Exponential(0.4)` in the CDF to 12 decimals. The class
has the full distribution API: `pdf`, `log_pdf`, `cdf`, `sf`, `log_sf`,
`hazard`, `quantile`, `isf`, `mean`, `variance`, `entropy` and seeded
`sample`.

## 2. Block maxima and GEV return levels (EQ-EVA-BM-001, EQ-EVA-RL-001)

`block_maxima(series, b)` keeps the maximum of each complete block of `b`
observations and drops an incomplete last block. `fit_gev` fits the GEV by
maximum likelihood through `inference.fit_distribution(…, GEV_FAMILY)`,
which also returns the observed-information covariance.

The `T`-block return level is the level one block maximum exceeds with
probability `1/T`:

```text
z_T = mu - (sigma/xi) (1 - y_T^(-xi)),   y_T = -log(1 - 1/T)
z_T = mu - sigma log y_T                 (xi = 0)
```

`gev_return_level(3, 2, xi, T)`:

| `xi` | `T = 2` | `T = 10` | `T = 100` | `T = 1000` |
|---|---|---|---|---|
| 0.25 | 3.7677 | 9.0417 | 20.2665 | 39.9817 |
| 0 | 3.7330 | 7.5007 | 12.2003 | 16.8145 |
| -0.2 | 3.7068 | 6.6242 | 9.0149 | 10.4879 |

The shape `xi` matters little at `T = 2` and dominates at `T = 1000`. This is
why the uncertainty in `xi` controls return-level intervals.

## 3. Return-level intervals: delta method vs profile likelihood

Seeded example: 80 maxima from `GEV(10, 2, 0.1)` (`seed=3`). The fit gives
`mu = 10.354 (0.283)`, `sigma = 2.216 (0.216)`, `xi = 0.102 (0.093)` with
standard errors in brackets and `log L = -194.589`.

| `T` | true `z_T` | estimate | 95% delta interval | 95% profile interval |
|---|---|---|---|---|
| 10 | 15.047 | 15.959 | (14.407, 17.511) | (14.681, 18.092) |
| 100 | 21.682 | 23.362 | (17.583, 29.142) | (19.588, 33.658) |

- **Delta method** (`fit.return_level(T)`, EQ-EVA-RL-002): `z_T ± 1.96 se`
  with `se² = grad' V grad`. The gradient in `xi` uses a series near
  `xi = 0`, so it has no `0/0` there. The interval is symmetric by
  construction.
- **Profile likelihood** (`fit.profile_return_level(T)`, EQ-EVA-PROF-001):
  the GEV is rewritten with `z_T` as a parameter, `(sigma, xi)` are maximized
  out, and the interval keeps every `z` with deviance
  `2 (l_max - l_p(z)) <= 3.841`. The test confirms the deviance equals
  `3.841` at both ends to `2e-3`.

The profile interval is skewed upward because the likelihood allows a
heavier tail much more readily than a lighter one. For long return periods,
prefer it over the Wald interval.

**Coverage check.** For 100 seeded samples of 50 maxima from
`GEV(0, 1, 0.1)`, the nominal 95% delta interval for `z_10` covered the truth
in 90 replicates (a separate 150-replicate run gave 90.7%). The
test requires a rate between 82% and 99%. The mild undercoverage is the
known small-sample behaviour of Wald intervals. It grows for shorter records
and longer periods: with 30 maxima and `T = 100` (200 replicates each for
`xi = -0.2, 0.1, 0.3`), nominal 95% delta intervals covered the truth in
76.5-86% of fits and profile intervals in 92.5-96.5%. Use the profile interval
for small samples.

## 4. Choosing a threshold (EQ-EVA-MRL-001, EQ-EVA-STAB-001)

The threshold trades bias (too low: the GPD does not hold yet) against
variance (too high: few excesses). Two diagnostics help:

- `mean_residual_life(values)`: the mean excess `e(u)` with normal-theory
  limits. Above a valid threshold `u0`, `e(u) = (sigma_u0 + xi (u - u0))/(1 - xi)`
  is linear in `u`.
- `threshold_stability(values, thresholds)`: the GPD `xi` and the modified
  scale `sigma* = sigma_u - xi u` at each threshold. Both are constant above a
  valid threshold.

On 4000 standard exponential draws (`seed=21`), where `xi = 0` and
`e(u) = 1` exactly:

| `u` | excesses | `e(u)` | 95% limits | `xi` | 95% limits | `sigma*` |
|---|---|---|---|---|---|---|
| 0.5 | 2418 | 0.988 | (0.950, 1.027) | -0.025 | (-0.066, 0.016) | 1.026 |
| 1.0 | 1478 | 0.973 | (0.924, 1.022) | -0.023 | (-0.078, 0.032) | 1.019 |
| 1.5 | 901 | 0.955 | (0.894, 1.017) | -0.020 | (-0.095, 0.054) | 1.005 |
| 2.0 | 528 | 0.982 | (0.905, 1.060) | -0.098 | (-0.183, -0.012) | 1.274 |

At `u = 2` the `xi` interval just misses 0. This happens in about 1 in 20
thresholds by chance, and the intervals at different thresholds are not
independent. Read the plots for a pattern, not for single points.

## 5. Fitting the GPD: MLE and PWM / L-moments

`fit_gpd(excesses, method="mle")` maximizes the GPD likelihood
(EQ-EVA-GPD-001) through `fit_distribution`. It starts from the PWM estimate,
moved inside the support if needed, and returns the observed-information
covariance. The MLE is regular only for `xi > -1/2` (Smith 1985). For
`xi < -1` the likelihood is unbounded.

`fit_gpd(excesses, method="pwm")` is the Hosking-Wallis (1987)
probability-weighted-moment estimator (EQ-EVA-PWM-001). With unbiased
`b_r`, it is identical to the two-L-moment estimator (Hosking 1990):

```text
xi = 2 - l1/l2,      sigma = l1 (l1/l2 - 1)
```

Hand check: for the excesses `1, 2, 3, 4, 5`, `b0 = 3` and `b1 = 2`, so
`l1 = 3`, `l2 = 1`, `xi = -1`, `sigma = 6`. That is the uniform law on
`[0, 6]`. `sample_l_moments` returns `(3, 1, 0, 0)` for this sample.

On 2000 excesses from `GPD(sigma = 1.5, xi = 0.2)` (`seed=11`):

| method | `sigma` | `xi` | log-likelihood |
|---|---|---|---|
| MLE | 1.5535 (se 0.0527) | 0.1527 (se 0.0258) | -3186.4544 |
| PWM | 1.5512 | 0.1541 | -3186.4559 |

The `xi` estimate is 1.8 standard errors below the truth for this seed. PWM
has no covariance in this library, so its intervals are not reported.

## 6. Exceedance rate, declustering, and POT return levels

**Rate (EQ-EVA-RATE-001).** `exceedance_rate` treats the number of
exceedances `k` out of `n` observations as binomial. It reports
`zeta_u = k/n` with variance `zeta_u (1 - zeta_u)/n`, and the annual rate
`lambda = n_y zeta_u`. `probability_of_exceedance(t)` is the Poisson
probability `1 - exp(-lambda t)` of at least one exceedance in `t` years.

**Return level (EQ-EVA-POT-001/002).** With `m = N n_y` observations in `N`
years:

```text
x_N = u + (sigma/xi) ((m zeta_u)^xi - 1)        (u + sigma log(m zeta_u) at xi = 0)
```

The delta-method interval combines the binomial variance of `zeta_u` with
the GPD covariance. `annual_exceedance_rate(x_N)` returns exactly `1/N`.

Example: ten "years" of daily data (3650 draws from `GPD(0, 1, 0.1)`,
`seed=5`), `u = 2`, `n_y = 365`. There are 601 exceedances, so
`zeta_u = 0.1647` and `lambda = 60.1` per year. The fit is `sigma = 1.104`
and `xi = 0.139`. Above `u = 2` the true values are `1.2` and `0.1`.

| `N` (years) | true `x_N` | estimate | 95% delta interval |
|---|---|---|---|
| 10 | 12.711 | 13.385 | (10.376, 16.394) |
| 50 | 16.676 | 18.229 | (12.482, 23.976) |
| 100 | 18.591 | 20.672 | (13.298, 28.046) |

**Declustering (EQ-EVA-DECL-001).** Exceedances in a storm or a heat wave
are not independent. `decluster_runs(series, u, r)` starts a cluster at an
exceedance and ends it after `r` consecutive non-exceedances. It keeps each
cluster's peak. The runs estimate of the extremal index is
`theta = clusters / exceedances`. Hand example (`u = 4`, `r = 2`):
`0 5 6 0 0 7 0 0 0 8 9 0` gives clusters `{5, 6}`, `{7}`, `{8, 9}`, peaks
`6, 7, 9` and `theta = 3/5`. With `r = 3`, the first two clusters merge.

`fit_pot(…, run_length=r)` fits the GPD to cluster peaks and counts
clusters in `zeta_u`. **Caveat:** on the independent series above,
`r = 3` still merges 601 exceedances into 348 clusters (`theta = 0.58`).
With an exceedance rate of 16%, runs of three non-exceedances are not rare,
so the runs method finds "clusters" in independent data. Use a higher
threshold or a shorter run length when exceedances are frequent.

## 7. Limitations

- No non-stationary (covariate-dependent) GEV/GPD, no point-process
  likelihood, and no `r`-largest order statistics.
- Profile intervals are implemented for GEV return levels only. POT return
  levels have delta-method intervals; the PWM fit has none.
- Return levels assume stationarity over the return period. A 100-year level
  from 10 years of data is an extrapolation: the interval reflects
  parameter uncertainty only, not model error.
