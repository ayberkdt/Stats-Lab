# Time Series: Correlograms, ARIMA, and Exponential Smoothing

A time series is a set of observations whose order matters: today's value
carries information about tomorrow's. This guide covers `probstat.timeseries`.
Every number below was computed by the library with the seed shown, and the
cross-checks named in each section were run against NumPy/SciPy in a scratch
script (those packages are not dependencies of the library).

## 1. Correlograms

The sample autocovariance uses the `1/n` divisor [EQ-TS-ACF-001], which keeps
the sample autocovariance matrix positive semidefinite. `acf(series, K)`
returns `rho_hat(0..K)`.

**AR(1) check.** For `x_t = 0.7 x_(t-1) + e_t` the theoretical ACF is
`0.7^k`. With `simulate_arma((0.7,), (), 4000, seed=101)`:

| lag | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| sample `rho_hat(k)` | 0.6898 | 0.4711 | 0.3248 | 0.2331 |
| `0.7^k` | 0.7000 | 0.4900 | 0.3430 | 0.2401 |
| Bartlett half-width (95%) | 0.0310 | 0.0433 | 0.0480 | 0.0500 |

The Bartlett band [EQ-TS-BAND-002] widens with the lag because the earlier
autocorrelations add sampling variance. The white-noise band
[EQ-TS-BAND-001] is the lag-1 value `1.96/sqrt(4000) = 0.0310`. Every
sample value lies within one Bartlett half-width of `0.7^k`.

**PACF and Durbin-Levinson.** `durbin_levinson` [EQ-TS-PACF-001] solves the
Yule-Walker equations for every order at once; its reflection coefficients
are the partial autocorrelations. For the AR(2) `phi = (0.5, -0.3)` the
theoretical PACF (from `arma_autocovariance`) is `0.384615` (that is
`0.5/1.3`), then `-0.3`. Lags 3-5 come out at about `1e-17`, which is zero
to rounding. That is the AR(p) cut-off. The sample PACF of
`simulate_arma((0.5, -0.3), (), 2000, seed=102)` is
`0.3637, -0.2892, 0.0052, 0.0134, 0.0103, 0.0010` against a band of
`+/-0.0438`. The NumPy cross-check (solving each Toeplitz system directly)
agreed to `6e-16`.

**Ljung-Box.** `Q = n(n+2) sum rho_hat(k)^2/(n-k)` [EQ-TS-LB-001] is compared
with `chi^2(h - p - q)`:

| series (n = 300, h = 10) | `Q` | p-value |
|---|---|---|
| white noise, `seed=103` | 8.175 | 0.612 |
| AR(1) `phi = 0.6`, `seed=104` | 172.93 | `6.9e-32` |

The statistic matched a NumPy evaluation to `9e-14`, and the p-value matched
`scipy.stats.chi2.sf`.

## 2. Differencing and decomposition

`difference(y, d, lag=1)` applies `(1 - B)^d`; `seasonal_difference(y, s, D)`
applies `(1 - B^s)^D`; `undifference` inverts ordinary differencing from the
first `d` values [EQ-TS-DIFF-001]. Applying `difference(., 2)` to `1, 4, 9,
16, 25, 36` gives the constant `2`, as expected for a quadratic.

`classical_decomposition(y, s, model=...)` [EQ-TS-DECOMP-001] uses a centred
moving average (`2 x s` for even `s`) as the trend. On
`10 + 0.5 t + (2, -1, 0.5, -1.5)[t mod 4]` it recovers the seasonal indices
exactly and leaves a zero remainder. The trend is `None` at the ends, where
the window is incomplete. In the multiplicative form the indices average one.
With a linear trend they are close to the true factors but not exact: on
`(10 + 0.5t)(1.2, 0.9, 1.1, 0.8)` the indices were
`1.1990, 0.9007, 1.1009, 0.7993` (24 points). The centred average of a
product is not the product of the averages. An STL-type robust decomposition
was **not** implemented (see ADR-0031).

## 3. ARMA polynomials: roots and stationarity

Sign convention: `phi(z) = 1 - sum phi_j z^j` and `theta(z) = 1 + sum theta_j
z^j` [EQ-TS-ARMA-001]. Two routes are used:

- `polynomial_roots` [EQ-TS-ROOT-001] runs the Durand-Kerner iteration plus
  Newton polishing. On six test polynomials, including complex pairs, a
  quintic, and the double root of `1 - 2z + z^2`, the roots agreed with
  `numpy.roots` to `6e-15`. The double root agreed only to `5e-10`, because
  repeated roots converge linearly.
- `is_stationary` / `is_invertible` [EQ-TS-ROOT-002] decide with the
  Schur-Cohn step-down recursion, which needs no root finding.
  `ar_root_check` reports both.

For `phi = (0.5, -0.3)` the roots are `0.8333 +/- 1.6245i`, with modulus
`1.8257 = sqrt(1/0.3)`, and the reflection coefficients are
`(0.3846, -0.3)`. So the model is stationary. `(1.2,)`, `(1.0,)`,
`(0.5, 0.5)`, and `(0.2, 0.9)` are rejected, and `arma_state_space`,
`simulate_arma`, and `arma_log_likelihood` raise `ValueError` for them. The
MA polynomial `1 + 1.5z` has its root at `-0.6667` and is not invertible.

## 4. Exact likelihood through the state-space form

`arma_state_space` builds the Harvey form [EQ-TS-SSM-001] with
`r = max(p, q + 1)`. The stationary state covariance comes from solving the
Lyapunov equation exactly. `arma_autocovariance` reads `gamma(k) = Z T^k P Z'`
off it. Against the `psi`-weight sum [EQ-TS-ARMA-002], with 3000 terms, it
agreed to `9e-16` for AR(1), ARMA(2,1), MA(1), and ARMA(3,2).

The likelihood [EQ-TS-LIK-001] is the prediction-error decomposition of the
Kalman filter. Two implementations share it:

- `ARMAStateSpace.kalman(series)` returns a full
  `filtering.KalmanFilterResult`. Its measurement step is
  `filtering.linear_gaussian_update`, the update that `kalman_filter` uses.
- `arma_log_likelihood` is a scalar `O(r^2)`-per-step specialization used for
  estimation. It freezes the covariance once it reaches steady state.

Checks:

| check | difference |
|---|---|
| AR(1) closed form [EQ-TS-LIK-002] vs both filters (`phi=0.7`, `sigma^2=0.8`, `mu=0.3`, n=200, seed 105; `log L = -269.5869`) | `5.7e-14` (both filters) |
| dense `N(mu 1, Toeplitz(gamma))` log-density, n=150, five models incl. non-invertible MA(1) `theta=1.5` | at most `6.8e-13` |
| ARMA(2,1) plus measurement noise `0.25`: `kalman(..., observation_variance=0.25)` vs `filtering.kalman_filter(model.to_linear_gaussian(0.25), ...)` | `0.0` |

`filtering.LinearGaussianStateSpace` requires a positive-definite observation
covariance, so a pure ARMA (observation noise zero) cannot be passed to
`kalman_filter` itself. ADR-0031 explains the workaround.

## 5. Estimation

`yule_walker(series, p)` [EQ-TS-YW-001] is always stationary. It matched a
direct NumPy solve to `4e-16`. `fit_arima(series, (p, d, q))` maximizes the
concentrated exact likelihood by BFGS. The optimizer works on `u` with
reflection coefficients `tanh(u)` [EQ-TS-PARAM-001], so every iterate is
stationary and invertible. Standard errors come from the inverse
central-difference Hessian [EQ-TS-SE-001].

| simulation | estimates | standard errors | `log L` |
|---|---|---|---|
| AR(2) `(0.5, -0.3)`, `mu = 1`, n = 800, seed 108 | `phi = (0.4131, -0.3107)`, `mu = 1.0267`, `sigma^2 = 0.9421` | `0.0336, 0.0336, 0.0382, 0.0471` | -1111.441 |
| MA(1) `theta = 0.6`, n = 800, seed 109 | `theta = 0.6002`, `sigma^2 = 1.0093` | `0.0286, 0.0505` | -1139.092 |

The AR(2) `phi_1` estimate is 2.6 standard errors from the truth. That is an
honest sampling outcome for this seed, and the test checks the 3-SE
criterion. For three models (n = 400-600), the maximum agreed with an
independent SciPy Nelder-Mead run on the same likelihood to `2.3e-12` in
`log L`. Information criteria (AIC / BIC) on the AR(2) data pick the true
order:

| order | AIC | BIC |
|---|---|---|
| (1,0,0) | 2310.07 | 2324.12 |
| **(2,0,0)** | **2230.88** | **2249.62** |
| (1,0,1) | 2259.23 | 2277.97 |
| (0,0,2) | 2257.69 | 2276.42 |

`fit.diagnostics(10)` runs Ljung-Box on the standardized innovations with
`p + q` degrees of freedom removed. For the AR(2) fit: `Q = 6.23`, `df = 8`,
`p = 0.621`.

## 6. Forecasting with exact intervals

`fit.forecast(h)` [EQ-TS-FC-001] propagates the filter's final predicted
state, keeps the full conditional covariance of the future differenced
values, and integrates `d` times. The intervals are exact given the
parameters. They omit parameter-estimation uncertainty.

- **ARIMA(0,1,0)**, `simulate_arima((), 1, (), 300, seed=110)`: the forecast
  stays at the last value, `8.8470`. Standard errors are
  `1.0113, 1.4301, 1.7516, 2.0225` for `h = 1..4`, which is
  `sqrt(sigma_hat^2 h)` with `sigma_hat^2 = 1.0226`. The ratio to `sqrt(h)`
  equals one to `2.2e-16` for `h = 1..8`.
- **AR(1)**, seed 111 (`phi_hat = 0.5081`, `mu_hat = 1.8016`): means and
  standard errors equal the closed forms `mu + phi^h (y_n - mu)` and
  `sigma sqrt(sum_{j<h} phi^{2j})` to `1e-10`. The standard error rises from
  `0.979` toward the marginal value.

## 7. Simulation and exponential smoothing

`simulate_arma(ar, ma, n, seed=..., burn_in=...)` [EQ-TS-SIM-001] uses the
child stream `"arma"` of a `RandomStream`. The default burn-in is
`200 + 10(p + q)`. `simulate_arima` integrates the ARMA path `d` times.

`simple_exponential_smoothing` [EQ-TS-ES-001] and `holt_linear`
[EQ-TS-ES-002] optimize `alpha` (and `beta`) over `(0, 1)` by BFGS on the
logit. The initial level and trend enter the one-step errors linearly, so
they are concentrated out exactly by least squares.

| data | result |
|---|---|
| `5 + N(0, 1)`, n = 200, seed 112 | `alpha = 2e-8` (no evidence of level change); forecast `5.0722` = least-squares level (the sample mean) |
| `3 + 0.7 t + N(0, 0.25)`, n = 150, seed 113 | `alpha = 1.2e-5`, `beta = 7.9e-5`, slope `0.6995` |
| ARIMA(0,1,1) with `theta = -0.6`, n = 300, seed 43 | `alpha = 0.355` (the equivalent ARIMA value is `1 + theta = 0.4`) |

Smoothing returns point forecasts only. No prediction intervals are provided
for it.
