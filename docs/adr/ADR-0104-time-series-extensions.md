# ADR-0104: Time-series extensions: GARCH volatility, vector autoregression, unit-root tests, and change points

- Status: Accepted
- Date: 2026-11-12

## Context

`probstat.timeseries` (ADR-0031) covered univariate second-order models:
correlograms, the Ljung-Box test, Durbin-Levinson, ARIMA on the exact
state-space likelihood, exponential smoothing, and the Schur-Cohn
stationarity check `is_stationary`. Four standard questions had no answer
in the library:

1. Does the variance cluster, and what is tomorrow's volatility? (GARCH,
   Bollerslev 1986.)
2. How do several series interact? (Vector autoregression: order selection,
   stability, impulse responses, variance decomposition, Granger causality;
   Lütkepohl 2005.)
3. Should a series be differenced? (Augmented Dickey-Fuller and KPSS.)
4. Did the mean or the variance change, and where? (CUSUM and PELT.)

The unit-root tests need reference distributions that are not available in
closed form (Dickey-Fuller) or are usually read from a four-point table
(KPSS). The requirement was to transcribe published coefficients only when
they can be checked against the primary publication, and otherwise to
compute p-values by a documented simulation.

## Decision

Four new modules of `probstat.timeseries`, each exported through the
package `__all__`; nothing in the existing modules changed.

**`garch`.** `fit_garch(series, p, q, distribution="normal"|"t")` maximizes
the conditional likelihood [EQ-TS-GARCH-002] by BFGS in unconstrained
coordinates mapped onto the stationary region [EQ-TS-GARCH-001]: `omega`
is `s^2 exp(u)`, the persistence `sum alpha + sum beta` is `logistic(u)`,
and its split among the `p + q` coefficients is a softmax. Every iterate
therefore has `omega > 0`, positive coefficients, and persistence below one;
the Student-t degrees of freedom are `2 + exp(u)`. The variance recursion
starts from the backcast `(1/n) sum e_t^2`. Standard errors are the inverse
central-difference Hessian in natural coordinates with per-coordinate
typical scales (so the step for `omega` stays relative); they are `None`
when an estimate sits on the boundary. `GARCHFit.forecast` implements the
variance recursion [EQ-TS-GARCH-003]; `GARCHFit.diagnostics` reports the
moments of the standardized residuals, Ljung-Box tests on `z` and `z^2`,
and Engle's LM test [EQ-TS-ARCHLM-001], also public as `arch_lm_test`.
`simulate_garch` uses the child stream `"garch"`. The order convention is
Bollerslev's (`q` ARCH terms, `p` GARCH terms), as in `tseries::garch`.

**`var`.** `fit_var` is multivariate least squares [EQ-TS-VAR-001] with
`Sigma_u` on `T - Kp - 1` degrees of freedom; `select_var_order` evaluates
AIC, SC, HQ, and FPE on the common sample of the largest order
[EQ-TS-VAR-002]. The companion eigenvalues [EQ-TS-VAR-003] come from a new
private routine `timeseries._eigen.eigenvalues` (stabilized elementary
reduction to Hessenberg form and the Francis double-shift QR iteration),
because `linalg` has only a symmetric eigensolver. `impulse_response`
(Cholesky-orthogonalized by default), `fevd`, `forecast` with MSE
matrices, and the Granger Wald/F test [EQ-TS-VAR-004 to 007] follow
Lütkepohl. The F denominator degrees of freedom are `K (T - Kp - 1)`.

**`unit_root`.** `adf_test` fits the ADF regression [EQ-TS-ADF-001] with
`"n"`, `"c"`, `"ct"`, or `"ctt"` deterministic terms and lag selection by
AIC or BIC up to Schwert's `floor(12 (n/100)^(1/4))`.

* *Critical values* [EQ-TS-ADF-002]: the `N = 1` rows of MacKinnon (2010),
  Queen's Economics Department Working Paper 1227, **Table 2** (`tau_nc`,
  `tau_c`), **Table 3** (`tau_ct`), and **Table 4** (`tau_ctt`), 1%, 5%,
  and 10%. The coefficients were read from the published PDF of the working
  paper (both its text layer and the rendered tables) and transcribed with
  the blank `beta_3` entries as zero. `T` is the number of observations in
  the test regression, as the paper defines it.
* *P-values* [EQ-TS-ADF-003]: the p-value surfaces of MacKinnon (1994,
  1996) could not be checked against those publications, so they are not
  used. Instead the statistic is compared with `simulations` (default 2000)
  Dickey-Fuller statistics of Gaussian random walks of the same regression
  length and deterministic terms, replication `r` drawn from the child
  stream `("adf", r)` of `RandomStream(seed)` (default `seed=0`). The
  p-value is `(1 + #{tau* <= tau}) / (1 + R)` with standard error
  `sqrt(p (1 - p) / R)`, at most `0.011` for `R = 2000`; results are
  cached per `(T, regression, R, seed)`. The finite-sample null law is
  exact for the Dickey-Fuller regression with normal errors; with lagged
  differences it is the same limit (Said and Dickey 1984).

`kpss_test` [EQ-TS-KPSS-001] uses the Bartlett long-run variance
(`newey_west_variance`) with the `"short"`/`"long"` truncation rules of
`tseries::kpss.test`. Its p-value is **exact asymptotically**
[EQ-TS-KPSS-002]: the limit is `sum lambda_k Z_k^2`, with
`lambda_k = 1/(k pi)^2` for the level case (Cramer-von Mises) and, for the
trend case, the eigenvalues of the kernel
`min(s,t) - st - 3 s(1-s) t(1-t)`. Writing that kernel as a rank-one
downdate of the Brownian-bridge kernel gives the secular equation
`1 = 3 sum_(k odd) c_k^2/(lambda_k - mu)`, `c_k^2 = 32/(k pi)^6`; summing
the partial fractions with `sum_(k odd) 1/(k pi)^4 = 1/96`,
`sum_(k odd) 1/(k pi)^2 = 1/8`, and
`sum_(k odd) 1/(omega^2 - (k pi)^2) = -tan(omega/2)/(4 omega)` reduces it to
`tan(omega/2) = omega/2`, so the odd eigenvalues are `1/(4 x_j^2)` with
`tan x_j = x_j`, and the even ones stay `1/(2 pi j)^2`. The survival
function is Imhof's inversion integral, evaluated by adaptive Gauss-Kronrod
quadrature with the 50 largest eigenvalues explicit and the rest through
their first three power sums (the first from the exact traces `1/6` and
`1/15`). The four critical values per case are stored to six decimals and
re-derived by a regression test.

**`changepoint`.** `cusum_test` [EQ-TS-CP-001] with an optional Newey-West
scale and the Kolmogorov tail (`kolmogorov_sf`); `pelt` [EQ-TS-CP-002] with
the Gaussian mean, variance, and mean-and-variance costs and the BIC/MBIC
penalties of R's `changepoint` package [EQ-TS-CP-003] (read from its
`cost_general_functions.c` and `penalty_decision.R`): BIC `(d + 1) log n`,
MBIC `(d + 2) log n` plus `log n_i` per segment. The pruning constant is
`K = 0` for the likelihood costs and `K = -log n` with the MBIC segment
term (`log a + log b - log(a + b) <= log n`). With a minimum segment length
`L` a pruned candidate stays usable for `L - 1` more steps, because the
pruning inequality only covers continuations of length at least `L`; this
makes `pelt` identical to optimal partitioning for every `L`. The mean
model's noise variance defaults to `1.4826 MAD(Delta x)/sqrt 2`
(`robust_noise_variance`); `variance=1.0` reproduces `cpt.mean` on
standardized data. Variance estimates are floored at `1e-11` times the
sample variance (scale-free, unlike the package's absolute `1e-11`).

The guide `docs/foundations/time-series.md` gained sections 8-11, a
method-choice table, and pitfalls (spurious regression, lag choice,
volatility clustering versus structural breaks, the opposite nulls of ADF
and KPSS, Monte Carlo error of simulated p-values, the GARCH order
convention); learning-path step S8 lists them. No lab was added.

## Consequences

- The ADF p-value costs about `R T` random-walk steps on first use (0.5 s
  to 1 s for `T` near 250 and `R = 2000`) and is free afterwards for the
  same `(T, regression, R, seed)`. Its resolution is `1/(R + 1)`; the
  standard error is reported with every result.
- The ADF critical values are strictly those of the Dickey-Fuller regression
  without lags at the regression length `T`; with many lags they are an
  approximation, as for every response-surface table.
- KPSS p-values are asymptotic; the four exact limiting critical values
  differ in the third decimal from the simulated table of the original
  paper as used by `tseries::kpss.test` (level `0.347/0.463/0.574/0.739`,
  trend `0.119/0.146/0.176/0.216` against `0.347305/0.461361/0.580615/
  0.743459` and `0.119220/0.147890/0.177468/0.217747` here).
- GARCH standard errors are `None` at boundary estimates (for example
  GARCH(2, 1) on GARCH(1, 1) data, where `beta_2` tends to zero); the
  estimates and the likelihood are still valid there.
- Orthogonalized impulse responses and the FEVD depend on the order of the
  variables; the docstrings and the guide say so.
- The Hessenberg QR routine is private to `timeseries`; if another package
  needs non-symmetric eigenvalues it should move to `linalg` (layer 0).
- R is not installed in the environment where the oracle values were
  produced, so the R packages named in the task (`rugarch`, `tseries`,
  `vars`, `urca`, `changepoint`) were not run; the oracles below replace
  them.

## Verification

All values below are asserted in `tests/test_timeseries_extensions.py`
(35 tests, about 14 s) unless stated otherwise; offline oracles used NumPy
2.2.6 and SciPy 1.14.1 on series generated by this library.

| method | oracle | result |
|---|---|---|
| GARCH(1, 1) normal, `simulate_garch(0.05, (0.1,), (0.85,), 1500, mean=0.02, seed=2026)` | the same likelihood in NumPy maximized by `scipy.optimize.minimize(method="SLSQP")` with bounds and the constraint `sum alpha + sum beta <= 0.999999` | `log L` equal to `4.5e-13`; estimates within `3e-8` |
| GARCH(1, 1) Student-t, seed 2027 | same | `log L` within `1.8e-10`; `nu` within `1.8e-5` |
| ARCH(2) and GARCH(2, 1) (offline only) | same | `log L` within `2e-12` and `3.4e-6` (boundary `beta_2`) |
| GARCH log-likelihood | direct normal and Student-t density sums | `1e-12` |
| GARCH(1, 1) forecast | closed form `sigma_bar^2 + (a+b)^(h-1)(sigma_(n+1)^2 - sigma_bar^2)` | `1e-12`, 60 steps |
| `simulate_garch` | unconditional variance `0.4` of ARCH(2), 100 000 draws | within 4 batch-means SE |
| ARCH(2) estimator, 20 seeds (offline) | true `(0.2, 0.3, 0.2)` | means `0.206, 0.306, 0.194` with SE `0.004, 0.011, 0.007` |
| VAR(2) coefficients, `Sigma_u` | `numpy.linalg.lstsq` | `1e-15` |
| companion eigenvalues | `numpy.linalg.eigvals`; 3000 random matrices up to `12 x 12` (offline) | moduli `1e-15` on the VAR; worst `2.2e-8` on matrices with repeated integer eigenvalues |
| impulse responses, FEVD | NumPy recursion; VAR(1) `A^i P` closed form; FEVD rows sum to one, `h = 1` shares `P_ij^2 / sum_j P_ij^2` | `9e-16`, `1e-12` |
| Granger Wald | NumPy `b' V^-1 b`, `scipy.stats.f.sf`, `chi2.sf` | `1e-12` relative |
| Granger size | 300 simulated VAR(1) without causality | rejection rate within 3 SE of 5% |
| order selection | NumPy `log det(U'U/T) + c m K^2 / T` | `8e-16`; AIC and FPE pick 2, BIC and HQ pick 1 |
| ADF statistic | `numpy.linalg.lstsq` t-ratio, regressions `c`, `ct`, lags 0 and 3 | `2e-15` |
| ADF critical values | MacKinnon (2010) Tables 2-4: `beta_inf` at `T = 1e9` and the formula at `T = 100` | exact to `1e-8` |
| simulated DF distribution | share of simulated statistics below each MacKinnon critical value, `T = 100`, `R = 20 000`, seed 2026, all four regressions (offline) | every `|z|` below 2; in the test `R = 5000` within 4 SE |
| ADF size and power | 200 random walks, `n = 101`; AR(1) `phi = 0.5` | rate within 3 SE of 5%; `p = 1/(R+1)` for the AR(1) |
| KPSS statistic | NumPy residuals, `cumsum`, Bartlett variance | `1e-16` |
| KPSS level limit | `1 - scipy.stats._hypotests._cdf_cvm_inf` (Cramer-von Mises) | within `5e-11` for `x` from `0.1` to `1.5` (tests: `1e-9`) |
| KPSS trend eigenvalues | brentq roots of the secular equation; eigenvalues of `M L' L M / T^2` at `T = 1500` (offline) | `1.5e-12` relative to `1/(4 x^2)`; four digits to the finite-`T` matrix |
| KPSS trend limit | Gil-Pelaez inversion with `scipy.integrate.quad` and 40 000 explicit eigenvalues | within `5e-11` |
| KPSS size | 300 white-noise series, `n = 250` | rate within 3 SE of 5% |
| CUSUM | NumPy statistic and arg-max; `scipy.stats.kstwobign.sf` | `1e-15`; tail `1e-14` |
| CUSUM size | 400 white-noise series | rate below 5% + 3 SE (asymptotic test, conservative) |
| PELT | brute-force optimal partitioning on 18 series (three models, three penalties, `L` in 1, 2, 3, 5), exhaustive enumeration on `n = 10`; 60 further random cases offline | identical change points, objective within `1e-8` |
| PELT recovery | means `0, 3, 3`, sd `1, 1, 3`, 900 points | `(300, 600)`; 129 751 of 405 450 costs evaluated |

The guide's examples run under `tests/test_documentation_examples.py`, and
every number they quote is checked there.
