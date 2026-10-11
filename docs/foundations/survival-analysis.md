# Survival Analysis for Right-Censored Data

`probstat.inference.survival` estimates time-to-event distributions when some
subjects leave follow-up before the event. Every number below was computed
with the library in this repository, and the same values are recomputed from
the defining formulas in `tests/test_survival_analysis.py`.

## 0. Censoring and the risk set

An observation is a pair `(t_i, delta_i)`. If `delta_i = 1`, the event was
seen at `t_i`. If `delta_i = 0`, the subject was **right-censored**: it was
still event-free when follow-up ended at `t_i`. Throwing censored subjects
away, or treating their times as event times, biases every summary toward
shorter survival.

Every method here assumes **non-informative censoring**: given the
covariates, the reason follow-up stops tells nothing about the subject's
future risk. If sicker subjects drop out more often, no estimator here
corrects for it.

`survival_data(times, events)` validates the sample. `risk_table()` returns,
for each distinct time, `n_j` (subjects with time `>= t_j`), `d_j` (events),
and `c_j` (censorings). A subject censored at `t_j` still counts in `n_j`.

Hand example used throughout (`+` marks censoring):
`1, 2, 2+, 3, 4+, 5, 5, 6+`.

| `t_j` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `n_j` | 8 | 7 | 5 | 4 | 3 | 1 |
| `d_j` | 1 | 1 | 1 | 0 | 2 | 0 |

## 1. Kaplan-Meier estimator (EQ-SURV-KM-001..003)

`kaplan_meier(data)` returns `S(t) = prod_{t_j <= t} (1 - d_j/n_j)` with:

- Greenwood's variance `S^2 sum d_j / (n_j (n_j - d_j))`; and
- log-log pointwise intervals `S^{exp(+/- z w)}`, where
  `w = sqrt(sum d_j/(n_j(n_j - d_j))) / |log S|`. These intervals always stay
  inside `(0, 1)`.

| `t_j` | `S(t_j)` | Greenwood se | 95% log-log interval |
|---|---|---|---|
| 1 | 0.8750 | 0.1169 | (0.3870, 0.9814) |
| 2 | 0.7500 | 0.1531 | (0.3148, 0.9309) |
| 3 | 0.6000 | 0.1817 | (0.1955, 0.8523) |
| 5 | 0.2000 | 0.1742 | (0.0096, 0.5727) |

The intervals are **pointwise**. Each covers `S(t)` at one fixed `t` with
about 95% probability; they do not form a band for the whole curve. When the
last subject at risk has the event, `S` drops to 0. The library then reports
variance 0 and the interval `(0, 0)`.

## 2. Median survival and other quantiles (EQ-SURV-MED-001)

`km.median()` is the first event time with `S(t) <= 0.5`. Its interval inverts
the pointwise bands:

- the lower limit is the first time the lower band reaches 0.5;
- the upper limit is the first time the upper band reaches 0.5.

Hand example: the median is `5`, with 95% interval `[1, inf)`. The upper band
never falls to 0.5 during follow-up, so the upper limit is unbounded. That
result is honest: eight subjects cannot bound the median from above.
`km.quantile(p)` gives other quantiles. A limit is `inf` when its curve does
not reach `1 - p`.

## 3. Restricted mean survival time (EQ-SURV-RMST-001)

`km.restricted_mean(tau)` is the area under the step curve up to `tau`,
`E[min(T, tau)]`. Its variance is
`sum_{t_j <= tau} A_j^2 d_j / (n_j (n_j - d_j))`, where `A_j` is the area
from `t_j` to `tau`.

Hand example, `tau = 6`:

- `RMST = 1 + 0.875 + 0.75 + 2(0.6) + 0.2 = 4.025`;
- `se = 0.631`;
- 95% interval `(2.788, 5.262)`.

The RMST is identified even when the median is not. A `tau` beyond the last
observed time is rejected while `S > 0`, because the curve is unknown there.

## 4. Nelson-Aalen cumulative hazard (EQ-SURV-NA-001)

`nelson_aalen(data)` returns `H(t) = sum d_j/n_j` with variance `sum d_j/n_j^2`.

Hand example:

- `H = 0.1250, 0.2679, 0.4679, 1.1345` at `t = 1, 2, 3, 5`;
- variances `0.0156, 0.0360, 0.0760, 0.2983`.

`exp(-H(t))` is never below the Kaplan-Meier value. At `t = 5` it is
`0.3216`, against `0.2000`.

## 5. Log-rank test (EQ-SURV-LR-001)

`log_rank_test(*groups)` compares observed and expected events at each pooled
event time, using the hypergeometric moments of each risk-set table. For `k`
groups, `U' V^{-1} U` on the first `k - 1` groups is approximately chi-square
with `k - 1` degrees of freedom.

Test groups: A = `3, 5, 5, 8, 10+, 12`, B = `1, 2, 4, 4+, 6, 7, 9+`,
C = `2, 6+, 11, 13, 14`.

| Groups | Statistic | df | p | Observed | Expected |
|---|---|---|---|---|---|
| A, B | 0.590 | 1 | 0.442 | (5, 5) | (6.107, 3.893) |
| A, B, C | 3.553 | 2 | 0.169 | (5, 5, 4) | (4.317, 2.978, 6.705) |

The test is most powerful when hazards are proportional. It can miss curves
that cross. With one binary covariate and no tied times, the Cox score test
(Section 7, Breslow ties) gives exactly the log-rank statistic, and the test
suite checks this.

## 6. Parametric AFT models (EQ-SURV-AFT-001)

`fit_parametric_survival(data, family, covariates=...)` fits the model
`log T = x' beta + sigma W` by censored maximum likelihood (BFGS with the
analytic gradient).

| `family` | `W` | Lifetime law |
|---|---|---|
| `"weibull"` | minimum extreme value | `Weibull(shape = 1/sigma, scale = exp(eta))` |
| `"exponential"` | same, with `sigma = 1` | `Exponential(rate = exp(-eta))` |
| `"lognormal"` | standard normal | `LogNormal(mu = eta, sigma)` |

Events contribute their density to the likelihood. Censored subjects
contribute the survival function. Standard errors come from the inverse
observed information. `weibull_parameters()` maps them to shape and scale by
the delta method.

Seeded check: 500 Weibull(shape 1.7, scale 5) lifetimes with uniform(0, 14)
censoring (146 censored, 29.2%).

| Quantity | Estimate | se |
|---|---|---|
| shape | 1.6551 | 0.0678 |
| scale | 5.0440 | 0.1628 |

Both estimates lie within one standard error of the truth. The AIC ranks
Weibull (1744.0) ahead of lognormal (1779.2) and exponential (1860.8).

In the exponential model with no covariates, the MLE is the closed form
`rate = d / sum t_i`, with `se(log mean) = 1/sqrt(d)`. Without censoring, the
lognormal and Weibull fits reproduce the ordinary MLEs. The tests compare the
Weibull fit with the independent profile-likelihood `fit_weibull_mle`.

`exp(beta_k)` is a **time ratio**: a unit increase in covariate `k`
multiplies survival time by `exp(beta_k)`. A seeded covariate effect of
`0.6` is recovered within three standard errors. For the Weibull model, the
equivalent proportional-hazards coefficient is `-beta_k / sigma`. The library
reports AFT coefficients only.

## 7. Cox proportional hazards (EQ-SURV-COX-001..004)

`fit_cox_model(data, covariates, ties="efron")` maximizes the partial
likelihood by Newton-Raphson from `beta = 0`, halving each step until the
likelihood does not decrease. It reports:

- `beta` with standard errors from the inverse information;
- hazard ratios `exp(beta)` with Wald intervals `exp(beta +/- z se)`;
- two-sided Wald p-values;
- the likelihood-ratio and score global tests;
- the Breslow baseline cumulative hazard, used by `survival(t, x)`.

Example from the tests: ten subjects, three censored, one continuous
covariate.

| Quantity | Value |
|---|---|
| `beta` | 1.5938 |
| se | 0.7647 |
| HR | 4.92 |
| 95% interval for HR | (1.10, 22.03) |
| Wald p | 0.037 |
| LR statistic | 5.687 |
| score statistic | 5.568 |
| Newton iterations | 4 |

An in-test scalar Newton iteration, written with explicit partial-likelihood
derivatives, reproduces these values to 8 decimals.

### Tied event times

- **Breslow** compares every tied event with the full risk set.
- **Efron** removes the tied events from the risk set in equal fractions, so
  term `r` of `d` uses `S0(R) - (r/d) S0(D)`.

Without ties the two coincide. With ties, Breslow's estimate is pulled toward
zero. Tie example from the tests (8 subjects, ties at `t = 1, 2, 4`):

| Method | `beta` | se | log partial likelihood at `beta = 0.4` |
|---|---|---|---|
| Breslow | 0.818 | 0.873 | -9.1853 |
| Efron | 0.887 | 0.874 | -8.8292 |

Efron is the default because it is closer to the exact discrete-time
likelihood.

## Limitations

- Only right censoring is supported. Left truncation (delayed entry),
  interval censoring, time-varying covariates, strata, frailties, and
  competing risks are not implemented.
- All intervals and tests are large-sample approximations: Greenwood, delta
  method, Wald, and chi-square. With few events, treat them as rough.
- The Kaplan-Meier intervals are pointwise. No simultaneous confidence band
  (Hall-Wellner, equal precision) is provided.
- The Cox model does not test the proportional-hazards assumption (for
  example with Schoenfeld residuals). Complete separation shows up as
  `converged=False` with a huge coefficient; no Firth correction is applied.
- The AFT standard errors use a central-difference Jacobian of the analytic
  gradient, so they are accurate to about six digits, not to machine
  precision.

## Sources

[REF-KAPLAN-MEIER-1958], [REF-GREENWOOD-1926], [REF-NELSON-1972],
[REF-AALEN-1978], [REF-MANTEL-1966], [REF-COX-1972], [REF-BRESLOW-1974],
[REF-EFRON-1977], [REF-KLEIN-MOESCHBERGER-2003].
