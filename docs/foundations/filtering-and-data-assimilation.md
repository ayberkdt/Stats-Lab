# Filtering, Smoothing, and Data Assimilation

State estimation asks what a dynamical system is doing *now* (filtering) or
*was doing* (smoothing) given noisy, indirect observations. This guide covers
`probstat.filtering` and `probstat.data_assimilation`. All numbers below were
produced by the library itself with the seeds stated; re-running the calls
reproduces them.

## 1. State-space models

A discrete-time state-space model has two equations [EQ-FILT-SSM-001]:

- **dynamics** `x_k = F x_(k-1) + B u_k + w_k`, with process noise
  `w_k ~ N(0, Q)`;
- **observation** `y_k = H x_k + v_k`, with measurement noise `v_k ~ N(0, R)`.

The prior state is `x_0 ~ N(m_0, P_0)`; observations `y_1..y_T` follow the
transitions. `LinearGaussianStateSpace` holds `F, Q, H, R, m_0, P_0` and an
optional control matrix `B`; `NonlinearStateSpace` replaces `F x` and `H x` by
callables `f(x)` and `h(x)` with optional analytic Jacobians
[EQ-FILT-SSM-002]. `simulate(model, T, seed=...)` draws a true trajectory and
observations, using separate random streams for the initial state, the
process noise, and the measurement noise.

**Process versus measurement noise.** `Q` says how much the dynamics are
trusted; `R` says how much the sensor is trusted. They push the gain in
opposite directions. For the scalar random walk (`F = H = 1`) the steady-state
gain from `steady_state_kalman` [EQ-FILT-KF-005] is:

| `q` | `r` | steady predicted variance `P^-` | gain `K` | filtered variance |
|---|---|---|---|---|
| 0.5 | 2.0 | 1.2808 | 0.3904 | 0.7808 |
| 2.0 | 2.0 | 3.2361 | 0.6180 | 1.2361 |
| 0.5 | 8.0 | 2.2656 | 0.2207 | 1.7656 |

More process noise means older information ages faster, so the filter leans
harder on new data (larger `K`). More measurement noise means each reading is
less informative (smaller `K`). For `q = r = 2` the predicted variance is
`1 + sqrt(5)` and the gain is the golden-ratio conjugate `0.6180`, matching
the closed form `P^- = (q + sqrt(q^2 + 4qr))/2`.

## 2. Sequential Bayesian updating

Every filter here implements, exactly or approximately, the two-step
recursion [EQ-FILT-BAYES-001]: **predict** with the dynamics, then **update**
with Bayes' rule using the new observation. For linear-Gaussian models both
steps stay Gaussian and the recursion is the Kalman filter.

- Prediction / covariance propagation [EQ-FILT-KF-001]:
  `m^- = F m + B u`, `P^- = F P F' + Q` (`kalman_predict`).
- Measurement update [EQ-FILT-KF-002]: innovation `v = y - H m^-`,
  `S = H P^- H' + R`, gain `K = P^- H' S^-1`, `m = m^- + K v`
  (`kalman_update`). The gain is formed by Cholesky solves.
- Covariance update in Joseph form [EQ-FILT-KF-003]:
  `P = (I - K H) P^- (I - K H)' + K R K'`, which stays symmetric positive
  semidefinite under rounding.

**Worked example.** Random walk with `q = 0.5`, `r = 2`, `m_0 = 0`,
`P_0 = 10`, observations `1.0, 2.0, 1.5`:

| step | `m^-` | `P^-` | `K` | `m` | `P` | log-likelihood term |
|---|---|---|---|---|---|---|
| 1 | 0.0000 | 10.5000 | 0.8400 | 0.8400 | 1.6800 | -2.2218 |
| 2 | 0.8400 | 2.1800 | 0.5215 | 1.4450 | 1.0431 | -1.7951 |
| 3 | 1.4450 | 1.5431 | 0.4355 | 1.4689 | 0.8710 | -1.5519 |

The first gain is large because the vague prior (`P_0 = 10`) is overwhelmed
by the first reading; the gain then shrinks towards the steady value 0.3904.
The sum of the last column is the log marginal likelihood
[EQ-FILT-KF-004], returned as `KalmanFilterResult.log_likelihood`. A `None`
observation is treated as missing: the update is skipped and the likelihood
term is zero.

`tests/test_uq7_filtering.py` checks the filter against batch Gaussian
conditioning: stacking `x_0..x_4` of a constant-velocity model into one
Gaussian vector and conditioning it with `inference.condition_linear_gaussian`
gives the same last filtered state, the same smoothed marginals, and the same
log-likelihood to nine decimals.

## 3. Filtering versus smoothing

The filter gives `p(x_k | y_1:k)`; the Rauch-Tung-Striebel smoother
(`rts_smoother`) runs backwards to give `p(x_k | y_1:T)` [EQ-FILT-RTS-001].
Because it also uses future observations, its marginal variances are never
larger than the filter's; at the last step the two coincide. For the worked
example above the smoothed means and variances are:

| `k` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| smoothed mean | 1.2559 | 1.3187 | 1.4612 | 1.4689 |
| smoothed variance | 1.2221 | 0.8224 | 0.7360 | 0.8710 |

At `k = 1` the smoothed variance 0.8224 is half the filtered 1.6800.

**Constant-velocity tracking.** State `(position, velocity)`, `dt = 1`,
white-noise acceleration `Q = q [[1/3, 1/2], [1/2, 1]]` with `q = 0.01`, and
position measured with `r = 4` [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001], prior
mean `(0, 1)` and prior covariance `diag(1, 0.5)`. Over 150 steps
(`simulate(..., seed=11)`):

| estimate | position RMSE | velocity RMSE |
|---|---|---|
| raw measurements | 2.050 | not measured |
| Kalman filter | 1.095 | 0.285 |
| RTS smoother | 0.615 | 0.155 |

The filter recovers the velocity, which is never observed, from the
correlation that `F` builds between position and velocity. 94.7% of the
filtered position errors fall inside two filtered standard deviations, close
to the nominal 95%.

<picture><source media="(prefers-color-scheme: dark)" srcset="../images/guides/filtering-hidden-velocity-dark.png"><img alt="True velocity, Kalman-filtered velocity with its 95% band, and the smoothed velocity over 150 time steps" src="../images/guides/filtering-hidden-velocity.png"></picture>

*Only position is measured, yet the filter infers the velocity from the correlation the dynamics build between the two states, with RMSE 0.285; the smoother, which also uses later observations, lowers it to 0.155, and the filtered 95% band holds the true velocity at 93% of the steps.*

## 4. Nonlinear Gaussian filters

- **Extended Kalman filter** (`extended_kalman_filter`) linearizes `f` and
  `h` at the current mean [EQ-FILT-EKF-001]. Jacobians are analytic when
  supplied, otherwise central differences [EQ-OPT-FD-001].
- **Unscented Kalman filter** (`unscented_kalman_filter`) pushes `2n + 1`
  sigma points through the exact maps and re-fits mean and covariance
  [EQ-FILT-UKF-001], [EQ-FILT-UKF-002]. Default weights use `alpha = 1`,
  `beta = 2`, `kappa = 0`.

On a linear model both reduce to the Kalman filter; the tests confirm equal
log-likelihoods to ten (EKF) and nine (UKF) decimals. The unscented transform
reproduces `E[X^2] = m^2 + P` exactly. `rts_smoother` also smooths EKF and
UKF output because every filter result stores `Cov(x_(k-1), x_k^-)`.

Neither filter reports its own linearization error. Strong nonlinearity
relative to the spread of `P`, or a multimodal posterior, calls for a particle
filter.

## 5. Particle filtering

`bootstrap_particle_filter` [EQ-FILT-PF-001] represents the filtering law by
weighted samples, propagates them with sampled process noise, weights them by
the observation likelihood, and resamples systematically [EQ-FILT-PF-003]
when the effective sample size [EQ-FILT-PF-002] drops below
`resample_threshold * N` (default 0.5). On the 10-step random walk
(`simulate(..., seed=1)`, filter `seed=3`):

| particles | log-likelihood estimate | resampling steps | minimum ESS |
|---|---|---|---|
| exact (Kalman) | -19.6049 | — | — |
| 200 | -19.2668 | 3 | 69.8 |
| 2000 | -19.4587 | 3 | 616.8 |

The likelihood estimate is unbiased, but its logarithm is biased downward and
its error falls only like `N^-1/2`. Particle filters degenerate in high state
or observation dimension.

**Proposals that look at `y_k`.** The bootstrap filter moves particles blind
to the new observation, so precise observations leave few particles with
weight. Two classical remedies return the same `ParticleFilterResult`:

- `auxiliary_particle_filter` [EQ-FILT-APF-001] (Pitt and Shephard)
  resamples the previous particles with first-stage weights
  `W_(k-1)^i g(y_k | f(x_(k-1)^i))`, propagates, and corrects with
  second-stage weights `g(y_k | x_k^j) / g(y_k | f(x_(k-1)^(a_j)))`. Its
  likelihood increment is `[sum_i W_(k-1)^i g(y_k | mu^i)] (1/N) sum_j w_k^j`
  [REF-DOUCET-JOHANSEN-2011, Section 4.2].
- `optimal_proposal_particle_filter` [EQ-FILT-OPT-001] samples
  `p(x_k | x_(k-1), y_k)` exactly. For a linear observation operator it is a
  Gaussian with the Kalman gain `K = Q H' (H Q H' + R)^-1`, and the weight is
  the predictive density `N(y_k; H f(x_(k-1)), H Q H' + R)`, which does not
  depend on the new particle. A nonlinear `h` raises `ValueError`; a
  `NonlinearStateSpace` whose `h` is linear passes `observation_matrix=H`,
  which is checked against `h`.

On an AR(1) with precise observations (`F = 0.9`, `q = 1`, `r = 0.5`, 10
steps, `simulate(..., seed=4)`, exact log-likelihood -18.4401), 100 runs with
`N = 100` and seeds 0-99 give:

| filter | mean log-likelihood | sd of log-likelihood | mean of `exp(estimate - exact)` |
|---|---|---|---|
| bootstrap | -18.6027 | 0.4175 | 0.927 ± 0.040 |
| auxiliary | -18.8829 | 0.7901 | 0.950 ± 0.133 |
| optimal proposal | -18.4756 | 0.1869 | 0.982 ± 0.019 |

(± is one standard error over the 100 runs.) The optimal proposal cuts the
standard deviation of the log-likelihood by more than half. The auxiliary
filter is worse than the bootstrap here: its look-ahead `g(y_k | mu)` uses
only `R = 0.5`, while the true predictive `p(y_k | x_(k-1))` has variance
`q + r = 1.5`, so the second-stage weights are heavy-tailed. Doucet and
Johansen warn about exactly this point-estimate first stage; the auxiliary
filter helps when `R` dominates the predictive spread. All three ratio means
are consistent with 1 (unbiasedness); the shortfall of the bootstrap and
auxiliary means is the right skew of the ratio, whose sample mean falls below
its expectation in most finite samples. `tests/test_particle_extensions.py`
checks unbiasedness with 200 replications and four standard errors.

**Particle smoothing.** `ffbs_particle_smoother` [EQ-FILT-FFBS-001]
(Godsill, Doucet, and West) needs a filter run made with
`store_history=True`, which keeps the weighted cloud of every step (the flag
changes no random draw and no returned number). Each trajectory is drawn
backwards: `x~_T` from the last cloud, then `x~_k = x_k^i` with weights
`W_k^i N(x~_(k+1); f(x_k^i), Q)`. The cost is `O(N T)` per trajectory, and
`Q` must be positive definite. On the AR(1) above (bootstrap filter `N = 500`,
`seed=3`; 200 trajectories, `seed=4`):

| `k` | filter mean, variance | FFBS mean, variance | RTS mean, variance |
|---|---|---|---|
| 0 | 0.0993, 0.9059 | 0.8802, 0.6055 | 0.8710, 0.6313 |
| 5 | 0.8049, 0.3704 | 1.0342, 0.2980 | 1.0982, 0.2978 |
| 10 | 3.2662, 0.3617 | 3.2424, 0.3372 | 3.3427, 0.3605 |

At `k = 0` the smoother moves the prior mean 0.10 to 0.88 using all ten
observations, as the RTS smoother does. The differences between FFBS and RTS
are Monte Carlo error of one run; the tests average 30 independent runs and
compare means and variances with the RTS values within four standard errors.

## Parameter estimation in state-space models

The filters above assume known `F, Q, H, R`. `probstat.filtering` estimates
static parameters `theta` three ways, all built on the filter likelihood
`l(theta) = log p(y_1:T | theta)`.

**Maximum likelihood.** `fit_state_space_mle(builder, theta_0, y)` takes a
builder `theta -> model`, minimizes `-l(theta)` with `minimize_bfgs`
(Kalman filter by default; `filter="ekf"` or `"ukf"` for nonlinear models,
whose likelihood is then approximate), and inverts the observed information
from `central_hessian` [EQ-FILT-MLE-001]. Variances are best passed on the
log scale. For the Nile series (100 annual flows) with the local-level model
`x_k = x_(k-1) + w_k`, `y_k = x_k + v_k`, and R's `StructTS` initialization
`x_0 ~ N(y_1, 1e4 var(y))`, the fit from `log(var(y)/100)` converges in 20
BFGS iterations to

| | level variance `q` | noise variance `r` | log-likelihood |
|---|---|---|---|
| `fit_state_space_mle` | 1469.17 | 15098.52 | -643.200988 |
| R `StructTS(Nile, "level")` | 1469.15 | 15098.58 | -643.200988 |

The two optimizers stop at slightly different points of a flat likelihood
(the log-likelihoods differ by `2e-10`). The standard error of `log q` is
0.87, against 0.21 for `log r`: the level variance is poorly determined from
100 points.

**EM.** `linear_gaussian_em(model, y, estimate=...)` is the algorithm of
Shumway and Stoffer [EQ-FILT-EM-001]. The E-step is the RTS smoother with the
lag-one covariances `Cov(x_k, x_(k-1) | y_1:T) = P^s_k G_(k-1)'`
(`smoothed_lag_one_covariances`); the M-step updates any subset of
`"transition"`, `"observation"`, `"process_covariance"`,
`"observation_covariance"`, and `"initial"` in closed form. Each step can
only raise the exact log-likelihood, and the returned trace lets you check
that. On an AR(1) plus noise (`F = 0.8`, `q = 1`, `r = 0.5`, 100 steps,
`simulate(..., seed=2026)`), estimating `F`, `Q`, `R` from `(0.5, 0.5, 1)`
raises the log-likelihood from -184.0309 to -167.94368 in 161 iterations
(tolerance `1e-10`); BFGS reaches the same maximum:

| | `F` | `q` | `r` |
|---|---|---|---|
| EM | 0.77879 | 0.96884 | 0.50151 |
| `fit_state_space_mle` | 0.77875 | 0.96906 | 0.50135 |
| truth | 0.8 | 1.0 | 0.5 |

EM is simple and stable but converges linearly: on the Nile model it needs
211 iterations to reach tolerance `1e-10` from `(1000, 10000)`, where BFGS
needs 20.

**Particle marginal Metropolis-Hastings.** For nonlinear models, or to get a
posterior instead of a point estimate,
`particle_marginal_metropolis_hastings(builder, log_prior, theta_0, y, ...)`
[EQ-FILT-PMMH-001] runs random-walk Metropolis-Hastings with a fresh
particle-filter estimate `p_hat(y | theta')` at each proposal. The estimate of
the current state is stored and reused, never recomputed; with an unbiased
`p_hat` this makes the chain target the exact posterior for any particle count
(Andrieu, Doucet, and Holenstein). Fewer particles make the estimate noisier,
which shows up as a lower acceptance rate and stickier chains, not as bias.
The result is an `inference.MCMCResult`, so `summary()` and the convergence
diagnostics apply. `tests/test_filtering_parameter_estimation.py` samples the
AR coefficient of a 12-step AR(1) plus noise with a uniform prior, 20
particles, and the optimal proposal, and compares the posterior mean and
variance with quadrature of the exact Kalman posterior within four Monte
Carlo standard errors.

**Out of scope: Rao-Blackwellized particle filters.** They sample only the
nonlinear part of the state and handle a conditionally linear-Gaussian part
with one Kalman filter per particle. `StateSpaceModel` has no way to declare
that split, so this would need a new model class (ADR-0113).

## 6. Data assimilation

Data assimilation combines a model forecast with observations to estimate the
state of a large system. `probstat.data_assimilation` consumes the same model
objects structurally (it does not import `probstat.filtering`).

**Stochastic ensemble Kalman filter.** `ensemble_kalman_filter` replaces the
Kalman covariance with the sample covariance of `N` model runs. Each member
is forecast with its own process noise and updated with its own perturbed
observation `y + e_i`, `e_i ~ N(0, R)` [EQ-DA-ENKF-001]. Without the
perturbation the analysis ensemble would be too narrow. On the
constant-velocity model (`q = 0.05`, `r = 1`, the same prior, 5 steps,
`seed=7` data, `seed=0` filter), the final analysis compared with the exact
filter is:

| | mean (pos, vel) | variance (pos, vel) |
|---|---|---|
| Kalman filter | (4.738, 0.636) | (0.537, 0.138) |
| EnKF, `N = 50` | (4.609, 0.675) | (0.670, 0.121) |
| EnKF, `N = 1000` | (4.716, 0.651) | (0.527, 0.135) |

The ensemble covariance has rank at most `N - 1` and relative error of order
`N^-1/2` [EQ-DA-ENKF-002]. Small ensembles therefore produce sampling error,
spurious long-range correlations, and usually under-dispersion. The library
exposes multiplicative inflation (`inflation >= 1`, applied to forecast
anomalies) as an explicit parameter. It does not inflate or localize anything
silently, and localization is not implemented.

**3D-Var.** `three_dimensional_variational` minimizes the background plus
observation cost [EQ-DA-3DVAR-001] by BFGS with an analytic gradient. For a
linear observation operator the minimizer equals one Kalman update. With the
first constant-velocity prediction `(1, 1)` and observation `1.9959`, 3D-Var
returns `(1.60019, 1.20776)` after 3 iterations, the same as the Kalman update
to rounding error (about `2e-16`). For the nonlinear operator `h(x) = x^2` with `x_b = 1`,
`B = 0.5`, `y = 2`, `R = 0.1`, the analysis is `x_a = 1.4040` with cost
0.1674 and Gauss-Newton analysis variance 0.01237. Unlike the Kalman filter,
3D-Var keeps `B` fixed instead of propagating it.

## Limitations

- Gaussian filters assume additive Gaussian noise; the EKF and UKF are
  approximations whose error is not estimated.
- The particle filters cover the transition-prior (bootstrap), auxiliary,
  and locally optimal proposals; the optimal proposal needs a linear
  observation operator. Particle smoothing is FFBS only (`O(M N T)`, no
  two-filter or fixed-lag smoother), and there is no Rao-Blackwellization.
  Parameter estimation has no gradient of the particle likelihood, so
  maximum likelihood is for Gaussian filters and particle models use PMMH.
- The EnKF is the stochastic perturbed-observation variant; there are no
  square-root or deterministic variants, no localization, and no adaptive
  inflation.
- 3D-Var is a single-time analysis; 4D-Var is not implemented.

## Common pitfalls

- **Covariances, not standard deviations.** `Q`, `R`, and `P_0` enter
  `LinearGaussianStateSpace` as covariance matrices.
- **Observations are sequences.** A scalar reading is `(1.0,)`; a bare float
  raises `TypeError`. `None` marks a missing observation: its update is
  skipped and its log-likelihood term is zero.
- **Index offset.** `filtered[0]` is the prior state, so `filtered` has one
  entry more than `predicted`.
- **Inflation is a covariance factor.** `inflation=1.0` (the default) means no
  inflation; the anomalies are scaled by `sqrt(inflation)`, and a value below
  one raises. Nothing is localized.
- **Unseeded runs differ.** `simulate`, `bootstrap_particle_filter`, and
  `ensemble_kalman_filter` default to `seed=None`.

The random walk of Section 2:

```python
import inspect
from probstat.data_assimilation import ensemble_kalman_filter
from probstat.filtering import LinearGaussianStateSpace, kalman_filter

walk = LinearGaussianStateSpace(((1.0,),), ((0.5,),), ((1.0,),), ((2.0,),), (0.0,), ((10.0,),))
full = kalman_filter(walk, [(1.0,), (2.0,), (1.5,)])
round(full.log_likelihood, 4)                     # -5.5687
gap = kalman_filter(walk, [(1.0,), None, (1.5,)])
gap.log_likelihood_terms[1]                       # 0.0, missing observation
len(gap.filtered), len(gap.predicted)             # (4, 3)
gap.filtered[0].mean                              # (0.0,), the prior
try:
    kalman_filter(walk, [1.0, 2.0])
except TypeError as error:
    message = str(error)
message                                           # 'observation must be iterable'
inspect.signature(ensemble_kalman_filter).parameters["inflation"].default   # 1.0
```

## Sources

- [REF-KALMAN-1960], [REF-RAUCH-TUNG-STRIEBEL-1965].
- [REF-SARKKA-2013, Chapters 4, 5, 7, 8, and 12].
- [REF-SIMON-2006], [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].
- [REF-JULIER-UHLMANN-2004].
- [REF-GORDON-SALMOND-SMITH-1993], [REF-KITAGAWA-1996],
  [REF-DOUCET-JOHANSEN-2011].
- [REF-PITT-SHEPHARD-1999], [REF-DOUCET-GODSILL-ANDRIEU-2000],
  [REF-GODSILL-DOUCET-WEST-2004].
- [REF-SHUMWAY-STOFFER-1982], [REF-DURBIN-KOOPMAN-2012],
  [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010].
- [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998],
  [REF-EVENSEN-2003], [REF-LORENC-1986].
