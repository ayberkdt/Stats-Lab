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
position measured with `r = 4` [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001]. Over 150
steps (`simulate(..., seed=11)`):

| estimate | position RMSE | velocity RMSE |
|---|---|---|
| raw measurements | 2.050 | not measured |
| Kalman filter | 1.095 | 0.285 |
| RTS smoother | 0.615 | 0.155 |

The filter recovers the velocity, which is never observed, from the
correlation that `F` builds between position and velocity. 94.7% of the
filtered position errors fall inside two filtered standard deviations, close
to the nominal 95%.

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

## 6. Data assimilation

Data assimilation combines a model forecast with observations to estimate the
state of a large system. `probstat.data_assimilation` consumes the same model
objects structurally (it does not import `probstat.filtering`).

**Stochastic ensemble Kalman filter.** `ensemble_kalman_filter` replaces the
Kalman covariance with the sample covariance of `N` model runs. Each member
is forecast with its own process noise and updated with its own perturbed
observation `y + e_i`, `e_i ~ N(0, R)` [EQ-DA-ENKF-001]. Without the
perturbation the analysis ensemble would be too narrow. On the
constant-velocity model (`q = 0.05`, `r = 1`, 5 steps, `seed=7` data,
`seed=0` filter), the final analysis compared with the exact filter is:

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
returns `(1.60019, 1.20776)` after 9 iterations, the same as the Kalman update
to about `1e-11`. For the nonlinear operator `h(x) = x^2` with `x_b = 1`,
`B = 0.5`, `y = 2`, `R = 0.1`, the analysis is `x_a = 1.4040` with cost
0.1674 and Gauss-Newton analysis variance 0.01237. Unlike the Kalman filter,
3D-Var keeps `B` fixed instead of propagating it.

## Limitations

- Gaussian filters assume additive Gaussian noise; the EKF and UKF are
  approximations whose error is not estimated.
- The particle filter uses the transition prior as proposal; there are no
  auxiliary or optimal proposals, no particle smoothing, and no
  Rao-Blackwellization.
- The EnKF is the stochastic perturbed-observation variant; there are no
  square-root or deterministic variants, no localization, and no adaptive
  inflation.
- 3D-Var is a single-time analysis; 4D-Var is not implemented.

## Sources

- [REF-KALMAN-1960], [REF-RAUCH-TUNG-STRIEBEL-1965].
- [REF-SARKKA-2013, Chapters 4, 5, 7, 8, and 12].
- [REF-SIMON-2006], [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].
- [REF-JULIER-UHLMANN-2004].
- [REF-GORDON-SALMOND-SMITH-1993], [REF-KITAGAWA-1996],
  [REF-DOUCET-JOHANSEN-2011].
- [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998],
  [REF-EVENSEN-2003], [REF-LORENC-1986].
