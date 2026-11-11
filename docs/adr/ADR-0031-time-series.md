# ADR-0031 — Time-series analysis package

- Status: Accepted
- Date: 2026-10-12

## Context

The library had state-space filtering (`probstat.filtering`, layer 6) and
hypothesis tests, but no tools for univariate time series: correlograms,
portmanteau tests, ARIMA estimation and forecasting, or exponential
smoothing. The package stays pure Python with no runtime dependencies. It must
take seeds through `RandomStream`, and it must reuse the existing optimizer,
the chi-square tail, and the Kalman machinery instead of re-implementing them.

## Decision

1. **New package `probstat.timeseries` at layer 7.** It may import
   `filtering`, `inference`, `optimization`, `linalg`, `modeling`, and the
   foundations. It does not import `workflow` and has no `visualization`
   dependency. It has four modules: `descriptive`, `polynomials`, `arma`, and
   `smoothing`.
2. **Sign convention.** The AR polynomial is `phi(z) = 1 - sum phi_j z^j` and
   the MA polynomial is `theta(z) = 1 + sum theta_j z^j`, as in
   Box-Jenkins-Reinsel-Ljung and Brockwell-Davis. The mean `mu` belongs to
   the differenced series, so it acts as a drift when `d > 0`. `fit_arima`
   includes it by default only when `d = 0`.
3. **Two independent root routes.** The stationarity and invertibility
   decision uses the Schur-Cohn step-down recursion, which is exact and needs
   no root finding. Durand-Kerner with Newton polishing reports the complex
   roots and minimum modulus. Durand-Kerner was chosen over companion-matrix
   QR because `linalg` has only a symmetric eigensolver, and a robust
   nonsymmetric QR with shifts is a larger, riskier addition. Durand-Kerner
   matched `numpy.roots` to `6e-15` on simple roots and to `5e-10` on a
   double root, where convergence is linear.
4. **Exact likelihood through the Harvey state space.** The model is
   `r = max(p, q+1)`, `y_t - mu = Z alpha_t`, with no measurement noise. The
   stationary initial covariance comes from an exact Lyapunov solve. This
   choice has one consequence:
   - **`filtering.LinearGaussianStateSpace` requires a positive-definite
     observation covariance**, so a pure ARMA cannot be passed to
     `filtering.kalman_filter`. `filtering` was left unchanged: relaxing that
     contract would touch the simulation, particle, and validation paths.
     Instead, `ARMAStateSpace.kalman` runs the predict step itself and calls
     `filtering.linear_gaussian_update`, the same Joseph-form update and log
     density that `kalman_filter` uses. It returns a
     `filtering.KalmanFilterResult`.
   - With positive observation noise, `ARMAStateSpace.to_linear_gaussian(r)`
     builds a real `LinearGaussianStateSpace`. The test checks that
     `kalman_filter` and `ARMAStateSpace.kalman` then agree; they matched to
     `0.0`.
   - Estimation uses a scalar specialization (`Z = e_1`), which costs
     `O(r^2)` per step. The covariance is frozen once it changes by less
     than `1e-13` relative. Its log-likelihood equals the general path and a
     dense Toeplitz MVN density to at most `6.8e-13`.
5. **MLE parameterization.** BFGS (`optimization.minimize_bfgs`) works on `u`
   with reflection coefficients `tanh(u)`, mapped to AR coefficients by the
   step-up recursion. MA coefficients are minus the step-up image, which
   keeps them invertible. `sigma^2` is concentrated out. Yule-Walker gives
   the AR starting values; MA starts at zero. Standard errors come from the
   inverse central-difference Hessian in `(phi, theta, mu, log sigma^2)`,
   with the delta method for `sigma^2`. They are `None` when the Hessian is
   not positive definite or a step leaves the stationary region.
   Convergence flags from BFGS are passed through as given. For example,
   ARIMA(1,1,1) fitted to a pure random walk (`seed=3`, n = 300) ends with
   `converged=False`, message "iteration limit reached", and
   `phi = 0.068`, `theta = -0.081`. This is the near-common-factor ridge
   `phi ~ -theta`, where the parameters are not identified.
6. **Forecast intervals are exact given the parameters.** The full
   conditional covariance of the future differenced values comes from the
   state space. Integration uses the weights `binom(h-j+d-1, d-1)`. The first
   `d` observations are treated as fixed (diffuse). Parameter uncertainty is
   not propagated.
7. **Exponential smoothing optimizes `alpha` and `beta` on the logit scale.**
   It concentrates the initial level and trend by exact linear least squares,
   because the one-step errors are affine in them. This is a project choice;
   it removes heuristic starting values. Only point forecasts are returned.
8. **STL was not implemented.** The brief allowed it only if correct. A
   faithful STL needs loess inner and outer loops, low-pass filtering, and
   robustness weights. No independent reference implementation (R or
   statsmodels) was available to verify it, so it was left out rather than
   shipped unverified. `REF-CLEVELAND-ETAL-1990` was not added for the same
   reason.

## Consequences

- `tests/test_timeseries.py` has 30 tests that run in about 2.7 s. They
  cover the AR(1) ACF against `phi^k`, the PACF cut-off, Ljung-Box on white
  noise versus AR, the Kalman likelihood against the closed-form AR(1)
  likelihood and against `kalman_filter`, AR(2) and MA(1) recovery within
  3 SE, `sqrt(h)` interval growth for ARIMA(0,1,0), stationarity rejection,
  and smoothing on a constant plus noise.
- Seasonal ARIMA, exogenous regressors, missing observations, the diffuse
  likelihood for `d > 0` in levels, and smoothing intervals are out of scope.
- Several bibliographic locators are chapter-level and were written from
  memory, not checked against the books: Jones 1980, Monahan 1984,
  Kerner 1966, and the Holt 2004 reprint. They should be verified before
  publication.
