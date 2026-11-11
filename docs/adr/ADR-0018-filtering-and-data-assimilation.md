# ADR-0018 — Filtering and data assimilation

- Status: Accepted
- Date: 2026-10-11

## Context

UQ-7 (ATHENS-8.1 to 8.15) needs dynamic state estimation: state-space models,
prediction and update steps, process versus measurement noise, sequential
Bayesian updating, filtering versus smoothing, the Kalman, extended, ensemble,
and particle filters, and data assimilation. The library must stay pure
Python, reuse the shared `linalg` and `optimization` layers, and keep the
package graph acyclic.

## Decision

1. Two new layer-6 packages: `probstat.filtering` (state-space models, KF,
   RTS, steady-state Riccati, EKF, UKF, bootstrap particle filter) and
   `probstat.data_assimilation` (stochastic EnKF, 3D-Var). Both depend only on
   `foundations`, `linalg`, and `optimization`. Neither imports the other:
   `data_assimilation` consumes models through the structural protocol
   `AdditiveGaussianModel`, which `LinearGaussianStateSpace` and
   `NonlinearStateSpace` satisfy. `visualization` moves to layer 9 so the gap
   above leaves room for later layers.
2. One time convention everywhere: `x_0 ~ N(m_0, P_0)` is the prior before
   the first observation, and observations `y_1..y_T` follow transitions.
   Filter results store filtered states for `k = 0..T` and predictions for
   `k = 1..T`.
3. Kalman covariance updates use the Joseph form. Gains are computed by
   Cholesky solves, never by explicit inversion. Indefinite covariances raise
   `ValueError`; there is no hidden jitter or repair. `R` must be positive
   definite, while `Q` and `P_0` may be singular positive semidefinite.
4. Every Gaussian filter result stores `Cov(x_(k-1), x_k^-)`, so one
   `rts_smoother` serves the KF, EKF (Jacobian cross-covariance), and UKF
   (sigma-point cross-covariance).
5. A `None` observation means a missing measurement: the update is skipped
   and the likelihood term is zero. Filters, smoother, PF, and EnKF all follow
   this convention.
6. The EnKF is the stochastic perturbed-observation variant. Multiplicative
   inflation is an explicit argument (`>= 1`). Localization and square-root
   variants are not implemented and are not emulated silently.
7. The particle filter is the bootstrap filter with systematic resampling
   triggered by an explicit ESS fraction. Moments are recorded before
   resampling, and the log-likelihood increment uses normalized previous
   weights, so adaptive resampling stays correct.
8. Seeds are `int | RandomStream | None`, and every stochastic source uses a
   named child stream (`initial`, `process`/`propagation`, `measurement`,
   `resampling`, `forecast`/`analysis` per step).
9. Inner sampling loops (particles, ensemble members) use private
   plain-summation helpers and skip per-component validation of internally
   generated vectors. The public model methods keep full validation.

## Consequences

The ATHENS-8 topics have tested implementations. Cross-checks tie them
together: KF equals batch Gaussian conditioning, EKF and UKF equal KF on
linear models, EnKF and PF converge to KF moments and likelihood. Run time
limits sizes to small state dimensions and thousands of particles or members.
The implementation targets teaching and verification, not operational
assimilation.

## Rejected alternatives

- Having `data_assimilation` import `filtering`: this violates the "import
  from layers <= 5" rule for UQ-7 packages and couples two siblings.
- The standard `(I - K H) P` covariance update: it is cheaper but can lose
  symmetry and definiteness under rounding.
- Automatic covariance inflation or jitter: it hides divergence instead of
  reporting it.
- Multinomial resampling as the default: it adds avoidable resampling noise.
