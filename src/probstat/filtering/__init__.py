"""State-space filtering and smoothing (sequential Bayesian state estimation).

``state_space`` defines linear-Gaussian and nonlinear additive-noise models;
``kalman`` provides the exact Kalman filter, RTS smoother, and steady-state
Riccati solution; ``extended`` provides the EKF and UKF; ``particle`` provides
the bootstrap particle filter. Ensemble and variational data assimilation live
in :mod:`probstat.data_assimilation`.

Main entry points:
    ``LinearGaussianStateSpace``: linear dynamics and observations with
    Gaussian noise.
    ``NonlinearStateSpace``: nonlinear maps with additive Gaussian noise.
    ``kalman_filter``: exact filtering and log marginal likelihood of a
    linear-Gaussian model.
    ``rts_smoother``: Rauch-Tung-Striebel smoothing of any filter result.
    ``extended_kalman_filter``: linearized filter for nonlinear models.
    ``unscented_kalman_filter``: sigma-point filter for nonlinear models.
    ``bootstrap_particle_filter``: sequential Monte Carlo filter.
    ``simulate``: seeded true trajectory and observations.

Assumptions:
    Process and measurement noise are additive, Gaussian, zero-mean, and
    independent of each other and over time; the model matrices and
    covariances are known. The Kalman filter is exact only for linear
    models. The EKF and UKF are Gaussian approximations whose error is not
    estimated; the particle filter uses the transition prior as proposal.

Common pitfalls:
    - ``process_covariance``, ``observation_covariance``, and
      ``initial_covariance`` are covariance matrices (variances), not
      standard deviations.
    - Every observation is a sequence, even for a scalar sensor (``(1.0,)``);
      ``None`` marks a missing observation, whose update is skipped and whose
      log-likelihood term is zero.
    - ``filtered`` has one more entry than ``predicted``: ``filtered[0]`` is
      the prior state and ``filtered[k]`` the state after observation ``k``.
    - ``bootstrap_particle_filter`` and ``simulate`` default to
      ``seed=None``; the particle log-likelihood is itself a random estimate.
    - Without analytic Jacobians the EKF uses central differences with the
      relative step ``jacobian_step`` (default ``1e-6``) of
      ``NonlinearStateSpace``.

Guide: docs/foundations/filtering-and-data-assimilation.md

References:
    [REF-KALMAN-1960], [REF-RAUCH-TUNG-STRIEBEL-1965], [REF-SARKKA-2013],
    [REF-JULIER-UHLMANN-2004], [REF-GORDON-SALMOND-SMITH-1993].

"""

from .extended import (
    UnscentedTransformResult,
    UnscentedWeights,
    extended_kalman_filter,
    unscented_kalman_filter,
    unscented_transform,
    unscented_update,
)
from .kalman import (
    GaussianState,
    KalmanFilterResult,
    MeasurementUpdate,
    SmootherResult,
    SteadyStateKalman,
    gain_from_cross_covariance,
    gaussian_log_density,
    kalman_filter,
    kalman_predict,
    kalman_update,
    linear_gaussian_update,
    rts_smoother,
    steady_state_kalman,
)
from .particle import (
    ParticleFilterResult,
    bootstrap_particle_filter,
    effective_sample_size,
    systematic_resample,
)
from .state_space import (
    LinearGaussianStateSpace,
    NonlinearStateSpace,
    StateSpaceModel,
    StateSpaceSimulation,
    simulate,
)

__all__ = [
    "GaussianState",
    "KalmanFilterResult",
    "LinearGaussianStateSpace",
    "MeasurementUpdate",
    "NonlinearStateSpace",
    "ParticleFilterResult",
    "SmootherResult",
    "StateSpaceModel",
    "StateSpaceSimulation",
    "SteadyStateKalman",
    "UnscentedTransformResult",
    "UnscentedWeights",
    "bootstrap_particle_filter",
    "effective_sample_size",
    "extended_kalman_filter",
    "gain_from_cross_covariance",
    "gaussian_log_density",
    "kalman_filter",
    "kalman_predict",
    "kalman_update",
    "linear_gaussian_update",
    "rts_smoother",
    "simulate",
    "steady_state_kalman",
    "systematic_resample",
    "unscented_kalman_filter",
    "unscented_transform",
    "unscented_update",
]
