"""State-space filtering and smoothing (sequential Bayesian state estimation).

``state_space`` defines linear-Gaussian and nonlinear additive-noise models;
``kalman`` provides the exact Kalman filter, RTS smoother, and steady-state
Riccati solution; ``extended`` provides the EKF and UKF; ``particle`` provides
the bootstrap particle filter. Ensemble and variational data assimilation live
in :mod:`probstat.data_assimilation`.
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
