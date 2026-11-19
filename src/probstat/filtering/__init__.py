"""State-space filtering and smoothing (sequential Bayesian state estimation).

``state_space`` defines linear-Gaussian and nonlinear additive-noise models;
``kalman`` provides the exact Kalman filter, RTS smoother, and steady-state
Riccati solution; ``extended`` provides the EKF and UKF; ``particle`` provides
the bootstrap particle filter; ``particle_proposals`` the auxiliary and
locally optimal particle filters; ``particle_smoothing`` the
forward-filtering backward-smoothing particle smoother; and
``parameter_estimation`` maximum likelihood, EM, and particle marginal
Metropolis-Hastings for static parameters. Ensemble and variational data
assimilation live in :mod:`probstat.data_assimilation`.

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
    ``auxiliary_particle_filter``: Pitt-Shephard look-ahead particle filter.
    ``optimal_proposal_particle_filter``: particle filter with the locally
    optimal proposal (linear observation operator).
    ``ffbs_particle_smoother``: smoothed trajectories from a stored particle
    filter run.
    ``fit_state_space_mle``: maximum-likelihood parameters with an
    observed-information covariance.
    ``linear_gaussian_em``: Shumway-Stoffer EM for linear-Gaussian models.
    ``particle_marginal_metropolis_hastings``: exact-posterior MCMC with a
    particle likelihood estimate.
    ``simulate``: seeded true trajectory and observations.

Assumptions:
    Process and measurement noise are additive, Gaussian, zero-mean, and
    independent of each other and over time; the model matrices and
    covariances are known. The Kalman filter is exact only for linear
    models. The EKF and UKF are Gaussian approximations whose error is not
    estimated; the bootstrap particle filter uses the transition prior as
    proposal, and the locally optimal proposal needs a linear observation
    operator. Parameter estimation assumes the model is identifiable in the
    parameters supplied to the builder.

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
    - Particle smoothing needs ``store_history=True`` on the filter run and a
      positive-definite ``Q``.
    - ``fit_state_space_mle`` reports ``covariance=None`` (with a
      ``message``) when the observed information is not positive definite or
      an estimate sits on a bound; nothing is regularized.
    - A PMMH chain mixes poorly when the log-likelihood estimate has a
      standard deviation well above one; increase ``particles``.
    - Without analytic Jacobians the EKF uses central differences with the
      relative step ``jacobian_step`` (default ``1e-6``) of
      ``NonlinearStateSpace``.

Guide: docs/foundations/filtering-and-data-assimilation.md

References:
    [REF-KALMAN-1960], [REF-RAUCH-TUNG-STRIEBEL-1965], [REF-SARKKA-2013],
    [REF-JULIER-UHLMANN-2004], [REF-GORDON-SALMOND-SMITH-1993],
    [REF-PITT-SHEPHARD-1999], [REF-GODSILL-DOUCET-WEST-2004],
    [REF-SHUMWAY-STOFFER-1982], [REF-ANDRIEU-DOUCET-HOLENSTEIN-2010].

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
from .parameter_estimation import (
    EMResult,
    StateSpaceFit,
    fit_state_space_mle,
    linear_gaussian_em,
    particle_marginal_metropolis_hastings,
    smoothed_lag_one_covariances,
)
from .particle import (
    ParticleFilterResult,
    bootstrap_particle_filter,
    effective_sample_size,
    systematic_resample,
)
from .particle_proposals import (
    auxiliary_particle_filter,
    optimal_proposal_particle_filter,
)
from .particle_smoothing import ParticleSmootherResult, ffbs_particle_smoother
from .state_space import (
    LinearGaussianStateSpace,
    NonlinearStateSpace,
    StateSpaceModel,
    StateSpaceSimulation,
    simulate,
)

__all__ = [
    "EMResult",
    "GaussianState",
    "KalmanFilterResult",
    "LinearGaussianStateSpace",
    "MeasurementUpdate",
    "NonlinearStateSpace",
    "ParticleFilterResult",
    "ParticleSmootherResult",
    "SmootherResult",
    "StateSpaceFit",
    "StateSpaceModel",
    "StateSpaceSimulation",
    "SteadyStateKalman",
    "UnscentedTransformResult",
    "UnscentedWeights",
    "auxiliary_particle_filter",
    "bootstrap_particle_filter",
    "effective_sample_size",
    "extended_kalman_filter",
    "ffbs_particle_smoother",
    "fit_state_space_mle",
    "gain_from_cross_covariance",
    "gaussian_log_density",
    "kalman_filter",
    "kalman_predict",
    "kalman_update",
    "linear_gaussian_em",
    "linear_gaussian_update",
    "optimal_proposal_particle_filter",
    "particle_marginal_metropolis_hastings",
    "rts_smoother",
    "simulate",
    "smoothed_lag_one_covariances",
    "steady_state_kalman",
    "systematic_resample",
    "unscented_kalman_filter",
    "unscented_transform",
    "unscented_update",
]
