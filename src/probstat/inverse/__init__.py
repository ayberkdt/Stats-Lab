"""Observation models and auditable inverse-problem calibration workflows.

``calibration`` holds the linear-Gaussian intercept-slope calibration with its
identifiability diagnostic. ``nonlinear`` declares a calibration problem on a
:class:`~probstat.modeling.ProbabilisticModel` whose joint input law is the
prior, with least squares, Fisher identifiability, the Laplace approximation,
and MCMC. ``discrepancy`` adds a Gaussian-process model discrepancy
(Kennedy-O'Hagan) and heteroscedastic noise, and ``prediction`` pushes
posterior draws through the forward model. ``regularization`` holds the
classical deterministic methods for ill-posed problems (Tikhonov, truncated
SVD, and the L-curve, GCV, and discrepancy parameter choices),
``test_problems`` discretized ill-posed problems with exact solutions, and
``ensemble_inversion`` the derivative-free ensemble Kalman inversion and
ES-MDA.

Main entry points:
    ``CalibrationProblem``: nonlinear calibration of model parameters from
    experiments, one ``NoiseModel`` per observed quantity.
    ``Experiment``: observed values with the conditions they were taken at.
    ``NoiseModel``: known or uncertain Gaussian or Student-t measurement
    noise, with an optional discrepancy variance and, for Gaussian noise,
    an optional ``ErrorCorrelation`` of the errors across experiments.
    ``predict``: posterior predictions under new conditions, latent or with
    measurement noise.
    ``DiscrepancyCalibration``: calibration with a Gaussian-process
    discrepancy over declared inputs.
    ``fisher_identifiability``: local identifiability from the prior-scaled
    Fisher information.
    ``calibrate_linear_gaussian``: conjugate posterior of an intercept and a
    slope under known noise.
    ``tikhonov``: Tikhonov solution by QR of ``[A; lambda L]``, with filter
    factors for ``L = I``; ``nonlinear_tikhonov`` for a nonlinear residual.
    ``truncated_svd`` and ``picard_coefficients``: SVD filtering and the
    discrete Picard condition.
    ``l_curve``, ``generalized_cross_validation``,
    ``discrepancy_principle``: choice of the Tikhonov parameter.
    ``backward_heat_problem``, ``second_derivative_problem``: test problems
    with exact solutions.
    ``ensemble_kalman_inversion``: derivative-free misfit minimization by
    iterated ensemble Kalman updates, with discrepancy-principle stopping.
    ``ensemble_smoother_mda``: ES-MDA, exact for linear-Gaussian problems as
    the ensemble grows; ``EnsembleInversionResult`` holds either run.

Assumptions:
    The forward model is deterministic given its inputs. Measurement errors
    are independent across observations unless a ``NoiseModel`` declares an
    ``ErrorCorrelation`` (Gaussian, per quantity, across experiments);
    ``DiscrepancyCalibration`` adds a correlated model discrepancy, not
    correlated measurement errors. The prior is
    the model's joint input law. The Fisher diagnostic is local and assumes
    Gaussian errors; the Laplace approximation is a Gaussian at the mode.
    Predictions outside the calibration domain are conditional mathematics;
    the data cannot confirm the model form there.

Common pitfalls:
    - ``predict`` defaults to ``include_noise=False``: its intervals cover the
      latent mean response, not a future measurement.
    - ``CalibrationProblem.fit_least_squares`` ignores the prior; it is the
      frequentist baseline, not a MAP estimate.
    - ``Experiment`` conditions may not name a calibration parameter, and an
      observed quantity without a ``NoiseModel`` is rejected; both raise
      ``ValueError`` when the problem is built.
    - ``calibrate_linear_gaussian`` raises for an unidentifiable design but
      accepts a weakly conditioned one; the message is only in
      ``identifiability.warning``, and no Python warning is emitted.
    - ``sample_posterior`` and ``predict`` default to ``seed=None`` (fresh
      entropy); pass an integer seed to reproduce draws.
    - ``tikhonov`` takes ``lambda`` and penalizes ``lambda^2 ||L x||^2``: a
      ridge weight ``alpha`` corresponds to ``lambda = sqrt(alpha)``.
    - ``discrepancy_principle`` needs the noise NORM ``delta ~ s sqrt(m)``,
      not the noise standard deviation ``s``.
    - ``nonlinear_tikhonov`` centers the penalty at ``x0`` unless
      ``prior_mean`` is given.
    - ``ensemble_kalman_inversion`` is an optimizer: its ensemble collapses,
      so its spread is not a posterior uncertainty. Both ensemble methods
      accept known-scale Gaussian noise only and raise ``ValueError``
      otherwise.

Guide: docs/foundations/inverse-problems-and-calibration.md
(deterministic regularization:
docs/foundations/regularization-and-ill-posed-problems.md)

References:
    [REF-TARANTOLA-2005], [REF-GELMAN-ETAL-2013], [REF-KENNEDY-OHAGAN-2001],
    [REF-BRYNJARSDOTTIR-OHAGAN-2014], [REF-HAARIO-SAKSMAN-TAMMINEN-2001],
    [REF-TIKHONOV-1963], [REF-HANSEN-2008], [REF-HANSEN-OLEARY-1993],
    [REF-IGLESIAS-LAW-STUART-2013], [REF-EMERICK-REYNOLDS-2013].

"""

from .calibration import (
    IdentifiabilityDiagnostic,
    LinearCalibrationResult,
    PredictionTarget,
    calibrate_linear_gaussian,
    diagnose_linear_identifiability,
)
from .discrepancy import (
    DISCREPANCY_KERNELS,
    DiscrepancyCalibration,
    DiscrepancyFit,
    DiscrepancyIdentifiability,
    DiscrepancyPosterior,
    DiscrepancyPrediction,
    DiscrepancyProcess,
    KnownNoise,
    LogLinearNoise,
    discrepancy_kernel,
)
from .ensemble_inversion import (
    EnsembleInversionResult,
    ensemble_kalman_inversion,
    ensemble_smoother_mda,
)
from .error_correlation import (
    ERROR_CORRELATION_KERNELS,
    ErrorCorrelation,
    error_correlation_kernel,
)
from .nonlinear import (
    CALIBRATION_GRADIENTS,
    CalibrationPosterior,
    CalibrationProblem,
    Experiment,
    FisherIdentifiability,
    LeastSquaresCalibration,
    NoiseModel,
    fisher_identifiability,
)
from .observation import ForwardResponse, GaussianObservationModel, ParameterVector
from .prediction import PosteriorPrediction, PredictedQuantity, predict
from .regularization import (
    NonlinearTikhonovResult,
    PicardCoefficients,
    RegularizationParameterChoice,
    TikhonovSolution,
    TruncatedSVDSolution,
    discrepancy_principle,
    generalized_cross_validation,
    l_curve,
    nonlinear_tikhonov,
    picard_coefficients,
    tikhonov,
    truncated_svd,
)
from .test_problems import (
    DiscreteIllPosedProblem,
    backward_heat_problem,
    second_derivative_problem,
)

__all__ = [
    "DISCREPANCY_KERNELS",
    "DiscrepancyCalibration",
    "DiscrepancyFit",
    "DiscrepancyIdentifiability",
    "DiscrepancyPosterior",
    "DiscrepancyPrediction",
    "DiscrepancyProcess",
    "KnownNoise",
    "LogLinearNoise",
    "discrepancy_kernel",
    "EnsembleInversionResult",
    "ensemble_kalman_inversion",
    "ensemble_smoother_mda",
    "CALIBRATION_GRADIENTS",
    "ERROR_CORRELATION_KERNELS",
    "CalibrationPosterior",
    "CalibrationProblem",
    "ErrorCorrelation",
    "Experiment",
    "FisherIdentifiability",
    "ForwardResponse",
    "GaussianObservationModel",
    "IdentifiabilityDiagnostic",
    "LeastSquaresCalibration",
    "LinearCalibrationResult",
    "NoiseModel",
    "ParameterVector",
    "PosteriorPrediction",
    "PredictedQuantity",
    "PredictionTarget",
    "calibrate_linear_gaussian",
    "diagnose_linear_identifiability",
    "error_correlation_kernel",
    "fisher_identifiability",
    "predict",
    "DiscreteIllPosedProblem",
    "NonlinearTikhonovResult",
    "PicardCoefficients",
    "RegularizationParameterChoice",
    "TikhonovSolution",
    "TruncatedSVDSolution",
    "backward_heat_problem",
    "discrepancy_principle",
    "generalized_cross_validation",
    "l_curve",
    "nonlinear_tikhonov",
    "picard_coefficients",
    "second_derivative_problem",
    "tikhonov",
    "truncated_svd",
]
