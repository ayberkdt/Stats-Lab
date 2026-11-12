"""Observation models and auditable inverse-problem calibration workflows.

``calibration`` holds the linear-Gaussian intercept-slope calibration with its
identifiability diagnostic. ``nonlinear`` declares a calibration problem on a
:class:`~probstat.modeling.ProbabilisticModel` whose joint input law is the
prior, with least squares, Fisher identifiability, the Laplace approximation,
and MCMC. ``discrepancy`` adds a Gaussian-process model discrepancy
(Kennedy-O'Hagan) and heteroscedastic noise, and ``prediction`` pushes
posterior draws through the forward model.

Main entry points:
    ``CalibrationProblem``: nonlinear calibration of model parameters from
    experiments, one ``NoiseModel`` per observed quantity.
    ``Experiment``: observed values with the conditions they were taken at.
    ``NoiseModel``: known or uncertain Gaussian or Student-t measurement
    noise, with an optional discrepancy variance.
    ``predict``: posterior predictions under new conditions, latent or with
    measurement noise.
    ``DiscrepancyCalibration``: calibration with a Gaussian-process
    discrepancy over declared inputs.
    ``fisher_identifiability``: local identifiability from the prior-scaled
    Fisher information.
    ``calibrate_linear_gaussian``: conjugate posterior of an intercept and a
    slope under known noise.

Assumptions:
    The forward model is deterministic given its inputs. Measurement errors
    are independent across observations (``DiscrepancyCalibration`` adds a
    correlated discrepancy, not correlated measurement errors). The prior is
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

Guide: docs/foundations/inverse-problems-and-calibration.md

References:
    [REF-TARANTOLA-2005], [REF-GELMAN-ETAL-2013], [REF-KENNEDY-OHAGAN-2001],
    [REF-BRYNJARSDOTTIR-OHAGAN-2014], [REF-HAARIO-SAKSMAN-TAMMINEN-2001].

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
from .nonlinear import (
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
    "CalibrationPosterior",
    "CalibrationProblem",
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
    "fisher_identifiability",
    "predict",
]
