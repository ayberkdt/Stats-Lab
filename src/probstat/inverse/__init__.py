"""Observation models and auditable inverse-problem calibration workflows."""

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
