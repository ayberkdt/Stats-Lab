"""Data assimilation: ensemble Kalman filtering and variational analysis.

``enkf`` provides the stochastic (perturbed-observation) ensemble Kalman
filter and its forecast/analysis building blocks; ``variational`` provides a
3D-Var cost minimization. Model types come from :mod:`probstat.filtering` but
are consumed structurally, so this package does not import it.
"""

from .enkf import (
    AdditiveGaussianModel,
    Ensemble,
    EnsembleAnalysis,
    EnsembleKalmanResult,
    ensemble_analysis,
    ensemble_forecast,
    ensemble_kalman_filter,
    inflate,
    initial_ensemble,
)
from .variational import VariationalAnalysis, three_dimensional_variational

__all__ = [
    "AdditiveGaussianModel",
    "Ensemble",
    "EnsembleAnalysis",
    "EnsembleKalmanResult",
    "VariationalAnalysis",
    "ensemble_analysis",
    "ensemble_forecast",
    "ensemble_kalman_filter",
    "inflate",
    "initial_ensemble",
    "three_dimensional_variational",
]
