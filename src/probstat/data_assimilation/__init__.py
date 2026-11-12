"""Data assimilation: ensemble Kalman filtering and variational analysis.

``enkf`` provides the stochastic (perturbed-observation) ensemble Kalman
filter and its forecast/analysis building blocks; ``variational`` provides a
3D-Var cost minimization. Model types come from :mod:`probstat.filtering` but
are consumed structurally, so this package does not import it.

Main entry points:
    ``ensemble_kalman_filter``: stochastic (perturbed-observation) ensemble
    Kalman filter over a sequence of observations.
    ``ensemble_forecast``: one forecast step of every ensemble member.
    ``ensemble_analysis``: one perturbed-observation analysis step.
    ``inflate``: multiplicative covariance inflation about the ensemble mean.
    ``three_dimensional_variational``: 3D-Var analysis by BFGS.

Assumptions:
    Noise is additive and Gaussian with known covariances. The ensemble
    covariance stands in for the true forecast covariance; it has rank at
    most ``N - 1`` and sampling error of order ``N^-1/2``. 3D-Var is a
    single-time analysis with a fixed background covariance.

Common pitfalls:
    - ``ensemble_kalman_filter`` defaults to ``inflation=1.0`` (no
      inflation) and never localizes; small ensembles then under-disperse.
    - ``inflation`` is a covariance factor: anomalies are scaled by
      ``sqrt(inflation)``, and a factor below one raises ``ValueError``.
    - ``ensemble_kalman_filter`` and ``initial_ensemble`` default to
      ``seed=None``; pass an integer seed to reproduce an ensemble.
    - ``three_dimensional_variational`` does not propagate the background
      covariance; repeated analyses reuse the ``B`` they are given.

Guide: docs/foundations/filtering-and-data-assimilation.md

References:
    [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998],
    [REF-EVENSEN-2003], [REF-LORENC-1986].

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
