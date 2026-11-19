"""Data assimilation: ensemble Kalman filtering, smoothing, and 3D-Var.

``enkf`` provides the ensemble type, the forecast step, multiplicative
inflation, and the stochastic (perturbed-observation) analysis;
``square_root`` the deterministic ETKF and serial EAKF analyses;
``localization`` the Gaspari-Cohn correlation function, Schur-product
localization of the stochastic gain, and the LETKF; ``relaxation`` the RTPP
and RTPS relaxations; ``ensemble_filter`` the filter cycle shared by every
analysis; ``smoother`` the fixed-lag ensemble Kalman smoother; and
``variational`` a 3D-Var cost minimization. Model types come from
:mod:`probstat.filtering` but are consumed structurally, so this package
does not import it.

Main entry points:
    ``ensemble_kalman_filter``: ensemble Kalman filter over a sequence of
    observations, with ``analysis="stochastic"`` (default), ``"etkf"``,
    ``"eakf"``, or ``"letkf"``, optional ``localization`` and
    ``relaxation``.
    ``ensemble_kalman_smoother``: fixed-lag EnKS reusing the analysis
    transforms of the filter.
    ``ensemble_forecast``: one forecast step of every ensemble member.
    ``ensemble_analysis``: one perturbed-observation analysis step.
    ``etkf_analysis``, ``eakf_analysis``, ``letkf_analysis``,
    ``localized_ensemble_analysis``: one deterministic, local, or localized
    analysis step.
    ``gaspari_cohn``, ``localization_matrix``, ``Localization``: compactly
    supported localization weights from coordinates.
    ``Relaxation``, ``relax_to_prior_perturbations``,
    ``relax_to_prior_spread``: RTPP and RTPS after the analysis.
    ``inflate``: multiplicative covariance inflation about the ensemble mean.
    ``three_dimensional_variational``: 3D-Var analysis by BFGS.

Assumptions:
    Noise is additive and Gaussian with known covariances. The ensemble
    covariance stands in for the true forecast covariance; it has rank at
    most ``N - 1`` and sampling error of order ``N^-1/2``. Localization
    assumes that true correlations decay with the distance between the
    given coordinates. 3D-Var is a single-time analysis with a fixed
    background covariance.

Common pitfalls:
    - ``ensemble_kalman_filter`` defaults to the stochastic analysis with
      ``inflation=1.0``, no localization, and no relaxation; small
      ensembles then under-disperse.
    - ``inflation`` is a covariance factor: anomalies are scaled by
      ``sqrt(inflation)``, and a factor below one raises ``ValueError``.
    - ``half_width`` of ``Localization`` is the Gaspari-Cohn ``c``: weights
      vanish beyond ``2c``, not beyond ``c``.
    - ``"etkf"`` and ``"eakf"`` are global and refuse a ``localization``;
      ``"letkf"`` requires one (``half_width=math.inf`` for none).
    - ``"eakf"`` needs a diagonal observation covariance.
    - ``ensemble_kalman_filter`` and ``initial_ensemble`` default to
      ``seed=None``; pass an integer seed to reproduce an ensemble.
    - ``three_dimensional_variational`` does not propagate the background
      covariance; repeated analyses reuse the ``B`` they are given.

Guide: docs/foundations/filtering-and-data-assimilation.md

References:
    [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998],
    [REF-EVENSEN-2003], [REF-LORENC-1986],
    [REF-BISHOP-ETHERTON-MAJUMDAR-2001], [REF-HUNT-KOSTELICH-SZUNYOGH-2007],
    [REF-ANDERSON-2001], [REF-GASPARI-COHN-1999],
    [REF-HOUTEKAMER-MITCHELL-2001], [REF-ZHANG-SNYDER-SUN-2004],
    [REF-WHITAKER-HAMILL-2012], [REF-EVENSEN-VANLEEUWEN-2000].

"""

from .enkf import (
    AdditiveGaussianModel,
    Ensemble,
    EnsembleAnalysis,
    ensemble_analysis,
    ensemble_forecast,
    inflate,
    initial_ensemble,
)
from .ensemble_filter import (
    ANALYSIS_METHODS,
    EnsembleKalmanResult,
    ensemble_kalman_filter,
)
from .localization import (
    Localization,
    gaspari_cohn,
    letkf_analysis,
    localization_matrix,
    localized_ensemble_analysis,
)
from .relaxation import (
    RELAXATION_METHODS,
    Relaxation,
    relax_to_prior_perturbations,
    relax_to_prior_spread,
)
from .smoother import EnsembleSmootherResult, ensemble_kalman_smoother
from .square_root import EnsembleTransformAnalysis, eakf_analysis, etkf_analysis
from .variational import VariationalAnalysis, three_dimensional_variational

__all__ = [
    "ANALYSIS_METHODS",
    "RELAXATION_METHODS",
    "AdditiveGaussianModel",
    "Ensemble",
    "EnsembleAnalysis",
    "EnsembleKalmanResult",
    "EnsembleSmootherResult",
    "EnsembleTransformAnalysis",
    "Localization",
    "Relaxation",
    "VariationalAnalysis",
    "eakf_analysis",
    "ensemble_analysis",
    "ensemble_forecast",
    "ensemble_kalman_filter",
    "ensemble_kalman_smoother",
    "etkf_analysis",
    "gaspari_cohn",
    "inflate",
    "initial_ensemble",
    "letkf_analysis",
    "localization_matrix",
    "localized_ensemble_analysis",
    "relax_to_prior_perturbations",
    "relax_to_prior_spread",
    "three_dimensional_variational",
]
