"""Data assimilation: ensemble Kalman filtering, smoothing, 3D-Var, and 4D-Var.

``enkf`` provides the ensemble type, the forecast step, multiplicative
inflation, and the stochastic (perturbed-observation) analysis;
``square_root`` the deterministic ETKF and serial EAKF analyses;
``localization`` the Gaspari-Cohn correlation function, Schur-product
localization of the stochastic gain, and the LETKF; ``relaxation`` the RTPP
and RTPS relaxations; ``ensemble_filter`` the filter cycle shared by every
analysis; ``smoother`` the fixed-lag ensemble Kalman smoother; and
``variational`` a 3D-Var cost minimization; and ``fourdvar`` strong-,
incremental, and weak-constraint 4D-Var over a window, with adjoint
gradients from reverse-mode automatic differentiation. Model types come from
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
    ``four_dimensional_variational``: strong-constraint 4D-Var of ``x_0``
    by BFGS with the reverse-mode (adjoint) gradient.
    ``incremental_four_dimensional_variational``: Gauss-Newton outer loops
    around exactly solved quadratic inner problems.
    ``weak_constraint_four_dimensional_variational``: 4D-Var over the whole
    trajectory with a model-error covariance ``Q``.
    ``four_dimensional_variational_cost``: the 4D-Var cost function with
    its terms and gradient.

Assumptions:
    Noise is additive and Gaussian with known covariances. The ensemble
    covariance stands in for the true forecast covariance; it has rank at
    most ``N - 1`` and sampling error of order ``N^-1/2``. Localization
    assumes that true correlations decay with the distance between the
    given coordinates. 3D-Var is a single-time analysis with a fixed
    background covariance. 4D-Var needs a differentiable model and, for
    the strong constraint, a perfect one; its Gauss-Newton covariance is
    exact only for linear models.

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
    - 4D-Var observations start after the first transition: ``y_k``
      observes ``M`` applied ``k`` times to ``x_0``, as in the filters. Pass
      a model's raw ``transition_function``, not its validating
      ``transition`` method, so that reverse mode can record it.
    - Long windows of a chaotic model make the 4D-Var cost multimodal; the
      result is a local minimum near the starting point.

Guide: docs/foundations/filtering-and-data-assimilation.md

References:
    [REF-EVENSEN-1994], [REF-BURGERS-VANLEEUWEN-EVENSEN-1998],
    [REF-EVENSEN-2003], [REF-LORENC-1986],
    [REF-BISHOP-ETHERTON-MAJUMDAR-2001], [REF-HUNT-KOSTELICH-SZUNYOGH-2007],
    [REF-ANDERSON-2001], [REF-GASPARI-COHN-1999],
    [REF-HOUTEKAMER-MITCHELL-2001], [REF-ZHANG-SNYDER-SUN-2004],
    [REF-WHITAKER-HAMILL-2012], [REF-EVENSEN-VANLEEUWEN-2000],
    [REF-LE-DIMET-TALAGRAND-1986], [REF-COURTIER-THEPAUT-HOLLINGSWORTH-1994],
    [REF-TREMOLET-2006].

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
from .fourdvar import (
    VARIATIONAL_FORMULATIONS,
    VARIATIONAL_GRADIENTS,
    FourDimensionalVariationalCost,
    FourDimensionalVariationalResult,
    four_dimensional_variational,
    four_dimensional_variational_cost,
    incremental_four_dimensional_variational,
    weak_constraint_four_dimensional_variational,
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
    "VARIATIONAL_FORMULATIONS",
    "VARIATIONAL_GRADIENTS",
    "AdditiveGaussianModel",
    "Ensemble",
    "EnsembleAnalysis",
    "EnsembleKalmanResult",
    "EnsembleSmootherResult",
    "EnsembleTransformAnalysis",
    "FourDimensionalVariationalCost",
    "FourDimensionalVariationalResult",
    "Localization",
    "Relaxation",
    "VariationalAnalysis",
    "eakf_analysis",
    "ensemble_analysis",
    "ensemble_forecast",
    "ensemble_kalman_filter",
    "ensemble_kalman_smoother",
    "etkf_analysis",
    "four_dimensional_variational",
    "four_dimensional_variational_cost",
    "gaspari_cohn",
    "incremental_four_dimensional_variational",
    "inflate",
    "initial_ensemble",
    "letkf_analysis",
    "localization_matrix",
    "localized_ensemble_analysis",
    "relax_to_prior_perturbations",
    "relax_to_prior_spread",
    "three_dimensional_variational",
    "weak_constraint_four_dimensional_variational",
]
