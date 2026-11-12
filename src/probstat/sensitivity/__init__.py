"""Local and global sensitivity: Sobol, Morris, DGSM, given-data, Shapley.

``stochastic`` decomposes the output variance of a stochastic simulator into
Sobol indices of its mean response and an intrinsic-noise index.

Main entry points:
    ``sobol_analysis``: first-order and total Sobol indices of one QoI of a
        probabilistic model, with optional intervals.
    ``sobol_analysis_all``: the same indices for several QoIs at the cost of
        one design.
    ``morris_screening``: Morris elementary effects (``mu*``, ``sigma``) for
        cheap screening.
    ``shapley_effects``: Shapley variance shares that remain valid for
        dependent inputs.
    ``dgsm``: derivative-based global sensitivity measures.
    ``estimate_sobol_indices``: pick-freeze estimator on caller-supplied base
        matrices, including second-order indices.
    ``central_difference_gradient``: local derivatives at one point.

Assumptions:
    Sobol indices, Morris screening, and the functional ANOVA behind them
    need mutually independent inputs and a square-integrable scalar QoI with
    non-zero variance. Shapley effects accept independent or Gaussian-copula
    inputs. Local derivatives describe one point and one step size, not the
    input distribution.

Common pitfalls:
    - ``sobol_analysis`` and ``morris_screening`` raise ``ValueError`` when
      the model's inputs are dependent (a non-identity copula); use
      ``shapley_effects`` instead of decorrelating the inputs by hand.
    - ``sobol_analysis`` defaults to ``bootstrap=0``, so it reports no
      interval (``interval_method == "none"``, bounds ``nan``), while
      ``workflow.UQStudy.sensitivity`` defaults to ``bootstrap=200``, which on
      a Sobol design gives replicate intervals over 8 randomizations. Rows of
      quasi-random or LHS designs are never bootstrapped.
    - Estimates are not clipped to ``[0, 1]``: small negative or above-one
      values are sampling noise, large ones point to too few rows or a
      broken design. The cost is ``size (d + 2)`` model runs.
    - Rank by total effect: an input that acts only through interactions has
      a first-order index near zero but a clearly positive total index.
    - Morris effects are measured on a grid whose step is half of each
      input's quantile range, so a response periodic with that period (for
      example ``sin(x)**2`` with ``x`` uniform on ``[-pi, pi]``) shows
      ``mu* = 0`` for that input; cross-check with ``sobol_analysis``.

Guide: docs/foundations/sensitivity-analysis.md

References:
    [REF-SOBOL-1993], [REF-SALTELLI-2010], [REF-JANSEN-1999],
    [REF-MORRIS-1991], [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007],
    [REF-SOBOL-KUCHERENKO-2009], [REF-OWEN-2014],
    [REF-SONG-NELSON-STAUM-2016].

"""

from .analysis import (
    LocalSensitivityResult,
    SensitivityDecision,
    SobolResult,
    central_difference_gradient,
    estimate_sobol_indices,
)
from .derivative_based import DGSMResult, dgsm, poincare_constant
from .given_data import GivenDataSensitivityResult, given_data_first_order
from .model_based import (
    IndexInterval,
    ModelSobolResult,
    MorrisResult,
    MultiQuantitySobolResult,
    morris_screening,
    sobol_analysis,
    sobol_analysis_all,
)
from .shapley import (
    ShapleyResult,
    exact_shapley_values,
    linear_gaussian_shapley,
    shapley_effects,
)
from .stochastic import StochasticSobolResult, stochastic_sobol_indices

__all__ = [
    "StochasticSobolResult",
    "stochastic_sobol_indices",
    "IndexInterval",
    "ModelSobolResult",
    "MorrisResult",
    "morris_screening",
    "sobol_analysis",
    "LocalSensitivityResult",
    "SensitivityDecision",
    "ShapleyResult",
    "SobolResult",
    "central_difference_gradient",
    "estimate_sobol_indices",
    "exact_shapley_values",
    "linear_gaussian_shapley",
    "shapley_effects",
    "DGSMResult",
    "GivenDataSensitivityResult",
    "MultiQuantitySobolResult",
    "dgsm",
    "given_data_first_order",
    "poincare_constant",
    "sobol_analysis_all",
]
