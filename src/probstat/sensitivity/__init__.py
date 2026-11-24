"""Local and global sensitivity: Sobol, Morris, DGSM, given-data, Shapley, HSIC.

``stochastic`` decomposes the output variance of a stochastic simulator into
Sobol indices of its mean response and an intrinsic-noise index.
``rank_based``, ``moment_independent``, ``kernel_indices`` and
``active_subspace`` add rank statistics, distribution-based indices, kernel
(MMD) indices and gradient-based dimension reduction.

Main entry points:
    ``sobol_analysis``: first-order and total Sobol indices of one QoI of a
        probabilistic model, with optional intervals and a choice of
        estimator (``SOBOL_ESTIMATORS``).
    ``sobol_analysis_all``: the same indices for several QoIs at the cost of
        one design.
    ``morris_screening``: Morris elementary effects (``mu``, ``mu*``,
        ``sigma``) for cheap screening.
    ``taylor_importance_factors``: quadratic-sum (first-order Taylor)
        importance factors of a model QoI around one point;
        ``quadratic_sum_importance`` does the same for a plain function.
    ``shapley_effects``: Shapley variance shares that remain valid for
        dependent inputs.
    ``dgsm``: derivative-based global sensitivity measures.
    ``estimate_sobol_indices``: pick-freeze estimator on caller-supplied base
        matrices, including second-order indices.
    ``central_difference_gradient``: local derivatives at one point.
    ``regression_sensitivity``: SRC, SRC^2, PCC (and SRRC, PRCC with
        ``rank=True``), ``R^2``, Student and Fisher tests from an existing
        input-output sample; ``model_regression_sensitivity`` samples and
        runs a model first.
    ``hsic_indices``: HSIC dependence measures, normalized ``R2_HSIC``, and
        permutation or Gamma independence tests from an existing sample,
        optionally of a target transform ``1{Y > c}``;
        ``model_hsic_indices`` samples and runs a model first.
    ``rank_sobol_indices``: all first-order Sobol indices from one i.i.d.
        sample by the lag-one rank statistic, with a disjoint-block standard
        error; ``model_rank_sobol_indices`` samples a model first.
    ``chatterjee_indices``: Chatterjee's rank correlation ``xi`` of the
        output on each input with an asymptotic or permutation test of
        independence; ``chatterjee_xi`` returns one coefficient.
    ``borgonovo_delta`` and ``pawn_indices``: moment-independent indices
        (density-based and CDF-based) from an existing sample, with a
        permutation null level.
    ``kernel_sobol_indices``: kernel (MMD) first-order indices by
        pick-freeze; ``model_kernel_sobol_indices`` draws the design.
    ``tornado_analysis``: one-at-a-time swings of a model QoI between two
        marginal quantiles (or explicit ranges), sorted for a tornado chart;
        ``tornado_function`` takes a callable with ``low / base / high``
        per input. It shows neither interactions, dependence, nor
        non-monotone behaviour; ``TornadoBar.nonmonotone`` is a cheap check.
    ``fast_analysis``: extended FAST first-order and total-effect indices
        from search curves with phase resampling; ``fast_indices`` for a
        plain callable on independent marginals.
    ``rbd_fast_analysis``: random balance design first-order indices from one
        sample, with the Tissot-Prieur bias correction; ``rbd_fast_indices``
        for a plain callable.
    ``active_subspace``: eigen-decomposition of ``E[grad f grad f']`` in the
        standard-normal space of a model, with eigenvalue gaps, activity
        scores and bootstrap diagnostics;
        ``active_subspace_from_gradients`` takes the gradients directly.

Assumptions:
    Sobol indices, Morris screening, and the functional ANOVA behind them
    need mutually independent inputs and a square-integrable scalar QoI with
    non-zero variance. Shapley effects accept independent or Gaussian-copula
    inputs. Local derivatives describe one point and one step size, not the
    input distribution; quadratic-sum importance factors inherit that
    locality and are trustworthy only for a model close to linear over the
    range of the inputs.

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
    - ``SRC^2`` are variance shares only for a near-linear model with
      independent inputs: read ``R^2`` (and the result's ``notes``) first;
      with ``R^2 < 0.7`` the indices miss the nonlinear part of the output.
    - HSIC values depend on the kernel bandwidth and are not variance
      shares; compare inputs through ``r2_hsic`` and the p-values. The
      permutation test costs ``O(n^2)`` per permutation and input.
    - Do not bootstrap ``rank_sobol_indices`` or ``chatterjee_indices``:
      repeated rows become neighbours after sorting and inflate both
      statistics. Use the ``blocks`` standard error or the built-in tests.
    - ``borgonovo_delta`` and ``pawn_indices`` are positive for an input
      without effect; read them against ``null_level`` (set
      ``permutations``), not against zero.
    - Kernel indices depend on the kernel and its bandwidth and need not sum
      to at most one; only the linear kernel gives variance shares.
    - Active-subspace eigenvectors depend on the scaling of the
      coordinates; ``active_subspace`` works in standard-normal space, and
      gradients passed to ``active_subspace_from_gradients`` must already
      be in coordinates of comparable scale.
    - A tornado chart is one-at-a-time: it misses interactions, refuses
      dependent inputs, and shows swing zero for ``x^2`` with ``x``
      symmetric about the baseline; read ``nonmonotone_inputs()``.
    - FAST and random balance design need independent inputs, an odd sample
      size, and ``N >= 4 M^2 max(2 d - 3, 1) + 1`` (extended FAST,
      ``fast_sample_size``); judge an estimate against
      its phase-resampling ``*_error``, and note that the raw random balance
      estimate is biased upwards by ``(2 M / N)(1 - S_i)``.
    - Morris effects are measured on a grid whose step is half of each
      input's quantile range, so a response periodic with that period (for
      example ``sin(x)**2`` with ``x`` uniform on ``[-pi, pi]``) shows
      ``mu* = 0`` for that input; cross-check with ``sobol_analysis``.

Guide: docs/foundations/sensitivity-analysis.md

References:
    [REF-SOBOL-1993], [REF-SALTELLI-2010], [REF-JANSEN-1999],
    [REF-MORRIS-1991], [REF-CAMPOLONGO-CARIBONI-SALTELLI-2007],
    [REF-SOBOL-KUCHERENKO-2009], [REF-OWEN-2014],
    [REF-SONG-NELSON-STAUM-2016], [REF-HELTON-ETAL-2006],
    [REF-DA-VEIGA-2015], [REF-GRETTON-ETAL-2008], [REF-GAMBOA-ETAL-2022],
    [REF-CHATTERJEE-2021], [REF-CONSTANTINE-DOW-WANG-2014],
    [REF-BORGONOVO-2007], [REF-PIANOSI-WAGENER-2015], [REF-DA-VEIGA-2021],
    [REF-CUKIER-1973], [REF-SALTELLI-TARANTOLA-CHAN-1999],
    [REF-TARANTOLA-GATELLI-MARA-2006], [REF-TISSOT-PRIEUR-2012],
    [REF-HOWARD-1988].

"""

from ._sobol_estimators import SOBOL_ESTIMATORS
from .active_subspace import (
    ActiveSubspaceResult,
    active_subspace,
    active_subspace_from_gradients,
)
from .analysis import (
    LocalSensitivityResult,
    SensitivityDecision,
    SobolResult,
    central_difference_gradient,
    estimate_sobol_indices,
)
from .derivative_based import DGSMResult, dgsm, poincare_constant
from .fast import (
    FastResult,
    RBDFastResult,
    fast_analysis,
    fast_frequencies,
    fast_indices,
    fast_sample_size,
    rbd_fast_analysis,
    rbd_fast_indices,
)
from .given_data import GivenDataSensitivityResult, given_data_first_order
from .hsic import HSICResult, hsic_indices, model_hsic_indices
from .kernel_indices import (
    KernelSobolResult,
    kernel_sobol_indices,
    model_kernel_sobol_indices,
)
from .model_based import (
    IndexInterval,
    ModelSobolResult,
    MorrisResult,
    MultiQuantitySobolResult,
    morris_screening,
    sobol_analysis,
    sobol_analysis_all,
)
from .moment_independent import (
    BorgonovoResult,
    PawnResult,
    borgonovo_delta,
    pawn_indices,
)
from .rank_based import (
    ChatterjeeResult,
    RankSobolResult,
    chatterjee_indices,
    chatterjee_xi,
    model_rank_sobol_indices,
    rank_sobol_indices,
)
from .regression_based import (
    RegressionSensitivityResult,
    model_regression_sensitivity,
    regression_sensitivity,
)
from .shapley import (
    ShapleyResult,
    exact_shapley_values,
    linear_gaussian_shapley,
    shapley_effects,
)
from .stochastic import StochasticSobolResult, stochastic_sobol_indices
from .taylor import (
    TaylorImportanceResult,
    quadratic_sum_importance,
    taylor_importance_factors,
)
from .tornado import TornadoBar, TornadoResult, tornado_analysis, tornado_function

__all__ = [
    "SOBOL_ESTIMATORS",
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
    "TaylorImportanceResult",
    "quadratic_sum_importance",
    "taylor_importance_factors",
    "HSICResult",
    "RegressionSensitivityResult",
    "hsic_indices",
    "model_hsic_indices",
    "model_regression_sensitivity",
    "regression_sensitivity",
    "ActiveSubspaceResult",
    "BorgonovoResult",
    "ChatterjeeResult",
    "KernelSobolResult",
    "PawnResult",
    "RankSobolResult",
    "active_subspace",
    "active_subspace_from_gradients",
    "borgonovo_delta",
    "chatterjee_indices",
    "chatterjee_xi",
    "kernel_sobol_indices",
    "model_kernel_sobol_indices",
    "model_rank_sobol_indices",
    "pawn_indices",
    "rank_sobol_indices",
    "FastResult",
    "RBDFastResult",
    "TornadoBar",
    "TornadoResult",
    "fast_analysis",
    "fast_frequencies",
    "fast_indices",
    "fast_sample_size",
    "rbd_fast_analysis",
    "rbd_fast_indices",
    "tornado_analysis",
    "tornado_function",
]
