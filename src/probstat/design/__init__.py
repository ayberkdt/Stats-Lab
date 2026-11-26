"""Optimal and Bayesian experimental design.

``classical`` covers approximate and exact optimal designs for linear and
locally linearized regression models: information matrices, D/A/E/I
criteria, the multiplicative and Fedorov-Wynn algorithms, the Kiefer-Wolfowitz
equivalence check, efficient rounding, and Fedorov exchange. ``bayesian``
scores experimental conditions of a calibration-style problem by expected
information gain (closed form, Laplace, nested Monte Carlo) and selects them
greedily. ``bayesian_optimization`` optimizes an expensive function over a
box with a Gaussian-process surrogate and an acquisition function (expected
improvement and its stable logarithm, confidence bounds, constrained expected
improvement, knowledge gradient, Thompson sampling).

Main entry points:
    ``d_optimal_design``: approximate D-optimal weights on a finite candidate
        set (multiplicative or Fedorov-Wynn).
    ``equivalence_check``: Kiefer-Wolfowitz certificate and D-efficiency
        bound of a design.
    ``efficient_rounding``: approximate weights to an exact ``n``-run design.
    ``exchange_exact_design``: Fedorov exchange improvement of an exact
        design.
    ``BayesianDesignProblem``: prior, forward model, and known Gaussian noise
        of a calibration-style design problem.
    ``expected_information_gain``: EIG of a set of conditions by the
        linearized, Laplace, or nested Monte Carlo estimator.
    ``greedy_design``: sequential selection of conditions by EIG.
    ``bayesian_optimization``: sequential minimization or maximization over
        a box (initial Latin hypercube, GP refit, acquisition maximized from
        a candidate set with local refinement).
    ``expected_improvement``, ``log_expected_improvement``,
        ``confidence_bound``, ``constrained_expected_improvement``,
        ``knowledge_gradient``, ``thompson_sample``: the acquisition
        functions on their own.
    ``linear_design_criteria``, ``greedy_linear_design`` and
        ``exhaustive_linear_design`` (experimental, 0.28): A-, D-, E- and
        goal-oriented sensor selection for linear Gaussian inverse problems.

Assumptions:
    Classical designs assume a regression model linear in its parameters (or
    linearized at a nominal value), homoscedastic uncorrelated errors, and a
    finite candidate set. Bayesian design needs known Gaussian noise scales;
    the Laplace estimator assumes a nearly Gaussian posterior. Bayesian
    optimization assumes a deterministic (or mildly noisy) objective that a
    stationary GP can represent, and a box-shaped search domain.

Common pitfalls:
    - ``information_matrix`` and the criteria are per run: weights sum to
      one, so the information of ``n`` runs is ``n M``. ``polynomial_basis(2)``
      has three terms ``(1, x, x^2)``.
    - A singular information matrix is not an error: ``design_criterion``
      returns ``-inf`` for D and ``inf`` for A and I (two support points for
      a quadratic, for example).
    - ``d_optimal_design`` returns when ``max_iterations`` (10000) is reached
      instead of raising; check ``converged``. Fedorov-Wynn with the default
      ``tolerance=1e-6`` on a 21-point quadratic grid stops unconverged.
    - Nested Monte Carlo defaults differ: ``nested_monte_carlo_information_gain``
      uses fresh inner samples (``reuse_samples=False``, biased upward),
      while ``expected_information_gain(method="nested")`` reuses the outer
      draws (``reuse_samples=True``, biased downward and never above
      ``log outer``). ``expected_information_gain`` itself defaults to
      ``method="laplace"``.
    - Bayesian optimization minimizes by default (``maximize=False``); the
      acquisition functions take the same flag. Its result is the best
      evaluated point, not a certified optimum, and ``seed=None`` is not
      reproducible. ``knowledge_gradient`` alone defaults to
      ``maximize=True``, the convention of ranking-and-selection.
    - A local design (``LocalLinearization``) is optimal only at its nominal
      parameter value. The EIG reported along a ``greedy_design`` path comes
      from the same common stream that chose the conditions; re-estimate the
      final design on an independent stream, as ``workflow.UQStudy.design``
      does.

Guide: docs/foundations/experimental-design.md

References:
    [REF-ATKINSON-DONEV-TOBIAS-2007], [REF-KIEFER-WOLFOWITZ-1960],
    [REF-PUKELSHEIM-RIEDER-1992], [REF-FEDOROV-1972],
    [REF-CHALONER-VERDINELLI-1995], [REF-RYAN-2003], [REF-HUAN-MARZOUK-2013],
    [REF-JONES-SCHONLAU-WELCH-1998], [REF-AMENT-ETAL-2023],
    [REF-SRINIVAS-ETAL-2010], [REF-GARDNER-ETAL-2014],
    [REF-FRAZIER-POWELL-DAYANIK-2008], [REF-ALEXANDERIAN-2021] (linear
    optimal design).

"""

from .bayesian import (
    BayesianDesignProblem,
    Condition,
    GreedyDesignResult,
    InformationGainEstimate,
    expected_information_gain,
    greedy_design,
    laplace_information_gain,
    linear_gaussian_information_gain,
    linearized_information_gain,
    nested_monte_carlo_information_gain,
)
from .bayesian_optimization import (
    ACQUISITIONS,
    INITIAL_DESIGNS,
    BayesianOptimizationResult,
    ThompsonDraw,
    bayesian_optimization,
    confidence_bound,
    constrained_expected_improvement,
    expected_improvement,
    gaussian_process_joint_posterior,
    gp_ucb_beta,
    improvement_function,
    knowledge_gradient,
    log_expected_improvement,
    log_improvement_function,
    probability_of_feasibility,
    thompson_sample,
)
from .classical import (
    ApproximateDesign,
    Basis,
    EquivalenceCheck,
    ExactDesign,
    ExchangeResult,
    LocalLinearization,
    OptimalDesignResult,
    d_efficiency,
    d_optimal_design,
    design_criterion,
    efficient_rounding,
    equivalence_check,
    exchange_exact_design,
    information_matrix,
    polynomial_basis,
    standardized_variances,
)
from .linear_optimal import (
    LinearDesignCriteria,
    LinearExhaustiveResult,
    LinearGreedyResult,
    exhaustive_linear_design,
    greedy_linear_design,
    linear_design_criteria,
)

__all__ = [
    "LinearDesignCriteria",
    "LinearExhaustiveResult",
    "LinearGreedyResult",
    "exhaustive_linear_design",
    "greedy_linear_design",
    "linear_design_criteria",
    "ApproximateDesign",
    "Basis",
    "BayesianDesignProblem",
    "Condition",
    "EquivalenceCheck",
    "ExactDesign",
    "ExchangeResult",
    "GreedyDesignResult",
    "InformationGainEstimate",
    "LocalLinearization",
    "OptimalDesignResult",
    "d_efficiency",
    "d_optimal_design",
    "design_criterion",
    "efficient_rounding",
    "equivalence_check",
    "exchange_exact_design",
    "expected_information_gain",
    "greedy_design",
    "information_matrix",
    "laplace_information_gain",
    "linear_gaussian_information_gain",
    "linearized_information_gain",
    "nested_monte_carlo_information_gain",
    "polynomial_basis",
    "standardized_variances",
    "ACQUISITIONS",
    "INITIAL_DESIGNS",
    "BayesianOptimizationResult",
    "ThompsonDraw",
    "bayesian_optimization",
    "confidence_bound",
    "constrained_expected_improvement",
    "expected_improvement",
    "gaussian_process_joint_posterior",
    "gp_ucb_beta",
    "improvement_function",
    "knowledge_gradient",
    "log_expected_improvement",
    "log_improvement_function",
    "probability_of_feasibility",
    "thompson_sample",
]
