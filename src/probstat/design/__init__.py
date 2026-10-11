"""Optimal and Bayesian experimental design.

``classical`` covers approximate and exact optimal designs for linear and
locally linearized regression models: information matrices, D/A/E/I
criteria, the multiplicative and Fedorov-Wynn algorithms, the Kiefer-Wolfowitz
equivalence check, efficient rounding, and Fedorov exchange. ``bayesian``
scores experimental conditions of a calibration-style problem by expected
information gain (closed form, Laplace, nested Monte Carlo) and selects them
greedily.
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

__all__ = [
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
]
