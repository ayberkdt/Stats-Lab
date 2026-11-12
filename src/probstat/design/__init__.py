"""Optimal and Bayesian experimental design.

``classical`` covers approximate and exact optimal designs for linear and
locally linearized regression models: information matrices, D/A/E/I
criteria, the multiplicative and Fedorov-Wynn algorithms, the Kiefer-Wolfowitz
equivalence check, efficient rounding, and Fedorov exchange. ``bayesian``
scores experimental conditions of a calibration-style problem by expected
information gain (closed form, Laplace, nested Monte Carlo) and selects them
greedily.

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

Assumptions:
    Classical designs assume a regression model linear in its parameters (or
    linearized at a nominal value), homoscedastic uncorrelated errors, and a
    finite candidate set. Bayesian design needs known Gaussian noise scales;
    the Laplace estimator assumes a nearly Gaussian posterior.

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
    - A local design (``LocalLinearization``) is optimal only at its nominal
      parameter value. The EIG reported along a ``greedy_design`` path comes
      from the same common stream that chose the conditions; re-estimate the
      final design on an independent stream, as ``workflow.UQStudy.design``
      does.

Guide: docs/foundations/experimental-design.md

References:
    [REF-ATKINSON-DONEV-TOBIAS-2007], [REF-KIEFER-WOLFOWITZ-1960],
    [REF-PUKELSHEIM-RIEDER-1992], [REF-FEDOROV-1972],
    [REF-CHALONER-VERDINELLI-1995], [REF-RYAN-2003], [REF-HUAN-MARZOUK-2013].

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
