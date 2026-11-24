"""Verification and uncertainty-quantification foundations.

The package holds the UQ vocabulary (QoIs, uncertainty sources, forward-model
contract, simulator-run failure policy, evaluation cache, budget errors),
solution verification (error budgets, observed order, Richardson extrapolation,
Grid Convergence Index, least-squares order, and the
:class:`SolutionVerification` numerical-uncertainty statement), simulations
recorded as computational experiments, and the auditable
:class:`UQWorkflowRecord` of a complete study.

Main entry points:
    ``UncertaintyBudget``: table of ``BudgetEntry`` contributions with the law
        of propagation (correlation included), Welch-Satterthwaite degrees of
        freedom, expanded uncertainty, and named combination rules
        (``root_sum_square``, ``worst_case_sum``, ``mixed_rule``) whose
        ``CombinedUncertainty`` records say standard uncertainty or bound.
    ``budget_from_model``: sensitivity coefficients by numerical
        differentiation, with a second-order linearisation check.
    ``monte_carlo_budget``: JCGM 101 propagation of distributions with
        symmetric and shortest coverage intervals;
        ``compare_with_law_of_propagation`` validates the GUM framework
        against it.
    ``QuantityOfInterest``: named, unit-carrying output that a decision
        depends on.
    ``verify_solution``: grid-refinement study to a ``SolutionVerification``
        statement ``phi_1 +/- U``.
    ``grid_convergence_index``: three-grid GCI of Celik et al. with
        non-uniform refinement ratios.
    ``least_squares_convergence``: observed order from four or more grids.
    ``observed_order_of_accuracy``: order from three uniformly refined
        solutions.
    ``refinement_study``: run a computational experiment over one varying
        numerical setting.
    ``EvaluationCache``: reuse of identical forward-model runs.
    ``UQWorkflowRecord``: auditable step-by-step record of a study.

Assumptions:
    Solution verification assumes a single leading error term, grids in the
    asymptotic range, and negligible iterative and round-off error relative to
    discretization error. ``NumericalErrorBudget`` entries are absolute bounds
    combined by the triangle inequality, not standard deviations. A
    verification statement applies to the QoI and settings of its own
    refinement study.

Common pitfalls:
    - ``observed_order_of_accuracy(coarse, medium, fine)`` takes the coarsest
      value first; reversing the order silently returns the negative order
      (``-2.0`` instead of ``2.0`` for ``1 + h^2`` on ``h = 1, 1/2, 1/4``).
      It uses absolute differences and does not classify the sequence, so an
      oscillatory sequence still returns a number. ``grid_convergence_index``
      and ``verify_solution`` sort the grids by spacing and classify
      convergence.
    - ``verify_solution`` defaults to ``theoretical_order=2.0``; for a
      first-order scheme pass the true order, because the asymptotic-range
      check compares against it. Its cautions are stored in the statement's
      ``warnings`` tuple, not emitted as Python warnings.
    - The three-grid GCI uses ``DEFAULT_SAFETY_FACTOR = 1.25``; an
      oscillatory three-grid sequence gets the bound
      ``UNRELIABLE_SAFETY_FACTOR * (phi_max - phi_min) / 2`` (factor 3.0) and
      no observed order.
    - ``U`` is an expanded uncertainty. ``standard_uncertainty()`` converts
      it as a uniform error (``U / sqrt(3)``) by default; ``"normal95"``
      gives ``U / 1.96``. State which one enters an uncertainty budget.

Guide: docs/foundations/vv-uq-foundations.md

References:
    [REF-ROACHE-1998, Chapters 4-5], [REF-CELIK-ETAL-2008], [REF-ROY-2005],
    [REF-ECA-HOEKSTRA-2014], [REF-OBERKAMPF-ROY-2010],
    [REF-SMITH-UQ-2024, Chapter 1].

"""

from .budget import (
    BudgetEntry,
    CombinedUncertainty,
    Contribution,
    ExpandedUncertainty,
    GumValidation,
    LinearizationCheck,
    ModelBudget,
    MonteCarloResult,
    QuantityKind,
    UncertaintyBudget,
    budget_from_model,
    compare_with_law_of_propagation,
    monte_carlo_budget,
    numerical_tolerance,
    sample_coverage_intervals,
)
from .evaluation import (
    FAILURE_POLICIES,
    EvaluationBudgetExceeded,
    EvaluationCache,
    FailedEvaluation,
    FailureLog,
    FailurePolicy,
    FrozenMapping,
    ModelEvaluationError,
    json_safe,
)
from .experiment import (
    ComputationalExperiment,
    RefinementStudy,
    refinement_study,
    run_computational_experiment,
)
from .taxonomy import (
    ForwardModel,
    QuantityOfInterest,
    UncertaintyNature,
    UncertaintySource,
    UncertaintySourceKind,
    evaluate_model,
    select_outputs,
)
from .verification import (
    DEFAULT_SAFETY_FACTOR,
    UNRELIABLE_SAFETY_FACTOR,
    ConvergenceType,
    GridConvergenceResult,
    LeastSquaresConvergence,
    NumericalErrorBudget,
    NumericalErrorDistribution,
    SolutionVerification,
    classify_convergence,
    grid_convergence_index,
    least_squares_convergence,
    observed_order_of_accuracy,
    representative_spacing,
    richardson_extrapolation,
    verify_solution,
)
from .workflow_record import (
    WORKFLOW_GUIDE,
    ProblemDirection,
    StageLike,
    StepGuide,
    StepStatus,
    UQWorkflowRecord,
    WorkflowEntry,
    WorkflowStep,
)

__all__ = [
    "DEFAULT_SAFETY_FACTOR",
    "FAILURE_POLICIES",
    "UNRELIABLE_SAFETY_FACTOR",
    "WORKFLOW_GUIDE",
    "BudgetEntry",
    "CombinedUncertainty",
    "ComputationalExperiment",
    "Contribution",
    "ConvergenceType",
    "EvaluationBudgetExceeded",
    "EvaluationCache",
    "ExpandedUncertainty",
    "FailedEvaluation",
    "FailureLog",
    "FailurePolicy",
    "ForwardModel",
    "FrozenMapping",
    "GridConvergenceResult",
    "GumValidation",
    "LeastSquaresConvergence",
    "LinearizationCheck",
    "ModelBudget",
    "ModelEvaluationError",
    "MonteCarloResult",
    "NumericalErrorBudget",
    "NumericalErrorDistribution",
    "ProblemDirection",
    "QuantityKind",
    "QuantityOfInterest",
    "RefinementStudy",
    "SolutionVerification",
    "StageLike",
    "StepGuide",
    "StepStatus",
    "UQWorkflowRecord",
    "UncertaintyBudget",
    "UncertaintyNature",
    "UncertaintySource",
    "UncertaintySourceKind",
    "WorkflowEntry",
    "WorkflowStep",
    "budget_from_model",
    "classify_convergence",
    "compare_with_law_of_propagation",
    "evaluate_model",
    "grid_convergence_index",
    "json_safe",
    "least_squares_convergence",
    "monte_carlo_budget",
    "numerical_tolerance",
    "observed_order_of_accuracy",
    "refinement_study",
    "representative_spacing",
    "richardson_extrapolation",
    "run_computational_experiment",
    "sample_coverage_intervals",
    "select_outputs",
    "verify_solution",
]
