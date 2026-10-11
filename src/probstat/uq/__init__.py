"""Verification and uncertainty-quantification foundations.

The package holds the UQ vocabulary (QoIs, uncertainty sources, forward-model
contract), solution verification (error budgets, observed order, Richardson
extrapolation, Grid Convergence Index, least-squares order, and the
:class:`SolutionVerification` numerical-uncertainty statement), simulations
recorded as computational experiments, and the auditable
:class:`UQWorkflowRecord` of a complete study.
"""

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
    "UNRELIABLE_SAFETY_FACTOR",
    "WORKFLOW_GUIDE",
    "ComputationalExperiment",
    "ConvergenceType",
    "ForwardModel",
    "GridConvergenceResult",
    "LeastSquaresConvergence",
    "NumericalErrorBudget",
    "NumericalErrorDistribution",
    "ProblemDirection",
    "QuantityOfInterest",
    "RefinementStudy",
    "SolutionVerification",
    "StageLike",
    "StepGuide",
    "StepStatus",
    "UQWorkflowRecord",
    "UncertaintyNature",
    "UncertaintySource",
    "UncertaintySourceKind",
    "WorkflowEntry",
    "WorkflowStep",
    "classify_convergence",
    "evaluate_model",
    "grid_convergence_index",
    "least_squares_convergence",
    "observed_order_of_accuracy",
    "refinement_study",
    "representative_spacing",
    "richardson_extrapolation",
    "run_computational_experiment",
    "verify_solution",
]
