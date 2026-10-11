"""Reliability, failure-probability, FORM/SORM, rare-event, and system analysis.

``analysis`` holds the original vector-level estimators. ``model_based``,
``simulation``, and ``systems`` work on a
:class:`~probstat.modeling.ProbabilisticModel` through
:class:`ModelLimitState`, sharing one joint input law across FORM, SORM,
importance sampling, subset simulation, system reliability, and inverse design.
"""

from .analysis import (
    DirectionalSamplingResult,
    DirectReliabilityResult,
    LimitState,
    LinearNormalFORMResult,
    WeightedFailureResult,
    estimate_directional_failure_probability,
    estimate_failure_probability,
    estimate_weighted_failure_probability,
    linear_normal_form,
)
from .model_based import (
    DesignParameterResult,
    FORMResult,
    InverseFORMResult,
    ModelLimitState,
    SORMResult,
    correlated_importance,
    design_for_target_beta,
    form_hlrf,
    generalized_reliability_index,
    inverse_form,
    principal_curvatures,
    second_order_probabilities,
    sorm,
)
from .simulation import (
    DesignPointImportanceResult,
    MethodComparison,
    MethodComparisonRow,
    ModelMonteCarloResult,
    SubsetSimulationResult,
    compare_reliability_methods,
    design_point_importance_sampling,
    model_monte_carlo,
    subset_simulation,
)
from .systems import (
    SeriesSystemBounds,
    SystemReliabilityResult,
    bivariate_normal_cdf,
    gauss_legendre,
    series_system_bounds,
    system_failure_probability,
)

__all__ = [
    "DesignParameterResult",
    "DesignPointImportanceResult",
    "DirectReliabilityResult",
    "DirectionalSamplingResult",
    "FORMResult",
    "InverseFORMResult",
    "LimitState",
    "LinearNormalFORMResult",
    "MethodComparison",
    "MethodComparisonRow",
    "ModelLimitState",
    "ModelMonteCarloResult",
    "SORMResult",
    "SeriesSystemBounds",
    "SubsetSimulationResult",
    "SystemReliabilityResult",
    "WeightedFailureResult",
    "bivariate_normal_cdf",
    "compare_reliability_methods",
    "correlated_importance",
    "design_for_target_beta",
    "design_point_importance_sampling",
    "estimate_directional_failure_probability",
    "estimate_failure_probability",
    "estimate_weighted_failure_probability",
    "form_hlrf",
    "gauss_legendre",
    "generalized_reliability_index",
    "inverse_form",
    "linear_normal_form",
    "model_monte_carlo",
    "principal_curvatures",
    "second_order_probabilities",
    "series_system_bounds",
    "sorm",
    "subset_simulation",
    "system_failure_probability",
]
