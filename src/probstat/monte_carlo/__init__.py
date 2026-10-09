"""Monte Carlo estimators, convergence diagnostics, and cost accounting."""

from .estimation import (
    MonteCarloEstimate,
    ProbabilityEstimate,
    convergence_trace,
    estimate_mean,
    estimate_probability,
    estimate_uniform_integral,
)
from .variance_reduction import (
    ControlVariateResult,
    CostedEstimate,
    EfficiencyComparison,
    ImportanceSamplingResult,
    StratifiedEstimate,
    allocate_stratified_budget,
    compare_efficiency,
    estimate_antithetic_uniform,
    estimate_conditional_expectation,
    estimate_control_variate,
    estimate_importance_sampling,
    estimate_stratified_uniform,
)

__all__ = [
    "MonteCarloEstimate",
    "ProbabilityEstimate",
    "ControlVariateResult",
    "CostedEstimate",
    "EfficiencyComparison",
    "ImportanceSamplingResult",
    "StratifiedEstimate",
    "allocate_stratified_budget",
    "compare_efficiency",
    "convergence_trace",
    "estimate_antithetic_uniform",
    "estimate_conditional_expectation",
    "estimate_control_variate",
    "estimate_importance_sampling",
    "estimate_mean",
    "estimate_probability",
    "estimate_stratified_uniform",
    "estimate_uniform_integral",
]
