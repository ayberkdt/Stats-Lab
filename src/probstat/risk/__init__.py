"""Quantitative risk analysis: scenarios x probabilities x consequences.

The package combines sample risk measures (expected loss, VaR, CVaR, loss
exceedance), consequence models applied to a
:class:`~probstat.modeling.ProbabilisticModel`, static fault trees, event
trees, lognormal fragility curves with hazard convolution, and decision
analysis with value-of-information measures. A risk matrix is provided only
as a communication aid with its documented limitations.
"""

from .consequences import (
    LOSS_KINDS,
    LossFunction,
    RiskAssessment,
    assess_risk,
    custom_loss,
    linear_loss,
    piecewise_linear_loss,
    step_loss,
)
from .decision import DecisionProblem, PreposteriorAnalysis, RiskMatrix
from .event_tree import (
    Branch,
    EventSequence,
    EventTree,
    Heading,
    Outcome,
    binary_heading,
)
from .fault_tree import (
    GATE_KINDS,
    BasicEvent,
    FaultTree,
    Gate,
    ImportanceMeasures,
    and_gate,
    k_of_n_gate,
    or_gate,
)
from .fragility import (
    FragilityFit,
    HazardConvolution,
    LognormalFragility,
    annual_failure_frequency,
    fit_lognormal_fragility,
    power_law_failure_frequency,
)
from .measures import (
    MEASURES,
    BootstrapInterval,
    bootstrap_risk_interval,
    conditional_value_at_risk,
    exceedance_probability,
    expected_loss,
    expected_shortfall,
    lognormal_conditional_value_at_risk,
    lognormal_value_at_risk,
    loss_exceedance_curve,
    normal_conditional_value_at_risk,
    normal_value_at_risk,
    poisson_exceedance_probability,
    return_period,
    value_at_risk,
)

__all__ = [
    "GATE_KINDS",
    "LOSS_KINDS",
    "MEASURES",
    "BasicEvent",
    "BootstrapInterval",
    "Branch",
    "DecisionProblem",
    "EventSequence",
    "EventTree",
    "FaultTree",
    "FragilityFit",
    "Gate",
    "HazardConvolution",
    "Heading",
    "ImportanceMeasures",
    "LognormalFragility",
    "LossFunction",
    "Outcome",
    "PreposteriorAnalysis",
    "RiskAssessment",
    "RiskMatrix",
    "and_gate",
    "annual_failure_frequency",
    "assess_risk",
    "binary_heading",
    "bootstrap_risk_interval",
    "conditional_value_at_risk",
    "custom_loss",
    "exceedance_probability",
    "expected_loss",
    "expected_shortfall",
    "fit_lognormal_fragility",
    "k_of_n_gate",
    "linear_loss",
    "lognormal_conditional_value_at_risk",
    "lognormal_value_at_risk",
    "loss_exceedance_curve",
    "normal_conditional_value_at_risk",
    "normal_value_at_risk",
    "or_gate",
    "piecewise_linear_loss",
    "poisson_exceedance_probability",
    "power_law_failure_frequency",
    "return_period",
    "step_loss",
    "value_at_risk",
]
