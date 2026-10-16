"""Engineering cost analysis: life-cycle cost, cost risk, and maintenance economics.

- :mod:`.discounting`: present value, annuities, Fisher real/nominal rates,
  escalation, equivalent annual cost.
- :mod:`.life_cycle`: a cost breakdown structure of uncertain elements with
  time profiles; converts to a :class:`~probstat.modeling.ProbabilisticModel`.
- :mod:`.moments` and :mod:`.cost_risk`: total-cost S-curves by the method of
  moments with correlation and by Gaussian-copula Monte Carlo, confidence
  levels, contingency, and joint cost-schedule confidence.
- :mod:`.learning`: Wright and Crawford learning curves and lot costs.
- :mod:`.maintenance`: age, block, and minimal-repair replacement policies.
- :mod:`.spares`: Poisson spare provisioning.
- :mod:`.unreliability`: expected failure cost and design trade-offs.
"""

from .cost_risk import (
    DEFAULT_LEVELS,
    FAMILIES,
    ConfidenceLevelRow,
    CostRiskResult,
    JointConfidenceLevel,
    cost_for_joint_confidence,
    cost_risk_analysis,
    fit_total_distribution,
    joint_confidence_level,
    life_cycle_cost_risk,
)
from .discounting import (
    COMPOUNDING,
    annuity_factor,
    capital_recovery_factor,
    continuous_rate,
    discount_factor,
    equivalent_annual_cost,
    escalate,
    net_present_value,
    nominal_rate,
    present_value,
    real_rate,
)
from .learning import THEORIES, LearningCurve, LotApproximation, learning_exponent
from .life_cycle import COST_CATEGORIES, CostBreakdown, CostElement, LifeCycleCostModel
from .maintenance import (
    CorrectiveCost,
    LifetimeLaw,
    RenewalFunction,
    ReplacementPolicyResult,
    age_replacement_cost_rate,
    expected_corrective_cost,
    minimal_repair_overhaul,
    optimize_age_replacement,
    optimize_block_replacement,
    optimize_periodic_overhaul,
    periodic_overhaul_cost_rate,
    renewal_function,
)
from .moments import MomentTotal, method_of_moments_total
from .spares import (
    SparePlan,
    expected_backorders,
    minimum_spares,
    optimal_spares,
    protection_level,
    spare_demand_mean,
    spare_plan_cost,
)
from .unreliability import (
    AlternativeEvaluation,
    DesignAlternative,
    FailureCostItem,
    UnreliabilityCost,
    compare_alternatives,
    cost_effectiveness_ratio,
    cost_of_unreliability,
    discounted_rate_factor,
    expected_failure_cost,
    incremental_cost_effectiveness,
    reliability_decision_problem,
)

__all__ = [
    "COMPOUNDING",
    "COST_CATEGORIES",
    "DEFAULT_LEVELS",
    "FAMILIES",
    "THEORIES",
    "AlternativeEvaluation",
    "ConfidenceLevelRow",
    "CorrectiveCost",
    "CostBreakdown",
    "CostElement",
    "CostRiskResult",
    "DesignAlternative",
    "FailureCostItem",
    "JointConfidenceLevel",
    "LearningCurve",
    "LifeCycleCostModel",
    "LifetimeLaw",
    "LotApproximation",
    "MomentTotal",
    "RenewalFunction",
    "ReplacementPolicyResult",
    "SparePlan",
    "UnreliabilityCost",
    "age_replacement_cost_rate",
    "annuity_factor",
    "capital_recovery_factor",
    "compare_alternatives",
    "continuous_rate",
    "cost_effectiveness_ratio",
    "cost_for_joint_confidence",
    "cost_of_unreliability",
    "cost_risk_analysis",
    "discount_factor",
    "discounted_rate_factor",
    "equivalent_annual_cost",
    "escalate",
    "expected_backorders",
    "expected_corrective_cost",
    "expected_failure_cost",
    "fit_total_distribution",
    "incremental_cost_effectiveness",
    "joint_confidence_level",
    "learning_exponent",
    "life_cycle_cost_risk",
    "method_of_moments_total",
    "minimal_repair_overhaul",
    "minimum_spares",
    "net_present_value",
    "nominal_rate",
    "optimal_spares",
    "optimize_age_replacement",
    "optimize_block_replacement",
    "optimize_periodic_overhaul",
    "periodic_overhaul_cost_rate",
    "present_value",
    "protection_level",
    "real_rate",
    "reliability_decision_problem",
    "renewal_function",
    "spare_demand_mean",
    "spare_plan_cost",
]
