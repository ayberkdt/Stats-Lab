"""Quantitative risk analysis: scenarios x probabilities x consequences.

The package combines sample risk measures (expected loss, VaR, CVaR, loss
exceedance), consequence models applied to a
:class:`~probstat.modeling.ProbabilisticModel`, static fault trees, event
trees, lognormal fragility curves with hazard convolution, and decision
analysis with value-of-information measures. A risk matrix is provided only
as a communication aid with its documented limitations. Extreme value
analysis (block maxima with the GEV, peaks over threshold with the GPD,
declustering, and return levels with intervals) lives in :mod:`.extremes`.

Engineering risk-management tools: FMECA with MIL-STD-1629A criticality
numbers (and RPN with its documented caveats) in :mod:`.fmeca`, a
quantitative risk register with exact totals and mitigation waterfalls in
:mod:`.register`, the MIL-STD-882E risk assessment matrix in
:mod:`.hazard_matrix`, and F-N curves in :mod:`.fn_curve`. Sequential decisions
under uncertainty (the Beta-Bernoulli bandit with Thompson sampling) live in
:mod:`.bandits`; contextual bandits with linear payoffs (LinUCB and linear
Thompson sampling) in :mod:`.contextual_bandits`; off-policy evaluation from
logged bandit data (IPS, self-normalized IPS, direct method, doubly robust)
in :mod:`.off_policy`; finite Markov decision processes (policy evaluation,
value iteration, policy iteration, backward induction) in
:mod:`.markov_decision`; and fixed-budget best-arm identification by
sequential halving in :mod:`.best_arm`.

Main entry points:
    ``assess_risk``: loss distribution, expected loss, VaR, CVaR, and
    exceedance curve of one model output.
    ``conditional_value_at_risk``: tail mean of a weighted loss sample.
    ``FaultTree``: minimal cut sets, exact top-event probability, and
    importance measures.
    ``EventTree``: sequence frequencies of initiating events.
    ``DecisionProblem``: expected values, EVPI, and preposterior analysis.
    ``RiskRegister``: exact exposure, ranking, and mitigation waterfalls.
    ``fit_pot``: peaks-over-threshold tail fit with return levels.
    ``fit_gev``: block-maxima tail fit with return levels.
    ``FiniteMDP`` with ``value_iteration`` / ``policy_iteration``: optimal
    values and policy of a finite discounted Markov decision process.
    ``doubly_robust``: value of a target policy from logged bandit data.
    ``LinUCB`` and ``LinearThompsonSampling``: contextual bandit policies.
    ``sequential_halving``: best arm within a fixed pull budget.

Assumptions:
    Losses are oriented so that larger is worse. Sample measures treat the
    weights as a probability distribution over the given losses and inherit
    the sampling error of the sample. Fault-tree basic events and register
    risks are independent, and an impact is independent of its occurrence.
    Extreme value fits assume a stationary series and, for POT, a threshold
    high enough for the GPD to hold and independent (or declustered)
    exceedances. A risk matrix is a communication aid only. A ``FiniteMDP``
    has known transition probabilities, every action available in every
    state, and a discount below one. Off-policy estimators assume
    independent records and logging propensities that are positive wherever
    the target policy acts.

Common pitfalls:
    - Sample risk measures read larger values as worse, while
      ``DecisionProblem`` defaults to ``maximize=True`` (payoffs are
      utilities). Negate gains before passing them as losses.
    - ``value_at_risk`` is the lower quantile and always an observed loss:
      ``value_at_risk(range(1, 11), 0.7)`` is ``7``, not an interpolated
      value.
    - ``fit_pot`` and ``exceedance_rate`` default to
      ``observations_per_year=1.0``, so rates and return periods are per
      observation until the sampling frequency is given.
    - ``assess_risk`` and ``bootstrap_risk_interval`` default to
      ``seed=None`` (fresh entropy); pass an integer seed to reproduce them.
    - ``Fmeca.inconsistent_items`` flags only mode ratios that sum to more
      than one; a smaller sum (missing modes) is visible only in
      ``mode_ratio_sums``.
    - ``inverse_propensity_score`` is unbiased only with the true logging
      propensities; a small ``effective_sample_size`` means the estimate
      rests on a few records, and ``clip`` trades that variance for bias.
    - ``value_iteration(tolerance=eps)`` bounds the loss of the returned
      policy by ``eps``; the number of sweeps grows like ``1 / (1 - gamma)``.

Guide: docs/foundations/risk-analysis.md

References:
    [REF-KAPLAN-GARRICK-1981], [REF-ROCKAFELLAR-URYASEV-2002],
    [REF-ARTZNER-ETAL-1999], [REF-VESELY-ETAL-1981], [REF-COX-2008],
    [REF-COLES-2001], [REF-LI-ETAL-2010], [REF-AGRAWAL-GOYAL-2013],
    [REF-DUDIK-LANGFORD-LI-2011], [REF-NORRIS-1997],
    [REF-KARNIN-KOREN-SOMEKH-2013].

"""

from .bandits import BanditRun, bernoulli_bandit
from .best_arm import HalvingRound, SequentialHalvingResult, sequential_halving
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
from .contextual_bandits import (
    LinearBanditRun,
    LinearThompsonSampling,
    LinUCB,
    simulate_linear_bandit,
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
from .extremes import (
    GPD_FAMILY,
    DeclusteredExceedances,
    ExceedanceRate,
    GEVFit,
    GPDFit,
    MeanResidualLifePoint,
    PeaksOverThresholdFit,
    ReturnLevel,
    ThresholdStabilityPoint,
    block_maxima,
    decluster_runs,
    exceedance_rate,
    fit_gev,
    fit_gpd,
    fit_pot,
    gev_return_level,
    gev_return_level_gradient,
    mean_residual_life,
    pot_return_level,
    sample_l_moments,
    threshold_excesses,
    threshold_stability,
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
from .fmeca import (
    SEVERITY_CLASSES,
    CriticalityMatrix,
    FailureMode,
    Fmeca,
    ItemCriticality,
    risk_priority_number,
)
from .fn_curve import FNCriterion, FNCurve, FNPoint, fn_curve, fn_curve_from_event_trees
from .fragility import (
    FragilityFit,
    HazardConvolution,
    LognormalFragility,
    annual_failure_frequency,
    fit_lognormal_fragility,
    power_law_failure_frequency,
)
from .hazard_matrix import (
    MIL_STD_882E_TABLE,
    PROBABILITY_LEVELS,
    RISK_LEVELS,
    SEVERITY_CATEGORIES,
    hazard_risk_level,
    mil_std_882e_risk_matrix,
    risk_assessment_code,
)
from .markov_decision import (
    FiniteHorizonSolution,
    FiniteMDP,
    MDPSolution,
    action_values,
    backward_induction,
    bellman_residual,
    greedy_policy,
    policy_evaluation,
    policy_iteration,
    value_iteration,
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
from .off_policy import (
    LoggedBanditData,
    OffPolicyEstimate,
    direct_method,
    doubly_robust,
    inverse_propensity_score,
    logged_bandit_data,
    self_normalized_ips,
)
from .register import (
    RANKING_KEYS,
    DiscreteImpact,
    ExposureDistribution,
    ImpactLaw,
    Risk,
    RiskRegister,
    WaterfallStep,
    register_from_rows,
)

__all__ = [
    "BanditRun",
    "bernoulli_bandit",
    "HalvingRound",
    "SequentialHalvingResult",
    "sequential_halving",
    "LinearBanditRun",
    "LinearThompsonSampling",
    "LinUCB",
    "simulate_linear_bandit",
    "FiniteHorizonSolution",
    "FiniteMDP",
    "MDPSolution",
    "action_values",
    "backward_induction",
    "bellman_residual",
    "greedy_policy",
    "policy_evaluation",
    "policy_iteration",
    "value_iteration",
    "LoggedBanditData",
    "OffPolicyEstimate",
    "direct_method",
    "doubly_robust",
    "inverse_propensity_score",
    "logged_bandit_data",
    "self_normalized_ips",
    "GATE_KINDS",
    "GPD_FAMILY",
    "LOSS_KINDS",
    "MEASURES",
    "MIL_STD_882E_TABLE",
    "PROBABILITY_LEVELS",
    "RANKING_KEYS",
    "RISK_LEVELS",
    "SEVERITY_CATEGORIES",
    "SEVERITY_CLASSES",
    "BasicEvent",
    "BootstrapInterval",
    "Branch",
    "CriticalityMatrix",
    "DecisionProblem",
    "DeclusteredExceedances",
    "DiscreteImpact",
    "EventSequence",
    "EventTree",
    "ExceedanceRate",
    "ExposureDistribution",
    "FNCriterion",
    "FNCurve",
    "FNPoint",
    "FailureMode",
    "FaultTree",
    "Fmeca",
    "FragilityFit",
    "GEVFit",
    "GPDFit",
    "Gate",
    "HazardConvolution",
    "Heading",
    "ImpactLaw",
    "ImportanceMeasures",
    "ItemCriticality",
    "LognormalFragility",
    "LossFunction",
    "MeanResidualLifePoint",
    "Outcome",
    "PeaksOverThresholdFit",
    "PreposteriorAnalysis",
    "ReturnLevel",
    "Risk",
    "RiskAssessment",
    "RiskMatrix",
    "RiskRegister",
    "ThresholdStabilityPoint",
    "WaterfallStep",
    "and_gate",
    "annual_failure_frequency",
    "assess_risk",
    "binary_heading",
    "block_maxima",
    "bootstrap_risk_interval",
    "conditional_value_at_risk",
    "custom_loss",
    "decluster_runs",
    "exceedance_probability",
    "exceedance_rate",
    "expected_loss",
    "expected_shortfall",
    "fit_gev",
    "fit_gpd",
    "fit_lognormal_fragility",
    "fit_pot",
    "fn_curve",
    "fn_curve_from_event_trees",
    "gev_return_level",
    "gev_return_level_gradient",
    "hazard_risk_level",
    "k_of_n_gate",
    "linear_loss",
    "lognormal_conditional_value_at_risk",
    "lognormal_value_at_risk",
    "loss_exceedance_curve",
    "mean_residual_life",
    "mil_std_882e_risk_matrix",
    "normal_conditional_value_at_risk",
    "normal_value_at_risk",
    "or_gate",
    "piecewise_linear_loss",
    "poisson_exceedance_probability",
    "pot_return_level",
    "power_law_failure_frequency",
    "register_from_rows",
    "return_period",
    "risk_assessment_code",
    "risk_priority_number",
    "sample_l_moments",
    "step_loss",
    "threshold_excesses",
    "threshold_stability",
    "value_at_risk",
]
