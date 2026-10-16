"""Exact system reliability: structures, BDDs, importance, CCF, phased missions.

A systems engineer describes a mission system from its components with
success-space structure expressions (:func:`series`, :func:`parallel`,
:func:`k_out_of_n`, :func:`bridge`) or imports fault-tree logic
(:func:`from_fault_tree`). :class:`SystemStructure` compiles the structure
into a reduced ordered binary decision diagram (:class:`BDD`) and computes,
exactly and without sampling, the reliability, minimal cut and path sets,
classical bounds, all standard importance measures, and the exact epistemic
mean and variance of the reliability. :class:`SystemModel` adds component
time models, common-cause failure groups (beta factor, multiple Greek
letter, alpha factor), MTTF, availability, and a bridge to
:class:`~probstat.modeling.ProbabilisticModel`; :class:`PhasedMission`
handles phased missions with non-repairable components.
:class:`MarkovComponent` and :class:`SubsystemComponent` let a Markov model
or a whole subsystem act as one block of a larger system (exact for
independent modules).

Repairable and standby systems are solved analytically with continuous-time
Markov chains (:class:`MarkovReliabilityModel` and its builders), phased
missions with repair with :class:`MarkovPhasedMission`, and discrete-time
missions with absorbing success or failure outcomes with
:func:`discrete_mission_outcome` and :class:`MissionStageModel`.
"""

from .adapters import MarkovComponent, SubsystemComponent
from .bdd import BDD, PolynomialMoments, VariableEffect
from .ccf import (
    AlphaFactorGroup,
    BetaFactorGroup,
    CCFEvent,
    CCFGroup,
    CommonCauseExpansion,
    MultipleGreekLetterGroup,
    expand_common_cause,
    normalisation,
    order_probabilities,
)
from .components import (
    ComponentModel,
    DemandComponent,
    ExponentialComponent,
    FixedProbability,
    LifeDistributionComponent,
    PeriodicallyTestedComponent,
    RepairableComponent,
    WeibullComponent,
)
from .importance import ImportanceMeasures, importance_measures, structural_importance
from .markov import (
    MarkovReliabilityModel,
    from_states,
    load_sharing_pair,
    parallel_redundant_pair,
    repairable_k_out_of_n,
    single_repairable_unit,
    standby_system,
)
from .phased import Phase, PhasedMission, PhasedMissionResult
from .phased_markov import (
    DiscreteMissionOutcome,
    MarkovPhasedMission,
    MarkovPhasedMissionResult,
    MissionPhase,
    MissionStage,
    MissionStageModel,
    discrete_mission_outcome,
)
from .structure import (
    Component,
    Expression,
    KOutOfN,
    Parallel,
    ReliabilityBounds,
    ReliabilityMoments,
    Series,
    SystemStructure,
    bridge,
    compile_expression,
    expression_components,
    from_fault_tree,
    k_out_of_n,
    parallel,
    series,
)
from .system import MTTFResult, SystemModel

__all__ = [
    "BDD",
    "AlphaFactorGroup",
    "BetaFactorGroup",
    "CCFEvent",
    "CCFGroup",
    "CommonCauseExpansion",
    "Component",
    "ComponentModel",
    "DemandComponent",
    "DiscreteMissionOutcome",
    "ExponentialComponent",
    "Expression",
    "FixedProbability",
    "ImportanceMeasures",
    "KOutOfN",
    "LifeDistributionComponent",
    "MarkovComponent",
    "MTTFResult",
    "MarkovPhasedMission",
    "MarkovPhasedMissionResult",
    "MarkovReliabilityModel",
    "MissionPhase",
    "MissionStage",
    "MissionStageModel",
    "MultipleGreekLetterGroup",
    "Parallel",
    "PeriodicallyTestedComponent",
    "Phase",
    "PhasedMission",
    "PhasedMissionResult",
    "PolynomialMoments",
    "ReliabilityBounds",
    "ReliabilityMoments",
    "RepairableComponent",
    "Series",
    "SubsystemComponent",
    "SystemModel",
    "SystemStructure",
    "VariableEffect",
    "WeibullComponent",
    "bridge",
    "compile_expression",
    "discrete_mission_outcome",
    "expand_common_cause",
    "expression_components",
    "from_fault_tree",
    "from_states",
    "importance_measures",
    "k_out_of_n",
    "load_sharing_pair",
    "normalisation",
    "order_probabilities",
    "parallel",
    "parallel_redundant_pair",
    "repairable_k_out_of_n",
    "series",
    "single_repairable_unit",
    "standby_system",
    "structural_importance",
]
