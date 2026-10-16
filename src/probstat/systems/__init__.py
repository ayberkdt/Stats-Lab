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
"""

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
from .phased import Phase, PhasedMission, PhasedMissionResult
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
    "ExponentialComponent",
    "Expression",
    "FixedProbability",
    "ImportanceMeasures",
    "KOutOfN",
    "LifeDistributionComponent",
    "MTTFResult",
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
    "SystemModel",
    "SystemStructure",
    "VariableEffect",
    "WeibullComponent",
    "bridge",
    "compile_expression",
    "expand_common_cause",
    "expression_components",
    "from_fault_tree",
    "importance_measures",
    "k_out_of_n",
    "normalisation",
    "order_probabilities",
    "parallel",
    "series",
    "structural_importance",
]
