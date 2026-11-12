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

Dynamic fault trees (:class:`DynamicFaultTree` with PAND, spare, FDEP and
SEQ gates) are quantified exactly by state-space generation into a
continuous-time Markov chain and cross-checked by
:func:`monte_carlo_unreliability`. :meth:`BDD.sift` reduces diagram sizes
by Rudell's sifting without changing the represented functions.

Repairable units and subsystems with general (non-exponential) sojourn laws
are semi-Markov models (:class:`SemiMarkovReliabilityModel`,
:func:`alternating_renewal_unit`, :class:`SemiMarkovComponent`). CCF
parameters can be epistemic inputs: :func:`ccf_parameter_uncertainty` gives
the exact (or Monte Carlo) mean and variance of the reliability and the
share of its variance due to CCF-parameter uncertainty, and
:func:`ccf_interval_bounds` bounds it over interval parameters with a
probability box.

Main entry points:
    ``SystemStructure``: exact reliability, cut and path sets, and bounds of
    a structure expression on a BDD.
    ``SystemModel``: component time models, common-cause groups, MTTF, and
    availability of one system.
    ``importance_measures``: Birnbaum, criticality, Fussell-Vesely, RAW, RRW,
    and structural importance.
    ``PhasedMission``: phased missions with non-repairable components.
    ``MarkovReliabilityModel``: availability, reliability, and MTTF of a
    continuous-time Markov chain with up and down states.
    ``DynamicFaultTree``: PAND, spare, FDEP, and SEQ gates by state-space
    generation.
    ``ccf_parameter_uncertainty``: mean and variance of the reliability over
    uncertain CCF parameters.

Assumptions:
    Component states are independent except through declared common-cause
    groups, shared Markov blocks, or dynamic gates; a component name that
    appears twice is the same physical component. The structure builders are
    monotone, and minimal cut sets and the classical bounds need monotone
    functions. Markov models assume exponential sojourns and constant rates
    (semi-Markov models relax the sojourn law); phased missions assume no
    repair during the mission. Rates and times must use one time unit.
    Composition through ``MarkovComponent`` and ``SubsystemComponent`` is
    exact only for modules that share nothing with the rest of the system.

Common pitfalls:
    - ``FixedProbability(p)`` takes the probability of working;
      ``DemandComponent(q)`` takes the probability of failing on demand.
    - ``1 - reliability`` loses a small unreliability to rounding. Call
      ``unreliability`` and pass ``unreliabilities=`` with the component
      complements.
    - Transient Markov measures (``reliability(t)``, ``availability(t)``)
      return a ``BoundedValue`` with ``value`` and ``error_bound``; the
      steady-state measures and the MTTF return plain floats.
    - ``SystemModel.mttf`` raises ``ValueError`` when ``R(t)`` does not tend
      to zero (for example with ``FixedProbability`` components).
    - ``ccf_basis="auto"`` splits the cumulative hazard for mission
      reliability and the probability for availability; the two agree only
      to first order in the member failure probability.

Guide: docs/foundations/system-reliability.md

References:
    [REF-BRYANT-1986], [REF-RAUZY-1993], [REF-ESARY-PROSCHAN-WALKUP-1967],
    [REF-MOSLEH-ETAL-1998], [REF-ZANG-SUN-TRIVEDI-1999],
    [REF-DUGAN-BAVUSO-BOYD-1992], [REF-LIMNIOS-OPRISAN-2001].

"""

from .adapters import MarkovComponent, SemiMarkovComponent, SubsystemComponent
from .bdd import BDD, PolynomialMoments, ReorderResult, VariableEffect
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
from .dynamic import (
    DFTMonteCarloEstimate,
    DFTStateSpace,
    DynamicBasicEvent,
    DynamicFaultTree,
    DynamicGate,
    and_gate,
    fdep_gate,
    monte_carlo_unreliability,
    or_gate,
    pand_gate,
    seq_gate,
    spare_gate,
    vote_gate,
)
from .epistemic_ccf import (
    CCFIntervalBounds,
    EpistemicCCFResult,
    ccf_interval_bounds,
    ccf_parameter_uncertainty,
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
from .semi_markov import SemiMarkovReliabilityModel, alternating_renewal_unit
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
    "CCFIntervalBounds",
    "CommonCauseExpansion",
    "Component",
    "ComponentModel",
    "DFTMonteCarloEstimate",
    "DFTStateSpace",
    "DemandComponent",
    "DiscreteMissionOutcome",
    "DynamicBasicEvent",
    "DynamicFaultTree",
    "DynamicGate",
    "EpistemicCCFResult",
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
    "ReorderResult",
    "RepairableComponent",
    "SemiMarkovComponent",
    "SemiMarkovReliabilityModel",
    "Series",
    "SubsystemComponent",
    "SystemModel",
    "SystemStructure",
    "VariableEffect",
    "WeibullComponent",
    "and_gate",
    "alternating_renewal_unit",
    "bridge",
    "ccf_interval_bounds",
    "ccf_parameter_uncertainty",
    "compile_expression",
    "discrete_mission_outcome",
    "expand_common_cause",
    "expression_components",
    "fdep_gate",
    "from_fault_tree",
    "from_states",
    "importance_measures",
    "k_out_of_n",
    "load_sharing_pair",
    "monte_carlo_unreliability",
    "normalisation",
    "or_gate",
    "order_probabilities",
    "pand_gate",
    "parallel",
    "parallel_redundant_pair",
    "repairable_k_out_of_n",
    "seq_gate",
    "series",
    "single_repairable_unit",
    "spare_gate",
    "standby_system",
    "structural_importance",
    "vote_gate",
]
