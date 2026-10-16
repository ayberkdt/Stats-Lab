"""System reliability models.

``markov`` owns analytic Markov reliability and availability models
(:class:`MarkovReliabilityModel` and builders for repairable units,
``k``-out-of-``n`` systems, redundant pairs, standby and load-sharing
systems); ``phased_markov`` owns phased missions solved with continuous-time
Markov chains and discrete-time stage models with absorbing outcomes.
"""

from .markov import (
    MarkovReliabilityModel,
    from_states,
    load_sharing_pair,
    parallel_redundant_pair,
    repairable_k_out_of_n,
    single_repairable_unit,
    standby_system,
)
from .phased_markov import (
    DiscreteMissionOutcome,
    MarkovPhasedMission,
    MissionPhase,
    MissionStage,
    MissionStageModel,
    PhasedMissionResult,
    discrete_mission_outcome,
)

__all__ = [
    "DiscreteMissionOutcome",
    "MarkovPhasedMission",
    "MarkovReliabilityModel",
    "MissionPhase",
    "MissionStage",
    "MissionStageModel",
    "PhasedMissionResult",
    "discrete_mission_outcome",
    "from_states",
    "load_sharing_pair",
    "parallel_redundant_pair",
    "repairable_k_out_of_n",
    "single_repairable_unit",
    "standby_system",
]
