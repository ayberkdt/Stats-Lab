"""Phased missions solved with Markov chains.

Continuous time: a :class:`MarkovPhasedMission` is a sequence of
:class:`MissionPhase` objects. Each phase has its own CTMC (its own
configuration, stresses and repair policy), a fixed duration and a set of
*success* states that meet the phase requirement. The state law at the end
of one phase, restricted to that phase's success states, is carried into
the next phase through an optional mapping matrix (identity by label by
default) [EQ-SYS-MKPM-001]. Within a phase the requirement is checked either
continuously (failure states are absorbing: any entry fails the mission) or
only at the end of the phase.

Discrete time: :func:`discrete_mission_outcome` analyses a mission written as
a discrete-time Markov chain with absorbing outcomes (success, abort, loss),
and :class:`MissionStageModel` builds such a chain from a list of stages with
per-stage success, retry, abort and failure probabilities
[EQ-SYS-MKPM-002].
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from probstat.foundations.validation import require_finite_real, require_probability
from probstat.linalg import Matrix, as_matrix
from probstat.processes.ctmc import BoundedValue, ContinuousTimeMarkovChain
from probstat.processes.markov import MarkovChain

State = Hashable
_ROW_TOLERANCE = 1e-10


@dataclass(frozen=True, eq=False)
class MissionPhase:
    """One phase of a phased mission.

    Args:
        name: Phase name.
        chain: CTMC of the system during this phase.
        duration: Non-negative phase duration.
        success_states: Labels of the states that satisfy the phase
            requirement; the other states are phase failures.
        entry_mapping: How the previous phase's success states map into this
            phase's states: ``None`` (identity by label), a mapping
            ``{(previous_label, label): probability}``, or a matrix with one
            row per previous state and one column per state of this phase.
            Rows must be non-negative and sum to at most one; a deficit is
            the probability of failing at the phase change (for example a
            reconfiguration failure). Must be ``None`` for the first phase.
        continuous: If true the requirement holds throughout the phase (the
            failure states are made absorbing); if false it is checked only
            at the end of the phase, so the system may fail and be repaired
            within the phase.

    Raises:
        ValueError: If the duration or the success states are invalid.

    References:
        [REF-SMOTHERMAN-ZEMOUDEH-1989], [REF-TRIVEDI-2002].

    """

    name: str
    chain: ContinuousTimeMarkovChain
    duration: float
    success_states: tuple[State, ...]
    entry_mapping: object = None
    continuous: bool = True
    failure_states: tuple[State, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Validate the phase definition.

        Raises:
            TypeError: If ``chain`` is not a CTMC or ``continuous`` not a bool.
            ValueError: If the duration or success states are invalid.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.chain, ContinuousTimeMarkovChain):
            raise TypeError("chain must be a ContinuousTimeMarkovChain")
        duration = require_finite_real(self.duration, name="duration")
        if duration < 0.0:
            raise ValueError("duration must be non-negative")
        if not isinstance(self.continuous, bool):
            raise TypeError("continuous must be a bool")
        chosen = {self.chain.index(state) for state in self.success_states}
        if not chosen:
            raise ValueError("success_states must be non-empty")
        states = self.chain.states
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "duration", duration)
        object.__setattr__(
            self,
            "success_states",
            tuple(label for i, label in enumerate(states) if i in chosen),
        )
        object.__setattr__(
            self,
            "failure_states",
            tuple(label for i, label in enumerate(states) if i not in chosen),
        )


@dataclass(frozen=True, slots=True)
class MarkovPhasedMissionResult:
    """Outcome of a phased-mission analysis.

    Args:
        phase_names: Phase names in order.
        phase_reliabilities: ``Pr(phase i succeeds | phases < i succeeded)``,
            including any loss at the phase's entry mapping; ``nan`` when the
            previous cumulative probability is zero.
        cumulative_success: ``Pr(phases 1..i all succeed)``.
        mission_success: ``Pr(every phase succeeds)``.
        error_bound: Absolute bound on the error of every cumulative value
            (sum of the per-phase uniformization bounds).
        final_states: State labels of the last phase.
        final_distribution: Sub-probability law over the last phase's states
            at mission end on the success event (zero on failure states).

    References:
        [REF-SMOTHERMAN-ZEMOUDEH-1989].

    """

    phase_names: tuple[str, ...]
    phase_reliabilities: tuple[float, ...]
    cumulative_success: tuple[float, ...]
    mission_success: float
    error_bound: float
    final_states: tuple[State, ...]
    final_distribution: tuple[float, ...]

    @property
    def mission_failure(self) -> float:
        """Return ``1 - mission_success``.

        References:
            [REF-SMOTHERMAN-ZEMOUDEH-1989]. No equation is evaluated.

        """
        return 1.0 - self.mission_success


def _mapping_matrix(
    mapping: object,
    previous: ContinuousTimeMarkovChain,
    current: ContinuousTimeMarkovChain,
    carried: Sequence[State],
) -> Matrix:
    """Return the validated phase-change matrix.

    Args:
        mapping: ``None``, ``{(previous, current): p}``, or a matrix.
        previous: Previous phase chain (rows).
        current: Current phase chain (columns).
        carried: Previous success states (rows that matter).

    Returns:
        Non-negative matrix with row sums at most one.

    Raises:
        ValueError: If the mapping is invalid.

    Equation:
        [EQ-SYS-MKPM-001].

    References:
        [REF-SMOTHERMAN-ZEMOUDEH-1989].

    """
    rows = previous.size
    columns = current.size
    if mapping is None:
        table = [[0.0] * columns for _ in range(rows)]
        for state in carried:
            try:
                column = current.index(state)
            except ValueError as error:
                raise ValueError(
                    f"success state {state!r} has no state of the same label in "
                    "the next phase; give an entry_mapping"
                ) from error
            table[previous.index(state)][column] = 1.0
        return tuple(tuple(row) for row in table)
    if isinstance(mapping, Mapping):
        table = [[0.0] * columns for _ in range(rows)]
        for key, value in mapping.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("entry_mapping keys must be (previous, current) pairs")
            probability = require_finite_real(value, name=f"entry_mapping{key!r}")
            table[previous.index(key[0])][current.index(key[1])] = probability
        matrix: Matrix = tuple(tuple(row) for row in table)
    else:
        matrix = as_matrix(
            cast(Iterable[Iterable[object]], mapping), name="entry_mapping"
        )
        if len(matrix) != rows or len(matrix[0]) != columns:
            raise ValueError(
                "entry_mapping must have one row per previous state and one "
                "column per current state"
            )
    for i, row in enumerate(matrix):
        if any(value < 0.0 for value in row):
            raise ValueError(f"entry_mapping row {i} has a negative entry")
        if math.fsum(row) > 1.0 + _ROW_TOLERANCE:
            raise ValueError(f"entry_mapping row {i} sums to more than one")
    return matrix


@dataclass(frozen=True, eq=False)
class MarkovPhasedMission:
    """A phased mission whose phases are continuous-time Markov chains.

    Args:
        phases: Non-empty sequence of :class:`MissionPhase`.
        initial: Initial law over the first phase's states (label, mapping,
            or probability vector).

    Raises:
        ValueError: If there are no phases, the first phase has an entry
            mapping, or a mapping is invalid.

    Equation:
        [EQ-SYS-MKPM-001].

    References:
        [REF-SMOTHERMAN-ZEMOUDEH-1989], [REF-TRIVEDI-2002].

    """

    phases: tuple[MissionPhase, ...]
    initial: object
    _mappings: tuple[Matrix | None, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the phases and precompute the phase-change matrices.

        Raises:
            TypeError: If a phase is not a :class:`MissionPhase`.
            ValueError: If the definition is invalid.

        References:
            Project validation policy. No equation is evaluated.

        """
        phases = tuple(self.phases)
        if not phases:
            raise ValueError("a mission needs at least one phase")
        if any(not isinstance(phase, MissionPhase) for phase in phases):
            raise TypeError("phases must be MissionPhase objects")
        if phases[0].entry_mapping is not None:
            raise ValueError("the first phase takes `initial`, not an entry_mapping")
        start = phases[0].chain.distribution(self.initial)
        mappings: list[Matrix | None] = [None]
        for previous, current in zip(phases, phases[1:], strict=False):
            mappings.append(
                _mapping_matrix(
                    current.entry_mapping,
                    previous.chain,
                    current.chain,
                    previous.success_states,
                )
            )
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "initial", start)
        object.__setattr__(self, "_mappings", tuple(mappings))

    def solve(self, tolerance: float = 1e-12) -> MarkovPhasedMissionResult:
        """Return the mission success probability and per-phase reliabilities.

        For each phase the entering sub-probability vector ``v`` (mass
        ``m``) is propagated over the phase duration by uniformization of the
        phase generator, with failure states absorbing when the phase is
        continuous; the success-state part is mapped into the next phase
        [EQ-SYS-MKPM-001]. The uniformization error bounds, scaled by the
        entering mass, add up because the mappings are non-expansive in L1.

        Args:
            tolerance: Uniformization truncation tolerance per phase.

        Returns:
            Mission result.

        Equation:
            [EQ-SYS-MKPM-001].

        References:
            [REF-SMOTHERMAN-ZEMOUDEH-1989], [REF-REIBMAN-TRIVEDI-1988].

        """
        vector = list(cast(tuple[float, ...], self.initial))
        bound = 0.0
        cumulative: list[float] = []
        conditional: list[float] = []
        previous_mass = 1.0
        for position, phase in enumerate(self.phases):
            mapping = self._mappings[position]
            if mapping is not None:
                carried = vector
                vector = [
                    math.fsum(carried[i] * mapping[i][j] for i in range(len(carried)))
                    for j in range(phase.chain.size)
                ]
            mass = math.fsum(vector)
            chain = (
                phase.chain.make_absorbing(phase.failure_states)
                if phase.continuous
                else phase.chain
            )
            end = [0.0] * chain.size
            if mass > 0.0:
                solution = chain.transient(
                    [value / mass for value in vector], phase.duration, tolerance
                )
                end = [value * mass for value in solution.probabilities]
                bound += solution.error_bound * mass
            success = {chain.index(state) for state in phase.success_states}
            vector = [value if i in success else 0.0 for i, value in enumerate(end)]
            achieved = math.fsum(vector)
            cumulative.append(achieved)
            conditional.append(
                achieved / previous_mass if previous_mass > 0.0 else math.nan
            )
            previous_mass = achieved
        return MarkovPhasedMissionResult(
            tuple(phase.name for phase in self.phases),
            tuple(conditional),
            tuple(cumulative),
            cumulative[-1],
            bound,
            self.phases[-1].chain.states,
            tuple(vector),
        )

    def success_probability(self, tolerance: float = 1e-12) -> BoundedValue:
        """Return the mission success probability with its error bound.

        Args:
            tolerance: Uniformization truncation tolerance per phase.

        Returns:
            Bounded success probability.

        Equation:
            [EQ-SYS-MKPM-001].

        References:
            [REF-SMOTHERMAN-ZEMOUDEH-1989].

        """
        result = self.solve(tolerance)
        return BoundedValue(result.mission_success, result.error_bound)


@dataclass(frozen=True, slots=True)
class DiscreteMissionOutcome:
    """Absorption outcome of a discrete-time mission model.

    Args:
        success_probability: Probability of absorption in a success state.
        failure_probability: Probability of absorption in a failure state.
        other_probability: Probability of absorption in any other absorbing
            state (for example abort).
        outcome_probabilities: Absorption probability per absorbing label.
        expected_steps: Expected number of steps until absorption.

    References:
        [REF-KEMENY-SNELL-1976, Chapter III].

    """

    success_probability: float
    failure_probability: float
    other_probability: float
    outcome_probabilities: Mapping[str, float]
    expected_steps: float


def discrete_mission_outcome(
    markov_chain: MarkovChain,
    start: int | str,
    success_states: Iterable[int | str],
    failure_states: Iterable[int | str],
) -> DiscreteMissionOutcome:
    """Return success/failure probabilities of a discrete-time mission.

    Uses :meth:`MarkovChain.absorption`: ``B = (I - Q)^-1 R`` and
    ``t = (I - Q)^-1 1`` [EQ-SYS-MKPM-002].

    Args:
        markov_chain: Chain whose closed classes are absorbing outcomes.
        start: Start state (index or label).
        success_states: Absorbing states that count as success.
        failure_states: Absorbing states that count as failure (disjoint from
            the success states); remaining absorbing states are "other".

    Returns:
        Outcome probabilities and expected number of steps.

    Raises:
        ValueError: If a listed state is not absorbing, the sets overlap, or
            the chain has no transient state.

    Equation:
        [EQ-SYS-MKPM-002], [EQ-PROC-MC-005].

    References:
        [REF-KEMENY-SNELL-1976, Chapter III], [REF-NORRIS-1997, Section 1.3].

    """
    if not isinstance(markov_chain, MarkovChain):
        raise TypeError("markov_chain must be a MarkovChain")
    analysis = markov_chain.absorption()
    origin = markov_chain.index(start)
    wins = {markov_chain.index(state) for state in success_states}
    losses = {markov_chain.index(state) for state in failure_states}
    if wins & losses:
        raise ValueError("success and failure states must be disjoint")
    for state in wins | losses:
        if state not in analysis.absorbing_states:
            raise ValueError(
                f"state {markov_chain.labels[state]!r} is not absorbing"
            )
    outcomes = {
        markov_chain.labels[state]: analysis.probability(origin, state)
        for state in analysis.absorbing_states
    }
    success = math.fsum(analysis.probability(origin, state) for state in wins)
    failure = math.fsum(analysis.probability(origin, state) for state in losses)
    other = math.fsum(
        analysis.probability(origin, state)
        for state in analysis.absorbing_states
        if state not in wins and state not in losses
    )
    return DiscreteMissionOutcome(
        success, failure, other, MappingProxyType(outcomes), analysis.steps(origin)
    )


@dataclass(frozen=True, slots=True)
class MissionStage:
    """One stage of a discrete mission.

    Each attempt of the stage ends in exactly one of: success (go to the next
    stage, or mission success after the last stage), retry (repeat the
    stage), abort (absorbing ``"abort"``), or failure (absorbing ``"loss"``).

    Args:
        name: Stage name (must not be ``"success"``, ``"abort"`` or
            ``"loss"``).
        success: Probability of completing the stage.
        retry: Probability of repeating the stage (``< 1``).
        abort: Probability of aborting the mission.
        failure: Probability of losing the mission; ``None`` means
            ``1 - success - retry - abort``.

    Raises:
        ValueError: If the probabilities are invalid or do not sum to one.

    References:
        [REF-KEMENY-SNELL-1976, Chapter III].

    """

    name: str
    success: float
    retry: float = 0.0
    abort: float = 0.0
    failure: float | None = None

    def __post_init__(self) -> None:
        """Validate the stage probabilities.

        Raises:
            ValueError: If the probabilities are invalid.

        References:
            Project validation policy. No equation is evaluated.

        """
        success = require_probability(self.success, name="success")
        retry = require_probability(self.retry, name="retry")
        abort = require_probability(self.abort, name="abort")
        if retry >= 1.0:
            raise ValueError("retry must be below one (the stage must end)")
        rest = 1.0 - math.fsum((success, retry, abort))
        if self.failure is None:
            if rest < -_ROW_TOLERANCE:
                raise ValueError("stage probabilities exceed one")
            failure = max(0.0, rest)
        else:
            failure = require_probability(self.failure, name="failure")
            if abs(rest - failure) > _ROW_TOLERANCE:
                raise ValueError("stage probabilities must sum to one")
        if str(self.name) in MissionStageModel.OUTCOMES:
            raise ValueError(f"stage name {self.name!r} is reserved")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "retry", retry)
        object.__setattr__(self, "abort", abort)
        object.__setattr__(self, "failure", failure)


@dataclass(frozen=True)
class MissionStageModel:
    """A discrete mission built from a sequence of stages.

    The chain has one transient state per stage and the absorbing outcomes
    ``"success"``, ``"abort"`` and ``"loss"``. The success probability is
    ``prod_i s_i / (1 - r_i)`` and the expected number of stage attempts
    satisfies ``T_i = (1 + s_i T_(i+1)) / (1 - r_i)`` with ``T_(n+1) = 0``
    [EQ-SYS-MKPM-002].

    Args:
        stages: Non-empty sequence of :class:`MissionStage` with distinct
            names.

    Raises:
        ValueError: If there are no stages or names repeat.

    Equation:
        [EQ-SYS-MKPM-002].

    References:
        [REF-KEMENY-SNELL-1976, Chapter III].

    """

    OUTCOMES = ("success", "abort", "loss")

    stages: tuple[MissionStage, ...]

    def __post_init__(self) -> None:
        """Validate the stages.

        Raises:
            TypeError: If a stage is not a :class:`MissionStage`.
            ValueError: If there are no stages or names repeat.

        References:
            Project validation policy. No equation is evaluated.

        """
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("a mission needs at least one stage")
        if any(not isinstance(stage, MissionStage) for stage in stages):
            raise TypeError("stages must be MissionStage objects")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be distinct")
        object.__setattr__(self, "stages", stages)

    def chain(self) -> MarkovChain:
        """Return the discrete-time chain of the mission.

        Returns:
            Chain with labels ``(*stage names, "success", "abort", "loss")``.

        Equation:
            [EQ-SYS-MKPM-002].

        References:
            [REF-KEMENY-SNELL-1976, Chapter III].

        """
        count = len(self.stages)
        size = count + 3
        success_index, abort_index, loss_index = count, count + 1, count + 2
        rows = [[0.0] * size for _ in range(size)]
        for i, stage in enumerate(self.stages):
            rows[i][i + 1 if i + 1 < count else success_index] += stage.success
            rows[i][i] += stage.retry
            rows[i][abort_index] += stage.abort
            rows[i][loss_index] += stage.failure or 0.0
        for index in (success_index, abort_index, loss_index):
            rows[index][index] = 1.0
        labels = (*(stage.name for stage in self.stages), *self.OUTCOMES)
        return MarkovChain(tuple(tuple(row) for row in rows), labels)

    def outcome(self) -> DiscreteMissionOutcome:
        """Return the mission outcome starting at the first stage.

        Returns:
            Success (``"success"``), failure (``"loss"``) and other
            (``"abort"``) probabilities and the expected number of attempts.

        Equation:
            [EQ-SYS-MKPM-002].

        References:
            [REF-KEMENY-SNELL-1976, Chapter III].

        """
        return discrete_mission_outcome(self.chain(), 0, ("success",), ("loss",))
