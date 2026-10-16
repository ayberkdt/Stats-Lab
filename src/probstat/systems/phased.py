"""Exact reliability of phased-mission systems with non-repairable components.

A phased mission (Zang, Sun and Trivedi 1999) runs through consecutive
phases ``1..J``; phase ``j`` lasts ``d_j`` and needs its own structure
``phi_j`` to work throughout the phase. Components are shared by the phases,
are not repaired during the mission, and may age at phase-dependent rates.
The mission succeeds when every phase succeeds.

Why "alive at the end of phase ``j``" suffices. Let ``x_{i,j} = 1`` mean
that component ``i`` is still working at the end of phase ``j``. Without
repair a component that works at the end of a phase worked during the whole
phase, so every component state is non-increasing in time within a phase.
Each ``phi_j`` is monotone, hence ``phi_j(x(t))`` is non-increasing in ``t``
and its minimum over the phase is attained at the phase end. The system
therefore works throughout phase ``j`` exactly when ``phi_j(x_{.,j}) = 1``,
and the mission succeeds exactly when ``AND_j phi_j(x_{.,j}) = 1``.

The variables ``x_{i,1} >= x_{i,2} >= ... >= x_{i,J}`` of one component are
dependent; they are determined by the component's *phase of failure*
``K_i in {1, ..., J, J + 1}`` (``J + 1``: survives the mission) through
``x_{i,j} = [j < K_i]``. Components are independent. The mission function is
built in one BDD with a component-major variable order, so the variables of
each component are consecutive. The probability is then evaluated exactly
bottom-up: at a node of component ``i``, enumerate ``K_i`` with its
probabilities and walk down deterministically through the component's
variables to the first node of a later component (or a terminal). This is
exact because the walk consumes the whole dependent block and the blocks of
different components are independent. (Zang et al. use phase algebra with a
reversed phase order instead; both give the exact probability.)

Phase-of-failure law: with per-phase models ``M_{i,j}`` applied to the age
within the phase (a component alive at the start of phase ``j`` survives it
with probability ``s_{i,j} = R_{i,j}(d_j)``), ``P(K_i = j) = prod_{l<j}
s_{i,l} (1 - s_{i,j})``. With a single model applied to the cumulative mission
time ``T_j = d_1 + ... + d_j``, ``P(K_i = j) = R_i(T_{j-1}) - R_i(T_j)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.validation import require_probability

from .bdd import BDD
from .components import ComponentModel
from .structure import (
    Component,
    Expression,
    KOutOfN,
    Parallel,
    Series,
    SystemStructure,
    compile_expression,
    expression_components,
)


def _phase_namer(phase: int) -> Callable[[str], str]:
    """Return ``component -> "<component>#<phase>"``.

    References:
        [REF-ZANG-SUN-TRIVEDI-1999]. No equation is evaluated.

    """

    def rename(component: str) -> str:
        """Return the phase variable name of a component.

        References:
            [REF-ZANG-SUN-TRIVEDI-1999]. No equation is evaluated.

        """
        return f"{component}#{phase}"

    return rename


@dataclass(frozen=True, slots=True)
class Phase:
    """One mission phase.

    Args:
        name: Phase label.
        duration: Phase length ``d_j > 0``.
        structure: Success structure required throughout the phase (an
            expression, a :class:`SystemStructure`, or a component name).

    References:
        [REF-ZANG-SUN-TRIVEDI-1999].

    """

    name: str
    duration: float
    structure: Expression

    def __post_init__(self) -> None:
        """Validate the fields.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If the name is empty or the duration not positive.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("phase names must be non-empty strings")
        if isinstance(self.duration, bool) or not isinstance(
            self.duration, int | float
        ):
            raise TypeError("duration must be a real number")
        duration = float(self.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        object.__setattr__(self, "duration", duration)
        structure: object = self.structure
        if isinstance(structure, SystemStructure):
            structure = structure.expression
        elif isinstance(structure, str):
            structure = Component(structure)
        if not isinstance(structure, Component | Series | Parallel | KOutOfN):
            raise TypeError("structure must be a structure expression")
        object.__setattr__(self, "structure", structure)


@dataclass(frozen=True, slots=True)
class PhasedMissionResult:
    """Exact phased-mission reliability with per-phase diagnostics.

    Args:
        phase_names: Phase labels in mission order.
        reliability: ``P(every phase succeeds)``.
        unreliability: ``1 - reliability``, computed directly.
        cumulative_reliability: ``P(phases 1..j all succeed)`` for each ``j``.
        conditional_reliability: ``P(phase j succeeds | phases 1..j-1
            succeeded)``; ``None`` when the condition has probability zero.
        phase_failure_probability: ``P(phases 1..j-1 succeed and phase j
            fails)``; these sum to the unreliability.

    References:
        [REF-ZANG-SUN-TRIVEDI-1999].

    """

    phase_names: tuple[str, ...]
    reliability: float
    unreliability: float
    cumulative_reliability: tuple[float, ...]
    conditional_reliability: tuple[float | None, ...]
    phase_failure_probability: tuple[float, ...]


class PhasedMission:
    """A phased-mission system with non-repairable components.

    Args:
        phases: Phases in mission order.
        components: Per component either one :class:`ComponentModel` (aging
            on the cumulative mission time) or one model per phase (each
            applied to the age within its phase, e.g. phase-dependent
            exponential rates).

    Raises:
        KeyError: If a used component has no model or a model names an
            unknown component.
        TypeError: If a model is invalid.
        ValueError: If phases are empty or repeat a name, or a per-phase
            model list has the wrong length.

    References:
        [REF-ZANG-SUN-TRIVEDI-1999], [REF-RAUZY-1993].

    """

    def __init__(
        self,
        phases: Sequence[Phase],
        components: Mapping[str, ComponentModel | Sequence[ComponentModel]],
    ) -> None:
        """Validate the mission and build the mission BDD.

        References:
            [REF-ZANG-SUN-TRIVEDI-1999]. No equation is evaluated.

        """
        self._phases = tuple(phases)
        if not self._phases:
            raise ValueError("a mission needs at least one phase")
        for index, phase in enumerate(self._phases):
            if not isinstance(phase, Phase):
                raise TypeError(f"phases[{index}] must be a Phase")
        names = [phase.name for phase in self._phases]
        if len(set(names)) != len(names):
            raise ValueError("phase names must be distinct")
        used: dict[str, None] = {}
        for phase in self._phases:
            for name in expression_components(phase.structure):
                used.setdefault(name, None)
        self._components = tuple(used)
        count = len(self._phases)
        models: dict[str, ComponentModel | tuple[ComponentModel, ...]] = {}
        for name, value in components.items():
            if name not in used:
                raise KeyError(f"model given for unknown component {name!r}")
            if isinstance(value, ComponentModel):
                models[name] = value
                continue
            sequence = tuple(value)
            if len(sequence) != count:
                raise ValueError(f"{name!r} needs one model per phase ({count})")
            for item in sequence:
                if not isinstance(item, ComponentModel):
                    raise TypeError(f"models of {name!r} must be ComponentModels")
            models[name] = sequence
        for name in self._components:
            if name not in models:
                raise KeyError(f"no model for component {name!r}")
        self._models: Mapping[str, ComponentModel | tuple[ComponentModel, ...]] = (
            MappingProxyType(models)
        )
        uses = [set(expression_components(phase.structure)) for phase in self._phases]
        order: list[str] = []
        self._variable: dict[str, tuple[str, int]] = {}
        for name in self._components:
            for j in range(count):
                if name in uses[j]:
                    label = f"{name}#{j + 1}"
                    order.append(label)
                    self._variable[label] = (name, j + 1)
        self._bdd = BDD(order)
        self._phase_roots = tuple(
            compile_expression(phase.structure, self._bdd, rename=_phase_namer(j + 1))
            for j, phase in enumerate(self._phases)
        )
        cumulative = []
        running = self._bdd.TRUE
        for root in self._phase_roots:
            running = self._bdd.apply_and(running, root)
            cumulative.append(running)
        self._cumulative = tuple(cumulative)

    @property
    def phases(self) -> tuple[Phase, ...]:
        """Return the phases in mission order.

        References:
            [REF-ZANG-SUN-TRIVEDI-1999].

        """
        return self._phases

    @property
    def components(self) -> tuple[str, ...]:
        """Return the components used by at least one phase.

        References:
            [REF-ZANG-SUN-TRIVEDI-1999].

        """
        return self._components

    @property
    def bdd(self) -> BDD:
        """Return the mission BDD (variables ``"<component>#<phase>"``).

        References:
            [REF-ZANG-SUN-TRIVEDI-1999], [REF-BRYANT-1986].

        """
        return self._bdd

    @property
    def mission_root(self) -> int:
        """Return the BDD node of ``AND_j phi_j(x_{., j})``.

        References:
            [REF-ZANG-SUN-TRIVEDI-1999].

        """
        return self._cumulative[-1]

    def phase_of_failure(self, component: str) -> tuple[float, ...]:
        """Return ``P(K = 1), ..., P(K = J), P(K = J + 1)`` for a component.

        ``K = j <= J`` means the component fails during phase ``j``;
        ``K = J + 1`` means it survives the mission.

        Args:
            component: Component name.

        Returns:
            The phase-of-failure distribution (sums to one).

        Raises:
            KeyError: If the component is unknown.

        Equation:
            [EQ-SYS-PMS-001].

        References:
            [REF-ZANG-SUN-TRIVEDI-1999].

        """
        model = self._models[component]
        durations = [phase.duration for phase in self._phases]
        result: list[float] = []
        if isinstance(model, tuple):
            alive = 1.0
            for phase_model, duration in zip(model, durations, strict=True):
                survive = require_probability(
                    phase_model.reliability(duration), name="reliability"
                )
                fail = require_probability(
                    phase_model.unreliability(duration), name="unreliability"
                )
                result.append(alive * fail)
                alive *= survive
            result.append(alive)
            return tuple(result)
        single = model
        end = 0.0
        previous_r, previous_f = 1.0, 0.0
        for duration in durations:
            end += duration
            r = require_probability(single.reliability(end), name="reliability")
            f = require_probability(single.unreliability(end), name="unreliability")
            # Subtract in whichever representation holds the smaller numbers.
            if previous_r < previous_f:
                result.append(max(0.0, previous_r - r))
            else:
                result.append(max(0.0, f - previous_f))
            previous_r, previous_f = r, f
        result.append(previous_r)
        return tuple(result)

    def _probability(
        self, root: int, laws: Mapping[str, Sequence[float]]
    ) -> tuple[float, float]:
        """Return ``(P(f = 1), P(f = 0))`` by phase-of-failure enumeration.

        References:
            [EQ-SYS-PMS-001], [REF-ZANG-SUN-TRIVEDI-1999].

        """
        bdd = self._bdd
        one: dict[int, float] = {0: 0.0, 1: 1.0}
        zero: dict[int, float] = {0: 1.0, 1: 0.0}
        for node in reversed(bdd.nodes(root)):
            name, _, _ = bdd.node(node)
            component = self._variable[name][0]
            hits_one: list[float] = []
            hits_zero: list[float] = []
            for outcome, mass in enumerate(laws[component], start=1):
                if mass == 0.0:
                    continue
                current = node
                while current >= 2:
                    label, low, high = bdd.node(current)
                    owner, phase = self._variable[label]
                    if owner != component:
                        break
                    current = high if phase < outcome else low
                hits_one.append(mass * one[current])
                hits_zero.append(mass * zero[current])
            one[node] = math.fsum(hits_one)
            zero[node] = math.fsum(hits_zero)
        return min(1.0, one[root]), min(1.0, zero[root])

    def evaluate(self) -> PhasedMissionResult:
        """Return the exact mission reliability and per-phase diagnostics.

        Returns:
            Mission reliability, cumulative and conditional phase
            reliabilities, and the probability that the mission fails in each
            phase.

        Equation:
            [EQ-SYS-PMS-001].

        References:
            [REF-ZANG-SUN-TRIVEDI-1999].

        """
        laws = {name: self.phase_of_failure(name) for name in self._components}
        cumulative: list[float] = []
        conditional: list[float | None] = []
        failures: list[float] = []
        previous_root = self._bdd.TRUE
        previous = 1.0
        final_zero = 0.0
        for root, phase_root in zip(self._cumulative, self._phase_roots, strict=True):
            value, final_zero = self._probability(root, laws)
            cumulative.append(value)
            conditional.append(value / previous if previous > 0.0 else None)
            fails_here = self._bdd.apply_and(
                previous_root, self._bdd.negate(phase_root)
            )
            failures.append(self._probability(fails_here, laws)[0])
            previous_root, previous = root, value
        return PhasedMissionResult(
            phase_names=tuple(phase.name for phase in self._phases),
            reliability=cumulative[-1],
            unreliability=final_zero,
            cumulative_reliability=tuple(cumulative),
            conditional_reliability=tuple(conditional),
            phase_failure_probability=tuple(failures),
        )

    def reliability(self) -> float:
        """Return the exact mission success probability.

        Returns:
            ``P(every phase succeeds)``.

        Equation:
            [EQ-SYS-PMS-001].

        References:
            [REF-ZANG-SUN-TRIVEDI-1999].

        """
        laws = {name: self.phase_of_failure(name) for name in self._components}
        return self._probability(self.mission_root, laws)[0]
