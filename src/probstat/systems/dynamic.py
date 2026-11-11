"""Dynamic fault trees quantified exactly by state-space generation.

A dynamic fault tree (Dugan, Bavuso and Boyd 1992; Stamatelatos et al. 2002,
Chapter 8) extends the static AND, OR and k-out-of-n (voting) gates with
gates whose output depends on the *order* of failures or on the *state of
use* of components:

* ``PAND`` (priority AND) fails when all inputs have failed in left-to-right
  order. Failures that occur at the same instant (possible only through a
  functional dependency) count as in order (inclusive PAND). Once an input
  fails before an input on its left, the gate can never fail.
* ``SPARE`` uses its primary unit and, when the unit in use fails, claims
  the first spare in its list that has not failed and is not in use by
  another spare gate (shared spare pools). It fails when the unit in use
  fails and no spare is available. A spare fails at ``alpha lambda`` while
  dormant and at ``lambda`` once in use, so ``alpha = 0`` is a cold spare,
  ``0 < alpha < 1`` a warm spare and ``alpha = 1`` a hot spare. Gates that
  need a spare at the same instant claim in declaration order.
* ``FDEP`` (functional dependency) forces its dependent basic events to
  fail at the instant its trigger (an event or a gate) fails. Its output is
  not used by other gates.
* ``SEQ`` (sequence enforcing) lets input ``j + 1`` (a basic event) fail
  only after input ``j`` has failed; before that its rate is zero. It fails
  when all inputs have failed.

Basic events are non-repairable with exponential lifetimes, so the system
is a continuous-time Markov chain whose state records which basic events
have failed, which unit each spare gate uses, and which PAND gates have
seen an out-of-order failure (EQ-SYS-DFT-001). The chain is generated
breadth-first from the all-working state; every state in which the top
event has occurred is merged into one absorbing failure state. The
unreliability ``F(t)`` is its transient probability (uniformization with a
rigorous error bound) and the MTTF is the mean time to absorption
(EQ-SYS-DFT-002). Elements that cannot influence the top event are pruned
before generation, and ``max_states`` stops a state explosion with a clear
error.

A tree with static gates only is also compiled to a BDD through the
fault-tree duality, and both paths give the same unreliability.
:func:`monte_carlo_unreliability` simulates the gate semantics with failure
times (not with the generated chain) and reports Clopper-Pearson intervals,
as an independent cross-check (EQ-SYS-DFT-003).

Closed forms that the tests use as oracles (EQ-SYS-DFT-004), for rates
``a``, ``b`` and ``lambda``:

* ``PAND(A, B)``: ``F(t) = a/(a+b) (1 - e^{-(a+b)t}) - e^{-bt} (1 - e^{-at})``;
* cold spare of two identical units: ``F(t) = 1 - e^{-lambda t}(1 + lambda t)``;
* hot spare of two identical units: ``F(t) = (1 - e^{-lambda t})^2``;
* warm spare with primary rate ``a``, spare rate ``b`` and dormancy
  ``alpha``: ``R(t) = e^{-at} + a e^{-bt} (1 - e^{-ct}) / c`` with
  ``c = a + alpha b - b`` (``R = e^{-at}(1 + a t)`` when ``c = 0``);
* ``FDEP`` with trigger ``T`` on the inputs of ``AND(A, B)``: the top event
  is ``T or (A and B)``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_non_negative_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.inference.confidence import clopper_pearson_interval
from probstat.processes.ctmc import BoundedValue, ContinuousTimeMarkovChain

from .structure import Component, Expression, KOutOfN, Parallel, Series, SystemStructure

STATIC_GATES = ("and", "or", "vote")
DYNAMIC_GATES = ("pand", "spare", "fdep", "seq")
FAILED_STATE = "failed"
_GATE_KINDS = STATIC_GATES + DYNAMIC_GATES
_MINIMUM_INPUTS = {
    "and": 1,
    "or": 1,
    "vote": 1,
    "pand": 2,
    "spare": 2,
    "fdep": 2,
    "seq": 2,
}


def _require_name(value: object, *, what: str) -> str:
    """Validate an element name.

    Args:
        value: Candidate name.
        what: Description used in error messages.

    Returns:
        The name.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If it is empty.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    if not value:
        raise ValueError(f"{what} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class DynamicBasicEvent:
    """A non-repairable basic event with an exponential lifetime.

    Args:
        name: Identifier, unique within the tree.
        rate: Failure rate ``lambda >= 0`` while active.
        dormancy: Dormancy factor ``alpha`` in ``[0, 1]``; the rate is
            ``alpha lambda`` while the event is an unused spare. It has no
            effect on events that are never spares.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the rate is negative or the dormancy is outside
            ``[0, 1]``.

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].

    """

    name: str
    rate: float
    dormancy: float = 1.0

    def __post_init__(self) -> None:
        """Validate the fields.

        References:
            Project validation policy. No equation is evaluated.

        """
        _require_name(self.name, what="basic event name")
        object.__setattr__(
            self, "rate", require_non_negative_finite_real(self.rate, name="rate")
        )
        object.__setattr__(
            self, "dormancy", require_probability(self.dormancy, name="dormancy")
        )


@dataclass(frozen=True, slots=True)
class DynamicGate:
    """A static or dynamic fault-tree gate.

    Args:
        name: Identifier, unique within the tree.
        kind: One of ``"and"``, ``"or"``, ``"vote"``, ``"pand"``,
            ``"spare"``, ``"fdep"`` and ``"seq"``.
        inputs: Input names. For ``"spare"`` the first input is the primary
            unit and the rest are the spares in order of use; for
            ``"fdep"`` the first input is the trigger and the rest are the
            dependent basic events; for ``"pand"`` and ``"seq"`` the order
            is the required failure order.
        k: Number of failed inputs that fails a ``"vote"`` gate.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the kind is unknown, an input repeats, the gate has
            too few inputs, or ``k`` is missing, misplaced or outside
            ``1..n``.

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].

    """

    name: str
    kind: str
    inputs: tuple[str, ...]
    k: int | None = None

    def __post_init__(self) -> None:
        """Validate the gate.

        References:
            Project validation policy. No equation is evaluated.

        """
        _require_name(self.name, what="gate name")
        if self.kind not in _GATE_KINDS:
            raise ValueError(f"unknown gate kind {self.kind!r}")
        if isinstance(self.inputs, str):
            raise TypeError("inputs must be a sequence of names, not a string")
        inputs = tuple(_require_name(item, what="gate input") for item in self.inputs)
        if len(set(inputs)) != len(inputs):
            raise ValueError(f"gate {self.name!r} repeats an input")
        if len(inputs) < _MINIMUM_INPUTS[self.kind]:
            raise ValueError(
                f"{self.kind} gate {self.name!r} needs at least "
                f"{_MINIMUM_INPUTS[self.kind]} inputs"
            )
        if self.name in inputs:
            raise ValueError(f"gate {self.name!r} lists itself as an input")
        if self.kind == "vote":
            if self.k is None:
                raise ValueError("a vote gate needs k")
            k = require_non_negative_integer(self.k, name="k")
            if not 1 <= k <= len(inputs):
                raise ValueError("k must satisfy 1 <= k <= number of inputs")
        elif self.k is not None:
            raise ValueError("k is only used by vote gates")
        object.__setattr__(self, "inputs", inputs)


def and_gate(name: str, *inputs: str) -> DynamicGate:
    """Return an AND gate: fails when every input has failed.

    Args:
        name: Gate name.
        *inputs: Input names.

    Returns:
        The gate.

    References:
        [REF-NASA-FTH-2002, Chapter 8]. No equation is evaluated.

    """
    return DynamicGate(name, "and", inputs)


def or_gate(name: str, *inputs: str) -> DynamicGate:
    """Return an OR gate: fails when any input has failed.

    Args:
        name: Gate name.
        *inputs: Input names.

    Returns:
        The gate.

    References:
        [REF-NASA-FTH-2002, Chapter 8]. No equation is evaluated.

    """
    return DynamicGate(name, "or", inputs)


def vote_gate(name: str, k: int, *inputs: str) -> DynamicGate:
    """Return a voting gate: fails when at least ``k`` inputs have failed.

    Args:
        name: Gate name.
        k: Failure threshold, ``1 <= k <= len(inputs)``.
        *inputs: Input names.

    Returns:
        The gate.

    References:
        [REF-NASA-FTH-2002, Chapter 8]. No equation is evaluated.

    """
    return DynamicGate(name, "vote", inputs, k)


def pand_gate(name: str, *inputs: str) -> DynamicGate:
    """Return a priority-AND gate: all inputs fail, left to right.

    Args:
        name: Gate name.
        *inputs: Input names in the required failure order.

    Returns:
        The gate.

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].
        No equation is evaluated.

    """
    return DynamicGate(name, "pand", inputs)


def spare_gate(name: str, primary: str, *spares: str) -> DynamicGate:
    """Return a spare gate with a primary unit and spares in order of use.

    Args:
        name: Gate name.
        primary: Primary basic event.
        *spares: Spare basic events (they may be shared with other spare
            gates).

    Returns:
        The gate.

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].
        No equation is evaluated.

    """
    return DynamicGate(name, "spare", (primary, *spares))


def fdep_gate(name: str, trigger: str, *dependents: str) -> DynamicGate:
    """Return a functional dependency: the trigger fails the dependents.

    Args:
        name: Gate name.
        trigger: Triggering event or gate.
        *dependents: Basic events forced to fail with the trigger.

    Returns:
        The gate.

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].
        No equation is evaluated.

    """
    return DynamicGate(name, "fdep", (trigger, *dependents))


def seq_gate(name: str, *inputs: str) -> DynamicGate:
    """Return a sequence-enforcing gate.

    Args:
        name: Gate name.
        *inputs: The first input is any event or gate; the others are basic
            events that can fail only after their predecessor has failed.

    Returns:
        The gate.

    References:
        [REF-NASA-FTH-2002, Chapter 8]. No equation is evaluated.

    """
    return DynamicGate(name, "seq", inputs)


@dataclass(frozen=True, slots=True)
class DFTStateSpace:
    """The generated Markov chain of a dynamic fault tree.

    State labels other than :data:`FAILED_STATE` are triples
    ``(failed events, unit in use per spare gate, PAND gates whose order
    was violated)``: a frozenset of names, a tuple with one name or
    ``None`` (gate failed) per spare gate in declaration order, and a
    frozenset of names.

    Args:
        chain: Continuous-time Markov chain over the reachable states.
        initial: Label of the all-working state.
        failure_state: :data:`FAILED_STATE` if the top event can occur,
            otherwise ``None``.
        transitions: Number of distinct transitions.
        absorbing_up_states: Working states without outgoing transitions
            (the top event can no longer occur from them).

    References:
        [EQ-SYS-DFT-001], [REF-DUGAN-BAVUSO-BOYD-1992].

    """

    chain: ContinuousTimeMarkovChain
    initial: Hashable
    failure_state: str | None
    transitions: int
    absorbing_up_states: tuple[Hashable, ...]

    @property
    def size(self) -> int:
        """Return the number of states of the generated chain.

        References:
            [EQ-SYS-DFT-001].

        """
        return self.chain.size


@dataclass(frozen=True, slots=True)
class DFTMonteCarloEstimate:
    """Monte Carlo estimate of the unreliability at one time.

    Args:
        time: Mission time ``t``.
        failures: Number of histories with the top event by ``t``.
        samples: Number of simulated histories.
        estimate: ``failures / samples``.
        lower: Lower Clopper-Pearson confidence limit.
        upper: Upper Clopper-Pearson confidence limit.
        confidence: Two-sided confidence level.

    References:
        [EQ-SYS-DFT-003], [REF-CLOPPER-PEARSON-1934].

    """

    time: float
    failures: int
    samples: int
    estimate: float
    lower: float
    upper: float
    confidence: float

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies in the confidence interval.

        Args:
            value: Candidate unreliability.

        Returns:
            ``lower <= value <= upper``.

        References:
            [REF-CLOPPER-PEARSON-1934]. No equation is evaluated.

        """
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class _Compiled:
    """Indexed form of a validated tree used by both quantification paths.

    References:
        [EQ-SYS-DFT-001], [REF-DUGAN-BAVUSO-BOYD-1992].

    """

    events: tuple[DynamicBasicEvent, ...]
    event_index: dict[str, int]
    element_index: dict[str, int]
    gate_order: tuple[DynamicGate, ...]
    gate_inputs: tuple[tuple[int, ...], ...]
    spare_gates: tuple[DynamicGate, ...]
    spare_units: tuple[tuple[int, ...], ...]
    spare_position: dict[str, int]
    pand_gates: tuple[DynamicGate, ...]
    pand_position: dict[str, int]
    fdeps: tuple[tuple[int, int], ...]
    pool: frozenset[int]
    predecessor: dict[int, int]
    top: int


class DynamicFaultTree:
    """A dynamic fault tree with exact Markov quantification.

    Args:
        events: Basic events.
        gates: Static and dynamic gates (FDEP gates included).
        top: Name of the top event (a gate or basic event, not an FDEP).
        max_states: Largest number of chain states that may be generated.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If names repeat, an input is unknown, the gates form a
            cycle, an FDEP output is used or is the top event, a spare or
            SEQ input violates its rules, or ``max_states`` is not positive.

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].

    """

    def __init__(
        self,
        events: Iterable[DynamicBasicEvent],
        gates: Iterable[DynamicGate],
        top: str,
        *,
        max_states: int = 10_000,
    ) -> None:
        """Validate the tree and prepare the indexed form.

        References:
            [REF-DUGAN-BAVUSO-BOYD-1992]. No equation is evaluated.

        """
        self._events = tuple(events)
        self._gates = tuple(gates)
        for event in self._events:
            if not isinstance(event, DynamicBasicEvent):
                raise TypeError("events must be DynamicBasicEvent instances")
        for gate in self._gates:
            if not isinstance(gate, DynamicGate):
                raise TypeError("gates must be DynamicGate instances")
        self._top = _require_name(top, what="top")
        limit = require_non_negative_integer(max_states, name="max_states")
        if limit < 1:
            raise ValueError("max_states must be positive")
        self._max_states = limit
        self._compiled = _compile(self._events, self._gates, self._top)
        self._space: DFTStateSpace | None = None
        self._structure: SystemStructure | None = None

    @property
    def top(self) -> str:
        """Return the name of the top event.

        References:
            [REF-NASA-FTH-2002, Chapter 8].

        """
        return self._top

    @property
    def events(self) -> tuple[DynamicBasicEvent, ...]:
        """Return the basic events that can influence the top event.

        References:
            [REF-NASA-FTH-2002, Chapter 8].

        """
        return self._compiled.events

    @property
    def is_static(self) -> bool:
        """Return whether every relevant gate is AND, OR or voting.

        References:
            [REF-NASA-FTH-2002, Chapter 8].

        """
        compiled = self._compiled
        if compiled.spare_gates or compiled.pand_gates or compiled.fdeps:
            return False
        return all(gate.kind in STATIC_GATES for gate in compiled.gate_order)

    # ------------------------------------------------------------------
    # static path
    # ------------------------------------------------------------------
    def static_structure(self) -> SystemStructure:
        """Return the success structure of a static tree on a BDD.

        OR gates become series structures, AND gates parallel structures and
        "at least ``k`` of ``n`` fail" becomes ``(n - k + 1)``-out-of-``n``
        success (EQ-SYS-STRUCT-004).

        Returns:
            Compiled structure whose components are the basic events.

        Raises:
            ValueError: If the tree has a dynamic gate.

        Equation:
            [EQ-SYS-STRUCT-004].

        References:
            [REF-VESELY-ETAL-1981], [REF-BARLOW-PROSCHAN-1975].

        """
        if not self.is_static:
            raise ValueError("the tree has dynamic gates; use the Markov path")
        if self._structure is None:
            gates = {gate.name: gate for gate in self._gates}
            memo: dict[str, Expression] = {}
            for gate in self._compiled.gate_order:
                items = tuple(
                    memo[name] if name in memo else Component(name)
                    for name in gate.inputs
                )
                if gate.kind == "or":
                    memo[gate.name] = Series(items)
                elif gate.kind == "and":
                    memo[gate.name] = Parallel(items)
                else:
                    k = gates[gate.name].k
                    assert k is not None
                    memo[gate.name] = KOutOfN(len(items) - k + 1, items)
            top = self._top
            expression = memo[top] if top in memo else Component(top)
            self._structure = SystemStructure(expression)
        return self._structure

    def _static_unreliability(self, t: float) -> BoundedValue:
        """Return ``F(t)`` of a static tree from its BDD.

        Each basic event works at ``t`` with ``e^{-lambda t}`` and has failed
        with ``-expm1(-lambda t)``, both passed to the BDD so that small
        unreliabilities keep their relative accuracy. Only rounding error
        remains, so the bound is zero.

        References:
            [EQ-SYS-BDD-001], [EQ-SYS-COMP-001], [REF-RAUZY-1993].

        """
        structure = self.static_structure()
        rates = {event.name: event.rate for event in self._compiled.events}
        works = {name: math.exp(-rates[name] * t) for name in structure.components}
        failed = {name: -math.expm1(-rates[name] * t) for name in structure.components}
        return BoundedValue(structure.unreliability(works, unreliabilities=failed), 0.0)

    # ------------------------------------------------------------------
    # Markov path
    # ------------------------------------------------------------------
    def state_space(self) -> DFTStateSpace:
        """Generate (once) and return the Markov chain of the tree.

        Returns:
            The chain, its initial state and size information.

        Raises:
            ValueError: If more than ``max_states`` states are reachable.

        Equation:
            [EQ-SYS-DFT-001].

        References:
            [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].

        """
        if self._space is None:
            self._space = _generate(self._compiled, self._max_states)
        return self._space

    def unreliability_curve(
        self,
        times: Iterable[float],
        *,
        method: str = "auto",
        tolerance: float = 1e-13,
    ) -> tuple[BoundedValue, ...]:
        """Return ``F(t)``, the probability that the top event occurred by ``t``.

        Args:
            times: Non-negative mission times.
            method: ``"markov"`` (state space), ``"bdd"`` (static trees
                only) or ``"auto"`` (BDD for a static tree, else Markov).
            tolerance: L1 truncation tolerance of uniformization.

        Returns:
            One value with an absolute error bound per time.

        Raises:
            ValueError: If the method is unknown, ``"bdd"`` is requested for a
                dynamic tree, or a time is invalid.

        Equation:
            [EQ-SYS-DFT-001], [EQ-SYS-BDD-001].

        References:
            [REF-DUGAN-BAVUSO-BOYD-1992], [REF-REIBMAN-TRIVEDI-1988].

        """
        checked = tuple(require_non_negative_finite_real(t, name="t") for t in times)
        if method not in ("auto", "markov", "bdd"):
            raise ValueError("method must be 'auto', 'markov' or 'bdd'")
        if method == "bdd" or (method == "auto" and self.is_static):
            return tuple(self._static_unreliability(t) for t in checked)
        space = self.state_space()
        if space.failure_state is None:
            return tuple(BoundedValue(0.0, 0.0) for _ in checked)
        solutions = space.chain.transient_many(space.initial, checked, tolerance)
        return tuple(solution.total((space.failure_state,)) for solution in solutions)

    def unreliability(
        self, t: float, *, method: str = "auto", tolerance: float = 1e-13
    ) -> BoundedValue:
        """Return ``F(t)`` at one time (see :meth:`unreliability_curve`).

        Args:
            t: Non-negative mission time.
            method: ``"auto"``, ``"markov"`` or ``"bdd"``.
            tolerance: L1 truncation tolerance of uniformization.

        Returns:
            Unreliability with an absolute error bound.

        Equation:
            [EQ-SYS-DFT-001].

        References:
            [REF-DUGAN-BAVUSO-BOYD-1992].

        """
        return self.unreliability_curve((t,), method=method, tolerance=tolerance)[0]

    def reliability(
        self, t: float, *, method: str = "auto", tolerance: float = 1e-13
    ) -> BoundedValue:
        """Return ``R(t) = 1 - F(t)``, summed over the working states.

        Args:
            t: Non-negative mission time.
            method: ``"auto"``, ``"markov"`` or ``"bdd"``.
            tolerance: L1 truncation tolerance of uniformization.

        Returns:
            Reliability with an absolute error bound.

        Raises:
            ValueError: As :meth:`unreliability_curve`.

        Equation:
            [EQ-SYS-DFT-001].

        References:
            [REF-DUGAN-BAVUSO-BOYD-1992].

        """
        checked = require_non_negative_finite_real(t, name="t")
        if method not in ("auto", "markov", "bdd"):
            raise ValueError("method must be 'auto', 'markov' or 'bdd'")
        if method == "bdd" or (method == "auto" and self.is_static):
            structure = self.static_structure()
            rates = {event.name: event.rate for event in self._compiled.events}
            works = {n: math.exp(-rates[n] * checked) for n in structure.components}
            failed = {n: -math.expm1(-rates[n] * checked) for n in structure.components}
            return BoundedValue(
                structure.reliability(works, unreliabilities=failed), 0.0
            )
        space = self.state_space()
        solution = space.chain.transient(space.initial, checked, tolerance)
        up = tuple(state for state in space.chain.states if state != FAILED_STATE)
        return solution.total(up)

    def mttf(self) -> float:
        """Return the mean time to the top event from the generated chain.

        Returns:
            ``E[T_top]`` (EQ-SYS-DFT-002); ``inf`` when the top event can be
            avoided forever with positive probability.

        Raises:
            ValueError: If the state space exceeds ``max_states``.

        Equation:
            [EQ-SYS-DFT-002].

        References:
            [REF-NORRIS-1997, Section 3.3], [REF-DUGAN-BAVUSO-BOYD-1992].

        """
        space = self.state_space()
        if space.failure_state is None or space.absorbing_up_states:
            return math.inf
        return space.chain.absorption(space.initial).initial_mean_time


# ----------------------------------------------------------------------
# validation and indexing
# ----------------------------------------------------------------------
def _compile(
    events: Sequence[DynamicBasicEvent], gates: Sequence[DynamicGate], top: str
) -> _Compiled:
    """Validate a tree, prune irrelevant elements and index the rest.

    Raises:
        ValueError: If the tree is invalid (see :class:`DynamicFaultTree`).

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-NASA-FTH-2002, Chapter 8].
        No equation is evaluated.

    """
    by_event = {event.name: event for event in events}
    by_gate = {gate.name: gate for gate in gates}
    if len(by_event) != len(events) or len(by_gate) != len(gates):
        raise ValueError("element names must be unique")
    clash = set(by_event) & set(by_gate)
    if clash:
        raise ValueError(f"names used for both an event and a gate: {sorted(clash)}")
    known = set(by_event) | set(by_gate)
    if top not in known:
        raise ValueError(f"unknown top event {top!r}")
    if top in by_gate and by_gate[top].kind == "fdep":
        raise ValueError("an FDEP gate cannot be the top event")
    primaries: dict[str, str] = {}
    spares_of: dict[str, list[str]] = {}
    followers: dict[str, str] = {}
    for gate in gates:
        for name in gate.inputs:
            if name not in known:
                raise ValueError(f"gate {gate.name!r} has unknown input {name!r}")
            if name in by_gate and by_gate[name].kind == "fdep":
                raise ValueError(f"FDEP gate {name!r} cannot be a gate input")
        if gate.kind == "spare":
            for name in gate.inputs:
                if name not in by_event:
                    raise ValueError(
                        f"spare gate {gate.name!r}: input {name!r} must be a "
                        "basic event"
                    )
            primary = gate.inputs[0]
            if primary in primaries:
                raise ValueError(f"basic event {primary!r} is the primary of two gates")
            primaries[primary] = gate.name
            for name in gate.inputs[1:]:
                spares_of.setdefault(name, []).append(gate.name)
        elif gate.kind == "fdep":
            for name in gate.inputs[1:]:
                if name not in by_event:
                    raise ValueError(
                        f"FDEP gate {gate.name!r}: dependent {name!r} must be a "
                        "basic event"
                    )
        elif gate.kind == "seq":
            for previous, name in zip(gate.inputs, gate.inputs[1:], strict=False):
                if name not in by_event:
                    raise ValueError(
                        f"SEQ gate {gate.name!r}: input {name!r} must be a basic event"
                    )
                if name in followers:
                    raise ValueError(f"basic event {name!r} follows in two SEQ gates")
                followers[name] = previous
    for name in primaries:
        if name in spares_of:
            raise ValueError(f"basic event {name!r} is both a primary and a spare")
    for name in followers:
        if name in primaries or name in spares_of:
            raise ValueError(f"basic event {name!r} is in a SEQ gate and a spare gate")
    # Prune elements that cannot influence the top event.
    relevant: set[str] = set()
    pending = [top]
    while True:
        while pending:
            name = pending.pop()
            if name in relevant:
                continue
            relevant.add(name)
            if name in by_gate:
                gate = by_gate[name]
                pending.extend(gate.inputs if gate.kind != "fdep" else gate.inputs[:1])
        added = False
        for gate in gates:
            if gate.name in relevant:
                continue
            if gate.kind == "fdep":
                needed = any(name in relevant for name in gate.inputs[1:])
            elif gate.kind == "spare":
                needed = any(name in relevant for name in gate.inputs)
            elif gate.kind == "seq":
                needed = any(name in relevant for name in gate.inputs[1:])
            else:
                needed = False
            if needed:
                pending.append(gate.name)
                added = True
        if not added and not pending:
            break
    # Topological order of the non-FDEP gates (inputs first); every cycle
    # is rejected, then the order is restricted to the relevant gates.
    full_order: list[DynamicGate] = []
    state: dict[str, int] = {}
    for root in by_gate:
        if by_gate[root].kind == "fdep" or state.get(root) == 2:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        state[root] = 1
        while stack:
            name, index = stack.pop()
            gate = by_gate[name]
            if index < len(gate.inputs):
                stack.append((name, index + 1))
                child = gate.inputs[index]
                if child in by_gate:
                    mark = state.get(child, 0)
                    if mark == 1:
                        raise ValueError(f"gates form a cycle through {child!r}")
                    if mark == 0:
                        state[child] = 1
                        stack.append((child, 0))
            else:
                state[name] = 2
                full_order.append(gate)
    order = [gate for gate in full_order if gate.name in relevant]
    kept_events = tuple(event for event in events if event.name in relevant)
    event_index = {event.name: i for i, event in enumerate(kept_events)}
    element_index = dict(event_index)
    for gate in order:
        element_index[gate.name] = len(element_index)
    gate_inputs = tuple(
        tuple(element_index[name] for name in gate.inputs) for gate in order
    )
    declared = [gate for gate in gates if gate.name in relevant]
    spare_gates = tuple(gate for gate in declared if gate.kind == "spare")
    pand_gates = tuple(gate for gate in declared if gate.kind == "pand")
    fdeps = tuple(
        (
            element_index[gate.inputs[0]],
            sum(1 << event_index[name] for name in gate.inputs[1:]),
        )
        for gate in declared
        if gate.kind == "fdep"
    )
    pool = frozenset(
        event_index[name] for gate in spare_gates for name in gate.inputs[1:]
    )
    predecessor = {
        event_index[name]: element_index[previous]
        for name, previous in followers.items()
        if name in event_index
    }
    return _Compiled(
        kept_events,
        event_index,
        element_index,
        tuple(order),
        gate_inputs,
        spare_gates,
        tuple(tuple(event_index[n] for n in gate.inputs) for gate in spare_gates),
        {gate.name: i for i, gate in enumerate(spare_gates)},
        pand_gates,
        {gate.name: i for i, gate in enumerate(pand_gates)},
        fdeps,
        pool,
        predecessor,
        element_index[top],
    )


# ----------------------------------------------------------------------
# state-space generation
# ----------------------------------------------------------------------
_Key = tuple[int, tuple[int, ...], int]


def _status(
    compiled: _Compiled, failed: int, use: Sequence[int], violated: int
) -> list[bool]:
    """Return the failure status of every element in a state.

    References:
        [EQ-SYS-DFT-001], [REF-DUGAN-BAVUSO-BOYD-1992].

    """
    count = len(compiled.events)
    status = [bool(failed >> i & 1) for i in range(count)]
    for gate, inputs in zip(compiled.gate_order, compiled.gate_inputs, strict=True):
        kind = gate.kind
        if kind == "or":
            value = any(status[i] for i in inputs)
        elif kind == "and" or kind == "seq":
            value = all(status[i] for i in inputs)
        elif kind == "vote":
            assert gate.k is not None
            value = sum(status[i] for i in inputs) >= gate.k
        elif kind == "pand":
            position = compiled.pand_position[gate.name]
            value = not violated >> position & 1 and all(status[i] for i in inputs)
        else:
            value = use[compiled.spare_position[gate.name]] < 0
        status.append(value)
    return status


def _settle(
    compiled: _Compiled, failed: int, use: Sequence[int], violated: int
) -> tuple[_Key, list[bool]]:
    """Apply spare claims and functional dependencies until nothing changes.

    All changes happen at one instant: a spare gate whose unit has failed
    claims its first available spare (gates in declaration order), and
    every FDEP whose trigger has failed fails its dependents. Afterwards a
    PAND gate whose failed inputs are not a prefix of its input list is
    marked as violated.

    Returns:
        The settled state key and the element statuses.

    References:
        [EQ-SYS-DFT-001], [REF-DUGAN-BAVUSO-BOYD-1992].

    """
    current = list(use)
    while True:
        claimed = False
        in_use = {
            units[u]
            for units, u in zip(compiled.spare_units, current, strict=True)
            if u >= 0
        }
        for g, units in enumerate(compiled.spare_units):
            u = current[g]
            if u < 0 or not failed >> units[u] & 1:
                continue
            claimed = True
            current[g] = -1
            for j in range(1, len(units)):
                if not failed >> units[j] & 1 and units[j] not in in_use:
                    current[g] = j
                    in_use.add(units[j])
                    break
        status = _status(compiled, failed, current, violated)
        forced = failed
        for trigger, dependents in compiled.fdeps:
            if status[trigger]:
                forced |= dependents
        if forced == failed and not claimed:
            break
        failed = forced
    for position, gate in enumerate(compiled.pand_gates):
        if violated >> position & 1:
            continue
        flags = [status[compiled.element_index[name]] for name in gate.inputs]
        first_working = flags.index(False) if False in flags else len(flags)
        if any(flags[first_working:]):
            violated |= 1 << position
    return (failed, tuple(current), violated), status


def _event_rate(
    compiled: _Compiled, index: int, key: _Key, status: Sequence[bool]
) -> float:
    """Return the current failure rate of a working basic event.

    References:
        [EQ-SYS-DFT-001], [REF-NASA-FTH-2002, Chapter 8].

    """
    event = compiled.events[index]
    previous = compiled.predecessor.get(index)
    if previous is not None:
        return event.rate if status[previous] else 0.0
    if index in compiled.pool:
        in_use = any(
            u >= 0 and units[u] == index
            for units, u in zip(compiled.spare_units, key[1], strict=True)
        )
        return event.rate if in_use else event.dormancy * event.rate
    return event.rate


def _label(compiled: _Compiled, key: _Key) -> Hashable:
    """Return the readable chain label of a working state.

    References:
        [EQ-SYS-DFT-001]. No further equation is evaluated.

    """
    failed, use, violated = key
    names = frozenset(
        event.name for i, event in enumerate(compiled.events) if failed >> i & 1
    )
    units = tuple(
        compiled.events[unit_list[u]].name if u >= 0 else None
        for unit_list, u in zip(compiled.spare_units, use, strict=True)
    )
    pands = frozenset(
        gate.name
        for position, gate in enumerate(compiled.pand_gates)
        if violated >> position & 1
    )
    return (names, units, pands)


def _generate(compiled: _Compiled, max_states: int) -> DFTStateSpace:
    """Generate the reachable chain breadth-first from the working state.

    Raises:
        ValueError: If more than ``max_states`` states are reachable.

    References:
        [EQ-SYS-DFT-001], [REF-DUGAN-BAVUSO-BOYD-1992].

    """
    initial, initial_status = _settle(compiled, 0, (0,) * len(compiled.spare_units), 0)
    if initial_status[compiled.top]:
        raise ValueError("the top event has occurred before any failure")
    labels: dict[_Key, Hashable] = {initial: _label(compiled, initial)}
    statuses: dict[_Key, list[bool]] = {initial: initial_status}
    queue: deque[_Key] = deque([initial])
    rates: dict[tuple[Hashable, Hashable], float] = {}
    reaches_failure = False
    absorbing: list[Hashable] = []

    def admit() -> None:
        """Raise when the state budget is exhausted.

        References:
            Project validation policy. No equation is evaluated.

        """
        if len(labels) + (1 if reaches_failure else 0) > max_states:
            raise ValueError(
                f"the dynamic fault tree has more than max_states={max_states} "
                "Markov states; raise max_states or simplify the tree"
            )

    while queue:
        key = queue.popleft()
        status = statuses.pop(key)
        source = labels[key]
        outgoing = False
        for index in range(len(compiled.events)):
            if key[0] >> index & 1:
                continue
            rate = _event_rate(compiled, index, key, status)
            if rate <= 0.0:
                continue
            successor, successor_status = _settle(
                compiled, key[0] | 1 << index, key[1], key[2]
            )
            target: Hashable
            if successor_status[compiled.top]:
                target = FAILED_STATE
                if not reaches_failure:
                    reaches_failure = True
                    admit()
            else:
                if successor not in labels:
                    labels[successor] = _label(compiled, successor)
                    statuses[successor] = successor_status
                    queue.append(successor)
                    admit()
                target = labels[successor]
            rates[(source, target)] = rates.get((source, target), 0.0) + rate
            outgoing = True
        if not outgoing:
            absorbing.append(source)
    states: list[Hashable] = list(labels.values())
    if reaches_failure:
        states.append(FAILED_STATE)
    chain = ContinuousTimeMarkovChain(tuple(states), rates)
    return DFTStateSpace(
        chain,
        labels[initial],
        FAILED_STATE if reaches_failure else None,
        len(rates),
        tuple(absorbing),
    )


# ----------------------------------------------------------------------
# Monte Carlo cross-check
# ----------------------------------------------------------------------
def _gate_times(
    compiled: _Compiled,
    times: Sequence[float],
    spare_failure: Sequence[float],
) -> list[float]:
    """Return every element's failure time from the basic-event times.

    Static gates use the order statistics of their input times (AND: the
    largest, OR: the smallest, vote: the ``k``-th smallest); a PAND gate
    fails at its last input time when the input times are non-decreasing;
    a SEQ gate at its last input time; a spare gate at the recorded time.
    Unfailed elements have time ``inf``.

    References:
        [EQ-SYS-DFT-003], [REF-DUGAN-BAVUSO-BOYD-1992].

    """
    values = list(times)
    for gate, inputs in zip(compiled.gate_order, compiled.gate_inputs, strict=True):
        kind = gate.kind
        child = [values[i] for i in inputs]
        if kind == "or":
            value = min(child)
        elif kind == "and" or kind == "seq":
            value = max(child)
        elif kind == "vote":
            assert gate.k is not None
            value = sorted(child)[gate.k - 1]
        elif kind == "pand":
            ordered = all(a <= b for a, b in zip(child, child[1:], strict=False))
            value = child[-1] if ordered else math.inf
        else:
            value = spare_failure[compiled.spare_position[gate.name]]
        values.append(value)
    return values


def _simulate_once(
    compiled: _Compiled, horizon: float, generator: _Exponentials
) -> float:
    """Simulate one history and return the top-event time (or ``inf``).

    Every working basic event carries a pending failure time. A dormant
    spare draws ``Exp(alpha lambda)``; when it is put into use at time
    ``s`` the pending time becomes ``s + Exp(lambda)`` (memorylessness). A
    SEQ follower draws ``s + Exp(lambda)`` when its predecessor fails at
    ``s``. At each failure the spare claims and FDEP cascades are resolved
    at that instant, and the gate times are recomputed from the failure
    times.

    References:
        [EQ-SYS-DFT-003], [REF-DUGAN-BAVUSO-BOYD-1992].

    """
    events = compiled.events
    count = len(events)
    fail_time = [math.inf] * count
    pending = [math.inf] * count
    active = [True] * count
    for i, event in enumerate(events):
        if i in compiled.predecessor:
            active[i] = False
        elif i in compiled.pool:
            active[i] = False
            pending[i] = generator.exponential(event.dormancy * event.rate)
            continue
        if active[i]:
            pending[i] = generator.exponential(event.rate)
    use = [0] * len(compiled.spare_units)
    spare_failure = [math.inf] * len(compiled.spare_units)
    while True:
        index = min(
            (i for i in range(count) if fail_time[i] == math.inf),
            key=lambda i: pending[i],
            default=None,
        )
        if index is None or pending[index] > horizon:
            return math.inf
        now = pending[index]
        fail_time[index] = now
        while True:
            changed = False
            in_use = {
                units[u]
                for units, u in zip(compiled.spare_units, use, strict=True)
                if u >= 0
            }
            for g, units in enumerate(compiled.spare_units):
                u = use[g]
                if u < 0 or fail_time[units[u]] > now:
                    continue
                changed = True
                use[g] = -1
                for j in range(1, len(units)):
                    unit = units[j]
                    if fail_time[unit] == math.inf and unit not in in_use:
                        use[g] = j
                        in_use.add(unit)
                        active[unit] = True
                        pending[unit] = now + generator.exponential(events[unit].rate)
                        break
                if use[g] < 0:
                    spare_failure[g] = now
            values = _gate_times(compiled, fail_time, spare_failure)
            for trigger, dependents in compiled.fdeps:
                if values[trigger] <= now:
                    for i in range(count):
                        if dependents >> i & 1 and fail_time[i] == math.inf:
                            fail_time[i] = now
                            changed = True
            for i, previous in compiled.predecessor.items():
                waiting = not active[i] and fail_time[i] == math.inf
                if waiting and values[previous] <= now:
                    active[i] = True
                    pending[i] = now + generator.exponential(events[i].rate)
            if not changed:
                break
        if values[compiled.top] <= now:
            return now


class _Exponentials:
    """Seeded exponential variates for the simulation.

    References:
        [REF-DEVROYE-1986, Chapter II]. No further equation is evaluated.

    """

    def __init__(self, stream: RandomStream) -> None:
        """Bind a reproducible generator.

        References:
            [REF-DEVROYE-1986, Chapter II]. No equation is evaluated.

        """
        self._generator = stream.generator()

    def exponential(self, rate: float) -> float:
        """Return an ``Exp(rate)`` variate, ``inf`` for a zero rate.

        References:
            [REF-DEVROYE-1986, Chapter II]. No equation is evaluated.

        """
        if rate <= 0.0:
            return math.inf
        return self._generator.expovariate(rate)


def monte_carlo_unreliability(
    tree: DynamicFaultTree,
    times: Iterable[float],
    *,
    samples: int,
    seed: int | RandomStream | None = None,
    confidence: float = 0.99,
) -> tuple[DFTMonteCarloEstimate, ...]:
    """Estimate ``F(t)`` by simulating the gate semantics with failure times.

    The simulation does not use the generated Markov chain: it draws
    failure times, resolves spare claims, functional dependencies and
    sequence enforcement at each failure instant, and evaluates the gates
    from the failure times (EQ-SYS-DFT-003). The number of histories whose
    top event occurs by ``t`` is binomial, so each estimate carries a
    Clopper-Pearson interval. Use it to cross-check the exact methods.

    Args:
        tree: Dynamic fault tree.
        times: Non-negative mission times.
        samples: Number of simulated histories (positive).
        seed: Integer seed or stream; ``None`` is not reproducible.
        confidence: Two-sided confidence level in ``(0, 1)``.

    Returns:
        One estimate per time, in the given order.

    Raises:
        TypeError: If ``tree`` is not a :class:`DynamicFaultTree`.
        ValueError: If ``samples`` is not positive or a time is invalid.

    Equation:
        [EQ-SYS-DFT-003], [EQ-INF-CI-CP-001].

    References:
        [REF-DUGAN-BAVUSO-BOYD-1992], [REF-CLOPPER-PEARSON-1934].

    """
    if not isinstance(tree, DynamicFaultTree):
        raise TypeError("tree must be a DynamicFaultTree")
    checked = tuple(require_non_negative_finite_real(t, name="t") for t in times)
    if not checked:
        raise ValueError("at least one time is required")
    count = require_non_negative_integer(samples, name="samples")
    if count < 1:
        raise ValueError("samples must be positive")
    generator = _Exponentials(resolve_stream(seed).child("dynamic-fault-tree"))
    horizon = max(checked)
    compiled = tree._compiled
    top_times = [_simulate_once(compiled, horizon, generator) for _ in range(count)]
    results: list[DFTMonteCarloEstimate] = []
    for t in checked:
        failures = sum(1 for value in top_times if value <= t)
        interval = clopper_pearson_interval(failures, count, confidence=confidence)
        results.append(
            DFTMonteCarloEstimate(
                t,
                failures,
                count,
                failures / count,
                interval.lower,
                interval.upper,
                interval.confidence,
            )
        )
    return tuple(results)
