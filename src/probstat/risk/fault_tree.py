"""Static fault trees: minimal cut sets, top-event probability, importance.

A fault tree is built from :class:`BasicEvent` leaves and AND / OR /
k-out-of-n :class:`Gate` nodes. Basic events are assumed statistically
independent; a basic event *name* may appear in several branches (a repeated
or shared event), and every occurrence denotes the same random event.

Quantification:

* ``minimal_cut_sets`` uses the top-down MOCUS expansion followed by Boolean
  absorption.
* ``top_event_probability`` is exact under independence. The default
  ``"shannon"`` method conditions on every repeated event (Shannon
  decomposition) and evaluates the remaining tree bottom-up, which is exact
  because non-repeated events occur once and therefore make sibling subtrees
  independent. ``"inclusion_exclusion"`` sums over unions of minimal cut sets
  and serves as an independent cross-check for small trees.
* ``rare_event_approximation`` and ``min_cut_upper_bound`` are the classical
  cut-set approximations.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.foundations import require_probability

GATE_KINDS = ("and", "or", "k_of_n")
_MAX_MOCUS_ROWS = 200_000


def _require_name(value: object, *, name: str) -> str:
    """Validate a non-empty identifier.

    Args:
        value: Candidate name.
        name: Field name for error messages.

    Returns:
        The name.

    Raises:
        TypeError: If the name is not text.
        ValueError: If the name is empty.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class BasicEvent:
    """An independent basic (primary) failure event.

    Args:
        name: Identifier; equal names denote the same event.
        probability: Failure probability in ``[0, 1]``.
        description: Optional meaning.

    References:
        [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

    """

    name: str
    probability: float
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the name and probability.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If the name is empty or the probability is invalid.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        _require_name(self.name, name="name")
        object.__setattr__(
            self,
            "probability",
            require_probability(self.probability, name=f"P({self.name})"),
        )


@dataclass(frozen=True, slots=True)
class Gate:
    """A logic gate combining child events.

    Args:
        name: Gate identifier.
        kind: ``"and"``, ``"or"``, or ``"k_of_n"``.
        inputs: Non-empty child basic events or gates.
        k: Required number of failed inputs for ``"k_of_n"`` gates.
        description: Optional meaning.

    References:
        [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

    """

    name: str
    kind: str
    inputs: tuple[BasicEvent | Gate, ...]
    k: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the gate type, inputs, and vote threshold.

        Raises:
            TypeError: If an input is not a basic event or gate.
            ValueError: If the kind is unknown, inputs are empty, or ``k`` is
                inconsistent with the kind.

        References:
            [REF-VESELY-ETAL-1981]. No equation is evaluated.

        """
        _require_name(self.name, name="name")
        if self.kind not in GATE_KINDS:
            raise ValueError(f"kind must be one of {GATE_KINDS}")
        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError("a gate needs at least one input")
        for index, child in enumerate(inputs):
            if not isinstance(child, BasicEvent | Gate):
                raise TypeError(f"inputs[{index}] must be a BasicEvent or Gate")
        object.__setattr__(self, "inputs", inputs)
        if self.kind == "k_of_n":
            if isinstance(self.k, bool) or not isinstance(self.k, int):
                raise ValueError("k_of_n gates need an integer k")
            if not 1 <= self.k <= len(inputs):
                raise ValueError("k must satisfy 1 <= k <= number of inputs")
        elif self.k is not None:
            raise ValueError("k is only valid for k_of_n gates")


def and_gate(name: str, *inputs: BasicEvent | Gate) -> Gate:
    """Return a gate that fails when every input fails.

    Args:
        name: Gate identifier.
        *inputs: Child events.

    Returns:
        AND gate.

    Equation:
        [EQ-RISK-FT-GATE-001].

    References:
        [REF-VESELY-ETAL-1981].

    """
    return Gate(name, "and", tuple(inputs))


def or_gate(name: str, *inputs: BasicEvent | Gate) -> Gate:
    """Return a gate that fails when at least one input fails.

    Args:
        name: Gate identifier.
        *inputs: Child events.

    Returns:
        OR gate.

    Equation:
        [EQ-RISK-FT-GATE-001].

    References:
        [REF-VESELY-ETAL-1981].

    """
    return Gate(name, "or", tuple(inputs))


def k_of_n_gate(name: str, k: int, *inputs: BasicEvent | Gate) -> Gate:
    """Return a voting gate that fails when at least ``k`` inputs fail.

    Args:
        name: Gate identifier.
        k: Vote threshold, ``1 <= k <= len(inputs)``.
        *inputs: Child events.

    Returns:
        k-out-of-n gate.

    Equation:
        [EQ-RISK-FT-GATE-001].

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """
    return Gate(name, "k_of_n", tuple(inputs), k)


def _at_least_k(probabilities: Iterable[float], k: int) -> float:
    """Return ``P(at least k of independent events occur)``.

    Args:
        probabilities: Independent event probabilities.
        k: Threshold count.

    Returns:
        Poisson-binomial upper-tail probability by dynamic programming.

    References:
        [EQ-RISK-FT-GATE-001], [REF-RAUSAND-HOYLAND-2004].

    """
    distribution = [1.0]
    for p in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1.0 - p)
            updated[count + 1] += mass * p
        distribution = updated
    return min(1.0, math.fsum(distribution[k:]))


def _gate_probability(
    node: BasicEvent | Gate, probabilities: Mapping[str, float]
) -> float:
    """Evaluate a node bottom-up assuming independent children.

    Args:
        node: Tree node.
        probabilities: Event-name probabilities (repeated events fixed to
            zero or one by the caller).

    Returns:
        Node failure probability.

    References:
        [EQ-RISK-FT-GATE-001], [REF-VESELY-ETAL-1981].

    """
    if isinstance(node, BasicEvent):
        return probabilities[node.name]
    children = [_gate_probability(child, probabilities) for child in node.inputs]
    if node.kind == "and":
        return math.prod(children)
    if node.kind == "or":
        return 1.0 - math.prod(1.0 - p for p in children)
    assert node.k is not None
    return _at_least_k(children, node.k)


@dataclass(frozen=True, slots=True)
class ImportanceMeasures:
    """Component importance measures of one basic event.

    Args:
        event: Basic-event name.
        probability: Event probability used in the evaluation.
        birnbaum: ``Q(1_i) - Q(0_i)``.
        fussell_vesely: ``P(union of minimal cut sets containing i) / Q``.
        risk_achievement_worth: ``Q(1_i) / Q``.
        risk_reduction_worth: ``Q / Q(0_i)`` (``inf`` if ``Q(0_i) = 0``).

    References:
        [REF-BIRNBAUM-1969], [REF-RAUSAND-HOYLAND-2004].

    """

    event: str
    probability: float
    birnbaum: float
    fussell_vesely: float
    risk_achievement_worth: float
    risk_reduction_worth: float


@dataclass(frozen=True)
class FaultTree:
    """A static coherent fault tree with independent basic events.

    Args:
        top: Top-event gate.
        max_repeated_events: Largest number of repeated events conditioned on
            by Shannon decomposition (cost grows as ``2^r``).

    References:
        [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

    """

    top: Gate
    max_repeated_events: int = 16
    _events: Mapping[str, BasicEvent] = field(init=False, repr=False, compare=False)
    _repeated: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _cut_sets: tuple[frozenset[str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Index events, detect repeated events, and compute cut sets.

        Raises:
            TypeError: If ``top`` is not a gate.
            ValueError: If one name denotes different events or gates, or a
                name is used for both a gate and a basic event.

        Equation:
            [EQ-RISK-FT-MOCUS-001].

        References:
            [REF-VESELY-ETAL-1981]. No probability is evaluated.

        """
        if not isinstance(self.top, Gate):
            raise TypeError("top must be a Gate")
        if (
            isinstance(self.max_repeated_events, bool)
            or not isinstance(self.max_repeated_events, int)
            or self.max_repeated_events < 0
        ):
            raise ValueError("max_repeated_events must be a non-negative integer")
        events: dict[str, BasicEvent] = {}
        gates: dict[str, Gate] = {}
        occurrences: dict[str, int] = {}
        stack: list[BasicEvent | Gate] = [self.top]
        while stack:
            node = stack.pop()
            if isinstance(node, BasicEvent):
                if node.name in gates:
                    raise ValueError(f"name {node.name!r} is both a gate and an event")
                known = events.setdefault(node.name, node)
                if known.probability != node.probability:
                    raise ValueError(
                        f"basic event {node.name!r} has conflicting probabilities"
                    )
                occurrences[node.name] = occurrences.get(node.name, 0) + 1
                continue
            if node.name in events:
                raise ValueError(f"name {node.name!r} is both a gate and an event")
            known_gate = gates.setdefault(node.name, node)
            if known_gate != node:
                raise ValueError(f"gate name {node.name!r} has conflicting definitions")
            stack.extend(node.inputs)
        object.__setattr__(
            self, "_events", MappingProxyType(dict(sorted(events.items())))
        )
        object.__setattr__(
            self,
            "_repeated",
            tuple(sorted(name for name, count in occurrences.items() if count > 1)),
        )
        object.__setattr__(self, "_cut_sets", _mocus(self.top))

    @property
    def basic_events(self) -> Mapping[str, BasicEvent]:
        """Return the read-only name-to-event index.

        References:
            [REF-VESELY-ETAL-1981].

        """
        return self._events

    @property
    def repeated_events(self) -> tuple[str, ...]:
        """Return names of basic events occurring more than once.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self._repeated

    @property
    def minimal_cut_sets(self) -> tuple[frozenset[str], ...]:
        """Return minimal cut sets sorted by order, then by names.

        Equation:
            [EQ-RISK-FT-MOCUS-001].

        References:
            [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

        """
        return self._cut_sets

    def _probabilities(self, overrides: Mapping[str, float] | None) -> dict[str, float]:
        """Merge validated probability overrides into the event table.

        Args:
            overrides: Optional event-name probabilities.

        Returns:
            Complete event probability table.

        Raises:
            KeyError: If an override names an unknown event.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        table = {name: event.probability for name, event in self._events.items()}
        for name, value in (overrides or {}).items():
            if name not in table:
                raise KeyError(f"unknown basic event {name!r}")
            table[name] = require_probability(value, name=f"P({name})")
        return table

    def top_event_probability(
        self,
        probabilities: Mapping[str, float] | None = None,
        *,
        method: str = "shannon",
    ) -> float:
        """Return the exact top-event probability under independence.

        Args:
            probabilities: Optional overrides of basic-event probabilities.
            method: ``"shannon"`` (condition on repeated events) or
                ``"inclusion_exclusion"`` over minimal cut sets.

        Returns:
            ``P(top)``.

        Raises:
            ValueError: If the method is unknown or its size guard is
                exceeded.

        Equation:
            [EQ-RISK-FT-SHANNON-001], [EQ-RISK-FT-IE-001],
            [EQ-RISK-FT-GATE-001].

        References:
            [REF-RAUSAND-HOYLAND-2004], [REF-VESELY-ETAL-1981].

        """
        table = self._probabilities(probabilities)
        if method == "shannon":
            return _shannon(self.top, table, self._repeated, self.max_repeated_events)
        if method == "inclusion_exclusion":
            return _inclusion_exclusion(self._cut_sets, table)
        raise ValueError("method must be 'shannon' or 'inclusion_exclusion'")

    def cut_set_probability(
        self,
        cut_set: Iterable[str],
        probabilities: Mapping[str, float] | None = None,
    ) -> float:
        """Return ``P(C) = prod_{i in C} p_i`` for one cut set.

        Args:
            cut_set: Basic-event names.
            probabilities: Optional overrides.

        Returns:
            Cut-set probability.

        Raises:
            KeyError: If a name is unknown.

        Equation:
            [EQ-RISK-FT-BOUND-001].

        References:
            [REF-VESELY-ETAL-1981].

        """
        table = self._probabilities(probabilities)
        return math.prod(table[name] for name in cut_set)

    def rare_event_approximation(
        self, probabilities: Mapping[str, float] | None = None
    ) -> float:
        """Return ``sum_j P(C_j)`` over minimal cut sets (an upper bound).

        Args:
            probabilities: Optional overrides.

        Returns:
            Rare-event approximation (may exceed one).

        Equation:
            [EQ-RISK-FT-BOUND-001].

        References:
            [REF-VESELY-ETAL-1981], [REF-RAUSAND-HOYLAND-2004].

        """
        table = self._probabilities(probabilities)
        return math.fsum(
            math.prod(table[name] for name in cut) for cut in self._cut_sets
        )

    def min_cut_upper_bound(
        self, probabilities: Mapping[str, float] | None = None
    ) -> float:
        """Return ``1 - prod_j (1 - P(C_j))``, an upper bound for coherent trees.

        Args:
            probabilities: Optional overrides.

        Returns:
            Min-cut upper bound.

        Equation:
            [EQ-RISK-FT-BOUND-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        table = self._probabilities(probabilities)
        return 1.0 - math.prod(
            1.0 - math.prod(table[name] for name in cut) for cut in self._cut_sets
        )

    def importance(
        self, probabilities: Mapping[str, float] | None = None
    ) -> Mapping[str, ImportanceMeasures]:
        """Return Birnbaum, Fussell-Vesely, RAW, and RRW for every event.

        Fussell-Vesely uses its cut-set definition, the probability that at
        least one minimal cut set containing the event occurs, divided by
        ``Q``; that union probability is evaluated exactly.

        Args:
            probabilities: Optional overrides.

        Returns:
            Read-only event-name mapping to importance measures. Ratios with a
            zero denominator are ``inf`` (positive numerator) or ``nan``.

        Raises:
            ValueError: If the repeated-event guard is exceeded by the tree or
                by the events shared among cut sets containing one event.

        Equation:
            [EQ-RISK-FT-IMP-001].

        References:
            [REF-BIRNBAUM-1969], [REF-RAUSAND-HOYLAND-2004].

        """
        table = self._probabilities(probabilities)
        q = self.top_event_probability(table)
        result: dict[str, ImportanceMeasures] = {}
        for name in self._events:
            q_one = self.top_event_probability({**table, name: 1.0})
            q_zero = self.top_event_probability({**table, name: 0.0})
            containing = tuple(cut for cut in self._cut_sets if name in cut)
            union = _union_probability(containing, table, self.max_repeated_events)
            result[name] = ImportanceMeasures(
                name,
                table[name],
                q_one - q_zero,
                _ratio(union, q),
                _ratio(q_one, q),
                _ratio(q, q_zero),
            )
        return MappingProxyType(result)


def _ratio(numerator: float, denominator: float) -> float:
    """Return a ratio with explicit infinite and undefined cases.

    Args:
        numerator: Non-negative numerator.
        denominator: Non-negative denominator.

    Returns:
        ``numerator / denominator``; ``inf`` or ``nan`` when the denominator
        is zero.

    References:
        [EQ-RISK-FT-IMP-001], [REF-RAUSAND-HOYLAND-2004].

    """
    if denominator > 0.0:
        return numerator / denominator
    return math.inf if numerator > 0.0 else math.nan


def _mocus(top: Gate) -> tuple[frozenset[str], ...]:
    """Return minimal cut sets by MOCUS expansion and absorption.

    Args:
        top: Top gate.

    Returns:
        Minimal cut sets sorted by order and names.

    Raises:
        ValueError: If the expansion exceeds the internal row guard.

    References:
        [EQ-RISK-FT-MOCUS-001], [REF-VESELY-ETAL-1981],
        [REF-RAUSAND-HOYLAND-2004].

    """
    rows: list[list[BasicEvent | Gate]] = [[top]]
    found: set[frozenset[str]] = set()
    processed = 0
    while rows:
        processed += 1
        if processed > _MAX_MOCUS_ROWS:
            raise ValueError("MOCUS expansion exceeded the row guard")
        row = rows.pop()
        position = next(
            (index for index, node in enumerate(row) if isinstance(node, Gate)), None
        )
        if position is None:
            found.add(frozenset(node.name for node in row))
            continue
        gate = row[position]
        assert isinstance(gate, Gate)
        rest = row[:position] + row[position + 1 :]
        if gate.kind == "and":
            rows.append(rest + list(gate.inputs))
        elif gate.kind == "or":
            rows.extend(rest + [child] for child in gate.inputs)
        else:
            assert gate.k is not None
            rows.extend(
                rest + list(combination)
                for combination in itertools.combinations(gate.inputs, gate.k)
            )
    ordered = sorted(found, key=lambda cut: (len(cut), sorted(cut)))
    minimal: list[frozenset[str]] = []
    for cut in ordered:
        if not any(kept <= cut for kept in minimal):
            minimal.append(cut)
    return tuple(minimal)


def _shannon(
    top: Gate,
    table: Mapping[str, float],
    repeated: tuple[str, ...],
    limit: int,
) -> float:
    """Condition on repeated events and evaluate the tree bottom-up.

    Args:
        top: Top gate.
        table: Event probabilities.
        repeated: Repeated event names.
        limit: Maximum number of repeated events.

    Returns:
        Exact top probability under independence.

    Raises:
        ValueError: If more repeated events than ``limit`` exist.

    References:
        [EQ-RISK-FT-SHANNON-001], [REF-RAUSAND-HOYLAND-2004].

    """
    if len(repeated) > limit:
        raise ValueError(
            f"{len(repeated)} repeated events exceed max_repeated_events={limit}"
        )
    total: list[float] = []
    for states in itertools.product((0, 1), repeat=len(repeated)):
        weight = 1.0
        conditioned = dict(table)
        for name, state in zip(repeated, states, strict=True):
            weight *= table[name] if state else 1.0 - table[name]
            conditioned[name] = float(state)
        if weight == 0.0:
            continue
        total.append(weight * _gate_probability(top, conditioned))
    return min(1.0, max(0.0, math.fsum(total)))


def _union_probability(
    cut_sets: tuple[frozenset[str], ...], table: Mapping[str, float], limit: int
) -> float:
    """Return the exact probability that at least one cut set occurs.

    Args:
        cut_sets: Cut sets of basic-event names.
        table: Event probabilities.
        limit: Maximum number of events shared between cut sets.

    Returns:
        Union probability (zero for an empty collection).

    References:
        [EQ-RISK-FT-SHANNON-001], [REF-RAUSAND-HOYLAND-2004].

    """
    if not cut_sets:
        return 0.0
    tree = FaultTree(
        or_gate(
            "__union__",
            *(
                and_gate(
                    f"__cut_{index}__",
                    *(BasicEvent(name, table[name]) for name in sorted(cut)),
                )
                for index, cut in enumerate(cut_sets)
            ),
        ),
        max_repeated_events=limit,
    )
    return tree.top_event_probability()


def _inclusion_exclusion(
    cut_sets: tuple[frozenset[str], ...],
    table: Mapping[str, float],
    *,
    max_cut_sets: int = 15,
) -> float:
    """Return ``P(union C_j)`` by inclusion-exclusion over cut sets.

    Args:
        cut_sets: Minimal cut sets.
        table: Event probabilities.
        max_cut_sets: Guard on the ``2^m - 1`` term count.

    Returns:
        Exact union probability under independence.

    Raises:
        ValueError: If there are more cut sets than the guard allows.

    References:
        [EQ-RISK-FT-IE-001], [REF-RAUSAND-HOYLAND-2004].

    """
    if len(cut_sets) > max_cut_sets:
        raise ValueError(
            f"{len(cut_sets)} cut sets exceed the inclusion-exclusion guard "
            f"of {max_cut_sets}"
        )
    terms: list[float] = []
    for order in range(1, len(cut_sets) + 1):
        sign = 1.0 if order % 2 else -1.0
        for combination in itertools.combinations(cut_sets, order):
            union = frozenset().union(*combination)
            terms.append(sign * math.prod(table[name] for name in union))
    return min(1.0, max(0.0, math.fsum(terms)))
