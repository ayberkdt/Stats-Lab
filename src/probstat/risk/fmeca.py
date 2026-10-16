"""Failure modes, effects, and criticality analysis (FMECA).

Two prioritisation schemes are provided, and they are *not* equivalent.

**Quantitative criticality (MIL-STD-1629A, Task 102).** For failure mode
``m`` of an item,

    C_m = beta * alpha * lambda_p * t

where ``lambda_p`` is the item (part) failure rate, ``alpha`` the fraction of
item failures that occur in this mode (failure-mode ratio), ``beta`` the
conditional probability that the mode produces the identified failure effect
(failure-effect probability), and ``t`` the operating time (or number of
cycles) per mission. For a small ``C_m``, it approximates the probability of
the mode with that effect during the mission. The item criticality number for
one severity class is ``C_r = sum C_m`` over the item's modes in that class.
The criticality matrix places items (or modes) by severity class against
criticality. Severity classes follow 1629A: I Catastrophic, II Critical,
III Marginal, IV Minor. The qualitative probability levels of 1629A are not
encoded here; bin edges for the matrix are user inputs.

**Risk priority number (RPN = S x O x D).** Severity, occurrence, and
detection are ranked on ordinal 1-10 scales and multiplied. The product of
ordinal ranks is not a meaningful quantity: equal RPNs can hide very
different risks, small changes in one rank change the RPN by very different
amounts, only 120 of the integers 1-1000 can occur, and the ordering depends
on the arbitrary scales (Bowles 2003). :meth:`Fmeca.rpn_ranking` is
provided for compatibility with existing worksheets; prefer the criticality
numbers, or quantitative risk measures, for decisions. The AIAG-VDA (2019)
Action Priority tables are not encoded (see the documentation).

Stability: this module is *experimental* (ADR-0063) inside a *provisional*
package; its names and signatures may change in any release.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations import require_finite_real, require_probability

SEVERITY_CLASSES = MappingProxyType(
    {1: "I Catastrophic", 2: "II Critical", 3: "III Marginal", 4: "IV Minor"}
)

_RATIO_TOLERANCE = 1e-9


def _text(value: object, *, name: str) -> str:
    """Validate non-empty text.

    Args:
        value: Candidate label.
        name: Field name for error messages.

    Returns:
        The label.

    Raises:
        TypeError: If it is not a string.
        ValueError: If it is blank.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _rank(value: object, *, name: str) -> int:
    """Validate an ordinal rank from 1 to 10.

    Args:
        value: Candidate rank.
        name: Field name for error messages.

    Returns:
        The rank.

    Raises:
        TypeError: If it is not an integer (booleans are rejected).
        ValueError: If it is outside ``1..10``.

    References:
        [REF-BOWLES-2003]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= 10:
        raise ValueError(f"{name} must lie in 1..10")
    return value


@dataclass(frozen=True, slots=True)
class FailureMode:
    """One failure mode of one item, with criticality and optional RPN ranks.

    Args:
        item: Item (part, function) identifier.
        mode: Failure-mode description, distinct within the item.
        severity_class: MIL-STD-1629A class 1 (I, Catastrophic) to 4 (IV,
            Minor).
        failure_rate: Item failure rate ``lambda_p`` (non-negative).
        mode_ratio: ``alpha`` in ``[0, 1]``.
        effect_probability: ``beta`` in ``[0, 1]``.
        operating_time: ``t`` per mission (non-negative, same time unit as
            the rate).
        effect: Optional end-effect description.
        severity_rank: Optional RPN severity rank ``S`` (1-10).
        occurrence_rank: Optional RPN occurrence rank ``O`` (1-10).
        detection_rank: Optional RPN detection rank ``D`` (1-10).

    References:
        [REF-MIL-STD-1629A-1980], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    item: str
    mode: str
    severity_class: int
    failure_rate: float
    mode_ratio: float
    effect_probability: float
    operating_time: float
    effect: str = ""
    severity_rank: int | None = None
    occurrence_rank: int | None = None
    detection_rank: int | None = None

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a value is outside its domain.

        References:
            [REF-MIL-STD-1629A-1980]. No equation is evaluated.

        """
        _text(self.item, name="item")
        _text(self.mode, name="mode")
        if (
            isinstance(self.severity_class, bool)
            or self.severity_class not in SEVERITY_CLASSES
        ):
            raise ValueError("severity_class must be 1, 2, 3, or 4")
        rate = require_finite_real(self.failure_rate, name="failure_rate")
        time = require_finite_real(self.operating_time, name="operating_time")
        if rate < 0.0 or time < 0.0:
            raise ValueError("failure_rate and operating_time must be non-negative")
        object.__setattr__(self, "failure_rate", rate)
        object.__setattr__(self, "operating_time", time)
        object.__setattr__(
            self, "mode_ratio", require_probability(self.mode_ratio, name="mode_ratio")
        )
        object.__setattr__(
            self,
            "effect_probability",
            require_probability(self.effect_probability, name="effect_probability"),
        )
        if not isinstance(self.effect, str):
            raise TypeError("effect must be a string")
        ranks = (self.severity_rank, self.occurrence_rank, self.detection_rank)
        for value, label in zip(
            ranks, ("severity", "occurrence", "detection"), strict=True
        ):
            if value is not None:
                _rank(value, name=f"{label}_rank")

    @property
    def criticality(self) -> float:
        """Return the failure-mode criticality number ``C_m = beta alpha lambda_p t``.

        Equation:
            [EQ-RISK-FMECA-001].

        References:
            [REF-MIL-STD-1629A-1980].

        """
        return (
            self.effect_probability
            * self.mode_ratio
            * self.failure_rate
            * self.operating_time
        )

    @property
    def rpn(self) -> int | None:
        """Return ``S x O x D`` when all three ranks are given, else ``None``.

        Equation:
            [EQ-RISK-FMECA-003].

        References:
            [REF-BOWLES-2003].

        """
        if None in (self.severity_rank, self.occurrence_rank, self.detection_rank):
            return None
        return (
            int(self.severity_rank)  # type: ignore[arg-type]
            * int(self.occurrence_rank)  # type: ignore[arg-type]
            * int(self.detection_rank)  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ItemCriticality:
    """Item criticality number for one severity class.

    Args:
        item: Item identifier.
        severity_class: Severity class 1-4.
        criticality: ``C_r = sum C_m`` over the item's modes in the class.
        modes: Modes contributing, in input order.

    References:
        [REF-MIL-STD-1629A-1980].

    """

    item: str
    severity_class: int
    criticality: float
    modes: tuple[str, ...]


@dataclass(frozen=True)
class CriticalityMatrix:
    """Counts and members of a severity-by-criticality grid.

    Args:
        criticality_edges: Strictly increasing interior bin edges on the
            criticality number (bins ``0..len(edges)``; a value equal to an
            edge falls in the higher bin).
        cells: ``(severity_class, criticality_bin) -> item or mode labels``.

    References:
        [REF-MIL-STD-1629A-1980].

    """

    criticality_edges: tuple[float, ...]
    cells: Mapping[tuple[int, int], tuple[str, ...]]

    def count(self, severity_class: int, criticality_bin: int) -> int:
        """Return how many entries fall in one cell.

        Args:
            severity_class: Class 1-4.
            criticality_bin: Bin index.

        Returns:
            Number of entries (zero for an empty cell).

        References:
            [REF-MIL-STD-1629A-1980]. No equation is evaluated.

        """
        return len(self.cells.get((severity_class, criticality_bin), ()))


@dataclass(frozen=True)
class Fmeca:
    """A FMECA worksheet: failure modes with criticality and RPN analyses.

    Args:
        modes: Failure modes; ``(item, mode)`` pairs must be distinct.

    References:
        [REF-MIL-STD-1629A-1980], [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    modes: tuple[FailureMode, ...]

    def __post_init__(self) -> None:
        """Validate the worksheet.

        Raises:
            TypeError: If an entry is not a :class:`FailureMode`.
            ValueError: If it is empty or an ``(item, mode)`` pair repeats.

        References:
            Project validation policy. No equation is evaluated.

        """
        modes = tuple(self.modes)
        if not modes:
            raise ValueError("at least one failure mode is required")
        for mode in modes:
            if not isinstance(mode, FailureMode):
                raise TypeError("modes must be FailureMode instances")
        keys = [(m.item, m.mode) for m in modes]
        if len(set(keys)) != len(keys):
            raise ValueError("(item, mode) pairs must be distinct")
        object.__setattr__(self, "modes", modes)

    def mode_ratio_sums(self) -> Mapping[str, float]:
        """Return ``sum alpha`` per item.

        The mode ratios of an item should sum to one when every failure mode
        is listed; a smaller sum means modes are missing, a larger one is an
        error in the data.

        References:
            [REF-MIL-STD-1629A-1980]. No equation is evaluated.

        """
        sums: dict[str, list[float]] = {}
        for m in self.modes:
            sums.setdefault(m.item, []).append(m.mode_ratio)
        return MappingProxyType({k: math.fsum(v) for k, v in sums.items()})

    def inconsistent_items(self) -> tuple[str, ...]:
        """Return items whose mode ratios sum to more than one.

        References:
            [REF-MIL-STD-1629A-1980]. No equation is evaluated.

        """
        return tuple(
            item
            for item, total in self.mode_ratio_sums().items()
            if total > 1.0 + _RATIO_TOLERANCE
        )

    def mode_criticality(self) -> tuple[tuple[FailureMode, float], ...]:
        """Return every mode with ``C_m``, in descending order of ``C_m``.

        Equation:
            [EQ-RISK-FMECA-001].

        References:
            [REF-MIL-STD-1629A-1980].

        """
        return tuple(
            sorted(((m, m.criticality) for m in self.modes), key=lambda p: -p[1])
        )

    def item_criticality(self) -> tuple[ItemCriticality, ...]:
        """Return ``C_r`` per (item, severity class), most severe class first.

        Within a class, items are in descending order of ``C_r``.

        Equation:
            [EQ-RISK-FMECA-002].

        References:
            [REF-MIL-STD-1629A-1980].

        """
        groups: dict[tuple[str, int], list[FailureMode]] = {}
        for m in self.modes:
            groups.setdefault((m.item, m.severity_class), []).append(m)
        rows = [
            ItemCriticality(
                item,
                severity,
                math.fsum(m.criticality for m in members),
                tuple(m.mode for m in members),
            )
            for (item, severity), members in groups.items()
        ]
        return tuple(sorted(rows, key=lambda r: (r.severity_class, -r.criticality)))

    def criticality_matrix(
        self, criticality_edges: Sequence[float], *, level: str = "item"
    ) -> CriticalityMatrix:
        """Place items (``C_r``) or modes (``C_m``) in a severity grid.

        Args:
            criticality_edges: Strictly increasing interior bin edges on the
                criticality number, chosen by the analyst (no standard
                defaults are assumed).
            level: ``"item"`` (``C_r`` per item and class) or ``"mode"``.

        Returns:
            Criticality matrix with labelled cells.

        Raises:
            ValueError: If edges do not increase or ``level`` is unknown.

        Equation:
            [EQ-RISK-FMECA-001], [EQ-RISK-FMECA-002].

        References:
            [REF-MIL-STD-1629A-1980].

        """
        edges = tuple(require_finite_real(v, name="edge") for v in criticality_edges)
        if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
            raise ValueError("criticality_edges must strictly increase")
        cells: dict[tuple[int, int], list[str]] = {}
        if level == "item":
            entries: Iterable[tuple[str, int, float]] = (
                (row.item, row.severity_class, row.criticality)
                for row in self.item_criticality()
            )
        elif level == "mode":
            entries = (
                (f"{m.item}: {m.mode}", m.severity_class, m.criticality)
                for m in self.modes
            )
        else:
            raise ValueError("level must be 'item' or 'mode'")
        for label, severity, value in entries:
            key = (severity, bisect.bisect_right(edges, value))
            cells.setdefault(key, []).append(label)
        return CriticalityMatrix(
            edges, MappingProxyType({k: tuple(v) for k, v in cells.items()})
        )

    def rpn_ranking(self) -> tuple[tuple[FailureMode, int], ...]:
        """Return modes with an RPN in descending RPN order (stable on ties).

        RPN multiplies ordinal ranks; read the module caveat before using it
        for decisions.

        Raises:
            ValueError: If no mode has all three ranks.

        Equation:
            [EQ-RISK-FMECA-003].

        References:
            [REF-BOWLES-2003].

        """
        scored = [(m, m.rpn) for m in self.modes if m.rpn is not None]
        if not scored:
            raise ValueError(
                "no failure mode has severity, occurrence and detection ranks"
            )
        return tuple(sorted(((m, int(r)) for m, r in scored), key=lambda p: -p[1]))


def risk_priority_number(severity: int, occurrence: int, detection: int) -> int:
    """Return ``RPN = S x O x D`` for ranks on 1-10 scales.

    The value is a product of ordinal ranks, not a measure of risk; see the
    module caveat (Bowles 2003).

    Args:
        severity: Severity rank ``S``.
        occurrence: Occurrence rank ``O``.
        detection: Detection rank ``D``.

    Returns:
        RPN in ``1..1000`` (only 120 distinct values are possible).

    Equation:
        [EQ-RISK-FMECA-003].

    References:
        [REF-BOWLES-2003].

    """
    return (
        _rank(severity, name="severity")
        * _rank(occurrence, name="occurrence")
        * _rank(detection, name="detection")
    )
