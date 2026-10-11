"""Event trees: initiating events, branch probabilities, and sequence risk.

An event tree starts from an initiating event with an annual frequency and
branches at successive headings (safety functions or barriers). Each path ends
in an :class:`Outcome` with a consequence. A sequence frequency is the
initiating frequency times the conditional branch probabilities along its
path; total risk is the frequency-weighted sum of consequences, the expected
consequence per year in the Kaplan-Garrick scenario view.

Branch probabilities are *conditional on the path so far*, so dependence
between headings must be encoded in the supplied numbers. Linking fault trees
with shared basic events across headings is not performed here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations import require_finite_real, require_probability

_BRANCH_SUM_TOLERANCE = 1e-9


def _require_label(value: object, *, name: str) -> str:
    """Validate a non-empty text label.

    Args:
        value: Candidate label.
        name: Field name for error messages.

    Returns:
        The label.

    Raises:
        TypeError: If the label is not text.
        ValueError: If the label is empty.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class Outcome:
    """A terminal end state with a scalar consequence.

    Args:
        name: End-state label (several paths may share one).
        consequence: Finite consequence (e.g. cost or fatalities) per event.

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-RAUSAND-HOYLAND-2004].

    """

    name: str
    consequence: float

    def __post_init__(self) -> None:
        """Validate the label and consequence.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If the label is empty or the consequence non-finite.

        References:
            Project validation policy. No equation is evaluated.

        """
        _require_label(self.name, name="name")
        object.__setattr__(
            self,
            "consequence",
            require_finite_real(self.consequence, name="consequence"),
        )


@dataclass(frozen=True, slots=True)
class Branch:
    """One conditional branch leaving a heading.

    Args:
        label: Branch label such as ``"success"`` or ``"failure"``.
        probability: Conditional probability given the path so far.
        next: Following heading or terminal outcome.

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """

    label: str
    probability: float
    next: Heading | Outcome

    def __post_init__(self) -> None:
        """Validate the label, probability, and successor.

        Raises:
            TypeError: If the successor has an invalid type.
            ValueError: If the label or probability is invalid.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        _require_label(self.label, name="label")
        object.__setattr__(
            self,
            "probability",
            require_probability(self.probability, name="probability"),
        )
        if not isinstance(self.next, Heading | Outcome):
            raise TypeError("next must be a Heading or Outcome")


@dataclass(frozen=True, slots=True)
class Heading:
    """A branching point (safety function) with exhaustive branches.

    Args:
        name: Heading label.
        branches: At least two branches whose probabilities sum to one.

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """

    name: str
    branches: tuple[Branch, ...]

    def __post_init__(self) -> None:
        """Validate the label and the branch partition.

        Raises:
            TypeError: If a branch has an invalid type.
            ValueError: If fewer than two branches exist, labels repeat, or
                probabilities do not sum to one within ``1e-9``.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        _require_label(self.name, name="name")
        branches = tuple(self.branches)
        if len(branches) < 2:
            raise ValueError("a heading needs at least two branches")
        for index, branch in enumerate(branches):
            if not isinstance(branch, Branch):
                raise TypeError(f"branches[{index}] must be a Branch")
        labels = [branch.label for branch in branches]
        if len(set(labels)) != len(labels):
            raise ValueError("branch labels must be distinct within a heading")
        total = math.fsum(branch.probability for branch in branches)
        if abs(total - 1.0) > _BRANCH_SUM_TOLERANCE:
            raise ValueError(
                f"branch probabilities of heading {self.name!r} sum to {total}"
            )
        object.__setattr__(self, "branches", branches)


def binary_heading(
    name: str,
    failure_probability: float,
    success: Heading | Outcome,
    failure: Heading | Outcome,
) -> Heading:
    """Return a two-branch heading with ``success`` and ``failure`` labels.

    Args:
        name: Heading label.
        failure_probability: Conditional failure probability.
        success: Successor after success.
        failure: Successor after failure.

    Returns:
        Binary heading.

    Equation:
        [EQ-RISK-ET-001].

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """
    q = require_probability(failure_probability, name="failure_probability")
    return Heading(
        name, (Branch("success", 1.0 - q, success), Branch("failure", q, failure))
    )


@dataclass(frozen=True, slots=True)
class EventSequence:
    """One accident sequence from initiating event to outcome.

    Args:
        outcome: End-state label.
        path: ``(heading, branch label)`` pairs along the sequence.
        conditional_probability: Product of branch probabilities.
        frequency: Initiating frequency times conditional probability.
        consequence: Outcome consequence.

    References:
        [REF-KAPLAN-GARRICK-1981], [REF-RAUSAND-HOYLAND-2004].

    """

    outcome: str
    path: tuple[tuple[str, str], ...]
    conditional_probability: float
    frequency: float
    consequence: float

    @property
    def risk(self) -> float:
        """Return the sequence risk ``f * c``.

        Equation:
            [EQ-RISK-DEF-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        return self.frequency * self.consequence


@dataclass(frozen=True)
class EventTree:
    """An initiating event followed by a tree of headings and outcomes.

    Args:
        initiating_event: Initiating-event label.
        frequency: Non-negative initiating-event frequency (per year).
        root: First heading, or an outcome for a trivial tree.

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-KAPLAN-GARRICK-1981].

    """

    initiating_event: str
    frequency: float
    root: Heading | Outcome

    def __post_init__(self) -> None:
        """Validate the initiating event.

        Raises:
            TypeError: If the root has an invalid type.
            ValueError: If the label is empty or the frequency negative.

        References:
            Project validation policy. No equation is evaluated.

        """
        _require_label(self.initiating_event, name="initiating_event")
        rate = require_finite_real(self.frequency, name="frequency")
        if rate < 0.0:
            raise ValueError("frequency must be non-negative")
        object.__setattr__(self, "frequency", rate)
        if not isinstance(self.root, Heading | Outcome):
            raise TypeError("root must be a Heading or Outcome")

    def sequences(self) -> tuple[EventSequence, ...]:
        """Enumerate all sequences in depth-first branch order.

        Returns:
            Sequences with frequencies ``f_IE prod_k p_k``.

        Equation:
            [EQ-RISK-ET-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        result: list[EventSequence] = []
        stack: list[tuple[Heading | Outcome, tuple[tuple[str, str], ...], float]] = [
            (self.root, (), 1.0)
        ]
        while stack:
            node, path, probability = stack.pop()
            if isinstance(node, Outcome):
                result.append(
                    EventSequence(
                        node.name,
                        path,
                        probability,
                        self.frequency * probability,
                        node.consequence,
                    )
                )
                continue
            for branch in reversed(node.branches):
                stack.append(
                    (
                        branch.next,
                        path + ((node.name, branch.label),),
                        probability * branch.probability,
                    )
                )
        return tuple(result)

    def total_frequency(self) -> float:
        """Return the sum of sequence frequencies (equals ``f_IE``).

        Equation:
            [EQ-RISK-ET-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return math.fsum(sequence.frequency for sequence in self.sequences())

    def total_risk(self) -> float:
        """Return expected consequence per unit time ``sum_i f_i c_i``.

        Equation:
            [EQ-RISK-DEF-001], [EQ-RISK-ET-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        return math.fsum(sequence.risk for sequence in self.sequences())

    def outcome_frequencies(self) -> Mapping[str, float]:
        """Return total frequency per end-state label.

        Equation:
            [EQ-RISK-ET-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        totals: dict[str, list[float]] = {}
        for sequence in self.sequences():
            totals.setdefault(sequence.outcome, []).append(sequence.frequency)
        return MappingProxyType({k: math.fsum(v) for k, v in totals.items()})

    def ranked_sequences(self, *, by: str = "risk") -> tuple[EventSequence, ...]:
        """Return sequences sorted in descending order of risk or frequency.

        Args:
            by: ``"risk"`` or ``"frequency"``.

        Returns:
            Sorted sequences (stable for ties).

        Raises:
            ValueError: If ``by`` is unknown.

        Equation:
            [EQ-RISK-DEF-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        if by == "risk":
            return tuple(sorted(self.sequences(), key=lambda s: -s.risk))
        if by == "frequency":
            return tuple(sorted(self.sequences(), key=lambda s: -s.frequency))
        raise ValueError("by must be 'risk' or 'frequency'")

    def consequence_exceedance_frequency(self, level: float) -> float:
        """Return the frequency of sequences with consequence ``> level``.

        Args:
            level: Finite consequence level.

        Returns:
            Exceedance frequency (one point of an F-N style curve).

        Equation:
            [EQ-RISK-EXC-001], [EQ-RISK-ET-001].

        References:
            [REF-KAPLAN-GARRICK-1981].

        """
        threshold = require_finite_real(level, name="level")
        return math.fsum(
            s.frequency for s in self.sequences() if s.consequence > threshold
        )
