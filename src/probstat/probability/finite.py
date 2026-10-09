"""Finite probability spaces, events, measures, and derived probability rules."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Generic, TypeVar

from probstat.foundations.validation import (
    require_non_negative_finite_real,
    require_probability,
)

OutcomeT = TypeVar("OutcomeT", bound=Hashable)
_NORMALIZATION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FiniteSampleSpace(Generic[OutcomeT]):
    """A non-empty finite set of distinct, hashable outcomes.

    This initial finite model uses the full power set as its sigma-algebra, so
    every subset of ``outcomes`` is an event. Input order is retained only for
    deterministic display and iteration. Sample-space ownership is identity-
    based: two separately constructed spaces remain different experiments even
    when they contain equal outcome labels.

    Args:
        outcomes: Distinct hashable outcomes. At least one outcome is required.

    Raises:
        TypeError: If an outcome is not hashable.
        ValueError: If the iterable is empty or repeats an outcome.

    References:
        [REF-ROSS-2019, Chapter 2], [REF-KOLMOGOROV-1956, Chapter I].

    """

    outcomes: frozenset[OutcomeT]
    _order: tuple[OutcomeT, ...] = field(compare=False, hash=False, repr=False)

    def __init__(self, outcomes: Iterable[OutcomeT]) -> None:
        """Materialize and validate a finite sample space.

        Args:
            outcomes: Distinct hashable outcomes.

        Raises:
            TypeError: If an outcome is not hashable.
            ValueError: If no outcome is supplied or an outcome is repeated.

        References:
            [REF-ROSS-2019, Chapter 2]. This constructor establishes a domain
            and evaluates no probability equation.

        """
        ordered: list[OutcomeT] = []
        seen: set[OutcomeT] = set()
        for index, outcome in enumerate(outcomes):
            try:
                is_duplicate = outcome in seen
            except TypeError as error:
                raise TypeError(f"outcomes[{index}] must be hashable") from error
            if is_duplicate:
                raise ValueError(f"outcomes[{index}] repeats an earlier outcome")
            seen.add(outcome)
            ordered.append(outcome)
        if not ordered:
            raise ValueError("a finite sample space must contain at least one outcome")
        object.__setattr__(self, "outcomes", frozenset(seen))
        object.__setattr__(self, "_order", tuple(ordered))

    def __len__(self) -> int:
        """Return the number of outcomes in the sample space.

        References:
            [REF-ROSS-2019, Chapter 2]. No probability equation is evaluated.

        """
        return len(self.outcomes)

    def __iter__(self) -> Iterator[OutcomeT]:
        """Iterate over outcomes in their validated input order.

        Returns:
            An iterator over each distinct outcome exactly once.

        References:
            [REF-ROSS-2019, Chapter 2]. No probability equation is evaluated.

        """
        return iter(self._order)

    def __contains__(self, outcome: object) -> bool:
        """Return whether ``outcome`` belongs to the sample space.

        Args:
            outcome: Candidate outcome.

        Returns:
            ``True`` exactly when the candidate is in ``outcomes``.

        References:
            [REF-ROSS-2019, Chapter 2]. No probability equation is evaluated.

        """
        return outcome in self.outcomes

    def event(self, members: Iterable[OutcomeT]) -> FiniteEvent[OutcomeT]:
        """Create an immutable event belonging to this sample space.

        Args:
            members: Outcomes to include in the event.

        Returns:
            A validated event, including the empty event when no members are
            supplied.

        Raises:
            TypeError: If a member is not hashable.
            ValueError: If a member is outside this sample space.

        References:
            [REF-ROSS-2019, Chapter 2]. Event construction evaluates no
            probability equation.

        """
        return FiniteEvent(self, members)

    def singleton(self, outcome: OutcomeT) -> FiniteEvent[OutcomeT]:
        """Create the event containing exactly one validated outcome.

        Args:
            outcome: The sole event member.

        Returns:
            A singleton event in this sample space.

        Raises:
            TypeError: If the outcome is not hashable.
            ValueError: If the outcome is outside this sample space.

        References:
            [REF-ROSS-2019, Chapter 2]. Event construction evaluates no
            probability equation.

        """
        return self.event((outcome,))

    @property
    def empty_event(self) -> FiniteEvent[OutcomeT]:
        """Return the impossible event containing no outcomes.

        References:
            [EQ-PROB-003], [REF-KOLMOGOROV-1956, Chapter I, Section 2].

        """
        return self.event(())

    @property
    def certain_event(self) -> FiniteEvent[OutcomeT]:
        """Return the certain event containing the entire sample space.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956, Chapter I, Section 2].

        """
        return self.event(self._order)


@dataclass(frozen=True, slots=True, init=False)
class FiniteEvent(Generic[OutcomeT]):
    """An immutable subset of a particular finite sample space.

    Args:
        space: The owning sample space.
        members: Outcomes in the event.

    Raises:
        TypeError: If a member is not hashable.
        ValueError: If a member is outside ``space``.

    References:
        [REF-ROSS-2019, Chapter 2].

    """

    space: FiniteSampleSpace[OutcomeT]
    members: frozenset[OutcomeT]

    def __init__(
        self, space: FiniteSampleSpace[OutcomeT], members: Iterable[OutcomeT]
    ) -> None:
        """Materialize event members and enforce sample-space ownership.

        Args:
            space: The owning finite sample space.
            members: Outcomes to include.

        Raises:
            TypeError: If a member is not hashable.
            ValueError: If a member is outside ``space``.

        References:
            [REF-ROSS-2019, Chapter 2]. No probability equation is evaluated.

        """
        try:
            materialized = frozenset(members)
        except TypeError as error:
            raise TypeError("event members must be hashable") from error
        outside = materialized - space.outcomes
        if outside:
            raise ValueError(f"event contains outcomes outside its space: {outside!r}")
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "members", materialized)

    def __len__(self) -> int:
        """Return the number of outcomes in the event.

        References:
            [REF-ROSS-2019, Chapter 2]. No probability equation is evaluated.

        """
        return len(self.members)

    def __contains__(self, outcome: object) -> bool:
        """Return whether ``outcome`` belongs to the event.

        Args:
            outcome: Candidate outcome.

        Returns:
            ``True`` exactly when the candidate is an event member.

        References:
            [REF-ROSS-2019, Chapter 2]. No probability equation is evaluated.

        """
        return outcome in self.members

    def _require_same_space(self, other: FiniteEvent[OutcomeT]) -> None:
        """Reject set operations across different sample spaces.

        Args:
            other: Event expected to share this event's sample space.

        Raises:
            TypeError: If ``other`` is not a finite event.
            ValueError: If the events belong to different sample spaces.

        References:
            [REF-ROSS-2019, Chapter 2]. This validator evaluates no equation.

        """
        if not isinstance(other, FiniteEvent):
            raise TypeError("other must be a FiniteEvent")
        if self.space != other.space:
            raise ValueError("events must belong to the same sample space")

    def complement(self) -> FiniteEvent[OutcomeT]:
        """Return all outcomes in the sample space but not in this event.

        Returns:
            The set complement relative to ``space``.

        Equation:
            [EQ-PROB-003].

        References:
            [REF-ROSS-2019, Chapter 2].

        """
        return self.space.event(self.space.outcomes - self.members)

    def union(self, other: FiniteEvent[OutcomeT]) -> FiniteEvent[OutcomeT]:
        """Return outcomes belonging to either event or both.

        Args:
            other: Event in the same sample space.

        Returns:
            The set union of both events.

        Raises:
            TypeError: If ``other`` is not a finite event.
            ValueError: If the events have different sample spaces.

        Equation:
            [EQ-PROB-004].

        References:
            [REF-ROSS-2019, Chapter 2].

        """
        self._require_same_space(other)
        return self.space.event(self.members | other.members)

    def intersection(self, other: FiniteEvent[OutcomeT]) -> FiniteEvent[OutcomeT]:
        """Return outcomes belonging to both events.

        Args:
            other: Event in the same sample space.

        Returns:
            The set intersection of both events.

        Raises:
            TypeError: If ``other`` is not a finite event.
            ValueError: If the events have different sample spaces.

        References:
            [EQ-PROB-004], [EQ-PROB-005], [REF-ROSS-2019, Chapters 2-3].

        """
        self._require_same_space(other)
        return self.space.event(self.members & other.members)

    def difference(self, other: FiniteEvent[OutcomeT]) -> FiniteEvent[OutcomeT]:
        """Return outcomes in this event but not in ``other``.

        Args:
            other: Event in the same sample space.

        Returns:
            The relative set difference.

        Raises:
            TypeError: If ``other`` is not a finite event.
            ValueError: If the events have different sample spaces.

        References:
            [REF-ROSS-2019, Chapter 2]. No new probability equation is used.

        """
        self._require_same_space(other)
        return self.space.event(self.members - other.members)

    def symmetric_difference(
        self, other: FiniteEvent[OutcomeT]
    ) -> FiniteEvent[OutcomeT]:
        """Return outcomes belonging to exactly one of two events.

        Args:
            other: Event in the same sample space.

        Returns:
            The symmetric set difference.

        Raises:
            TypeError: If ``other`` is not a finite event.
            ValueError: If the events have different sample spaces.

        References:
            [REF-ROSS-2019, Chapter 2]. No new probability equation is used.

        """
        self._require_same_space(other)
        return self.space.event(self.members ^ other.members)

    def is_disjoint(self, other: FiniteEvent[OutcomeT]) -> bool:
        """Return whether two events have no shared outcome.

        Args:
            other: Event in the same sample space.

        Returns:
            ``True`` when the intersection is empty.

        Raises:
            TypeError: If ``other`` is not a finite event.
            ValueError: If the events have different sample spaces.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956, Chapter I, Section 2].

        """
        self._require_same_space(other)
        return self.members.isdisjoint(other.members)

    def is_subset_of(self, other: FiniteEvent[OutcomeT]) -> bool:
        """Return whether every member also belongs to ``other``.

        Args:
            other: Event in the same sample space.

        Returns:
            ``True`` when this event is a subset of ``other``.

        Raises:
            TypeError: If ``other`` is not a finite event.
            ValueError: If the events have different sample spaces.

        References:
            [REF-ROSS-2019, Chapter 2]. No new probability equation is used.

        """
        self._require_same_space(other)
        return self.members <= other.members


@dataclass(frozen=True, slots=True, init=False)
class FiniteProbabilityMeasure(Generic[OutcomeT]):
    """A validated probability measure on a finite sample space.

    The supplied non-negative weights must cover the sample space exactly and
    sum to one within ``1e-12`` absolute tolerance. They are then divided by
    their computed total so the stored representation is normalized while
    preserving relative weights.

    Args:
        space: Non-empty finite sample space.
        weights: One non-negative finite weight per outcome, with no extras.

    Raises:
        TypeError: If a weight is not real.
        ValueError: If keys do not match the sample space, a weight is invalid,
            or the total is not one within tolerance.

    Equation:
        [EQ-PROB-002].

    References:
        [REF-KOLMOGOROV-1956, Chapter I], [REF-ROSS-2019, Chapter 2].

    """

    space: FiniteSampleSpace[OutcomeT]
    _weights: Mapping[OutcomeT, float] = field(repr=False)

    def __init__(
        self, space: FiniteSampleSpace[OutcomeT], weights: Mapping[OutcomeT, Real]
    ) -> None:
        """Validate complete outcome weights and freeze the normalized mapping.

        Args:
            space: Non-empty finite sample space.
            weights: Mapping with exactly one weight for every outcome.

        Raises:
            TypeError: If a weight is not real.
            ValueError: If keys or weight values violate the measure contract.

        Equation:
            [EQ-PROB-001], [EQ-PROB-002].

        References:
            [REF-KOLMOGOROV-1956, Chapter I], [REF-ROSS-2019, Chapter 2].

        """
        supplied_outcomes = frozenset(weights)
        missing = space.outcomes - supplied_outcomes
        extra = supplied_outcomes - space.outcomes
        if missing or extra:
            raise ValueError(
                f"weight keys must equal the sample space; missing={missing!r}, "
                f"extra={extra!r}"
            )
        validated: dict[OutcomeT, float] = {}
        for outcome in space:
            weight = require_probability(weights[outcome], name=f"weight[{outcome!r}]")
            validated[outcome] = weight
        total = math.fsum(validated.values())
        if not math.isclose(
            total, 1.0, rel_tol=0.0, abs_tol=_NORMALIZATION_TOLERANCE
        ):
            raise ValueError(f"weights must sum to one; received {total!r}")
        normalized = {outcome: weight / total for outcome, weight in validated.items()}
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "_weights", MappingProxyType(normalized))

    @classmethod
    def uniform(
        cls, space: FiniteSampleSpace[OutcomeT]
    ) -> FiniteProbabilityMeasure[OutcomeT]:
        """Construct the uniform probability measure on ``space``.

        Args:
            space: Non-empty finite sample space.

        Returns:
            A measure assigning every outcome probability ``1 / |Omega|``.

        Equation:
            [EQ-PROB-002].

        References:
            [REF-ROSS-2019, Chapter 2].

        """
        mass = 1.0 / len(space)
        return cls(space, {outcome: mass for outcome in space})

    @property
    def weights(self) -> Mapping[OutcomeT, float]:
        """Return a read-only outcome-to-probability mapping.

        Equation:
            [EQ-PROB-002].

        References:
            [REF-KOLMOGOROV-1956, Chapter I].

        """
        return self._weights

    def _require_owned_event(self, event: FiniteEvent[OutcomeT]) -> None:
        """Require an event belonging to this measure's sample space.

        Args:
            event: Candidate finite event.

        Raises:
            TypeError: If ``event`` is not a finite event.
            ValueError: If ``event`` belongs to another sample space.

        References:
            [REF-ROSS-2019, Chapters 2-3]. No equation is evaluated.

        """
        if not isinstance(event, FiniteEvent):
            raise TypeError("event must be a FiniteEvent")
        if event.space != self.space:
            raise ValueError("event must belong to the measure's sample space")

    def probability(self, event: FiniteEvent[OutcomeT]) -> float:
        """Return the probability mass assigned to ``event``.

        Args:
            event: Event in this measure's sample space.

        Returns:
            The sum of outcome weights over the event.

        Raises:
            TypeError: If ``event`` is not a finite event.
            ValueError: If ``event`` belongs to another sample space.

        Equation:
            [EQ-PROB-002].

        References:
            [REF-KOLMOGOROV-1956, Chapter I], [REF-ROSS-2019, Chapter 2].

        """
        self._require_owned_event(event)
        return math.fsum(self._weights[outcome] for outcome in event.members)

    def conditional_probability(
        self, event: FiniteEvent[OutcomeT], given: FiniteEvent[OutcomeT]
    ) -> float:
        """Return ``P(event | given)`` when the conditioning event is possible.

        Args:
            event: Event whose conditional probability is requested.
            given: Conditioning event with strictly positive probability.

        Returns:
            Conditional probability of ``event`` given ``given``.

        Raises:
            TypeError: If either argument is not a finite event.
            ValueError: If ownership differs or ``P(given) == 0``.

        Equation:
            [EQ-PROB-005].

        References:
            [REF-ROSS-2019, Chapter 3].

        """
        self._require_owned_event(event)
        self._require_owned_event(given)
        given_probability = self.probability(given)
        if given_probability == 0.0:
            raise ValueError("conditional probability requires P(given) > 0")
        joint = self.probability(event.intersection(given))
        return joint / given_probability

    def are_independent(
        self,
        first: FiniteEvent[OutcomeT],
        second: FiniteEvent[OutcomeT],
        *,
        tolerance: float = 1e-12,
    ) -> bool:
        """Test event independence using a symmetric product criterion.

        Args:
            first: First event.
            second: Second event in the same sample space.
            tolerance: Non-negative absolute numerical comparison tolerance.

        Returns:
            Whether ``P(first intersection second)`` equals
            ``P(first) * P(second)`` within ``tolerance``.

        Raises:
            TypeError: If events or ``tolerance`` have invalid types.
            ValueError: If ownership differs or ``tolerance`` is invalid.

        Equation:
            [EQ-PROB-006].

        References:
            [REF-ROSS-2019, Chapter 3].

        """
        self._require_owned_event(first)
        self._require_owned_event(second)
        checked_tolerance = require_non_negative_finite_real(
            tolerance, name="tolerance"
        )
        joint = self.probability(first.intersection(second))
        product = self.probability(first) * self.probability(second)
        return math.isclose(
            joint, product, rel_tol=0.0, abs_tol=checked_tolerance
        )

    def total_probability(
        self,
        event: FiniteEvent[OutcomeT],
        partition: Iterable[FiniteEvent[OutcomeT]],
    ) -> float:
        """Evaluate the law of total probability over a positive partition.

        Args:
            event: Target event.
            partition: Non-empty, pairwise-disjoint events whose union is the
                entire sample space and whose probabilities are all positive.

        Returns:
            ``P(event)`` reconstructed from conditional probabilities.

        Raises:
            TypeError: If an item is not a finite event.
            ValueError: If event ownership differs, the cells are not a
                partition, or any cell has zero probability.

        Equation:
            [EQ-PROB-007].

        References:
            [REF-ROSS-2019, Chapter 3].

        """
        self._require_owned_event(event)
        cells = tuple(partition)
        self._validate_positive_partition(cells)
        terms = (
            self.conditional_probability(event, cell) * self.probability(cell)
            for cell in cells
        )
        return math.fsum(terms)

    def _validate_positive_partition(
        self, cells: tuple[FiniteEvent[OutcomeT], ...]
    ) -> None:
        """Validate a non-empty, disjoint, exhaustive, positive partition.

        Args:
            cells: Materialized candidate partition.

        Raises:
            TypeError: If an item is not a finite event.
            ValueError: If ownership, disjointness, coverage, or positivity
                fails.

        References:
            [EQ-PROB-007], [REF-ROSS-2019, Chapter 3].

        """
        if not cells:
            raise ValueError("partition must contain at least one event")
        covered: set[OutcomeT] = set()
        for index, cell in enumerate(cells):
            self._require_owned_event(cell)
            overlap = covered & cell.members
            if overlap:
                raise ValueError(
                    f"partition cell {index} overlaps earlier cells: {overlap!r}"
                )
            if self.probability(cell) == 0.0:
                raise ValueError(
                    f"partition cell {index} must have positive probability"
                )
            covered.update(cell.members)
        if covered != self.space.outcomes:
            missing = self.space.outcomes - covered
            raise ValueError(f"partition does not cover the sample space: {missing!r}")

    def bayes_probability(
        self,
        hypothesis: FiniteEvent[OutcomeT],
        evidence: FiniteEvent[OutcomeT],
    ) -> float:
        """Return posterior event probability using Bayes' rule.

        Args:
            hypothesis: Hypothesis event with strictly positive prior
                probability.
            evidence: Evidence event with strictly positive marginal
                probability.

        Returns:
            ``P(hypothesis | evidence)`` evaluated by the Bayes factorization.

        Raises:
            TypeError: If either argument is not a finite event.
            ValueError: If ownership differs or either required probability is
                zero.

        Equation:
            [EQ-PROB-008].

        References:
            [REF-ROSS-2019, Chapter 3].

        """
        self._require_owned_event(hypothesis)
        self._require_owned_event(evidence)
        prior = self.probability(hypothesis)
        evidence_probability = self.probability(evidence)
        if prior == 0.0:
            raise ValueError("Bayes' rule requires P(hypothesis) > 0")
        if evidence_probability == 0.0:
            raise ValueError("Bayes' rule requires P(evidence) > 0")
        likelihood = self.conditional_probability(evidence, hypothesis)
        return likelihood * prior / evidence_probability

    def expectation(self, value: Callable[[OutcomeT], Real]) -> float:
        """Return the expectation of a finite real-valued outcome function.

        This bridge treats ``value`` as a finite random variable without yet
        introducing a separate random-variable object. A dedicated abstraction
        is planned for the next probability-foundation round.

        Args:
            value: Function mapping every outcome to a finite real number.

        Returns:
            Probability-weighted average of the function values.

        Raises:
            TypeError: If a function value is not real.
            ValueError: If a function value is non-finite.

        Equation:
            [EQ-PROB-009].

        References:
            [REF-ROSS-2019, Chapter 4].

        """
        terms: list[float] = []
        for outcome in self.space:
            result = value(outcome)
            if isinstance(result, bool) or not isinstance(result, Real):
                raise TypeError(f"value({outcome!r}) must be a real number")
            numeric_result = float(result)
            if not math.isfinite(numeric_result):
                raise ValueError(f"value({outcome!r}) must be finite")
            terms.append(numeric_result * self._weights[outcome])
        return math.fsum(terms)
