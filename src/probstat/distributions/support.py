"""Immutable support objects for discrete and continuous distributions."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class IntegerSupport:
    """A bounded or unbounded inclusive integer support.

    Args:
        lower: Inclusive integer lower bound.
        upper: Optional inclusive integer upper bound.

    References:
        [REF-ROSS-2019, Chapter 4].

    """

    lower: int
    upper: int | None

    def __post_init__(self) -> None:
        """Validate support bounds.

        Raises:
            TypeError: If a bound is not an integer.
            ValueError: If ``upper < lower``.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        if isinstance(self.lower, bool) or not isinstance(self.lower, int):
            raise TypeError("lower must be an integer")
        if self.upper is not None:
            if isinstance(self.upper, bool) or not isinstance(self.upper, int):
                raise TypeError("upper must be an integer or None")
            if self.upper < self.lower:
                raise ValueError("upper must not be below lower")

    def contains(self, value: object) -> bool:
        """Return whether ``value`` is an integer in the support.

        Args:
            value: Candidate support value.

        Returns:
            Membership result without coercion.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if value < self.lower:
            return False
        return self.upper is None or value <= self.upper

    def finite_values(self) -> tuple[int, ...]:
        """Enumerate a bounded integer support.

        Returns:
            Every supported integer in ascending order.

        Raises:
            ValueError: If the support is unbounded above.

        References:
            [REF-ROSS-2019, Chapter 4]. No equation is evaluated.

        """
        if self.upper is None:
            raise ValueError("an unbounded integer support cannot be enumerated")
        return tuple(range(self.lower, self.upper + 1))


@dataclass(frozen=True, slots=True, init=False)
class FiniteSupport(Generic[ValueT]):
    """A non-empty ordered collection of distinct hashable support values.

    Args:
        values: Distinct hashable values in deterministic display order.

    References:
        [REF-ROSS-2019, Chapters 4-5].

    """

    values: tuple[ValueT, ...]
    _members: frozenset[ValueT] = field(repr=False)

    def __init__(self, values: Iterable[ValueT]) -> None:
        """Materialize values and reject empty, repeated, or unhashable input.

        Args:
            values: Candidate support values.

        Raises:
            TypeError: If a value is unhashable.
            ValueError: If input is empty or repeats a value.

        References:
            [REF-ROSS-2019, Chapters 4-5]. No equation is evaluated.

        """
        ordered: list[ValueT] = []
        seen: set[ValueT] = set()
        for index, value in enumerate(values):
            try:
                duplicate = value in seen
            except TypeError as error:
                raise TypeError(f"values[{index}] must be hashable") from error
            if duplicate:
                raise ValueError(f"values[{index}] repeats an earlier value")
            seen.add(value)
            ordered.append(value)
        if not ordered:
            raise ValueError("finite support must contain at least one value")
        object.__setattr__(self, "values", tuple(ordered))
        object.__setattr__(self, "_members", frozenset(seen))

    def __iter__(self) -> Iterator[ValueT]:
        """Iterate through support values in declared order.

        Returns:
            Iterator over every support value exactly once.

        References:
            [REF-ROSS-2019, Chapters 4-5]. No equation is evaluated.

        """
        return iter(self.values)

    def __len__(self) -> int:
        """Return the number of distinct support values.

        References:
            [REF-ROSS-2019, Chapters 4-5]. No equation is evaluated.

        """
        return len(self.values)

    def contains(self, value: object) -> bool:
        """Return whether ``value`` belongs to the finite support.

        Args:
            value: Candidate support value.

        Returns:
            Membership result; unhashable values return ``False``.

        References:
            [REF-ROSS-2019, Chapters 4-5]. No equation is evaluated.

        """
        try:
            return value in self._members
        except TypeError:
            return False


@dataclass(frozen=True, slots=True)
class RealSupport:
    """An interval support over the extended real line.

    ``None`` denotes an unbounded side. This class describes support membership;
    distribution parameters remain the responsibility of constraint objects.

    Args:
        lower: Optional finite lower bound.
        upper: Optional finite upper bound.
        lower_closed: Whether a finite lower bound is included.
        upper_closed: Whether a finite upper bound is included.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3].

    """

    lower: float | None = None
    upper: float | None = None
    lower_closed: bool = False
    upper_closed: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize finite support endpoints.

        Raises:
            TypeError: If a supplied endpoint is not real.
            ValueError: If an endpoint is non-finite, bounds are reversed, or
                equal bounds exclude their sole possible value.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. No equation is evaluated.

        """
        from probstat.foundations.validation import require_finite_real

        lower = (
            None
            if self.lower is None
            else require_finite_real(self.lower, name="lower")
        )
        upper = (
            None
            if self.upper is None
            else require_finite_real(self.upper, name="upper")
        )
        if lower is not None and upper is not None:
            if lower > upper:
                raise ValueError("lower must not exceed upper")
            if lower == upper and not (self.lower_closed and self.upper_closed):
                raise ValueError("equal support bounds must both be closed")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, value: object) -> bool:
        """Return whether a finite real value belongs to this support.

        Args:
            value: Candidate real value.

        Returns:
            Membership result without coercing non-real values.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. No equation is evaluated.

        """
        from probstat.foundations.validation import require_finite_real

        try:
            numeric = require_finite_real(value, name="value")
        except (TypeError, ValueError):
            return False
        if self.lower is not None:
            if numeric < self.lower or (
                numeric == self.lower and not self.lower_closed
            ):
                return False
        if self.upper is not None:
            if numeric > self.upper or (
                numeric == self.upper and not self.upper_closed
            ):
                return False
        return True

