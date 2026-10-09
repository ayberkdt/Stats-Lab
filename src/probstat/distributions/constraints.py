"""Reusable immutable parameter constraints for probability distributions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class RealIntervalConstraint:
    """A finite-real interval used to validate scalar parameters.

    Args:
        lower: Optional finite lower bound.
        upper: Optional finite upper bound.
        lower_closed: Whether ``lower`` is included.
        upper_closed: Whether ``upper`` is included.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3].

    """

    lower: float | None = None
    upper: float | None = None
    lower_closed: bool = True
    upper_closed: bool = True

    def __post_init__(self) -> None:
        """Validate that the configured interval is non-empty.

        Raises:
            ValueError: If a bound is non-finite, reversed, or defines an empty
                interval.

        References:
            [REF-PYTHON-MATH]. This validates a software constraint and evaluates
            no probability equation.

        """
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
                raise ValueError("an interval with equal bounds must be closed")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, value: object) -> bool:
        """Return whether ``value`` satisfies the interval constraint.

        Args:
            value: Candidate scalar.

        Returns:
            ``False`` for invalid or out-of-interval values.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. No equation is evaluated.

        """
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

    def validate(self, value: object, *, name: str) -> float:
        """Return a valid scalar or raise a parameter-specific error.

        Args:
            value: Candidate scalar.
            name: Parameter name used in error messages.

        Returns:
            The validated finite float.

        Raises:
            TypeError: If ``value`` is not real.
            ValueError: If ``value`` is non-finite or outside the interval.

        References:
            [REF-CASELLA-BERGER-2002, Chapter 3]. No equation is evaluated.

        """
        numeric = require_finite_real(value, name=name)
        if not self.contains(numeric):
            raise ValueError(f"{name} lies outside {self!r}")
        return numeric


@dataclass(frozen=True, slots=True)
class IntegerIntervalConstraint:
    """An integer interval used to validate count parameters.

    Args:
        lower: Optional inclusive lower bound.
        upper: Optional inclusive upper bound.

    References:
        [REF-ROSS-2019, Chapters 1 and 4].

    """

    lower: int | None = None
    upper: int | None = None

    def __post_init__(self) -> None:
        """Validate integer bounds and their ordering.

        Raises:
            TypeError: If a bound is not an integer.
            ValueError: If the lower bound exceeds the upper bound.

        References:
            [REF-ROSS-2019, Chapters 1 and 4]. No equation is evaluated.

        """
        for name, bound in (("lower", self.lower), ("upper", self.upper)):
            if bound is not None and (
                isinstance(bound, bool) or not isinstance(bound, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError("lower must not exceed upper")

    def contains(self, value: object) -> bool:
        """Return whether ``value`` is an integer inside the interval.

        Args:
            value: Candidate integer.

        Returns:
            ``False`` for non-integers and out-of-range integers.

        References:
            [REF-ROSS-2019, Chapters 1 and 4]. No equation is evaluated.

        """
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if self.lower is not None and value < self.lower:
            return False
        return self.upper is None or value <= self.upper

    def validate(self, value: object, *, name: str) -> int:
        """Return a valid integer or raise a parameter-specific error.

        Args:
            value: Candidate integer.
            name: Parameter name used in error messages.

        Returns:
            The validated integer.

        Raises:
            TypeError: If ``value`` is not an integer.
            ValueError: If ``value`` lies outside the interval.

        References:
            [REF-ROSS-2019, Chapters 1 and 4]. No equation is evaluated.

        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not self.contains(value):
            raise ValueError(f"{name} lies outside {self!r}")
        return value


@dataclass(frozen=True, slots=True)
class ProbabilitySimplexConstraint:
    """A non-empty probability-vector constraint with unit-sum tolerance.

    Args:
        tolerance: Finite non-negative absolute tolerance for the unit sum.

    References:
        [EQ-DIST-CAT-001], [REF-ROSS-2019, Section 4.6].

    """

    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        """Validate and normalize the configured comparison tolerance.

        Raises:
            TypeError: If ``tolerance`` is not real.
            ValueError: If ``tolerance`` is negative or non-finite.

        References:
            [REF-PYTHON-MATH]. This validates a software constraint and evaluates
            no probability equation.

        """
        checked = require_non_negative_finite_real(
            self.tolerance, name="tolerance"
        )
        object.__setattr__(self, "tolerance", checked)

    def validate(self, values: Iterable[object], *, name: str) -> tuple[float, ...]:
        """Validate and return a normalized non-empty probability vector.

        Args:
            values: Candidate probabilities.
            name: Parameter name used in indexed error messages.

        Returns:
            Probabilities divided by their near-unit computed total.

        Raises:
            TypeError: If an element is not real.
            ValueError: If the vector is empty, an element is outside ``[0,1]``,
                or its total differs from one beyond ``tolerance``.

        Equation:
            [EQ-DIST-CAT-001].

        References:
            [REF-ROSS-2019, Section 4.6].

        """
        checked = tuple(
            require_probability(value, name=f"{name}[{index}]")
            for index, value in enumerate(values)
        )
        if not checked:
            raise ValueError(f"{name} must contain at least one probability")
        total = math.fsum(checked)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=self.tolerance):
            raise ValueError(f"{name} must sum to one; received {total!r}")
        return tuple(value / total for value in checked)


FINITE_REAL = RealIntervalConstraint()
PROBABILITY = RealIntervalConstraint(0.0, 1.0)
NON_NEGATIVE_FINITE = RealIntervalConstraint(0.0, None)
POSITIVE_FINITE = RealIntervalConstraint(0.0, None, lower_closed=False)
NON_NEGATIVE_INTEGER = IntegerIntervalConstraint(0, None)
POSITIVE_INTEGER = IntegerIntervalConstraint(1, None)
SIMPLEX = ProbabilitySimplexConstraint()

