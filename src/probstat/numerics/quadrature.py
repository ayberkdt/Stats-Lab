"""Dependency-free composite quadrature with explicit evaluation costs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)


@dataclass(frozen=True, slots=True)
class QuadratureResult:
    """A finite quadrature estimate and its deterministic work count.

    Args:
        value: Finite integral estimate.
        evaluations: Positive number of integrand evaluations.
        method: Non-empty quadrature method label.
        lower: Finite lower integration bound.
        upper: Finite upper bound greater than ``lower``.
        intervals: Positive number of equal subintervals.

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapters 2-3].

    """

    value: float
    evaluations: int
    method: str
    lower: float
    upper: float
    intervals: int

    def __post_init__(self) -> None:
        """Validate numerical result metadata.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a field has an invalid numerical domain.

        References:
            [REF-DAVIS-RABINOWITZ-1984]. No new equation is evaluated.

        """
        value = require_finite_real(self.value, name="value")
        lower = require_finite_real(self.lower, name="lower")
        upper = require_finite_real(self.upper, name="upper")
        evaluations = require_non_negative_integer(
            self.evaluations, name="evaluations"
        )
        intervals = require_non_negative_integer(self.intervals, name="intervals")
        if evaluations == 0:
            raise ValueError("evaluations must be positive")
        if intervals == 0:
            raise ValueError("intervals must be positive")
        if upper <= lower:
            raise ValueError("upper must be greater than lower")
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")
        if not self.method.strip():
            raise ValueError("method must not be empty")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "intervals", intervals)


def composite_trapezoid(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    intervals: int,
) -> QuadratureResult:
    """Approximate a finite-interval integral by composite trapezoids.

    Args:
        function: Scalar integrand returning finite real values.
        lower: Finite lower integration bound.
        upper: Finite upper bound strictly greater than ``lower``.
        intervals: Positive number of equal subintervals.

    Returns:
        Estimate with exactly ``intervals + 1`` function evaluations.

    Raises:
        TypeError: If arguments or integrand values have invalid types.
        ValueError: If bounds, interval count, or values have invalid domains.

    Equation:
        [EQ-NUM-TRAP-001].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    checked_lower, checked_upper, checked_intervals = _quadrature_inputs(
        function, lower, upper, intervals
    )
    width = (checked_upper - checked_lower) / checked_intervals
    endpoints = 0.5 * (
        _evaluate(function, checked_lower, index=0)
        + _evaluate(function, checked_upper, index=checked_intervals)
    )
    interior = math.fsum(
        _evaluate(function, checked_lower + index * width, index=index)
        for index in range(1, checked_intervals)
    )
    return QuadratureResult(
        width * (endpoints + interior),
        checked_intervals + 1,
        "composite trapezoid",
        checked_lower,
        checked_upper,
        checked_intervals,
    )


def composite_simpson(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    intervals: int,
) -> QuadratureResult:
    """Approximate a finite-interval integral by composite Simpson quadrature.

    Args:
        function: Scalar integrand returning finite real values.
        lower: Finite lower integration bound.
        upper: Finite upper bound strictly greater than ``lower``.
        intervals: Positive even number of equal subintervals.

    Returns:
        Estimate with exactly ``intervals + 1`` function evaluations.

    Raises:
        TypeError: If arguments or integrand values have invalid types.
        ValueError: If bounds, interval count, parity, or values are invalid.

    Equation:
        [EQ-NUM-SIMP-001].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    checked_lower, checked_upper, checked_intervals = _quadrature_inputs(
        function, lower, upper, intervals
    )
    if checked_intervals % 2:
        raise ValueError("Simpson quadrature requires an even interval count")
    width = (checked_upper - checked_lower) / checked_intervals
    weighted = [
        _evaluate(function, checked_lower, index=0),
        _evaluate(function, checked_upper, index=checked_intervals),
    ]
    for index in range(1, checked_intervals):
        coefficient = 4.0 if index % 2 else 2.0
        weighted.append(
            coefficient
            * _evaluate(function, checked_lower + index * width, index=index)
        )
    return QuadratureResult(
        width * math.fsum(weighted) / 3.0,
        checked_intervals + 1,
        "composite Simpson",
        checked_lower,
        checked_upper,
        checked_intervals,
    )


def _quadrature_inputs(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    intervals: int,
) -> tuple[float, float, int]:
    """Validate common finite-interval quadrature inputs.

    Args:
        function: Candidate callable integrand.
        lower: Candidate lower bound.
        upper: Candidate upper bound.
        intervals: Candidate positive interval count.

    Returns:
        Validated lower bound, upper bound, and interval count.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If bounds or interval count have invalid domains.

    References:
        [REF-DAVIS-RABINOWITZ-1984]. No quadrature equation is evaluated.

    """
    if not callable(function):
        raise TypeError("function must be callable")
    checked_lower = require_finite_real(lower, name="lower")
    checked_upper = require_finite_real(upper, name="upper")
    checked_intervals = require_non_negative_integer(intervals, name="intervals")
    if checked_upper <= checked_lower:
        raise ValueError("upper must be greater than lower")
    if checked_intervals == 0:
        raise ValueError("intervals must be positive")
    return checked_lower, checked_upper, checked_intervals


def _evaluate(
    function: Callable[[float], float], point: float, *, index: int
) -> float:
    """Evaluate an integrand and require a finite real result.

    Args:
        function: Validated callable integrand.
        point: Finite quadrature node.
        index: Node index used in error messages.

    Returns:
        Finite real integrand value.

    Raises:
        TypeError: If the result is not real.
        ValueError: If the result is non-finite.

    References:
        Project numerical validation policy. No equation is evaluated.

    """
    return require_finite_real(function(point), name=f"function(node {index})")
