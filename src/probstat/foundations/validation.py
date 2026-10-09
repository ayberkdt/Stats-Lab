"""Shared input validation with no upward package dependencies."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real


def require_non_negative_integer(value: int, *, name: str) -> int:
    """Validate and return a non-negative integer.

    Args:
        value: Candidate integer. Boolean values are rejected even though
            ``bool`` is an ``int`` subclass in Python.
        name: Parameter name used in an error message.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` is negative.

    References:
        [REF-PYTHON-DATAMODEL]. This is a software-domain validator and does
        not evaluate a mathematical equation.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def require_probability(value: float, *, name: str = "p") -> float:
    """Validate and return a finite probability in the closed unit interval.

    Args:
        value: Candidate real number.
        name: Parameter name used in an error message.

    Returns:
        The probability as a ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is non-finite or lies outside ``[0, 1]``.

    References:
        [EQ-PROB-001], [REF-KOLMOGOROV-1956].

    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def coerce_finite_sample(
    values: Iterable[Real], *, name: str = "values"
) -> tuple[float, ...]:
    """Materialize a non-empty iterable of finite real observations.

    Args:
        values: Observations to validate and materialize once.
        name: Parameter name used in an error message.

    Returns:
        An immutable tuple of finite floating-point observations.

    Raises:
        TypeError: If an observation is not a real number.
        ValueError: If the sample is empty or contains a non-finite value.

    References:
        [REF-NIST-EDA]. This validator establishes the finite-sample domain and
        does not itself evaluate a statistical equation.

    """
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{index}] must be a real number")
        observation = float(value)
        if not math.isfinite(observation):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(observation)
    if not result:
        raise ValueError(f"{name} must contain at least one observation")
    return tuple(result)

