"""Shared validation and numeric helpers of the generative package.

The flow layers and energy models evaluate one program on plain floats and
on reverse-mode tape variables, so the helpers here are written for any
numeric kind: :func:`primal` reads the innermost float of a variable, and
:func:`standard_normal_log_density` is evaluated with operators only.

References:
    [REF-PROBSTAT-VALIDATION] and [REF-GRIEWANK-WALTHER-2008, Chapter 3].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..foundations.random_streams import RandomStream
from ..numerics import Dual, ReverseVariable

LOG_TWO_PI = math.log(2.0 * math.pi)
Point = tuple[float, ...]


def primal(value: Any) -> float:
    """Return the innermost float of a float, dual, or tape variable.

    Args:
        value: Number, :class:`~probstat.numerics.ReverseVariable`, or
            :class:`~probstat.numerics.Dual`.

    Returns:
        The underlying real value.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    while isinstance(value, ReverseVariable | Dual):
        value = value.value
    return float(value)


def as_points(data: object, *, name: str = "data", minimum: int = 1) -> list[Point]:
    """Validate a data set of points and return it as tuples of floats.

    A sequence of numbers is read as one-dimensional points.

    Args:
        data: Sequence of points (sequences of reals) or of reals.
        name: Argument name used in error messages.
        minimum: Smallest admissible number of points.

    Returns:
        The points as a list of float tuples of one common dimension.

    Raises:
        TypeError: If ``data`` is not a sequence of reals or of sequences.
        ValueError: For too few points, mixed dimensions, an empty point, or
            a non-finite coordinate.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if isinstance(data, str | bytes) or not isinstance(data, Sequence):
        raise TypeError(f"{name} must be a sequence of points")
    points: list[Point] = []
    for index, item in enumerate(data):
        if isinstance(item, Sequence) and not isinstance(item, str | bytes):
            coordinates = tuple(_finite(value, name) for value in item)
        else:
            coordinates = (_finite(item, name),)
        if not coordinates:
            raise ValueError(f"{name}[{index}] is an empty point")
        if points and len(coordinates) != len(points[0]):
            raise ValueError(f"{name} points must share one dimension")
        points.append(coordinates)
    if len(points) < minimum:
        raise ValueError(f"{name} needs at least {minimum} points")
    return points


def _finite(value: object, name: str) -> float:
    """Return one coordinate as a finite float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} coordinates must be real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} coordinates must be finite")
    return result


def as_point(value: object, dimension: int, *, name: str = "x") -> Point:
    """Validate one point of a known dimension.

    Args:
        value: A sequence of reals, or a real number when ``dimension == 1``.
        dimension: Required length.
        name: Argument name used in error messages.

    Returns:
        The point as a float tuple.

    Raises:
        TypeError: If ``value`` is not a point.
        ValueError: If the length is wrong or a coordinate is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    point = as_points([value], name=name)[0]
    if len(point) != dimension:
        raise ValueError(f"{name} must have dimension {dimension}")
    return point


def positive_integer(value: object, *, name: str) -> int:
    """Validate a positive integer setting.

    Raises:
        ValueError: If ``value`` is not an integer of at least one.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_real(value: object, *, name: str) -> float:
    """Validate a positive finite real setting.

    Raises:
        ValueError: If ``value`` is not a positive finite real number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def resolve(seed: int | RandomStream) -> RandomStream:
    """Return ``seed`` as a :class:`RandomStream`.

    Raises:
        TypeError: If ``seed`` is neither an integer nor a stream.

    References:
        [REF-STEELE-LEA-FLOOD-2014]. No equation is evaluated.

    """
    if isinstance(seed, RandomStream):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or a RandomStream")
    return RandomStream(seed)


def standard_normal_log_density(z: Sequence[Any]) -> Any:
    """Return ``log N(z; 0, I)`` for floats or tape variables.

    Args:
        z: Point of any numeric kind.

    Returns:
        ``-|z|^2 / 2 - d log(2 pi) / 2``.

    References:
        [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Section 2.1].

    """
    total: Any = 0.0
    for value in z:
        total = total + value * value
    return -0.5 * total - 0.5 * len(z) * LOG_TWO_PI


def mean_and_standard_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample mean and its standard error ``s / sqrt(n)``.

    Args:
        values: At least one finite value; one value has standard error 0.

    Returns:
        ``(mean, standard error)``.

    References:
        [EQ-GEN-DIAG-002], [REF-OWEN-2013, Chapter 2].

    """
    n = len(values)
    mean = math.fsum(values) / n
    if n < 2:
        return mean, 0.0
    variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance / n)
