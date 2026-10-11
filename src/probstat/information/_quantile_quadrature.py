"""Quadrature over a law's quantile function by the normal substitution.

An expectation ``E[g(X)] = int_0^1 g(Q(u)) du`` is rewritten with
``u = Phi(z)`` as ``int g(Q(Phi(z))) phi(z) dz`` and evaluated by composite
Simpson quadrature on ``[-z_max, z_max]``. The substitution removes the
endpoint singularities of unbounded quantile functions, and the lower-tail
probability ``Phi(-|z|)`` is formed with ``erfc`` so that upper-tail nodes call
``law.isf`` (when available) without cancellation in ``1 - u``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from probstat.foundations import require_finite_real

_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)
_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


def lower_tail(z: float) -> float:
    """Return the standard normal lower-tail probability ``Phi(-|z|)``.

    Args:
        z: Finite standard normal abscissa.

    Returns:
        ``erfc(|z|/sqrt 2)/2`` in ``(0, 1/2]``.

    Equation:
        [EQ-OT-W1D-001].

    References:
        [REF-PEYRE-CUTURI-2019, Remark 2.30].

    """
    return 0.5 * math.erfc(abs(z) * _INV_SQRT_TWO)


def quantile_at(law: object, z: float) -> float:
    """Return ``Q(Phi(z))`` using ``isf`` for upper-tail nodes when available.

    Args:
        law: Object with ``quantile(u)`` and optionally ``isf(u)``.
        z: Finite standard normal abscissa.

    Returns:
        The quantile of ``law`` at ``Phi(z)``.

    Raises:
        TypeError: If ``law`` has no callable ``quantile``.

    Equation:
        [EQ-OT-W1D-001].

    References:
        [REF-PEYRE-CUTURI-2019, Remark 2.30].

    """
    quantile = getattr(law, "quantile", None)
    if not callable(quantile):
        raise TypeError(f"{type(law).__name__} has no quantile method")
    tail = lower_tail(z)
    if z <= 0.0:
        return float(quantile(tail))
    isf = getattr(law, "isf", None)
    if callable(isf):
        return float(isf(tail))
    return float(quantile(1.0 - tail))


def simpson_normal_nodes(
    intervals: int, z_max: float
) -> tuple[tuple[float, float], ...]:
    """Return Simpson nodes ``z_k`` and weights ``w_k phi(z_k)`` on ``[-z_max, z_max]``.

    Args:
        intervals: Positive even number of subintervals.
        z_max: Positive truncation point; ``Phi(-8)`` is about ``6e-16``.

    Returns:
        Pairs ``(z_k, weight_k)`` whose weights sum to about one.

    Raises:
        TypeError: If ``intervals`` is not an integer.
        ValueError: If ``intervals`` is not positive and even or ``z_max`` is
            not positive.

    Equation:
        [EQ-OT-W1D-001].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    if isinstance(intervals, bool) or not isinstance(intervals, int):
        raise TypeError("intervals must be an integer")
    if intervals <= 0 or intervals % 2:
        raise ValueError("intervals must be a positive even integer")
    limit = require_finite_real(z_max, name="z_max")
    if not limit > 0.0:
        raise ValueError("z_max must be positive")
    width = 2.0 * limit / intervals
    nodes: list[tuple[float, float]] = []
    for index in range(intervals + 1):
        z = -limit + index * width
        if index in (0, intervals):
            simpson = 1.0
        elif index % 2:
            simpson = 4.0
        else:
            simpson = 2.0
        density = _INV_SQRT_TWO_PI * math.exp(-0.5 * z * z)
        nodes.append((z, simpson * width / 3.0 * density))
    return tuple(nodes)


def quantile_expectation(
    function: Callable[[float], float],
    *,
    intervals: int,
    z_max: float,
) -> float:
    """Return ``int function(z) phi(z) dz`` by composite Simpson quadrature.

    Args:
        function: Integrand in the normal variable ``z``; it typically
            evaluates quantile functions through :func:`quantile_at`.
        intervals: Positive even number of subintervals.
        z_max: Positive truncation point.

    Returns:
        The quadrature estimate.

    Raises:
        ValueError: If an integrand value is not finite.

    Equation:
        [EQ-OT-W1D-001].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    terms = []
    for z, weight in simpson_normal_nodes(intervals, z_max):
        value = require_finite_real(function(z), name="integrand")
        terms.append(weight * value)
    return math.fsum(terms)
