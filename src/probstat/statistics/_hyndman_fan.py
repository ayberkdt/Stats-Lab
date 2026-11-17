"""The nine Hyndman-Fan sample-quantile definitions evaluated on a sorted sample.

Every definition is a weighted average ``(1 - gamma) x_(j) + gamma x_(j+1)``
of two adjacent order statistics, with ``x_(0) = x_(1)`` and
``x_(n+1) = x_(n)`` at the ends. The position is computed in exact rational
arithmetic from the floating-point level ``p``. When ``p`` is the
floating-point value of a rational at which the position is an integer (for
example ``p = 0.07`` and ``n = 100``, where ``n p = 7`` although the product
``100 * 0.07`` rounds to ``7.000000000000001``), the position is taken to be
that integer, so the result depends on the decimal level the caller wrote
rather than on the rounding of a product.

References:
    [EQ-STAT-QUANT-002], [REF-HYNDMAN-FAN-1996].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

# Plotting-position constants (alpha, beta) of the continuous types 4-9:
# p_k = (k - alpha) / (n + 1 - alpha - beta), so that the position of p is
# h = alpha + p (n + 1 - alpha - beta).
_CONTINUOUS: dict[int, tuple[Fraction, Fraction]] = {
    4: (Fraction(0), Fraction(1)),
    5: (Fraction(1, 2), Fraction(1, 2)),
    6: (Fraction(0), Fraction(0)),
    7: (Fraction(1), Fraction(1)),
    8: (Fraction(1, 3), Fraction(1, 3)),
    9: (Fraction(3, 8), Fraction(3, 8)),
}


def _snapped_position(
    probability: float, offset: Fraction, scale: Fraction
) -> Fraction:
    """Return the exact position ``offset + p scale``, snapped to an integer.

    The position is snapped to the nearest integer ``k`` when ``p`` equals
    the floating-point value of ``(k - offset) / scale``, the level at which
    the position is exactly ``k``.

    Args:
        probability: Level ``p`` in ``[0, 1]``.
        offset: Exact additive constant of the position.
        scale: Exact positive multiplier of ``p``.

    Returns:
        Exact rational position.

    References:
        [EQ-STAT-QUANT-002], [REF-HYNDMAN-FAN-1996].

    """
    position = offset + Fraction(probability) * scale
    nearest = round(position)
    if float((nearest - offset) / scale) == probability:
        return Fraction(nearest)
    return position


def _order_statistic(ordered: Sequence[float], index: int) -> float:
    """Return ``x_(index)`` with the ends clamped to ``x_(1)`` and ``x_(n)``.

    Args:
        ordered: Ascending sample.
        index: One-based order-statistic index (any integer).

    Returns:
        The clamped order statistic.

    References:
        [EQ-STAT-QUANT-002], [REF-HYNDMAN-FAN-1996].

    """
    return ordered[min(max(index, 1), len(ordered)) - 1]


def hyndman_fan_quantile(
    ordered: Sequence[float], probability: float, kind: int
) -> float:
    """Return the Hyndman-Fan type ``kind`` quantile of a sorted sample.

    Discontinuous types (``j = floor(n p + m)``, ``g = n p + m - j``):
    type 1 (``m = 0``, ``gamma = 0`` if ``g = 0`` else 1, the inverse
    empirical CDF), type 2 (``m = 0``, ``gamma = 1/2`` if ``g = 0`` else 1,
    averaging at discontinuities), and type 3 (``m = -1/2``, ``gamma = 0`` if
    ``g = 0`` and ``j`` is even else 1, the nearest even order statistic).
    Continuous types interpolate ``gamma = g`` at the position
    ``h = alpha + p (n + 1 - alpha - beta)``, ``j = floor(h)``,
    ``g = h - j``: type 4 (``alpha = 0, beta = 1``), 5 (``1/2, 1/2``),
    6 (``0, 0``), 7 (``1, 1``), 8 (``1/3, 1/3``), and 9 (``3/8, 3/8``).

    Args:
        ordered: Non-empty ascending sample.
        probability: Level ``p`` in ``[0, 1]``.
        kind: Hyndman-Fan type from 1 to 9.

    Returns:
        Sample quantile.

    Raises:
        ValueError: If ``kind`` is not an integer from 1 to 9.

    Equation:
        [EQ-STAT-QUANT-002].

    References:
        [REF-HYNDMAN-FAN-1996].

    """
    n = len(ordered)
    if kind in (1, 2, 3):
        offset = Fraction(-1, 2) if kind == 3 else Fraction(0)
        position = _snapped_position(probability, offset, Fraction(n))
        j = math.floor(position)
        on_step = position == j
        lower = _order_statistic(ordered, j)
        upper = _order_statistic(ordered, j + 1)
        if kind == 1:
            return lower if on_step else upper
        if kind == 2:
            return 0.5 * (lower + upper) if on_step else upper
        return lower if on_step and j % 2 == 0 else upper
    if kind not in _CONTINUOUS:
        raise ValueError("method must be an integer from 1 to 9")
    alpha, beta = _CONTINUOUS[kind]
    position = _snapped_position(probability, alpha, n + 1 - alpha - beta)
    j = math.floor(position)
    fraction = float(position - j)
    lower = _order_statistic(ordered, j)
    upper = _order_statistic(ordered, j + 1)
    if fraction == 0.0 or lower == upper:
        return lower
    return lower + fraction * (upper - lower)
