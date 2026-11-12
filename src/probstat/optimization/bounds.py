"""Box constraints expressed as a smooth reparametrization.

:class:`BoxTransform` maps unconstrained coordinates onto a box so that the
unconstrained minimizers can handle simple bounds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.linalg import Vector

Bounds = Sequence[tuple[float | None, float | None]]


@dataclass(frozen=True, slots=True)
class BoxTransform:
    """Smooth map between unconstrained coordinates ``z`` and a box for ``x``.

    Per coordinate with bounds ``(l, u)``:

    * two finite bounds: ``x = l + (u - l)(1 + sin z)/2``;
    * lower bound only: ``x = l - 1 + sqrt(z^2 + 1)``;
    * upper bound only: ``x = u + 1 - sqrt(z^2 + 1)``;
    * no bound: ``x = z``.

    These are the parameter transformations of MINUIT. Every ``z`` maps
    inside the box, so the objective is never evaluated outside it, and a
    minimum on a bound becomes an ordinary stationary point in ``z`` (where
    ``dx/dz = 0``). The price is a distorted metric near the bounds; start
    points exactly on a bound are moved inward (see :meth:`to_inner`).

    Args:
        lower: Lower bound per coordinate (``-inf`` when absent).
        upper: Upper bound per coordinate (``inf`` when absent).

    References:
        [REF-JAMES-ROOS-1975].

    """

    lower: Vector
    upper: Vector

    @classmethod
    def from_bounds(cls, bounds: Bounds, size: int) -> BoxTransform:
        """Validate ``(low, high)`` pairs (``None`` or infinite for no bound).

        Args:
            bounds: One pair per coordinate.
            size: Expected number of coordinates.

        Returns:
            Validated transform.

        Raises:
            ValueError: If the length differs or a pair has ``low >= high``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        pairs = tuple(bounds)
        if len(pairs) != size:
            raise ValueError("bounds must have one (low, high) pair per coordinate")
        lower: list[float] = []
        upper: list[float] = []
        for index, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError("each bound must be a (low, high) pair")
            low_raw, high_raw = pair
            low = -math.inf if low_raw is None else float(low_raw)
            high = math.inf if high_raw is None else float(high_raw)
            if math.isnan(low) or math.isnan(high) or low == math.inf:
                raise ValueError(f"invalid bound for coordinate {index}")
            if high == -math.inf or not low < high:
                raise ValueError(f"bounds for coordinate {index} need low < high")
            lower.append(low)
            upper.append(high)
        return cls(tuple(lower), tuple(upper))

    def to_inner(self, x: Vector) -> Vector:
        """Map a point in the closed box to unconstrained coordinates.

        A coordinate on (or outside) a bound is first moved inside by
        ``1e-3`` of the interval width (two-sided) or ``1e-3 max(1, |b|)``
        (one-sided), because ``dx/dz`` vanishes on the bound.

        Args:
            x: Point inside or on the boundary of the box.

        Returns:
            Unconstrained coordinates ``z``.

        Raises:
            ValueError: If a coordinate lies outside the box.

        Equation:
            [EQ-OPT-BOX-001].

        References:
            [REF-JAMES-ROOS-1975].

        """
        inner: list[float] = []
        for value, low, high in zip(x, self.lower, self.upper, strict=True):
            if value < low or value > high:
                raise ValueError("x0 must lie inside the bounds")
            finite_low, finite_high = math.isfinite(low), math.isfinite(high)
            if finite_low and finite_high:
                margin = 1e-3 * (high - low)
                value = min(max(value, low + margin), high - margin)
                ratio = 2.0 * (value - low) / (high - low) - 1.0
                inner.append(math.asin(min(1.0, max(-1.0, ratio))))
            elif finite_low:
                value = max(value, low + 1e-3 * max(1.0, abs(low)))
                inner.append(math.sqrt((value - low + 1.0) ** 2 - 1.0))
            elif finite_high:
                value = min(value, high - 1e-3 * max(1.0, abs(high)))
                inner.append(math.sqrt((high - value + 1.0) ** 2 - 1.0))
            else:
                inner.append(value)
        return tuple(inner)

    def to_outer(self, z: Vector) -> Vector:
        """Map unconstrained coordinates to the box (clamped against rounding).

        Args:
            z: Unconstrained coordinates.

        Returns:
            Point inside the closed box.

        Equation:
            [EQ-OPT-BOX-001].

        References:
            [REF-JAMES-ROOS-1975].

        """
        outer: list[float] = []
        for value, low, high in zip(z, self.lower, self.upper, strict=True):
            finite_low, finite_high = math.isfinite(low), math.isfinite(high)
            if finite_low and finite_high:
                x = low + 0.5 * (high - low) * (1.0 + math.sin(value))
            elif finite_low:
                x = low - 1.0 + math.hypot(value, 1.0)
            elif finite_high:
                x = high + 1.0 - math.hypot(value, 1.0)
            else:
                x = value
            outer.append(min(max(x, low), high))
        return tuple(outer)

    def derivative(self, z: Vector) -> Vector:
        """Return the diagonal Jacobian ``dx_i/dz_i``.

        Args:
            z: Unconstrained coordinates.

        Returns:
            Derivative of each coordinate map.

        Equation:
            [EQ-OPT-BOX-001].

        References:
            [REF-JAMES-ROOS-1975].

        """
        result: list[float] = []
        for value, low, high in zip(z, self.lower, self.upper, strict=True):
            finite_low, finite_high = math.isfinite(low), math.isfinite(high)
            if finite_low and finite_high:
                result.append(0.5 * (high - low) * math.cos(value))
            elif finite_low:
                result.append(value / math.hypot(value, 1.0))
            elif finite_high:
                result.append(-value / math.hypot(value, 1.0))
            else:
                result.append(1.0)
        return tuple(result)
