"""Double-exponential (tanh-sinh) quadrature on sub-intervals of ``(0, 1)``.

Probability-scale integrals such as ``int_0^1 |Q_1(u) - Q_2(u)|^p du`` or
the quantile representation of the CRPS have integrands that are unbounded
(but integrable) at ``u -> 0`` and ``u -> 1``. The tanh-sinh substitution
clusters nodes double-exponentially near both ends, so such integrals
converge quickly without the integrand ever being evaluated at an endpoint.
The integrand receives both ``u`` and its complement ``1 - u``, computed
without cancellation, so upper-tail quantiles can be taken from an inverse
survival function.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .integration_result import truncation_term

# Nodes run until min(x, 1 - x) reaches about 1e-300 on the unit scale,
# i.e. pi sinh(t) <= ln(1e300). Stopping at 1e-15 (as earlier versions did)
# is harmless for bounded integrands but drops a mass of order
# eps^(1 - a) / (1 - a) for an end singularity u^-a, which reached several
# per cent for |Q_t(u)|^2 with Student-t quantiles of 2.2 degrees of freedom.
_T_MAX = math.asinh(math.log(1.0e300) / math.pi)
# Inside this range the integrand must be finite; beyond it (u closer than
# 1e-15 to an end) an overflowing or failing evaluation contributes nothing,
# because the nodes there carry weight below 1e-15 times the integrand.
_T_SAFE = math.asinh(math.log(1.0e15) / math.pi)
_EPS = 2.220446049250313e-16


@dataclass(frozen=True, slots=True)
class DEQuadrature:
    """Tanh-sinh estimate with its convergence evidence.

    Args:
        value: Integral estimate.
        error_estimate: Absolute difference between the last two levels
            plus the estimated mass beyond the outermost nodes.
        evaluations: Number of integrand evaluations.
        converged: Whether the error estimate met the tolerance before
            ``max_level``; ``False`` means the estimate (and its error
            estimate) should not be trusted.
        skipped_nodes: Nodes closer than ``1e-15`` to an end whose evaluation
            failed or overflowed and was counted as zero. A non-zero count
            with a strongly singular integrand can hide real mass.

    References:
        [REF-TAKAHASI-MORI-1974].

    """

    value: float
    error_estimate: float
    evaluations: int
    converged: bool = True
    skipped_nodes: int = 0


def tanh_sinh(
    integrand: Callable[[float, float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = 1.0e-11,
    max_level: int = 9,
    absolute_tolerance: float = 0.0,
    lower_complement: float | None = None,
    upper_complement: float | None = None,
) -> DEQuadrature:
    """Integrate ``g(u)`` over ``(lower, upper)`` with ``0 <= lower < upper <= 1``.

    With ``u = lower + (upper - lower) x`` and
    ``x = 1 / (1 + exp(-pi sinh t))``, the integral becomes
    ``(upper - lower) int w(t) g(u(t)) dt`` with
    ``w(t) = pi cosh(t) x (1 - x)``, which the trapezoidal rule with step
    ``h = 2^-k`` integrates; each level halves ``h`` and reuses the previous
    nodes. The error estimate is the difference of the last two levels plus
    a truncation term for the mass beyond the outermost nodes (about
    ``1e-300`` from the ends), estimated from ``|w g|`` there and its decay
    rate by :func:`probstat.numerics.integration_result.truncation_term`.
    Iteration stops when the error estimate is within ``tolerance``
    relative to the estimate (absolute when the estimate is zero) after at
    least three levels.

    Limits near one lose their distance to one when rounded (``1 - 6e-16``
    is stored as ``1 - 6.66e-16``). ``lower_complement`` and
    ``upper_complement`` give ``1 - lower`` and ``1 - upper`` exactly; the
    integrand then receives complements measured from them, so a kink at a
    complement-scale breakpoint sits exactly on the piece boundary.

    Args:
        integrand: Function of ``(u, 1 - u)`` returning a finite real value.
        lower: Lower limit in ``[0, 1)``.
        upper: Upper limit in ``(lower, 1]``.
        tolerance: Relative stopping tolerance.
        max_level: Largest refinement level.
        absolute_tolerance: Absolute stopping floor, for integrals whose true
            value is zero or below the rounding noise of the integrand.
        lower_complement: Exact ``1 - lower`` (default: computed). With
            complements given, ``lower == upper`` is allowed when the
            complements differ.
        upper_complement: Exact ``1 - upper`` (default: computed).

    Returns:
        Estimate, error estimate, evaluation count, convergence flag, and the
        number of skipped near-end nodes.

    Notes:
        Singularities are resolved only at ``u = 0`` and ``u = 1`` (or at a
        limit whose complement is given): the integrand receives ``u`` and
        ``1 - u``, so a singularity at another interior limit is approached
        only to the spacing of floating-point numbers near that limit.
        Split such integrals at the singular point by a change of variable
        instead. Very strong end singularities (``u^-a`` with ``a`` close to
        one) carry mass closer to the end than ``1e-300``; the truncation
        term then keeps the error estimate large and such cases return
        ``converged=False``.

    Raises:
        ValueError: If the limits are invalid or the integrand is non-finite.

    Equation:
        [EQ-VAL-QUAD-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974].

    """
    lower_c = 1.0 - lower if lower_complement is None else float(lower_complement)
    upper_c = 1.0 - upper if upper_complement is None else float(upper_complement)
    if lower_complement is None and upper_complement is None:
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("limits must satisfy 0 <= lower < upper <= 1")
    elif not (
        0.0 <= lower <= upper <= 1.0
        and 0.0 <= upper_c < lower_c <= 1.0
        and abs(lower + lower_c - 1.0) <= 4.0 * _EPS
        and abs(upper + upper_c - 1.0) <= 4.0 * _EPS
    ):
        raise ValueError(
            "limits must satisfy 0 <= lower <= upper <= 1 with consistent "
            "complements 1 - upper < 1 - lower"
        )
    # In the upper half the complements carry the precision.
    width = lower_c - upper_c if lower >= 0.5 else upper - lower
    evaluations = 0
    skipped = 0
    outer: dict[float, float] = {}

    def term(t: float) -> float:
        """Return the weighted integrand at one trapezoid node.

        References:
            [EQ-VAL-QUAD-DE-001].

        """
        nonlocal evaluations, skipped
        s = math.pi * math.sinh(t)
        if s >= 0.0:
            e = math.exp(-s)
            x, xc = 1.0 / (1.0 + e), e / (1.0 + e)
        else:
            e = math.exp(s)
            x, xc = e / (1.0 + e), 1.0 / (1.0 + e)
        u = lower + width * x
        uc = upper_c + width * xc
        evaluations += 1
        if abs(t) <= _T_SAFE:
            value = integrand(u, uc)
            if not math.isfinite(value):
                raise ValueError("integrand returned a non-finite value")
        else:
            try:
                value = integrand(u, uc)
            except (ArithmeticError, ValueError):
                skipped += 1
                return 0.0
            if not math.isfinite(value):
                skipped += 1
                return 0.0
        weighted = math.pi * math.cosh(t) * x * xc * value
        return weighted if math.isfinite(weighted) else 0.0

    def level_sum(points: list[float]) -> float:
        """Return the sum of terms, keeping the moduli near ``t_max``.

        References:
            [EQ-VAL-QUAD-DE-001].

        """
        terms = [term(t) for t in points]
        for t, value in zip(points, terms, strict=True):
            if abs(t) >= _T_MAX - 2.0:
                outer[t] = abs(value)
        return math.fsum(terms)

    count = int(_T_MAX)
    total = level_sum([float(j) for j in range(-count, count + 1)])
    estimate = width * total
    previous = estimate
    step = 1.0
    error = math.inf
    converged = False
    for level in range(1, max_level + 1):
        step *= 0.5
        nodes = int(_T_MAX / step)
        odd = level_sum([j * step for j in range(-nodes, nodes + 1) if j % 2 != 0])
        total = 0.5 * total + step * odd
        estimate = width * total
        for t in [t for t in outer if abs(t) < _T_MAX - 2.0 * step]:
            del outer[t]
        truncation = width * truncation_term(outer, step)
        error = abs(estimate - previous) + truncation
        previous = estimate
        allowed = max(tolerance * max(abs(estimate), 1e-300), absolute_tolerance)
        if level >= 3 and error <= allowed:
            converged = True
            break
    return DEQuadrature(estimate, error, evaluations, converged, skipped)


def warn_if_unreliable(
    result: DEQuadrature, what: str, *, stacklevel: int = 3
) -> float:
    """Return ``result.value``, warning when convergence was not reached.

    Args:
        result: Tanh-sinh estimate.
        what: Short description of the quantity, used in the warning.
        stacklevel: Warning stack level (the default points at the caller of
            the function that called this helper).

    Returns:
        The estimate.

    Warns:
        RuntimeWarning: If the levels did not converge, or if near-end nodes
            were skipped while the integral did not converge.

    References:
        [REF-TAKAHASI-MORI-1974]. No equation is evaluated.

    """
    if not result.converged:
        warnings.warn(
            f"{what}: tanh-sinh quadrature did not converge (error estimate "
            f"{result.error_estimate:.3g}, {result.skipped_nodes} "
            "skipped near-end nodes); the value may be inaccurate",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return result.value


Breakpoint = float | tuple[float, float]
"""A breakpoint ``u`` or a pair ``(u, 1 - u)`` with an exact complement."""


def _canonical_breakpoint(point: Breakpoint) -> tuple[float, float]:
    """Return ``(u, 1 - u)`` keeping the precise coordinate of each half.

    Below one half ``u`` is precise and the complement is derived from it;
    from one half on the complement is precise and ``u`` is derived.

    References:
        [EQ-VAL-QUAD-DE-001]. No equation is evaluated.

    """
    if isinstance(point, tuple):
        u, complement = float(point[0]), float(point[1])
    else:
        u = float(point)
        complement = 1.0 - u
    if u < 0.5:
        return u, 1.0 - u
    return 1.0 - complement, complement


def tanh_sinh_pieces(
    integrand: Callable[[float, float], float],
    breakpoints: Sequence[Breakpoint],
    *,
    tolerance: float = 1.0e-11,
    max_level: int = 9,
    absolute_tolerance: float = 0.0,
) -> DEQuadrature:
    """Integrate over ``(0, 1)`` split at interior breakpoints (kinks).

    Each piece is integrated by :func:`tanh_sinh`, which clusters nodes at its
    ends, so a kink placed at a breakpoint costs no accuracy. Values, error
    estimates, evaluations, and skipped nodes add up. Convergence is judged
    on the sum, ``sum errors <= max(tolerance |sum values|,
    absolute_tolerance)``: a tiny piece whose own relative tolerance is out
    of reach (a value of ``1e-11`` next to pieces of ``1e-3``) does not make
    an accurate total unreliable.

    Args:
        integrand: Function of ``(u, 1 - u)``.
        breakpoints: Interior points in ``(0, 1)``, in any order, each a
            float ``u`` or a pair ``(u, 1 - u)`` whose complement is exact
            (needed for kinks within ``1e-15`` of one; see
            :func:`sign_change_points`).
        tolerance: Relative stopping tolerance per piece and for the sum.
        max_level: Largest refinement level per piece.
        absolute_tolerance: Absolute stopping floor per piece and for the
            sum.

    Returns:
        Combined estimate.

    Raises:
        ValueError: If a breakpoint is outside ``(0, 1)``.

    Equation:
        [EQ-VAL-QUAD-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974].

    """
    points = sorted(
        {_canonical_breakpoint(b) for b in breakpoints},
        key=lambda pair: (pair[0], -pair[1]),
    )
    if any(not (0.0 < u and 0.0 < c and u <= 1.0 and c <= 1.0) for u, c in points):
        raise ValueError("breakpoints must lie strictly inside (0, 1)")
    edges = [(0.0, 1.0), *points, (1.0, 0.0)]
    parts = [
        tanh_sinh(
            integrand,
            left[0],
            right[0],
            tolerance=tolerance,
            max_level=max_level,
            absolute_tolerance=absolute_tolerance,
            lower_complement=left[1],
            upper_complement=right[1],
        )
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]
    value = math.fsum(part.value for part in parts)
    error = math.fsum(part.error_estimate for part in parts)
    return DEQuadrature(
        value,
        error,
        sum(part.evaluations for part in parts),
        error <= max(tolerance * abs(value), absolute_tolerance),
        sum(part.skipped_nodes for part in parts),
    )


def sign_change_points(
    function: Callable[[float, float], float],
    *,
    grid: int = 64,
    complements: bool = False,
) -> tuple[Breakpoint, ...]:
    """Locate sign changes of ``f(u)`` on ``(0, 1)`` by a grid plus bisection.

    Used to put the kinks of ``|Q_0(u) - Q_1(u)|`` (where two quantile
    functions cross) at breakpoints. The grid has ``grid`` equal cells plus
    geometric nodes ``2^-k`` and ``1 - 2^-k`` (``k = 8..60``) so that
    crossings deep in either tail are found; in the upper half the bisection
    works on the complement ``1 - u`` so it keeps relative precision there.
    Besides sign changes, the points where ``f`` leaves or reaches exact
    zero are located (by bisection on ``f == 0``): there ``|f|`` jumps or has
    a kink, as for step quantile functions of discrete laws that coincide on
    an interval. A point where ``f`` is exactly zero is returned exactly.
    Crossings closer together than the grid spacing can be missed.

    Args:
        function: Function of ``(u, 1 - u)``.
        grid: Number of equal grid cells.
        complements: Return pairs ``(u, 1 - u)`` whose complement keeps full
            precision near one (``1 - 6.2e-16`` is not representable as a
            float ``u``), as accepted by :func:`tanh_sinh_pieces`. The
            default returns the floats ``u``.

    Returns:
        Located points, increasing (floats, or pairs with ``complements``).

    References:
        [REF-TAKAHASI-MORI-1974]. No equation is evaluated.

    """
    pairs = {((k + 0.5) / grid, 1.0 - (k + 0.5) / grid) for k in range(grid)}
    for k in range(8, 61):
        tail = 2.0**-k
        pairs.add((tail, 1.0 - tail))
        pairs.add((1.0 - tail, tail))
    nodes = sorted(pairs, key=lambda pair: (pair[0], -pair[1]))
    values = [function(u, c) for u, c in nodes]

    def midpoint(
        low: tuple[float, float], high: tuple[float, float]
    ) -> tuple[float, float]:
        """Return the bisection point, on the complement in the upper half.

        References:
            Bisection; no equation is evaluated.

        """
        if low[0] > 0.5:
            complement = 0.5 * (low[1] + high[1])
            return 1.0 - complement, complement
        point = 0.5 * (low[0] + high[0])
        return point, 1.0 - point

    found: list[tuple[float, float]] = []
    for (left, right), (f_left, f_right) in zip(
        zip(nodes[:-1], nodes[1:], strict=True),
        zip(values[:-1], values[1:], strict=True),
        strict=True,
    ):
        if (f_left == 0.0) != (f_right == 0.0):
            # f leaves (or reaches) exact zero inside the cell: keep the
            # zero end of the final bracket, so an isolated zero is exact.
            zero_left = f_left == 0.0
            low, high = left, right
            for _ in range(200):
                middle = midpoint(low, high)
                if middle in (low, high):
                    break
                if (function(*middle) == 0.0) == zero_left:
                    low = middle
                else:
                    high = middle
            found.append(low if zero_left else high)
            continue
        if f_left * f_right >= 0.0:
            continue
        low, high, f_low = left, right, f_left
        for _ in range(200):
            middle = midpoint(low, high)
            if middle in (low, high):
                break
            f_middle = function(*middle)
            if f_middle == 0.0:
                low = high = middle
                break
            if (f_middle > 0.0) == (f_low > 0.0):
                low, f_low = middle, f_middle
            else:
                high = middle
        found.append(midpoint(low, high) if low != high else low)
    located = sorted(
        {pair for pair in found if pair[0] > 0.0 and pair[1] > 0.0},
        key=lambda pair: (pair[0], -pair[1]),
    )
    if complements:
        return tuple(located)
    return tuple(sorted({u for u, _ in located if u < 1.0}))
