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


@dataclass(frozen=True, slots=True)
class DEQuadrature:
    """Tanh-sinh estimate with its convergence evidence.

    Args:
        value: Integral estimate.
        error_estimate: Absolute difference between the last two levels.
        evaluations: Number of integrand evaluations.
        converged: Whether two successive levels met the tolerance before
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
) -> DEQuadrature:
    """Integrate ``g(u)`` over ``(lower, upper)`` with ``0 <= lower < upper <= 1``.

    With ``u = lower + (upper - lower) x`` and
    ``x = 1 / (1 + exp(-pi sinh t))``, the integral becomes
    ``(upper - lower) int w(t) g(u(t)) dt`` with
    ``w(t) = pi cosh(t) x (1 - x)``, which the trapezoidal rule with step
    ``h = 2^-k`` integrates; each level halves ``h`` and reuses the previous
    nodes. Iteration stops when two successive levels agree within
    ``tolerance`` relative to the estimate (absolute when the estimate is
    zero) after at least three levels.

    Args:
        integrand: Function of ``(u, 1 - u)`` returning a finite real value.
        lower: Lower limit in ``[0, 1)``.
        upper: Upper limit in ``(lower, 1]``.
        tolerance: Relative stopping tolerance.
        max_level: Largest refinement level.
        absolute_tolerance: Absolute stopping floor, for integrals whose true
            value is zero or below the rounding noise of the integrand.

    Returns:
        Estimate, last-level difference, evaluation count, convergence flag,
        and the number of skipped near-end nodes.

    Notes:
        Singularities are resolved only at ``u = 0`` and ``u = 1``: the
        integrand receives ``u`` and ``1 - u``, so a singularity at an
        interior limit (``lower > 0`` or ``upper < 1``) is approached only to
        the spacing of floating-point numbers near that limit. Split such
        integrals at the singular point by a change of variable instead.
        Very strong end singularities (``u^-a`` with ``a`` close to one) can
        carry mass closer to the end than ``1e-300`` and return
        ``converged=False``.

    Raises:
        ValueError: If the limits are invalid or the integrand is non-finite.

    Equation:
        [EQ-VAL-QUAD-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974].

    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("limits must satisfy 0 <= lower < upper <= 1")
    width = upper - lower
    upper_complement = 1.0 - upper
    evaluations = 0
    skipped = 0

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
        uc = upper_complement + width * xc
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

    count = int(_T_MAX)
    total = math.fsum(term(float(j)) for j in range(-count, count + 1))
    estimate = width * total
    previous = estimate
    step = 1.0
    difference = math.inf
    converged = False
    for level in range(1, max_level + 1):
        step *= 0.5
        nodes = int(_T_MAX / step)
        odd = math.fsum(term(j * step) for j in range(-nodes, nodes + 1) if j % 2 != 0)
        total = 0.5 * total + step * odd
        estimate = width * total
        difference = abs(estimate - previous)
        previous = estimate
        allowed = max(tolerance * max(abs(estimate), 1e-300), absolute_tolerance)
        if level >= 3 and difference <= allowed:
            converged = True
            break
    return DEQuadrature(estimate, difference, evaluations, converged, skipped)


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
            f"{what}: tanh-sinh quadrature did not converge (last-level "
            f"difference {result.error_estimate:.3g}, {result.skipped_nodes} "
            "skipped near-end nodes); the value may be inaccurate",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return result.value


def tanh_sinh_pieces(
    integrand: Callable[[float, float], float],
    breakpoints: Sequence[float],
    *,
    tolerance: float = 1.0e-11,
    max_level: int = 9,
    absolute_tolerance: float = 0.0,
) -> DEQuadrature:
    """Integrate over ``(0, 1)`` split at interior breakpoints (kinks).

    Each piece is integrated by :func:`tanh_sinh`, which clusters nodes at its
    ends, so a kink placed at a breakpoint costs no accuracy. Values, error
    estimates, evaluations, and skipped nodes add up; the result is converged
    only if every piece is.

    Args:
        integrand: Function of ``(u, 1 - u)``.
        breakpoints: Interior points in ``(0, 1)``, in any order.
        tolerance: Relative stopping tolerance per piece.
        max_level: Largest refinement level per piece.
        absolute_tolerance: Absolute stopping floor per piece.

    Returns:
        Combined estimate.

    Raises:
        ValueError: If a breakpoint is outside ``(0, 1)``.

    Equation:
        [EQ-VAL-QUAD-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974].

    """
    points = sorted(set(breakpoints))
    if any(not 0.0 < b < 1.0 for b in points):
        raise ValueError("breakpoints must lie strictly inside (0, 1)")
    edges = [0.0, *points, 1.0]
    parts = [
        tanh_sinh(
            integrand,
            left,
            right,
            tolerance=tolerance,
            max_level=max_level,
            absolute_tolerance=absolute_tolerance,
        )
        for left, right in zip(edges[:-1], edges[1:], strict=True)
    ]
    return DEQuadrature(
        math.fsum(part.value for part in parts),
        math.fsum(part.error_estimate for part in parts),
        sum(part.evaluations for part in parts),
        all(part.converged for part in parts),
        sum(part.skipped_nodes for part in parts),
    )


def sign_change_points(
    function: Callable[[float, float], float], *, grid: int = 64
) -> tuple[float, ...]:
    """Locate sign changes of ``f(u)`` on ``(0, 1)`` by a grid plus bisection.

    Used to put the kinks of ``|Q_0(u) - Q_1(u)|`` (where two quantile
    functions cross) at breakpoints. The grid has ``grid`` equal cells plus
    geometric nodes ``2^-k`` and ``1 - 2^-k`` (``k = 8..60``) so that
    crossings deep in either tail are found; in the upper half the bisection
    works on the complement ``1 - u`` so it keeps relative precision there.
    Crossings closer together than the grid spacing can be missed.

    Args:
        function: Function of ``(u, 1 - u)``.
        grid: Number of equal grid cells.

    Returns:
        Located crossing points, increasing.

    References:
        [REF-TAKAHASI-MORI-1974]. No equation is evaluated.

    """
    pairs = {((k + 0.5) / grid, 1.0 - (k + 0.5) / grid) for k in range(grid)}
    for k in range(8, 61):
        tail = 2.0**-k
        pairs.add((tail, 1.0 - tail))
        pairs.add((1.0 - tail, tail))
    nodes = sorted(pairs)
    values = [function(u, c) for u, c in nodes]
    found: list[float] = []
    for (left, right), (f_left, f_right) in zip(
        zip(nodes[:-1], nodes[1:], strict=True),
        zip(values[:-1], values[1:], strict=True),
        strict=True,
    ):
        if f_left == 0.0:
            found.append(left[0])
            continue
        if f_left * f_right >= 0.0:
            continue
        low, high, f_low = left, right, f_left
        for _ in range(80):
            if low[0] > 0.5:
                complement = 0.5 * (low[1] + high[1])
                middle = (1.0 - complement, complement)
            else:
                point = 0.5 * (low[0] + high[0])
                middle = (point, 1.0 - point)
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
        found.append(0.5 * (low[0] + high[0]))
    return tuple(point for point in found if 0.0 < point < 1.0)
