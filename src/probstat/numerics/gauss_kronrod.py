"""Globally adaptive Gauss-Kronrod (G7-K15) quadrature with error estimates.

Each panel ``[a, b]`` is integrated by the 15-point Kronrod rule, whose seven
odd-indexed nodes are the 7-point Gauss-Legendre nodes, so the difference of
the two estimates costs no extra evaluations. QUADPACK turns that difference
into a local error estimate; the global algorithm (QAG without extrapolation)
repeatedly bisects the panel with the largest local error until the summed
error is below ``max(abs_tol, rel_tol |I|)`` or a subdivision limit is hit.
Infinite ranges use QUADPACK's QAGI substitution ``x = a + (1 - t)/t``.

The nodes and weights are the 33-digit constants of QUADPACK's ``dqk15.f``
(Netlib), originally tabulated by Kronrod; ``tests/test_adaptive_quadrature.py``
checks them in exact rational arithmetic (K15 exact through degree 22, G7
through degree 13, residuals below ``1e-32``).
"""

from __future__ import annotations

import heapq
import math
import sys
from collections.abc import Callable

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

from .integration_result import (
    ROUNDOFF_FACTOR,
    IntegrationResult,
    check_limit,
    check_tolerances,
)

# Kronrod abscissae on [-1, 1] (non-negative half, descending). Entries 1, 3,
# 5 and 7 (0-based) are the Gauss-Legendre 7-point nodes; entry 7 is zero.
K15_NODES = (
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,
    0.586087235467691130294144838258730,
    0.405845151377397166906606412076961,
    0.207784955007898467600689403773245,
    0.0,
)
K15_WEIGHTS = (
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
)
# Gauss weights for K15_NODES[1], [3], [5] and the centre.
G7_WEIGHTS = (
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
)
_TINY = sys.float_info.min
_EPS = sys.float_info.epsilon


def kronrod_panel(
    function: Callable[[float], float], lower: float, upper: float
) -> tuple[float, float, float]:
    """Apply the G7-K15 pair to one finite panel.

    The local error estimate is QUADPACK's
    ``resasc min(1, (200 |K15 - G7| / resasc)^1.5)``, floored at
    ``50 eps int |f|``, where ``resasc`` approximates
    ``int |f - mean(f)|`` over the panel.

    Args:
        function: Integrand returning finite reals.
        lower: Finite left end.
        upper: Finite right end.

    Returns:
        ``(K15 estimate, local error estimate, K15 estimate of int |f|)``.

    Equation:
        [EQ-NUM-GK15-001].

    References:
        [REF-PIESSENS-QUADPACK-1983, Section 2.2 and routine QK15],
        [REF-KRONROD-1965].

    """
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    f_centre = function(centre)
    gauss = f_centre * G7_WEIGHTS[3]
    kronrod = f_centre * K15_WEIGHTS[7]
    absolute = abs(kronrod)
    left: list[float] = []
    right: list[float] = []
    for index in range(7):
        offset = half * K15_NODES[index]
        f_left = function(centre - offset)
        f_right = function(centre + offset)
        left.append(f_left)
        right.append(f_right)
        pair = f_left + f_right
        kronrod += K15_WEIGHTS[index] * pair
        absolute += K15_WEIGHTS[index] * (abs(f_left) + abs(f_right))
        if index % 2 == 1:
            gauss += G7_WEIGHTS[index // 2] * pair
    mean = 0.5 * kronrod
    spread = K15_WEIGHTS[7] * abs(f_centre - mean) + math.fsum(
        K15_WEIGHTS[index] * (abs(left[index] - mean) + abs(right[index] - mean))
        for index in range(7)
    )
    width = abs(half)
    result = kronrod * half
    absolute *= width
    spread *= width
    error = abs((kronrod - gauss) * half)
    if spread != 0.0 and error != 0.0:
        error = spread * min(1.0, (200.0 * error / spread) ** 1.5)
    if absolute > _TINY / ROUNDOFF_FACTOR:
        error = max(ROUNDOFF_FACTOR * absolute, error)
    return result, error, absolute


def gauss_kronrod(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 1.0e-10,
    limit: int = 200,
) -> IntegrationResult:
    """Integrate ``f`` over ``[lower, upper]`` by adaptive G7-K15 bisection.

    Panels are kept in a max-heap on their local error; the worst panel is
    bisected until the summed error estimate is at most
    ``max(abs_tol, rel_tol |I|)`` (converged), the number of panels reaches
    ``limit``, or the worst panel is too narrow to bisect in floating point
    (both reported with ``converged=False``). There is no epsilon-algorithm
    extrapolation (QAGS) and no QUADPACK roundoff detection, so an endpoint
    singularity such as ``x^-1/2`` converges geometrically in the number of
    bisections rather than faster.

    Infinite limits use ``x = a + (1 - t)/t`` on ``t in (0, 1]`` (QAGI), with
    ``f(x) + f(-x)`` folded for the whole line; nodes never reach ``t = 0``.

    Args:
        function: Integrand returning finite reals at every node.
        lower: Lower limit (``-inf`` allowed).
        upper: Upper limit greater than ``lower`` (``inf`` allowed).
        abs_tol: Absolute error target.
        rel_tol: Relative error target (at least ``50 eps`` if ``abs_tol``
            is zero).
        limit: Maximum number of panels (at least one; one means a single
            G7-K15 application).

    Returns:
        Estimate, summed error estimate, evaluations, panels, and flag.

    Raises:
        TypeError: If an argument or an integrand value has an invalid type.
        ValueError: If the limits or tolerances are invalid, or the integrand
            returns a non-finite value.

    Equation:
        [EQ-NUM-GK15-001], [EQ-NUM-QAG-001].

    References:
        [REF-PIESSENS-QUADPACK-1983, Sections 2.2-2.3 and routines QAG,
        QAGI], [REF-KRONROD-1965], [REF-LAURIE-1997].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    a = check_limit(lower)
    b = check_limit(upper)
    if not a < b:
        raise ValueError("upper must be greater than lower")
    absolute_target, relative_target = check_tolerances(abs_tol, rel_tol)
    panel_limit = require_non_negative_integer(limit, name="limit")
    if panel_limit < 1:
        raise ValueError("limit must be at least 1")
    evaluations = 0

    def value_at(x: float) -> float:
        """Evaluate the user integrand and require a finite real value.

        References:
            Project numerical validation policy. No equation is evaluated.

        """
        nonlocal evaluations
        evaluations += 1
        return require_finite_real(function(x), name=f"function({x!r})")

    integrand: Callable[[float], float]
    if math.isfinite(a) and math.isfinite(b):
        integrand = value_at
        start, stop = a, b
    else:

        def integrand(t: float) -> float:
            """Return the QAGI-transformed integrand at ``t in (0, 1)``.

            References:
                [REF-PIESSENS-QUADPACK-1983, routine QAGI].

            """
            distance = (1.0 - t) / t
            if math.isfinite(a):
                total = value_at(a + distance)
            elif math.isfinite(b):
                total = value_at(b - distance)
            else:
                total = value_at(distance) + value_at(-distance)
            weighted = total / (t * t)
            if not math.isfinite(weighted):
                raise ValueError("transformed integrand overflowed near t = 0")
            return weighted

        start, stop = 0.0, 1.0

    first_value, first_error, _ = kronrod_panel(integrand, start, stop)
    heap: list[tuple[float, float, float, float]] = [
        (-first_error, start, stop, first_value)
    ]
    area = first_value
    error_sum = first_error
    while (
        error_sum > max(absolute_target, relative_target * abs(area))
        and len(heap) < panel_limit
    ):
        negative_error, left, right, value = heapq.heappop(heap)
        middle = 0.5 * (left + right)
        too_narrow = max(abs(left), abs(right)) <= (1.0 + 100.0 * _EPS) * (
            abs(middle) + 1000.0 * _TINY
        )
        if too_narrow or not left < middle < right:
            heapq.heappush(heap, (negative_error, left, right, value))
            break
        value_1, error_1, _ = kronrod_panel(integrand, left, middle)
        value_2, error_2, _ = kronrod_panel(integrand, middle, right)
        area += value_1 + value_2 - value
        error_sum += error_1 + error_2 + negative_error
        heapq.heappush(heap, (-error_1, left, middle, value_1))
        heapq.heappush(heap, (-error_2, middle, right, value_2))
    estimate = math.fsum(panel[3] for panel in heap)
    error_estimate = math.fsum(-panel[0] for panel in heap)
    converged = error_estimate <= max(absolute_target, relative_target * abs(estimate))
    intervals = tuple(sorted((panel[1], panel[2]) for panel in heap))
    return IntegrationResult(
        value=estimate,
        error_estimate=error_estimate,
        evaluations=evaluations,
        converged=converged,
        rule="gauss-kronrod-15",
        intervals=intervals,
        levels=0,
    )
