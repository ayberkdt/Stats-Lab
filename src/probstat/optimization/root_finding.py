"""Bracketed scalar root finding by Brent's method."""

from __future__ import annotations

import math
from collections.abc import Callable

from probstat.foundations.validation import require_finite_real

from ._common import _finite_value
from .results import RootResult


def brent_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> RootResult:
    """Find a root in a sign-changing bracket by Brent's method.

    Args:
        function: Continuous scalar function.
        lower: Bracket endpoint.
        upper: Bracket endpoint with ``f(lower) f(upper) <= 0``.
        tolerance: Absolute bracket-width tolerance.
        max_iterations: Maximum iterations.

    Returns:
        Root result.

    Raises:
        ValueError: If the bracket does not change sign.

    Equation:
        [EQ-OPT-BRENT-001].

    References:
        [REF-BRENT-1973, Chapter 4].

    """
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    fa = _finite_value(function(a), "f(lower)")
    fb = _finite_value(function(b), "f(upper)")
    evaluations = 2
    if fa * fb > 0.0:
        raise ValueError("root bracket must change sign")
    c, fc = b, fb
    d = e = b - a
    for iteration in range(max_iterations):
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tolerance_1 = 2.0 * 2.0**-52 * abs(b) + 0.5 * tolerance
        half_width = 0.5 * (c - b)
        if abs(half_width) <= tolerance_1 or fb == 0.0:
            return RootResult(b, fb, iteration, evaluations, True)
        if abs(e) >= tolerance_1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * half_width * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * half_width * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * half_width * q - abs(tolerance_1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = half_width
        else:
            d = e = half_width
        a, fa = b, fb
        b += d if abs(d) > tolerance_1 else math.copysign(tolerance_1, half_width)
        fb = _finite_value(function(b), "f(b)")
        evaluations += 1
    return RootResult(b, fb, max_iterations, evaluations, False)
