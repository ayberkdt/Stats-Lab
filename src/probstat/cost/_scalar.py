"""Brent's derivative-free minimization of a unimodal function on an interval."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))
_SQRT_EPS = math.sqrt(2.220446049250313e-16)


@dataclass(frozen=True, slots=True)
class ScalarMinimum:
    """Result of a bracketed scalar minimization.

    Args:
        x: Approximate minimizer.
        value: Function value at ``x``.
        evaluations: Function evaluations.
        converged: Whether the bracket shrank below the tolerance.

    References:
        [REF-BRENT-1973, Chapter 5].

    """

    x: float
    value: float
    evaluations: int
    converged: bool


def brent_minimize(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> ScalarMinimum:
    """Minimize ``f`` on ``[lower, upper]`` by golden section with parabolic steps.

    For a unimodal function the result is within about
    ``sqrt(eps) |x| + tolerance`` of the minimizer; for a multimodal function
    it is a local minimum inside the bracket.

    Args:
        function: Finite-valued scalar function.
        lower: Left end of the bracket.
        upper: Right end of the bracket (greater than ``lower``).
        tolerance: Absolute tolerance on ``x``.
        max_iterations: Iteration limit.

    Returns:
        Minimum with evaluation count and convergence flag.

    Raises:
        ValueError: If the bracket is empty.

    Equation:
        [EQ-COST-BRENTMIN-001].

    References:
        [REF-BRENT-1973, Chapter 5].

    """
    a, b = float(lower), float(upper)
    if not a < b:
        raise ValueError("upper must exceed lower")
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = function(x)
    evaluations = 1
    d = e = 0.0
    for _ in range(max_iterations):
        middle = 0.5 * (a + b)
        tol1 = _SQRT_EPS * abs(x) + tolerance / 3.0
        tol2 = 2.0 * tol1
        if abs(x - middle) <= tol2 - 0.5 * (b - a):
            return ScalarMinimum(x, fx, evaluations, True)
        use_golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            previous = e
            e = d
            if abs(p) < abs(0.5 * q * previous) and q * (a - x) < p < q * (b - x):
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if x < middle else -tol1
                use_golden = False
        if use_golden:
            e = (b - x) if x < middle else (a - x)
            d = _GOLDEN * e
        u = x + (d if abs(d) >= tol1 else math.copysign(tol1, d))
        fu = function(u)
        evaluations += 1
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v in (x, w):
                v, fv = u, fu
    return ScalarMinimum(x, fx, evaluations, False)
