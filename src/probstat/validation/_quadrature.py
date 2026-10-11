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
from collections.abc import Callable
from dataclasses import dataclass

# Nodes are kept while min(x, 1 - x) >= 1e-15 on the unit scale, i.e.
# pi sinh(t) <= ln(1e15); the omitted end masses are below 1e-15 each.
_T_MAX = math.asinh(math.log(1.0e15) / math.pi)


@dataclass(frozen=True, slots=True)
class DEQuadrature:
    """Tanh-sinh estimate with its convergence evidence.

    Args:
        value: Integral estimate.
        error_estimate: Absolute difference between the last two levels.
        evaluations: Number of integrand evaluations.

    References:
        [REF-TAKAHASI-MORI-1974].

    """

    value: float
    error_estimate: float
    evaluations: int


def tanh_sinh(
    integrand: Callable[[float, float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = 1.0e-11,
    max_level: int = 9,
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

    Returns:
        Estimate, last-level difference, and evaluation count.

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

    def term(t: float) -> float:
        """Return the weighted integrand at one trapezoid node.

        References:
            [EQ-VAL-QUAD-DE-001].

        """
        nonlocal evaluations
        s = math.pi * math.sinh(t)
        if s >= 0.0:
            e = math.exp(-s)
            x, xc = 1.0 / (1.0 + e), e / (1.0 + e)
        else:
            e = math.exp(s)
            x, xc = e / (1.0 + e), 1.0 / (1.0 + e)
        u = lower + width * x
        uc = upper_complement + width * xc
        value = integrand(u, uc)
        evaluations += 1
        if not math.isfinite(value):
            raise ValueError("integrand returned a non-finite value")
        return math.pi * math.cosh(t) * x * xc * value

    count = int(_T_MAX)
    total = math.fsum(term(float(j)) for j in range(-count, count + 1))
    estimate = width * total
    previous = estimate
    step = 1.0
    difference = math.inf
    for level in range(1, max_level + 1):
        step *= 0.5
        nodes = int(_T_MAX / step)
        odd = math.fsum(term(j * step) for j in range(-nodes, nodes + 1) if j % 2 != 0)
        total = 0.5 * total + step * odd
        estimate = width * total
        difference = abs(estimate - previous)
        previous = estimate
        if level >= 3 and difference <= tolerance * max(abs(estimate), 1e-300):
            break
    return DEQuadrature(estimate, difference, evaluations)
