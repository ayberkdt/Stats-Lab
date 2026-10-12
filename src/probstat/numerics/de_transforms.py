"""Double-exponential quadrature on finite, half-infinite and infinite ranges.

Each rule substitutes ``x = phi(t)`` so that the transformed integrand
``f(phi(t)) phi'(t)`` decays double-exponentially as ``|t| -> inf`` and then
applies the trapezoidal rule with step ``h = 2^-k``, halving ``h`` and reusing
the previous nodes, exactly as :func:`probstat.numerics.tanh_sinh` does on
``(0, 1)``:

* tanh-sinh on ``[a, b]``: ``x = a + (b - a)/(1 + exp(-pi sinh t))``;
* exp-sinh on ``[a, inf)``: ``x = a + exp((pi/2) sinh t)`` (mirrored for
  ``(-inf, b]``);
* sinh-sinh on ``(-inf, inf)``: ``x = sinh((pi/2) sinh t)``.

The error estimate is the difference of the last two levels, floored at
``50 eps`` times the trapezoid estimate of ``int |f|``, plus a node-rounding
term ``h sum |w_k f(x_k)| min(1, ulp(x_k)/d_k)`` with ``d_k`` the distance to
the nearest finite end. The last term is about ``eps int |f|`` when that end
is zero, but it is large when an integrand singular at a non-zero end ``c``
is sampled closer to ``c`` than ``ulp(c)``: the integrand then sees the
rounded node, an error the level difference cannot detect.

Evaluation policy: inside the *safe range* the integrand must return a finite
real value, otherwise ``ValueError`` is raised. Outside it a raising or
non-finite evaluation contributes zero, because those nodes carry negligible
weight for any integrable ``f``; they are still evaluated and counted. The
safe range excludes nodes closer than ``1e-15`` of the width to a finite end
(tanh-sinh), closer than ``1e-15`` to the finite end (exp-sinh), rounded onto
a finite end, or farther than ``1e15`` from it (exp-sinh) or from zero
(sinh-sinh).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

from probstat.foundations.validation import require_finite_real

from .integration_result import (
    ROUNDOFF_FACTOR,
    IntegrationResult,
    check_limit,
    check_tolerances,
)

# Finite ends: nodes until min(x, 1 - x) ~ exp(-pi sinh t) reaches 1e-300.
_T_MAX_FINITE = math.asinh(math.log(1.0e300) / math.pi)
_T_SAFE_FINITE = math.asinh(math.log(1.0e15) / math.pi)
# Infinite ranges: exp((pi/2) sinh t) runs from 1e-300 to 1e300.
_T_MAX_INFINITE = math.asinh(2.0 * math.log(1.0e300) / math.pi)
_S_SAFE_INFINITE = math.log(1.0e15)
_X_SAFE = 1.0e15

# (node, weight dx/dt, failure tolerated, relative node-rounding error)
NodeMap = Callable[[float], tuple[float, float, bool, float]]


def _double_exponential(
    function: Callable[[float], float],
    node_map: NodeMap,
    t_max: float,
    *,
    rule: str,
    interval: tuple[float, float],
    abs_tol: float,
    rel_tol: float,
    max_level: int,
) -> IntegrationResult:
    """Run the level-halving trapezoidal rule on a transformed integrand.

    Args:
        function: User integrand.
        node_map: Map ``t -> (x, dx/dt, tolerant, rho)`` where ``rho`` is
            ``min(1, ulp(x)/d)`` for the distance ``d`` to the nearest finite
            end (zero without finite ends).
        t_max: Truncation of the ``t`` range.
        rule: Rule label for the result.
        interval: Original integration range for the result.
        abs_tol: Validated absolute tolerance.
        rel_tol: Validated relative tolerance.
        max_level: Largest halving level (at least 3).

    Returns:
        Estimate, level-difference error estimate, evaluations, and flag.

    Raises:
        ValueError: If the integrand is non-finite inside the safe range.

    Equation:
        [EQ-NUM-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974], [REF-MORI-SUGIHARA-2001].

    """
    evaluations = 0

    def term(t: float) -> tuple[float, float, float]:
        """Return ``f(x(t)) x'(t)``, its modulus, and its rounding share.

        References:
            [EQ-NUM-DE-001].

        """
        nonlocal evaluations
        x, weight, tolerant, rounding = node_map(t)
        evaluations += 1
        if tolerant:
            try:
                raw = function(x)
            except (ArithmeticError, ValueError):
                return 0.0, 0.0, 0.0
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"function({x!r}) must be a real number")
            value = float(raw)
            if not math.isfinite(value):
                return 0.0, 0.0, 0.0
        else:
            value = require_finite_real(function(x), name=f"function({x!r})")
        weighted = weight * value
        if not math.isfinite(weighted):
            if tolerant:
                return 0.0, 0.0, 0.0
            raise ValueError(f"weighted integrand overflowed at x = {x!r}")
        return weighted, abs(weighted), abs(weighted) * rounding

    def level_sum(points: list[float]) -> tuple[float, float, float]:
        """Return the sums of terms, moduli, and rounding shares.

        References:
            [EQ-NUM-DE-001].

        """
        triples = [term(t) for t in points]
        return (
            math.fsum(p[0] for p in triples),
            math.fsum(p[1] for p in triples),
            math.fsum(p[2] for p in triples),
        )

    count = int(t_max)
    total, absolute, rounding = level_sum([float(j) for j in range(-count, count + 1)])
    previous = total
    step = 1.0
    error = math.inf
    levels = 0
    converged = False
    for level in range(1, max_level + 1):
        step *= 0.5
        nodes = int(t_max / step)
        odd_total, odd_absolute, odd_rounding = level_sum(
            [j * step for j in range(-nodes, nodes + 1) if j % 2 != 0]
        )
        total = 0.5 * total + step * odd_total
        absolute = 0.5 * absolute + step * odd_absolute
        rounding = 0.5 * rounding + step * odd_rounding
        error = max(abs(total - previous), ROUNDOFF_FACTOR * absolute) + rounding
        previous = total
        levels = level
        if level >= 3 and error <= max(abs_tol, rel_tol * abs(total)):
            converged = True
            break
    return IntegrationResult(
        value=total,
        error_estimate=error,
        evaluations=evaluations,
        converged=converged,
        rule=rule,
        intervals=(interval,),
        levels=levels,
    )


def _check_common(
    function: Callable[[float], float],
    abs_tol: float,
    rel_tol: float,
    max_level: int,
) -> tuple[float, float, int]:
    """Validate the arguments shared by the double-exponential rules.

    Args:
        function: Candidate integrand.
        abs_tol: Absolute tolerance.
        rel_tol: Relative tolerance.
        max_level: Largest halving level.

    Returns:
        Validated tolerances and level limit.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If a tolerance or the level limit is invalid.

    References:
        Project numerical validation policy. No equation is evaluated.

    """
    if not callable(function):
        raise TypeError("function must be callable")
    absolute_target, relative_target = check_tolerances(abs_tol, rel_tol)
    if isinstance(max_level, bool) or not isinstance(max_level, int):
        raise TypeError("max_level must be an integer")
    if not 3 <= max_level <= 16:
        raise ValueError("max_level must lie in [3, 16]")
    return absolute_target, relative_target, max_level


def tanh_sinh_interval(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 1.0e-10,
    max_level: int = 9,
) -> IntegrationResult:
    """Integrate ``f`` over a finite ``[lower, upper]`` by tanh-sinh quadrature.

    The integrand is never evaluated at an end unless a node rounds onto it,
    so integrable end singularities such as ``x^-1/2`` or ``log x`` at
    ``x = 0`` converge at the double-exponential rate. A singularity at a
    non-zero end ``c`` is resolved only down to ``ulp(c)``; substitute so the
    singular end sits at zero, or use :func:`probstat.numerics.tanh_sinh`,
    which passes the complement ``1 - u`` without cancellation.

    Args:
        function: Integrand returning finite reals inside the safe range.
        lower: Finite lower limit.
        upper: Finite upper limit greater than ``lower``.
        abs_tol: Absolute error target.
        rel_tol: Relative error target.
        max_level: Largest halving level, in ``[3, 16]``.

    Returns:
        Estimate, error estimate, evaluations, levels, and flag.

    Raises:
        TypeError: If an argument or integrand value has an invalid type.
        ValueError: If limits or tolerances are invalid, or the integrand is
            non-finite inside the safe range.

    Equation:
        [EQ-NUM-DE-001], [EQ-VAL-QUAD-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974], [REF-MORI-SUGIHARA-2001].

    """
    targets = _check_common(function, abs_tol, rel_tol, max_level)
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    if not a < b:
        raise ValueError("upper must be greater than lower")
    width = b - a
    if not math.isfinite(width):
        raise ValueError("upper - lower must be finite")

    def node_map(t: float) -> tuple[float, float, bool, float]:
        """Return the tanh-sinh node, Jacobian, tolerance flag, and rho.

        References:
            [EQ-VAL-QUAD-DE-001].

        """
        s = math.pi * math.sinh(t)
        if s >= 0.0:
            e = math.exp(-s)
            fraction, complement = 1.0 / (1.0 + e), e / (1.0 + e)
            x = b - width * complement
        else:
            e = math.exp(s)
            fraction, complement = e / (1.0 + e), 1.0 / (1.0 + e)
            x = a + width * fraction
        weight = width * math.pi * math.cosh(t) * fraction * complement
        tolerant = abs(t) > _T_SAFE_FINITE or x <= a or x >= b
        distance = width * min(fraction, complement)
        rounding = min(1.0, math.ulp(x) / distance) if distance > 0.0 else 1.0
        return x, weight, tolerant, rounding

    return _double_exponential(
        function,
        node_map,
        _T_MAX_FINITE,
        rule="tanh-sinh",
        interval=(a, b),
        abs_tol=targets[0],
        rel_tol=targets[1],
        max_level=targets[2],
    )


def exp_sinh(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 1.0e-10,
    max_level: int = 9,
) -> IntegrationResult:
    """Integrate over a half-line ``[a, inf)`` or ``(-inf, b]`` by exp-sinh.

    With ``u = exp((pi/2) sinh t)`` the node is ``a + u`` (or ``b - u``) and
    ``dx/dt = (pi/2) cosh(t) u``. Nodes cluster double-exponentially at the
    finite end (``u`` down to ``1e-300``), so end singularities at zero are
    handled, and run out to ``u = 1e300``, so algebraic decay faster than
    ``1/x`` converges. Slowly decaying oscillatory integrands such as
    ``sin(x)/x`` are not supported (no QAWO-type rule); the level differences
    then stay large and ``converged`` is ``False``.

    Args:
        function: Integrand returning finite reals inside the safe range.
        lower: Finite lower limit, or ``-inf``.
        upper: ``inf``, or a finite upper limit when ``lower`` is ``-inf``.
        abs_tol: Absolute error target.
        rel_tol: Relative error target.
        max_level: Largest halving level, in ``[3, 16]``.

    Returns:
        Estimate, error estimate, evaluations, levels, and flag.

    Raises:
        TypeError: If an argument or integrand value has an invalid type.
        ValueError: If not exactly one limit is infinite, tolerances are
            invalid, or the integrand is non-finite inside the safe range.

    Equation:
        [EQ-NUM-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974], [REF-MORI-SUGIHARA-2001].

    """
    targets = _check_common(function, abs_tol, rel_tol, max_level)
    a = check_limit(lower)
    b = check_limit(upper)
    if math.isfinite(a) and b == math.inf:
        anchor, sign = a, 1.0
    elif a == -math.inf and math.isfinite(b):
        anchor, sign = b, -1.0
    else:
        raise ValueError("exp_sinh needs one finite limit and one infinite limit")

    def node_map(t: float) -> tuple[float, float, bool, float]:
        """Return the exp-sinh node, Jacobian, tolerance flag, and rho.

        References:
            [EQ-NUM-DE-001].

        """
        s = 0.5 * math.pi * math.sinh(t)
        u = math.exp(s)
        x = anchor + sign * u
        weight = 0.5 * math.pi * math.cosh(t) * u
        tolerant = abs(s) > _S_SAFE_INFINITE or x == anchor
        return x, weight, tolerant, min(1.0, math.ulp(x) / u)

    return _double_exponential(
        function,
        node_map,
        _T_MAX_INFINITE,
        rule="exp-sinh",
        interval=(a, b),
        abs_tol=targets[0],
        rel_tol=targets[1],
        max_level=targets[2],
    )


def sinh_sinh(
    function: Callable[[float], float],
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 1.0e-10,
    max_level: int = 9,
) -> IntegrationResult:
    """Integrate over the whole real line by sinh-sinh quadrature.

    The node is ``x = sinh((pi/2) sinh t)`` with
    ``dx/dt = (pi/2) cosh(t) cosh((pi/2) sinh t)``; nodes reach
    ``|x| ~ 5e299``. The integrand should decay faster than ``1/|x|`` in
    both tails and be smooth on the line (no interior singularity or kink;
    split the range there instead).

    Args:
        function: Integrand returning finite reals for ``|x| <= 1e15``.
        abs_tol: Absolute error target.
        rel_tol: Relative error target.
        max_level: Largest halving level, in ``[3, 16]``.

    Returns:
        Estimate, error estimate, evaluations, levels, and flag.

    Raises:
        TypeError: If an argument or integrand value has an invalid type.
        ValueError: If tolerances are invalid or the integrand is non-finite
            inside the safe range.

    Equation:
        [EQ-NUM-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974], [REF-MORI-SUGIHARA-2001].

    """
    targets = _check_common(function, abs_tol, rel_tol, max_level)

    def node_map(t: float) -> tuple[float, float, bool, float]:
        """Return the sinh-sinh node, Jacobian, tolerance flag, and rho.

        References:
            [EQ-NUM-DE-001].

        """
        s = 0.5 * math.pi * math.sinh(t)
        x = math.sinh(s)
        weight = 0.5 * math.pi * math.cosh(t) * math.cosh(s)
        return x, weight, abs(x) > _X_SAFE, 0.0

    return _double_exponential(
        function,
        node_map,
        _T_MAX_INFINITE,
        rule="sinh-sinh",
        interval=(-math.inf, math.inf),
        abs_tol=targets[0],
        rel_tol=targets[1],
        max_level=targets[2],
    )
