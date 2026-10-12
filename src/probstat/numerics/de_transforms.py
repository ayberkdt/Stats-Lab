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
``50 eps`` times the trapezoid estimate of ``int |f|``, plus two terms the
level difference cannot see:

* node rounding, ``h sum |w_k f(x_k)| min(1, 2 alpha_k ulp(x_k)/d_k)`` over
  nodes within ``1e4 ulp(x_k)`` of a finite end, with ``d_k`` the distance to
  that end and ``alpha_k`` the local log-log slope of ``|f|`` against it. An
  integrand singular at a non-zero end ``c`` (``alpha`` of order one) is
  sampled closer to ``c`` than ``ulp(c)`` and sees the rounded node; a smooth
  integrand on a shifted range such as ``[1e8, 1e8 + 1]`` has ``alpha`` near
  zero and no such error;
* truncation, the mass beyond the outermost node on each side, estimated
  from ``|w f|`` there and its decay rate, so slowly decaying tails such as
  ``x^-1.025`` at infinity are not reported as converged.

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
    truncation_term,
)

# Finite ends: nodes until min(x, 1 - x) ~ exp(-pi sinh t) reaches 1e-300.
_T_MAX_FINITE = math.asinh(math.log(1.0e300) / math.pi)
_T_SAFE_FINITE = math.asinh(math.log(1.0e15) / math.pi)
# Infinite ranges: exp((pi/2) sinh t) runs from 1e-300 to 1e300.
_T_MAX_INFINITE = math.asinh(2.0 * math.log(1.0e300) / math.pi)
_S_SAFE_INFINITE = math.log(1.0e15)
_X_SAFE = 1.0e15
# Nodes farther than 1e4 ulp(x) from the end carry only the generic rounding
# of x that every rule shares (about ulp(x) |f'| per unit length); the
# rounding term counts the nodes inside this zone.
_ROUNDING_ZONE = 1.0e-4

# (node, weight dx/dt, failure tolerated, ulp(x)/d for the exact distance d to
# the nearest finite end, rounded distance |x - end| computed from the node)
NodeMap = Callable[[float], tuple[float, float, bool, float, float]]


def _rounding_shares(
    records: list[tuple[float, float, float, float]],
) -> float:
    """Return the node-rounding error estimate of nodes ordered outwards.

    Each record is ``(|w f|, |f|, ulp(x)/d, d_rounded)`` for nodes on one
    side of ``t = 0``, ordered away from the centre. A node rounded by
    ``ulp(x)`` changes ``f`` by about ``alpha ulp(x)/d`` relative, where
    ``alpha = |d log|f| / d log d|`` is the local log-log slope of ``|f|``
    against the distance to the end. The slope is estimated from the
    nearest node further inside with a different rounded distance (inside
    the rounding zone neighbouring nodes share one rounded node, so their
    values carry no slope information); for a node rounded onto the end the
    change of ``log|f|`` across that last step is used, and without any
    reference node ``alpha = 1``. A smooth integrand has ``alpha`` near zero
    and no rounding share; an integrand singular at a non-zero end keeps
    ``alpha`` of order one.

    Args:
        records: Node records ordered outwards.

    Returns:
        ``sum |w f| min(1, 2 alpha ulp(x)/d)`` over the records with
        ``ulp(x)/d >= 1e-4`` (the factor two covers the nodes rounded onto
        the end, whose mass is lost; farther nodes carry only the generic
        rounding of ``x`` common to every rule).

    References:
        [EQ-NUM-DE-001], [REF-MORI-SUGIHARA-2001].

    """
    shares: list[float] = []
    # The innermost nodes have no node further inside; they use the first
    # node further out with a different rounded distance instead.
    reference: tuple[float, float] | None = None
    first_distance = records[0][3] if records else math.nan
    for _, value, _, distance in records:
        if distance != first_distance and value > 0.0 and distance > 0.0:
            reference = (distance, value)
            break
    group_distance = math.nan
    group_representative: tuple[float, float] | None = None
    for modulus, value, ratio, distance in records:
        if distance != group_distance:
            if group_representative is not None:
                reference = group_representative
            group_distance = distance
            group_representative = None
        if group_representative is None and value > 0.0 and distance > 0.0:
            group_representative = (distance, value)
        if modulus == 0.0 or ratio < _ROUNDING_ZONE:
            continue
        slope = 1.0
        if reference is not None and value > 0.0:
            change = abs(math.log(value / reference[1]))
            if distance > 0.0:
                spread = abs(math.log(distance / reference[0]))
                slope = change / spread if spread > 0.0 else 1.0
            else:
                slope = change
        shares.append(modulus * min(1.0, 2.0 * slope * ratio))
    return math.fsum(shares)


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

    The error estimate is the level difference (floored at ``50 eps`` times
    the estimate of ``int |f|``), plus the node-rounding term of
    :func:`_rounding_shares`, plus the truncation term of
    :func:`probstat.numerics.integration_result.truncation_term` (the mass
    beyond the outermost node on each side, from the edge modulus ``|w f|``
    and its decay rate). A slowly decaying tail (``x^-1.025`` at infinity,
    ``x^-0.97`` at zero) therefore keeps the run from being reported as
    converged.

    Args:
        function: User integrand.
        node_map: Map ``t -> (x, dx/dt, tolerant, ulp(x)/d, d_rounded)``
            with ``d`` the exact and ``d_rounded`` the rounded distance to
            the nearest finite end (``ulp(x)/d`` is zero without finite
            ends).
        t_max: Truncation of the ``t`` range.
        rule: Rule label for the result.
        interval: Original integration range for the result.
        abs_tol: Validated absolute tolerance.
        rel_tol: Validated relative tolerance.
        max_level: Largest halving level (at least 3).

    Returns:
        Estimate, error estimate, evaluations, and flag.

    Raises:
        ValueError: If the integrand is non-finite inside the safe range.

    Equation:
        [EQ-NUM-DE-001].

    References:
        [REF-TAKAHASI-MORI-1974], [REF-MORI-SUGIHARA-2001].

    """
    evaluations = 0
    outer: dict[float, float] = {}

    def term(t: float) -> tuple[float, float, float, float, float]:
        """Return ``f(x(t)) x'(t)``, its modulus, ``|f|``, and rounding data.

        References:
            [EQ-NUM-DE-001].

        """
        nonlocal evaluations
        x, weight, tolerant, ratio, distance = node_map(t)
        evaluations += 1
        if tolerant:
            try:
                raw = function(x)
            except (ArithmeticError, ValueError):
                return 0.0, 0.0, 0.0, ratio, distance
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise TypeError(f"function({x!r}) must be a real number")
            value = float(raw)
            if not math.isfinite(value):
                return 0.0, 0.0, 0.0, ratio, distance
        else:
            value = require_finite_real(function(x), name=f"function({x!r})")
        weighted = weight * value
        if not math.isfinite(weighted):
            if tolerant:
                return 0.0, 0.0, 0.0, ratio, distance
            raise ValueError(f"weighted integrand overflowed at x = {x!r}")
        return weighted, abs(weighted), abs(value), ratio, distance

    def level_sum(points: list[float]) -> tuple[float, float, float]:
        """Return the sums of terms, moduli, and rounding shares.

        ``points`` must be increasing. Moduli of the nodes near ``t_max``
        are kept for the truncation term.

        References:
            [EQ-NUM-DE-001].

        """
        terms = [term(t) for t in points]
        for t, p in zip(points, terms, strict=True):
            if abs(t) >= t_max - 2.0:
                outer[t] = p[1]
        records = [(p[1], p[2], p[3], p[4]) for p in terms]
        right = [r for t, r in zip(points, records, strict=True) if t >= 0.0]
        left = [r for t, r in zip(points, records, strict=True) if t < 0.0]
        return (
            math.fsum(p[0] for p in terms),
            math.fsum(p[1] for p in terms),
            _rounding_shares(right) + _rounding_shares(left[::-1]),
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
        for t in [t for t in outer if abs(t) < t_max - 2.0 * step]:
            del outer[t]
        truncation = truncation_term(outer, step)
        error = (
            max(abs(total - previous), ROUNDOFF_FACTOR * absolute)
            + rounding
            + truncation
        )
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

    def node_map(t: float) -> tuple[float, float, bool, float, float]:
        """Return the tanh-sinh node, Jacobian, tolerance flag, and rounding.

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
        ratio = math.ulp(x) / distance if distance > 0.0 else math.inf
        rounded = b - x if s >= 0.0 else x - a
        return x, weight, tolerant, ratio, max(rounded, 0.0)

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

    def node_map(t: float) -> tuple[float, float, bool, float, float]:
        """Return the exp-sinh node, Jacobian, tolerance flag, and rounding.

        References:
            [EQ-NUM-DE-001].

        """
        s = 0.5 * math.pi * math.sinh(t)
        u = math.exp(s)
        x = anchor + sign * u
        weight = 0.5 * math.pi * math.cosh(t) * u
        tolerant = abs(s) > _S_SAFE_INFINITE or x == anchor
        return x, weight, tolerant, math.ulp(x) / u, abs(x - anchor)

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

    def node_map(t: float) -> tuple[float, float, bool, float, float]:
        """Return the sinh-sinh node, Jacobian, tolerance flag, and rounding.

        References:
            [EQ-NUM-DE-001].

        """
        s = 0.5 * math.pi * math.sinh(t)
        x = math.sinh(s)
        weight = 0.5 * math.pi * math.cosh(t) * math.cosh(s)
        return x, weight, abs(x) > _X_SAFE, 0.0, math.inf

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
