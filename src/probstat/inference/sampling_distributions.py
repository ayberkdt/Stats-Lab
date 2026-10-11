"""Tail probabilities of the sampling distributions used by classical tests.

The Student-t, chi-square, and F laws arise as exact sampling distributions of
test statistics computed from independent normal observations. Their upper
tails are evaluated directly through the regularized incomplete gamma and beta
functions so that small p-values are not lost to ``1 - CDF`` cancellation.
The noncentral t distribution, needed for the power of t tests, is evaluated by
adaptive Gauss-Legendre quadrature of its exact normal/chi mixture
representation.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from probstat.distributions._special import (
    log1pmx,
    regularized_beta,
    regularized_gamma_q,
    stirling_error,
)
from probstat.foundations.validation import require_finite_real

_SQRT_HALF = math.sqrt(0.5)
_GAUSS_ORDER = 10
_NONCENTRAL_TOLERANCE = 1e-13
_MAX_BISECTIONS = 48


def _gauss_legendre_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]`` by Newton's method.

    Each node is a root of ``P_n``, refined from the Tricomi-type starting
    value ``cos(pi (i - 1/4)/(n + 1/2))`` with the three-term recurrence;
    the weight is ``2 / ((1 - x^2) P_n'(x)^2)``.

    Args:
        order: Number of nodes ``n >= 1``.

    Returns:
        Nodes and positive weights (summing to two).

    Equation:
        [EQ-INF-DIST-NCT-002].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Section 2.7].

    """
    nodes: list[float] = []
    weights: list[float] = []
    for index in range(1, order + 1):
        x = math.cos(math.pi * (index - 0.25) / (order + 0.5))
        derivative = 1.0
        for _ in range(100):
            previous, current = 1.0, x
            for degree in range(2, order + 1):
                previous, current = current, (
                    (2 * degree - 1) * x * current - (degree - 1) * previous
                ) / degree
            derivative = order * (x * current - previous) / (x * x - 1.0)
            step = current / derivative
            x -= step
            if abs(step) <= 1e-16:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * derivative * derivative))
    return tuple(nodes), tuple(weights)


_GAUSS_NODES, _GAUSS_WEIGHTS = _gauss_legendre_rule(_GAUSS_ORDER)


def _gauss_panel(function: Callable[[float], float], low: float, high: float) -> float:
    """Return the fixed-order Gauss-Legendre estimate on ``[low, high]``.

    References:
        [EQ-INF-DIST-NCT-002], [REF-DAVIS-RABINOWITZ-1984, Section 2.7].

    """
    half = 0.5 * (high - low)
    middle = 0.5 * (high + low)
    return half * math.fsum(
        weight * function(middle + half * node)
        for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS, strict=True)
    )


def _adaptive_integral(
    function: Callable[[float], float],
    breakpoints: list[float],
    tolerance: float,
) -> float:
    """Integrate by globally bisected Gauss-Legendre panels between breakpoints.

    A panel is accepted when its two halves agree with the whole-panel
    estimate to within its share of ``tolerance`` (the share halves with each
    bisection); otherwise both halves are refined, to at most 48 levels.

    Args:
        function: Smooth integrand on each breakpoint interval.
        breakpoints: Increasing finite interval ends.
        tolerance: Absolute error target for the whole integral.

    Returns:
        Integral estimate.

    Equation:
        [EQ-INF-DIST-NCT-002].

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 6].

    """
    pieces: list[float] = []
    count = len(breakpoints) - 1
    stack = [
        (low, high, _gauss_panel(function, low, high), tolerance / count, 0)
        for low, high in zip(breakpoints[:-1], breakpoints[1:], strict=True)
        if high > low
    ]
    while stack:
        low, high, whole, share, depth = stack.pop()
        middle = 0.5 * (low + high)
        left = _gauss_panel(function, low, middle)
        right = _gauss_panel(function, middle, high)
        if abs(left + right - whole) <= share or depth >= _MAX_BISECTIONS:
            pieces.append(left + right)
            continue
        stack.append((low, middle, left, 0.5 * share, depth + 1))
        stack.append((middle, high, right, 0.5 * share, depth + 1))
    return math.fsum(pieces)


def standard_normal_cdf(value: float) -> float:
    """Return the standard normal distribution function ``Phi(z)``.

    Args:
        value: Finite evaluation point.

    Returns:
        ``Phi(z) = erfc(-z / sqrt(2)) / 2``, accurate in both tails.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is not finite.

    Equation:
        [EQ-INF-DIST-NORM-001].

    References:
        [REF-NIST-DLMF, Section 7.2], [REF-CASELLA-BERGER-2002, Chapter 3].

    """
    z = require_finite_real(value, name="value")
    return 0.5 * math.erfc(-z * _SQRT_HALF)


def standard_normal_sf(value: float) -> float:
    """Return the standard normal survival function ``1 - Phi(z)``.

    Args:
        value: Finite evaluation point.

    Returns:
        ``erfc(z / sqrt(2)) / 2`` without subtractive cancellation.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is not finite.

    Equation:
        [EQ-INF-DIST-NORM-001].

    References:
        [REF-NIST-DLMF, Section 7.2], [REF-CASELLA-BERGER-2002, Chapter 3].

    """
    z = require_finite_real(value, name="value")
    return 0.5 * math.erfc(z * _SQRT_HALF)


def _positive_df(value: float, name: str) -> float:
    """Validate a strictly positive finite degrees-of-freedom parameter.

    Args:
        value: Candidate degrees of freedom.
        name: Parameter name used in error messages.

    Returns:
        Validated float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is not finite and positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def student_t_sf(value: float, degrees_of_freedom: float) -> float:
    """Return the Student-t upper tail ``P(T >= t)`` for ``nu`` degrees of freedom.

    Args:
        value: Finite evaluation point ``t``.
        degrees_of_freedom: Positive, possibly fractional, ``nu``.

    Returns:
        ``I_{nu/(nu+t^2)}(nu/2, 1/2) / 2`` for ``t >= 0`` and its complement
        for ``t < 0``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``degrees_of_freedom`` is not positive.

    Equation:
        [EQ-DIST-T-001], [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17], [REF-CASELLA-BERGER-2002, Chapter 5].

    """
    t = require_finite_real(value, name="value")
    nu = _positive_df(degrees_of_freedom, "degrees_of_freedom")
    if t == 0.0:
        return 0.5
    tail = 0.5 * regularized_beta(nu / (nu + t * t), 0.5 * nu, 0.5)
    return tail if t > 0.0 else 1.0 - tail


def student_t_two_sided_p(value: float, degrees_of_freedom: float) -> float:
    """Return the two-sided Student-t tail probability ``P(|T| >= |t|)``.

    Args:
        value: Finite observed statistic ``t``.
        degrees_of_freedom: Positive ``nu``.

    Returns:
        ``I_{nu/(nu+t^2)}(nu/2, 1/2)``, equal to one at ``t = 0``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``degrees_of_freedom`` is not positive.

    Equation:
        [EQ-DIST-T-001], [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17], [REF-CASELLA-BERGER-2002, Chapter 5].

    """
    t = require_finite_real(value, name="value")
    nu = _positive_df(degrees_of_freedom, "degrees_of_freedom")
    if t == 0.0:
        return 1.0
    return regularized_beta(nu / (nu + t * t), 0.5 * nu, 0.5)


def chi_square_sf(value: float, degrees_of_freedom: float) -> float:
    """Return the chi-square upper tail ``P(X >= x)`` for ``k`` degrees of freedom.

    Args:
        value: Finite non-negative evaluation point; negative values give one.
        degrees_of_freedom: Positive ``k``.

    Returns:
        ``Q(k/2, x/2)``, the upper regularized incomplete gamma function.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``degrees_of_freedom`` is not positive.

    Equation:
        [EQ-INF-DIST-CHI2-001], [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.2], [REF-CASELLA-BERGER-2002, Chapter 5].

    """
    x = require_finite_real(value, name="value")
    k = _positive_df(degrees_of_freedom, "degrees_of_freedom")
    if x <= 0.0:
        return 1.0
    return regularized_gamma_q(0.5 * k, 0.5 * x)


def f_sf(value: float, numerator_df: float, denominator_df: float) -> float:
    """Return the Fisher-Snedecor F upper tail ``P(F >= f)``.

    Args:
        value: Finite non-negative evaluation point; negative values give one.
        numerator_df: Positive numerator degrees of freedom ``d1``.
        denominator_df: Positive denominator degrees of freedom ``d2``.

    Returns:
        ``I_{d2/(d2 + d1 f)}(d2/2, d1/2)``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If a degrees-of-freedom parameter is not positive.

    Equation:
        [EQ-INF-DIST-F-001], [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17], [REF-CASELLA-BERGER-2002, Chapter 5].

    """
    f = require_finite_real(value, name="value")
    d1 = _positive_df(numerator_df, "numerator_df")
    d2 = _positive_df(denominator_df, "denominator_df")
    if f <= 0.0:
        return 1.0
    return regularized_beta(d2 / (d2 + d1 * f), 0.5 * d2, 0.5 * d1)


def noncentral_t_cdf(
    value: float, degrees_of_freedom: float, noncentrality: float
) -> float:
    """Return the noncentral Student-t distribution function.

    ``T' = (Z + delta) / sqrt(V / nu)`` with ``Z`` standard normal independent
    of ``V ~ chi^2_nu``. Conditioning on ``S = sqrt(V / nu)`` gives
    ``P(T' <= t) = E[Phi(t S - delta)]``. The expectation is integrated over
    a range that holds all but a negligible (below ``1e-30``) part of the law
    of ``S`` by adaptive Gauss-Legendre quadrature, with breakpoints at the
    chi-scale mode ``S = 1`` and at ``S = delta/t``, where the normal factor
    switches from 0 to 1 over a width ``1/|t|``. For ``nu < 2`` the variable
    ``u = s^nu`` removes the ``s^(nu-1)`` endpoint singularity of the
    density. (A fixed 2000-panel Simpson rule erred by up to ``7e-4``, for
    example at ``(t, nu, delta) = (2000, 30, 1950)``.) With zero
    noncentrality the exact central CDF is returned instead.

    Args:
        value: Finite evaluation point ``t``.
        degrees_of_freedom: Degrees of freedom ``nu >= 1``.
        noncentrality: Finite noncentrality parameter ``delta``.

    Returns:
        ``P(T' <= t)`` in ``[0, 1]``; the quadrature targets an absolute
        error of ``1e-13``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``degrees_of_freedom`` is below one.

    Equation:
        [EQ-INF-DIST-NCT-001], [EQ-INF-DIST-NCT-002].

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995, noncentral t chapter],
        [REF-DAVIS-RABINOWITZ-1984, Chapters 2 and 6].

    """
    t = require_finite_real(value, name="value")
    nu = require_finite_real(degrees_of_freedom, name="degrees_of_freedom")
    delta = require_finite_real(noncentrality, name="noncentrality")
    if nu < 1.0:
        raise ValueError("degrees_of_freedom must be at least one")
    if delta == 0.0:
        return 1.0 - student_t_sf(t, nu)
    spread = 12.0 / math.sqrt(nu)
    lower = max(0.0, 1.0 - spread)
    upper = 1.0 + spread
    half = 0.5 * nu
    log_constant = math.log(2.0) + half * math.log(half) - math.lgamma(half)
    # log g(s) = log 2 + log(nu/2)/2 - log sqrt(2 pi) - stirling_error(nu/2)
    #            + (nu/2) [log(s^2) - (s^2 - 1)] - log s, free of the
    # O(nu log nu) cancellation in (nu/2) log(nu/2) - lgamma(nu/2).
    log_front = (
        math.log(2.0)
        + 0.5 * math.log(half)
        - 0.5 * math.log(2.0 * math.pi)
        - stirling_error(half)
    )

    def integrand(s: float) -> float:
        """Return ``Phi(t s - delta) g_nu(s)`` for the chi-scale density ``g``.

        References:
            [EQ-INF-DIST-NCT-001], [EQ-NUM-STIRLING-001].

        """
        if s <= 0.0:
            density = math.exp(log_constant) if nu == 1.0 else 0.0
        else:
            excess = s * s - 1.0
            if abs(excess) <= 0.5:
                core = log1pmx(excess)
            else:
                core = 2.0 * math.log(s) - excess
            density = math.exp(log_front + half * core - math.log(s))
        return 0.5 * math.erfc(-(t * s - delta) * _SQRT_HALF) * density

    def transformed(u: float) -> float:
        """Return the integrand in ``u = s^nu``: ``Phi(t s - delta) g(s) ds/du``.

        With ``s = u^(1/nu)``, ``g_nu(s) ds = C exp(-nu u^(2/nu)/2) du / nu``,
        which is bounded and continuous at ``u = 0``.

        References:
            [EQ-INF-DIST-NCT-002]. No new equation is introduced.

        """
        s = u ** (1.0 / nu) if u > 0.0 else 0.0
        density = math.exp(log_constant - 0.5 * nu * s * s) / nu
        return 0.5 * math.erfc(-(t * s - delta) * _SQRT_HALF) * density

    # Breakpoints resolve both features at their own scales: the chi-scale
    # bump (mode 1, standard deviation about 1/sqrt(2 nu)) and the normal
    # step at delta/t (width 1/|t|; Phi is within 1e-15 of 0 or 1 beyond
    # 8 widths), so no panel can straddle a feature unseen by its nodes.
    spread_s = 1.0 / math.sqrt(2.0 * nu)
    candidates = [1.0 + k * spread_s for k in (-3.0, 0.0, 3.0)]
    if t != 0.0:
        step_centre = delta / t
        candidates += [step_centre + k / abs(t) for k in (-8.0, -1.0, 0.0, 1.0, 8.0)]
    points = {lower, upper}
    points.update(point for point in candidates if lower < point < upper)
    breakpoints = sorted(points)
    if nu < 2.0:
        mapped = sorted({point**nu for point in breakpoints})
        value = _adaptive_integral(transformed, mapped, _NONCENTRAL_TOLERANCE)
    else:
        value = _adaptive_integral(integrand, breakpoints, _NONCENTRAL_TOLERANCE)
    return max(0.0, min(1.0, value))
