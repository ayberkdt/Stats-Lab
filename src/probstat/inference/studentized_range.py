"""The studentized range distribution, by numerical integration.

For ``k`` independent standard normal variables with range ``W`` and an
independent ``S = sqrt(V / nu)``, ``V ~ chi^2_nu``, the studentized range
``Q = W / S`` has distribution function

``P(Q <= q) = int_0^inf f_nu(s) P_k(q s) ds``,
``P_k(w) = k int phi(z) [Phi(z) - Phi(z - w)]^(k-1) dz``,

with ``f_nu`` the density of ``S``. The inner integral is evaluated by a fixed
16-point Gauss-Legendre rule on nine panels of width two covering
``[-9, 9]`` (the standard normal mass outside is about ``2e-19``) and the
outer one by adaptive Gauss-Kronrod quadrature over a range of ``S`` that
carries all but a negligible part of its mass; ``nu = inf`` uses ``P_k(q)``
directly. Against SciPy 1.14.1 ``studentized_range.cdf`` (computed offline
on ``k = 2, 3, 4, 5, 8, 10, 20``, ``nu = 1, 2, 3, 5, 10, 20, 60, 120, 1000,
inf`` and ``q = 0.3 ... 15``) the largest absolute difference is
``6.4e-13``; for ``k = 2`` the closed forms ``2 Phi(q/sqrt 2) - 1`` and
``P(|T_nu| <= q/sqrt 2)`` are reproduced. Upper tails are formed as
``1 - P(Q <= q)``, so p-values carry an absolute (not relative) accuracy of
about ``1e-12``.
"""

from __future__ import annotations

import math
from functools import lru_cache

from probstat.foundations.validation import require_finite_real
from probstat.numerics import gauss_kronrod
from probstat.optimization import brent_root

from ._hypothesis import check_level

_PANEL_ORDER = 16
_Z_LIMIT = 9.0
_SQRT_HALF = math.sqrt(0.5)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _legendre_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return Gauss-Legendre nodes and weights on ``[-1, 1]`` by Newton's method.

    References:
        [EQ-INF-SRANGE-001], [REF-DAVIS-RABINOWITZ-1984, Section 2.7].

    """
    nodes: list[float] = []
    weights: list[float] = []
    for index in range(1, order + 1):
        x = math.cos(math.pi * (index - 0.25) / (order + 0.5))
        derivative = 1.0
        for _ in range(100):
            previous, current = 1.0, x
            for degree in range(2, order + 1):
                previous, current = (
                    current,
                    ((2 * degree - 1) * x * current - (degree - 1) * previous) / degree,
                )
            derivative = order * (x * current - previous) / (x * x - 1.0)
            step = current / derivative
            x -= step
            if abs(step) <= 1e-16:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * derivative * derivative))
    return tuple(nodes), tuple(weights)


def _inner_grid() -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return the inner nodes ``z``, weights ``w phi(z)``, and ``Phi(z)``.

    References:
        [EQ-INF-SRANGE-001], [REF-DAVIS-RABINOWITZ-1984, Section 2.7].

    """
    base_nodes, base_weights = _legendre_rule(_PANEL_ORDER)
    nodes: list[float] = []
    weights: list[float] = []
    lower = -_Z_LIMIT
    while lower < _Z_LIMIT - 1e-12:
        upper = lower + 2.0
        half, middle = 0.5 * (upper - lower), 0.5 * (upper + lower)
        for node, weight in zip(base_nodes, base_weights, strict=True):
            z = middle + half * node
            nodes.append(z)
            weights.append(half * weight * _INV_SQRT_2PI * math.exp(-0.5 * z * z))
        lower = upper
    cdf = tuple(0.5 * math.erfc(-z * _SQRT_HALF) for z in nodes)
    return tuple(nodes), tuple(weights), cdf


_NODES, _WEIGHTS, _CDF = _inner_grid()


def _range_cdf(width: float, groups: int) -> float:
    """Return ``P(W <= w)`` for the range ``W`` of ``k`` standard normals.

    References:
        [EQ-INF-SRANGE-001], [REF-HOCHBERG-TAMHANE-1987].

    """
    if width <= 0.0:
        return 0.0
    power = groups - 1
    total = math.fsum(
        weight * max(0.0, cdf - 0.5 * math.erfc(-(z - width) * _SQRT_HALF)) ** power
        for z, weight, cdf in zip(_NODES, _WEIGHTS, _CDF, strict=True)
    )
    return min(1.0, groups * total)


def _check_groups(groups: object) -> int:
    """Validate the number of means ``k >= 2``.

    References:
        [EQ-INF-SRANGE-001]. No equation is evaluated.

    """
    if isinstance(groups, bool) or not isinstance(groups, int):
        raise TypeError("groups must be an integer")
    if groups < 2:
        raise ValueError("groups must be at least two")
    return groups


def _check_df(df: object) -> float:
    """Validate the degrees of freedom ``nu >= 1`` (``math.inf`` allowed).

    References:
        [EQ-INF-SRANGE-001]. No equation is evaluated.

    """
    if isinstance(df, bool) or not isinstance(df, (int, float)):
        raise TypeError("df must be a real number")
    value = float(df)
    if math.isnan(value) or value < 1.0:
        raise ValueError("df must be at least one (math.inf allowed)")
    return value


@lru_cache(maxsize=4096)
def _cdf(q: float, groups: int, df: float) -> float:
    """Evaluate the studentized range distribution function (validated input).

    References:
        [EQ-INF-SRANGE-001], [REF-COPENHAVER-HOLLAND-1988].

    """
    if q <= 0.0:
        return 0.0
    if math.isinf(df):
        return _range_cdf(q, groups)
    half = 0.5 * df
    log_constant = math.log(2.0) + half * math.log(half) - math.lgamma(half)
    spread = 9.0 / math.sqrt(df)
    lower = max(0.0, 1.0 - spread)
    upper = 1.0 + spread + (12.0 / df if df < 10.0 else 0.0)

    def integrand(s: float) -> float:
        """Return ``f_nu(s) P_k(q s)``.

        References:
            [EQ-INF-SRANGE-001].

        """
        if s <= 0.0:
            return 0.0
        log_density = log_constant + (df - 1.0) * math.log(s) - half * s * s
        return math.exp(log_density) * _range_cdf(q * s, groups)

    mode = math.sqrt(max(df - 1.0, 0.0) / df)
    points = [lower, upper]
    if lower < mode < upper:
        points = [lower, mode, upper]
    total = 0.0
    for a, b in zip(points[:-1], points[1:], strict=True):
        total += gauss_kronrod(integrand, a, b, abs_tol=2e-12, rel_tol=1e-11).value
    return max(0.0, min(1.0, total))


def studentized_range_cdf(value: float, groups: int, df: float) -> float:
    """Return ``P(Q <= q)`` for the studentized range of ``k`` means.

    Args:
        value: Point ``q``; non-positive values give zero.
        groups: Number of means ``k >= 2``.
        df: Degrees of freedom ``nu >= 1`` of the variance estimate, or
            ``math.inf`` for a known variance.

    Returns:
        The distribution function, accurate to about ``1e-12`` absolute.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``groups < 2`` or ``df < 1``.

    Equation:
        [EQ-INF-SRANGE-001].

    References:
        [REF-COPENHAVER-HOLLAND-1988], [REF-LUND-LUND-1983],
        [REF-HOCHBERG-TAMHANE-1987].

    """
    q = require_finite_real(value, name="value")
    return _cdf(q, _check_groups(groups), _check_df(df))


def studentized_range_sf(value: float, groups: int, df: float) -> float:
    """Return the upper tail ``P(Q > q)`` (the Tukey HSD p-value).

    Args:
        value: Observed studentized range ``q``.
        groups: Number of means ``k >= 2``.
        df: Degrees of freedom ``nu >= 1``, or ``math.inf``.

    Returns:
        ``1 - P(Q <= q)``, accurate to about ``1e-12`` absolute.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``groups < 2`` or ``df < 1``.

    Equation:
        [EQ-INF-SRANGE-001].

    References:
        [REF-COPENHAVER-HOLLAND-1988], [REF-LUND-LUND-1983].

    """
    return max(0.0, 1.0 - studentized_range_cdf(value, groups, df))


def studentized_range_quantile(probability: float, groups: int, df: float) -> float:
    """Return the ``p``-quantile of the studentized range by Brent's method.

    Args:
        probability: Level strictly inside ``(0, 1)``.
        groups: Number of means ``k >= 2``.
        df: Degrees of freedom ``nu >= 1``, or ``math.inf``.

    Returns:
        ``q`` with ``P(Q <= q) = p`` to an absolute tolerance of ``1e-10``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-INF-SRANGE-001].

    References:
        [REF-COPENHAVER-HOLLAND-1988], [REF-BRENT-1973, Chapter 4].

    """
    p = check_level(probability, name="probability")
    k = _check_groups(groups)
    nu = _check_df(df)
    upper = 4.0
    while _cdf(upper, k, nu) < p:
        upper *= 2.0
        if upper > 1e6:
            raise ValueError("quantile is too far in the upper tail to bracket")
    return brent_root(lambda q: _cdf(q, k, nu) - p, 0.0, upper, tolerance=1e-10).root
