"""Internal numerical helpers shared by the continuous distribution families.

The helpers cover three needs that recur across families:

* tail-accurate quantile inversion by a bracketed Brent iteration on a
  log-probability residual in a transformed coordinate;
* log-space standard-normal tails that stay finite far beyond ``erfc``
  underflow; and
* call-local variate generators (open-interval uniforms and Marsaglia-Tsang
  gamma variates in log form).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

_EPSILON = math.ulp(1.0)
_RESIDUAL_CLAMP = 1e200
_MAX_BRENT_ITERATIONS = 300
_MAX_BRACKET_EXPANSIONS = 64
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_TWO = math.sqrt(2.0)

LOG_MIN = -744.0
LOG_MAX = 709.0
SINH_LIMIT = 710.0
LOGIT_LIMIT = 744.0


def safe_log(value: float) -> float:
    """Return ``log(value)`` with ``log(0) = -inf``.

    Args:
        value: Non-negative probability or density value.

    Returns:
        Natural logarithm, negative infinity at zero.

    References:
        [REF-PYTHON-MATH]. No probability equation is evaluated.

    """
    return math.log(value) if value > 0.0 else -math.inf


def expit(t: float) -> float:
    """Return the logistic function ``1/(1+exp(-t))`` without overflow.

    Args:
        t: Finite real argument.

    Returns:
        Value in ``[0, 1]`` with full relative accuracy for negative ``t``.

    Equation:
        [EQ-DIST-LOGIS-002].

    References:
        [REF-JKB-1995, Chapter 23].

    """
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))
    exponential = math.exp(t)
    return exponential / (1.0 + exponential)


def _clamped(value: float) -> float:
    """Map a possibly infinite residual into a large finite range.

    Args:
        value: Residual value, possibly infinite.

    Returns:
        The residual clipped to ``[-1e200, 1e200]``.

    References:
        [REF-BRENT-1973, Chapter 4]. This is a floating-point guard only.

    """
    if value != value:
        raise ArithmeticError("quantile residual evaluated to NaN")
    return max(-_RESIDUAL_CLAMP, min(_RESIDUAL_CLAMP, value))


def brent_increasing(
    residual: Callable[[float], float],
    low: float,
    high: float,
    residual_low: float,
    residual_high: float,
) -> float:
    """Locate a root of a non-decreasing residual inside a sign-change bracket.

    This is Brent's zeroin combination of bisection, secant, and inverse
    quadratic interpolation with tolerance ``2 eps |t| + 5e-31``.

    Args:
        residual: Non-decreasing residual function of the coordinate ``t``.
        low: Bracket end with non-positive residual.
        high: Bracket end with non-negative residual.
        residual_low: Residual at ``low``.
        residual_high: Residual at ``high``.

    Returns:
        Coordinate ``t`` with a near-zero residual.

    Equation:
        [EQ-DIST-INV-001].

    References:
        [REF-BRENT-1973, Chapter 4].

    """
    a, b = low, high
    fa, fb = _clamped(residual_low), _clamped(residual_high)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c, fc = b, fb
    d = e = b - a
    for _ in range(_MAX_BRENT_ITERATIONS):
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tolerance = 2.0 * _EPSILON * abs(b) + 0.5e-30
        midpoint = 0.5 * (c - b)
        if abs(midpoint) <= tolerance or fb == 0.0:
            return b
        if abs(e) >= tolerance and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * midpoint * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * midpoint * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            limit = min(3.0 * midpoint * q - abs(tolerance * q), abs(e * q))
            if 2.0 * p < limit:
                e = d
                d = p / q
            else:
                d = midpoint
                e = d
        else:
            d = midpoint
            e = d
        a, fa = b, fb
        if abs(d) > tolerance:
            b += d
        else:
            b += math.copysign(tolerance, midpoint)
        fb = _clamped(residual(b))
    return b


def invert_increasing(
    residual: Callable[[float], float],
    start: float,
    minimum: float,
    maximum: float,
) -> float:
    """Bracket and solve ``residual(t) = 0`` for a non-decreasing residual.

    The bracket grows geometrically from ``start`` until the residual changes
    sign or a coordinate limit is reached; Brent's method then refines it.

    Args:
        residual: Non-decreasing residual of the coordinate ``t``.
        start: Initial coordinate guess.
        minimum: Finite lower coordinate limit.
        maximum: Finite upper coordinate limit.

    Returns:
        Root coordinate, or the violated limit when no sign change exists
        inside ``[minimum, maximum]``.

    Equation:
        [EQ-DIST-INV-001].

    References:
        [REF-BRENT-1973, Chapter 4].

    """
    origin = min(max(start, minimum), maximum)
    value = _clamped(residual(origin))
    if value == 0.0:
        return origin
    step = 1.0
    if value < 0.0:
        low, value_low = origin, value
        for _ in range(_MAX_BRACKET_EXPANSIONS):
            high = min(origin + step, maximum)
            value_high = _clamped(residual(high))
            if value_high >= 0.0:
                return brent_increasing(residual, low, high, value_low, value_high)
            if high >= maximum:
                return maximum
            low, value_low = high, value_high
            step *= 2.0
        return low
    high, value_high = origin, value
    for _ in range(_MAX_BRACKET_EXPANSIONS):
        low = max(origin - step, minimum)
        value_low = _clamped(residual(low))
        if value_low <= 0.0:
            return brent_increasing(residual, low, high, value_low, value_high)
        if low <= minimum:
            return minimum
        high, value_high = low, value_low
        step *= 2.0
    return high


def solve_tail_quantile(
    lower_probability: float,
    upper_probability: float,
    *,
    log_cdf: Callable[[float], float],
    log_sf: Callable[[float], float],
    to_x: Callable[[float], float],
    start: float,
    minimum: float,
    maximum: float,
) -> float:
    """Invert whichever tail carries the smaller probability, in log space.

    ``lower_probability`` and ``upper_probability`` must sum to one
    mathematically. The smaller one is matched through ``log F(x) = log p``
    or ``log S(x) = log q``, so far-tail quantiles keep relative accuracy.

    Args:
        lower_probability: Target ``p = F(x)`` in ``(0, 1)``.
        upper_probability: Target ``q = S(x)`` in ``(0, 1)``.
        log_cdf: Log cumulative distribution function of ``x``.
        log_sf: Log survival function of ``x``.
        to_x: Increasing map from coordinate ``t`` to the support.
        start: Initial coordinate guess.
        minimum: Finite lower coordinate limit.
        maximum: Finite upper coordinate limit.

    Returns:
        Quantile ``x`` in the distribution support.

    Equation:
        [EQ-DIST-INV-001], [EQ-DIST-ISF-001].

    References:
        [REF-BRENT-1973, Chapter 4], [REF-NIST-DLMF-GAMMA, Section 8.10].

    """
    if lower_probability <= upper_probability:
        target = math.log(lower_probability)

        def residual(t: float) -> float:
            """Return ``log F(x(t)) - log p``, with infinite ``x`` as limits.

            References:
                [EQ-DIST-INV-001], [REF-BRENT-1973].

            """
            point = to_x(t)
            if point == math.inf:
                return -target
            if point == -math.inf:
                return -math.inf
            return log_cdf(point) - target

    else:
        target = math.log(upper_probability)

        def residual(t: float) -> float:
            """Return ``log q - log S(x(t))``, with infinite ``x`` as limits.

            References:
                [EQ-DIST-INV-001], [REF-BRENT-1973].

            """
            point = to_x(t)
            if point == math.inf:
                return math.inf
            if point == -math.inf:
                return target
            return target - log_sf(point)

    return to_x(invert_increasing(residual, start, minimum, maximum))


def bounded_exp(t: float) -> float:
    """Return ``exp(t)`` for a coordinate limited to the finite float range.

    Args:
        t: Coordinate in ``[LOG_MIN, LOG_MAX]``.

    Returns:
        Positive value ``exp(t)``.

    References:
        [REF-PYTHON-MATH]. No probability equation is evaluated.

    """
    return math.exp(min(max(t, LOG_MIN), LOG_MAX))


def log_standard_normal_sf(z: float) -> float:
    """Return ``log(1 - Phi(z))`` beyond the underflow point of ``erfc``.

    For ``z < 35`` the value is ``log(erfc(z/sqrt 2)/2)``. Further out the
    Mills ratio ``R(z) = (1 - Phi(z))/phi(z)`` is evaluated by its continued
    fraction ``1/(z+1/(z+2/(z+3/(z+...))))``.

    Args:
        z: Real argument, possibly infinite.

    Returns:
        Natural logarithm of the standard-normal upper tail.

    Equation:
        [EQ-DIST-TN-004].

    References:
        [REF-NIST-DLMF, Sections 7.2 and 7.9].

    """
    if z == math.inf:
        return -math.inf
    if z == -math.inf:
        return 0.0
    if z < 35.0:
        if z < -5.0:
            return math.log1p(-0.5 * math.erfc(-z / _SQRT_TWO))
        return math.log(0.5 * math.erfc(z / _SQRT_TWO))
    fraction = z
    for index in range(40, 0, -1):
        fraction = z + index / fraction
    return -0.5 * z * z - _LOG_SQRT_TWO_PI - math.log(fraction)


def log_standard_normal_cdf(z: float) -> float:
    """Return ``log Phi(z)`` with full accuracy in the lower tail.

    Args:
        z: Real argument, possibly infinite.

    Returns:
        Natural logarithm of the standard-normal CDF.

    Equation:
        [EQ-DIST-TN-004].

    References:
        [REF-NIST-DLMF, Sections 7.2 and 7.9].

    """
    return log_standard_normal_sf(-z)


def log_diff_exp(larger: float, smaller: float) -> float:
    """Return ``log(exp(larger) - exp(smaller))`` for ``larger >= smaller``.

    Args:
        larger: Larger logarithm, finite or ``-inf``.
        smaller: Smaller logarithm, possibly ``-inf``.

    Returns:
        Log of the positive difference, ``-inf`` when the two coincide.

    References:
        [REF-PYTHON-MATH]. No probability equation is evaluated.

    """
    if larger == -math.inf or smaller >= larger:
        return -math.inf
    if smaller == -math.inf:
        return larger
    gap = smaller - larger
    if gap > -math.log(2.0):
        return larger + math.log(-math.expm1(gap))
    return larger + math.log1p(-math.exp(gap))


def log_normal_interval_mass(lower: float, upper: float) -> float:
    """Return ``log(Phi(upper) - Phi(lower))`` without tail cancellation.

    Intervals in one half-line subtract the smaller tail in log space; an
    interval containing zero uses ``erf``, whose two terms have opposite signs.

    Args:
        lower: Lower standardized endpoint, possibly ``-inf``.
        upper: Upper standardized endpoint, possibly ``inf``.

    Returns:
        Log standard-normal probability of ``[lower, upper]``.

    Equation:
        [EQ-DIST-TN-002], [EQ-DIST-TN-004].

    References:
        [REF-NIST-DLMF, Sections 7.2 and 7.9].

    """
    if upper <= lower:
        return -math.inf
    if lower >= 0.0:
        return log_diff_exp(
            log_standard_normal_sf(lower), log_standard_normal_sf(upper)
        )
    if upper <= 0.0:
        return log_diff_exp(
            log_standard_normal_cdf(upper), log_standard_normal_cdf(lower)
        )
    high = 1.0 if upper == math.inf else math.erf(upper / _SQRT_TWO)
    low = -1.0 if lower == -math.inf else math.erf(lower / _SQRT_TWO)
    return math.log(0.5 * (high - low))


def log_normal_short_mass(anchor: float, width: float) -> float:
    """Return ``log(Phi(anchor + width) - Phi(anchor))`` for a short interval.

    The integrand ``phi(anchor) exp(-anchor s - s^2/2)`` is expanded in its
    Taylor series ``sum c_n s^n`` with ``c_0 = 1``, ``c_1 = -anchor`` and
    ``(n+1) c_(n+1) = -anchor c_n - c_(n-1)``, then integrated termwise. The
    width enters directly, so no difference of nearly equal standardized
    points is formed. Intended for ``width (|anchor| + width) <= 1/2``.

    Args:
        anchor: Finite standardized interval start.
        width: Positive standardized interval width.

    Returns:
        Log standard-normal probability of ``[anchor, anchor + width]``.

    Equation:
        [EQ-DIST-TN-005].

    References:
        [REF-NIST-DLMF, Section 7.2], [REF-JKB-1994, Chapter 13].

    """
    previous, current = 0.0, 1.0
    power = width
    total = width
    small_terms = 0
    for order in range(1, 200):
        following = (-anchor * current - previous) / order
        power *= width
        term = following * power / (order + 1)
        total += term
        previous, current = current, following
        small_terms = small_terms + 1 if abs(term) <= 1e-17 * abs(total) else 0
        if small_terms >= 2:
            break
    return -0.5 * anchor * anchor - _LOG_SQRT_TWO_PI + math.log(total)


def open_unit(generator: random.Random) -> float:
    """Draw a uniform variate strictly inside ``(0, 1)``.

    Args:
        generator: Call-local pseudo-random generator.

    Returns:
        Uniform value excluding both endpoints.

    References:
        [REF-DEVROYE-1986, Chapter II], [REF-MATSUMOTO-NISHIMURA-1998].

    """
    while True:
        value = generator.random()
        if value > 0.0:
            return value


def log_standard_gamma_variate(shape: float, generator: random.Random) -> float:
    """Draw ``log G`` for ``G ~ Gamma(shape, 1)`` by Marsaglia-Tsang.

    For ``shape >= 1`` the squeeze-and-reject method with ``d = shape - 1/3``
    and ``c = 1/sqrt(9 d)`` returns ``d v``. For ``shape < 1`` the boost
    ``G = G' U^(1/shape)`` with ``G' ~ Gamma(shape + 1, 1)`` is applied in log
    space so tiny variates do not underflow before the caller combines them.

    Args:
        shape: Finite positive shape.
        generator: Call-local pseudo-random generator.

    Returns:
        Natural logarithm of one gamma variate.

    Equation:
        [EQ-DIST-GAM-005].

    References:
        [REF-MARSAGLIA-TSANG-2000].

    """
    if shape < 1.0:
        boosted = log_standard_gamma_variate(shape + 1.0, generator)
        return boosted + math.log(open_unit(generator)) / shape
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = generator.gauss(0.0, 1.0)
        v = 1.0 + c * x
        if v <= 0.0:
            continue
        v = v * v * v
        u = open_unit(generator)
        squared = x * x
        if u < 1.0 - 0.0331 * squared * squared:
            return math.log(d * v)
        if math.log(u) < 0.5 * squared + d * (1.0 - v + math.log(v)):
            return math.log(d * v)
