"""Internal special functions needed by probability-distribution algorithms."""

from __future__ import annotations

import math
from fractions import Fraction

from .constraints import NON_NEGATIVE_FINITE, POSITIVE_FINITE, PROBABILITY

_EPSILON = 8.0 * math.ulp(1.0)
_FPMIN = 1e-300
_MAX_ITERATIONS = 10_000
# Terms needed by the P(a, x) series near x ~ a grow like sqrt(a): the term
# ratio x/(a+k) makes the k-th term about exp(-k^2/(2a)), which falls below
# the convergence threshold after roughly 8.6 sqrt(a) terms.
_SERIES_SQRT_FACTOR = 12.0
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
# Above this shape the log prefactors are formed from Stirling differences
# instead of differences of lgamma values of size ~a log a.
LARGE_SHAPE = 100.0
_STIRLING_SERIES_START = 15.0
_TEMME_MIN_SHAPE = 1e6
_TEMME_MAX_OFFSET = 0.3
# Taylor coefficients of Temme's c_0(eta) = 1/(lambda-1) - 1/eta and
# c_1(eta) = 1/eta^3 - 1/(lambda-1)^3 - 1/(lambda-1)^2 - 1/(12(lambda-1)),
# derived in exact rational arithmetic by reverting
# mu - log(1 + mu) = eta^2/2 (mu = lambda - 1) and checked against the closed
# forms in 80-digit arithmetic. Eighteen terms leave a remainder below 1e-17
# for |eta| <= 0.36 (the region |lambda - 1| <= 0.3).
_TEMME_C0 = (
    -1.0 / 3.0,
    1.0 / 12.0,
    -2.0 / 135.0,
    1.0 / 864.0,
    1.0 / 2835.0,
    -139.0 / 777600.0,
    1.0 / 25515.0,
    -571.0 / 261273600.0,
    -281.0 / 151559100.0,
    163879.0 / 197522841600.0,
    -5221.0 / 29554024500.0,
    5246819.0 / 782190452736000.0,
    5459.0 / 531972441000.0,
    -534703531.0 / 122021710626816000.0,
    91207079.0 / 99704934754425000.0,
    -4483131259.0 / 175711263302615040000.0,
    -2650986803.0 / 45465450248017800000.0,
    432261921612371.0 / 17743323368298066739200000.0,
)
_TEMME_C1 = (
    -1.0 / 540.0,
    -1.0 / 288.0,
    1.0 / 378.0,
    -77.0 / 77760.0,
    1.0 / 4860.0,
    -1.0 / 2488320.0,
    -2743.0 / 151559100.0,
    41969.0 / 5486745600.0,
    -11.0 / 6823440.0,
    47207.0 / 10158317568000.0,
    3761.0 / 27280638000.0,
    -3599669.0 / 62575236218880.0,
    61903187.0 / 5179477130100000.0,
    -4193939.0 / 239062943268864000.0,
    -2570401.0 / 2547084047508000.0,
    54310133948197.0 / 130465613002191667200000.0,
    -2459127719.0 / 28715021209274400000.0,
    28105097117.0 / 463228955069786357760000.0,
)
# B_(2k) / (2k (2k - 1)) for k = 1..7: coefficients of a^(1-2k) in the
# Stirling error lgamma(a) - (a - 1/2) log a + a - log sqrt(2 pi).
_STIRLING_COEFFICIENTS = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
)


def stirling_error(a: float) -> float:
    """Return ``lgamma(a) - [(a - 1/2) log a - a + log sqrt(2 pi)]``.

    For ``a >= 15`` the asymptotic series
    ``sum_(k=1..7) B_(2k) / (2k (2k-1) a^(2k-1))`` is summed directly (its
    truncation error is below ``4e-18`` there); smaller arguments use the
    defining difference, whose terms are then of modest size.

    Args:
        a: Finite positive argument.

    Returns:
        The Stirling-series remainder, positive and of order ``1/(12 a)``
        for large ``a``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    if a < _STIRLING_SERIES_START:
        return math.lgamma(a) - (a - 0.5) * math.log(a) + a - _LOG_SQRT_TWO_PI
    inverse_square = 1.0 / (a * a)
    total = 0.0
    for coefficient in reversed(_STIRLING_COEFFICIENTS):
        total = total * inverse_square + coefficient
    return total / a


def log1pmx(t: float) -> float:
    """Return ``log(1 + t) - t`` without cancellation for small ``|t|``.

    For ``|t| <= 1/2`` the identity ``log(1+t) - t = -t y + 2 sum_(k>=1)
    y^(2k+1)/(2k+1)`` with ``y = t/(2+t)`` is summed (``|y| <= 1/3``, so the
    series gains about one decimal digit per term); otherwise the direct
    difference is used.

    Args:
        t: Real argument greater than ``-1``.

    Returns:
        Non-positive value ``log(1 + t) - t``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1992] (function ``rlog1``).

    """
    if abs(t) > 0.5:
        return math.log1p(t) - t
    y = t / (2.0 + t)
    square = y * y
    power = y * square
    series = 0.0
    denominator = 3.0
    while True:
        term = power / denominator
        series += term
        if abs(term) <= 1e-17 * abs(series):
            break
        power *= square
        denominator += 2.0
    return 2.0 * series - t * y


def log_gamma_kernel(a: float, x: float, rate: float = 1.0) -> float:
    """Return ``a log y - y - lgamma(a)`` at ``y = rate x``, accurately for large ``a``.

    Below :data:`LARGE_SHAPE` the three terms are added directly. Above it the
    equivalent form ``a [log(y/a) - (y/a - 1)] + log(a)/2 - log sqrt(2 pi) -
    stirling_error(a)`` is used: every term is then of order one near the
    mode instead of order ``a log a``, so ``exp`` of the result keeps full
    relative accuracy for shapes up to ``1e15`` and beyond. When
    ``rate != 1`` the offset ``rate x - a`` is formed in exact rational
    arithmetic, so the product ``rate x`` is never rounded first.

    Args:
        a: Finite positive shape.
        x: Finite positive point.
        rate: Finite positive multiplier; ``rate x`` must be finite.

    Returns:
        Logarithm of ``y^a exp(-y) / Gamma(a)`` with ``y = rate x``.

    Equation:
        [EQ-NUM-STIRLING-001], [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11], [REF-DIDONATO-MORRIS-1992].

    """
    if a < LARGE_SHAPE:
        if rate == 1.0:
            return -x + a * math.log(x) - math.lgamma(a)
        scaled = rate * x
        return -scaled + a * (math.log(rate) + math.log(x)) - math.lgamma(a)
    if rate == 1.0:
        offset = x - a
    else:
        offset = float(Fraction(rate) * Fraction(x) - Fraction(a))
    t = offset / a
    if abs(t) <= 0.5:
        core = a * log1pmx(t)
    else:
        core = a * (math.log(rate) + math.log(x) - math.log(a)) - offset
    return core + 0.5 * math.log(a) - _LOG_SQRT_TWO_PI - stirling_error(a)


def log_gamma_ratio(a: float, b: float) -> float:
    """Return ``lgamma(a + b) - lgamma(a)`` accurately when ``a`` is large.

    Above :data:`LARGE_SHAPE` the identity ``b log a - b + (a + b - 1/2)
    log(1 + b/a) + stirling_error(a + b) - stirling_error(a)`` replaces the
    difference of two numbers of size ``a log a``.

    Args:
        a: Finite positive argument.
        b: Finite positive increment.

    Returns:
        ``log(Gamma(a + b) / Gamma(a))``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    if a < LARGE_SHAPE:
        return math.lgamma(a + b) - math.lgamma(a)
    return (
        b * math.log(a)
        - b
        + (a + b - 0.5) * math.log1p(b / a)
        + stirling_error(a + b)
        - stirling_error(a)
    )


def log_beta_kernel(
    a: float,
    b: float,
    x: float,
    complement: float,
    offset: Fraction | None = None,
) -> float:
    """Return ``a log x + b log(1 - x) - log B(a, b)`` without cancellation.

    Below :data:`LARGE_SHAPE` (for the larger shape) the terms are added
    directly. Otherwise, with ``1 + t = x (a+b)/a`` and ``1 + s = (1-x)
    (a+b)/b`` (so ``a t + b s = 0``), the kernel equals
    ``a [log(1+t) - t] + b [log(1+s) - s] + log(ab/(a+b))/2 - log sqrt(2 pi)
    - stirling_error(a) - stirling_error(b) + stirling_error(a+b)``. Both
    bracketed terms are non-positive, so nothing cancels. The shared offset
    ``d = x (a+b) - a`` is formed in exact rational arithmetic from whichever
    of ``x`` and ``1 - x`` is the smaller (and hence more accurate) input.

    Args:
        a: Finite positive first shape.
        b: Finite positive second shape.
        x: Point strictly inside ``(0, 1)``.
        complement: ``1 - x`` formed without cancellation.
        offset: Optional exact value of ``x (a + b) - a`` supplied by callers
            that know ``x`` only as a rounded ratio of exact inputs (for
            example the F law, where it equals ``ab (f - 1)/(af + b)``).

    Returns:
        Logarithm of ``x^a (1 - x)^b / B(a, b)``.

    Equation:
        [EQ-NUM-STIRLING-001], [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17], [REF-DIDONATO-MORRIS-1992].

    """
    log_point = math.log(x) if x <= 0.5 else math.log1p(-complement)
    log_other = math.log(complement) if complement <= 0.5 else math.log1p(-x)
    if max(a, b) < LARGE_SHAPE:
        return (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * log_point
            + b * log_other
        )
    first, second = Fraction(a), Fraction(b)
    if offset is None:
        total = first + second
        if x <= 0.5:
            offset = Fraction(x) * total - first
        else:
            offset = second - Fraction(complement) * total
    t = float(offset / first)
    s = float(-offset / second)
    if abs(t) <= 0.5:
        first_part = a * log1pmx(t)
    else:
        # log(1 + t) = log(x (a+b)/a); the logarithm of x is already accurate.
        first_part = a * (log_point + math.log1p(b / a) - t)
    if abs(s) <= 0.5:
        second_part = b * log1pmx(s)
    else:
        second_part = b * (log_other + math.log1p(a / b) - s)
    return (
        first_part
        + second_part
        + 0.5 * (math.log(a) + math.log(b) - math.log(a + b))
        - _LOG_SQRT_TWO_PI
        - stirling_error(a)
        - stirling_error(b)
        + stirling_error(a + b)
    )


def log_beta_function(a: float, b: float) -> float:
    """Return ``log B(a, b)`` with Stirling differences for large arguments.

    Args:
        a: Finite positive first argument.
        b: Finite positive second argument.

    Returns:
        ``lgamma(a) + lgamma(b) - lgamma(a + b)``; when the larger argument
        exceeds :data:`LARGE_SHAPE` the large-argument pair is combined
        through :func:`log_gamma_ratio` or the Stirling form, so the result
        keeps absolute accuracy near ``1e-15`` even when it is small compared
        with ``a log a``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 5.11 and 5.12].

    """
    small, large = min(a, b), max(a, b)
    if large < LARGE_SHAPE:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if small < LARGE_SHAPE:
        return math.lgamma(small) - log_gamma_ratio(large, small)
    return (
        _LOG_SQRT_TWO_PI
        + (small - 0.5) * math.log(small)
        + (large - 0.5) * math.log(large)
        - (a + b - 0.5) * math.log(a + b)
        + stirling_error(small)
        + stirling_error(large)
        - stirling_error(a + b)
    )


def regularized_gamma_p(a: float, x: float) -> float:
    """Return the lower regularized incomplete gamma function ``P(a, x)``.

    A convergent series is used for ``x < a + 1``; otherwise the complement of
    a modified-Lentz continued fraction for ``Q(a, x)`` is used. For
    ``a >= 1e6`` and ``|x - a| <= 0.3 a``, where both expansions would need
    ``O(sqrt(a))`` terms, Temme's uniform asymptotic expansion is used.

    Args:
        a: Finite positive shape parameter.
        x: Finite non-negative evaluation point.

    Returns:
        A value clamped to ``[0, 1]`` against floating-point roundoff.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain.
        ArithmeticError: If the selected expansion does not converge.

    Equation:
        [EQ-NUM-GAMMA-001], [EQ-NUM-GAMMA-002].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 8.2, 8.9, and 8.12], [REF-TEMME-1979].

    """
    shape = POSITIVE_FINITE.validate(a, name="a")
    point = NON_NEGATIVE_FINITE.validate(x, name="x")
    if point == 0.0:
        return 0.0
    if _use_temme(shape, point):
        return _gamma_temme(shape, point)[0]
    if point < shape + 1.0:
        return _gamma_series_p(shape, point)
    return _clamp_probability(1.0 - _gamma_continued_fraction_q(shape, point))


def regularized_gamma_q(a: float, x: float) -> float:
    """Return the upper regularized incomplete gamma function ``Q(a, x)``.

    Args:
        a: Finite positive shape parameter.
        x: Finite non-negative evaluation point.

    Returns:
        The complement ``1 - P(a, x)`` evaluated through the numerically
        appropriate expansion (series, continued fraction, or Temme's uniform
        expansion for ``a >= 1e6`` near ``x = a``) and clamped to ``[0, 1]``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain.
        ArithmeticError: If the selected expansion does not converge.

    Equation:
        [EQ-NUM-GAMMA-001], [EQ-NUM-GAMMA-002].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 8.2, 8.9, and 8.12], [REF-TEMME-1979].

    """
    shape = POSITIVE_FINITE.validate(a, name="a")
    point = NON_NEGATIVE_FINITE.validate(x, name="x")
    if point == 0.0:
        return 1.0
    if _use_temme(shape, point):
        return _gamma_temme(shape, point)[1]
    if point < shape + 1.0:
        return _clamp_probability(1.0 - _gamma_series_p(shape, point))
    return _gamma_continued_fraction_q(shape, point)


def _use_temme(a: float, x: float) -> bool:
    """Return whether Temme's uniform expansion should evaluate ``P, Q``.

    Args:
        a: Validated positive shape.
        x: Validated positive point.

    Returns:
        ``True`` for ``a >= 1e6`` and ``|x - a| <= 0.3 a``.

    References:
        [EQ-NUM-GAMMA-002], [REF-TEMME-1979].

    """
    return a >= _TEMME_MIN_SHAPE and abs(x - a) <= _TEMME_MAX_OFFSET * a


def _gamma_temme(a: float, x: float) -> tuple[float, float]:
    """Evaluate ``(P(a, x), Q(a, x))`` by Temme's uniform asymptotic expansion.

    With ``lambda = x/a`` and ``eta = sign(lambda - 1) sqrt(2 (lambda - 1 -
    log lambda))``, ``Q = erfc(eta sqrt(a/2))/2 + R`` and
    ``P = erfc(-eta sqrt(a/2))/2 - R``, where
    ``R = exp(-a eta^2/2) / sqrt(2 pi a) (c_0(eta) + c_1(eta)/a)``. The
    coefficients are summed from their Taylor series in ``eta`` (exact
    rational coefficients, convergent for ``|eta| < 2 sqrt(pi)``); the
    omitted ``c_2(eta)/a^2`` term (``c_2(0) = 25/6048``) is below ``5e-15``
    relative to ``R`` for ``a >= 1e6``.

    Args:
        a: Shape with ``a >= 1e6``.
        x: Point with ``|x - a| <= 0.3 a``.

    Returns:
        The pair ``(P, Q)``, each clamped to ``[0, 1]``.

    Equation:
        [EQ-NUM-GAMMA-002].

    References:
        [REF-TEMME-1979], [REF-NIST-DLMF-GAMMA, Section 8.12].

    """
    t = (x - a) / a
    half_square = -log1pmx(t)
    eta = math.copysign(math.sqrt(2.0 * half_square), t)
    c0 = 0.0
    for coefficient in reversed(_TEMME_C0):
        c0 = c0 * eta + coefficient
    c1 = 0.0
    for coefficient in reversed(_TEMME_C1):
        c1 = c1 * eta + coefficient
    remainder = (
        math.exp(-a * half_square) / math.sqrt(2.0 * math.pi * a) * (c0 + c1 / a)
    )
    argument = eta * math.sqrt(0.5 * a)
    lower = 0.5 * math.erfc(-argument) - remainder
    upper = 0.5 * math.erfc(argument) + remainder
    return _clamp_probability(lower), _clamp_probability(upper)


def _gamma_series_p(a: float, x: float) -> float:
    """Evaluate ``P(a, x)`` through its convergent power series.

    Args:
        a: Validated positive shape.
        x: Validated positive point satisfying the series branch criterion.

    Returns:
        Lower regularized incomplete gamma value.

    Raises:
        ArithmeticError: If convergence is not reached within the iteration
            cap ``10_000 + 12 sqrt(a)``. Near ``x ~ a`` the series needs about
            ``8.6 sqrt(a)`` terms, so the work grows like ``sqrt(a)`` for very
            large shapes; the prefactor ``x^a e^-x / Gamma(a)`` comes from
            :func:`log_gamma_kernel`.

    Equation:
        [EQ-NUM-GAMMA-001], [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.9].

    """
    term = 1.0 / a
    total = term
    denominator = a
    limit = _MAX_ITERATIONS + int(_SERIES_SQRT_FACTOR * math.sqrt(a))
    for _ in range(limit):
        denominator += 1.0
        term *= x / denominator
        total += term
        if abs(term) <= abs(total) * _EPSILON:
            return _clamp_probability(total * math.exp(log_gamma_kernel(a, x)))
    raise ArithmeticError("regularized-gamma series did not converge")


def _gamma_continued_fraction_q(a: float, x: float) -> float:
    """Evaluate ``Q(a, x)`` with a stabilized continued fraction.

    Args:
        a: Validated positive shape.
        x: Validated positive point satisfying the continued-fraction criterion.

    Returns:
        Upper regularized incomplete gamma value.

    Raises:
        ArithmeticError: If convergence is not reached within the iteration cap.

    Equation:
        [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.9].

    """
    b = x + 1.0 - a
    if abs(b) < _FPMIN:
        b = _FPMIN
    c = 1.0 / _FPMIN
    d = 1.0 / b
    fraction = d
    limit = _MAX_ITERATIONS + int(_SERIES_SQRT_FACTOR * math.sqrt(a))
    for index in range(1, limit + 1):
        coefficient = -float(index) * (float(index) - a)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + coefficient / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= _EPSILON:
            return _clamp_probability(math.exp(log_gamma_kernel(a, x)) * fraction)
    raise ArithmeticError("regularized-gamma continued fraction did not converge")


def _clamp_probability(value: float) -> float:
    """Clamp a nearly valid computed probability to the unit interval.

    Args:
        value: Floating result from a converged special-function expansion.

    Returns:
        ``value`` restricted to ``[0, 1]``.

    References:
        [REF-NIST-DLMF-GAMMA]. This is a roundoff guard and introduces no new
        mathematical equation.

    """
    return max(0.0, min(1.0, value))


def regularized_beta(x: float, a: float, b: float) -> float:
    """Return the regularized incomplete beta function ``I_x(a, b)``.

    The implementation chooses the direct or symmetry-transformed continued
    fraction so the smaller tail is evaluated before any complement.

    Args:
        x: Finite evaluation point in ``[0, 1]``.
        a: Finite positive first shape parameter.
        b: Finite positive second shape parameter.

    Returns:
        A value clamped to ``[0, 1]`` against floating-point roundoff.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain.
        ArithmeticError: If the continued fraction does not converge.

    Equation:
        [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Sections 8.17 and 8.18].

    """
    point = PROBABILITY.validate(x, name="x")
    return regularized_beta_pair(point, 1.0 - point, a, b)


def regularized_beta_pair(x: float, complement: float, a: float, b: float) -> float:
    """Return ``I_x(a, b)`` when ``x`` and ``1 - x`` are both supplied.

    Callers that can form ``1 - x`` directly (for example ``z^2/(nu + z^2)``
    beside ``nu/(nu + z^2)``) pass it as ``complement``, so neither the
    logarithmic prefactor nor the symmetry-branch continued fraction has to
    recover a small quantity from a difference with one.

    Args:
        x: Finite evaluation point in ``[0, 1]``.
        complement: ``1 - x`` evaluated without cancellation.
        a: Finite positive first shape parameter.
        b: Finite positive second shape parameter.

    Returns:
        A value clamped to ``[0, 1]`` against floating-point roundoff.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain or
            ``|x + complement - 1| > 1e-12``.
        ArithmeticError: If the continued fraction does not converge.

    Equation:
        [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Sections 8.17 and 8.18].

    """
    point = PROBABILITY.validate(x, name="x")
    other = PROBABILITY.validate(complement, name="complement")
    first_shape = POSITIVE_FINITE.validate(a, name="a")
    second_shape = POSITIVE_FINITE.validate(b, name="b")
    if abs(point + other - 1.0) > 1e-12:
        raise ValueError("complement must equal 1 - x")
    if point == 0.0:
        return 0.0
    if other == 0.0:
        return 1.0
    series = _beta_large_shape(first_shape, second_shape, point, other)
    if series is not None:
        return series
    log_front = log_beta_kernel(first_shape, second_shape, point, other)
    split = (first_shape + 1.0) / (first_shape + second_shape + 2.0)
    if point < split:
        result = (
            math.exp(log_front)
            * _beta_continued_fraction(first_shape, second_shape, point)
            / first_shape
        )
        return _clamp_probability(result)
    reflected = (
        math.exp(log_front)
        * _beta_continued_fraction(second_shape, first_shape, other)
        / second_shape
    )
    return _clamp_probability(1.0 - reflected)


_LOG_SINHC_TERMS = 80
_BETA_SERIES_MIN_SHAPE = 1e4
# -log(1 - t) <= 1, the region where the sinh series converges quickly.
_BETA_SERIES_MAX_POINT = -math.expm1(-1.0)
_log_sinhc_cache: list[float] = []


def _beta_large_shape(a: float, b: float, x: float, y: float) -> float | None:
    """Return ``I_x(a, b)`` from the large-shape gamma series when it applies.

    The series of :func:`_beta_gamma_series` is used when the larger shape is
    at least ``1e4``, the point on the side of the smaller shape satisfies
    ``-log(1 - t) <= 1``, and the first correction ratio
    ``|s - 1| s (s + 1) / (24 N^2)`` is at most one (so for ``s`` up to about
    ``(24 L^2)^(1/3)``; for example ``s <= 1.3e3`` at ``L = 1e4``). There the continued fraction converges slowly near
    ``x ~ a/(a+b)`` and its local stopping test (``|delta - 1| <= 8 eps``)
    is met while many factors ``1 + O(1/a)`` remain, which cost up to three
    significant digits for ``a ~ 1e13``. Of the pair ``I_t(s, L)`` and its
    complement the smaller one is computed and the other obtained by
    subtraction from one.

    Args:
        a: Validated first shape.
        b: Validated second shape.
        x: Point strictly inside ``(0, 1)``.
        y: ``1 - x`` formed without cancellation.

    Returns:
        ``I_x(a, b)``, or ``None`` when the series does not apply.

    Equation:
        [EQ-NUM-BETA-002].

    References:
        [REF-DIDONATO-MORRIS-1992], [REF-NIST-DLMF-BETA, Section 8.17].

    """
    if max(a, b) < _BETA_SERIES_MIN_SHAPE:
        return None
    if a >= b:
        large, small, point, complement = a, b, y, x
    else:
        large, small, point, complement = b, a, x, y
    if point > _BETA_SERIES_MAX_POINT:
        return None
    big_n = large + 0.5 * (small - 1.0)
    if abs(small - 1.0) * small * (small + 1.0) > 24.0 * big_n * big_n:
        # The first correction e_1 Gamma(s+2)/(Gamma(s) N^2) exceeds one:
        # both shapes are large and the series would need too many terms.
        return None
    lower_side = _beta_gamma_series(large, small, point, complement, upper=True)
    if lower_side is None:
        return None
    if lower_side > 0.5:
        other = _beta_gamma_series(large, small, point, complement, upper=False)
        if other is None:
            return None
        lower_side = 1.0 - other
        upper_side = other
    else:
        upper_side = 1.0 - lower_side
    # lower_side = I_t(small, large); upper_side = I_(1-t)(large, small).
    return _clamp_probability(upper_side if a >= b else lower_side)


def _log_sinhc_coefficients() -> list[float]:
    """Return ``l_k = B_(2k) / (2k (2k)!)``, the series of ``log(sinh(v/2)/(v/2))``.

    ``log(sinh(v/2)/(v/2)) = sum_(k>=1) l_k v^(2k)`` for ``|v| < 2 pi``. The
    Bernoulli numbers are generated once in exact rational arithmetic
    (Akiyama-Tanigawa recurrence) and cached as floats.

    Returns:
        Coefficients ``l_1, ..., l_80`` (``l_1 = 1/24``, ``l_2 = -1/2880``).

    Equation:
        [EQ-NUM-BETA-002].

    References:
        [REF-NIST-DLMF, Sections 4.19 and 24.2] (the hyperbolic form follows
        from ``log(sin z / z)`` with ``z -> i z``).

    """
    if _log_sinhc_cache:
        return _log_sinhc_cache
    order = 2 * _LOG_SINHC_TERMS
    row = [Fraction(0)] * (order + 1)
    bernoulli = [Fraction(0)] * (order + 1)
    for m in range(order + 1):
        row[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            row[j - 1] = j * (row[j - 1] - row[j])
        bernoulli[m] = row[0]
    factorial = Fraction(1)
    for k in range(1, _LOG_SINHC_TERMS + 1):
        factorial *= (2 * k - 1) * (2 * k)
        _log_sinhc_cache.append(float(bernoulli[2 * k] / (2 * k * factorial)))
    return _log_sinhc_cache


def _beta_gamma_series(
    large: float, small: float, point: float, complement: float, *, upper: bool
) -> float | None:
    """Return ``I_t(s, L)`` or ``I_(1-t)(L, s)`` by a gamma-function series.

    With ``v = -log(1 - u)`` the beta integral becomes
    ``I_t(s, L) = Gamma(s+L)/(Gamma(s) Gamma(L)) int_0^w v^(s-1)
    phi(v) exp(-N v) dv`` with ``w = -log(1 - t)``, ``N = L + (s-1)/2`` and
    ``phi(v) = (sinh(v/2)/(v/2))^(s-1) = sum_n e_n v^(2n)`` (convergent for
    ``|v| < 2 pi``). Integrating term by term gives the convergent series

    ``I_t(s, L) = Gamma(s+L)/(Gamma(L) N^s) sum_n e_n [Gamma(s+2n)/Gamma(s)]
    N^(-2n) P(s + 2n, N w)``,

    and the same terms with ``Q`` in place of ``P`` give the complementary
    integral over ``[w, inf)``, an asymptotic expansion in ``1/N`` that is
    truncated before its terms grow (the large-``L`` expansion behind
    DiDonato and Morris's ``BGRAT``). The ``P`` series has non-negative terms
    for ``s >= 1``. The prefactor's logarithm ``L log1pmx(s/L) + (s - 1/2)
    log(1 + s/L) - s log(1 + (s-1)/(2L)) + stirling(L+s) - stirling(L)``
    is formed from small terms only, so no ``s log L``-sized cancellation
    occurs. ``e_n`` follow from ``n e_n = (s - 1) sum_(k=1..n) k l_k
    e_(n-k)`` with the coefficients of :func:`_log_sinhc_coefficients`.

    Args:
        large: Large shape ``L``.
        small: Small shape ``s``.
        point: Small-side point ``t`` with ``-log(1 - t) <= 1``.
        complement: ``1 - t`` formed without cancellation.
        upper: Return ``I_t(s, L)`` (``P`` series) when true, else
            ``I_(1-t)(L, s)`` (``Q`` series).

    Returns:
        The requested value (accurate in relative terms when it is the
        smaller of the pair), or ``None`` if the series does not settle within
        80 terms.

    Equation:
        [EQ-NUM-BETA-002], [EQ-NUM-GAMMA-001], [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1992], [REF-NIST-DLMF-BETA, Section 8.17].

    """
    big_n = large + 0.5 * (small - 1.0)
    w = -math.log1p(-point) if point <= 0.5 else -math.log(complement)
    z = big_n * w
    ratio = small / large
    log_prefactor = (
        large * log1pmx(ratio)
        + (small - 0.5) * math.log1p(ratio)
        - small * math.log1p(0.5 * (small - 1.0) / large)
        + stirling_error(large + small)
        - stirling_error(large)
    )
    coefficients = _log_sinhc_coefficients()
    exponent = small - 1.0
    e_values = [1.0]
    weight = 1.0
    total = 0.0
    previous = math.inf
    for n in range(_LOG_SINHC_TERMS):
        shape = small + 2.0 * n
        if n > 0:
            following = math.fsum(
                k * coefficients[k - 1] * e_values[n - k] for k in range(1, n + 1)
            )
            e_values.append(exponent * following / n)
            weight *= (shape - 2.0) * (shape - 1.0) / (big_n * big_n)
        factor = e_values[n] * weight
        if upper:
            term = factor * regularized_gamma_p(shape, z)
        else:
            term = factor * regularized_gamma_q(shape, z)
            if n > 1 and abs(term) > abs(previous):
                break
            previous = term
        total += term
        if n > 0 and abs(term) <= 1e-17 * abs(total):
            break
    else:
        return None
    return _clamp_probability(math.exp(log_prefactor) * total)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued-fraction factor.

    Args:
        a: Validated positive first shape.
        b: Validated positive second shape.
        x: Validated point strictly between zero and one.

    Returns:
        Continued-fraction multiplier used by ``I_x(a,b)``.

    Raises:
        ArithmeticError: If convergence is not reached within the iteration cap.

    Equation:
        [EQ-NUM-BETA-001].

    References:
        [REF-NIST-DLMF-BETA, Section 8.17].

    """
    total_shape = a + b
    a_plus_one = a + 1.0
    a_minus_one = a - 1.0
    c = 1.0
    d = 1.0 - total_shape * x / a_plus_one
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    fraction = d
    for index in range(1, _MAX_ITERATIONS + 1):
        doubled = 2.0 * index
        coefficient = (
            index * (b - index) * x / ((a_minus_one + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        fraction *= d * c

        coefficient = -(
            (a + index)
            * (total_shape + index)
            * x
            / ((a + doubled) * (a_plus_one + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= _EPSILON:
            return fraction
    raise ArithmeticError("regularized-beta continued fraction did not converge")
