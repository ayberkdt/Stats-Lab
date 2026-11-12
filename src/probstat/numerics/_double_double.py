"""Double-double arithmetic for the exponents of special-function prefactors.

A double-double number is an unevaluated sum ``hi + lo`` of two floats with
``|lo| <= ulp(hi)/2``, which carries about 106 significant bits. The
error-free transformations ``two_sum`` (Knuth) and ``two_product`` (Dekker,
with Veltkamp's splitting) produce the exact rounding error of one float
addition or multiplication; the double-double operations built on them have
relative errors of a few ``2^-104`` [EQ-NUM-DD-001].

The kernels below need such numbers only where a float result is the
exponential or the complementary error function of a quantity that is
rounded, or that cancels, before the outer function is applied: the
relative error of ``exp(E)`` is the absolute error of ``E``, and the
relative error of ``erfc(w)`` is about ``2 w^2`` times the relative error of
``w``. Forming ``E`` and ``w`` with an absolute error far below ``2^-53``
removes that growth. The logarithm is table driven (``log(j/64)`` stored as
double-doubles) so that one evaluation costs a few double-double operations
[EQ-NUM-DD-002].
"""

from __future__ import annotations

import math
from decimal import Context, Decimal
from functools import lru_cache

DoubleDouble = tuple[float, float]

#: ``log sqrt(2 pi)`` split into a float and its rounding error.
LOG_SQRT_TWO_PI = (0.9189385332046728, -3.8782941580672414e-17)
#: ``1/sqrt(2)`` split into a float and its rounding error.
INV_SQRT_TWO = (0.7071067811865476, -4.833646656726457e-17)
# Cody-Waite split of ln 2: the leading part has 32 significant bits, so
# k * _LN2_HIGH is exact for |k| < 2^21; ln 2 - _LN2_HIGH - _LN2_LOW ~ 1e-26.
_LN2_HIGH = 6.93147180369123816490e-01
_LN2_LOW = 1.90821492927058770002e-10
_TWO_OVER_SQRT_PI = 1.1283791670955126
_TWO_THIRDS = (0.6666666666666666, 3.700743415417188e-17)
_SPLITTER = 134217729.0  # 2^27 + 1
_SPLIT_LIMIT = 1e300
_SQRT_HALF = math.sqrt(0.5)
_TABLE_SCALE = 64.0
_LOG1PMX_SERIES_LIMIT = 1.0 / 16.0
_ERFC_UNDERFLOW = 27.5
# 21, 19, ..., 5: the bracket y^4/5 + ... + y^20/21 of dd_log1pmx (Horner order).
_ODD_DENOMINATORS = tuple(float(k) for k in range(21, 4, -2))
#: Shapes from which ``stirling_error`` is summed from its asymptotic series.
STIRLING_SERIES_START = 15.0
# B_(2k) / (2k (2k - 1)) for k = 1..7 (DLMF 5.11.1).
_STIRLING_COEFFICIENTS = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
)
_log_table: dict[int, DoubleDouble] = {}


def two_sum(a: float, b: float) -> DoubleDouble:
    """Return ``(s, e)`` with ``s = fl(a + b)`` and ``s + e = a + b`` exactly.

    Args:
        a: Finite float.
        b: Finite float.

    Returns:
        The rounded sum and its exact rounding error.

    Equation:
        [EQ-NUM-DD-001].

    References:
        [REF-HIGHAM-2002, Section 4.3], [REF-DEKKER-1971].

    """
    total = a + b
    virtual = total - a
    return total, (a - (total - virtual)) + (b - virtual)


def _fast_two_sum(a: float, b: float) -> DoubleDouble:
    """Return the exact sum ``a + b`` as ``(s, e)`` when ``|a| >= |b|``.

    Args:
        a: Float of the larger magnitude (or zero).
        b: Float of the smaller magnitude.

    Returns:
        Normalized double-double sum.

    Equation:
        [EQ-NUM-DD-001].

    References:
        [REF-DEKKER-1971].

    """
    total = a + b
    return total, b - (total - a)


def two_product(a: float, b: float) -> DoubleDouble:
    """Return ``(p, e)`` with ``p = fl(a b)`` and ``p + e = a b`` exactly.

    Veltkamp's splitting divides each factor into halves of at most 26
    significant bits, whose products are exact.

    Args:
        a: Finite float.
        b: Finite float.

    Returns:
        The rounded product and its exact rounding error. The error part is
        zero when a factor exceeds ``1e300`` (where splitting would
        overflow) or the product is not finite; it is inexact only when
        partial products underflow, below ``1e-290``.

    Equation:
        [EQ-NUM-DD-001].

    References:
        [REF-DEKKER-1971].

    """
    product = a * b
    if (
        not (-_SPLIT_LIMIT <= a <= _SPLIT_LIMIT and -_SPLIT_LIMIT <= b <= _SPLIT_LIMIT)
        or product - product != 0.0
    ):
        return product, 0.0
    scaled = _SPLITTER * a
    a_high = scaled - (scaled - a)
    a_low = a - a_high
    scaled = _SPLITTER * b
    b_high = scaled - (scaled - b)
    b_low = b - b_high
    error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + (
        a_low * b_low
    )
    return product, error


def dd_add(x: DoubleDouble, y: DoubleDouble) -> DoubleDouble:
    """Return the double-double sum ``x + y``.

    Args:
        x: Double-double summand.
        y: Double-double summand.

    Returns:
        Normalized sum; its absolute error is a few ``2^-104 max(|x|, |y|)``.

    Equation:
        [EQ-NUM-DD-001].

    References:
        [REF-DEKKER-1971].

    """
    first, second = x[0], y[0]
    total = first + second
    virtual = total - first
    error = (first - (total - virtual)) + (second - virtual) + x[1] + y[1]
    head = total + error
    return head, error - (head - total)


def dd_mul(x: DoubleDouble, y: DoubleDouble) -> DoubleDouble:
    """Return the double-double product ``x y``.

    Args:
        x: Double-double factor.
        y: Double-double factor.

    Returns:
        Normalized product with relative error of a few ``2^-104``.

    Equation:
        [EQ-NUM-DD-001].

    References:
        [REF-DEKKER-1971].

    """
    product, error = two_product(x[0], y[0])
    error += x[0] * y[1] + x[1] * y[0]
    head = product + error
    return head, error - (head - product)


def dd_div(x: DoubleDouble, y: DoubleDouble) -> DoubleDouble:
    """Return the double-double quotient ``x / y`` by one correction step.

    Args:
        x: Double-double numerator.
        y: Double-double denominator with ``y[0] != 0``.

    Returns:
        Normalized quotient with relative error of a few ``2^-104``.

    Equation:
        [EQ-NUM-DD-001].

    References:
        [REF-DEKKER-1971].

    """
    first = x[0] / y[0]
    product, error = two_product(first, y[0])
    correction = (((x[0] - product) - error) + x[1] - first * y[1]) / y[0]
    head = first + correction
    return head, correction - (head - first)


def dd_exp(x: DoubleDouble) -> float:
    """Return ``exp(hi + lo)`` rounded to a float.

    The exponential of the leading part is the library ``exp`` of an exact
    argument; the trailing part, at most half an ulp of ``hi``, enters as
    the factor ``1 + lo`` (its square is below ``2^-90`` for ``|hi| < 750``).

    Args:
        x: Double-double exponent.

    Returns:
        ``exp(x)`` with the error of one ``math.exp`` call plus one rounding.

    Equation:
        [EQ-NUM-DD-001].

    References:
        [REF-DEKKER-1971], [REF-PYTHON-MATH].

    """
    return math.exp(x[0]) * (1.0 + x[1])


def _table_log(index: int) -> DoubleDouble:
    """Return ``log(index/64)`` as a double-double, computed once in decimal.

    Args:
        index: Integer between 45 and 91 (``index/64`` in ``[0.70, 1.43]``).

    Returns:
        The correctly rounded leading float and the rounded remainder.

    Equation:
        [EQ-NUM-DD-002].

    References:
        [REF-PYTHON-DECIMAL], [REF-DEKKER-1971].

    """
    cached = _log_table.get(index)
    if cached is None:
        context = Context(prec=40)
        value = context.ln(Decimal(index)) - 6 * context.ln(Decimal(2))
        high = float(value)
        cached = (high, float(value - Decimal(high)))
        _log_table[index] = cached
    return cached


def dd_log(x: DoubleDouble) -> DoubleDouble:
    """Return ``log x`` for a positive finite double-double ``x``.

    With ``x = m 2^k``, ``m`` in ``[sqrt(1/2), sqrt 2)`` and ``c = j/64``
    the nearest table point, ``log x = k ln 2 + log c + 2 atanh(y)`` with
    ``y = (m - c)/(m + c)`` (``|y| < 0.0056``; ``m - c`` is exact) and
    ``2 atanh(y) = 2y + 2y (y^2/3 + y^4/5 + y^6/7 + y^8/9)``, the bracket in
    floats. ``k ln 2`` uses the Cody-Waite split, exact in its leading part.

    Args:
        x: Double-double with ``x[0] > 0`` finite and normal.

    Returns:
        ``log x`` with absolute error below ``1e-22`` (measured at most
        ``3.2e-23`` over ``|log x| <= 700``).

    Equation:
        [EQ-NUM-DD-002].

    References:
        [REF-NIST-DLMF-ELEMENTARY, Section 4.24], [REF-DEKKER-1971].

    """
    mantissa, exponent = math.frexp(x[0])
    if mantissa < _SQRT_HALF:
        mantissa *= 2.0
        exponent -= 1
    low = math.ldexp(x[1], -exponent)
    index = int(mantissa * _TABLE_SCALE + 0.5)
    center = index / _TABLE_SCALE
    difference = two_sum(mantissa - center, low)
    denominator = two_sum(2.0 * center, difference[0])
    y = dd_div(difference, (denominator[0], denominator[1] + difference[1]))
    square = y[0] * y[0]
    bracket = square * (
        1.0 / 3.0 + square * (1.0 / 5.0 + square * (1.0 / 7.0 + square / 9.0))
    )
    table = _table_log(index)
    scale = float(exponent)
    first, first_error = two_sum(scale * _LN2_HIGH, table[0])
    second, second_error = two_sum(first, 2.0 * y[0])
    trailing = (
        first_error
        + second_error
        + scale * _LN2_LOW
        + table[1]
        + 2.0 * y[1]
        + 2.0 * y[0] * bracket
    )
    return _fast_two_sum(second, trailing)


def dd_log1pmx(t: DoubleDouble) -> DoubleDouble:
    """Return ``log(1 + t) - t`` for a double-double ``|t| <= 1/2``.

    For ``|t| <= 1/16``: ``log(1 + t) - t = -t y + (2/3) y^3 + 2y (y^4/5 +
    ... + y^20/21)`` with ``y = t/(2 + t)`` (``|y| < 0.033``; the omitted
    terms are below ``1e-29`` relative); the first two terms are formed in
    double-double arithmetic and the rest, below ``t^3/40`` relative to the
    total, in floats. Otherwise :func:`dd_log` of ``1 + t`` minus ``t``,
    which cancels by at most a factor ``40``.

    Args:
        t: Double-double with ``|t[0]| <= 0.5``.

    Returns:
        ``log1p(t) - t`` with relative error below ``1e-20``.

    Equation:
        [EQ-NUM-DD-002], [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1992] (function ``rlog1``), [REF-DEKKER-1971].

    """
    if abs(t[0]) > _LOG1PMX_SERIES_LIMIT:
        one_plus = two_sum(1.0, t[0])
        logarithm = dd_log(_fast_two_sum(one_plus[0], one_plus[1] + t[1]))
        return dd_add(logarithm, (-t[0], -t[1]))
    y = dd_div(t, dd_add((2.0, 0.0), t))
    linear = dd_mul(t, y)
    square = dd_mul(y, y)
    cubic = dd_mul(dd_mul(square, y), _TWO_THIRDS)
    s = square[0]
    bracket = 0.0
    for denominator in _ODD_DENOMINATORS:
        bracket = bracket * s + 1.0 / denominator
    rest = 2.0 * y[0] * s * s * bracket
    return dd_add((-linear[0], -linear[1]), (cubic[0], cubic[1] + rest))


def stirling_error_float(a: float) -> float:
    """Return ``lgamma(a) - (a - 1/2) log a + a - log sqrt(2 pi)`` for ``a >= 15``.

    Args:
        a: Float with ``a >= 15``.

    Returns:
        ``sum_(k=1..7) B_(2k)/(2k (2k-1) a^(2k-1))``; the truncation error
        is below ``4e-18`` and the value below ``1/180``.

    Equation:
        [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    inverse_square = 1.0 / (a * a)
    total = 0.0
    for coefficient in reversed(_STIRLING_COEFFICIENTS):
        total = total * inverse_square + coefficient
    return total / a


def dd_stirling_error(a: DoubleDouble) -> DoubleDouble:
    """Return ``sigma(a) = lgamma(a) - (a - 1/2) log a + a - log sqrt(2 pi)``.

    For ``a >= 15`` the asymptotic series is summed in floats at the leading
    part (its value is below ``1/180``, so its rounding, and the omitted
    ``sigma'(a) lo``, are far below ``2^-53`` in absolute terms). Smaller
    ``a`` are lifted to ``z = a + n >= 15`` by
    ``Gamma(a) = Gamma(z) / (a (a+1) ... (a+n-1))``:
    ``sigma(a) = (z - 1/2) log z - (a - 1/2) log a - n + sigma(z) -
    log prod_(k<n) (a + k)``, every term in double-double arithmetic.

    Args:
        a: Positive finite double-double.

    Returns:
        ``sigma(a)`` with absolute error below ``2e-18``.

    Equation:
        [EQ-NUM-DD-002], [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 5.5 and 5.11], [REF-DEKKER-1971].

    """
    if a[0] >= STIRLING_SERIES_START:
        return stirling_error_float(a[0]), 0.0
    count = math.ceil(STIRLING_SERIES_START - a[0])
    product = a
    for offset in range(1, count):
        product = dd_mul(product, dd_add(a, (float(offset), 0.0)))
    lifted = dd_add(a, (float(count), 0.0))
    lifted_term = dd_mul(dd_add(lifted, (-0.5, 0.0)), dd_log(lifted))
    own_term = dd_mul(dd_add(a, (-0.5, 0.0)), dd_log(a))
    total = dd_add(lifted_term, (-own_term[0], -own_term[1]))
    total = dd_add(total, (-float(count), 0.0))
    total = dd_add(total, (stirling_error_float(lifted[0]), 0.0))
    logarithm = dd_log(product)
    return dd_add(total, (-logarithm[0], -logarithm[1]))


@lru_cache(maxsize=4096)
def half_log_minus_stirling(a: DoubleDouble) -> DoubleDouble:
    """Return ``log(a)/2 - sigma(a)``, the shape term of the Stirling prefactors.

    The value depends on the shape only, so it is cached: a distribution's
    CDF evaluated at many points, or a quantile search, pays for the lifted
    ``sigma`` of a small shape once.

    Args:
        a: Positive finite double-double shape.

    Returns:
        ``log(a)/2 - sigma(a)`` as a double-double.

    Equation:
        [EQ-NUM-DD-002], [EQ-NUM-STIRLING-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11], [REF-DEKKER-1971].

    """
    logarithm = dd_log(a)
    correction = dd_stirling_error(a)
    return dd_add(
        (0.5 * logarithm[0], 0.5 * logarithm[1]), (-correction[0], -correction[1])
    )


def dd_scaled_log_ratio(a: float, numerator: DoubleDouble) -> DoubleDouble:
    """Return ``a [log(1 + t) - t]`` with ``1 + t = numerator / a``.

    For ``|t| <= 1/2`` the series of :func:`dd_log1pmx` is used; otherwise
    ``log(numerator / a) - t`` is formed from :func:`dd_log`.

    Args:
        a: Finite positive float.
        numerator: Positive double-double ``a (1 + t)``.

    Returns:
        The non-positive value ``a log1pmx(t)``.

    Equation:
        [EQ-NUM-DD-002], [EQ-NUM-STIRLING-001].

    References:
        [REF-DIDONATO-MORRIS-1992], [REF-DEKKER-1971].

    """
    shape = (a, 0.0)
    offset = dd_add(numerator, (-a, 0.0))
    t = dd_div(offset, shape)
    if abs(t[0]) <= 0.5:
        return dd_mul(shape, dd_log1pmx(t))
    logarithm = dd_log(dd_div(numerator, shape))
    return dd_mul(shape, dd_add(logarithm, (-t[0], -t[1])))


def erfc_double_double(w: DoubleDouble) -> float:
    """Return ``erfc(hi + lo)`` from ``math.erfc(hi)`` and a first-order correction.

    ``erfc(hi + lo) = erfc(hi) - (2/sqrt(pi)) exp(-hi^2) lo + O(hi lo^2
    exp(-hi^2))`` (DLMF 7.10.1). The correction is about ``2 hi^2`` times
    the relative size of ``lo``, which is exactly the error that rounding
    the argument to ``hi`` would cause; the second-order term is below
    ``hi^4 2^-106`` relative. The result therefore carries the error of
    ``math.erfc`` at an exact float argument only.

    Args:
        w: Double-double argument.

    Returns:
        ``erfc(w)`` as a float.

    Equation:
        [EQ-NUM-DD-003].

    References:
        [REF-NIST-DLMF, Sections 7.2 and 7.10], [REF-DEKKER-1971].

    """
    high, low = w
    value = math.erfc(high)
    if low == 0.0 or not math.isfinite(high) or abs(high) > _ERFC_UNDERFLOW:
        return value
    return value - _TWO_OVER_SQRT_PI * math.exp(-high * high) * low
