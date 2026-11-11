"""Multiple-precision reference evaluations of special functions.

Every function evaluates in :mod:`decimal` arithmetic at a working precision
of the requested significant digits plus guard digits, stops each series or
continued fraction with an explicit truncation test, and returns a
:class:`decimal.Decimal` rounded to the requested number of significant
digits. The requested precision defaults to the active decimal context
(``decimal.getcontext().prec``), so a caller sets it once with
``decimal.localcontext``. Exact rational results use :mod:`fractions`.

The algorithms are deliberately simple and independent of the float kernels
in :mod:`probstat.distributions` (series, continued fractions, Stirling's
series with its rigorous remainder bound for real arguments), so they can
serve as references for those kernels. :func:`compare_to_reference`
measures a float implementation against a reference in units in the last
place (ULPs).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    InvalidOperation,
    Overflow,
    getcontext,
    localcontext,
)
from fractions import Fraction
from statistics import NormalDist

#: Guard digits carried beyond the requested precision.
GUARD_DIGITS = 10
#: Largest accepted number of significant digits.
MAX_DIGITS = 5000
_MAX_TERMS = 500_000
_MAX_REFINEMENTS = 8
_BERNOULLI: list[Fraction] = [Fraction(1)]

RealInput = Decimal | Fraction | int | float | str


def _working(precision: int) -> Context:
    """Return a fresh round-half-even context with ``precision`` digits.

    Args:
        precision: Significant decimal digits of the context.

    Returns:
        Decimal context with the default exponent range and traps.

    References:
        [REF-PYTHON-DECIMAL]. No mathematical equation is evaluated.

    """
    return Context(prec=precision, rounding=ROUND_HALF_EVEN)


def _requested_digits(digits: int | None) -> int:
    """Return the validated number of requested significant digits.

    Args:
        digits: Requested digits, or ``None`` for the active context precision.

    Returns:
        Integer between 1 and :data:`MAX_DIGITS`.

    Raises:
        TypeError: If ``digits`` is not an integer.
        ValueError: If ``digits`` lies outside ``[1, MAX_DIGITS]``.

    References:
        [REF-PYTHON-DECIMAL]. No mathematical equation is evaluated.

    """
    value = getcontext().prec if digits is None else digits
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("digits must be an integer")
    if not 1 <= value <= MAX_DIGITS:
        raise ValueError(f"digits must lie in [1, {MAX_DIGITS}]")
    return value


def _to_decimal(value: object, *, name: str) -> Decimal:
    """Convert a real input to a finite ``Decimal`` in the active context.

    Integers, floats, decimal strings, and ``Decimal`` values convert
    exactly; a ``Fraction`` is divided out at the active precision.

    Args:
        value: Candidate real number.
        name: Argument name for error messages.

    Returns:
        Finite decimal value.

    Raises:
        TypeError: If ``value`` is not a supported real type.
        ValueError: If ``value`` is not finite or not a number.

    References:
        [REF-PYTHON-DECIMAL], [REF-PYTHON-FRACTIONS]. No mathematical equation
        is evaluated.

    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int | float):
        result = Decimal(value)
    elif isinstance(value, Fraction):
        result = Decimal(value.numerator) / Decimal(value.denominator)
    elif isinstance(value, str):
        try:
            result = Decimal(value)
        except InvalidOperation as error:
            raise ValueError(f"{name} is not a decimal number: {value!r}") from error
    else:
        raise TypeError(f"{name} must be a real number")
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _to_fraction(value: object, *, name: str) -> Fraction:
    """Convert a real input to an exact ``Fraction``.

    Args:
        value: Candidate real number (int, float, Decimal, str, Fraction).
        name: Argument name for error messages.

    Returns:
        Exact rational value.

    Raises:
        TypeError: If ``value`` is not a supported real type.
        ValueError: If ``value`` is not finite.

    References:
        [REF-PYTHON-FRACTIONS]. No mathematical equation is evaluated.

    """
    if isinstance(value, Fraction):
        return value
    return Fraction(_to_decimal(value, name=name))


def _rounded(value: Decimal, digits: int) -> Decimal:
    """Round ``value`` to ``digits`` significant digits, half to even.

    Args:
        value: Value computed at working precision.
        digits: Requested significant digits.

    Returns:
        The correctly rounded ``digits``-digit decimal of ``value``.

    References:
        [REF-PYTHON-DECIMAL]. No mathematical equation is evaluated.

    """
    with localcontext(_working(digits)):
        return +value


def _tolerance(precision: int) -> Decimal:
    """Return ``10**-precision``, the relative truncation tolerance.

    Args:
        precision: Working digits.

    Returns:
        Decimal power of ten.

    References:
        [REF-PYTHON-DECIMAL]. No mathematical equation is evaluated.

    """
    return Decimal(1).scaleb(-precision)


def _magnitude_digits(value: Decimal) -> int:
    """Return extra digits needed to resolve an absolute error of size ``value``.

    Args:
        value: Magnitude of a quantity whose absolute error matters.

    Returns:
        ``max(0, ceil(log10 |value|)) + 1``.

    References:
        [REF-PYTHON-DECIMAL]. No mathematical equation is evaluated.

    """
    if value == 0:
        return 1
    return max(0, value.copy_abs().adjusted() + 1) + 1


def _lentz(
    leading: Decimal,
    elements: Iterator[tuple[Decimal, Decimal]],
    precision: int,
) -> tuple[Decimal, int]:
    """Evaluate ``b0 + a1/(b1 + a2/(b2 + ...))`` by the modified Lentz method.

    Each step multiplies the running value by ``Delta_j = C_j D_j``; the
    iteration stops when ``|Delta_j - 1| <= 10**-precision``, i.e. when two
    successive convergents agree to the working precision. For a fraction
    with positive elements the convergents bracket the value, so this
    difference bounds the truncation error.

    Args:
        leading: Nonzero leading term ``b0``.
        elements: Iterator of partial numerators and denominators
            ``(a_j, b_j)`` for ``j = 1, 2, ...``.
        precision: Working digits (the active context must carry them).

    Returns:
        The value and the number of elements consumed.

    Raises:
        ArithmeticError: If the fraction has not converged after the iteration
            cap.

    References:
        [EQ-NUM-REFERENCE-003], [REF-LENTZ-1976], [REF-THOMPSON-BARNETT-1986].

    """
    tiny = Decimal(1).scaleb(-4 * precision - 20)
    tolerance = _tolerance(precision)
    value = leading if leading != 0 else tiny
    c_term = value
    d_term = Decimal(0)
    for index, (numerator, denominator) in enumerate(elements, start=1):
        d_term = denominator + numerator * d_term
        if d_term == 0:
            d_term = tiny
        c_term = denominator + numerator / c_term
        if c_term == 0:
            c_term = tiny
        d_term = 1 / d_term
        delta = c_term * d_term
        value *= delta
        if (delta - 1).copy_abs() <= tolerance:
            return value, index
        if index >= _MAX_TERMS:
            break
    raise ArithmeticError("continued fraction did not converge")


def _complement(compute: Callable[[int], Decimal], precision: int) -> Decimal:
    """Return ``1 - v`` to ``precision`` relative digits for ``v = compute(w)``.

    ``compute(w)`` must return ``v`` with relative error ``10**-w``. When
    ``1 - v`` is small, cancellation costs ``-log10(1 - v)`` digits, so the
    value is recomputed with that many extra digits.

    Args:
        compute: Evaluator of ``v`` at a given working precision.
        precision: Required relative digits of the complement.

    Returns:
        ``1 - v`` at the requested relative accuracy.

    Raises:
        ArithmeticError: If the complement cannot be resolved.

    References:
        [EQ-NUM-REFERENCE-003], [REF-HIGHAM-2002, Section 1.7].

    """
    working = precision
    for _ in range(_MAX_REFINEMENTS):
        value = compute(working)
        with localcontext(_working(working)):
            result = 1 - value
        if result != 0:
            lost = max(0, -result.adjusted())
            if lost <= working - precision:
                return result
            working = precision + lost + 2
        else:
            working *= 2
        if working > 4 * MAX_DIGITS:
            break
    raise ArithmeticError("complement could not be resolved")


# ---------------------------------------------------------------------------
# Elementary constants and functions
# ---------------------------------------------------------------------------


def _pi(precision: int) -> Decimal:
    """Return ``pi`` to ``precision`` digits from ``pi = 6 arcsin(1/2)``.

    The arcsine series has positive terms whose ratio is below ``1/4``, so
    the tail after a term ``t`` is at most ``t/3``; summation stops when a
    term falls below ``10**-(precision + 2)``.

    Args:
        precision: Working digits.

    Returns:
        ``pi`` at working precision.

    References:
        [EQ-NUM-REFERENCE-001], [REF-NIST-DLMF-ELEMENTARY, Equation 4.24.1],
        [REF-PYTHON-DECIMAL, Recipes].

    """
    with localcontext(_working(precision + 2)):
        tolerance = _tolerance(precision + 2)
        x_square = Decimal(1) / 4
        term = Decimal(1) / 2
        total = term
        index = 0
        while term > tolerance:
            odd = 2 * index + 1
            term = term * odd * odd * x_square / ((odd + 1) * (odd + 2))
            total += term
            index += 1
        return 6 * total


def reference_pi(*, digits: int | None = None) -> Decimal:
    """Return ``pi`` rounded to ``digits`` significant digits.

    Args:
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``pi`` as a ``Decimal``.

    Raises:
        TypeError: If ``digits`` is not an integer.
        ValueError: If ``digits`` is out of range.

    Equation:
        [EQ-NUM-REFERENCE-001].

    References:
        [REF-NIST-DLMF-ELEMENTARY, Equation 4.24.1], [REF-PYTHON-DECIMAL].

    """
    requested = _requested_digits(digits)
    return _rounded(_pi(requested + GUARD_DIGITS), requested)


def reference_exp(x: RealInput, *, digits: int | None = None) -> Decimal:
    """Return ``exp(x)`` rounded to ``digits`` significant digits.

    Uses ``Decimal.exp``, which is correctly rounded at the working precision.

    Args:
        x: Finite real argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``exp(x)`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``x`` is not finite or ``digits`` is out of range.
        OverflowError: If ``exp(x)`` exceeds the decimal exponent range.

    References:
        [REF-PYTHON-DECIMAL]. No series is truncated here.

    """
    requested = _requested_digits(digits)
    with localcontext(_working(requested + GUARD_DIGITS)):
        value = _to_decimal(x, name="x")
        try:
            result = value.exp()
        except Overflow as error:
            raise OverflowError("exp(x) exceeds the decimal range") from error
    return _rounded(result, requested)


def reference_log(x: RealInput, *, digits: int | None = None) -> Decimal:
    """Return the natural logarithm ``ln x`` rounded to ``digits`` digits.

    Uses ``Decimal.ln``, which is correctly rounded at the working precision.

    Args:
        x: Finite positive argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``ln x`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``x <= 0``, ``x`` is not finite, or ``digits`` is out of
            range.

    References:
        [REF-PYTHON-DECIMAL]. No series is truncated here.

    """
    requested = _requested_digits(digits)
    with localcontext(_working(requested + GUARD_DIGITS)):
        value = _to_decimal(x, name="x")
        if value <= 0:
            raise ValueError("x must be positive")
        result = value.ln()
    return _rounded(result, requested)


# ---------------------------------------------------------------------------
# Error function and the normal distribution
# ---------------------------------------------------------------------------


def _fraction_threshold(precision: int) -> Decimal:
    """Return the argument above which the ``erfc`` continued fraction is used.

    Args:
        precision: Working digits.

    Returns:
        ``max(3, sqrt(precision) / 2)``.

    References:
        [EQ-NUM-REFERENCE-002]. No equation is evaluated.

    """
    return Decimal(max(3.0, 0.5 * math.sqrt(precision)))


def _erf_series(x: Decimal, precision: int) -> Decimal:
    """Return ``erf x`` for ``x >= 0`` from the positive-term series.

    ``erf x = (2/sqrt pi) exp(-x^2) sum_n 2^n x^(2n+1) / (1 3 ... (2n+1))``.
    Consecutive terms have ratio ``r_n = 2x^2/(2n+3)``; summation stops once
    ``r_n <= 1/2`` and the term is below ``10**-precision`` of the sum, so the
    neglected tail (at most ``t r/(1 - r) <= t``) is below the tolerance.

    Args:
        x: Non-negative argument.
        precision: Working digits.

    Returns:
        ``erf x`` at working precision.

    Raises:
        ArithmeticError: If the series exceeds the iteration cap.

    References:
        [EQ-NUM-REFERENCE-002], [REF-NIST-DLMF, Equation 7.6.2].

    """
    with localcontext(_working(precision + 3)):
        if x == 0:
            return Decimal(0)
        tolerance = _tolerance(precision + 2)
        x_square = x * x
        term = x
        total = x
        index = 0
        while True:
            index += 1
            term = term * 2 * x_square / (2 * index + 1)
            total += term
            if term <= tolerance * total and 4 * x_square <= 2 * index + 3:
                break
            if index >= _MAX_TERMS:
                raise ArithmeticError("erf series did not converge")
        root_pi = _pi(precision + 3).sqrt()
        return 2 * (-x_square).exp() * total / root_pi


def _erfc_fraction(x: Decimal, precision: int) -> Decimal:
    """Return ``erfc x`` for ``x > 0`` from Laplace's continued fraction.

    ``sqrt(pi) exp(x^2) erfc x = x/(x^2 + (1/2)/(1 + 1/(x^2 + (3/2)/(1 + ...))))``;
    every element is positive, so successive convergents bracket the value
    and the Lentz stopping test bounds the truncation error.

    Args:
        x: Positive argument.
        precision: Working digits.

    Returns:
        ``erfc x`` at working precision.

    References:
        [EQ-NUM-REFERENCE-002], [REF-NIST-DLMF, Equation 7.9.1],
        [REF-LENTZ-1976].

    """
    with localcontext(_working(precision + 3)):
        x_square = x * x

        def elements() -> Iterator[tuple[Decimal, Decimal]]:
            """Yield ``(a_j, b_j)`` for ``j >= 2`` of the fraction.

            References:
                [REF-NIST-DLMF, Equation 7.9.1].

            """
            index = 2
            while True:
                yield Decimal(index - 1) / 2, x_square if index % 2 else Decimal(1)
                index += 1

        denominator, _ = _lentz(x_square, elements(), precision + 2)
        root_pi = _pi(precision + 3).sqrt()
        return (-x_square).exp() * x / (denominator * root_pi)


def _erf_value(x: Decimal, precision: int) -> Decimal:
    """Return ``erf x`` for any finite ``x`` at working precision.

    Args:
        x: Finite argument.
        precision: Working digits.

    Returns:
        ``erf x``.

    References:
        [EQ-NUM-REFERENCE-002], [REF-NIST-DLMF, Sections 7.6 and 7.9].

    """
    magnitude = x.copy_abs()
    if magnitude < _fraction_threshold(precision):
        value = _erf_series(magnitude, precision)
    else:
        with localcontext(_working(precision + 2)):
            value = 1 - _erfc_fraction(magnitude, precision)
    return value.copy_negate() if x < 0 else value


def _erfc_value(x: Decimal, precision: int) -> Decimal:
    """Return ``erfc x`` with ``precision`` relative digits for finite ``x``.

    Negative arguments use ``erfc x = 1 + erf |x|`` (no cancellation); large
    positive arguments use the continued fraction; small positive arguments
    use ``1 - erf x`` with guard digits for the cancellation.

    Args:
        x: Finite argument.
        precision: Working digits.

    Returns:
        ``erfc x``.

    References:
        [EQ-NUM-REFERENCE-002], [REF-NIST-DLMF, Sections 7.6 and 7.9].

    """
    if x <= 0:
        with localcontext(_working(precision + 2)):
            return 1 + _erf_value(-x, precision)
    if x >= _fraction_threshold(precision):
        return _erfc_fraction(x, precision)
    return _complement(lambda working: _erf_series(x, working), precision)


def reference_erf(x: RealInput, *, digits: int | None = None) -> Decimal:
    """Return the error function ``erf x`` rounded to ``digits`` digits.

    For ``|x|`` below ``max(3, sqrt(w)/2)`` (``w`` working digits) the
    positive-term series of ``exp(-x^2)`` times a confluent sum is used;
    above it, ``1 - erfc |x|`` with Laplace's continued fraction. Each
    expansion stops with a rigorous tail bound (geometric tail for the
    series, bracketing convergents for the fraction).

    Args:
        x: Finite real argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``erf x`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``x`` is not finite or ``digits`` is out of range.

    Equation:
        [EQ-NUM-REFERENCE-002].

    References:
        [REF-NIST-DLMF, Equations 7.6.2 and 7.9.1].

    """
    requested = _requested_digits(digits)
    working = requested + GUARD_DIGITS
    with localcontext(_working(working)):
        value = _to_decimal(x, name="x")
    return _rounded(_erf_value(value, working), requested)


def reference_erfc(x: RealInput, *, digits: int | None = None) -> Decimal:
    """Return the complementary error function ``erfc x`` to ``digits`` digits.

    The result has full relative accuracy in the upper tail, where it is
    tiny: large arguments use the continued fraction directly and moderate
    ones recompute ``1 - erf x`` with as many extra digits as the
    cancellation removes.

    Args:
        x: Finite real argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``erfc x`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``x`` is not finite or ``digits`` is out of range.

    Equation:
        [EQ-NUM-REFERENCE-002].

    References:
        [REF-NIST-DLMF, Equations 7.6.2 and 7.9.1].

    """
    requested = _requested_digits(digits)
    working = requested + GUARD_DIGITS
    with localcontext(_working(working)):
        value = _to_decimal(x, name="x")
    return _rounded(_erfc_value(value, working), requested)


def _normal_cdf_value(x: Decimal, precision: int) -> Decimal:
    """Return ``Phi(x) = erfc(-x/sqrt 2)/2`` at working precision.

    The relative condition number of ``Phi`` at ``x`` grows like ``x^2`` in
    the lower tail, so ``log10(1 + x^2)`` extra digits absorb the rounding of
    ``-x/sqrt 2``.

    Args:
        x: Finite argument.
        precision: Working digits.

    Returns:
        ``Phi(x)``.

    References:
        [EQ-NUM-REFERENCE-002], [REF-NIST-DLMF, Equation 7.2.2].

    """
    extra = _magnitude_digits(x * x + 1)
    working = precision + extra
    with localcontext(_working(working)):
        argument = -x / Decimal(2).sqrt()
        return _erfc_value(argument, working) / 2


def reference_normal_cdf(x: RealInput, *, digits: int | None = None) -> Decimal:
    """Return the standard normal CDF ``Phi(x)`` rounded to ``digits`` digits.

    ``Phi(x) = erfc(-x / sqrt 2) / 2``, evaluated with full relative accuracy
    in both tails (``Phi(-40)`` is about ``3.7e-350`` and is still resolved).

    Args:
        x: Finite real argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``Phi(x)`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``x`` is not finite or ``digits`` is out of range.

    Equation:
        [EQ-NUM-REFERENCE-002].

    References:
        [REF-NIST-DLMF, Equations 7.2.2, 7.6.2, and 7.9.1].

    """
    requested = _requested_digits(digits)
    working = requested + GUARD_DIGITS
    with localcontext(_working(working)):
        value = _to_decimal(x, name="x")
    return _rounded(_normal_cdf_value(value, working), requested)


def _normal_density(x: Decimal, precision: int) -> Decimal:
    """Return ``phi(x) = exp(-x^2/2) / sqrt(2 pi)`` at working precision.

    Args:
        x: Finite argument.
        precision: Working digits.

    Returns:
        Standard normal density.

    References:
        [EQ-NUM-REFERENCE-004], [REF-NIST-DLMF, Equation 7.2.2].

    """
    with localcontext(_working(precision + 2)):
        return (-(x * x) / 2).exp() / (2 * _pi(precision + 2)).sqrt()


def _lower_quantile(probability: Decimal, precision: int) -> Decimal:
    """Return ``Phi^-1(p)`` for ``0 < p < 1/2`` by Newton steps on ``log Phi``.

    The float quantile (or the tail asymptote ``x^2 ~ 2L - log(4 pi L)``,
    ``L = -log p``, below the float range) starts the iteration
    ``x <- x - (log Phi(x) - log p) Phi(x) / phi(x)``; ``log Phi`` is concave,
    so the steps converge quadratically. Iteration stops when a correction
    falls below ``10**-precision`` of ``|x|``: the remaining error is of the
    order of the square of that correction.

    Args:
        probability: Probability strictly between 0 and 1/2.
        precision: Working digits.

    Returns:
        The lower-tail quantile.

    Raises:
        ArithmeticError: If the iteration does not converge.

    References:
        [EQ-NUM-REFERENCE-004], [REF-NIST-DLMF, Section 7.17].

    """
    with localcontext(_working(precision + 4)):
        log_p = probability.ln()
        approximate = float(probability)
        if approximate > 1e-300:
            x = Decimal(NormalDist().inv_cdf(approximate))
        else:
            tail = -log_p
            x = -(2 * tail - (4 * _pi(precision) * tail).ln()).sqrt()
        tolerance = _tolerance(precision)
        floor = _tolerance(2 * precision)
        for _ in range(200):
            cdf = _normal_cdf_value(x, precision + 4)
            step = (cdf.ln() - log_p) * cdf / _normal_density(x, precision + 4)
            x -= step
            if step.copy_abs() <= tolerance * max(x.copy_abs(), floor):
                return x
    raise ArithmeticError("normal quantile iteration did not converge")


def reference_normal_quantile(
    probability: RealInput, *, digits: int | None = None
) -> Decimal:
    """Return the standard normal quantile ``Phi^-1(p)`` to ``digits`` digits.

    Upper-tail probabilities use ``Phi^-1(p) = -Phi^-1(1 - p)`` with ``1 - p``
    formed exactly, so both tails keep full relative accuracy. Near
    ``p = 1/2`` the quantile is small and ``-log10|p - 1/2|`` extra digits are
    carried.

    Args:
        probability: Probability strictly between 0 and 1.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        The quantile as a ``Decimal`` (exactly zero at ``p = 1/2``).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``p`` is not strictly between 0 and 1, or ``digits`` is
            out of range.

    Equation:
        [EQ-NUM-REFERENCE-004].

    References:
        [REF-NIST-DLMF, Sections 7.2 and 7.17].

    """
    requested = _requested_digits(digits)
    exact = _to_fraction(probability, name="probability")
    if not 0 < exact < 1:
        raise ValueError("probability must lie strictly between zero and one")
    half = Fraction(1, 2)
    if exact == half:
        return Decimal(0)
    offset = abs(exact - half)
    with localcontext(_working(20)):
        scale = Decimal(offset.numerator) / Decimal(offset.denominator)
    extra = max(0, -scale.adjusted())
    working = requested + GUARD_DIGITS + extra
    lower = exact if exact < half else 1 - exact
    with localcontext(_working(working)):
        tail = Decimal(lower.numerator) / Decimal(lower.denominator)
    quantile = _lower_quantile(tail, working)
    return _rounded(quantile if exact < half else quantile.copy_negate(), requested)


# ---------------------------------------------------------------------------
# Gamma and beta functions
# ---------------------------------------------------------------------------


def _bernoulli(index: int) -> Fraction:
    """Return the Bernoulli number ``B_index`` exactly.

    Numbers are generated once by the recurrence
    ``sum_(k=0..n-1) C(n, k) B_k = 0`` (``n >= 2``) and cached.

    Args:
        index: Non-negative index.

    Returns:
        ``B_index`` as a fraction (``B_1 = -1/2``).

    References:
        [EQ-NUM-REFERENCE-005], [REF-NIST-DLMF-BERNOULLI, Equation 24.5.3].

    """
    while len(_BERNOULLI) <= index:
        # With n = m + 1 the recurrence gives B_m = -sum_(k<m) C(m+1, k) B_k / (m+1).
        order = len(_BERNOULLI) + 1
        total = sum(
            (math.comb(order, k) * _BERNOULLI[k] for k in range(order - 1)),
            Fraction(0),
        )
        _BERNOULLI.append(-total / order)
    return _BERNOULLI[index]


def _stirling(z: Decimal, precision: int) -> Decimal:
    """Return ``log Gamma(z)`` for large ``z`` from Stirling's series.

    ``log Gamma(z) ~ (z - 1/2) log z - z + log(2 pi)/2 +
    sum_k B_(2k) / (2k (2k - 1) z^(2k - 1))``. For real ``z > 0`` the
    remainder after any number of terms is bounded in magnitude by the first
    neglected term, so summation stops before the first term below
    ``10**-precision`` of the leading part.

    Args:
        z: Argument, at least about ``0.37 precision + 10``.
        precision: Working digits.

    Returns:
        ``log Gamma(z)``.

    Raises:
        ArithmeticError: If the terms stop decreasing before the tolerance.

    References:
        [EQ-NUM-REFERENCE-005], [REF-NIST-DLMF-GAMMA, Equation 5.11.1 and
        Section 5.11(ii)].

    """
    with localcontext(_working(precision + 4)):
        leading = (z - Decimal("0.5")) * z.ln() - z + (2 * _pi(precision + 4)).ln() / 2
        tolerance = _tolerance(precision + 2) * max(leading.copy_abs(), Decimal(1))
        inverse_square = 1 / (z * z)
        power = 1 / z
        total = leading
        previous: Decimal | None = None
        for k in range(1, _MAX_TERMS):
            coefficient = _bernoulli(2 * k) / (2 * k * (2 * k - 1))
            term = (
                Decimal(coefficient.numerator) / Decimal(coefficient.denominator)
            ) * power
            size = term.copy_abs()
            if size <= tolerance:
                return total
            if previous is not None and size >= previous:
                break
            total += term
            previous = size
            power *= inverse_square
    raise ArithmeticError("Stirling series terms stopped decreasing")


def _log_gamma_once(x: Decimal, precision: int) -> tuple[Decimal, Decimal]:
    """Return ``log Gamma(x)`` and the size of the cancelling terms.

    The argument is shifted to ``z = x + m >= 0.37 precision + 10`` with
    ``log Gamma(x) = log Gamma(z) - log(x (x + 1) ... (x + m - 1))``.

    Args:
        x: Positive argument.
        precision: Working digits.

    Returns:
        ``(log Gamma(x), max(|log Gamma(z)|, |log product|))``.

    References:
        [EQ-NUM-REFERENCE-005], [REF-NIST-DLMF-GAMMA, Equations 5.5.1 and
        5.11.1].

    """
    shift_to = Decimal(math.ceil(0.37 * precision) + 10)
    with localcontext(_working(precision + 4)):
        steps = max(0, math.ceil(shift_to - x))
        product = Decimal(1)
        for k in range(steps):
            product *= x + k
        shifted = _stirling(x + steps, precision + 2)
        log_product = product.ln()
        return shifted - log_product, max(shifted.copy_abs(), log_product.copy_abs())


def _log_gamma_value(x: Decimal, precision: int) -> Decimal:
    """Return ``log Gamma(x)`` with ``precision`` relative digits for ``x > 0``.

    Near the zeros ``x = 1`` and ``x = 2`` the shift recurrence cancels; the
    evaluation is repeated with ``log10(size / |result|)`` extra digits.

    Args:
        x: Positive argument.
        precision: Working digits.

    Returns:
        ``log Gamma(x)`` (exactly zero at ``x = 1`` and ``x = 2``).

    Raises:
        ArithmeticError: If the cancellation cannot be resolved.

    References:
        [EQ-NUM-REFERENCE-005], [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    if x in (1, 2):
        return Decimal(0)
    working = precision
    for _ in range(_MAX_REFINEMENTS):
        value, size = _log_gamma_once(x, working)
        if value != 0:
            lost = max(0, size.adjusted() - value.adjusted() + 1)
            if lost <= working - precision:
                return value
            working = precision + lost + 2
        else:
            working *= 2
    raise ArithmeticError("log-gamma cancellation could not be resolved")


def reference_log_gamma(x: RealInput, *, digits: int | None = None) -> Decimal:
    """Return ``log Gamma(x)`` for ``x > 0`` rounded to ``digits`` digits.

    Stirling's series is summed at ``z = x + m`` with ``z`` of order the
    working precision and its truncation is bounded by the first neglected
    term (a rigorous bound for real ``z``); the shift is undone by one
    logarithm of the shift product. Relative accuracy is kept near
    the zeros at ``x = 1`` and ``x = 2``.

    Args:
        x: Finite positive argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``log Gamma(x)`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``x <= 0``, ``x`` is not finite, or ``digits`` is out of
            range.

    Equation:
        [EQ-NUM-REFERENCE-005].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11], [REF-NIST-DLMF-BERNOULLI].

    """
    requested = _requested_digits(digits)
    working = requested + GUARD_DIGITS
    with localcontext(_working(working)):
        value = _to_decimal(x, name="x")
    if value <= 0:
        raise ValueError("x must be positive")
    return _rounded(_log_gamma_value(value, working), requested)


def _scaled_exponential(exponent: Callable[[int], Decimal], precision: int) -> Decimal:
    """Return ``exp(E)`` with relative accuracy ``10**-precision``.

    ``E`` is evaluated with ``log10 |E|`` extra digits so that its absolute
    error, which becomes the relative error of ``exp(E)``, stays below the
    tolerance.

    Args:
        exponent: Evaluator of ``E`` at a given working precision.
        precision: Working digits.

    Returns:
        ``exp(E)``.

    References:
        [EQ-NUM-REFERENCE-006], [REF-HIGHAM-2002, Section 1.7].

    """
    rough = exponent(precision)
    working = precision + _magnitude_digits(rough)
    with localcontext(_working(working)):
        return exponent(working).exp()


def _gamma_series_p(a: Decimal, x: Decimal, precision: int) -> Decimal:
    """Return ``P(a, x)`` from the series ``x^a e^-x sum x^k / Gamma(a+k+1)``.

    Terms ``t_k = t_(k-1) x/(a + k)`` are positive; summation stops when the
    ratio is at most ``1/2`` and the term is below ``10**-precision`` of the
    sum, which bounds the tail by the tolerance.

    Args:
        a: Positive shape.
        x: Positive argument.
        precision: Working digits.

    Returns:
        ``P(a, x)``.

    Raises:
        ArithmeticError: If the series exceeds the iteration cap.

    References:
        [EQ-NUM-REFERENCE-006], [REF-NIST-DLMF-GAMMA, Equation 8.7.1].

    """
    with localcontext(_working(precision + 3)):
        tolerance = _tolerance(precision + 2)
        term = Decimal(1)
        total = Decimal(1)
        index = 0
        while True:
            index += 1
            term = term * x / (a + index)
            total += term
            if term <= tolerance * total and 2 * x <= a + index + 1:
                break
            if index >= _MAX_TERMS:
                raise ArithmeticError("incomplete gamma series did not converge")

    def exponent(working: int) -> Decimal:
        """Return ``a log x - x - log Gamma(a + 1)``.

        References:
            [EQ-NUM-REFERENCE-006].

        """
        with localcontext(_working(working + 2)):
            return a * x.ln() - x - _log_gamma_value(a + 1, working)

    with localcontext(_working(precision + 3)):
        return _scaled_exponential(exponent, precision + 2) * total


def _gamma_fraction_q(a: Decimal, x: Decimal, precision: int) -> Decimal:
    """Return ``Q(a, x)`` from the continued fraction of ``Gamma(a, x)``.

    ``x^-a e^x Gamma(a, x) = (1/x)/(1 + ((1-a)/x)/(1 + (1/x)/(1 + ((2-a)/x)/
    (1 + (2/x)/(1 + ...)))))``, evaluated by the modified Lentz method; for
    integer ``a`` the fraction terminates and is exact.

    Args:
        a: Positive shape.
        x: Argument with ``x >= a + 1``.
        precision: Working digits.

    Returns:
        ``Q(a, x)``.

    References:
        [EQ-NUM-REFERENCE-006], [REF-NIST-DLMF-GAMMA, Equation 8.9.2],
        [REF-LENTZ-1976].

    """
    with localcontext(_working(precision + 3)):
        inverse = 1 / x

        def elements() -> Iterator[tuple[Decimal, Decimal]]:
            """Yield ``(a_j, 1)`` for ``j >= 2``: ``(k-a)/x`` then ``k/x``.

            References:
                [REF-NIST-DLMF-GAMMA, Equation 8.9.2].

            """
            k = 1
            while True:
                yield (k - a) * inverse, Decimal(1)
                yield k * inverse, Decimal(1)
                k += 1

        denominator, _ = _lentz(Decimal(1), elements(), precision + 2)
        fraction = inverse / denominator

    def exponent(working: int) -> Decimal:
        """Return ``a log x - x - log Gamma(a)``.

        References:
            [EQ-NUM-REFERENCE-006].

        """
        with localcontext(_working(working + 2)):
            return a * x.ln() - x - _log_gamma_value(a, working)

    with localcontext(_working(precision + 3)):
        return _scaled_exponential(exponent, precision + 2) * fraction


def _gamma_arguments(
    a: RealInput, x: RealInput, working: int
) -> tuple[Decimal, Decimal]:
    """Validate and convert incomplete-gamma arguments.

    Args:
        a: Shape candidate.
        x: Argument candidate.
        working: Working digits.

    Returns:
        ``(a, x)`` as decimals.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``a <= 0``, ``x < 0``, or either is not finite.

    References:
        [EQ-NUM-REFERENCE-006]. No equation is evaluated.

    """
    with localcontext(_working(working)):
        shape = _to_decimal(a, name="a")
        point = _to_decimal(x, name="x")
    if shape <= 0:
        raise ValueError("a must be positive")
    if point < 0:
        raise ValueError("x must be non-negative")
    return shape, point


def reference_regularized_gamma_p(
    a: RealInput, x: RealInput, *, digits: int | None = None
) -> Decimal:
    """Return the regularized lower incomplete gamma ``P(a, x)``.

    For ``x < a + 1`` the positive series is summed; otherwise
    ``P = 1 - Q`` with ``Q`` from the continued fraction. Either way the
    result keeps ``digits`` relative digits (complements are recomputed with
    the digits their cancellation removes).

    Args:
        a: Finite positive shape.
        x: Finite non-negative argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``P(a, x)`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``a <= 0``, ``x < 0``, an argument is not finite, or
            ``digits`` is out of range.

    Equation:
        [EQ-NUM-REFERENCE-006].

    References:
        [REF-NIST-DLMF-GAMMA, Equations 8.7.1 and 8.9.2].

    """
    requested = _requested_digits(digits)
    working = requested + GUARD_DIGITS
    shape, point = _gamma_arguments(a, x, working)
    if point == 0:
        return Decimal(0)
    if point < shape + 1:
        value = _gamma_series_p(shape, point, working)
    else:
        value = _complement(
            lambda w: _gamma_fraction_q(shape, point, w), working
        )
    return _rounded(value, requested)


def reference_regularized_gamma_q(
    a: RealInput, x: RealInput, *, digits: int | None = None
) -> Decimal:
    """Return the regularized upper incomplete gamma ``Q(a, x) = 1 - P(a, x)``.

    Args:
        a: Finite positive shape.
        x: Finite non-negative argument.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``Q(a, x)`` as a ``Decimal``, with full relative accuracy in the upper
        tail.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``a <= 0``, ``x < 0``, an argument is not finite, or
            ``digits`` is out of range.

    Equation:
        [EQ-NUM-REFERENCE-006].

    References:
        [REF-NIST-DLMF-GAMMA, Equations 8.7.1 and 8.9.2].

    """
    requested = _requested_digits(digits)
    working = requested + GUARD_DIGITS
    shape, point = _gamma_arguments(a, x, working)
    if point == 0:
        return Decimal(1)
    if point < shape + 1:
        value = _complement(lambda w: _gamma_series_p(shape, point, w), working)
    else:
        value = _gamma_fraction_q(shape, point, working)
    return _rounded(value, requested)


def _beta_fraction(x: Decimal, a: Decimal, b: Decimal, precision: int) -> Decimal:
    """Return ``I_x(a, b)`` from its continued fraction (``x`` below the mode).

    ``I_x(a, b) = x^a (1-x)^b / (a B(a, b)) / (1 + d_1/(1 + d_2/(1 + ...)))``
    with ``d_(2m) = m(b-m)x / ((a+2m-1)(a+2m))`` and
    ``d_(2m+1) = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))``, evaluated by the
    modified Lentz method.

    Args:
        x: Argument with ``0 < x < (a + 1)/(a + b + 2)``.
        a: Positive first shape.
        b: Positive second shape.
        precision: Working digits.

    Returns:
        ``I_x(a, b)``.

    References:
        [EQ-NUM-REFERENCE-007], [REF-NIST-DLMF-BETA, Equations 8.17.22 and
        8.17.23], [REF-LENTZ-1976].

    """
    with localcontext(_working(precision + 3)):
        exact = Fraction(x)
        complement = 1 - exact
        y = Decimal(complement.numerator) / Decimal(complement.denominator)

        def elements() -> Iterator[tuple[Decimal, Decimal]]:
            """Yield ``(d_j, 1)`` for ``j >= 2``.

            References:
                [REF-NIST-DLMF-BETA, Equation 8.17.23].

            """
            m = 1
            while True:
                even = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
                odd = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
                yield even, Decimal(1)
                yield odd, Decimal(1)
                m += 1

        first = -(a + b) * x / (a + 1)
        tail, _ = _lentz(Decimal(1), elements(), precision + 2)
        denominator = 1 + first / tail

    def exponent(working: int) -> Decimal:
        """Return ``a log x + b log(1-x) - log a - log B(a, b)``.

        References:
            [EQ-NUM-REFERENCE-007].

        """
        with localcontext(_working(working + 2)):
            log_beta = (
                _log_gamma_value(a, working)
                + _log_gamma_value(b, working)
                - _log_gamma_value(a + b, working)
            )
            return a * x.ln() + b * y.ln() - a.ln() - log_beta

    with localcontext(_working(precision + 3)):
        return _scaled_exponential(exponent, precision + 2) / denominator


def reference_regularized_beta(
    x: RealInput, a: RealInput, b: RealInput, *, digits: int | None = None
) -> Decimal:
    """Return the regularized incomplete beta function ``I_x(a, b)``.

    Below ``x = (a + 1)/(a + b + 2)`` the continued fraction converges
    quickly; above it the symmetry ``I_x(a, b) = 1 - I_(1-x)(b, a)`` is used
    with the complement resolved to full relative accuracy.

    Args:
        x: Argument in ``[0, 1]``.
        a: Finite positive first shape.
        b: Finite positive second shape.
        digits: Significant digits; ``None`` uses the active decimal context.

    Returns:
        ``I_x(a, b)`` as a ``Decimal``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``x`` is outside ``[0, 1]``, a shape is not positive,
            an argument is not finite, or ``digits`` is out of range.

    Equation:
        [EQ-NUM-REFERENCE-007].

    References:
        [REF-NIST-DLMF-BETA, Equations 8.17.4, 8.17.22, and 8.17.23].

    """
    requested = _requested_digits(digits)
    working = requested + GUARD_DIGITS
    with localcontext(_working(working)):
        point = _to_decimal(x, name="x")
        first = _to_decimal(a, name="a")
        second = _to_decimal(b, name="b")
    if not 0 <= point <= 1:
        raise ValueError("x must lie in [0, 1]")
    if first <= 0 or second <= 0:
        raise ValueError("a and b must be positive")
    if point == 0:
        return Decimal(0)
    if point == 1:
        return Decimal(1)
    with localcontext(_working(working)):
        switch = (first + 1) / (first + second + 2)
    if point < switch:
        value = _beta_fraction(point, first, second, working)
    else:
        mirrored = Fraction(1) - Fraction(point)
        with localcontext(_working(working + 3)):
            other = Decimal(mirrored.numerator) / Decimal(mirrored.denominator)
        value = _complement(
            lambda w: _beta_fraction(other, second, first, w), working
        )
    return _rounded(value, requested)


def exact_regularized_beta(x: RealInput, a: int, b: int) -> Fraction:
    """Return ``I_x(a, b)`` exactly for positive integer shapes.

    ``I_x(m, n - m + 1) = sum_(j=m..n) C(n, j) x^j (1 - x)^(n - j)``, the
    binomial upper tail, evaluated in rational arithmetic.

    Args:
        x: Argument in ``[0, 1]``; floats and decimals convert exactly.
        a: Positive integer first shape.
        b: Positive integer second shape.

    Returns:
        ``I_x(a, b)`` as an exact fraction.

    Raises:
        TypeError: If a shape is not an integer or ``x`` is not real.
        ValueError: If ``x`` is outside ``[0, 1]`` or a shape is not positive.

    Equation:
        [EQ-NUM-REFERENCE-007].

    References:
        [REF-NIST-DLMF-BETA, Equation 8.17.5], [REF-PYTHON-FRACTIONS].

    """
    for name, shape in (("a", a), ("b", b)):
        if isinstance(shape, bool) or not isinstance(shape, int):
            raise TypeError(f"{name} must be an integer")
        if shape <= 0:
            raise ValueError(f"{name} must be positive")
    point = _to_fraction(x, name="x")
    if not 0 <= point <= 1:
        raise ValueError("x must lie in [0, 1]")
    total = a + b - 1
    other = 1 - point
    return sum(
        (
            math.comb(total, j) * point**j * other ** (total - j)
            for j in range(a, total + 1)
        ),
        Fraction(0),
    )


# ---------------------------------------------------------------------------
# Comparison of float implementations against a reference
# ---------------------------------------------------------------------------


def ulp_error(value: float, reference: Decimal) -> float:
    """Return ``|value - reference|`` in units in the last place of the reference.

    The unit is ``math.ulp`` of the reference rounded to the nearest float
    (the spacing of doubles at the correctly rounded result), so a correctly
    rounded implementation scores at most 0.5.

    Args:
        value: Float result of the implementation under test.
        reference: High-precision reference value.

    Returns:
        Error in ULPs; ``inf`` when ``value`` is NaN or infinite.

    Raises:
        ValueError: If the reference is not finite or overflows a float.

    Equation:
        [EQ-NUM-REFERENCE-008].

    References:
        [REF-GOLDBERG-1991, Section 1.2], [REF-PYTHON-MATH].

    """
    if not isinstance(reference, Decimal) or not reference.is_finite():
        raise ValueError("reference must be a finite Decimal")
    rounded = float(reference)
    if not math.isfinite(rounded):
        raise ValueError("reference lies outside the float range")
    if not math.isfinite(value):
        return math.inf
    with localcontext(_working(60)):
        difference = (Decimal(value) - reference).copy_abs()
        return float(difference / Decimal(math.ulp(rounded)))


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    """Errors of a float implementation against a multiple-precision reference.

    Args:
        points: Evaluation points (argument tuples).
        values: Float results of the implementation.
        references: Reference values.
        ulp_errors: Absolute errors in ULPs of the rounded reference.
        relative_errors: ``|value - reference| / |reference|`` (``inf`` for a
            nonzero error at a zero reference, ``0`` when both vanish).
        digits: Decimal digits of the references.

    References:
        [EQ-NUM-REFERENCE-008], [REF-GOLDBERG-1991, Section 1.2].

    """

    points: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    references: tuple[Decimal, ...]
    ulp_errors: tuple[float, ...]
    relative_errors: tuple[float, ...]
    digits: int

    @property
    def max_ulp(self) -> float:
        """Return the largest error in ULPs.

        References:
            [EQ-NUM-REFERENCE-008].

        """
        return max(self.ulp_errors)

    @property
    def mean_ulp(self) -> float:
        """Return the mean error in ULPs.

        References:
            [EQ-NUM-REFERENCE-008].

        """
        return math.fsum(self.ulp_errors) / len(self.ulp_errors)

    @property
    def max_relative_error(self) -> float:
        """Return the largest relative error.

        References:
            [EQ-NUM-REFERENCE-008].

        """
        return max(self.relative_errors)

    @property
    def worst_point(self) -> tuple[float, ...]:
        """Return the point with the largest error in ULPs.

        References:
            [EQ-NUM-REFERENCE-008].

        """
        return self.points[self.ulp_errors.index(self.max_ulp)]

    def within(self, budget: float) -> bool:
        """Return whether every error is at most ``budget`` ULPs.

        Args:
            budget: Non-negative ULP budget.

        Returns:
            ``True`` when ``max_ulp <= budget``.

        References:
            [EQ-NUM-REFERENCE-008].

        """
        return self.max_ulp <= budget


def compare_to_reference(
    function: Callable[..., float],
    reference: Callable[..., Decimal],
    points: Sequence[float | Sequence[float]],
    *,
    digits: int = 40,
) -> ReferenceComparison:
    """Measure a float implementation against a multiple-precision reference.

    Each point is passed positionally to both callables (a scalar point is a
    one-argument call). The reference runs inside a decimal context of
    ``digits`` significant digits, so the reference functions of this module
    pick that precision up when their ``digits`` argument is left at
    ``None``.

    Args:
        function: Float implementation under test.
        reference: Callable returning a ``Decimal`` reference value.
        points: Non-empty sequence of scalars or argument tuples.
        digits: Decimal digits of the reference evaluations (at least 20, so
            that the reference resolves a float ULP).

    Returns:
        Per-point errors and their maxima.

    Raises:
        TypeError: If ``digits`` is not an integer or a reference value is not
            a ``Decimal``.
        ValueError: If ``points`` is empty, ``digits < 20``, or a reference is
            not finite in float range.

    Equation:
        [EQ-NUM-REFERENCE-008].

    References:
        [REF-GOLDBERG-1991, Section 1.2], [REF-PYTHON-DECIMAL].

    """
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise TypeError("digits must be an integer")
    if not 20 <= digits <= MAX_DIGITS:
        raise ValueError(f"digits must lie in [20, {MAX_DIGITS}]")
    arguments = tuple(
        tuple(float(v) for v in point)
        if isinstance(point, Sequence)
        else (float(point),)
        for point in points
    )
    if not arguments:
        raise ValueError("points must not be empty")
    values: list[float] = []
    references: list[Decimal] = []
    ulps: list[float] = []
    relatives: list[float] = []
    for point in arguments:
        value = float(function(*point))
        with localcontext(_working(digits)):
            exact = reference(*point)
        if not isinstance(exact, Decimal):
            raise TypeError("reference must return a Decimal")
        values.append(value)
        references.append(exact)
        ulps.append(ulp_error(value, exact))
        if not math.isfinite(value):
            relatives.append(math.inf)
        elif exact == 0:
            relatives.append(0.0 if value == 0.0 else math.inf)
        else:
            with localcontext(_working(30)):
                gap = (Decimal(value) - exact).copy_abs()
                relatives.append(float(gap / exact.copy_abs()))
    return ReferenceComparison(
        points=arguments,
        values=tuple(values),
        references=tuple(references),
        ulp_errors=tuple(ulps),
        relative_errors=tuple(relatives),
        digits=digits,
    )
