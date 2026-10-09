"""Internal special functions needed by probability-distribution algorithms."""

from __future__ import annotations

import math

from .constraints import NON_NEGATIVE_FINITE, POSITIVE_FINITE, PROBABILITY

_EPSILON = 8.0 * math.ulp(1.0)
_FPMIN = 1e-300
_MAX_ITERATIONS = 10_000


def regularized_gamma_p(a: float, x: float) -> float:
    """Return the lower regularized incomplete gamma function ``P(a, x)``.

    A convergent series is used for ``x < a + 1``; otherwise the complement of
    a modified-Lentz continued fraction for ``Q(a, x)`` is used.

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
        [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 8.2 and 8.9].

    """
    shape = POSITIVE_FINITE.validate(a, name="a")
    point = NON_NEGATIVE_FINITE.validate(x, name="x")
    if point == 0.0:
        return 0.0
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
        appropriate expansion and clamped to ``[0, 1]``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument violates its domain.
        ArithmeticError: If the selected expansion does not converge.

    Equation:
        [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 8.2 and 8.9].

    """
    shape = POSITIVE_FINITE.validate(a, name="a")
    point = NON_NEGATIVE_FINITE.validate(x, name="x")
    if point == 0.0:
        return 1.0
    if point < shape + 1.0:
        return _clamp_probability(1.0 - _gamma_series_p(shape, point))
    return _gamma_continued_fraction_q(shape, point)


def _gamma_series_p(a: float, x: float) -> float:
    """Evaluate ``P(a, x)`` through its convergent power series.

    Args:
        a: Validated positive shape.
        x: Validated positive point satisfying the series branch criterion.

    Returns:
        Lower regularized incomplete gamma value.

    Raises:
        ArithmeticError: If convergence is not reached within the iteration cap.

    Equation:
        [EQ-NUM-GAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Section 8.9].

    """
    term = 1.0 / a
    total = term
    denominator = a
    for _ in range(_MAX_ITERATIONS):
        denominator += 1.0
        term *= x / denominator
        total += term
        if abs(term) <= abs(total) * _EPSILON:
            log_scale = -x + a * math.log(x) - math.lgamma(a)
            return _clamp_probability(total * math.exp(log_scale))
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
    for index in range(1, _MAX_ITERATIONS + 1):
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
            log_scale = -x + a * math.log(x) - math.lgamma(a)
            return _clamp_probability(math.exp(log_scale) * fraction)
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
    first_shape = POSITIVE_FINITE.validate(a, name="a")
    second_shape = POSITIVE_FINITE.validate(b, name="b")
    if point == 0.0:
        return 0.0
    if point == 1.0:
        return 1.0
    log_front = (
        math.lgamma(first_shape + second_shape)
        - math.lgamma(first_shape)
        - math.lgamma(second_shape)
        + first_shape * math.log(point)
        + second_shape * math.log1p(-point)
    )
    split = (first_shape + 1.0) / (first_shape + second_shape + 2.0)
    if point < split:
        result = (
            math.exp(log_front)
            * _beta_continued_fraction(first_shape, second_shape, point)
            / first_shape
        )
        return _clamp_probability(result)
    complement = (
        math.exp(log_front)
        * _beta_continued_fraction(
            second_shape, first_shape, 1.0 - point
        )
        / second_shape
    )
    return _clamp_probability(1.0 - complement)


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
            index
            * (b - index)
            * x
            / ((a_minus_one + doubled) * (a + doubled))
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

