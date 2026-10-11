"""Digamma and trigamma functions for positive real arguments."""

from __future__ import annotations

import math

from probstat.foundations.validation import require_finite_real

_ASYMPTOTIC_START = 10.0
# B_(2k) / (2k) for k = 1..6: coefficients of x^(-2k) in log x - psi(x) - 1/(2x).
_DIGAMMA_COEFFICIENTS = (
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
)
# B_(2k) for k = 1..6: coefficients of x^(-2k-1) in psi'(x) - 1/x - 1/(2x^2).
_TRIGAMMA_COEFFICIENTS = (
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
)


def _validated_positive(value: object, *, name: str) -> float:
    """Return a finite positive float or raise.

    Args:
        value: Candidate argument.
        name: Argument name for error messages.

    Returns:
        Validated positive float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is non-finite or not positive.

    References:
        [REF-PYTHON-MATH]. No mathematical equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _even_power_series(coefficients: tuple[float, ...], inverse_square: float) -> float:
    """Return ``sum_k c_k u^k`` for ``u = x^-2`` by Horner's rule (``k >= 1``).

    Args:
        coefficients: Series coefficients ``c_1, c_2, ...``.
        inverse_square: ``u = 1/x^2``.

    Returns:
        Truncated series value.

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.11].

    """
    total = 0.0
    for coefficient in reversed(coefficients):
        total = (total + coefficient) * inverse_square
    return total


def log_minus_digamma(x: float) -> float:
    """Return ``log x - psi(x)`` without cancellation for large ``x``.

    For ``x >= 10`` the asymptotic series
    ``1/(2x) + sum_(k=1..6) B_(2k)/(2k x^(2k))`` is summed directly; smaller
    arguments first apply ``psi(x) = psi(x + 1) - 1/x``.

    Args:
        x: Finite positive argument.

    Returns:
        Positive value ``log x - psi(x)``.

    Raises:
        TypeError: If ``x`` is not real.
        ValueError: If ``x`` is non-finite or not positive.

    Equation:
        [EQ-NUM-DIGAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 5.5 and 5.11].

    """
    value = _validated_positive(x, name="x")
    if value < _ASYMPTOTIC_START:
        return math.log(value) - digamma(value)
    return 0.5 / value + _even_power_series(
        _DIGAMMA_COEFFICIENTS, 1.0 / (value * value)
    )


def digamma(x: float) -> float:
    """Return ``psi(x) = d/dx log Gamma(x)`` for ``x > 0``.

    The recurrence ``psi(x) = psi(x + 1) - 1/x`` lifts the argument to at
    least 10, where the asymptotic series
    ``log x - 1/(2x) - sum B_(2k)/(2k x^(2k))`` is truncated after ``k = 6``.

    Args:
        x: Finite positive argument.

    Returns:
        Digamma value.

    Raises:
        TypeError: If ``x`` is not real.
        ValueError: If ``x`` is non-finite or not positive.

    Equation:
        [EQ-NUM-DIGAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 5.5 and 5.11].

    """
    value = _validated_positive(x, name="x")
    shift = 0.0
    while value < _ASYMPTOTIC_START:
        shift += 1.0 / value
        value += 1.0
    return math.log(value) - log_minus_digamma(value) - shift


def trigamma(x: float) -> float:
    """Return ``psi'(x)`` for ``x > 0``.

    The recurrence ``psi'(x) = psi'(x + 1) + 1/x^2`` lifts the argument to at
    least 10, where ``1/x + 1/(2x^2) + sum B_(2k)/x^(2k+1)`` is truncated
    after ``k = 6``.

    Args:
        x: Finite positive argument.

    Returns:
        Trigamma value.

    Raises:
        TypeError: If ``x`` is not real.
        ValueError: If ``x`` is non-finite or not positive.

    Equation:
        [EQ-NUM-DIGAMMA-001].

    References:
        [REF-NIST-DLMF-GAMMA, Sections 5.11 and 5.15].

    """
    value = _validated_positive(x, name="x")
    shift = 0.0
    while value < _ASYMPTOTIC_START:
        shift += 1.0 / (value * value)
        value += 1.0
    inverse = 1.0 / value
    inverse_square = inverse * inverse
    series = inverse * _even_power_series(_TRIGAMMA_COEFFICIENTS, inverse_square)
    return inverse + 0.5 * inverse_square + series + shift
