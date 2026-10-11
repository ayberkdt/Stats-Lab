"""Tail probabilities of the sampling distributions used by classical tests.

The Student-t, chi-square, and F laws arise as exact sampling distributions of
test statistics computed from independent normal observations. Their upper
tails are evaluated directly through the regularized incomplete gamma and beta
functions so that small p-values are not lost to ``1 - CDF`` cancellation.
The noncentral t distribution, needed for the power of t tests, is evaluated by
quadrature of its exact normal/chi mixture representation.
"""

from __future__ import annotations

import math

from probstat.distributions._special import regularized_beta, regularized_gamma_q
from probstat.foundations.validation import require_finite_real

_SQRT_HALF = math.sqrt(0.5)
_NONCENTRAL_INTERVALS = 2000


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
    ``P(T' <= t) = E[Phi(t S - delta)]``. The expectation is integrated by
    composite Simpson quadrature with 2000 subintervals over a range that
    holds all but a negligible (below ``1e-30``) part of the law of ``S``.
    With zero noncentrality the exact central CDF is returned instead.

    Args:
        value: Finite evaluation point ``t``.
        degrees_of_freedom: Degrees of freedom ``nu >= 1``.
        noncentrality: Finite noncentrality parameter ``delta``.

    Returns:
        ``P(T' <= t)`` in ``[0, 1]``; quadrature error is typically below
        ``1e-9`` in absolute terms.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``degrees_of_freedom`` is below one.

    Equation:
        [EQ-INF-DIST-NCT-001], [EQ-NUM-SIMP-001].

    References:
        [REF-JOHNSON-KOTZ-BALAKRISHNAN-1995, noncentral t chapter],
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

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
    log_constant = math.log(2.0) + 0.5 * nu * math.log(0.5 * nu) - math.lgamma(0.5 * nu)

    def integrand(s: float) -> float:
        """Return ``Phi(t s - delta) g_nu(s)`` for the chi-scale density ``g``.

        References:
            [EQ-INF-DIST-NCT-001]. No new equation is introduced.

        """
        if s <= 0.0:
            density = math.exp(log_constant) if nu == 1.0 else 0.0
        else:
            density = math.exp(
                log_constant + (nu - 1.0) * math.log(s) - 0.5 * nu * s * s
            )
        return 0.5 * math.erfc(-(t * s - delta) * _SQRT_HALF) * density

    width = (upper - lower) / _NONCENTRAL_INTERVALS
    terms = [integrand(lower), integrand(upper)]
    for index in range(1, _NONCENTRAL_INTERVALS):
        coefficient = 4.0 if index % 2 else 2.0
        terms.append(coefficient * integrand(lower + index * width))
    return max(0.0, min(1.0, width * math.fsum(terms) / 3.0))
