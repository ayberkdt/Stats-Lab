"""Null distribution of the Anderson-Darling statistic for a fully specified law.

When the null CDF ``F`` is completely specified (no parameter estimated from
the tested sample), ``U_i = F(X_i)`` are independent uniforms and the law of
the Anderson-Darling statistic ``A_n^2`` [EQ-INF-GOF-AD-001] depends on ``n``
only. Marsaglia and Marsaglia (2004) evaluate its limit ``ADinf(z)`` by a
series with a two-term recursion [EQ-INF-GOF-AD-002] and correct it for
finite ``n`` by an empirical error curve fitted to ``10^10`` simulated
statistics [EQ-INF-GOF-AD-003]. For ``n = 1`` the law is exact and
elementary [EQ-INF-GOF-AD-004].

These probabilities are *not* valid after the parameters of ``F`` have been
estimated from the same data: the statistic is then stochastically smaller,
so the p-values are too large. Use ``inference.normality_tests`` for a
normal null with both parameters estimated, or
``inference.bootstrap_goodness_of_fit`` for any fitted family.

References:
    [REF-MARSAGLIA-MARSAGLIA-2004], [REF-ANDERSON-DARLING-1952],
    [REF-ANDERSON-DARLING-1954].

"""

from __future__ import annotations

import math
from numbers import Real

from probstat.foundations.validation import require_non_negative_integer

_PI_SQUARED_OVER_8 = math.pi * math.pi / 8.0
_C0_FACTOR = math.pi / math.sqrt(2.0)  # c_0 = pi e^-t (2t)^(-1/2)
_C1_FACTOR = math.pi * math.sqrt(math.pi / 2.0)  # c_1 = pi (pi/2)^(1/2) erfc(t^(1/2))
_MAX_RECURSION_TERMS = 200
_MAX_SERIES_TERMS = 100
_NEGLIGIBLE_EXPONENT = 150.0
_SMALLEST_ARGUMENT = 0.01  # ADinf(0.01) is about 5e-53
# 1 - ADinf(40) is about 6.5e-19, below half an ulp of 1; the alternating
# series loses all accuracy for much larger arguments.
_LARGEST_ARGUMENT = 40.0
_RIGHT_CROSSING = 0.8


def anderson_darling_asymptotic_cdf(statistic: float) -> float:
    """Evaluate the limiting null CDF ``ADinf(z) = lim P(A_n^2 < z)``.

    The series of Marsaglia and Marsaglia (2004, Section 2) is summed until
    its terms no longer change the double-precision value; the authors
    report about 15 correct digits, which a 30-digit quadrature of the
    Anderson-Darling integral reproduces to ``5e-16``.

    Args:
        statistic: Real argument ``z``; ``+inf`` is allowed.

    Returns:
        ``ADinf(z)`` in ``[0, 1]``; ``0`` for ``z <= 0.01`` (the true value
        is below ``1e-52`` there) and ``1`` for ``z >= 40`` (where
        ``1 - ADinf(z) < 1e-18``). ``1 - ADinf(z)`` is resolved only to
        about ``1e-15`` absolute.

    Raises:
        TypeError: If ``statistic`` is not a real number.
        ValueError: If ``statistic`` is NaN.

    Equation:
        [EQ-INF-GOF-AD-002].

    References:
        [REF-MARSAGLIA-MARSAGLIA-2004, Section 2],
        [REF-ANDERSON-DARLING-1952].

    """
    z = _require_statistic(statistic)
    if z <= _SMALLEST_ARGUMENT:
        return 0.0
    if z >= _LARGEST_ARGUMENT:
        return 1.0
    scale = 1.0 / z
    total = scale * _series_term(z, 0)
    for index in range(1, _MAX_SERIES_TERMS):
        scale *= (0.5 - index) / index  # binomial(-1/2, j) / z
        updated = total + (4 * index + 1) * scale * _series_term(z, index)
        if updated == total:
            break
        total = updated
    return min(1.0, max(0.0, total))


def anderson_darling_cdf(statistic: float, sample_size: int) -> float:
    """Evaluate the finite-sample null CDF ``P(A_n^2 < z)``.

    For ``n >= 2`` this is ``x + errfix(n, x)`` with ``x = ADinf(z)``
    (Marsaglia and Marsaglia 2004, Section 3), clipped to ``[0, 1]``. The
    authors state an absolute accuracy of ``5e-5`` at ``n = 8, 16, 32, 64,
    128`` and better than ``5e-4`` for other ``n``; seeded simulations of
    four million statistics found deviations of at most ``5e-4`` for
    ``n = 8, 10, 50``, ``6e-4`` for ``n = 5``, ``0.004`` for ``n = 3``, and
    ``0.012`` for ``n = 2``. The
    correction leaves an upper-tail floor ``P(A_n^2 >= z) >= 6e-4 / n``
    (the residual of its last polynomial at ``x = 1``), so very small
    p-values are reported conservatively. For ``n = 1`` the exact law
    ``sqrt(1 - 4 exp(-1 - z))`` is used.

    Args:
        statistic: Real argument ``z``; ``+inf`` is allowed.
        sample_size: Positive number of observations ``n``.

    Returns:
        ``P(A_n^2 < z)`` in ``[0, 1]``. The upper-tail p-value of an
        observed statistic is one minus this value (the C routine
        ``ADtest`` of the paper returns the CDF value itself).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``statistic`` is NaN or ``sample_size`` is zero.

    Equation:
        [EQ-INF-GOF-AD-003], [EQ-INF-GOF-AD-004].

    References:
        [REF-MARSAGLIA-MARSAGLIA-2004, Sections 1 and 3].

    """
    z = _require_statistic(statistic)
    count = require_non_negative_integer(sample_size, name="sample_size")
    if count == 0:
        raise ValueError("sample_size must be positive")
    if count == 1:
        return _single_observation_cdf(z)
    if math.isinf(z):
        return 0.0 if z < 0.0 else 1.0
    limit = anderson_darling_asymptotic_cdf(z)
    return min(1.0, max(0.0, limit + _error_correction(count, limit)))


def _require_statistic(statistic: object) -> float:
    """Validate a real, non-NaN statistic argument.

    Args:
        statistic: Candidate argument.

    Returns:
        The argument as a float (infinite values allowed).

    Raises:
        TypeError: If the argument is not real.
        ValueError: If the argument is NaN.

    References:
        [EQ-INF-GOF-AD-001].

    """
    if isinstance(statistic, bool) or not isinstance(statistic, Real):
        raise TypeError("statistic must be a real number")
    value = float(statistic)
    if math.isnan(value):
        raise ValueError("statistic must not be NaN")
    return value


def _series_term(z: float, index: int) -> float:
    """Evaluate ``f(z, j)`` by its power series in ``z / 8``.

    ``f(z, j) = sum_k c_k (z/8)^k / k!`` with ``t = (4j+1)^2 pi^2 / (8z)``,
    ``c_0 = pi e^-t (2t)^(-1/2)``, ``c_1 = pi (pi/2)^(1/2) erfc(t^(1/2))``,
    and ``c_(k+1) = ((k - 1/2 - t) c_k + t c_(k-1)) / k``.

    Args:
        z: Positive finite argument.
        index: Series index ``j >= 0``.

    Returns:
        ``f(z, j)``; ``0`` once ``t > 150`` (the factor ``e^-t`` underflows
        the double-precision sum).

    References:
        [EQ-INF-GOF-AD-002], [REF-MARSAGLIA-MARSAGLIA-2004, Section 2].

    """
    t = (4 * index + 1) ** 2 * _PI_SQUARED_OVER_8 / z
    if t > _NEGLIGIBLE_EXPONENT:
        return 0.0
    previous = _C0_FACTOR * math.exp(-t) / math.sqrt(t)
    current = _C1_FACTOR * math.erfc(math.sqrt(t))
    power = z / 8.0  # (z/8)^k / k!
    total = previous + current * power
    for order in range(1, _MAX_RECURSION_TERMS):
        following = ((order - 0.5 - t) * current + t * previous) / order
        previous, current = current, following
        power *= z / (8.0 * (order + 1))
        updated = total + following * power
        if updated == total:
            break
        total = updated
    return total


def _error_correction(count: int, limit: float) -> float:
    """Evaluate the finite-``n`` correction ``errfix(n, x)``.

    Args:
        count: Sample size ``n >= 2``.
        limit: Asymptotic CDF value ``x = ADinf(z)`` in ``[0, 1]``.

    Returns:
        The additive correction to ``x``.

    References:
        [EQ-INF-GOF-AD-003], [REF-MARSAGLIA-MARSAGLIA-2004, Section 3].

    """
    n = float(count)
    if limit > _RIGHT_CROSSING:
        x = limit
        tail = -130.2137 + (
            745.2337
            - (1705.091 - (1950.646 - (1116.360 - 255.7844 * x) * x) * x) * x
        ) * x
        return tail / n
    crossing = 0.01265 + 0.1757 / n
    if limit < crossing:
        u = limit / crossing
        dip = math.sqrt(u) * (1.0 - u) * (49.0 * u - 102.0)
        return (0.0037 / n**3 + 0.00078 / n**2 + 0.00006 / n) * dip
    u = (limit - crossing) / (_RIGHT_CROSSING - crossing)
    middle = -0.00022633 + (
        6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * u) * u) * u) * u
    ) * u
    return (0.04213 / n + 0.01365 / n**2) * middle


def _single_observation_cdf(z: float) -> float:
    """Return the exact ``P(A_1^2 < z) = sqrt(1 - 4 exp(-1 - z))``.

    With one observation ``A_1^2 = -1 - log(U (1 - U))`` and ``U`` uniform,
    so ``A_1^2 < z`` exactly when ``U`` lies between the roots of
    ``u (1 - u) = exp(-1 - z)``, an interval of length
    ``sqrt(1 - 4 exp(-1 - z))`` for ``z > log 4 - 1``.

    Args:
        z: Real argument (infinite allowed).

    Returns:
        The exact CDF value.

    References:
        [EQ-INF-GOF-AD-004], [REF-MARSAGLIA-MARSAGLIA-2004, Section 1].

    """
    if z <= math.log(4.0) - 1.0:
        return 0.0
    if math.isinf(z):
        return 1.0
    return math.sqrt(-math.expm1(math.log(4.0) - 1.0 - z))
