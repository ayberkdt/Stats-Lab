"""Null distribution of the Cramer-von Mises statistic for a fully specified law.

When the null CDF ``F`` is completely specified (no parameter estimated from
the tested sample), ``U_i = F(X_i)`` are independent uniforms and the law of
``W_n^2 = 1/(12 n) + sum_i (U_(i) - (2i - 1)/(2n))^2`` [EQ-INF-GOF-CVM-001]
depends on ``n`` only. This module evaluates

- the limit law ``V(x) = lim P(W_n^2 <= x)`` (Anderson and Darling, 1952), the
  law of ``sum_k Z_k^2 / (k pi)^2`` with independent standard normal
  ``Z_k``, by a series in the modified Bessel function
  ``K_(1/4)`` for the lower part [EQ-INF-GOF-CVM-002] and by a line integral
  for the upper tail [EQ-INF-GOF-CVM-003];
- the finite-``n`` approximation ``V(x) + psi_1(x)/n`` of Csorgo and Faraway
  (1996) with an error of order ``n^-2`` [EQ-INF-GOF-CVM-004];
- the exact law for ``n = 1`` [EQ-INF-GOF-CVM-005].

The Bessel function is evaluated here by the trapezoidal rule on
``K_nu(z) = int_0^inf exp(-z cosh t) cosh(nu t) dt``, whose error decays
geometrically in the number of nodes; the standard library has no Bessel
function.

These probabilities are *not* valid after the parameters of ``F`` have been
estimated from the same data: the statistic is then stochastically smaller,
so the p-values are too large. Use ``inference.normality_tests`` for a
normal null with both parameters estimated, or
``inference.bootstrap_goodness_of_fit`` for any fitted family.

References:
    [REF-ANDERSON-DARLING-1952], [REF-CSORGO-FARAWAY-1996],
    [REF-STEPHENS-1974].

"""

from __future__ import annotations

import math
from numbers import Real

from probstat.foundations.validation import require_non_negative_integer

_SERIES_SWITCH = 0.6
_TAIL_TERMS = 3
_TAIL_NODES = 64
_MAX_SERIES_TERMS = 400
_TOLERANCE = 1e-18
_NEGLIGIBLE_ARGUMENT = 400.0  # exp(-2 z) underflows beyond this
_LARGEST_STATISTIC = 50.0  # P(W^2 > 50) is far below 1e-30
_SQRT_PI = math.sqrt(math.pi)
_PI_POW_1_5 = math.pi**1.5


def cramer_von_mises_asymptotic_cdf(statistic: float) -> float:
    """Evaluate the limiting null CDF ``V(x) = lim P(W_n^2 <= x)``.

    Below ``x = 0.6`` the series in ``K_(1/4)`` of Anderson and Darling is
    summed to double precision; from there on ``1 - V(x)`` is computed
    directly from its own series (see :func:`cramer_von_mises_asymptotic_sf`),
    so the value near one keeps its absolute accuracy of about ``1e-16``.

    Args:
        statistic: Real argument ``x``; ``+inf`` is allowed.

    Returns:
        ``V(x)`` in ``[0, 1]``; ``0`` for ``x <= 0``. The 95 and 99 percent
        points are ``0.461361...`` and ``0.743459...``.

    Raises:
        TypeError: If ``statistic`` is not a real number.
        ValueError: If ``statistic`` is NaN.

    Equation:
        [EQ-INF-GOF-CVM-002], [EQ-INF-GOF-CVM-003].

    References:
        [REF-ANDERSON-DARLING-1952], [REF-CSORGO-FARAWAY-1996].

    """
    x = _require_statistic(statistic)
    if x <= 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0
    if x < _SERIES_SWITCH:
        return min(1.0, max(0.0, _lower_series(x)))
    return min(1.0, max(0.0, 1.0 - _upper_tail(x)))


def cramer_von_mises_asymptotic_sf(statistic: float) -> float:
    """Evaluate the limiting null tail ``1 - V(x) = lim P(W_n^2 > x)``.

    For ``x >= 0.6`` the tail is the alternating sum of the line integrals
    ``(1/pi) int_((2j-1)^2 pi^2)^((2j)^2 pi^2) sqrt(-sqrt(y)/sin sqrt(y))
    exp(-x y/2) dy/y`` [EQ-INF-GOF-CVM-003], three of which are summed (the
    fourth is below ``exp(-2 pi^2 x)`` times the first, under ``1e-5`` of it
    for ``x >= 0.6`` and far below the double-precision resolution of the
    first for the larger arguments where it matters). The integrable
    inverse-square-root singularities at both ends are removed by
    ``u = sqrt(y) = (4j - 1) pi/2 - (pi/2) cos(phi)``, after which the
    integrand is an even, ``2 pi``-periodic analytic function of ``phi`` and
    the trapezoidal rule with 64 nodes converges geometrically. The relative
    error is below about ``3e-14`` up to ``x = 30`` (below ``1e-14`` up to
    ``x = 10``), checked against a 150-digit evaluation of the Bessel series.

    Args:
        statistic: Real argument ``x``; ``+inf`` is allowed.

    Returns:
        ``P(W^2 > x)`` in ``[0, 1]``; ``1`` for ``x <= 0``.

    Raises:
        TypeError: If ``statistic`` is not a real number.
        ValueError: If ``statistic`` is NaN.

    Equation:
        [EQ-INF-GOF-CVM-002], [EQ-INF-GOF-CVM-003].

    References:
        [REF-ANDERSON-DARLING-1952].

    """
    x = _require_statistic(statistic)
    if x <= 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    if x < _SERIES_SWITCH:
        return min(1.0, max(0.0, 1.0 - _lower_series(x)))
    return min(1.0, max(0.0, _upper_tail(x)))


def cramer_von_mises_cdf(statistic: float, sample_size: int) -> float:
    """Return ``P(W_n^2 <= x)`` for a fully specified continuous null.

    ``n = 1`` is exact: ``W_1^2 = 1/12 + (U - 1/2)^2`` gives
    ``2 sqrt(x - 1/12)`` on ``[1/12, 1/3]`` [EQ-INF-GOF-CVM-005]. For
    ``n >= 2`` the value is the approximation ``V(x) + psi_1(x)/n`` of
    Csorgo and Faraway (1996, equations 1.8 and 1.10) inside the support
    ``(1/(12 n), n/3)`` of ``W_n^2`` (``0`` below and ``1`` above)
    [EQ-INF-GOF-CVM-004]. Its error is of order ``n^-2``. Against seeded
    simulations of 3 million samples the upper-tail probabilities
    ``p <= 0.5`` were within about ``0.01`` of the truth at ``n = 2``,
    ``0.004`` at ``n = 3``, ``0.003`` at ``n = 4`` and ``0.001`` for
    ``n >= 5``; the relative error is large in the far tail at ``n = 2`` (a
    factor of about 9 at ``p = 10^-3``). The correction is not reliable
    close to the ends of the support.

    Args:
        statistic: Real argument ``x``; ``+inf`` is allowed.
        sample_size: Positive sample size ``n``.

    Returns:
        The probability in ``[0, 1]``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``statistic`` is NaN or ``sample_size`` is not positive.

    Equation:
        [EQ-INF-GOF-CVM-004], [EQ-INF-GOF-CVM-005].

    References:
        [REF-CSORGO-FARAWAY-1996].

    """
    x = _require_statistic(statistic)
    count = _require_count(sample_size)
    return 1.0 - _finite_sf(x, count)


def cramer_von_mises_sf(statistic: float, sample_size: int) -> float:
    """Return ``P(W_n^2 > x)``, the p-value of a statistic, for ``n`` observations.

    The tail is computed as ``(1 - V(x)) - psi_1(x)/n`` so that the large
    arguments keep the absolute accuracy of ``1 - V``; the correction
    ``psi_1`` itself is resolved to about ``1e-15``, so tails below about
    ``1e-12`` are not resolved. See :func:`cramer_von_mises_cdf` for the
    accuracy against simulation.

    Args:
        statistic: Real argument ``x``; ``+inf`` is allowed.
        sample_size: Positive sample size ``n``.

    Returns:
        The tail probability in ``[0, 1]``: ``1`` at or below ``1/(12 n)``,
        ``0`` at or above ``n/3`` (for ``n >= 2``).

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``statistic`` is NaN or ``sample_size`` is not positive.

    Equation:
        [EQ-INF-GOF-CVM-004], [EQ-INF-GOF-CVM-005].

    References:
        [REF-CSORGO-FARAWAY-1996].

    """
    x = _require_statistic(statistic)
    count = _require_count(sample_size)
    return _finite_sf(x, count)


def _finite_sf(x: float, count: int) -> float:
    """Return the finite-``n`` null tail for a validated statistic and size.

    Args:
        x: Real non-NaN statistic (infinite allowed).
        count: Positive sample size.

    Returns:
        ``P(W_n^2 > x)`` in ``[0, 1]``.

    References:
        [EQ-INF-GOF-CVM-004], [EQ-INF-GOF-CVM-005].

    """
    if count == 1:
        if x <= 1.0 / 12.0:
            return 1.0
        if x >= 1.0 / 3.0:
            return 0.0
        return 1.0 - 2.0 * math.sqrt(x - 1.0 / 12.0)
    if x <= 1.0 / (12.0 * count):
        return 1.0
    if x >= count / 3.0 or x > _LARGEST_STATISTIC:
        return 0.0
    tail = cramer_von_mises_asymptotic_sf(x)
    correction = _psi_one(x) / count
    return min(1.0, max(0.0, tail - correction))


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
        [EQ-INF-GOF-CVM-001].

    """
    if isinstance(statistic, bool) or not isinstance(statistic, Real):
        raise TypeError("statistic must be a real number")
    value = float(statistic)
    if math.isnan(value):
        raise ValueError("statistic must not be NaN")
    return value


def _require_count(sample_size: object) -> int:
    """Validate a positive integer sample size.

    Args:
        sample_size: Candidate argument.

    Returns:
        The sample size.

    Raises:
        TypeError: If the argument is not an integer.
        ValueError: If it is not positive.

    References:
        [EQ-INF-GOF-CVM-001].

    """
    try:
        count = require_non_negative_integer(sample_size, name="sample_size")
    except ValueError:
        raise ValueError("sample_size must be positive") from None
    if count == 0:
        raise ValueError("sample_size must be positive")
    return count


def _scaled_bessel_k(order: float, argument: float) -> float:
    """Evaluate ``exp(z) K_nu(z)`` for ``z > 0`` by the trapezoidal rule.

    ``K_nu(z) = int_0^inf exp(-z cosh t) cosh(nu t) dt`` has an even
    integrand that is analytic in a strip, so the trapezoidal rule with step
    ``h`` has an error of order ``exp(-c/h)``; the step ``min(1/8, 1/(4
    sqrt(1+z)))`` follows the width ``z^(-1/2)`` of the integrand. Nodes are
    added until the integrand is below ``1e-18`` of the running sum. The
    scaling by ``exp(z)`` keeps large arguments from underflowing. Agreement
    with the reference Bessel function is about ``1e-15`` relative for
    ``z`` from ``5e-4`` to ``200`` and ``nu`` from ``1/4`` to ``5/4``.

    Args:
        order: Order ``nu >= 0``.
        argument: Positive finite argument ``z``.

    Returns:
        ``exp(z) K_nu(z)``.

    References:
        [EQ-INF-GOF-CVM-002].

    """
    step = min(0.125, 0.25 / math.sqrt(1.0 + argument))
    total = 0.5
    index = 1
    while True:
        node = index * step
        base = -2.0 * argument * math.sinh(0.5 * node) ** 2
        if base + order * node < -745.0:
            break
        value = 0.5 * (math.exp(base + order * node) + math.exp(base - order * node))
        total += value
        if index > 8 and value < _TOLERANCE * total:
            break
        index += 1
    return step * total


def _decayed_bessel(order: float, argument: float) -> float:
    """Return ``exp(-z) K_nu(z)`` without overflow for large ``z``.

    Args:
        order: Order ``nu >= 0``.
        argument: Positive finite argument ``z``.

    Returns:
        ``exp(-z) K_nu(z) = exp(-2z) (exp(z) K_nu(z))``.

    References:
        [EQ-INF-GOF-CVM-002].

    """
    if argument > _NEGLIGIBLE_ARGUMENT:
        return 0.0
    return math.exp(-2.0 * argument) * _scaled_bessel_k(order, argument)


def _lower_series(x: float) -> float:
    """Return ``V(x)`` from the Anderson-Darling series in ``K_(1/4)``.

    ``V(x) = (pi^(3/2) sqrt(x))^(-1) sum_(j>=0) Gamma(j + 1/2)/j!
    sqrt(4j + 1) exp(-q_j) K_(1/4)(q_j)`` with ``q_j = (4j+1)^2/(16 x)``.

    Args:
        x: Positive finite argument below the switch point.

    Returns:
        The series value (not clamped).

    References:
        [EQ-INF-GOF-CVM-002], [REF-ANDERSON-DARLING-1952].

    """
    total = 0.0
    for index in range(_MAX_SERIES_TERMS):
        shape = 4 * index + 1
        argument = shape * shape / (16.0 * x)
        weight = math.exp(math.lgamma(index + 0.5) - math.lgamma(index + 1.0))
        term = (
            weight
            * math.sqrt(shape)
            * _decayed_bessel(0.25, argument)
            / (_PI_POW_1_5 * math.sqrt(x))
        )
        total += term
        if index > 2 and term < _TOLERANCE:
            break
    return total


def _upper_tail(x: float) -> float:
    """Return ``1 - V(x)`` from the alternating line integrals.

    Args:
        x: Positive finite argument.

    Returns:
        The tail value (not clamped).

    References:
        [EQ-INF-GOF-CVM-003].

    """
    return (
        math.fsum(
            (-1) ** (index + 1) * _line_integral(x, index)
            for index in range(1, _TAIL_TERMS + 1)
        )
        / math.pi
    )


def _line_integral(x: float, index: int) -> float:
    """Evaluate the ``index``-th line integral of the tail series.

    ``I_j = int_((2j-1) pi)^(2 j pi) 2 exp(-x u^2/2) / sqrt(-u sin u) du``
    with ``u = (4j-1) pi/2 - (pi/2) cos(phi)``. Then ``-sin u = sin(pi s^2) =
    sin(pi c^2)`` for ``s = sin(phi/2)``, ``c = cos(phi/2)`` and
    ``du = pi s c d(phi)``, so ``I_j = pi int_0^pi exp(-x u^2/2) u^(-1/2)
    r(phi) d(phi)`` with ``r = 2 s c / sqrt(sin(pi s^2))`` a smooth even
    periodic function (``r(0) = r(pi) = 2/sqrt(pi)``); the trapezoidal rule
    is applied on ``[0, pi]``.

    Args:
        x: Positive finite argument.
        index: Interval index ``j >= 1``.

    Returns:
        The integral ``I_j``.

    References:
        [EQ-INF-GOF-CVM-003].

    """
    nodes = _TAIL_NODES
    centre = (4 * index - 1) * math.pi / 2.0
    total = 0.0
    for node in range(nodes + 1):
        angle = math.pi * node / nodes
        cosine = math.cos(angle)
        sine_sq = 0.5 * (1.0 - cosine)
        cosine_sq = 0.5 * (1.0 + cosine)
        point = centre - 0.5 * math.pi * cosine
        if sine_sq <= 0.5:
            phase = math.pi * sine_sq
            near = math.sqrt(cosine_sq)
        else:
            phase = math.pi * cosine_sq
            near = math.sqrt(sine_sq)
        if phase == 0.0:
            ratio = 2.0 * near / _SQRT_PI
        else:
            ratio = 2.0 * math.sqrt(sine_sq * cosine_sq) / math.sqrt(math.sin(phase))
        value = math.exp(-0.5 * x * point * point) / math.sqrt(point) * ratio
        total += value * (0.5 if node in (0, nodes) else 1.0)
    return math.pi * total * math.pi / nodes


def _psi_one(x: float) -> float:
    """Return the first-order finite-sample correction ``psi_1(x)``.

    ``psi_1(x) = V(x)/12 - sum_(k>=0) A_k(x) / (pi k!)`` where ``A_k`` is
    a sum of five terms in ``K_(1/4)``, ``K_(3/4)``, ``K_(5/4)``, with the
    gamma factors ``Gamma(k + 1/2)/k!`` formed through log-gamma [Csorgo and
    Faraway (1996), equation 1.10, as implemented in SciPy]. The ``V(x)/12``
    term is evaluated as ``(1 - S(x))/12`` with ``S = 1 - V`` so that it keeps
    precision.

    Args:
        x: Positive finite argument inside the support.

    Returns:
        ``psi_1(x)``.

    References:
        [EQ-INF-GOF-CVM-004], [REF-CSORGO-FARAWAY-1996].

    """
    root = 2.0 * math.sqrt(x)
    power_a = x**0.75
    power_b = x**1.25
    total = 0.0
    for index in range(_MAX_SERIES_TERMS):
        shape = 2 * index + 1
        gamma_a = math.exp(math.lgamma(index + 0.5) - math.lgamma(index + 1.0))
        gamma_b = gamma_a * (index + 0.5)
        first = (
            shape * gamma_a * _weighted_two((4 * index + 3) / root) / (9.0 * power_a)
        )
        second = gamma_a * _weighted_three((4 * index + 1) / root) / (72.0 * power_b)
        third = (
            2.0
            * (shape + 2)
            * gamma_b
            * _weighted_three((4 * index + 5) / root)
            / (12.0 * power_b)
        )
        fourth = (
            7.0
            * shape
            * gamma_a
            * _weighted_two((4 * index + 1) / root)
            / (144.0 * power_a)
        )
        fifth = (
            7.0
            * shape
            * gamma_a
            * _weighted_two((4 * index + 5) / root)
            / (144.0 * power_a)
        )
        term = -(first + second + third + fourth + fifth) / math.pi
        total += term
        if index > 2 and abs(term) < _TOLERANCE:
            break
    return (1.0 - cramer_von_mises_asymptotic_sf(x)) / 12.0 + total


def _weighted_two(y: float) -> float:
    """Return ``exp(-z) (y/2)^(3/2) (K_(1/4)(z) + K_(3/4)(z)) / sqrt(pi)``.

    Args:
        y: Positive argument with ``z = y^2/4``.

    Returns:
        The value used by the second-order terms of ``psi_1``.

    References:
        [EQ-INF-GOF-CVM-004].

    """
    z = y * y / 4.0
    bessel = _decayed_bessel(0.25, z) + _decayed_bessel(0.75, z)
    return float((y / 2.0) ** 1.5 * bessel / _SQRT_PI)


def _weighted_three(y: float) -> float:
    """Return ``exp(-z) (y/2)^(5/2) (2 K_(1/4) + 3 K_(3/4) - K_(5/4)) / sqrt(pi)``.

    Args:
        y: Positive argument with ``z = y^2/4``.

    Returns:
        The value used by the third-order terms of ``psi_1``.

    References:
        [EQ-INF-GOF-CVM-004].

    """
    z = y * y / 4.0
    bessel = (
        2.0 * _decayed_bessel(0.25, z)
        + 3.0 * _decayed_bessel(0.75, z)
        - _decayed_bessel(1.25, z)
    )
    return float((y / 2.0) ** 2.5 * bessel / _SQRT_PI)
