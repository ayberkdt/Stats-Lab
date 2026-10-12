"""Exact and bounded limit-theorem results for the sample mean.

This module answers limit-theorem questions with exact distributions and
proven inequalities, not with simulation. For i.i.d. summands with mean
``mu``, variance ``sigma^2``, and third absolute central moment ``rho``, the
standardized mean is ``Z_n = sqrt(n) (Xbar_n - mu) / sigma``.

* Exact laws of the sum where closed forms exist: Bernoulli sums are
  binomial, exponential sums are gamma, normal sums are normal.
* The Kolmogorov distance ``sup_z |P(Z_n <= z) - Phi(z)|`` evaluated from the
  exact law and compared with the Berry-Esseen bound ``C rho / (sigma^3
  sqrt(n))`` using Shevtsova's constant ``C = 0.4748``.
* The one-term Edgeworth expansion, which corrects ``Phi`` for skewness.
* Chebyshev, Hoeffding, and Bernstein tail bounds for ``Xbar_n - mu`` next to
  the exact tail probability.
* Delta-method variances of ``g(Xbar_n)`` next to exact variances.

Simulation demonstrations of the laws of large numbers and the central limit
theorem live separately in :mod:`probstat.monte_carlo.limit_simulation`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from probstat.distributions._special import (
    regularized_gamma_p,
    regularized_gamma_q,
)
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

SummandKind = Literal["bernoulli", "exponential", "normal"]

BERRY_ESSEEN_CONSTANT = 0.4748
"""Shevtsova's (2011) upper bound for the i.i.d. Berry-Esseen constant."""

_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
_GRID_POINTS = 1601
_GOLDEN_STEPS = 60
_INTEGER_TOLERANCE = 1e-9


def _positive(value: object, *, name: str) -> float:
    """Validate a strictly positive finite real.

    Args:
        value: Candidate value.
        name: Parameter name used in error messages.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _sample_size(value: object) -> int:
    """Validate a positive integer sample size.

    Args:
        value: Candidate sample size.

    Returns:
        The sample size.

    Raises:
        ValueError: If the size is not a positive integer.

    References:
        Project validation policy. No equation is evaluated.

    """
    n = require_non_negative_integer(value, name="n")
    if n < 1:
        raise ValueError("n must be a positive integer")
    return n


def standard_normal_cdf(z: float) -> float:
    """Return ``Phi(z)`` through the complementary error function.

    Args:
        z: Evaluation point (infinite values are allowed).

    Returns:
        Standard normal distribution function.

    Equation:
        [EQ-LIM-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """
    return 0.5 * math.erfc(-z / _SQRT_TWO)


def _normal_density(z: float) -> float:
    """Return the standard normal density ``phi(z)``.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """
    return math.exp(-0.5 * z * z) / _SQRT_TWO_PI


@dataclass(frozen=True, slots=True)
class SummandLaw:
    """Moments and support of one i.i.d. summand with a closed-form sum law.

    Args:
        kind: ``"bernoulli"``, ``"exponential"``, or ``"normal"``.
        parameter: Success probability, rate, or ``(mean, sd)`` pair.
        mean: Mean ``mu``.
        variance: Variance ``sigma^2``.
        third_central_moment: ``E[(X - mu)^3]``.
        third_absolute_central_moment: ``rho = E|X - mu|^3``.
        lower: Lower support bound (``-inf`` when unbounded).
        upper: Upper support bound (``inf`` when unbounded).

    References:
        [REF-CASELLA-BERGER-2002, Sections 3.2-3.3], [EQ-LIM-010].

    """

    kind: SummandKind
    parameter: tuple[float, ...]
    mean: float
    variance: float
    third_central_moment: float
    third_absolute_central_moment: float
    lower: float
    upper: float

    @property
    def standard_deviation(self) -> float:
        """Return ``sigma``.

        References:
            [REF-CASELLA-BERGER-2002, Section 2.3].

        """
        return math.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """Return ``gamma = E[(X - mu)^3] / sigma^3``.

        References:
            [EQ-LIM-010], [REF-HALL-1992, Chapter 2].

        """
        return self.third_central_moment / (self.variance * math.sqrt(self.variance))

    @property
    def lattice(self) -> bool:
        """Return whether the summand is supported on a lattice.

        References:
            [REF-HALL-1992, Chapter 2].

        """
        return self.kind == "bernoulli"

    @property
    def bounded(self) -> bool:
        """Return whether the support is a bounded interval.

        References:
            [REF-HOEFFDING-1963].

        """
        return math.isfinite(self.lower) and math.isfinite(self.upper)


def bernoulli_summand(p: float) -> SummandLaw:
    """Return the Bernoulli(``p``) summand, whose sums are binomial.

    ``rho = p (1 - p) (p^2 + (1 - p)^2)`` and
    ``E[(X - p)^3] = p (1 - p) (1 - 2 p)``.

    Args:
        p: Success probability in ``(0, 1)``.

    Returns:
        Summand description.

    Raises:
        ValueError: If ``p`` is not in ``(0, 1)``.

    Equation:
        [EQ-LIM-010].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.2].

    """
    q = require_finite_real(p, name="p")
    if not 0.0 < q < 1.0:
        raise ValueError("p must lie strictly between zero and one")
    variance = q * (1.0 - q)
    return SummandLaw(
        "bernoulli",
        (q,),
        q,
        variance,
        variance * (1.0 - 2.0 * q),
        variance * (q * q + (1.0 - q) * (1.0 - q)),
        0.0,
        1.0,
    )


def exponential_summand(rate: float) -> SummandLaw:
    """Return the Exponential(``rate``) summand, whose sums are gamma.

    For rate one, ``E|X - 1|^3 = 12/e - 2``; moments of order three scale
    with ``rate^-3``.

    Args:
        rate: Positive rate ``lambda``.

    Returns:
        Summand description.

    Raises:
        ValueError: If the rate is not positive.

    Equation:
        [EQ-LIM-010].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """
    lam = _positive(rate, name="rate")
    scale3 = lam**-3
    return SummandLaw(
        "exponential",
        (lam,),
        1.0 / lam,
        1.0 / (lam * lam),
        2.0 * scale3,
        (12.0 / math.e - 2.0) * scale3,
        0.0,
        math.inf,
    )


def normal_summand(mean: float, standard_deviation: float) -> SummandLaw:
    """Return the Normal(``mean``, ``sd^2``) summand, whose sums are normal.

    ``E|X - mu|^3 = 2 sqrt(2/pi) sigma^3``.

    Args:
        mean: Finite mean.
        standard_deviation: Positive standard deviation.

    Returns:
        Summand description.

    Raises:
        ValueError: If the standard deviation is not positive.

    Equation:
        [EQ-LIM-010].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """
    mu = require_finite_real(mean, name="mean")
    sigma = _positive(standard_deviation, name="standard_deviation")
    return SummandLaw(
        "normal",
        (mu, sigma),
        mu,
        sigma * sigma,
        0.0,
        2.0 * math.sqrt(2.0 / math.pi) * sigma**3,
        -math.inf,
        math.inf,
    )


def berry_esseen_bound(
    third_absolute_moment: float,
    standard_deviation: float,
    n: int,
    *,
    constant: float = BERRY_ESSEEN_CONSTANT,
) -> float:
    """Return the Berry-Esseen bound ``C rho / (sigma^3 sqrt(n))``.

    It bounds ``sup_z |P(Z_n <= z) - Phi(z)|`` for i.i.d. summands with a
    finite third moment. The default ``C = 0.4748`` is Shevtsova's (2011)
    bound for identically distributed summands. The value is not capped at
    one even though a Kolmogorov distance never exceeds one.

    Args:
        third_absolute_moment: ``rho = E|X - mu|^3 > 0``.
        standard_deviation: ``sigma > 0``.
        n: Positive sample size.
        constant: Positive Berry-Esseen constant.

    Returns:
        Upper bound on the Kolmogorov distance to ``Phi``.

    Raises:
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-LIM-001].

    References:
        [REF-SHEVTSOVA-2011], [REF-CASELLA-BERGER-2002, Section 5.5.3].

    """
    rho = _positive(third_absolute_moment, name="third_absolute_moment")
    sigma = _positive(standard_deviation, name="standard_deviation")
    c = _positive(constant, name="constant")
    size = _sample_size(n)
    return c * rho / (sigma**3 * math.sqrt(size))


# Stirling-formula errors log(k!) - log(sqrt(2 pi k) (k/e)^k) for k = 1..15,
# evaluated with mpmath at 50 significant digits and rounded to doubles; they
# agree with the integer entries of the table in R's stirlerr.c.
_STIRLING_ERRORS = (
    0.0,
    0.08106146679532726,
    0.0413406959554093,
    0.02767792568499834,
    0.020790672103765093,
    0.016644691189821193,
    0.013876128823070748,
    0.01189670994589177,
    0.010411265261972096,
    0.009255462182712733,
    0.00833056343336287,
    0.007573675487951841,
    0.00694284010720953,
    0.006408994188004207,
    0.0059513701127588475,
    0.005554733551962801,
)
_LOG_TWO_PI = math.log(2.0 * math.pi)
# exp(-750) / sqrt(2 pi) underflows to zero, so masses whose deviance exceeds
# this are exactly zero in floating point and need not be evaluated.
_DEVIANCE_CUTOFF = 750.0


def _stirling_error(k: int) -> float:
    """Return ``delta(k) = log(k!) - log(sqrt(2 pi k) (k/e)^k)``.

    Tabulated for ``k <= 15``; otherwise the Stirling-De Moivre series
    ``1/(12k) - 1/(360k^3) + 1/(1260k^5) - 1/(1680k^7) + 1/(1188k^9)``,
    truncated as in Loader's implementation.

    References:
        [REF-LOADER-2000, Equation 4].

    """
    if k <= 15:
        return _STIRLING_ERRORS[k]
    square = float(k) * float(k)
    s0, s1, s2, s3, s4 = 1 / 12, 1 / 360, 1 / 1260, 1 / 1680, 1 / 1188
    if k > 500:
        return (s0 - s1 / square) / k
    if k > 80:
        return (s0 - (s1 - s2 / square) / square) / k
    if k > 35:
        return (s0 - (s1 - (s2 - s3 / square) / square) / square) / k
    return (s0 - (s1 - (s2 - (s3 - s4 / square) / square) / square) / square) / k


def _deviance_term(x: float, mean: float) -> float:
    """Return ``x log(x / mean) + mean - x`` without cancellation.

    For ``|x - mean| < 0.1 (x + mean)`` the series
    ``(x - mean)^2/(x + mean) + 2 x sum_j v^(2j+1)/(2j+1)`` with
    ``v = (x - mean)/(x + mean)`` is summed; otherwise the closed form.

    References:
        [REF-LOADER-2000, Equation 6].

    """
    if abs(x - mean) < 0.1 * (x + mean):
        v = (x - mean) / (x + mean)
        total = (x - mean) * v
        power = 2.0 * x * v
        v_squared = v * v
        for j in range(1, 1000):
            power *= v_squared
            updated = total + power / (2 * j + 1)
            if updated == total:
                return updated
            total = updated
        return total
    return x * math.log(x / mean) + mean - x


def _binomial_pmf(n: int, p: float) -> list[float]:
    """Return the ``Binomial(n, p)`` probability mass function as a list.

    Loader's saddle-point form
    ``p(k) = sqrt(n / (2 pi k (n - k))) exp(delta(n) - delta(k) - delta(n - k)
    - D_0(k, n p) - D_0(n - k, n q))`` keeps full relative precision for
    every ``n``, where the ``lgamma`` difference ``log n! - log k! - ...``
    loses about ``n eps`` (``1e-8`` relative at ``n = 4e6``). Masses whose
    deviance ``D_0(k, n p) + D_0(n - k, n q)`` exceeds 750 underflow to zero
    and are left at zero without evaluation.

    References:
        [REF-LOADER-2000, Equations 3-6], [EQ-LIM-002],
        [REF-CASELLA-BERGER-2002, Section 3.2].

    """
    q = 1.0 - p
    mean_p = n * p
    mean_q = n * q
    pmf = [0.0] * (n + 1)
    if n == 0:
        return [1.0]
    # The end masses q^n and p^n, in the forms of Loader's implementation.
    pmf[0] = math.exp(
        -_deviance_term(n, mean_q) - mean_p if p < 0.1 else n * math.log(q)
    )
    pmf[n] = math.exp(
        -_deviance_term(n, mean_p) - mean_q if q < 0.1 else n * math.log(p)
    )
    if n == 1:
        return pmf

    def deviance(k: int) -> float:
        """Return the binomial deviance ``D(k; n, p)``.

        References:
            [REF-LOADER-2000, Equation 6].

        """
        return _deviance_term(k, mean_p) + _deviance_term(n - k, mean_q)

    # The deviance is convex in k with its minimum at n p: bisect for the
    # range where masses can be representable.
    mode = min(max(int(mean_p), 1), n - 1)
    low, high = 1, mode
    if deviance(low) > _DEVIANCE_CUTOFF:
        while high - low > 1:
            middle = (low + high) // 2
            if deviance(middle) > _DEVIANCE_CUTOFF:
                low = middle
            else:
                high = middle
        first = high
    else:
        first = 1
    low, high = mode, n - 1
    if deviance(high) > _DEVIANCE_CUTOFF:
        while high - low > 1:
            middle = (low + high) // 2
            if deviance(middle) > _DEVIANCE_CUTOFF:
                high = middle
            else:
                low = middle
        last = low
    else:
        last = n - 1
    log_n_error = _stirling_error(n)
    for k in range(first, last + 1):
        # log((n - k)/n) to absolute rounding error: log1p(-k/n) would
        # amplify the rounding of k/n by n/(n - k) when k is close to n.
        log_fraction = math.log1p(-k / n) if 2 * k < n else math.log((n - k) / n)
        exponent = (
            log_n_error
            - _stirling_error(k)
            - _stirling_error(n - k)
            - deviance(k)
            - 0.5 * (_LOG_TWO_PI + math.log(k) + log_fraction)
        )
        pmf[k] = math.exp(exponent)
    return pmf


def standardized_mean_cdf(summand: SummandLaw, n: int, z: float) -> float:
    """Return the exact ``P(Z_n <= z)`` from the closed-form law of the sum.

    Bernoulli: ``P(S <= floor(n p + z sqrt(n p q)))`` with ``S ~ Bin(n, p)``.
    Exponential: ``P(n, n + z sqrt(n))`` (regularized lower incomplete gamma,
    free of the rate). Normal: ``Phi(z)``.

    Args:
        summand: Summand law.
        n: Positive sample size.
        z: Finite evaluation point.

    Returns:
        Exact distribution function of the standardized mean.

    Equation:
        [EQ-LIM-002].

    References:
        [REF-CASELLA-BERGER-2002, Sections 3.2, 3.3, and 5.2],
        [REF-NIST-DLMF-GAMMA, Section 8.2].

    """
    size = _sample_size(n)
    point = require_finite_real(z, name="z")
    if summand.kind == "normal":
        return standard_normal_cdf(point)
    if summand.kind == "exponential":
        argument = size + point * math.sqrt(size)
        return 0.0 if argument <= 0.0 else regularized_gamma_p(size, argument)
    p = summand.parameter[0]
    threshold = size * p + point * math.sqrt(size * p * (1.0 - p))
    upper = math.floor(threshold + _INTEGER_TOLERANCE)
    if upper < 0:
        return 0.0
    if upper >= size:
        return 1.0
    return min(1.0, math.fsum(_binomial_pmf(size, p)[: upper + 1]))


def edgeworth_cdf(z: float, n: int, skewness: float) -> float:
    """Return the one-term Edgeworth approximation to ``P(Z_n <= z)``.

    ``G_n(z) = Phi(z) - phi(z) gamma (z^2 - 1) / (6 sqrt(n))``. The error is
    ``o(n^-1/2)`` uniformly in ``z`` for non-lattice summands with a finite
    third moment; lattice summands need an additional discontinuous term.

    Args:
        z: Finite evaluation point.
        n: Positive sample size.
        skewness: Summand skewness ``gamma``.

    Returns:
        Edgeworth approximation (not necessarily inside ``[0, 1]``).

    Equation:
        [EQ-LIM-004].

    References:
        [REF-HALL-1992, Chapter 2].

    """
    point = require_finite_real(z, name="z")
    size = _sample_size(n)
    gamma = require_finite_real(skewness, name="skewness")
    correction = gamma * (point * point - 1.0) / (6.0 * math.sqrt(size))
    return standard_normal_cdf(point) - _normal_density(point) * correction


def _golden_maximum(
    function: Callable[[float], float], low: float, high: float
) -> float:
    """Return the maximum found by golden-section search on ``[low, high]``.

    References:
        [REF-BRENT-1973, Chapter 5].

    """
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(_GOLDEN_STEPS):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = function(d)
    return max(fc, fd, function(low), function(high))


def _continuous_distance(
    exact: Callable[[float], float],
    approximation: Callable[[float], float],
    low: float,
    high: float,
    kink: float,
) -> float:
    """Return ``sup |exact - approximation|`` by a grid plus local refinement.

    The supremum is located on a uniform grid (with ``kink`` added as a node
    when it lies inside the window) and every grid local maximum among the
    five largest is refined by golden-section search on its neighbouring
    cells.

    References:
        [EQ-LIM-003], [REF-BRENT-1973, Chapter 5].

    """

    def gap(z: float) -> float:
        """Return the absolute approximation error at ``z``.

        References:
            [EQ-LIM-003].

        """
        return abs(exact(z) - approximation(z))

    step = (high - low) / (_GRID_POINTS - 1)
    nodes = {low + index * step for index in range(_GRID_POINTS)}
    if low < kink < high:
        nodes.add(kink)
    grid = sorted(nodes)
    values = [gap(z) for z in grid]
    peaks = [
        index
        for index in range(len(grid))
        if (index == 0 or values[index] >= values[index - 1])
        and (index == len(grid) - 1 or values[index] >= values[index + 1])
    ]
    peaks.sort(key=lambda index: values[index], reverse=True)
    best = max(values)
    for index in peaks[:5]:
        left = grid[max(index - 1, 0)]
        right = grid[min(index + 1, len(grid) - 1)]
        if left < kink < right:
            best = max(
                best,
                _golden_maximum(gap, left, kink),
                _golden_maximum(gap, kink, right),
            )
        elif right > left:
            best = max(best, _golden_maximum(gap, left, right))
    return best


def kolmogorov_distance_to_normal(
    summand: SummandLaw, n: int, *, edgeworth: bool = False
) -> float:
    """Return the exact ``sup_z |P(Z_n <= z) - A_n(z)|``.

    ``A_n`` is ``Phi`` or, with ``edgeworth=True``, the one-term Edgeworth
    approximation. For the lattice Bernoulli case the supremum is attained at
    a jump, so both one-sided limits at each of the ``n + 1`` atoms are
    compared with ``Phi`` exactly (the binomial masses use Loader's
    saddle-point form, accurate to rounding for any ``n``). For the
    exponential case the continuous gap is maximized numerically on
    ``[max(-sqrt(n) - 8, -40), 10]`` (1601-node grid, step at most
    ``50/1600`` whatever ``n``, golden-section refinement). Below the window
    the gap is below ``Phi(-9) = 1.2e-19`` (the exact law is zero below
    ``-sqrt(n)``, and both laws underflow below ``-40``); above it the gap is
    at most ``Q(n, n + 10 sqrt(n))``, which decreases in ``n`` from
    ``exp(-11) = 1.7e-5`` at ``n = 1`` (against a supremum of ``0.159``) to
    ``1.8e-15`` at ``n = 100``, far below the supremum for every ``n``. The
    normal case is exactly zero.

    Args:
        summand: Summand law.
        n: Positive sample size.
        edgeworth: Compare with the Edgeworth approximation instead of
            ``Phi``; only defined here for non-lattice summands.

    Returns:
        Kolmogorov distance.

    Raises:
        ValueError: If an Edgeworth comparison is requested for a lattice
            summand.

    Equation:
        [EQ-LIM-002], [EQ-LIM-003], [EQ-LIM-004].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.3], [REF-HALL-1992, Chapter 2].

    """
    size = _sample_size(n)
    if summand.kind == "normal":
        return 0.0
    if summand.kind == "bernoulli":
        if edgeworth:
            raise ValueError(
                "the one-term Edgeworth expansion needs a non-lattice summand"
            )
        p = summand.parameter[0]
        scale = math.sqrt(size * p * (1.0 - p))
        cumulative = 0.0
        distance = 0.0
        for k, mass in enumerate(_binomial_pmf(size, p)):
            phi = standard_normal_cdf((k - size * p) / scale)
            distance = max(distance, abs(phi - cumulative))
            cumulative += mass
            distance = max(distance, abs(min(cumulative, 1.0) - phi))
        return distance
    gamma = summand.skewness
    root = math.sqrt(size)

    def standardized(z: float) -> float:
        """Return the standardized value of the rounded sum ``n + z sqrt(n)``.

        For large ``n`` the sum is rounded to ``ulp(n)``, which moves the
        exact law by up to ``phi(z) ulp(n) / (2 sqrt(n))`` (``1e-13`` at
        ``n = 1e8``); both laws are therefore compared at the point the
        rounded sum represents, so a maximizer cannot pick up that noise.

        References:
            [EQ-LIM-002].

        """
        return ((size + z * root) - size) / root

    def exact(z: float) -> float:
        """Return the exact standardized-mean CDF of the exponential sum.

        References:
            [EQ-LIM-002].

        """
        argument = size + z * root
        return 0.0 if argument <= 0.0 else regularized_gamma_p(size, argument)

    def approximation(z: float) -> float:
        """Return the selected normal or Edgeworth approximation.

        References:
            [EQ-LIM-004].

        """
        point = standardized(z)
        if edgeworth:
            return edgeworth_cdf(point, size, gamma)
        return standard_normal_cdf(point)

    # A fixed window keeps the grid step (at most 50/1600) independent of n;
    # below -40 both distribution functions are zero in floating point.
    low = max(-root - 8.0, -40.0)
    return _continuous_distance(exact, approximation, low, 10.0, -root)


@dataclass(frozen=True, slots=True)
class CLTErrorRow:
    """Exact central-limit error at one sample size.

    Args:
        n: Sample size.
        kolmogorov_normal: Exact ``sup |F_n - Phi|``.
        kolmogorov_edgeworth: Exact ``sup |F_n - G_n|`` or ``None`` for a
            lattice summand.
        berry_esseen: Berry-Esseen bound with Shevtsova's constant.

    References:
        [EQ-LIM-001], [EQ-LIM-003], [EQ-LIM-004].

    """

    n: int
    kolmogorov_normal: float
    kolmogorov_edgeworth: float | None
    berry_esseen: float


def clt_error_profile(
    summand: SummandLaw, sizes: Iterable[int]
) -> tuple[CLTErrorRow, ...]:
    """Tabulate exact CLT and Edgeworth errors against the Berry-Esseen bound.

    Args:
        summand: Summand law (for the normal case all distances are zero).
        sizes: Positive sample sizes.

    Returns:
        One row per sample size.

    Equation:
        [EQ-LIM-001], [EQ-LIM-003], [EQ-LIM-004].

    References:
        [REF-SHEVTSOVA-2011], [REF-HALL-1992, Chapter 2].

    """
    rows: list[CLTErrorRow] = []
    for raw in sizes:
        n = _sample_size(raw)
        edgeworth = (
            None
            if summand.lattice
            else kolmogorov_distance_to_normal(summand, n, edgeworth=True)
        )
        rows.append(
            CLTErrorRow(
                n,
                kolmogorov_distance_to_normal(summand, n),
                edgeworth,
                berry_esseen_bound(
                    summand.third_absolute_central_moment,
                    summand.standard_deviation,
                    n,
                ),
            )
        )
    return tuple(rows)


def chebyshev_bound(variance: float, n: int, t: float) -> float:
    """Return Chebyshev's ``P(|Xbar_n - mu| >= t) <= sigma^2 / (n t^2)``.

    Args:
        variance: Summand variance ``sigma^2 > 0``.
        n: Positive sample size.
        t: Positive deviation.

    Returns:
        Two-sided tail bound, capped at one.

    Equation:
        [EQ-LIM-005].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.6].

    """
    sigma2 = _positive(variance, name="variance")
    deviation = _positive(t, name="t")
    return min(1.0, sigma2 / (_sample_size(n) * deviation * deviation))


def hoeffding_bound(
    lower: float, upper: float, n: int, t: float, *, two_sided: bool = False
) -> float:
    """Return Hoeffding's bound ``exp(-2 n t^2 / (b - a)^2)``.

    It bounds ``P(Xbar_n - mu >= t)`` for independent summands in
    ``[a, b]``; the two-sided version doubles it.

    Args:
        lower: Lower support bound ``a``.
        upper: Upper support bound ``b > a``.
        n: Positive sample size.
        t: Positive deviation.
        two_sided: Bound ``P(|Xbar_n - mu| >= t)`` instead.

    Returns:
        Tail bound, capped at one.

    Raises:
        ValueError: If ``upper <= lower``.

    Equation:
        [EQ-LIM-006].

    References:
        [REF-HOEFFDING-1963], [REF-WASSERMAN-2004, Chapter 4].

    """
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    if b <= a:
        raise ValueError("upper must exceed lower")
    deviation = _positive(t, name="t")
    value = math.exp(-2.0 * _sample_size(n) * deviation * deviation / (b - a) ** 2)
    return min(1.0, 2.0 * value if two_sided else value)


def bernstein_bound(
    variance: float, bound: float, n: int, t: float, *, two_sided: bool = False
) -> float:
    """Return Bernstein's ``exp(-n t^2 / (2 (sigma^2 + c t / 3)))``.

    It bounds ``P(Xbar_n - mu >= t)`` for independent summands with variance
    ``sigma^2`` and ``|X_i - mu| <= c``; the two-sided version doubles it.

    Args:
        variance: Summand variance ``sigma^2 > 0``.
        bound: Almost-sure bound ``c > 0`` on ``|X - mu|``.
        n: Positive sample size.
        t: Positive deviation.
        two_sided: Bound ``P(|Xbar_n - mu| >= t)`` instead.

    Returns:
        Tail bound, capped at one.

    Equation:
        [EQ-LIM-007].

    References:
        [REF-BOUCHERON-LUGOSI-MASSART-2013, Corollary 2.11].

    """
    sigma2 = _positive(variance, name="variance")
    c = _positive(bound, name="bound")
    deviation = _positive(t, name="t")
    exponent = (
        _sample_size(n) * deviation * deviation / (2.0 * (sigma2 + c * deviation / 3.0))
    )
    value = math.exp(-exponent)
    return min(1.0, 2.0 * value if two_sided else value)


def _exact_tails(summand: SummandLaw, n: int, t: float) -> tuple[float, float]:
    """Return exact ``P(Xbar - mu >= t)`` and ``P(Xbar - mu <= -t)``.

    References:
        [EQ-LIM-002], [REF-CASELLA-BERGER-2002, Section 5.2].

    """
    if summand.kind == "normal":
        scaled = math.sqrt(n) * t / summand.standard_deviation
        upper = 0.5 * math.erfc(scaled / _SQRT_TWO)
        return upper, upper
    if summand.kind == "exponential":
        lam = summand.parameter[0]
        upper = regularized_gamma_q(n, n * (1.0 + lam * t))
        lower_argument = n * (1.0 - lam * t)
        lower = regularized_gamma_p(n, lower_argument) if lower_argument > 0.0 else 0.0
        return upper, lower
    p = summand.parameter[0]
    pmf = _binomial_pmf(n, p)
    high = math.ceil(n * (p + t) - _INTEGER_TOLERANCE)
    low = math.floor(n * (p - t) + _INTEGER_TOLERANCE)
    upper = math.fsum(pmf[max(high, 0) :]) if high <= n else 0.0
    lower = math.fsum(pmf[: low + 1]) if low >= 0 else 0.0
    return min(upper, 1.0), min(lower, 1.0)


def sample_mean_upper_tail(summand: SummandLaw, n: int, t: float) -> float:
    """Return the exact one-sided tail ``P(Xbar_n - mu >= t)``.

    Bernoulli uses the binomial upper tail at ``ceil(n (p + t))``;
    exponential uses ``Q(n, n (1 + lambda t))``; normal uses
    ``1 - Phi(sqrt(n) t / sigma)``.

    Args:
        summand: Summand law.
        n: Positive sample size.
        t: Positive deviation.

    Returns:
        Exact tail probability.

    Equation:
        [EQ-LIM-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.2], [REF-NIST-DLMF-GAMMA,
        Section 8.2].

    """
    return _exact_tails(summand, _sample_size(n), _positive(t, name="t"))[0]


def sample_mean_two_sided_tail(summand: SummandLaw, n: int, t: float) -> float:
    """Return the exact two-sided tail ``P(|Xbar_n - mu| >= t)``.

    Args:
        summand: Summand law.
        n: Positive sample size.
        t: Positive deviation.

    Returns:
        Exact tail probability.

    Equation:
        [EQ-LIM-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.2].

    """
    upper, lower = _exact_tails(summand, _sample_size(n), _positive(t, name="t"))
    return min(1.0, upper + lower)


@dataclass(frozen=True, slots=True)
class TailBoundComparison:
    """Exact sample-mean tails next to distribution-free bounds.

    Args:
        n: Sample size.
        t: Deviation.
        exact_upper: Exact ``P(Xbar - mu >= t)``.
        exact_two_sided: Exact ``P(|Xbar - mu| >= t)``.
        chebyshev: Two-sided Chebyshev bound.
        hoeffding: One-sided Hoeffding bound, ``None`` if unbounded support.
        bernstein: One-sided Bernstein bound, ``None`` if unbounded support.

    References:
        [EQ-LIM-005], [EQ-LIM-006], [EQ-LIM-007].

    """

    n: int
    t: float
    exact_upper: float
    exact_two_sided: float
    chebyshev: float
    hoeffding: float | None
    bernstein: float | None


def compare_tail_bounds(summand: SummandLaw, n: int, t: float) -> TailBoundComparison:
    """Compare exact sample-mean tails with Chebyshev, Hoeffding, Bernstein.

    Hoeffding uses the support ``[lower, upper]``; Bernstein uses
    ``c = max(mu - lower, upper - mu)``. Both are omitted for unbounded
    summands, where their hypotheses fail.

    Args:
        summand: Summand law.
        n: Positive sample size.
        t: Positive deviation.

    Returns:
        Comparison record.

    Equation:
        [EQ-LIM-002], [EQ-LIM-005], [EQ-LIM-006], [EQ-LIM-007].

    References:
        [REF-HOEFFDING-1963], [REF-BOUCHERON-LUGOSI-MASSART-2013,
        Corollary 2.11], [REF-CASELLA-BERGER-2002, Section 3.6].

    """
    size = _sample_size(n)
    deviation = _positive(t, name="t")
    upper, lower = _exact_tails(summand, size, deviation)
    hoeffding: float | None = None
    bernstein: float | None = None
    if summand.bounded:
        hoeffding = hoeffding_bound(summand.lower, summand.upper, size, deviation)
        c = max(summand.mean - summand.lower, summand.upper - summand.mean)
        bernstein = bernstein_bound(summand.variance, c, size, deviation)
    return TailBoundComparison(
        size,
        deviation,
        upper,
        min(1.0, upper + lower),
        chebyshev_bound(summand.variance, size, deviation),
        hoeffding,
        bernstein,
    )


def delta_method_variance(derivative: float, variance: float, n: int) -> float:
    """Return the delta-method variance ``g'(mu)^2 sigma^2 / n`` of ``g(Xbar)``.

    Args:
        derivative: ``g'(mu)``.
        variance: Summand variance ``sigma^2 >= 0``.
        n: Positive sample size.

    Returns:
        First-order variance approximation.

    Equation:
        [EQ-LIM-008].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """
    slope = require_finite_real(derivative, name="derivative")
    sigma2 = require_finite_real(variance, name="variance")
    if sigma2 < 0.0:
        raise ValueError("variance must be non-negative")
    return slope * slope * sigma2 / _sample_size(n)


@dataclass(frozen=True, slots=True)
class DeltaMethodComparison:
    """Exact variance of ``g(Xbar_n)`` next to its delta-method approximation.

    Args:
        n: Sample size.
        exact_variance: Exact ``Var g(Xbar_n)``.
        delta_variance: ``g'(mu)^2 sigma^2 / n``.

    References:
        [EQ-LIM-008].

    """

    n: int
    exact_variance: float
    delta_variance: float

    @property
    def ratio(self) -> float:
        """Return ``exact / delta`` (infinite when the delta variance is zero).

        References:
            [EQ-LIM-008].

        """
        if self.delta_variance == 0.0:
            return math.inf if self.exact_variance > 0.0 else 1.0
        return self.exact_variance / self.delta_variance


def delta_method_normal_square(
    mean: float, standard_deviation: float, n: int
) -> DeltaMethodComparison:
    """Compare ``Var(Xbar^2)`` for normal data with the delta method.

    Exact: ``4 mu^2 sigma^2 / n + 2 sigma^4 / n^2``. Delta:
    ``4 mu^2 sigma^2 / n``.

    Args:
        mean: Finite ``mu``.
        standard_deviation: Positive ``sigma``.
        n: Positive sample size.

    Returns:
        Comparison record.

    Equation:
        [EQ-LIM-008].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """
    mu = require_finite_real(mean, name="mean")
    sigma = _positive(standard_deviation, name="standard_deviation")
    size = _sample_size(n)
    s2 = sigma * sigma / size
    return DeltaMethodComparison(
        size, 4.0 * mu * mu * s2 + 2.0 * s2 * s2, 4.0 * mu * mu * s2
    )


def delta_method_normal_exp(
    mean: float, standard_deviation: float, n: int
) -> DeltaMethodComparison:
    """Compare ``Var(exp(Xbar))`` for normal data with the delta method.

    ``exp(Xbar)`` is lognormal: exact ``(e^{s^2} - 1) e^{2 mu + s^2}`` with
    ``s^2 = sigma^2 / n``; delta ``e^{2 mu} s^2``.

    Args:
        mean: Finite ``mu``.
        standard_deviation: Positive ``sigma``.
        n: Positive sample size.

    Returns:
        Comparison record.

    Equation:
        [EQ-LIM-008].

    References:
        [REF-CASELLA-BERGER-2002, Sections 3.3 and 5.5.4].

    """
    mu = require_finite_real(mean, name="mean")
    sigma = _positive(standard_deviation, name="standard_deviation")
    size = _sample_size(n)
    s2 = sigma * sigma / size
    exact = math.expm1(s2) * math.exp(2.0 * mu + s2)
    return DeltaMethodComparison(size, exact, math.exp(2.0 * mu) * s2)


def delta_method_exponential_reciprocal(rate: float, n: int) -> DeltaMethodComparison:
    """Compare ``Var(1 / Xbar)`` for exponential data with the delta method.

    ``Xbar ~ Gamma(n, n lambda)`` gives exact
    ``(n lambda)^2 / ((n - 1)^2 (n - 2))`` for ``n >= 3``; the delta method
    gives ``lambda^2 / n``.

    Args:
        rate: Positive ``lambda``.
        n: Sample size ``n >= 3``.

    Returns:
        Comparison record.

    Raises:
        ValueError: If ``n < 3`` (the exact variance is infinite).

    Equation:
        [EQ-LIM-008], [EQ-EST-010].

    References:
        [REF-CASELLA-BERGER-2002, Sections 5.5.4 and 7.3].

    """
    lam = _positive(rate, name="rate")
    size = _sample_size(n)
    if size < 3:
        raise ValueError("Var(1/Xbar) is finite only for n >= 3")
    exact = (size * lam) ** 2 / ((size - 1) ** 2 * (size - 2))
    return DeltaMethodComparison(size, exact, lam * lam / size)


def delta_method_bernoulli(
    function: Callable[[float], float],
    derivative: float,
    p: float,
    n: int,
) -> DeltaMethodComparison:
    """Compare ``Var g(Xbar)`` for Bernoulli data with the delta method.

    The exact variance is a finite sum over the binomial law of ``n Xbar``.

    Args:
        function: Transformation ``g`` on ``[0, 1]``.
        derivative: ``g'(p)``.
        p: Success probability in ``(0, 1)``.
        n: Positive sample size.

    Returns:
        Comparison record.

    Equation:
        [EQ-LIM-008].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """
    summand = bernoulli_summand(p)
    size = _sample_size(n)
    pmf = _binomial_pmf(size, summand.mean)
    values = [float(function(k / size)) for k in range(size + 1)]
    mean = math.fsum(w * v for w, v in zip(pmf, values, strict=True))
    exact = math.fsum(w * (v - mean) ** 2 for w, v in zip(pmf, values, strict=True))
    return DeltaMethodComparison(
        size, exact, delta_method_variance(derivative, summand.variance, size)
    )
