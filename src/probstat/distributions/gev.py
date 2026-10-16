"""Generalized extreme value distribution with a stable Gumbel limit."""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._base import ContinuousDistribution
from ._continuous import open_unit
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .gumbel import EULER_GAMMA
from .support import RealSupport

# zeta(k) for k = 2..12, used by the small-shape moment series.
_ZETA = (
    math.pi**2 / 6.0,
    1.2020569031595942,
    math.pi**4 / 90.0,
    1.0369277551433699,
    math.pi**6 / 945.0,
    1.0083492773819228,
    math.pi**8 / 9450.0,
    1.0020083928260822,
    math.pi**10 / 93555.0,
    1.0004941886041195,
    691.0 * math.pi**12 / 638512875.0,
)
_SERIES_THRESHOLD = 0.01
_TINY_PRODUCT = 1e-280
_EXP_LIMIT = 709.0


def _log_gamma_one_minus_over(shape: float) -> float:
    """Return ``lgamma(1 - xi) / xi`` by its Maclaurin series.

    ``log Gamma(1 - xi) = gamma_E xi + sum_(k>=2) zeta(k) xi^k / k`` for
    ``|xi| < 1``.

    Args:
        shape: Small shape ``xi`` with ``|xi| < 0.01``.

    Returns:
        Series value of ``lgamma(1 - xi) / xi`` (``gamma_E`` at zero).

    Equation:
        [EQ-DIST-GEV-003].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.7].

    """
    total = 0.0
    for index in range(len(_ZETA) - 1, -1, -1):
        order = index + 2
        total = total * shape + _ZETA[index] / order
    return EULER_GAMMA + shape * total


def _variance_ratio_series(shape: float) -> float:
    """Return ``(lgamma(1 - 2 xi) - 2 lgamma(1 - xi)) / xi^2`` by series.

    The ``gamma_E`` terms cancel exactly, leaving
    ``sum_(k>=2) zeta(k) (2^k - 2) xi^(k-2) / k``, whose limit is
    ``zeta(2) = pi^2/6``.

    Args:
        shape: Small shape ``xi`` with ``|xi| < 0.01``.

    Returns:
        Series value of the scaled log-gamma difference.

    Equation:
        [EQ-DIST-GEV-003].

    References:
        [REF-NIST-DLMF-GAMMA, Section 5.7].

    """
    total = 0.0
    for index in range(len(_ZETA) - 1, -1, -1):
        order = index + 2
        total = total * shape + _ZETA[index] * (2.0**order - 2.0) / order
    return total


@dataclass(frozen=True, slots=True)
class GeneralizedExtremeValue(ContinuousDistribution):
    """A GEV law ``F(x) = exp(-(1 + xi z)^(-1/xi))``, ``z = (x - mu)/sigma``.

    ``xi > 0`` is the heavy-tailed Frechet type with lower endpoint
    ``mu - sigma/xi``; ``xi < 0`` is the reversed-Weibull type with upper
    endpoint ``mu - sigma/xi``; ``xi = 0`` is the Gumbel law for maxima,
    ``exp(-exp(-z))``. Every formula uses ``log1p(xi z)/xi`` or
    ``expm1(-xi log L)/xi``, so the ``xi -> 0`` limit is continuous. This is
    the climate/hydrology sign convention; some libraries use ``c = -xi``.

    Args:
        mu: Finite location.
        sigma: Finite positive scale.
        xi: Finite shape.

    References:
        [REF-COLES-2001, Chapter 3], [REF-JKB-1995, Chapter 22].

    """

    mu: float = 0.0
    sigma: float = 1.0
    xi: float = 0.0

    def __post_init__(self) -> None:
        """Validate location, scale, and shape.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``sigma <= 0``.

        References:
            [EQ-DIST-GEV-001], [REF-COLES-2001, Chapter 3].

        """
        object.__setattr__(self, "mu", FINITE_REAL.validate(self.mu, name="mu"))
        object.__setattr__(
            self, "sigma", POSITIVE_FINITE.validate(self.sigma, name="sigma")
        )
        object.__setattr__(self, "xi", FINITE_REAL.validate(self.xi, name="xi"))

    @property
    def endpoint(self) -> float | None:
        """Return the finite support endpoint ``mu - sigma/xi``, if any.

        Returns:
            Lower endpoint for ``xi > 0``, upper endpoint for ``xi < 0``, and
            ``None`` for ``xi = 0`` or when the endpoint exceeds float range.

        Equation:
            [EQ-DIST-GEV-001].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        if self.xi == 0.0:
            return None
        try:
            value = self.mu - self.sigma / self.xi
        except OverflowError:
            return None
        return value if math.isfinite(value) else None

    @property
    def support(self) -> RealSupport:
        """Return the half-line or real line on which ``1 + xi z > 0``.

        References:
            [EQ-DIST-GEV-001], [REF-COLES-2001, Section 3.1].

        """
        end = self.endpoint
        if end is None:
            return RealSupport()
        if self.xi > 0.0:
            return RealSupport(end, None, lower_closed=True)
        return RealSupport(None, end, upper_closed=True)

    def _reduced(self, x: float) -> tuple[float, float]:
        """Return ``w = xi z`` and ``y = log(1 + w)/xi`` (``y = z`` at ``xi = 0``).

        Args:
            x: Finite real evaluation point.

        Returns:
            Pair ``(w, y)``; ``y`` is ``-inf`` below a lower endpoint and
            ``inf`` above an upper endpoint.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GEV-001].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        z = (require_finite_real(x, name="x") - self.mu) / self.sigma
        w = self.xi * z
        if w <= -1.0:
            return w, (-math.inf if self.xi > 0.0 else math.inf)
        if abs(w) < _TINY_PRODUCT:
            return w, z
        return w, math.log1p(w) / self.xi

    @staticmethod
    def _exponent(y: float) -> float:
        """Return ``t = exp(-y)`` with overflow mapped to infinity.

        Args:
            y: Reduced variate.

        Returns:
            ``t`` such that ``F = exp(-t)``.

        Equation:
            [EQ-DIST-GEV-001].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        return math.inf if -y > _EXP_LIMIT else math.exp(-y)

    def log_pdf(self, x: float) -> float:
        """Evaluate ``-log sigma - (1 + xi) y - exp(-y)`` on the support.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density, ``-inf`` outside the support; at an upper endpoint
            (``xi < 0``) it is ``-inf``, ``-log sigma``, or ``inf`` for
            ``xi > -1``, ``xi = -1``, or ``xi < -1``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GEV-001].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        w, y = self._reduced(x)
        if w < -1.0:
            return -math.inf
        if w == -1.0:
            if self.xi > -1.0:
                return -math.inf
            if self.xi == -1.0:
                return -math.log(self.sigma)
            return math.inf
        t = self._exponent(y)
        if t == math.inf:
            return -math.inf
        return -math.log(self.sigma) - (1.0 + self.xi) * y - t

    def pdf(self, x: float) -> float:
        """Evaluate the GEV density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GEV-001].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        log_density = self.log_pdf(x)
        if log_density == math.inf:
            return math.inf
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate ``F(x) = exp(-exp(-y))``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GEV-001].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        _, y = self._reduced(x)
        return math.exp(-self._exponent(y))

    def sf(self, x: float) -> float:
        """Evaluate the direct survival ``-expm1(-exp(-y))``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting the CDF from one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GEV-001].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        _, y = self._reduced(x)
        return -math.expm1(-self._exponent(y))

    def _from_tails(self, lower: float, upper: float) -> float:
        """Return ``mu + sigma expm1(-xi log L)/xi`` with ``L = -log F``.

        ``L`` is ``-log(p)`` or ``-log1p(-q)``, whichever argument is smaller.

        Args:
            lower: Lower-tail probability ``p`` in ``(0, 1)``.
            upper: Complementary upper-tail probability ``q``.

        Returns:
            Physical quantile.

        Raises:
            OverflowError: If the quantile exceeds float range.

        Equation:
            [EQ-DIST-GEV-002].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        exponent = -math.log(lower) if lower <= upper else -math.log1p(-upper)
        log_exponent = math.log(exponent)
        power = -self.xi * log_exponent
        if abs(power) < _TINY_PRODUCT:
            reduced = -log_exponent
        else:
            if power > _EXP_LIMIT:
                raise OverflowError("GEV quantile exceeds float range")
            reduced = math.expm1(power) / self.xi
        result = self.mu + self.sigma * reduced
        if not math.isfinite(result):
            raise OverflowError("GEV quantile exceeds float range")
        return result

    def quantile(self, probability: float) -> float:
        """Return the closed-form GEV quantile.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Quantile; the endpoints map to the support ends.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.
            OverflowError: If a finite quantile exceeds float range.

        Equation:
            [EQ-DIST-GEV-002].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        p = require_probability(probability, name="probability")
        end = self.endpoint
        if p == 0.0:
            return end if end is not None and self.xi > 0.0 else -math.inf
        if p == 1.0:
            return end if end is not None and self.xi < 0.0 else math.inf
        return self._from_tails(p, 1.0 - p)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function without forming ``1 - q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Return level exceeded with probability ``q``.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.
            OverflowError: If a finite quantile exceeds float range.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-GEV-002].

        References:
            [REF-COLES-2001, Section 3.1].

        """
        q = require_probability(probability, name="probability")
        end = self.endpoint
        if q == 0.0:
            return end if end is not None and self.xi < 0.0 else math.inf
        if q == 1.0:
            return end if end is not None and self.xi > 0.0 else -math.inf
        return self._from_tails(1.0 - q, q)

    @property
    def mean(self) -> float:
        """Return ``mu + sigma (Gamma(1 - xi) - 1)/xi`` (``mu + sigma gamma_E``).

        Returns:
            Mean, or ``inf`` for ``xi >= 1``.

        A mean beyond the float range is returned as ``+-inf`` (properties
        never raise).

        Equation:
            [EQ-DIST-GEV-003].

        References:
            [REF-COLES-2001, Section 3.1], [REF-NIST-DLMF-GAMMA, Section 5.7].

        """
        if self.xi >= 1.0:
            return math.inf
        if abs(self.xi) < _SERIES_THRESHOLD:
            if abs(self.xi) < _TINY_PRODUCT:
                factor = EULER_GAMMA
            else:
                series = self.xi * _log_gamma_one_minus_over(self.xi)
                factor = math.expm1(series) / self.xi
        else:
            try:
                factor = (math.gamma(1.0 - self.xi) - 1.0) / self.xi
            except OverflowError:
                return -math.inf if self.xi < 0.0 else math.inf
        result = self.mu + self.sigma * factor
        if not math.isfinite(result):
            return -math.inf if self.xi < 0.0 else math.inf
        return result

    @property
    def variance(self) -> float:
        """Return ``sigma^2 (Gamma(1-2 xi) - Gamma(1-xi)^2) / xi^2``.

        The ``xi -> 0`` limit ``sigma^2 pi^2 / 6`` is reached through a
        cancellation-free log-gamma series for ``|xi| < 0.01``.

        Returns:
            Variance, or ``inf`` for ``xi >= 1/2``.

        A variance beyond the float range is returned as ``inf``.

        Equation:
            [EQ-DIST-GEV-003].

        References:
            [REF-COLES-2001, Section 3.1], [REF-NIST-DLMF-GAMMA, Section 5.7].

        """
        if self.xi >= 0.5:
            return math.inf
        if abs(self.xi) < _SERIES_THRESHOLD:
            ratio = _variance_ratio_series(self.xi)
            difference = ratio * self.xi * self.xi
            relative = math.expm1(difference) / difference if difference else 1.0
            log_first = self.xi * _log_gamma_one_minus_over(self.xi)
            factor = math.exp(2.0 * log_first) * ratio * relative
        else:
            try:
                first = math.gamma(1.0 - self.xi)
                second = math.gamma(1.0 - 2.0 * self.xi)
                factor = (second - first * first) / (self.xi * self.xi)
            except OverflowError:
                return math.inf
        result = self.sigma * self.sigma * factor
        if not math.isfinite(result):
            return math.inf
        return max(result, 0.0)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible GEV observations by inverse transform.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.
            OverflowError: If a variate exceeds float range.

        Equation:
            [EQ-DIST-GEV-002].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        observations: list[float] = []
        for _ in range(checked_size):
            u = open_unit(generator)
            observations.append(self._from_tails(u, 1.0 - u))
        return tuple(observations)
