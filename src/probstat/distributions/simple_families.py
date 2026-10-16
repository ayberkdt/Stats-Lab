"""Closed-form continuous families: Cauchy, Laplace, Rayleigh, and Pareto (type I).

Every distribution function, quantile, and inverse survival function here
has an elementary closed form; each is written so that its tail keeps
relative accuracy (for example ``F(x) = atan2(1, -z)/pi`` for the Cauchy law
instead of ``1/2 + atan(z)/pi``, which cancels for ``z << 0``). Sampling is
by inversion of an open-interval uniform from the call-local stream.
"""

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
from .support import RealSupport

EULER_GAMMA = 0.57721566490153286061


@dataclass(frozen=True, slots=True)
class Cauchy(ContinuousDistribution):
    """The Cauchy law with location ``loc`` and scale ``scale``.

    ``f(x) = 1 / (pi s (1 + z^2))`` with ``z = (x - loc)/s``. No moment of
    order one or higher exists: ``mean``, ``variance``, ``skewness`` and
    ``kurtosis`` are ``nan`` (undefined), the median is ``loc``. It equals
    ``StudentT(1, loc, scale)``.

    Args:
        loc: Finite location (median).
        scale: Finite positive scale (half the interquartile range).

    References:
        [REF-JKB-1994, Chapter 16].

    """

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate location and scale.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``scale <= 0``.

        References:
            [EQ-DIST-CAUCHY-001], [REF-JKB-1994, Chapter 16].

        """
        object.__setattr__(self, "loc", FINITE_REAL.validate(self.loc, name="loc"))
        object.__setattr__(
            self, "scale", POSITIVE_FINITE.validate(self.scale, name="scale")
        )

    @property
    def support(self) -> RealSupport:
        """Return the real line.

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return RealSupport()

    def _z(self, x: float) -> float:
        """Return the validated standardized point.

        References:
            [EQ-DIST-CAUCHY-001]. No new equation.

        """
        return (require_finite_real(x, name="x") - self.loc) / self.scale

    def log_pdf(self, x: float) -> float:
        """Return ``-log(pi s) - log(1 + z^2)``.

        Equation:
            [EQ-DIST-CAUCHY-001].

        References:
            [REF-JKB-1994, Chapter 16].

        """
        z = abs(self._z(x))
        if z > 1e8:
            tail = 2.0 * math.log(z) + math.log1p(1.0 / (z * z))
        else:
            tail = math.log1p(z * z)
        return -math.log(math.pi * self.scale) - tail

    def pdf(self, x: float) -> float:
        """Return the Cauchy density.

        Equation:
            [EQ-DIST-CAUCHY-001].

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return math.exp(self.log_pdf(x))

    def cdf(self, x: float) -> float:
        """Return ``F(x) = atan2(1, -z) / pi`` (accurate in the lower tail).

        Equation:
            [EQ-DIST-CAUCHY-001].

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return math.atan2(1.0, -self._z(x)) / math.pi

    def sf(self, x: float) -> float:
        """Return ``S(x) = atan2(1, z) / pi`` (accurate in the upper tail).

        Equation:
            [EQ-DIST-CAUCHY-001].

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return math.atan2(1.0, self._z(x)) / math.pi

    def quantile(self, probability: float) -> float:
        """Return ``loc - s / tan(pi p)`` (``p < 1/2``) or ``loc + s / tan(pi (1-p))``.

        Equation:
            [EQ-DIST-CAUCHY-001].

        References:
            [REF-JKB-1994, Chapter 16].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        if p == 0.5:
            return self.loc
        if p < 0.5:
            return self.loc - self.scale / math.tan(math.pi * p)
        return self.loc + self.scale / math.tan(math.pi * (1.0 - p))

    def isf(self, probability: float) -> float:
        """Return ``loc + s / tan(pi q)``.

        Equation:
            [EQ-DIST-CAUCHY-001], [EQ-DIST-ISF-001].

        References:
            [REF-JKB-1994, Chapter 16].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return -math.inf
        if q == 0.5:
            return self.loc
        if q < 0.5:
            return self.loc + self.scale / math.tan(math.pi * q)
        return self.loc - self.scale / math.tan(math.pi * (1.0 - q))

    @property
    def mean(self) -> float:
        """Return ``nan``: the Cauchy mean is undefined.

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return math.nan

    @property
    def variance(self) -> float:
        """Return ``nan``: the variance about an undefined mean is undefined.

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return math.nan

    @property
    def skewness(self) -> float:
        """Return ``nan`` (undefined).

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return math.nan

    @property
    def kurtosis(self) -> float:
        """Return ``nan`` (undefined).

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return math.nan

    @property
    def median(self) -> float:
        """Return the median ``loc``.

        References:
            [REF-JKB-1994, Chapter 16].

        """
        return self.loc

    @property
    def entropy(self) -> float:
        """Return ``log(4 pi s)``.

        Equation:
            [EQ-DIST-CAUCHY-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        return math.log(4.0 * math.pi * self.scale)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw Cauchy variates by inversion.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-CAUCHY-001].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(self.quantile(open_unit(generator)) for _ in range(count))


@dataclass(frozen=True, slots=True)
class Laplace(ContinuousDistribution):
    """The Laplace (double exponential) law ``f = exp(-|x - loc|/b) / (2 b)``.

    Args:
        loc: Finite location (mean, median, and mode).
        scale: Finite positive scale ``b``.

    References:
        [REF-JKB-1995, Chapter 24].

    """

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate location and scale.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``scale <= 0``.

        References:
            [EQ-DIST-LAPLACE-001], [REF-JKB-1995, Chapter 24].

        """
        object.__setattr__(self, "loc", FINITE_REAL.validate(self.loc, name="loc"))
        object.__setattr__(
            self, "scale", POSITIVE_FINITE.validate(self.scale, name="scale")
        )

    @property
    def support(self) -> RealSupport:
        """Return the real line.

        References:
            [REF-JKB-1995, Chapter 24].

        """
        return RealSupport()

    def _z(self, x: float) -> float:
        """Return the validated standardized point.

        References:
            [EQ-DIST-LAPLACE-001]. No new equation.

        """
        return (require_finite_real(x, name="x") - self.loc) / self.scale

    def log_pdf(self, x: float) -> float:
        """Return ``-|z| - log(2 b)``.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        return -abs(self._z(x)) - math.log(2.0 * self.scale)

    def pdf(self, x: float) -> float:
        """Return the Laplace density.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        return math.exp(self.log_pdf(x))

    def log_cdf(self, x: float) -> float:
        """Return ``log F``: ``z - log 2`` below the median.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        z = self._z(x)
        if z <= 0.0:
            return z - math.log(2.0)
        return math.log1p(-0.5 * math.exp(-z))

    def log_sf(self, x: float) -> float:
        """Return ``log S``: ``-z - log 2`` above the median.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        z = self._z(x)
        if z >= 0.0:
            return -z - math.log(2.0)
        return math.log1p(-0.5 * math.exp(z))

    def cdf(self, x: float) -> float:
        """Return ``exp(z)/2`` for ``z <= 0`` and ``1 - exp(-z)/2`` otherwise.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        z = self._z(x)
        return 0.5 * math.exp(z) if z <= 0.0 else 1.0 - 0.5 * math.exp(-z)

    def sf(self, x: float) -> float:
        """Return ``exp(-z)/2`` for ``z >= 0`` and ``1 - exp(z)/2`` otherwise.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        z = self._z(x)
        return 0.5 * math.exp(-z) if z >= 0.0 else 1.0 - 0.5 * math.exp(z)

    def quantile(self, probability: float) -> float:
        """Return ``loc + b log(2p)`` (``p <= 1/2``) or ``loc - b log(2(1-p))``.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        if p <= 0.5:
            return self.loc + self.scale * math.log(2.0 * p)
        return self.loc - self.scale * math.log(2.0 * (1.0 - p))

    def isf(self, probability: float) -> float:
        """Return ``loc - b log(2q)`` (``q <= 1/2``) or ``loc + b log(2(1-q))``.

        Equation:
            [EQ-DIST-LAPLACE-001], [EQ-DIST-ISF-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return -math.inf
        if q <= 0.5:
            return self.loc - self.scale * math.log(2.0 * q)
        return self.loc + self.scale * math.log(2.0 * (1.0 - q))

    @property
    def mean(self) -> float:
        """Return ``loc``.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        return self.loc

    @property
    def variance(self) -> float:
        """Return ``2 b^2``.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        return 2.0 * self.scale * self.scale

    @property
    def skewness(self) -> float:
        """Return ``0``.

        References:
            [REF-JKB-1995, Chapter 24].

        """
        return 0.0

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``3``.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-JKB-1995, Chapter 24].

        """
        return 3.0

    @property
    def entropy(self) -> float:
        """Return ``1 + log(2 b)``.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        return 1.0 + math.log(2.0 * self.scale)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw Laplace variates by inversion.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-LAPLACE-001].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(self.quantile(open_unit(generator)) for _ in range(count))


@dataclass(frozen=True, slots=True)
class Rayleigh(ContinuousDistribution):
    """The Rayleigh law ``f(x) = x / s^2 exp(-x^2 / (2 s^2))`` on ``x >= 0``.

    Args:
        scale: Finite positive scale ``s`` (the mode).

    References:
        [REF-JKB-1994, Chapter 18].

    """

    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the scale.

        Raises:
            TypeError: If ``scale`` is not real.
            ValueError: If ``scale`` is non-finite or non-positive.

        References:
            [EQ-DIST-RAYLEIGH-001], [REF-JKB-1994, Chapter 18].

        """
        object.__setattr__(
            self, "scale", POSITIVE_FINITE.validate(self.scale, name="scale")
        )

    @property
    def support(self) -> RealSupport:
        """Return ``[0, inf)``.

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return RealSupport(0.0, None, lower_closed=True)

    def _half_square(self, x: float) -> float:
        """Return ``x^2 / (2 s^2)`` (``0`` below the support).

        References:
            [EQ-DIST-RAYLEIGH-001]. No new equation.

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 0.0
        ratio = point / self.scale
        return 0.5 * ratio * ratio

    def log_pdf(self, x: float) -> float:
        """Return ``log x - 2 log s - x^2/(2 s^2)`` (``-inf`` for ``x <= 0``).

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return -math.inf
        return math.log(point) - 2.0 * math.log(self.scale) - self._half_square(point)

    def pdf(self, x: float) -> float:
        """Return the Rayleigh density.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        value = self.log_pdf(x)
        return math.exp(value) if value > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Return ``-expm1(-x^2/(2 s^2))``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return -math.expm1(-self._half_square(x))

    def sf(self, x: float) -> float:
        """Return ``exp(-x^2/(2 s^2))``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return math.exp(-self._half_square(x))

    def log_sf(self, x: float) -> float:
        """Return ``-x^2/(2 s^2)``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return -self._half_square(x)

    def quantile(self, probability: float) -> float:
        """Return ``s sqrt(-2 log(1 - p))``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        p = require_probability(probability, name="probability")
        if p == 1.0:
            return math.inf
        return self.scale * math.sqrt(-2.0 * math.log1p(-p))

    def isf(self, probability: float) -> float:
        """Return ``s sqrt(-2 log q)``.

        Equation:
            [EQ-DIST-RAYLEIGH-001], [EQ-DIST-ISF-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        return self.scale * math.sqrt(-2.0 * math.log(q))

    @property
    def mean(self) -> float:
        """Return ``s sqrt(pi/2)``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return self.scale * math.sqrt(0.5 * math.pi)

    @property
    def variance(self) -> float:
        """Return ``(4 - pi) s^2 / 2``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return 0.5 * (4.0 - math.pi) * self.scale * self.scale

    @property
    def skewness(self) -> float:
        """Return ``2 sqrt(pi) (pi - 3) / (4 - pi)^(3/2)``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return 2.0 * math.sqrt(math.pi) * (math.pi - 3.0) / math.pow(4.0 - math.pi, 1.5)

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``-(6 pi^2 - 24 pi + 16) / (4 - pi)^2``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        pi = math.pi
        return -(6.0 * pi * pi - 24.0 * pi + 16.0) / (4.0 - pi) ** 2

    @property
    def entropy(self) -> float:
        """Return ``1 + log(s / sqrt 2) + gamma_E / 2``.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-JKB-1994, Chapter 18].

        """
        return 1.0 + math.log(self.scale / math.sqrt(2.0)) + 0.5 * EULER_GAMMA

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw Rayleigh variates ``s sqrt(-2 log U)``.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-RAYLEIGH-001].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(self.isf(open_unit(generator)) for _ in range(count))


@dataclass(frozen=True, slots=True)
class Pareto(ContinuousDistribution):
    """The Pareto type I law ``S(x) = (x_m / x)^alpha`` on ``x >= x_m``.

    Moments: ``E[X] = alpha x_m/(alpha - 1)`` for ``alpha > 1`` (``inf``
    otherwise); the variance is ``inf`` for ``alpha <= 2`` (``E[X^2]``
    diverges on a positive support), the skewness ``inf`` for
    ``2 < alpha <= 3`` and the excess kurtosis ``inf`` for ``2 < alpha <= 4``;
    both are ``nan`` when the variance is infinite.

    Args:
        scale: Finite positive minimum ``x_m``.
        shape: Finite positive tail index ``alpha``.

    References:
        [REF-JKB-1994, Chapter 20].

    """

    scale: float = 1.0
    shape: float = 1.0

    def __post_init__(self) -> None:
        """Validate scale and shape.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or non-positive.

        References:
            [EQ-DIST-PARETO-001], [REF-JKB-1994, Chapter 20].

        """
        object.__setattr__(
            self, "scale", POSITIVE_FINITE.validate(self.scale, name="scale")
        )
        object.__setattr__(
            self, "shape", POSITIVE_FINITE.validate(self.shape, name="shape")
        )

    @property
    def support(self) -> RealSupport:
        """Return ``[x_m, inf)``.

        References:
            [REF-JKB-1994, Chapter 20].

        """
        return RealSupport(self.scale, None, lower_closed=True)

    def _log_ratio(self, x: float) -> float:
        """Return ``log(x / x_m)``, or ``-inf`` below the support.

        References:
            [EQ-DIST-PARETO-001]. No new equation.

        """
        point = require_finite_real(x, name="x")
        if point < self.scale:
            return -math.inf
        return math.log(point / self.scale)

    def log_pdf(self, x: float) -> float:
        """Return ``log(alpha / x_m) - (alpha + 1) log(x / x_m)``.

        Equation:
            [EQ-DIST-PARETO-001].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        ratio = self._log_ratio(x)
        if ratio == -math.inf:
            return -math.inf
        return math.log(self.shape / self.scale) - (self.shape + 1.0) * ratio

    def pdf(self, x: float) -> float:
        """Return the Pareto density.

        Equation:
            [EQ-DIST-PARETO-001].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        value = self.log_pdf(x)
        return math.exp(value) if value > -745.2 else 0.0

    def log_sf(self, x: float) -> float:
        """Return ``-alpha log(x / x_m)`` (``0`` below the support).

        Equation:
            [EQ-DIST-PARETO-001].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        ratio = self._log_ratio(x)
        return 0.0 if ratio == -math.inf else -self.shape * ratio

    def sf(self, x: float) -> float:
        """Return ``(x_m / x)^alpha``.

        Equation:
            [EQ-DIST-PARETO-001].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        return math.exp(self.log_sf(x))

    def cdf(self, x: float) -> float:
        """Return ``-expm1(-alpha log(x / x_m))``.

        Equation:
            [EQ-DIST-PARETO-001].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        return -math.expm1(self.log_sf(x))

    def quantile(self, probability: float) -> float:
        """Return ``x_m exp(-log(1 - p) / alpha)``.

        Equation:
            [EQ-DIST-PARETO-001].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        p = require_probability(probability, name="probability")
        if p == 1.0:
            return math.inf
        exponent = -math.log1p(-p) / self.shape
        return self.scale * math.exp(exponent) if exponent < 709.0 else math.inf

    def isf(self, probability: float) -> float:
        """Return ``x_m q^(-1/alpha)``.

        Equation:
            [EQ-DIST-PARETO-001], [EQ-DIST-ISF-001].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        exponent = -math.log(q) / self.shape
        return self.scale * math.exp(exponent) if exponent < 709.0 else math.inf

    @property
    def mean(self) -> float:
        """Return ``alpha x_m / (alpha - 1)`` or ``inf`` for ``alpha <= 1``.

        Equation:
            [EQ-DIST-PARETO-002].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        if self.shape <= 1.0:
            return math.inf
        return self.shape * self.scale / (self.shape - 1.0)

    @property
    def variance(self) -> float:
        """Return ``x_m^2 alpha / ((alpha - 1)^2 (alpha - 2))`` or ``inf``.

        Equation:
            [EQ-DIST-PARETO-002].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        a = self.shape
        if a <= 2.0:
            return math.inf
        return self.scale * self.scale * a / ((a - 1.0) ** 2 * (a - 2.0))

    @property
    def skewness(self) -> float:
        """Return the skewness ``2 (1 + a)/(a - 3) sqrt((a - 2)/a)`` for ``a > 3``.

        Equation:
            [EQ-DIST-PARETO-002].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        a = self.shape
        if a > 3.0:
            return 2.0 * (1.0 + a) / (a - 3.0) * math.sqrt((a - 2.0) / a)
        return math.inf if a > 2.0 else math.nan

    @property
    def kurtosis(self) -> float:
        """Return ``6 (a^3 + a^2 - 6 a - 2) / (a (a - 3)(a - 4))`` for ``a > 4``.

        Equation:
            [EQ-DIST-PARETO-002].

        References:
            [REF-JKB-1994, Chapter 20].

        """
        a = self.shape
        if a > 4.0:
            return 6.0 * (a**3 + a * a - 6.0 * a - 2.0) / (a * (a - 3.0) * (a - 4.0))
        return math.inf if a > 2.0 else math.nan

    @property
    def entropy(self) -> float:
        """Return ``log(x_m / alpha) + 1/alpha + 1``.

        Equation:
            [EQ-DIST-PARETO-002].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        return math.log(self.scale / self.shape) + 1.0 / self.shape + 1.0

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw Pareto variates ``x_m U^(-1/alpha)``.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-PARETO-001].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(self.isf(open_unit(generator)) for _ in range(count))
