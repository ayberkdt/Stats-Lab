"""Pareto (type I) law in closed form.

The distribution function, quantile, and inverse survival function have
elementary closed forms written so that the upper tail keeps relative
accuracy. Sampling is by inversion of an open-interval uniform from the
call-local stream.

References:
    [REF-JKB-1994, Chapter 20].

"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)

from ._autodiff import admath, real_argument
from ._base import ContinuousDistribution
from ._continuous import open_unit
from .constraints import POSITIVE_FINITE
from .support import RealSupport


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
            self, "scale", POSITIVE_FINITE.validate_parameter(self.scale, name="scale")
        )
        object.__setattr__(
            self, "shape", POSITIVE_FINITE.validate_parameter(self.shape, name="shape")
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
        point = real_argument(x, name="x")
        if point < self.scale:
            return -math.inf
        return admath.log(point / self.scale)

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
        return admath.log(self.shape / self.scale) - (self.shape + 1.0) * ratio

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
