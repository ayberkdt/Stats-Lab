"""Rayleigh law on the non-negative half-line in closed form.

The distribution function, quantile, and inverse survival function have
elementary closed forms written so that the tails keep relative accuracy.
Sampling is by inversion of an open-interval uniform from the call-local
stream.

References:
    [REF-JKB-1994, Chapter 18].

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
from probstat.numerics.special import EULER_GAMMA

from ._base import ContinuousDistribution
from ._continuous import density_from_log, open_unit
from .constraints import POSITIVE_FINITE
from .support import RealSupport


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
        return density_from_log(value)

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
