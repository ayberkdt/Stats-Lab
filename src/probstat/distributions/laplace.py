"""Laplace (double exponential) law in closed form.

The distribution function, quantile, and inverse survival function have
elementary closed forms written so that each tail keeps relative accuracy.
Sampling is by inversion of an open-interval uniform from the call-local
stream.

References:
    [REF-JKB-1995, Chapter 24].

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
from ._continuous import density_from_log, open_unit
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport


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
        return density_from_log(self.log_pdf(x))

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
