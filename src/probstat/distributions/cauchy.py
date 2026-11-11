"""Cauchy law with a tail-accurate closed-form distribution function.

The distribution function, quantile, and inverse survival function have
elementary closed forms written so that the tails keep relative accuracy
(``F(x) = atan2(1, -z)/pi`` instead of ``1/2 + atan(z)/pi``, which cancels
for ``z << 0``). Sampling is by inversion of an open-interval uniform from
the call-local stream.

References:
    [REF-JKB-1994, Chapter 16].

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
        return density_from_log(self.log_pdf(x))

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
