"""Logistic distribution with location and positive scale."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._continuous import expit, open_unit
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class Logistic:
    """A logistic law with ``F(x) = 1 / (1 + exp(-(x - mu)/s))``.

    Args:
        mu: Finite location (mean, median, and mode).
        s: Finite positive scale; the standard deviation is ``s pi / sqrt 3``.

    References:
        [REF-JKB-1995, Chapter 23].

    """

    mu: float = 0.0
    s: float = 1.0

    def __post_init__(self) -> None:
        """Validate location and scale.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``s <= 0``.

        References:
            [EQ-DIST-LOGIS-001], [REF-JKB-1995, Chapter 23].

        """
        object.__setattr__(self, "mu", FINITE_REAL.validate(self.mu, name="mu"))
        object.__setattr__(self, "s", POSITIVE_FINITE.validate(self.s, name="s"))

    @property
    def support(self) -> RealSupport:
        """Return the entire real line.

        References:
            [EQ-DIST-LOGIS-001], [REF-JKB-1995, Chapter 23].

        """
        return RealSupport()

    def _standardized(self, x: float) -> float:
        """Validate ``x`` and return ``z = (x - mu) / s``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Standardized point.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        References:
            [EQ-DIST-LOGIS-001], [REF-JKB-1995, Chapter 23].

        """
        return (require_finite_real(x, name="x") - self.mu) / self.s

    def log_pdf(self, x: float) -> float:
        """Evaluate ``-|z| - log s - 2 log(1 + exp(-|z|))``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Natural log density, using the law's symmetry for stability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LOGIS-001].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        magnitude = abs(self._standardized(x))
        return -magnitude - math.log(self.s) - 2.0 * math.log1p(math.exp(-magnitude))

    def pdf(self, x: float) -> float:
        """Evaluate the logistic density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LOGIS-001].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        log_density = self.log_pdf(x)
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate the logistic CDF ``expit(z)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower-tail probability with relative accuracy for ``z < 0``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LOGIS-002].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        return expit(max(self._standardized(x), -800.0))

    def sf(self, x: float) -> float:
        """Evaluate the direct logistic survival function ``expit(-z)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability with relative accuracy for ``z > 0``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LOGIS-002].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        return expit(max(-self._standardized(x), -800.0))

    def quantile(self, probability: float) -> float:
        """Return ``mu + s (log p - log(1 - p))``.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Real quantile, with infinite endpoints at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-LOGIS-003].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        return self.mu + self.s * (math.log(p) - math.log1p(-p))

    def isf(self, probability: float) -> float:
        """Return ``mu + s (log(1 - q) - log q)`` without forming ``1 - q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Real quantile, with infinite endpoints at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-LOGIS-003].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return -math.inf
        return self.mu + self.s * (math.log1p(-q) - math.log(q))

    @property
    def mean(self) -> float:
        """Return ``E[X] = mu``.

        Equation:
            [EQ-DIST-LOGIS-004].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        return self.mu

    @property
    def variance(self) -> float:
        """Return ``Var(X) = s^2 pi^2 / 3``.

        Equation:
            [EQ-DIST-LOGIS-004].

        References:
            [REF-JKB-1995, Chapter 23].

        """
        return self.s * self.s * math.pi * math.pi / 3.0

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible logistic observations by inverse transform.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-LOGIS-003].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        observations: list[float] = []
        for _ in range(checked_size):
            u = open_unit(generator)
            observations.append(self.mu + self.s * (math.log(u) - math.log1p(-u)))
        return tuple(observations)
