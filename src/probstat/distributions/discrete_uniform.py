"""Discrete uniform law on a finite range of consecutive integers.

The support uses exact integer arithmetic.

References:
    [REF-JKB-2005, Chapter 6].

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

from ._base import DiscreteDistribution
from .support import IntegerSupport


@dataclass(frozen=True, slots=True)
class DiscreteUniform(DiscreteDistribution):
    """The discrete uniform law on the integers ``low, low + 1, ..., high``.

    Args:
        low: Integer minimum.
        high: Integer maximum with ``high >= low``.

    References:
        [REF-JKB-2005, Chapter 6].

    """

    low: int
    high: int

    def __post_init__(self) -> None:
        """Validate the integer bounds.

        Raises:
            TypeError: If a bound is not an integer.
            ValueError: If ``high < low``.

        References:
            [EQ-DIST-DUNIF-001], [REF-JKB-2005, Chapter 6].

        """
        for name in ("low", "high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.high < self.low:
            raise ValueError("high must not be below low")

    @property
    def count(self) -> int:
        """Return the number of support points ``n = high - low + 1``.

        References:
            [EQ-DIST-DUNIF-001]. No new equation.

        """
        return self.high - self.low + 1

    @property
    def support(self) -> IntegerSupport:
        """Return the integer support.

        References:
            [EQ-DIST-DUNIF-001]. No new equation.

        """
        return IntegerSupport(self.low, self.high)

    def log_pmf(self, k: int) -> float:
        """Return ``-log n`` on the support and ``-inf`` elsewhere.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if self.low <= k <= self.high:
            return -math.log(self.count)
        return -math.inf

    def pmf(self, k: int) -> float:
        """Return ``1/n`` on the support and ``0`` elsewhere.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return 0.0 if self.log_pmf(k) == -math.inf else 1.0 / self.count

    def cdf(self, x: float) -> float:
        """Return ``(floor(x) - low + 1)/n`` clipped to ``[0, 1]``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        point = math.floor(require_finite_real(x, name="x"))
        if point < self.low:
            return 0.0
        if point >= self.high:
            return 1.0
        return (point - self.low + 1) / self.count

    def sf(self, x: float) -> float:
        """Return ``(high - floor(x))/n`` clipped to ``[0, 1]``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        point = math.floor(require_finite_real(x, name="x"))
        if point < self.low:
            return 1.0
        if point >= self.high:
            return 0.0
        return (self.high - point) / self.count

    def quantile(self, probability: float) -> float:
        """Return the smallest ``k`` with ``F(k) >= p`` (integer-valued float).

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        p = require_probability(probability, name="probability")
        k = self.low + max(math.ceil(p * self.count) - 1, 0)
        while k > self.low and self.cdf(k - 1) >= p:
            k -= 1
        while self.cdf(k) < p:
            k += 1
        return float(k)

    @property
    def mean(self) -> float:
        """Return ``(low + high) / 2``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return 0.5 * (self.low + self.high)

    @property
    def variance(self) -> float:
        """Return ``(n^2 - 1) / 12``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return (self.count * self.count - 1) / 12.0

    @property
    def skewness(self) -> float:
        """Return ``0`` (``nan`` for a single point).

        References:
            [REF-JKB-2005, Chapter 6].

        """
        return 0.0 if self.count > 1 else math.nan

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``-6 (n^2 + 1) / (5 (n^2 - 1))``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-JKB-2005, Chapter 6].

        """
        n2 = self.count * self.count
        return -6.0 * (n2 + 1) / (5.0 * (n2 - 1)) if n2 > 1 else math.nan

    @property
    def entropy(self) -> float:
        """Return ``log n``.

        Equation:
            [EQ-DIST-DUNIF-001].

        References:
            [REF-COVER-THOMAS-2006, Chapter 2].

        """
        return math.log(self.count)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[int, ...]:
        """Draw uniform integers with ``randrange``.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of integers in ``[low, high]``.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        References:
            [REF-MATSUMOTO-NISHIMURA-1998]. No equation is evaluated.

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(generator.randrange(self.low, self.high + 1) for _ in range(count))
