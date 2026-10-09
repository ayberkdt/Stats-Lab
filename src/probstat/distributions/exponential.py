"""Exponential waiting-time distribution parameterized by rate."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from .constraints import POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class Exponential:
    """An exponential waiting-time law with positive rate ``rate``.

    Args:
        rate: Finite rate parameter ``lambda > 0``; the scale is ``1/rate``.

    References:
        [REF-ROSS-2019, Chapter 5].

    """

    rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize the positive rate.

        Raises:
            TypeError: If ``rate`` is not real.
            ValueError: If ``rate`` is non-finite or non-positive.

        References:
            [EQ-DIST-EXP-001], [REF-ROSS-2019, Chapter 5].

        """
        object.__setattr__(
            self, "rate", POSITIVE_FINITE.validate(self.rate, name="rate")
        )

    @property
    def support(self) -> RealSupport:
        """Return non-negative real support.

        References:
            [EQ-DIST-EXP-001], [REF-ROSS-2019, Chapter 5].

        """
        return RealSupport(0.0, None, lower_closed=True)

    def log_pdf(self, x: float) -> float:
        """Evaluate natural log exponential density at ``x``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density on support and negative infinity below zero.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-EXP-001].

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            return -math.inf
        return math.log(self.rate) - self.rate * point

    def pdf(self, x: float) -> float:
        """Evaluate exponential probability density at ``x``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density on non-negative support and zero below it.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-EXP-001].

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        log_density = self.log_pdf(x)
        return 0.0 if log_density == -math.inf else math.exp(log_density)

    def cdf(self, x: float) -> float:
        """Evaluate the exponential cumulative distribution stably.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``1-exp(-rate*x)`` on support and zero below it.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-EXP-002].

        References:
            [REF-ROSS-2019, Chapter 5], [REF-PYTHON-MATH].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            return 0.0
        return -math.expm1(-self.rate * point)

    def sf(self, x: float) -> float:
        """Evaluate the direct exponential survival function.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``exp(-rate*x)`` on support and one below it.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-EXP-002].

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            return 1.0
        return math.exp(-self.rate * point)

    def quantile(self, probability: float) -> float:
        """Return the closed-form generalized inverse CDF.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Non-negative waiting time, including positive infinity at one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-EXP-005].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        target = require_probability(probability, name="probability")
        if target == 1.0:
            return math.inf
        return -math.log1p(-target) / self.rate

    def hazard(self, x: float) -> float:
        """Return the exponential law's memoryless constant hazard.

        Args:
            x: Finite real evaluation point.

        Returns:
            Zero below support and ``rate`` on support.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-EXP-006].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 2].

        """
        point = require_finite_real(x, name="x")
        return 0.0 if point < 0.0 else self.rate

    @property
    def mean(self) -> float:
        """Return mean waiting time ``1/rate``.

        Equation:
            [EQ-DIST-EXP-003].

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        return 1.0 / self.rate

    @property
    def variance(self) -> float:
        """Return waiting-time variance ``1/rate^2``.

        Equation:
            [EQ-DIST-EXP-004].

        References:
            [REF-ROSS-2019, Chapter 5].

        """
        return 1.0 / (self.rate * self.rate)

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible observations by inverse transformation.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local generator.

        Returns:
            Tuple of non-negative waiting times.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter II]. Sampling inverts
            [EQ-DIST-EXP-005].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        return tuple(
            -math.log1p(-generator.random()) / self.rate
            for _ in range(checked_size)
        )
