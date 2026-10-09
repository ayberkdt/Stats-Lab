"""Continuous uniform distribution on a finite closed interval."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from .constraints import FINITE_REAL
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class Uniform:
    """A continuous uniform law on ``[lower, upper]``.

    Args:
        lower: Finite lower endpoint.
        upper: Finite upper endpoint strictly greater than ``lower``.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """

    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize interval endpoints.

        Raises:
            TypeError: If an endpoint is not real.
            ValueError: If an endpoint is non-finite or ``upper <= lower``.

        References:
            [EQ-DIST-UNIF-001], [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        lower = FINITE_REAL.validate(self.lower, name="lower")
        upper = FINITE_REAL.validate(self.upper, name="upper")
        if upper <= lower:
            raise ValueError("upper must be strictly greater than lower")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def support(self) -> RealSupport:
        """Return the closed finite interval support.

        References:
            [EQ-DIST-UNIF-001], [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        return RealSupport(
            self.lower, self.upper, lower_closed=True, upper_closed=True
        )

    @property
    def width(self) -> float:
        """Return interval width ``upper-lower``.

        References:
            [EQ-DIST-UNIF-001], [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        return self.upper - self.lower

    def log_pdf(self, x: float) -> float:
        """Evaluate natural log density at ``x``.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``-log(width)`` on support and negative infinity elsewhere.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-UNIF-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        point = require_finite_real(x, name="x")
        if point < self.lower or point > self.upper:
            return -math.inf
        return -math.log(self.width)

    def pdf(self, x: float) -> float:
        """Evaluate the continuous uniform density at ``x``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Constant density on support and zero outside it.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-UNIF-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        log_density = self.log_pdf(x)
        return 0.0 if log_density == -math.inf else math.exp(log_density)

    def cdf(self, x: float) -> float:
        """Evaluate the continuous uniform cumulative distribution.

        Args:
            x: Finite real evaluation point.

        Returns:
            Accumulated probability through ``x``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-UNIF-002].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        point = require_finite_real(x, name="x")
        if point <= self.lower:
            return 0.0
        if point >= self.upper:
            return 1.0
        return (point - self.lower) / self.width

    def sf(self, x: float) -> float:
        """Evaluate the direct survival probability ``P(X>x)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without CDF subtraction.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-UNIF-002].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        point = require_finite_real(x, name="x")
        if point < self.lower:
            return 1.0
        if point >= self.upper:
            return 0.0
        return (self.upper - point) / self.width

    def quantile(self, probability: float) -> float:
        """Return the closed-form generalized inverse CDF.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Corresponding point in the closed support interval.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-UNIF-005].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        target = require_probability(probability, name="probability")
        return self.lower + target * self.width

    def hazard(self, x: float) -> float:
        """Return uniform hazard where survival probability is positive.

        Args:
            x: Finite real evaluation point below ``upper``.

        Returns:
            Zero below support or ``1/(upper-x)`` inside support.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite or survival is zero.

        Equation:
            [EQ-DIST-UNIF-006].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 2].

        """
        point = require_finite_real(x, name="x")
        if point < self.lower:
            return 0.0
        if point >= self.upper:
            raise ValueError("hazard requires positive survival probability")
        return 1.0 / (self.upper - point)

    @property
    def mean(self) -> float:
        """Return midpoint expectation ``(lower+upper)/2``.

        Equation:
            [EQ-DIST-UNIF-003].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        return 0.5 * (self.lower + self.upper)

    @property
    def variance(self) -> float:
        """Return variance ``width^2/12``.

        Equation:
            [EQ-DIST-UNIF-004].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        return self.width * self.width / 12.0

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible continuous uniform observations.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local generator.

        Returns:
            Tuple of observations in the support interval.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter II]. Sampling inverts
            [EQ-DIST-UNIF-005].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        return tuple(
            self.lower + generator.random() * self.width
            for _ in range(checked_size)
        )
