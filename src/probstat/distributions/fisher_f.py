"""Fisher-Snedecor F distribution with two positive degrees of freedom."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._continuous import (
    LOG_MAX,
    LOG_MIN,
    bounded_exp,
    log_standard_gamma_variate,
    safe_log,
    solve_tail_quantile,
)
from ._special import (
    LARGE_SHAPE,
    log_beta_function,
    log_beta_kernel,
    regularized_beta_pair,
)
from .constraints import POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class FisherF:
    """The F law of ``(V_1/nu1) / (V_2/nu2)`` for independent chi-squares.

    Args:
        nu1: Finite positive numerator degrees of freedom.
        nu2: Finite positive denominator degrees of freedom.

    References:
        [REF-JKB-1995, Chapter 27], [REF-NIST-DISTRIBUTIONS, F Distribution].

    """

    nu1: float
    nu2: float

    def __post_init__(self) -> None:
        """Validate both degrees of freedom.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or non-positive.

        References:
            [EQ-DIST-F-001], [REF-JKB-1995, Chapter 27].

        """
        object.__setattr__(self, "nu1", POSITIVE_FINITE.validate(self.nu1, name="nu1"))
        object.__setattr__(self, "nu2", POSITIVE_FINITE.validate(self.nu2, name="nu2"))

    @property
    def support(self) -> RealSupport:
        """Return non-negative real support.

        References:
            [EQ-DIST-F-001], [REF-JKB-1995, Chapter 27].

        """
        return RealSupport(0.0, None, lower_closed=True)

    def _ratios(self, point: float) -> tuple[float, float]:
        """Return ``y = nu1 x/(nu1 x + nu2)`` and ``1 - y`` without subtraction.

        Args:
            point: Validated positive point.

        Returns:
            The incomplete-beta argument and its complement.

        Equation:
            [EQ-DIST-F-002].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        numerator = self.nu1 * point
        if numerator == math.inf:
            return 1.0, 0.0
        denominator = numerator + self.nu2
        return numerator / denominator, self.nu2 / denominator

    def log_pdf(self, x: float) -> float:
        """Evaluate the F log-density, including its endpoint limit.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density; at zero it is ``inf`` for ``nu1 < 2``, ``0`` for
            ``nu1 = 2``, and ``-inf`` for ``nu1 > 2``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-F-001], [EQ-NUM-STIRLING-001].

        References:
            [REF-JKB-1995, Chapter 27], [REF-NIST-DLMF-GAMMA, Section 5.11].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            return -math.inf
        half1, half2 = 0.5 * self.nu1, 0.5 * self.nu2
        log_beta = log_beta_function(half1, half2)
        if point == 0.0:
            if self.nu1 < 2.0:
                return math.inf
            if self.nu1 > 2.0:
                return -math.inf
            return 0.0
        first, second = self._ratios(point)
        if second == 0.0:
            return -math.inf
        if max(half1, half2) >= LARGE_SHAPE and first > 0.0:
            # Exact offset y (a+b) - a = a b (x - 1) / (a x + b) keeps the
            # cancellation-free kernel independent of the rounding of y.
            a, b, value = Fraction(half1), Fraction(half2), Fraction(point)
            offset = a * b * (value - 1) / (a * value + b)
            return (
                log_beta_kernel(half1, half2, first, second, offset)
                - math.log(point)
            )
        # f(x) = y^(nu1/2) (1-y)^(nu2/2) / (x B(nu1/2, nu2/2)).
        return (
            half1 * math.log(first)
            + half2 * math.log(second)
            - math.log(point)
            - log_beta
        )

    def pdf(self, x: float) -> float:
        """Evaluate the F probability density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value, possibly infinite at zero.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-F-001].

        References:
            [REF-JKB-1995, Chapter 27].

        """
        log_density = self.log_pdf(x)
        if log_density == math.inf:
            return math.inf
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate the F CDF ``I_y(nu1/2, nu2/2)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-F-002], [EQ-NUM-BETA-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 0.0
        first, second = self._ratios(point)
        return regularized_beta_pair(first, second, 0.5 * self.nu1, 0.5 * self.nu2)

    def sf(self, x: float) -> float:
        """Evaluate the direct F survival function ``I_(1-y)(nu2/2, nu1/2)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting the CDF from one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-F-002], [EQ-NUM-BETA-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 1.0
        first, second = self._ratios(point)
        return regularized_beta_pair(second, first, 0.5 * self.nu2, 0.5 * self.nu1)

    def _invert(self, lower: float, upper: float) -> float:
        """Solve the tail equation on the log-coordinate ``t = log x``.

        Args:
            lower: Target lower-tail probability in ``(0, 1)``.
            upper: Complementary target upper-tail probability.

        Returns:
            Positive quantile.

        Equation:
            [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-BRENT-1973, Chapter 4].

        """
        return solve_tail_quantile(
            lower,
            upper,
            log_cdf=lambda point: safe_log(self.cdf(point)),
            log_sf=lambda point: safe_log(self.sf(point)),
            to_x=bounded_exp,
            start=0.0,
            minimum=LOG_MIN,
            maximum=LOG_MAX,
        )

    def quantile(self, probability: float) -> float:
        """Return the F quantile by tail-aware numerical inversion.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Non-negative quantile, ``inf`` at one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-BRENT-1973, Chapter 4].

        """
        target = require_probability(probability, name="probability")
        if target == 0.0:
            return 0.0
        if target == 1.0:
            return math.inf
        return self._invert(target, 1.0 - target)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function ``x`` with ``S(x) = q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Critical value of the F law.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-BRENT-1973, Chapter 4].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return 0.0
        return self._invert(1.0 - q, q)

    @property
    def mean(self) -> float:
        """Return ``E[X] = nu2 / (nu2 - 2)``.

        Returns:
            Mean, or ``inf`` when ``nu2 <= 2``.

        Equation:
            [EQ-DIST-F-003].

        References:
            [REF-JKB-1995, Chapter 27].

        """
        if self.nu2 <= 2.0:
            return math.inf
        return self.nu2 / (self.nu2 - 2.0)

    @property
    def variance(self) -> float:
        """Return ``2 nu2^2 (nu1+nu2-2) / (nu1 (nu2-2)^2 (nu2-4))``.

        Returns:
            Variance, or ``inf`` when ``nu2 <= 4``.

        Equation:
            [EQ-DIST-F-003].

        References:
            [REF-JKB-1995, Chapter 27].

        """
        if self.nu2 <= 4.0:
            return math.inf
        shifted = self.nu2 - 2.0
        return (
            2.0
            * self.nu2
            * self.nu2
            * (self.nu1 + self.nu2 - 2.0)
            / (self.nu1 * shifted * shifted * (self.nu2 - 4.0))
        )

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible F observations as scaled chi-square ratios.

        ``X = (G_1 / (nu1/2)) / (G_2 / (nu2/2))`` with independent
        ``G_i ~ Gamma(nu_i/2, 1)``, combined in log space.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            Tuple of non-negative pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-F-004], [EQ-DIST-GAM-005].

        References:
            [REF-DEVROYE-1986, Chapter IX], [REF-MARSAGLIA-TSANG-2000].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        half1, half2 = 0.5 * self.nu1, 0.5 * self.nu2
        offset = math.log(half2) - math.log(half1)
        observations: list[float] = []
        for _ in range(checked_size):
            logarithm = (
                log_standard_gamma_variate(half1, generator)
                - log_standard_gamma_variate(half2, generator)
                + offset
            )
            if logarithm > LOG_MAX:
                raise OverflowError("F sample exceeds float range")
            observations.append(math.exp(logarithm) if logarithm > -745.2 else 0.0)
        return tuple(observations)
