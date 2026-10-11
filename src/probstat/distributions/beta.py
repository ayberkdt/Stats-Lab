"""Beta distribution on the unit interval with two positive shapes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._continuous import (
    LOGIT_LIMIT,
    expit,
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
class Beta:
    """A beta law with shapes ``alpha`` and ``beta`` on ``[0, 1]``.

    Density ``f(x) = x^(alpha-1) (1-x)^(beta-1) / B(alpha, beta)``.
    ``Beta(1, 1)`` is ``Uniform(0, 1)``.

    Args:
        alpha: Finite positive first shape.
        beta: Finite positive second shape.

    References:
        [REF-JKB-1995, Chapter 25], [REF-NIST-DISTRIBUTIONS, Beta
        Distribution].

    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        """Validate and normalize both shapes.

        Raises:
            TypeError: If a shape is not real.
            ValueError: If a shape is non-finite or non-positive.

        References:
            [EQ-DIST-BETA-001], [REF-JKB-1995, Chapter 25].

        """
        object.__setattr__(
            self, "alpha", POSITIVE_FINITE.validate(self.alpha, name="alpha")
        )
        object.__setattr__(
            self, "beta", POSITIVE_FINITE.validate(self.beta, name="beta")
        )

    @property
    def support(self) -> RealSupport:
        """Return the closed unit interval.

        References:
            [EQ-DIST-BETA-001], [REF-JKB-1995, Chapter 25].

        """
        return RealSupport(0.0, 1.0, lower_closed=True, upper_closed=True)

    def _log_beta_function(self) -> float:
        """Return ``log B(alpha, beta)``, via Stirling differences when large.

        References:
            [EQ-NUM-STIRLING-001], [REF-NIST-DLMF-GAMMA, Section 5.12].

        """
        return log_beta_function(self.alpha, self.beta)

    def log_pdf(self, x: float) -> float:
        """Evaluate the beta log-density, including endpoint limits.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density; ``-inf`` outside ``[0, 1]`` and possibly ``inf`` at an
            endpoint whose shape is below one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-BETA-001], [EQ-NUM-STIRLING-001].

        References:
            [REF-JKB-1995, Chapter 25], [REF-NIST-DLMF-GAMMA, Section 5.11].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0 or point > 1.0:
            return -math.inf
        normalizer = self._log_beta_function()
        if point == 0.0 or point == 1.0:
            shape = self.alpha if point == 0.0 else self.beta
            if shape < 1.0:
                return math.inf
            if shape > 1.0:
                return -math.inf
            return -normalizer
        if max(self.alpha, self.beta) >= LARGE_SHAPE:
            # alpha log x + beta log(1-x) - log B as one cancellation-free
            # kernel, then remove one power of x and of 1 - x.
            return (
                log_beta_kernel(self.alpha, self.beta, point, 1.0 - point)
                - math.log(point)
                - math.log1p(-point)
            )
        return (
            (self.alpha - 1.0) * math.log(point)
            + (self.beta - 1.0) * math.log1p(-point)
            - normalizer
        )

    def pdf(self, x: float) -> float:
        """Evaluate the beta probability density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value, possibly infinite at an endpoint.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-BETA-001].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        log_density = self.log_pdf(x)
        if log_density == math.inf:
            return math.inf
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate the beta CDF ``I_x(alpha, beta)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Regularized incomplete beta probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-BETA-002], [EQ-NUM-BETA-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 0.0
        if point >= 1.0:
            return 1.0
        return regularized_beta_pair(point, 1.0 - point, self.alpha, self.beta)

    def sf(self, x: float) -> float:
        """Evaluate the direct beta survival function ``I_(1-x)(beta, alpha)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting the CDF from one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-BETA-002], [EQ-NUM-BETA-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 1.0
        if point >= 1.0:
            return 0.0
        return regularized_beta_pair(1.0 - point, point, self.beta, self.alpha)

    def _invert(self, lower: float, upper: float) -> float:
        """Solve the tail equation on the logit coordinate.

        Args:
            lower: Target lower-tail probability in ``(0, 1)``.
            upper: Complementary target upper-tail probability.

        Returns:
            Quantile in ``[0, 1]``.

        Equation:
            [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-BRENT-1973, Chapter 4].

        """
        center = self.alpha / (self.alpha + self.beta)
        return solve_tail_quantile(
            lower,
            upper,
            log_cdf=lambda point: safe_log(self.cdf(point)),
            log_sf=lambda point: safe_log(self.sf(point)),
            to_x=expit,
            start=math.log(center) - math.log1p(-center),
            minimum=-LOGIT_LIMIT,
            maximum=LOGIT_LIMIT,
        )

    def quantile(self, probability: float) -> float:
        """Return the beta quantile by tail-aware numerical inversion.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Quantile in ``[0, 1]``.

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
            return 1.0
        return self._invert(target, 1.0 - target)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function ``x`` with ``S(x) = q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Quantile matched through ``log S(x) = log q``.

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
            return 1.0
        if q == 1.0:
            return 0.0
        return self._invert(1.0 - q, q)

    @property
    def mean(self) -> float:
        """Return ``E[X] = alpha / (alpha + beta)``.

        Equation:
            [EQ-DIST-BETA-003].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Return ``alpha beta / ((alpha+beta)^2 (alpha+beta+1))``.

        Equation:
            [EQ-DIST-BETA-003].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        total = self.alpha + self.beta
        return (self.alpha / total) * (self.beta / total) / (total + 1.0)

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible beta observations as gamma ratios.

        ``X = G_a / (G_a + G_b)`` with independent ``G_a ~ Gamma(alpha, 1)``
        and ``G_b ~ Gamma(beta, 1)``, combined in log space as
        ``expit(log G_a - log G_b)``.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            Tuple of observations in ``[0, 1]``.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-BETA-004], [EQ-DIST-GAM-005].

        References:
            [REF-DEVROYE-1986, Chapter IX], [REF-MARSAGLIA-TSANG-2000].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        observations: list[float] = []
        for _ in range(checked_size):
            first = log_standard_gamma_variate(self.alpha, generator)
            second = log_standard_gamma_variate(self.beta, generator)
            observations.append(expit(first - second))
        return tuple(observations)
