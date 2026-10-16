"""Negative binomial distribution for failures before a target success."""

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
from ._special import regularized_beta
from .constraints import POSITIVE_FINITE, POSITIVE_INTEGER
from .support import IntegerSupport


@dataclass(frozen=True, slots=True)
class NegativeBinomial(DiscreteDistribution):
    """Failures before the ``r``-th success in independent Bernoulli trials.

    Args:
        r: Positive integer target number of successes.
        p: Common success probability satisfying ``0 < p <= 1``.

    References:
        [REF-ROSS-2019, Section 4.5].

    """

    r: int
    p: float

    def __post_init__(self) -> None:
        """Validate and normalize the success target and probability.

        Raises:
            TypeError: If ``r`` is not an integer or ``p`` is not real.
            ValueError: If ``r < 1`` or ``p`` is outside ``(0, 1]``.

        References:
            [EQ-DIST-NBIN-001], [REF-ROSS-2019, Section 4.5].

        """
        target = POSITIVE_INTEGER.validate(self.r, name="r")
        probability = POSITIVE_FINITE.validate(self.p, name="p")
        if probability > 1.0:
            raise ValueError("p must not exceed one")
        object.__setattr__(self, "r", target)
        object.__setattr__(self, "p", probability)

    @property
    def support(self) -> IntegerSupport:
        """Return unbounded non-negative integer support.

        References:
            [EQ-DIST-NBIN-001], [REF-ROSS-2019, Section 4.5].

        """
        return IntegerSupport(0, None)

    def log_pmf(self, k: int) -> float:
        """Evaluate natural log mass at failure count ``k``.

        Args:
            k: Candidate failure count before the ``r``-th success.

        Returns:
            Log mass, or negative infinity outside the support.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-NBIN-001].

        References:
            [REF-ROSS-2019, Section 4.5], [REF-NIST-DLMF-GAMMA, Section 5.2].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 0:
            return -math.inf
        if self.p == 1.0:
            return 0.0 if k == 0 else -math.inf
        log_coefficient = (
            math.lgamma(k + self.r)
            - math.lgamma(k + 1.0)
            - math.lgamma(self.r)
        )
        return (
            log_coefficient
            + self.r * math.log(self.p)
            + k * math.log1p(-self.p)
        )

    def pmf(self, k: int) -> float:
        """Evaluate probability of ``k`` failures before the ``r``-th success.

        Args:
            k: Candidate failure count.

        Returns:
            Point mass, or zero outside the support.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-NBIN-001].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        log_mass = self.log_pmf(k)
        return 0.0 if log_mass == -math.inf else math.exp(log_mass)

    def cdf(self, x: float) -> float:
        """Evaluate cumulative probability through ``floor(x)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Cumulative probability evaluated by a regularized beta function.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.
            ArithmeticError: If the beta continued fraction does not converge.

        Equation:
            [EQ-DIST-NBIN-002].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 0.0
        return regularized_beta(self.p, float(self.r), cutoff + 1.0)

    def sf(self, x: float) -> float:
        """Evaluate the direct survival probability ``P(X > x)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability evaluated without CDF subtraction.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.
            ArithmeticError: If the beta continued fraction does not converge.

        Equation:
            [EQ-DIST-NBIN-002].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 1.0
        return regularized_beta(1.0 - self.p, cutoff + 1.0, float(self.r))

    def quantile(self, probability: float) -> float:
        """Return the generalized inverse CDF by bracketing and bisection.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Smallest supported count whose CDF reaches the target; positive
            infinity at probability one for a non-degenerate law.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.
            ArithmeticError: If beta evaluation fails to converge.

        Equation:
            [EQ-DIST-NBIN-005].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        target = require_probability(probability, name="probability")
        if target == 0.0 or self.p == 1.0:
            return 0.0
        if target == 1.0:
            return math.inf
        lower = -1
        upper = max(1, math.ceil(self.mean))
        while self.cdf(upper) < target:
            lower = upper
            upper *= 2
        while upper - lower > 1:
            midpoint = (lower + upper) // 2
            if self.cdf(midpoint) >= target:
                upper = midpoint
            else:
                lower = midpoint
        return float(upper)

    @property
    def mean(self) -> float:
        """Return expected failures ``r(1-p)/p``.

        Equation:
            [EQ-DIST-NBIN-003].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        return self.r * (1.0 - self.p) / self.p

    @property
    def variance(self) -> float:
        """Return failure-count variance ``r(1-p)/p^2``.

        Equation:
            [EQ-DIST-NBIN-004].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        return self.r * (1.0 - self.p) / (self.p * self.p)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[int, ...]:
        """Draw reproducible counts as sums of geometric waiting counts.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of failures before the ``r``-th success.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter III]. Sampling uses the independent-sum
            construction implied by [EQ-DIST-NBIN-001].

        """
        checked_size = require_non_negative_integer(size, name="size")
        if self.p == 1.0:
            return (0,) * checked_size
        generator = resolve_stream(seed).generator()
        log_failure = math.log1p(-self.p)
        return tuple(
            sum(
                math.floor(math.log1p(-generator.random()) / log_failure)
                for _ in range(self.r)
            )
            for _ in range(checked_size)
        )
