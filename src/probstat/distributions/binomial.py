"""Binomial distribution with stable log-mass and finite-support operations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._autodiff import admath
from ._base import DiscreteDistribution
from ._special import regularized_beta_pair
from .constraints import NON_NEGATIVE_INTEGER, PROBABILITY
from .support import IntegerSupport

# Largest trial count whose distribution function is a direct mass sum;
# above it the regularized incomplete beta function is used.
_DIRECT_SUM_MAX_TRIALS = 256


@dataclass(frozen=True, slots=True)
class Binomial(DiscreteDistribution):
    """The number of successes in ``n`` independent Bernoulli trials.

    Args:
        n: Non-negative integer trial count.
        p: Success probability in ``[0, 1]``.

    References:
        [REF-ROSS-2019, Section 4.3].

    """

    n: int
    p: float

    def __post_init__(self) -> None:
        """Validate and normalize trial count and success probability.

        Raises:
            TypeError: If ``n`` is not an integer or ``p`` is not real.
            ValueError: If ``n < 0`` or ``p`` is outside ``[0, 1]``.

        References:
            [EQ-DIST-BIN-001], [REF-ROSS-2019, Section 4.3].

        """
        object.__setattr__(self, "n", NON_NEGATIVE_INTEGER.validate(self.n, name="n"))
        object.__setattr__(self, "p", PROBABILITY.validate_parameter(self.p, name="p"))

    @property
    def support(self) -> IntegerSupport:
        """Return the finite integer support ``{0, ..., n}``.

        References:
            [EQ-DIST-BIN-001], [REF-ROSS-2019, Section 4.3].

        """
        return IntegerSupport(0, self.n)

    def log_pmf(self, k: int) -> float:
        """Evaluate the natural logarithm of binomial probability mass.

        Args:
            k: Candidate integer success count.

        Returns:
            ``log(P(X=k))`` or negative infinity outside the support.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-BIN-001].

        References:
            [REF-ROSS-2019, Section 4.3], [REF-NIST-DLMF-GAMMA, Section 5.2].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if not self.support.contains(k):
            return -math.inf
        if self.p == 0.0:
            return 0.0 if k == 0 else -math.inf
        if self.p == 1.0:
            return 0.0 if k == self.n else -math.inf
        log_coefficient = (
            admath.lgamma(self.n + 1.0)
            - admath.lgamma(k + 1.0)
            - admath.lgamma(self.n - k + 1.0)
        )
        return (
            log_coefficient
            + k * admath.log(self.p)
            + (self.n - k) * admath.log1p(-self.p)
        )

    def pmf(self, k: int) -> float:
        """Evaluate binomial probability mass at integer ``k``.

        Args:
            k: Candidate integer success count.

        Returns:
            ``P(X=k)``; zero outside ``{0, ..., n}``.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-BIN-001].

        References:
            [REF-ROSS-2019, Section 4.3].

        """
        log_mass = self.log_pmf(k)
        return 0.0 if log_mass == -math.inf else math.exp(log_mass)

    def cdf(self, x: float) -> float:
        """Evaluate the binomial cumulative distribution function.

        For ``n <= 256`` the lower tail is summed below the mean and the
        complementary upper tail above it (``math.fsum``, near-exact). For
        larger ``n`` the identity ``F(k) = I_(1-p)(n - k, k + 1)`` is used:
        the regularized incomplete beta function selects the smaller tail
        before any complement and needs a bounded number of
        continued-fraction terms instead of the ``O(n)`` terms of a sum.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``P(X <= x)`` in ``[0, 1]``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-BIN-002], [EQ-NUM-BETA-001].

        References:
            [REF-ROSS-2019, Section 4.3], [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 0.0
        if cutoff >= self.n:
            return 1.0
        if self.n <= _DIRECT_SUM_MAX_TRIALS:
            if cutoff < self.mean:
                result = math.fsum(self.pmf(k) for k in range(cutoff + 1))
            else:
                upper_tail = math.fsum(
                    self.pmf(k) for k in range(cutoff + 1, self.n + 1)
                )
                result = 1.0 - upper_tail
            return max(0.0, min(1.0, result))
        if self.p == 0.0:
            return 1.0
        if self.p == 1.0:
            return 0.0
        return regularized_beta_pair(
            1.0 - self.p, self.p, float(self.n - cutoff), cutoff + 1.0
        )

    def sf(self, x: float) -> float:
        """Evaluate the direct binomial survival probability ``P(X>x)``.

        The upper-tail masses are summed for ``n <= 256``; for larger ``n``
        ``S(k) = I_p(k + 1, n - k)`` is evaluated directly. Neither form
        subtracts the CDF from one, so small upper tails keep their relative
        accuracy.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without CDF subtraction.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-BIN-002], [EQ-NUM-BETA-001].

        References:
            [REF-ROSS-2019, Section 4.3], [REF-NIST-DLMF-BETA, Section 8.17].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 1.0
        if cutoff >= self.n:
            return 0.0
        if self.n <= _DIRECT_SUM_MAX_TRIALS:
            # Each mass carries the rounding of exp(log mass), so the sum can
            # exceed one by a few units in the last place; clamp it as the
            # CDF is clamped (an unclamped S(0) > 1 made isf(1) wrong).
            upper_tail = math.fsum(self.pmf(k) for k in range(cutoff + 1, self.n + 1))
            return max(0.0, min(1.0, upper_tail))
        if self.p == 0.0:
            return 0.0
        if self.p == 1.0:
            return 1.0
        return regularized_beta_pair(
            self.p, 1.0 - self.p, cutoff + 1.0, float(self.n - cutoff)
        )

    def quantile(self, probability: float) -> float:
        """Return the binomial generalized inverse CDF by binary search.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Smallest supported count whose CDF reaches the target; at
            probability one the largest count with positive mass (``n``, or
            ``0`` when ``p = 0``), even where the floating-point CDF rounds
            to one at a smaller count.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-BIN-005].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        target = require_probability(probability, name="probability")
        if target == 1.0:
            return 0.0 if self.p == 0.0 else float(self.n)
        lower = 0
        upper = self.n
        while lower < upper:
            midpoint = (lower + upper) // 2
            if self.cdf(midpoint) >= target:
                upper = midpoint
            else:
                lower = midpoint + 1
        return float(lower)

    @property
    def mean(self) -> float:
        """Return expected success count ``n p``.

        Equation:
            [EQ-DIST-BIN-003].

        References:
            [REF-ROSS-2019, Section 4.3].

        """
        return self.n * self.p

    @property
    def variance(self) -> float:
        """Return success-count variance ``n p (1-p)``.

        Equation:
            [EQ-DIST-BIN-004].

        References:
            [REF-ROSS-2019, Section 4.3].

        """
        return self.n * self.p * (1.0 - self.p)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[int, ...]:
        """Draw reproducible binomial observations by Bernoulli aggregation.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of success counts.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter X]. Sampling uses the defining Bernoulli-
            sum construction and introduces no new equation.

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(
            sum(generator.random() < self.p for _ in range(self.n))
            for _ in range(checked_size)
        )

