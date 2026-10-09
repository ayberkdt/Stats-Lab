"""Binomial distribution with stable log-mass and finite-support operations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

from .constraints import NON_NEGATIVE_INTEGER, PROBABILITY
from .support import IntegerSupport


@dataclass(frozen=True, slots=True)
class Binomial:
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
        object.__setattr__(self, "p", PROBABILITY.validate(self.p, name="p"))

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
            math.lgamma(self.n + 1.0)
            - math.lgamma(k + 1.0)
            - math.lgamma(self.n - k + 1.0)
        )
        return (
            log_coefficient
            + k * math.log(self.p)
            + (self.n - k) * math.log1p(-self.p)
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

        The lower tail is summed directly below the mean; above it, the upper
        tail is summed and complemented. ``math.fsum`` reduces accumulation
        error while the branch avoids subtracting a near-one tail.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``P(X <= x)`` in ``[0, 1]``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-BIN-002].

        References:
            [REF-ROSS-2019, Section 4.3].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 0.0
        if cutoff >= self.n:
            return 1.0
        if cutoff < self.mean:
            result = math.fsum(self.pmf(k) for k in range(cutoff + 1))
        else:
            upper_tail = math.fsum(
                self.pmf(k) for k in range(cutoff + 1, self.n + 1)
            )
            result = 1.0 - upper_tail
        return max(0.0, min(1.0, result))

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

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[int, ...]:
        """Draw reproducible binomial observations by Bernoulli aggregation.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local generator.

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
        generator = random.Random(seed)
        return tuple(
            sum(generator.random() < self.p for _ in range(self.n))
            for _ in range(checked_size)
        )

