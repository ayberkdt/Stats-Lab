"""Geometric distribution for failures before the first success."""

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
from .constraints import POSITIVE_FINITE
from .support import IntegerSupport


@dataclass(frozen=True, slots=True)
class Geometric(DiscreteDistribution):
    """Failures before the first success in independent Bernoulli trials.

    This zero-based parameterization has support ``{0, 1, ...}``. It is not the
    alternate convention that counts the successful trial itself.

    Args:
        p: Success probability satisfying ``0 < p <= 1``.

    References:
        [REF-ROSS-2019, Section 4.5].

    """

    p: float

    def __post_init__(self) -> None:
        """Validate and normalize the success probability.

        Raises:
            TypeError: If ``p`` is not real.
            ValueError: If ``p`` is non-finite or outside ``(0, 1]``.

        References:
            [EQ-DIST-GEOM-001], [REF-ROSS-2019, Section 4.5].

        """
        checked = POSITIVE_FINITE.validate(self.p, name="p")
        if checked > 1.0:
            raise ValueError("p must not exceed one")
        object.__setattr__(self, "p", checked)

    @property
    def support(self) -> IntegerSupport:
        """Return unbounded non-negative integer support.

        References:
            [EQ-DIST-GEOM-001], [REF-ROSS-2019, Section 4.5].

        """
        return IntegerSupport(0, None)

    def log_pmf(self, k: int) -> float:
        """Evaluate natural log mass at failure count ``k``.

        Args:
            k: Candidate failure count.

        Returns:
            Log mass, or negative infinity outside the support.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-GEOM-001].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 0:
            return -math.inf
        if self.p == 1.0:
            return 0.0 if k == 0 else -math.inf
        return math.log(self.p) + k * math.log1p(-self.p)

    def pmf(self, k: int) -> float:
        """Evaluate probability of exactly ``k`` failures before success.

        Args:
            k: Candidate failure count.

        Returns:
            ``P(X=k)``; zero outside the support.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-GEOM-001].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        log_mass = self.log_pmf(k)
        return 0.0 if log_mass == -math.inf else math.exp(log_mass)

    def cdf(self, x: float) -> float:
        """Evaluate probability of at most ``floor(x)`` failures.

        Args:
            x: Finite real evaluation point.

        Returns:
            Cumulative probability in ``[0, 1]``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GEOM-002].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 0.0
        if self.p == 1.0:
            return 1.0
        if cutoff == 0:
            return self.p
        return -math.expm1((cutoff + 1) * math.log1p(-self.p))

    def sf(self, x: float) -> float:
        """Evaluate the direct survival probability ``P(X > x)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting a near-one CDF.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GEOM-002].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 1.0
        if self.p == 1.0:
            return 0.0
        return math.exp((cutoff + 1) * math.log1p(-self.p))

    def quantile(self, probability: float) -> float:
        """Return the generalized inverse CDF at ``probability``.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Smallest supported count whose CDF reaches ``probability``;
            positive infinity at probability one for a non-degenerate law.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-GEOM-005].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        target = require_probability(probability, name="probability")
        if target == 0.0 or self.p == 1.0:
            return 0.0
        if target == 1.0:
            return math.inf
        estimate = math.ceil(math.log1p(-target) / math.log1p(-self.p) - 1.0)
        return float(max(0, estimate))

    @property
    def mean(self) -> float:
        """Return expected failures ``(1-p)/p``.

        Equation:
            [EQ-DIST-GEOM-003].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        return (1.0 - self.p) / self.p

    @property
    def variance(self) -> float:
        """Return failure-count variance ``(1-p)/p^2``.

        Equation:
            [EQ-DIST-GEOM-004].

        References:
            [REF-ROSS-2019, Section 4.5].

        """
        return (1.0 - self.p) / (self.p * self.p)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[int, ...]:
        """Draw reproducible failure counts by inverse transformation.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of non-negative failure counts.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter III]. Sampling inverts
            [EQ-DIST-GEOM-002].

        """
        checked_size = require_non_negative_integer(size, name="size")
        if self.p == 1.0:
            return (0,) * checked_size
        generator = resolve_stream(seed).generator()
        log_failure = math.log1p(-self.p)
        return tuple(
            math.floor(math.log1p(-generator.random()) / log_failure)
            for _ in range(checked_size)
        )
