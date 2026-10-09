"""Poisson distribution with stable log mass, tails, and exact sampling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._special import regularized_gamma_p, regularized_gamma_q
from .constraints import NON_NEGATIVE_FINITE
from .support import IntegerSupport


@dataclass(frozen=True, slots=True)
class Poisson:
    """A Poisson count distribution with finite non-negative rate ``lam``.

    ``lam=0`` is supported as the degenerate distribution concentrated at zero.

    Args:
        lam: Expected non-negative event count.

    References:
        [REF-ROSS-2019, Section 4.8].

    """

    lam: float

    def __post_init__(self) -> None:
        """Validate and normalize the rate parameter.

        Raises:
            TypeError: If ``lam`` is not real.
            ValueError: If ``lam`` is negative or non-finite.

        References:
            [EQ-DIST-POIS-001], [REF-ROSS-2019, Section 4.8].

        """
        object.__setattr__(
            self, "lam", NON_NEGATIVE_FINITE.validate(self.lam, name="lam")
        )

    @property
    def support(self) -> IntegerSupport:
        """Return unbounded non-negative integer support.

        References:
            [EQ-DIST-POIS-001], [REF-ROSS-2019, Section 4.8].

        """
        return IntegerSupport(0, None)

    def log_pmf(self, k: int) -> float:
        """Evaluate natural log Poisson mass at integer ``k``.

        Args:
            k: Candidate event count.

        Returns:
            Log mass, or negative infinity outside the support.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-POIS-001].

        References:
            [REF-ROSS-2019, Section 4.8], [REF-NIST-DLMF-GAMMA, Section 5.2].

        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 0:
            return -math.inf
        if self.lam == 0.0:
            return 0.0 if k == 0 else -math.inf
        return k * math.log(self.lam) - self.lam - math.lgamma(k + 1.0)

    def pmf(self, k: int) -> float:
        """Evaluate Poisson probability mass at integer ``k``.

        Args:
            k: Candidate event count.

        Returns:
            ``P(X=k)``; zero outside the support.

        Raises:
            TypeError: If ``k`` is not an integer.

        Equation:
            [EQ-DIST-POIS-001].

        References:
            [REF-ROSS-2019, Section 4.8].

        """
        log_mass = self.log_pmf(k)
        return 0.0 if log_mass == -math.inf else math.exp(log_mass)

    def cdf(self, x: float) -> float:
        """Evaluate ``P(X <= x)`` through a regularized gamma tail.

        Args:
            x: Finite real evaluation point.

        Returns:
            Cumulative probability in ``[0, 1]``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.
            ArithmeticError: If the gamma expansion does not converge.

        Equation:
            [EQ-DIST-POIS-002].

        References:
            [REF-NIST-DLMF-GAMMA, Equation 8.4.10].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 0.0
        if self.lam == 0.0:
            return 1.0
        return regularized_gamma_q(cutoff + 1.0, self.lam)

    def sf(self, x: float) -> float:
        """Evaluate the stable survival probability ``P(X > x)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting a near-one CDF.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.
            ArithmeticError: If the gamma expansion does not converge.

        Equation:
            [EQ-DIST-POIS-002].

        References:
            [REF-NIST-DLMF-GAMMA, Equation 8.4.10].

        """
        point = require_finite_real(x, name="x")
        cutoff = math.floor(point)
        if cutoff < 0:
            return 1.0
        if self.lam == 0.0:
            return 0.0
        return regularized_gamma_p(cutoff + 1.0, self.lam)

    def quantile(self, probability: float) -> float:
        """Return the Poisson generalized inverse CDF by adaptive bracketing.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Smallest supported count whose CDF reaches the target; positive
            infinity at probability one for a positive-rate law.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.
            ArithmeticError: If a gamma expansion does not converge.

        Equation:
            [EQ-DIST-POIS-005].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 2].

        """
        target = require_probability(probability, name="probability")
        if target == 0.0 or self.lam == 0.0:
            return 0.0
        if target == 1.0:
            return math.inf
        lower = -1
        upper = max(1, math.ceil(self.lam))
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
        """Return expected count ``lam``.

        Equation:
            [EQ-DIST-POIS-003].

        References:
            [REF-ROSS-2019, Section 4.8].

        """
        return self.lam

    @property
    def variance(self) -> float:
        """Return count variance ``lam``.

        Equation:
            [EQ-DIST-POIS-004].

        References:
            [REF-ROSS-2019, Section 4.8].

        """
        return self.lam

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[int, ...]:
        """Draw reproducible exact Poisson observations.

        Knuth's product method is used below rate 10; transformed rejection is
        used at larger rates to avoid linear work and exponential underflow.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local generator.

        Returns:
            Tuple of non-negative event counts.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter X], [REF-HORMANN-1993]. Sampling
            introduces no new probability equation.

        """
        checked_size = require_non_negative_integer(size, name="size")
        if self.lam == 0.0:
            return (0,) * checked_size
        generator = random.Random(seed)
        sampler = _sample_knuth if self.lam < 10.0 else _sample_ptrs
        return tuple(sampler(self.lam, generator) for _ in range(checked_size))


def _sample_knuth(lam: float, generator: random.Random) -> int:
    """Draw one low-rate Poisson count with Knuth's product method.

    Args:
        lam: Positive rate below 10.
        generator: Call-local pseudo-random generator.

    Returns:
        One exact Poisson variate.

    References:
        [REF-DEVROYE-1986, Chapter X]. No new probability equation is defined.

    """
    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= generator.random()
    return count - 1


def _sample_ptrs(lam: float, generator: random.Random) -> int:
    """Draw one high-rate Poisson count by transformed rejection.

    Args:
        lam: Rate at least 10.
        generator: Call-local pseudo-random generator.

    Returns:
        One exact Poisson variate.

    References:
        [REF-HORMANN-1993]. No new probability equation is defined.

    """
    root = math.sqrt(lam)
    b = 0.931 + 2.53 * root
    a = -0.059 + 0.02483 * b
    inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
    squeeze = 0.9277 - 3.6224 / (b - 2.0)
    log_rate = math.log(lam)
    while True:
        centered_uniform = generator.random() - 0.5
        vertical_uniform = generator.random()
        distance = 0.5 - abs(centered_uniform)
        if distance <= 0.0:
            continue
        candidate = math.floor(
            (2.0 * a / distance + b) * centered_uniform + lam + 0.43
        )
        if distance >= 0.07 and vertical_uniform <= squeeze:
            return candidate
        if candidate < 0 or (distance < 0.013 and vertical_uniform > distance):
            continue
        if vertical_uniform == 0.0:
            return candidate
        log_acceptance = math.log(
            vertical_uniform
            * inverse_alpha
            / (a / (distance * distance) + b)
        )
        log_mass_without_constant = (
            -lam + candidate * log_rate - math.lgamma(candidate + 1.0)
        )
        if log_acceptance <= log_mass_without_constant:
            return candidate

