"""Univariate normal distribution with validated finite parameters."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_non_negative_integer,
)


@dataclass(frozen=True, slots=True)
class Normal:
    """A univariate normal distribution parameterized by mean and scale.

    Args:
        mu: Finite location parameter.
        sigma: Finite, strictly positive scale parameter.

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3].

    """

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize the location and scale parameters.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``sigma <= 0``.

        References:
            [EQ-DIST-NORM-001], [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        location = coerce_finite_sample((self.mu,), name="mu")[0]
        scale = coerce_finite_sample((self.sigma,), name="sigma")[0]
        if scale <= 0.0:
            raise ValueError("sigma must be strictly positive")
        object.__setattr__(self, "mu", location)
        object.__setattr__(self, "sigma", scale)

    def pdf(self, x: float) -> float:
        """Evaluate the normal probability density function at ``x``.

        Args:
            x: Finite real-valued evaluation point.

        Returns:
            Probability density at ``x``. A density is not a point probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is not finite.

        Equation:
            [EQ-DIST-NORM-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        point = coerce_finite_sample((x,), name="x")[0]
        z = (point - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2.0 * math.pi))

    def cdf(self, x: float) -> float:
        """Evaluate the normal cumulative distribution function at ``x``.

        Args:
            x: Finite real-valued evaluation point.

        Returns:
            ``P(X <= x)`` computed through the error function.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is not finite.

        Equation:
            [EQ-DIST-NORM-002].

        References:
            [REF-NIST-DLMF, Equation 7.2.1].

        """
        point = coerce_finite_sample((x,), name="x")[0]
        standardized = (point - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(standardized))

    @property
    def mean(self) -> float:
        """Return expected value ``E[X]``.

        Equation:
            [EQ-DIST-NORM-003].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        return self.mu

    @property
    def variance(self) -> float:
        """Return variance ``Var(X)``.

        Equation:
            [EQ-DIST-NORM-004].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        return self.sigma**2

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible pseudo-random normal observations.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            A tuple of ``size`` pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        References:
            [REF-DEVROYE-1986, Chapter IX]. The standard-library generator is
            treated as an implementation dependency, not a new equation.

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        return tuple(generator.gauss(self.mu, self.sigma) for _ in range(checked_size))

