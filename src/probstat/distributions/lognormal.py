"""Lognormal distribution with explicit log-location parameterization."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class LogNormal:
    """A lognormal law parameterized by ``log(X) ~ Normal(mu, sigma)``.

    Args:
        mu: Finite mean of the natural logarithm of ``X``.
        sigma: Finite positive standard deviation of the natural logarithm.

    References:
        [REF-NIST-DISTRIBUTIONS, Lognormal Distribution].

    """

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize log-location and log-scale parameters.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``sigma <= 0``.

        References:
            [EQ-DIST-LN-001], [REF-NIST-DISTRIBUTIONS].

        """
        object.__setattr__(self, "mu", FINITE_REAL.validate(self.mu, name="mu"))
        object.__setattr__(
            self, "sigma", POSITIVE_FINITE.validate(self.sigma, name="sigma")
        )

    @property
    def support(self) -> RealSupport:
        """Return strictly positive real support.

        References:
            [EQ-DIST-LN-001], [REF-NIST-DISTRIBUTIONS].

        """
        return RealSupport(0.0, None, lower_closed=False)

    def log_pdf(self, x: float) -> float:
        """Evaluate the natural logarithm of the lognormal density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density on positive support and negative infinity otherwise.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LN-001].

        References:
            [REF-NIST-DISTRIBUTIONS, Lognormal Distribution].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return -math.inf
        logarithm = math.log(point)
        z = (logarithm - self.mu) / self.sigma
        return (
            -0.5 * z * z
            - math.log(point)
            - math.log(self.sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )

    def pdf(self, x: float) -> float:
        """Evaluate the lognormal probability density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density on positive support and zero otherwise.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LN-001].

        References:
            [REF-NIST-DISTRIBUTIONS, Lognormal Distribution].

        """
        log_density = self.log_pdf(x)
        return 0.0 if log_density == -math.inf else math.exp(log_density)

    def cdf(self, x: float) -> float:
        """Evaluate the lognormal cumulative distribution function.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``P(X <= x)`` with a stable error-function branch.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LN-002].

        References:
            [REF-NIST-DISTRIBUTIONS], [REF-NIST-DLMF, Equation 7.2.2].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 0.0
        z = (math.log(point) - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * math.erfc(-z)

    def sf(self, x: float) -> float:
        """Evaluate the direct lognormal survival function.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``P(X > x)`` without subtracting a near-one CDF.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-LN-002].

        References:
            [REF-NIST-DISTRIBUTIONS], [REF-NIST-DLMF, Equation 7.2.2].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 1.0
        z = (math.log(point) - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * math.erfc(z)

    def quantile(self, probability: float) -> float:
        """Return the lognormal generalized inverse CDF.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Positive quantile, with endpoints zero and positive infinity.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.
            OverflowError: If a finite mathematical quantile exceeds float range.

        Equation:
            [EQ-DIST-LN-003].

        References:
            [REF-NIST-DISTRIBUTIONS], [REF-PYTHON-STATISTICS].

        """
        target = require_probability(probability, name="probability")
        if target == 0.0:
            return 0.0
        if target == 1.0:
            return math.inf
        normal_quantile = NormalDist(mu=self.mu, sigma=self.sigma).inv_cdf(target)
        try:
            return math.exp(normal_quantile)
        except OverflowError as error:
            raise OverflowError("lognormal quantile exceeds float range") from error

    def isf(self, probability: float) -> float:
        """Return the inverse survival function ``x`` with ``S(x) = q``.

        Evaluated directly from ``q`` so that upper-tail quantiles keep full
        relative accuracy where ``1 - q`` would round to one.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            ``S^-1(q) = F^-1(1 - q)``.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001].

        References:
            [REF-NIST-DLMF, Section 7.17], [REF-PYTHON-STATISTICS].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return 0.0
        try:
            return math.exp(self.mu - self.sigma * NormalDist().inv_cdf(q))
        except OverflowError as error:
            raise OverflowError("lognormal quantile exceeds float range") from error

    @property
    def mean(self) -> float:
        """Return the lognormal expected value.

        Raises:
            OverflowError: If the mathematical mean exceeds float range.

        Equation:
            [EQ-DIST-LN-004].

        References:
            [REF-NIST-DISTRIBUTIONS, Lognormal Distribution].

        """
        try:
            result = math.exp(self.mu + 0.5 * self.sigma**2)
        except OverflowError as error:
            raise OverflowError("lognormal mean exceeds float range") from error
        if not math.isfinite(result):
            raise OverflowError("lognormal mean exceeds float range")
        return result

    @property
    def variance(self) -> float:
        """Return the lognormal variance.

        Raises:
            OverflowError: If the mathematical variance exceeds float range.

        Equation:
            [EQ-DIST-LN-004].

        References:
            [REF-NIST-DISTRIBUTIONS, Lognormal Distribution].

        """
        squared_scale = self.sigma**2
        try:
            result = math.expm1(squared_scale) * math.exp(
                2.0 * self.mu + squared_scale
            )
        except OverflowError as error:
            raise OverflowError("lognormal variance exceeds float range") from error
        if not math.isfinite(result):
            raise OverflowError("lognormal variance exceeds float range")
        return result

    def sample(self, size: int = 1, *, seed: int | None = None) -> tuple[float, ...]:
        """Draw reproducible pseudo-random lognormal observations.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Optional seed for a call-local random-number generator.

        Returns:
            Tuple of strictly positive pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.
            OverflowError: If a generated logarithm exceeds float range.

        Equation:
            [EQ-DIST-LN-003].

        References:
            [REF-DEVROYE-1986, Chapter IX].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = random.Random(seed)
        observations: list[float] = []
        for _ in range(checked_size):
            try:
                observation = math.exp(generator.gauss(self.mu, self.sigma))
            except OverflowError as error:
                raise OverflowError("lognormal sample exceeds float range") from error
            if observation == 0.0 or not math.isfinite(observation):
                raise OverflowError("lognormal sample is outside positive float range")
            observations.append(observation)
        return tuple(observations)
