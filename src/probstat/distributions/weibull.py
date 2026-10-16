"""Two-parameter Weibull distribution with positive shape and scale."""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._base import ContinuousDistribution
from .constraints import POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class Weibull(ContinuousDistribution):
    """A Weibull law with shape ``k`` and scale ``scale``.

    Args:
        shape: Finite positive shape parameter ``k``.
        scale: Finite positive scale parameter ``lambda``.

    References:
        [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

    """

    shape: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize shape and scale.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or non-positive.

        References:
            [EQ-DIST-WEIB-001], [REF-NIST-DISTRIBUTIONS].

        """
        object.__setattr__(
            self, "shape", POSITIVE_FINITE.validate(self.shape, name="shape")
        )
        object.__setattr__(
            self, "scale", POSITIVE_FINITE.validate(self.scale, name="scale")
        )

    @property
    def support(self) -> RealSupport:
        """Return non-negative real support.

        References:
            [EQ-DIST-WEIB-001], [REF-NIST-DISTRIBUTIONS].

        """
        return RealSupport(0.0, None, lower_closed=True)

    def _powered_ratio(self, point: float) -> float:
        """Return ``(point/scale)^shape`` with overflow mapped to infinity.

        Args:
            point: A validated positive point.

        Returns:
            Non-negative power or positive infinity on floating-point overflow.

        References:
            [EQ-DIST-WEIB-001], [REF-PYTHON-MATH].

        """
        try:
            return math.pow(point / self.scale, self.shape)
        except OverflowError:
            return math.inf

    def log_pdf(self, x: float) -> float:
        """Evaluate the Weibull log-density, including its endpoint limit.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density; the value at zero may be infinite when ``shape < 1``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-WEIB-001].

        References:
            [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            return -math.inf
        if point == 0.0:
            if self.shape < 1.0:
                return math.inf
            if self.shape == 1.0:
                return -math.log(self.scale)
            return -math.inf
        powered = self._powered_ratio(point)
        return (
            math.log(self.shape)
            - math.log(self.scale)
            + (self.shape - 1.0) * math.log(point / self.scale)
            - powered
        )

    def pdf(self, x: float) -> float:
        """Evaluate the Weibull probability density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density on non-negative support and zero below support.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-WEIB-001].

        References:
            [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

        """
        log_density = self.log_pdf(x)
        if log_density == -math.inf:
            return 0.0
        if log_density == math.inf:
            return math.inf
        return math.exp(log_density)

    def cdf(self, x: float) -> float:
        """Evaluate the Weibull CDF with stable near-zero subtraction.

        Args:
            x: Finite real evaluation point.

        Returns:
            Cumulative probability at ``x``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-WEIB-002].

        References:
            [REF-NIST-DISTRIBUTIONS], [REF-PYTHON-MATH].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 0.0
        powered = self._powered_ratio(point)
        return 1.0 if powered == math.inf else -math.expm1(-powered)

    def sf(self, x: float) -> float:
        """Evaluate the direct Weibull survival function.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting the CDF from one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-WEIB-002].

        References:
            [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 1.0
        return math.exp(-self._powered_ratio(point))

    def quantile(self, probability: float) -> float:
        """Return the Weibull generalized inverse CDF.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Non-negative quantile, with positive infinity at one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.
            OverflowError: If a finite mathematical quantile exceeds float range.

        Equation:
            [EQ-DIST-WEIB-003].

        References:
            [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

        """
        target = require_probability(probability, name="probability")
        if target == 1.0:
            return math.inf
        powered = math.pow(-math.log1p(-target), 1.0 / self.shape)
        result = self.scale * powered
        if not math.isfinite(result):
            raise OverflowError("Weibull quantile exceeds float range")
        return result

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
        result = float(self.scale * (-math.log(q)) ** (1.0 / self.shape))
        if not math.isfinite(result):
            raise OverflowError("Weibull quantile exceeds float range")
        return result

    @property
    def mean(self) -> float:
        """Return the Weibull expected value.

        Raises:
            OverflowError: If the mathematical mean exceeds float range.

        Equation:
            [EQ-DIST-WEIB-004].

        References:
            [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

        """
        try:
            result = self.scale * math.gamma(1.0 + 1.0 / self.shape)
        except (OverflowError, ValueError) as error:
            raise OverflowError("Weibull mean exceeds float range") from error
        if not math.isfinite(result):
            raise OverflowError("Weibull mean exceeds float range")
        return result

    @property
    def variance(self) -> float:
        """Return the Weibull variance.

        Raises:
            OverflowError: If the mathematical variance exceeds float range.

        Equation:
            [EQ-DIST-WEIB-005].

        References:
            [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

        """
        try:
            first = math.gamma(1.0 + 1.0 / self.shape)
            second = math.gamma(1.0 + 2.0 / self.shape)
            result = self.scale**2 * (second - first**2)
        except (OverflowError, ValueError) as error:
            raise OverflowError("Weibull variance exceeds float range") from error
        if not math.isfinite(result):
            raise OverflowError("Weibull variance exceeds float range")
        return max(0.0, result)

    def hazard(self, x: float) -> float:
        """Evaluate the Weibull hazard rate, including its zero limit.

        Args:
            x: Finite real point on or above the support boundary.

        Returns:
            Instantaneous failure rate.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite or below zero.

        Equation:
            [EQ-DIST-WEIB-006].

        References:
            [REF-NIST-DISTRIBUTIONS, Weibull Distribution].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            raise ValueError("hazard is undefined below support")
        if point == 0.0:
            if self.shape < 1.0:
                return math.inf
            if self.shape == 1.0:
                return 1.0 / self.scale
            return 0.0
        try:
            return (self.shape / self.scale) * math.pow(
                point / self.scale, self.shape - 1.0
            )
        except OverflowError:
            return math.inf

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible inverse-transform Weibull observations.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of non-negative pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-WEIB-003].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        observations: list[float] = []
        for _ in range(checked_size):
            observation = self.scale * math.pow(
                -math.log1p(-generator.random()), 1.0 / self.shape
            )
            if not math.isfinite(observation):
                raise OverflowError("Weibull sample exceeds float range")
            observations.append(observation)
        return tuple(observations)
