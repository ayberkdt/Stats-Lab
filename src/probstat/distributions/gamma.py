"""Gamma distribution in the shape-rate parameterization."""

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
    log_gamma_kernel,
    regularized_gamma_p,
    regularized_gamma_q,
)
from .constraints import POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class Gamma(ContinuousDistribution):
    """A gamma law with shape ``k`` and rate ``lambda`` (scale ``1/lambda``).

    Density ``f(x) = lambda^k x^(k-1) exp(-lambda x) / Gamma(k)`` on ``x >= 0``.
    ``Gamma(1, lambda)`` is ``Exponential(lambda)`` and ``Gamma(nu/2, 1/2)``
    is the chi-square law with ``nu`` degrees of freedom.

    Args:
        shape: Finite positive shape ``k``.
        rate: Finite positive rate ``lambda``.

    References:
        [REF-JKB-1994, Chapter 17], [REF-NIST-DISTRIBUTIONS, Gamma
        Distribution].

    """

    shape: float
    rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize shape and rate.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or non-positive.

        References:
            [EQ-DIST-GAM-001], [REF-JKB-1994, Chapter 17].

        """
        object.__setattr__(
            self, "shape", POSITIVE_FINITE.validate(self.shape, name="shape")
        )
        object.__setattr__(
            self, "rate", POSITIVE_FINITE.validate(self.rate, name="rate")
        )

    @classmethod
    def from_scale(cls, shape: float, scale: float) -> Gamma:
        """Build a gamma law from shape and scale ``theta = 1/rate``.

        Args:
            shape: Finite positive shape ``k``.
            scale: Finite positive scale ``theta``.

        Returns:
            ``Gamma(shape, 1/scale)``.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or non-positive.

        Equation:
            [EQ-DIST-GAM-001].

        References:
            [REF-JKB-1994, Chapter 17].

        """
        checked = POSITIVE_FINITE.validate(scale, name="scale")
        return cls(shape, 1.0 / checked)

    @property
    def scale(self) -> float:
        """Return the scale ``theta = 1/rate``.

        References:
            [EQ-DIST-GAM-001], [REF-JKB-1994, Chapter 17].

        """
        return 1.0 / self.rate

    @property
    def support(self) -> RealSupport:
        """Return non-negative real support.

        References:
            [EQ-DIST-GAM-001], [REF-JKB-1994, Chapter 17].

        """
        return RealSupport(0.0, None, lower_closed=True)

    def _scaled(self, point: float) -> float:
        """Return ``lambda x`` with overflow mapped to infinity.

        Args:
            point: Validated non-negative point.

        Returns:
            Scaled point, possibly infinite.

        References:
            [EQ-DIST-GAM-002], [REF-PYTHON-MATH].

        """
        return self.rate * point

    def log_pdf(self, x: float) -> float:
        """Evaluate the gamma log-density, including its endpoint limit.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density; at zero it is ``inf`` for ``k < 1``, ``log lambda``
            for ``k = 1``, and ``-inf`` for ``k > 1``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GAM-001], [EQ-NUM-STIRLING-001].

        References:
            [REF-JKB-1994, Chapter 17], [REF-NIST-DLMF-GAMMA, Section 5.11].

        """
        point = require_finite_real(x, name="x")
        if point < 0.0:
            return -math.inf
        if point == 0.0:
            if self.shape < 1.0:
                return math.inf
            if self.shape == 1.0:
                return math.log(self.rate)
            return -math.inf
        scaled = self._scaled(point)
        if scaled == math.inf:
            return -math.inf
        if self.shape >= LARGE_SHAPE and scaled > 0.0:
            # k log(lambda x) - lambda x - lgamma(k) without differencing
            # lgamma(k) (size k log k) against k log(lambda x).
            return log_gamma_kernel(self.shape, point, self.rate) - math.log(point)
        return (
            self.shape * math.log(self.rate)
            + (self.shape - 1.0) * math.log(point)
            - scaled
            - math.lgamma(self.shape)
        )

    def pdf(self, x: float) -> float:
        """Evaluate the gamma probability density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value, possibly infinite at zero when ``k < 1``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GAM-001].

        References:
            [REF-JKB-1994, Chapter 17].

        """
        log_density = self.log_pdf(x)
        if log_density == math.inf:
            return math.inf
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate the gamma CDF as ``P(k, lambda x)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower regularized incomplete gamma probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GAM-002], [EQ-NUM-GAMMA-001].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.2].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 0.0
        scaled = self._scaled(point)
        if scaled == math.inf:
            return 1.0
        return regularized_gamma_p(self.shape, scaled)

    def sf(self, x: float) -> float:
        """Evaluate the direct gamma survival function ``Q(k, lambda x)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting the CDF from one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GAM-002], [EQ-NUM-GAMMA-001].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.2].

        """
        point = require_finite_real(x, name="x")
        if point <= 0.0:
            return 1.0
        scaled = self._scaled(point)
        if scaled == math.inf:
            return 0.0
        return regularized_gamma_q(self.shape, scaled)

    def _invert(self, lower: float, upper: float) -> float:
        """Solve the tail equation on the log-coordinate ``t = log x``.

        Args:
            lower: Target lower-tail probability in ``(0, 1)``.
            upper: Complementary target upper-tail probability.

        Returns:
            Positive quantile.

        Equation:
            [EQ-DIST-INV-001], [EQ-DIST-GAM-003].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.10], [REF-BRENT-1973, Chapter 4].

        """
        return solve_tail_quantile(
            lower,
            upper,
            log_cdf=lambda point: safe_log(self.cdf(point)),
            log_sf=lambda point: safe_log(self.sf(point)),
            to_x=bounded_exp,
            start=math.log(self.shape / self.rate),
            minimum=LOG_MIN,
            maximum=LOG_MAX,
        )

    def quantile(self, probability: float) -> float:
        """Return the gamma quantile by tail-aware numerical inversion.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Non-negative quantile, ``0`` at zero and ``inf`` at one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-GAM-003], [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.10], [REF-BRENT-1973, Chapter 4].

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
            Quantile matched through ``log Q(k, lambda x) = log q``.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-GAMMA, Section 8.10], [REF-BRENT-1973, Chapter 4].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return 0.0
        return self._invert(1.0 - q, q)

    @property
    def mean(self) -> float:
        """Return ``E[X] = k / lambda``.

        Equation:
            [EQ-DIST-GAM-004].

        References:
            [REF-JKB-1994, Chapter 17].

        """
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        """Return ``Var(X) = k / lambda^2``.

        Equation:
            [EQ-DIST-GAM-004].

        References:
            [REF-JKB-1994, Chapter 17].

        """
        return self.shape / (self.rate * self.rate)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible gamma observations by Marsaglia-Tsang.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of non-negative pseudo-random observations; for very small
            shapes a variate may underflow to ``0.0``.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-GAM-005].

        References:
            [REF-MARSAGLIA-TSANG-2000].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        log_scale = -math.log(self.rate)
        observations: list[float] = []
        for _ in range(checked_size):
            logarithm = log_standard_gamma_variate(self.shape, generator) + log_scale
            if logarithm > LOG_MAX:
                raise OverflowError("gamma sample exceeds float range")
            observations.append(math.exp(logarithm) if logarithm > -745.2 else 0.0)
        return tuple(observations)
