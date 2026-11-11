"""Beta-PERT law on a finite interval with a most likely value.

The beta-PERT law is a rescaled :class:`~probstat.distributions.Beta` and
inherits its tail-accurate distribution functions.

References:
    [REF-MALCOLM-1959], [REF-VOSE-2008].

"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
)
from probstat.numerics.special import digamma

from ._base import ContinuousDistribution
from ._special import log_beta_function
from .beta import Beta
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class PERT(ContinuousDistribution):
    """The beta-PERT law on ``[low, high]`` with most likely value ``mode``.

    ``X = low + (high - low) B`` with ``B ~ Beta(alpha, beta)``,
    ``alpha = 1 + lam (mode - low)/(high - low)`` and
    ``beta = 1 + lam (high - mode)/(high - low)``. The standard PERT choice
    ``lam = 4`` gives the mean ``(low + 4 mode + high)/6`` of the classical
    PERT three-point estimate; other ``lam > 0`` give the "modified PERT"
    family (larger ``lam`` concentrates the law around the mode).

    Args:
        low: Finite minimum.
        mode: Most likely value with ``low <= mode <= high``.
        high: Finite maximum with ``low < high``.
        lam: Positive shape weight ``lam`` (default 4).

    References:
        [REF-MALCOLM-1959], [REF-VOSE-2008].

    """

    low: float
    mode: float
    high: float
    lam: float = 4.0

    def __post_init__(self) -> None:
        """Validate the three points and the shape weight.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If ``low < high`` and ``low <= mode <= high`` fail or
                ``lam <= 0``.

        References:
            [EQ-DIST-PERT-001], [REF-MALCOLM-1959].

        """
        low = FINITE_REAL.validate(self.low, name="low")
        mode = FINITE_REAL.validate(self.mode, name="mode")
        high = FINITE_REAL.validate(self.high, name="high")
        lam = POSITIVE_FINITE.validate(self.lam, name="lam")
        if not low < high:
            raise ValueError("low must be below high")
        if not low <= mode <= high:
            raise ValueError("mode must lie in [low, high]")
        for name, value in (("low", low), ("mode", mode), ("high", high)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "lam", lam)

    @property
    def width(self) -> float:
        """Return ``high - low``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        return self.high - self.low

    @property
    def alpha(self) -> float:
        """Return ``1 + lam (mode - low)/(high - low)``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        return 1.0 + self.lam * (self.mode - self.low) / self.width

    @property
    def beta(self) -> float:
        """Return ``1 + lam (high - mode)/(high - low)``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        return 1.0 + self.lam * (self.high - self.mode) / self.width

    @property
    def standard(self) -> Beta:
        """Return the underlying ``Beta(alpha, beta)`` law on ``[0, 1]``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        return Beta(self.alpha, self.beta)

    @property
    def support(self) -> RealSupport:
        """Return ``[low, high]``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        return RealSupport(self.low, self.high, lower_closed=True, upper_closed=True)

    def _unit(self, x: float) -> float:
        """Map a validated point to ``(x - low)/(high - low)`` clipped to ``[0, 1]``.

        References:
            [EQ-DIST-PERT-001]. No new equation.

        """
        point = require_finite_real(x, name="x")
        return min(max((point - self.low) / self.width, 0.0), 1.0)

    def log_pdf(self, x: float) -> float:
        """Return ``log f_B(u) - log(high - low)`` (``-inf`` outside).

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        point = require_finite_real(x, name="x")
        if point < self.low or point > self.high:
            return -math.inf
        return self.standard.log_pdf(self._unit(point)) - math.log(self.width)

    def pdf(self, x: float) -> float:
        """Return the PERT density.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        value = self.log_pdf(x)
        return math.exp(value) if value > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Return ``F_B((x - low)/(high - low))``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        return self.standard.cdf(self._unit(x))

    def sf(self, x: float) -> float:
        """Return ``S_B((x - low)/(high - low))``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        return self.standard.sf(self._unit(x))

    def quantile(self, probability: float) -> float:
        """Return ``low + (high - low) Q_B(p)``.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        return self.low + self.width * self.standard.quantile(probability)

    def isf(self, probability: float) -> float:
        """Return ``low + (high - low) isf_B(q)``.

        Equation:
            [EQ-DIST-PERT-001], [EQ-DIST-ISF-001].

        References:
            [REF-VOSE-2008].

        """
        return self.low + self.width * self.standard.isf(probability)

    @property
    def mean(self) -> float:
        """Return ``(low + lam mode + high) / (lam + 2)``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-VOSE-2008].

        """
        return (self.low + self.lam * self.mode + self.high) / (self.lam + 2.0)

    @property
    def variance(self) -> float:
        """Return ``(mean - low)(high - mean) / (lam + 3)``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-VOSE-2008].

        """
        mean = self.mean
        return (mean - self.low) * (self.high - mean) / (self.lam + 3.0)

    @property
    def skewness(self) -> float:
        """Return the beta skewness of ``Beta(a, b)``.

        ``2 (b - a) sqrt(a + b + 1) / ((a + b + 2) sqrt(a b))``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        a, b = self.alpha, self.beta
        return (
            2.0 * (b - a) * math.sqrt(a + b + 1.0) / ((a + b + 2.0) * math.sqrt(a * b))
        )

    @property
    def kurtosis(self) -> float:
        """Return the beta excess kurtosis.

        ``6 [(a - b)^2 (a + b + 1) - a b (a + b + 2)] / (a b (a + b + 2)
        (a + b + 3))``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-JKB-1995, Chapter 25].

        """
        a, b = self.alpha, self.beta
        total = a + b
        numerator = (a - b) ** 2 * (total + 1.0) - a * b * (total + 2.0)
        return 6.0 * numerator / (a * b * (total + 2.0) * (total + 3.0))

    @property
    def entropy(self) -> float:
        """Return the differential entropy of the rescaled beta law.

        ``log B(a, b) - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b) +
        log(high - low)``.

        Equation:
            [EQ-DIST-PERT-002].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        a, b = self.alpha, self.beta
        return (
            log_beta_function(a, b)
            - (a - 1.0) * digamma(a)
            - (b - 1.0) * digamma(b)
            + (a + b - 2.0) * digamma(a + b)
            + math.log(self.width)
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw PERT variates by rescaling beta variates.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` or ``seed`` has the wrong type.
            ValueError: If ``size`` or an integer seed is negative.

        Equation:
            [EQ-DIST-PERT-001].

        References:
            [REF-VOSE-2008].

        """
        draws = self.standard.sample(size, seed=resolve_stream(seed))
        return tuple(self.low + self.width * value for value in draws)
