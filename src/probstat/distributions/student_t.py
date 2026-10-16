"""Location-scale Student-t distribution with positive real degrees of freedom."""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.numerics.special import digamma

from ._base import ContinuousDistribution
from ._continuous import (
    SINH_LIMIT,
    log_standard_gamma_variate,
    safe_log,
    solve_tail_quantile,
)
from ._special import (
    LARGE_SHAPE,
    log_beta_function,
    log_gamma_ratio,
    regularized_beta_pair,
)
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class StudentT(ContinuousDistribution):
    """A Student-t law ``X = mu + sigma T`` with ``T ~ t(nu)``.

    The standard density is ``Gamma((nu+1)/2) / (sqrt(nu pi) Gamma(nu/2))
    (1 + t^2/nu)^(-(nu+1)/2)``. ``nu = 1`` is the Cauchy law and
    ``nu -> inf`` approaches ``Normal(mu, sigma)``.

    Args:
        nu: Finite positive degrees of freedom (need not be an integer).
        mu: Finite location.
        sigma: Finite positive scale.

    References:
        [REF-JKB-1995, Chapter 28], [REF-NIST-DISTRIBUTIONS, t Distribution].

    """

    nu: float
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        """Validate degrees of freedom, location, and scale.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``nu, sigma <= 0``.

        References:
            [EQ-DIST-STT-001], [REF-JKB-1995, Chapter 28].

        """
        object.__setattr__(self, "nu", POSITIVE_FINITE.validate(self.nu, name="nu"))
        object.__setattr__(self, "mu", FINITE_REAL.validate(self.mu, name="mu"))
        object.__setattr__(
            self, "sigma", POSITIVE_FINITE.validate(self.sigma, name="sigma")
        )

    @property
    def support(self) -> RealSupport:
        """Return the entire real line.

        References:
            [EQ-DIST-STT-001], [REF-JKB-1995, Chapter 28].

        """
        return RealSupport()

    def _upper_standard_tail(self, magnitude: float) -> float:
        """Return ``P(T > |t|)`` for the standard t law.

        Uses ``I_(nu/(nu+t^2))(nu/2, 1/2)/2`` with both ``nu/(nu+t^2)`` and
        its complement ``t^2/(nu+t^2)`` formed directly, so neither the centre
        nor the tails (nor ``nu -> inf``) lose accuracy to ``1 - x``. When
        ``t^2`` or ``nu + t^2`` would overflow, the same ratios are formed
        from ``r = sqrt(nu)/|t|`` as ``r^2/(1+r^2)`` and ``1/(1+r^2)``; when
        even ``r^2`` underflows (``x < 1e-300``) the leading term
        ``x^(nu/2) / (nu B(nu/2, 1/2))`` of the incomplete-beta series is
        used in log space, whose relative error ``O(x)`` is far below
        rounding. Hence ``StudentT(1).cdf(-1e200) = 1/(pi 1e200)``.

        Args:
            magnitude: Non-negative standardized point ``|t|``.

        Returns:
            Upper-tail probability in ``[0, 1/2]``.

        Equation:
            [EQ-DIST-STT-002], [EQ-DIST-T-001], [EQ-NUM-BETA-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-JKB-1995, Chapter 28].

        """
        if magnitude == math.inf:
            return 0.0
        squared = magnitude * magnitude
        denominator = self.nu + squared
        if denominator != math.inf:
            return 0.5 * regularized_beta_pair(
                self.nu / denominator, squared / denominator, 0.5 * self.nu, 0.5
            )
        ratio = math.sqrt(self.nu) / magnitude
        ratio_squared = ratio * ratio
        if ratio_squared >= 1e-300:
            scale = 1.0 + ratio_squared
            return 0.5 * regularized_beta_pair(
                ratio_squared / scale, 1.0 / scale, 0.5 * self.nu, 0.5
            )
        # I_x(nu/2, 1/2) = x^(nu/2) (1-x)^(1/2) / ((nu/2) B(nu/2, 1/2)) (1 + O(x))
        # with log x = 2 log r - log1p(r^2) and x < 1e-300.
        log_tail = (
            self.nu * math.log(ratio)
            - math.log(self.nu)
            - log_beta_function(0.5 * self.nu, 0.5)
        )
        return math.exp(log_tail) if log_tail > -745.2 else 0.0

    def _standardized(self, x: float) -> float:
        """Validate ``x`` and return ``(x - mu) / sigma``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Standardized point.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        References:
            [EQ-DIST-STT-001], [REF-JKB-1995, Chapter 28].

        """
        return (require_finite_real(x, name="x") - self.mu) / self.sigma

    def log_pdf(self, x: float) -> float:
        """Evaluate the location-scale Student-t log-density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Natural log density.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-STT-001], [EQ-NUM-STIRLING-001].

        References:
            [REF-JKB-1995, Chapter 28], [REF-NIST-DLMF-GAMMA, Section 5.11].

        """
        z = self._standardized(x)
        ratio = abs(z) / math.sqrt(self.nu)
        if ratio > 1e150:
            log_term = 2.0 * math.log(ratio)
        else:
            log_term = math.log1p(ratio * ratio)
        if 0.5 * self.nu >= LARGE_SHAPE:
            # lgamma((nu+1)/2) - lgamma(nu/2) from Stirling differences.
            log_ratio = log_gamma_ratio(0.5 * self.nu, 0.5)
        else:
            log_ratio = math.lgamma(0.5 * (self.nu + 1.0)) - math.lgamma(0.5 * self.nu)
        return (
            log_ratio
            - 0.5 * math.log(self.nu * math.pi)
            - math.log(self.sigma)
            - 0.5 * (self.nu + 1.0) * log_term
        )

    def pdf(self, x: float) -> float:
        """Evaluate the location-scale Student-t density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Positive density value.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-STT-001].

        References:
            [REF-JKB-1995, Chapter 28].

        """
        log_density = self.log_pdf(x)
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate the Student-t CDF through the incomplete beta function.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``P(X <= x)`` with relative accuracy in the lower tail.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-STT-002], [EQ-DIST-T-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        z = self._standardized(x)
        if z < 0.0:
            return self._upper_standard_tail(-z)
        return 1.0 - self._upper_standard_tail(z)

    def sf(self, x: float) -> float:
        """Evaluate the direct Student-t survival function.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``P(X > x)`` with relative accuracy in the upper tail.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-STT-002], [EQ-DIST-T-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17].

        """
        z = self._standardized(x)
        if z > 0.0:
            return self._upper_standard_tail(z)
        return 1.0 - self._upper_standard_tail(-z)

    def _invert(self, lower: float, upper: float) -> float:
        """Solve the tail equation on the coordinate ``t = asinh(z)``.

        Args:
            lower: Target lower-tail probability in ``(0, 1)``.
            upper: Complementary target upper-tail probability.

        Returns:
            Real quantile.

        Equation:
            [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-BRENT-1973, Chapter 4].

        """
        if lower == upper:
            return self.mu
        return solve_tail_quantile(
            lower,
            upper,
            log_cdf=lambda point: safe_log(self.cdf(point)),
            log_sf=lambda point: safe_log(self.sf(point)),
            to_x=lambda t: self.mu + self.sigma * math.sinh(t),
            start=0.0,
            minimum=-SINH_LIMIT,
            maximum=SINH_LIMIT,
        )

    def quantile(self, probability: float) -> float:
        """Return the Student-t quantile by tail-aware numerical inversion.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Real quantile, with infinite endpoints at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-INV-001], [EQ-DIST-STT-002].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-BRENT-1973, Chapter 4].

        """
        target = require_probability(probability, name="probability")
        if target == 0.0:
            return -math.inf
        if target == 1.0:
            return math.inf
        return self._invert(target, 1.0 - target)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function ``x`` with ``S(x) = q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Real quantile, with infinite endpoints at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-INV-001].

        References:
            [REF-NIST-DLMF-BETA, Section 8.17], [REF-BRENT-1973, Chapter 4].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf
        if q == 1.0:
            return -math.inf
        return self._invert(1.0 - q, q)

    @property
    def mean(self) -> float:
        """Return ``E[X] = mu`` for ``nu > 1``.

        Returns:
            ``mu``, or ``nan`` for ``nu <= 1`` where the mean is undefined
            (the integral is of the form ``inf - inf``).

        Equation:
            [EQ-DIST-STT-003].

        References:
            [REF-JKB-1995, Chapter 28].

        """
        if self.nu <= 1.0:
            return math.nan
        return self.mu

    @property
    def variance(self) -> float:
        """Return ``sigma^2 nu / (nu - 2)`` for ``nu > 2``.

        Returns:
            Variance, ``inf`` for ``1 < nu <= 2``, and ``nan`` for ``nu <= 1``
            (a central moment about an undefined mean).

        Equation:
            [EQ-DIST-STT-003].

        References:
            [REF-JKB-1995, Chapter 28].

        """
        if self.nu <= 1.0:
            return math.nan
        if self.nu <= 2.0:
            return math.inf
        return self.sigma * self.sigma * self.nu / (self.nu - 2.0)

    @property
    def skewness(self) -> float:
        """Return ``0`` for ``nu > 3`` and ``nan`` otherwise.

        Equation:
            [EQ-DIST-STT-005].

        References:
            [REF-JKB-1995, Chapter 28].

        """
        return 0.0 if self.nu > 3.0 else math.nan

    @property
    def kurtosis(self) -> float:
        """Return the excess kurtosis ``6 / (nu - 4)`` for ``nu > 4``.

        Returns:
            ``6/(nu - 4)``; ``inf`` for ``2 < nu <= 4`` (divergent fourth
            moment, finite variance); ``nan`` for ``nu <= 2``.

        Equation:
            [EQ-DIST-STT-005].

        References:
            [REF-JKB-1995, Chapter 28].

        """
        if self.nu > 4.0:
            return 6.0 / (self.nu - 4.0)
        return math.inf if self.nu > 2.0 else math.nan

    @property
    def entropy(self) -> float:
        """Return the differential entropy of the location-scale t law.

        ``H = (nu+1)/2 [psi((nu+1)/2) - psi(nu/2)] + log(sqrt(nu)
        B(nu/2, 1/2)) + log sigma``.

        Equation:
            [EQ-DIST-STT-005].

        References:
            [REF-JKB-1995, Chapter 28], [REF-COVER-THOMAS-2006, Chapter 8].

        """
        half = 0.5 * self.nu
        return (
            0.5
            * (self.nu + 1.0)
            * (digamma(half + 0.5) - digamma(half))
            + 0.5 * math.log(self.nu)
            + log_beta_function(half, 0.5)
            + math.log(self.sigma)
        )

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible Student-t observations as normal/chi ratios.

        ``T = Z / sqrt(V / nu)`` with ``Z ~ N(0, 1)`` and
        ``V = 2 G``, ``G ~ Gamma(nu/2, 1)``.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-STT-004], [EQ-DIST-GAM-005].

        References:
            [REF-DEVROYE-1986, Chapter IX], [REF-MARSAGLIA-TSANG-2000].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        log_half_nu = math.log(0.5 * self.nu)
        observations: list[float] = []
        for _ in range(checked_size):
            normal = generator.gauss(0.0, 1.0)
            log_gamma = log_standard_gamma_variate(0.5 * self.nu, generator)
            factor = math.exp(min(0.5 * (log_half_nu - log_gamma), 709.0))
            observations.append(self.mu + self.sigma * normal * factor)
        return tuple(observations)
