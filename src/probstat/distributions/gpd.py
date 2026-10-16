"""Generalized Pareto distribution with a stable exponential limit."""

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
from ._continuous import open_unit
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport

_TINY_PRODUCT = 1e-280
_EXP_LIMIT = 709.0


@dataclass(frozen=True, slots=True)
class GeneralizedPareto(ContinuousDistribution):
    """A GPD law ``S(x) = (1 + xi z)^(-1/xi)``, ``z = (x - mu)/sigma >= 0``.

    The Pickands-Balkema-de Haan theorem makes this the limit law of
    threshold excesses: ``mu`` plays the role of the threshold. ``xi > 0`` is
    heavy-tailed (Pareto type), ``xi = 0`` is the exponential law with mean
    ``sigma`` and ``xi < 0`` has the finite upper endpoint ``mu - sigma/xi``
    (``xi = -1`` is the uniform law on it). Every formula uses
    ``log1p(xi z)/xi`` or ``expm1(xi L)/xi``, so the ``xi -> 0`` limit is
    continuous; the sign convention matches :class:`GeneralizedExtremeValue`.

    Args:
        mu: Finite location (threshold).
        sigma: Finite positive scale.
        xi: Finite shape.

    References:
        [REF-PICKANDS-1975], [REF-BALKEMA-DEHAAN-1974],
        [REF-COLES-2001, Chapter 4].

    """

    mu: float = 0.0
    sigma: float = 1.0
    xi: float = 0.0

    def __post_init__(self) -> None:
        """Validate location, scale, and shape.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite or ``sigma <= 0``.

        References:
            [EQ-DIST-GPD-001], [REF-COLES-2001, Chapter 4].

        """
        object.__setattr__(self, "mu", FINITE_REAL.validate(self.mu, name="mu"))
        object.__setattr__(
            self, "sigma", POSITIVE_FINITE.validate(self.sigma, name="sigma")
        )
        object.__setattr__(self, "xi", FINITE_REAL.validate(self.xi, name="xi"))

    @property
    def upper_endpoint(self) -> float | None:
        """Return the finite upper endpoint ``mu - sigma/xi`` for ``xi < 0``.

        Returns:
            Upper endpoint, or ``None`` for ``xi >= 0`` or float overflow.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        if self.xi >= 0.0:
            return None
        try:
            value = self.mu - self.sigma / self.xi
        except OverflowError:
            return None
        return value if math.isfinite(value) else None

    @property
    def support(self) -> RealSupport:
        """Return ``[mu, inf)`` or ``[mu, mu - sigma/xi]``.

        References:
            [EQ-DIST-GPD-001], [REF-COLES-2001, Section 4.2].

        """
        end = self.upper_endpoint
        return RealSupport(
            self.mu, end, lower_closed=True, upper_closed=end is not None
        )

    def _reduced(self, x: float) -> float:
        """Return ``y = log1p(xi z)/xi`` (``y = z`` at ``xi = 0``), ``S = e^-y``.

        Args:
            x: Finite real evaluation point.

        Returns:
            ``-inf`` below ``mu`` (not used for probabilities), ``inf`` at or
            beyond a finite upper endpoint, otherwise the reduced variate.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        z = (require_finite_real(x, name="x") - self.mu) / self.sigma
        if z < 0.0:
            return -math.inf
        w = self.xi * z
        if w <= -1.0:
            return math.inf
        if abs(w) < _TINY_PRODUCT:
            return z
        return math.log1p(w) / self.xi

    def log_pdf(self, x: float) -> float:
        """Evaluate ``-log sigma - (1 + xi) y`` on the support.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density, ``-inf`` off the support. At a finite upper endpoint
            the value is ``-inf``, ``-log sigma`` or ``inf`` for
            ``-1 < xi < 0``, ``xi = -1`` or ``xi < -1``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        y = self._reduced(x)
        if y == -math.inf:
            return -math.inf
        if y == math.inf:
            end = self.upper_endpoint
            point = float(x)
            if end is None or point > end:
                return -math.inf
            if self.xi > -1.0:
                return -math.inf
            if self.xi == -1.0:
                return -math.log(self.sigma)
            return math.inf
        return -math.log(self.sigma) - (1.0 + self.xi) * y

    def pdf(self, x: float) -> float:
        """Evaluate the GPD density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        log_density = self.log_pdf(x)
        if log_density == math.inf:
            return math.inf
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def log_sf(self, x: float) -> float:
        """Evaluate ``log S(x) = -y`` without forming ``S``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log survival probability (``0`` below ``mu``, ``-inf`` beyond a
            finite upper endpoint).

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        y = self._reduced(x)
        return 0.0 if y == -math.inf else -y

    def sf(self, x: float) -> float:
        """Evaluate the direct survival function ``exp(-y)``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        log_survival = self.log_sf(x)
        return math.exp(log_survival) if log_survival > -745.2 else 0.0

    def cdf(self, x: float) -> float:
        """Evaluate ``F(x) = -expm1(-y)`` with full accuracy near ``mu``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        y = self._reduced(x)
        if y == -math.inf:
            return 0.0
        return -math.expm1(-y)

    def hazard(self, x: float) -> float:
        """Return the hazard ``1/(sigma (1 + xi z))`` on the support.

        Args:
            x: Finite real evaluation point.

        Returns:
            Zero below ``mu``; ``inf`` at or beyond a finite upper endpoint.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-GPD-001].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        z = (require_finite_real(x, name="x") - self.mu) / self.sigma
        if z < 0.0:
            return 0.0
        denominator = 1.0 + self.xi * z
        if denominator <= 0.0:
            return math.inf
        return 1.0 / (self.sigma * denominator)

    def _from_log_tail(self, exponent: float) -> float:
        """Return ``mu + sigma expm1(xi L)/xi`` for ``L = -log S >= 0``.

        Args:
            exponent: Non-negative ``L``.

        Returns:
            Physical quantile.

        Raises:
            OverflowError: If the quantile exceeds float range.

        Equation:
            [EQ-DIST-GPD-002].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        power = self.xi * exponent
        if abs(power) < _TINY_PRODUCT:
            reduced = exponent
        else:
            if power > _EXP_LIMIT:
                raise OverflowError("GPD quantile exceeds float range")
            reduced = math.expm1(power) / self.xi
        result = self.mu + self.sigma * reduced
        if not math.isfinite(result):
            raise OverflowError("GPD quantile exceeds float range")
        end = self.upper_endpoint
        return min(result, end) if end is not None else result

    def quantile(self, probability: float) -> float:
        """Return the closed-form GPD quantile with ``L = -log1p(-p)``.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Quantile; ``p = 0`` gives ``mu`` and ``p = 1`` the upper endpoint
            (``inf`` for ``xi >= 0``).

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.
            OverflowError: If a finite quantile exceeds float range.

        Equation:
            [EQ-DIST-GPD-002].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return self.mu
        if p == 1.0:
            end = self.upper_endpoint
            return end if end is not None else math.inf
        return self._from_log_tail(-math.log1p(-p))

    def isf(self, probability: float) -> float:
        """Return the inverse survival function with ``L = -log q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Level exceeded with probability ``q``, accurate for tiny ``q``.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.
            OverflowError: If a finite quantile exceeds float range.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-GPD-002].

        References:
            [REF-COLES-2001, Section 4.2].

        """
        q = require_probability(probability, name="probability")
        if q == 1.0:
            return self.mu
        if q == 0.0:
            end = self.upper_endpoint
            return end if end is not None else math.inf
        return self._from_log_tail(-math.log(q))

    @property
    def mean(self) -> float:
        """Return ``mu + sigma/(1 - xi)``.

        Returns:
            Mean, or ``inf`` for ``xi >= 1``.

        Equation:
            [EQ-DIST-GPD-003].

        References:
            [REF-COLES-2001, Section 4.2], [REF-HOSKING-WALLIS-1987].

        """
        if self.xi >= 1.0:
            return math.inf
        return self.mu + self.sigma / (1.0 - self.xi)

    @property
    def variance(self) -> float:
        """Return ``sigma^2 / ((1 - xi)^2 (1 - 2 xi))``.

        Returns:
            Variance, or ``inf`` for ``xi >= 1/2``.

        Equation:
            [EQ-DIST-GPD-003].

        References:
            [REF-HOSKING-WALLIS-1987].

        """
        if self.xi >= 0.5:
            return math.inf
        one_minus = 1.0 - self.xi
        return self.sigma * self.sigma / (one_minus * one_minus * (1.0 - 2.0 * self.xi))

    @property
    def entropy(self) -> float:
        """Return the differential entropy ``log sigma + xi + 1`` in nats.

        ``-log S(X)`` is standard exponential, so ``E[y] = 1`` and
        ``h = log sigma + (1 + xi) E[y]`` for every real ``xi``.

        Equation:
            [EQ-DIST-GPD-003].

        References:
            [REF-COVER-THOMAS-2006, Chapter 8], [REF-COLES-2001, Section 4.2].

        """
        return math.log(self.sigma) + self.xi + 1.0

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible GPD observations by inverse survival transform.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.
            OverflowError: If a variate exceeds float range.

        Equation:
            [EQ-DIST-GPD-002].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(
            self._from_log_tail(-math.log(open_unit(generator)))
            for _ in range(checked_size)
        )
