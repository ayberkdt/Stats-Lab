"""Univariate normal distribution with validated finite parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._autodiff import admath, real_argument
from ._base import ContinuousDistribution
from ._continuous import (
    density_from_log,
    log_standard_normal_cdf,
    log_standard_normal_sf,
    standard_normal_sf_dd,
    standardized_double_double,
)
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport


@dataclass(frozen=True, slots=True)
class Normal(ContinuousDistribution):
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
        location = FINITE_REAL.validate_parameter(self.mu, name="mu")
        scale = POSITIVE_FINITE.validate_parameter(self.sigma, name="sigma")
        object.__setattr__(self, "mu", location)
        object.__setattr__(self, "sigma", scale)

    @property
    def support(self) -> RealSupport:
        """Return support over the entire real line.

        References:
            [EQ-DIST-NORM-001], [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        return RealSupport()

    def log_pdf(self, x: float) -> float:
        """Evaluate the natural logarithm of the normal density.

        Args:
            x: Finite real-valued evaluation point.

        Returns:
            Natural log density at ``x``.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is not finite.

        Equation:
            [EQ-DIST-NORM-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 3.3].

        """
        point = real_argument(x, name="x")
        z = (point - self.mu) / self.sigma
        return -0.5 * z * z - admath.log(self.sigma) - 0.5 * math.log(
            2.0 * math.pi
        )

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
        return density_from_log(self.log_pdf(x))

    def cdf(self, x: float) -> float:
        """Evaluate the normal cumulative distribution function at ``x``.

        Args:
            x: Finite real-valued evaluation point.

        Returns:
            ``P(X <= x)`` computed through the error function. The
            standardized point is carried in double-double arithmetic, so the
            error stays within a few ulp over the whole range instead of
            growing like ``(1 + z^2)`` ulp in the tails (ADR-0089).

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is not finite.

        Equation:
            [EQ-DIST-NORM-002], [EQ-DIST-NORM-007].

        References:
            [REF-NIST-DLMF, Equation 7.2.1], [REF-DEKKER-1971].

        """
        point = require_finite_real(x, name="x")
        high, low = standardized_double_double(point, self.mu, self.sigma)
        # erfc(-z/sqrt 2)/2 equals (1 + erf(z/sqrt 2))/2 but keeps full
        # relative accuracy in the lower tail, where 1 + erf cancels.
        return standard_normal_sf_dd((-high, -low))

    def log_cdf(self, x: float) -> float:
        """Return ``log Phi((x - mu)/sigma)``, finite far beyond ``erfc`` underflow.

        Equation:
            [EQ-DIST-TN-004], [EQ-DIST-NORM-007].

        References:
            [REF-NIST-DLMF, Sections 7.2 and 7.9].

        """
        point = require_finite_real(x, name="x")
        return log_standard_normal_cdf(
            *standardized_double_double(point, self.mu, self.sigma)
        )

    def log_sf(self, x: float) -> float:
        """Return ``log(1 - Phi((x - mu)/sigma))`` without underflow.

        Equation:
            [EQ-DIST-TN-004], [EQ-DIST-NORM-007].

        References:
            [REF-NIST-DLMF, Sections 7.2 and 7.9].

        """
        point = require_finite_real(x, name="x")
        return log_standard_normal_sf(
            *standardized_double_double(point, self.mu, self.sigma)
        )

    def sf(self, x: float) -> float:
        """Evaluate the normal survival function through ``erfc``.

        Args:
            x: Finite real-valued evaluation point.

        Returns:
            Upper-tail probability evaluated without near-one subtraction,
            from a double-double standardized point (ADR-0089).

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-NORM-005], [EQ-DIST-NORM-007].

        References:
            [REF-NIST-DLMF, Equation 7.2.2], [REF-DEKKER-1971].

        """
        point = require_finite_real(x, name="x")
        return standard_normal_sf_dd(
            standardized_double_double(point, self.mu, self.sigma)
        )

    def quantile(self, probability: float) -> float:
        """Return the normal generalized inverse CDF.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Real quantile, with infinite endpoints at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-NORM-006].

        References:
            [REF-PYTHON-STATISTICS].

        """
        target = require_probability(probability, name="probability")
        if target == 0.0:
            return -math.inf
        if target == 1.0:
            return math.inf
        return NormalDist(mu=self.mu, sigma=self.sigma).inv_cdf(target)

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
            return -math.inf
        return self.mu - self.sigma * NormalDist().inv_cdf(q)

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

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible pseudo-random normal observations.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

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
        generator = resolve_stream(seed).generator()
        return tuple(generator.gauss(self.mu, self.sigma) for _ in range(checked_size))

