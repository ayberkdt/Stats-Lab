"""Normal distribution truncated to a finite or one-sided interval."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    LOGIT_LIMIT,
    SINH_LIMIT,
    bounded_exp,
    expit,
    log_normal_interval_mass,
    log_normal_short_mass,
    open_unit,
    solve_tail_quantile,
)
from .constraints import FINITE_REAL, POSITIVE_FINITE
from .support import RealSupport

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_SHORT_INTERVAL = 0.5


def _log_mass_from(anchor: float, width: float, far: float) -> float:
    """Return ``log(Phi(anchor + width) - Phi(anchor))`` (``far = anchor + width``).

    Short intervals use the width-based series so the width is never
    recovered as a difference of standardized points.

    Args:
        anchor: Finite standardized start.
        width: Non-negative standardized width.
        far: Standardized end, possibly infinite.

    Returns:
        Log standard-normal interval probability.

    Equation:
        [EQ-DIST-TN-002], [EQ-DIST-TN-005].

    References:
        [REF-NIST-DLMF, Sections 7.2 and 7.9].

    """
    if width <= 0.0:
        return -math.inf
    if width * (abs(anchor) + width) <= _SHORT_INTERVAL:
        return log_normal_short_mass(anchor, width)
    return log_normal_interval_mass(min(anchor, far), max(anchor, far))


def _short_interval_moments(centre: float, half_width: float) -> tuple[float, float]:
    """Return ``(E[u], Var[u])/h^k`` for ``u = z - c`` on a short interval.

    On ``[c - h, c + h]`` the standardized density is proportional to
    ``w(u) = exp(-c u - u^2/2)``, ``|u| <= h``. Its Maclaurin coefficients
    satisfy ``(n+1) p_(n+1) = -c p_n - p_(n-1)``; with ``q_n = p_n h^n`` the
    moments are ``int u^k w = 2 h^(k+1) sum_(n+k even) q_n/(n+k+1)``. For
    ``h (|c| + h) <= 1/2`` the scaled coefficients decay roughly like
    ``(h (|c| + h))^n / n!``, so the sums converge in a few dozen terms
    without cancellation.

    Args:
        centre: Standardized interval midpoint ``c``.
        half_width: Standardized half-width ``h > 0``.

    Returns:
        ``(E[u]/h, Var[u]/h^2)``; the caller multiplies by ``h`` and ``h^2``
        in physical units.

    Equation:
        [EQ-DIST-TN-006].

    References:
        [REF-JKB-1994, Chapter 13], [REF-NIST-DLMF, Section 7.6].

    """
    tilt = centre * half_width
    curvature = half_width * half_width
    previous, current = 0.0, 1.0  # q_(-1), q_0
    sums = [1.0, 0.0, 1.0 / 3.0]  # contributions of q_0 to S_0, S_1, S_2
    index = 0
    while True:
        following = (-tilt * current - curvature * previous) / (index + 1)
        index += 1
        previous, current = current, following
        for k in range(3):
            if (index + k) % 2 == 0:
                sums[k] += current / (index + k + 1)
        # Odd coefficients vanish when c = 0, so two consecutive terms must
        # be negligible before stopping.
        negligible = 1e-18 * abs(sums[0])
        if abs(current) <= negligible and abs(previous) <= negligible:
            break
        if index > 400:
            break
    first = sums[1] / sums[0]
    second = sums[2] / sums[0]
    return first, max(second - first * first, 0.0)


@dataclass(frozen=True, slots=True)
class TruncatedNormal(ContinuousDistribution):
    """A ``Normal(mu, sigma)`` law conditioned on ``lower <= X <= upper``.

    ``None`` denotes an unbounded side, so one-sided truncations such as a
    positive normal (``lower=0``) are supported. Probabilities are evaluated
    in log space and short intervals next to a truncation point use the
    physical distance to that point, so truncation regions deep in a normal
    tail (for example ``lower = mu + 40 sigma``) remain well defined.

    Args:
        mu: Finite location of the parent normal law.
        sigma: Finite positive scale of the parent normal law.
        lower: Optional finite lower truncation point.
        upper: Optional finite upper truncation point, ``upper > lower``.

    References:
        [REF-JKB-1994, Chapter 13], [REF-NIST-DLMF, Sections 7.2 and 7.9].

    """

    mu: float = 0.0
    sigma: float = 1.0
    lower: float | None = None
    upper: float | None = None
    _alpha: float = field(init=False, repr=False, compare=False)
    _beta: float = field(init=False, repr=False, compare=False)
    _log_mass: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate parameters and cache the log normalizing mass.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If a parameter is non-finite, ``sigma <= 0``,
                ``upper <= lower``, or the interval mass underflows to zero.

        Equation:
            [EQ-DIST-TN-002].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        mu = FINITE_REAL.validate(self.mu, name="mu")
        sigma = POSITIVE_FINITE.validate(self.sigma, name="sigma")
        lower = (
            None
            if self.lower is None
            else FINITE_REAL.validate(self.lower, name="lower")
        )
        upper = (
            None
            if self.upper is None
            else FINITE_REAL.validate(self.upper, name="upper")
        )
        if lower is not None and upper is not None and not lower < upper:
            raise ValueError("lower must be strictly less than upper")
        alpha = -math.inf if lower is None else (lower - mu) / sigma
        beta = math.inf if upper is None else (upper - mu) / sigma
        if lower is not None and upper is not None:
            log_mass = _log_mass_from(alpha, (upper - lower) / sigma, beta)
        else:
            log_mass = log_normal_interval_mass(alpha, beta)
        if not math.isfinite(log_mass):
            raise ValueError("truncation interval has numerically zero mass")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "_alpha", alpha)
        object.__setattr__(self, "_beta", beta)
        object.__setattr__(self, "_log_mass", log_mass)

    @property
    def support(self) -> RealSupport:
        """Return the truncation interval, closed at finite ends.

        References:
            [EQ-DIST-TN-001], [REF-JKB-1994, Chapter 13].

        """
        return RealSupport(
            self.lower,
            self.upper,
            lower_closed=self.lower is not None,
            upper_closed=self.upper is not None,
        )

    def log_pdf(self, x: float) -> float:
        """Evaluate ``log phi(z) - log sigma - log Z`` inside the interval.

        Args:
            x: Finite real evaluation point.

        Returns:
            Log density, ``-inf`` outside the truncation interval.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TN-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        point = require_finite_real(x, name="x")
        if (self.lower is not None and point < self.lower) or (
            self.upper is not None and point > self.upper
        ):
            return -math.inf
        z = (point - self.mu) / self.sigma
        return -0.5 * z * z - _LOG_SQRT_TWO_PI - math.log(self.sigma) - self._log_mass

    def pdf(self, x: float) -> float:
        """Evaluate the truncated normal density.

        Args:
            x: Finite real evaluation point.

        Returns:
            Density value, zero outside the truncation interval.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TN-001].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        log_density = self.log_pdf(x)
        return math.exp(log_density) if log_density > -745.2 else 0.0

    def _log_cdf(self, point: float) -> float:
        """Return ``log F(point)`` for a physical point (may be infinite).

        References:
            [EQ-DIST-TN-002], [EQ-DIST-TN-005], [REF-NIST-DLMF, Section 7.9].

        """
        if self.lower is not None and point <= self.lower:
            return -math.inf
        if self.upper is not None and point >= self.upper:
            return 0.0
        z = (point - self.mu) / self.sigma
        if self.lower is not None:
            width = (point - self.lower) / self.sigma
            mass = _log_mass_from(self._alpha, width, z)
        else:
            mass = log_normal_interval_mass(-math.inf, z)
        return min(0.0, mass - self._log_mass)

    def _log_sf(self, point: float) -> float:
        """Return ``log S(point)`` for a physical point (may be infinite).

        References:
            [EQ-DIST-TN-002], [EQ-DIST-TN-005], [REF-NIST-DLMF, Section 7.9].

        """
        if self.upper is not None and point >= self.upper:
            return -math.inf
        if self.lower is not None and point <= self.lower:
            return 0.0
        z = (point - self.mu) / self.sigma
        if self.upper is not None:
            width = (self.upper - point) / self.sigma
            # Mass of [beta - w, beta] equals the mass of [-beta, -beta + w].
            mass = _log_mass_from(-self._beta, width, -z)
        else:
            mass = log_normal_interval_mass(z, math.inf)
        return min(0.0, mass - self._log_mass)

    def cdf(self, x: float) -> float:
        """Evaluate ``(Phi(z) - Phi(alpha)) / Z`` in log space.

        Args:
            x: Finite real evaluation point.

        Returns:
            Lower-tail probability.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TN-002].

        References:
            [REF-JKB-1994, Chapter 13], [REF-NIST-DLMF, Section 7.9].

        """
        logarithm = self._log_cdf(require_finite_real(x, name="x"))
        return math.exp(logarithm) if logarithm > -745.2 else 0.0

    def sf(self, x: float) -> float:
        """Evaluate the direct survival ``(Phi(beta) - Phi(z)) / Z``.

        Args:
            x: Finite real evaluation point.

        Returns:
            Upper-tail probability without subtracting the CDF from one.

        Raises:
            TypeError: If ``x`` is not real.
            ValueError: If ``x`` is non-finite.

        Equation:
            [EQ-DIST-TN-002].

        References:
            [REF-JKB-1994, Chapter 13], [REF-NIST-DLMF, Section 7.9].

        """
        logarithm = self._log_sf(require_finite_real(x, name="x"))
        return math.exp(logarithm) if logarithm > -745.2 else 0.0

    def _coordinate(self, t: float) -> float:
        """Map an unconstrained coordinate ``t`` into the support.

        Both ends finite: ``l + (u - l) expit(t)`` (anchored at the nearer
        end); one finite end: ``l + sigma exp(t)`` or ``u - sigma exp(-t)``;
        none: ``mu + sigma sinh(t)``.

        Args:
            t: Real coordinate.

        Returns:
            Physical point.

        Equation:
            [EQ-DIST-INV-001].

        References:
            [REF-BRENT-1973, Chapter 4].

        """
        if self.lower is not None and self.upper is not None:
            width = self.upper - self.lower
            if t <= 0.0:
                return self.lower + width * expit(t)
            return self.upper - width * expit(-t)
        if self.lower is not None:
            return self.lower + self.sigma * bounded_exp(t)
        if self.upper is not None:
            return self.upper - self.sigma * bounded_exp(-t)
        return self.mu + self.sigma * math.sinh(t)

    def _invert(self, lower: float, upper: float) -> float:
        """Solve the log-tail equation on the support coordinate.

        Args:
            lower: Target lower-tail probability in ``(0, 1)``.
            upper: Complementary target upper-tail probability.

        Returns:
            Physical quantile clipped to the truncation interval.

        Equation:
            [EQ-DIST-INV-001], [EQ-DIST-TN-002].

        References:
            [REF-BRENT-1973, Chapter 4], [REF-NIST-DLMF, Section 7.9].

        """
        if self.lower is not None and self.upper is not None:
            start, minimum, maximum = 0.0, -LOGIT_LIMIT, LOGIT_LIMIT
        elif self.lower is not None:
            # exp(t) is the distance above lower in sigma units (~1/alpha
            # deep in the upper tail).
            alpha = self._alpha
            distance = 1.0 / (1.0 + alpha) if alpha >= 0.0 else 1.0 - alpha
            start, minimum, maximum = math.log(distance), LOG_MIN, LOG_MAX
        elif self.upper is not None:
            beta = self._beta
            distance = 1.0 / (1.0 - beta) if beta <= 0.0 else 1.0 + beta
            start, minimum, maximum = -math.log(distance), -LOG_MAX, -LOG_MIN
        else:
            start, minimum, maximum = 0.0, -SINH_LIMIT, SINH_LIMIT
        result = solve_tail_quantile(
            lower,
            upper,
            log_cdf=self._log_cdf,
            log_sf=self._log_sf,
            to_x=self._coordinate,
            start=start,
            minimum=minimum,
            maximum=maximum,
        )
        if self.lower is not None:
            result = max(result, self.lower)
        if self.upper is not None:
            result = min(result, self.upper)
        return result

    def quantile(self, probability: float) -> float:
        """Return the truncated normal quantile by log-tail inversion.

        Args:
            probability: Cumulative probability in ``[0, 1]``.

        Returns:
            Quantile in the truncation interval, with the interval ends (or
            infinities) at zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is non-finite or outside ``[0, 1]``.

        Equation:
            [EQ-DIST-INV-001], [EQ-DIST-TN-002].

        References:
            [REF-BRENT-1973, Chapter 4], [REF-NIST-DLMF, Section 7.9].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return -math.inf if self.lower is None else self.lower
        if p == 1.0:
            return math.inf if self.upper is None else self.upper
        return self._invert(p, 1.0 - p)

    def isf(self, probability: float) -> float:
        """Return the inverse survival function ``x`` with ``S(x) = q``.

        Args:
            probability: Upper-tail probability ``q`` in ``[0, 1]``.

        Returns:
            Quantile in the truncation interval.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-DIST-ISF-001], [EQ-DIST-INV-001].

        References:
            [REF-BRENT-1973, Chapter 4], [REF-NIST-DLMF, Section 7.9].

        """
        q = require_probability(probability, name="probability")
        if q == 0.0:
            return math.inf if self.upper is None else self.upper
        if q == 1.0:
            return -math.inf if self.lower is None else self.lower
        return self._invert(1.0 - q, q)

    def _density_ratio(self, z: float) -> float:
        """Return ``phi(z) / Z`` (zero at infinite ``z``).

        References:
            [EQ-DIST-TN-003], [REF-JKB-1994, Chapter 13].

        """
        if not math.isfinite(z):
            return 0.0
        return math.exp(-0.5 * z * z - _LOG_SQRT_TWO_PI - self._log_mass)

    def _short_moments(self) -> tuple[float, float] | None:
        """Return ``(mean, variance)`` by the short-interval series, if it applies.

        For two finite ends with ``w (|alpha| + w) <= 1/2`` (``w`` the
        standardized width) the closed forms below subtract nearly equal
        density ratios: ``(0, 1, -1e-8, 1e-8)`` gave a variance of
        ``5.55e-17`` instead of ``3.33e-17``. The moments are then taken from
        the series about the interval midpoint, in physical units.

        Returns:
            Mean and variance, or ``None`` when the closed forms are used.

        Equation:
            [EQ-DIST-TN-006].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        if self.lower is None or self.upper is None:
            return None
        width = (self.upper - self.lower) / self.sigma
        if width * (abs(self._alpha) + width) > _SHORT_INTERVAL:
            return None
        half_physical = 0.5 * (self.upper - self.lower)
        midpoint = self.lower + half_physical
        centre = (midpoint - self.mu) / self.sigma
        first, second = _short_interval_moments(centre, half_physical / self.sigma)
        return midpoint + half_physical * first, half_physical * half_physical * second

    @property
    def mean(self) -> float:
        """Return ``mu + sigma (phi(alpha) - phi(beta)) / Z``.

        Short two-sided intervals use the midpoint series of
        :meth:`_short_moments` instead of the density-ratio difference.

        Equation:
            [EQ-DIST-TN-003], [EQ-DIST-TN-006].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        short = self._short_moments()
        if short is not None:
            return short[0]
        shift = self._density_ratio(self._alpha) - self._density_ratio(self._beta)
        return self.mu + self.sigma * shift

    @property
    def variance(self) -> float:
        """Return ``sigma^2 [1 + (alpha phi(alpha) - beta phi(beta))/Z - m^2]``.

        ``m = (phi(alpha) - phi(beta))/Z``; infinite endpoints contribute zero.
        The bracket loses relative accuracy as a finite end moves far into a
        normal tail (roughly ``alpha^4`` units of roundoff). Short two-sided
        intervals (``w (|alpha| + w) <= 1/2``) use the cancellation-free
        midpoint series of :meth:`_short_moments`.

        Equation:
            [EQ-DIST-TN-003], [EQ-DIST-TN-006].

        References:
            [REF-JKB-1994, Chapter 13].

        """
        short = self._short_moments()
        if short is not None:
            return short[1]
        ratio_alpha = self._density_ratio(self._alpha)
        ratio_beta = self._density_ratio(self._beta)
        weighted_alpha = self._alpha * ratio_alpha if ratio_alpha > 0.0 else 0.0
        weighted_beta = self._beta * ratio_beta if ratio_beta > 0.0 else 0.0
        shift = ratio_alpha - ratio_beta
        bracket = 1.0 + weighted_alpha - weighted_beta - shift * shift
        return self.sigma * self.sigma * max(bracket, 0.0)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw reproducible truncated normal observations by inverse transform.

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of observations inside the truncation interval.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative.

        Equation:
            [EQ-DIST-INV-001], [EQ-DIST-TN-002].

        References:
            [REF-DEVROYE-1986, Chapter II].

        """
        checked_size = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        observations: list[float] = []
        for _ in range(checked_size):
            u = open_unit(generator)
            observations.append(self._invert(u, 1.0 - u))
        return tuple(observations)
