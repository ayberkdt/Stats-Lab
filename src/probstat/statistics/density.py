"""Gaussian kernel density estimation with rule-of-thumb bandwidths."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.distributions.normal import Normal
from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_non_negative_integer,
    require_probability,
)
from probstat.optimization import brent_root

from .order import quantile

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def kde_bandwidth(values: Iterable[float], *, rule: str = "silverman") -> float:
    """Return a rule-of-thumb bandwidth for a Gaussian kernel.

    ``silverman``: ``0.9 min(s, IQR/1.34) n^(-1/5)``, robust to heavy tails
    and mild multimodality; ``normal``: ``1.06 s n^(-1/5)``, optimal for
    normal data. A zero IQR falls back to ``s``.

    Args:
        values: At least two finite observations.
        rule: ``"silverman"`` or ``"normal"``.

    Returns:
        Positive bandwidth, or ``0`` for a constant sample.

    Raises:
        ValueError: For fewer than two observations or an unknown rule.

    Equation:
        [EQ-STAT-KDE-001].

    References:
        [REF-SILVERMAN-1986, Section 3.4.2].

    """
    sample = coerce_finite_sample(values, name="values")
    size = len(sample)
    if size < 2:
        raise ValueError("a bandwidth needs at least two observations")
    centre = math.fsum(sample) / size
    sd = math.sqrt(math.fsum((v - centre) ** 2 for v in sample) / (size - 1))
    if rule == "normal":
        return 1.06 * sd * math.pow(size, -0.2)
    if rule != "silverman":
        raise ValueError("rule must be 'silverman' or 'normal'")
    iqr = quantile(sample, 0.75) - quantile(sample, 0.25)
    spread = min(sd, iqr / 1.34) if iqr > 0.0 else sd
    return 0.9 * spread * math.pow(size, -0.2)


@dataclass(frozen=True, slots=True)
class KernelDensity:
    """Gaussian kernel density estimate ``f_h(x) = (1/nh) sum phi((x - x_i)/h)``.

    Args:
        sample: Observations in ascending order.
        bandwidth: Kernel standard deviation ``h > 0``.

    References:
        [REF-SILVERMAN-1986, Chapter 3].

    """

    sample: tuple[float, ...]
    bandwidth: float

    def pdf(self, x: float) -> float:
        """Return the estimated density at ``x``.

        Equation:
            [EQ-STAT-KDE-001].

        References:
            [REF-SILVERMAN-1986, Section 3.3].

        """
        h = self.bandwidth
        terms = (math.exp(-0.5 * ((x - v) / h) ** 2) for v in self.sample)
        return math.fsum(terms) * _INV_SQRT_2PI / (len(self.sample) * h)

    def cdf(self, x: float) -> float:
        """Return the estimated distribution function at ``x``.

        Equation:
            [EQ-STAT-KDE-001].

        References:
            [REF-SILVERMAN-1986, Section 3.3].

        """
        h = self.bandwidth
        # Phi(z) = erfc(-z / sqrt 2) / 2 keeps relative accuracy in the lower
        # tail, where 0.5 (1 + erf(z / sqrt 2)) cancels to zero.
        return math.fsum(
            0.5 * math.erfc(-(x - v) / (h * _SQRT2)) for v in self.sample
        ) / len(self.sample)

    def sf(self, x: float) -> float:
        """Return the estimated survival function ``1 - F_h(x)`` without cancellation.

        Equation:
            [EQ-STAT-KDE-001].

        References:
            [REF-SILVERMAN-1986, Section 3.3].

        """
        h = self.bandwidth
        return math.fsum(
            0.5 * math.erfc((x - v) / (h * _SQRT2)) for v in self.sample
        ) / len(self.sample)

    def quantile(self, probability: float) -> float:
        """Return ``x`` with ``F_h(x) = p`` by Brent's method.

        The root lies in ``[min x_i + h z_p, max x_i + h z_p]`` where
        ``z_p`` is the standard normal quantile, because ``F_h`` is an average
        of ``Phi((x - x_i)/h)``; in the upper half the residual ``q - S_h(x)``
        is used so upper-tail quantiles keep their accuracy.

        Args:
            probability: Probability in ``[0, 1]``.

        Returns:
            Quantile (``-inf``/``inf`` at ``0``/``1``).

        Raises:
            ValueError: If ``probability`` lies outside ``[0, 1]``.

        Equation:
            [EQ-STAT-KDE-002].

        References:
            [REF-SILVERMAN-1986, Section 3.3], [REF-BRENT-1973, Chapter 4].

        """
        p = require_probability(probability, name="probability")
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        standard = Normal()
        h = self.bandwidth
        if p <= 0.5:
            z = standard.quantile(p)

            def residual(t: float) -> float:
                """Return ``F_h(t) - p``.

                References:
                    [EQ-STAT-KDE-002]. No new equation.

                """
                return self.cdf(t) - p

        else:
            q = 1.0 - p
            z = standard.isf(q)

            def residual(t: float) -> float:
                """Return ``S_h(t) - q`` (decreasing in ``t``).

                References:
                    [EQ-STAT-KDE-002]. No new equation.

                """
                return self.sf(t) - q

        low = self.sample[0] + h * z
        high = self.sample[-1] + h * z
        if low == high:
            return low
        tolerance = 1e-13 * max(1.0, abs(low))
        return brent_root(residual, low, high, tolerance=tolerance).root

    @property
    def mean(self) -> float:
        """Return the mean of the estimate, the sample mean.

        Equation:
            [EQ-STAT-KDE-002].

        References:
            [REF-SILVERMAN-1986, Section 3.3].

        """
        return math.fsum(self.sample) / len(self.sample)

    @property
    def variance(self) -> float:
        """Return ``sum (x_i - xbar)^2 / n + h^2``, the variance of the estimate.

        Equation:
            [EQ-STAT-KDE-002].

        References:
            [REF-SILVERMAN-1986, Section 3.3].

        """
        centre = self.mean
        spread = math.fsum((v - centre) ** 2 for v in self.sample) / len(self.sample)
        return spread + self.bandwidth * self.bandwidth

    def draw(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Draw from the estimate: a resampled observation plus ``h Z``.

        (Named ``draw`` because ``sample`` is the data field.)

        Args:
            size: Number of draws, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Equation:
            [EQ-STAT-KDE-002].

        References:
            [REF-SILVERMAN-1986, Section 6.4].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(
            generator.choice(self.sample) + self.bandwidth * generator.gauss(0.0, 1.0)
            for _ in range(count)
        )

    def grid(self, size: int = 128, *, extend: float = 3.0) -> tuple[float, ...]:
        """Return ``size`` equally spaced points covering the sample +/- ``extend h``.

        Raises:
            ValueError: If ``size < 2`` or ``extend`` is negative.

        References:
            [REF-SILVERMAN-1986, Chapter 3]. No equation is evaluated.

        """
        if size < 2 or not extend >= 0.0:
            raise ValueError("size must be at least 2 and extend non-negative")
        low = self.sample[0] - extend * self.bandwidth
        high = self.sample[-1] + extend * self.bandwidth
        step = (high - low) / (size - 1)
        return tuple(low + i * step for i in range(size))


def gaussian_kde(
    values: Iterable[float],
    *,
    bandwidth: float | str = "silverman",
) -> KernelDensity:
    """Return a Gaussian kernel density estimate of a univariate sample.

    Args:
        values: At least two finite observations, not all equal.
        bandwidth: Positive kernel standard deviation, or a rule name accepted
            by :func:`kde_bandwidth`.

    Returns:
        :class:`KernelDensity` with ``pdf``, ``cdf``, ``sf``, ``quantile``,
        ``mean``, ``variance`` (so it can serve as a model input marginal),
        ``draw``, and ``grid``.

    Raises:
        ValueError: For fewer than two observations, a constant sample with a
            rule bandwidth, or a non-positive bandwidth.

    Equation:
        [EQ-STAT-KDE-001].

    References:
        [REF-SILVERMAN-1986, Chapter 3].

    """
    sample = tuple(sorted(coerce_finite_sample(values, name="values")))
    if len(sample) < 2:
        raise ValueError("a density estimate needs at least two observations")
    if isinstance(bandwidth, str):
        h = kde_bandwidth(sample, rule=bandwidth)
        if h == 0.0:
            raise ValueError("a rule bandwidth is zero for a constant sample")
    else:
        h = float(bandwidth)
        if not (math.isfinite(h) and h > 0.0):
            raise ValueError("bandwidth must be positive and finite")
    return KernelDensity(sample, h)
