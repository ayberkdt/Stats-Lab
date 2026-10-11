"""Gaussian kernel density estimation with rule-of-thumb bandwidths."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations import coerce_finite_sample

from .order import quantile

_STANDARD = NormalDist()
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
        return math.fsum(_STANDARD.cdf((x - v) / h) for v in self.sample) / len(
            self.sample
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
        :class:`KernelDensity` with ``pdf``, ``cdf``, and ``grid``.

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
