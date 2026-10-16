"""Empirical distribution of a sample: step ECDF or continuous interpolation.

``Empirical(sample)`` with ``method="linear"`` (the default) is the
continuous law whose quantile function is the Hyndman-Fan type-7 sample
quantile ``Q(p) = x_(j+1) + (h - j)(x_(j+2) - x_(j+1))``, ``h = (n-1) p``,
``j = floor(h)``: its CDF interpolates linearly between the points
``(x_(i), (i-1)/(n-1))`` of the sorted sample, its density is
``1 / ((n-1)(x_(i+1) - x_(i)))`` on each gap, and its support is
``[x_(1), x_(n)]``. It is a continuous marginal usable as a model input.
Tied observations create atoms (vertical CDF segments); ``pdf`` then
reports the density of the continuous part only.

``method="step"`` is the classical ECDF ``F_n(x) = #{x_i <= x}/n`` with the
type-1 quantile ``Q(p) = x_(ceil(n p))``: a discrete law on the sample
points with the sample moments (``1/n`` variance). It has no density, so
``pdf`` raises; it is meant for bootstrap-style resampling, not as a
continuous model input.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)

from ._base import ContinuousDistribution
from ._continuous import open_unit

METHODS = ("linear", "step")


@dataclass(frozen=True, slots=True, init=False)
class Empirical(ContinuousDistribution):
    """The empirical law of a finite sample (see the module docstring).

    Args:
        sample: At least two finite observations (any order).
        method: ``"linear"`` (continuous, type-7 quantile) or ``"step"``
            (ECDF, type-1 quantile).

    References:
        [REF-HYNDMAN-FAN-1996], [REF-VAN-DER-VAART-1998, Chapter 19].

    """

    values: tuple[float, ...]
    method: str = "linear"
    _sum: float = field(default=0.0, repr=False, compare=False)

    def __init__(self, sample: Iterable[float], method: str = "linear") -> None:
        """Sort and validate the sample.

        Raises:
            TypeError: If an observation is not real.
            ValueError: If fewer than two observations are given, the method
                is unknown, or (linear) all observations are equal.

        References:
            Project validation policy. No equation is evaluated.

        """
        values = tuple(sorted(coerce_finite_sample(sample, name="sample")))
        if len(values) < 2:
            raise ValueError("an empirical law needs at least two observations")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if method == "linear" and values[0] == values[-1]:
            raise ValueError("a constant sample has no continuous interpolation")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "_sum", math.fsum(values))

    @property
    def size(self) -> int:
        """Return the sample size ``n``.

        References:
            Project validation policy. No equation is evaluated.

        """
        return len(self.values)

    def pdf(self, x: float) -> float:
        """Return ``1/((n-1)(x_(i+1) - x_(i)))`` on the gap containing ``x``.

        Raises:
            ValueError: For ``method="step"``, which has no density.

        Equation:
            [EQ-DIST-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        point = require_finite_real(x, name="x")
        if self.method == "step":
            raise ValueError("the step empirical law has no density")
        values = self.values
        if point < values[0] or point > values[-1]:
            return 0.0
        index = bisect.bisect_right(values, point) - 1
        index = min(index, len(values) - 2)
        gap = values[index + 1] - values[index]
        while gap == 0.0 and index > 0:
            index -= 1
            gap = values[index + 1] - values[index]
        return 1.0 / ((len(values) - 1) * gap) if gap > 0.0 else 0.0

    def log_pdf(self, x: float) -> float:
        """Return ``log pdf(x)`` (``-inf`` outside the support).

        Raises:
            ValueError: For ``method="step"``.

        Equation:
            [EQ-DIST-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        value = self.pdf(x)
        return math.log(value) if value > 0.0 else -math.inf

    def cdf(self, x: float) -> float:
        """Return the interpolated CDF (linear) or ``#{x_i <= x}/n`` (step).

        Equation:
            [EQ-DIST-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996], [REF-VAN-DER-VAART-1998, Chapter 19].

        """
        point = require_finite_real(x, name="x")
        values = self.values
        count = bisect.bisect_right(values, point)
        if self.method == "step":
            return count / len(values)
        if count == 0:
            return 0.0
        if count >= len(values):
            return 1.0
        low, high = values[count - 1], values[count]
        return ((count - 1) + (point - low) / (high - low)) / (len(values) - 1)

    def sf(self, x: float) -> float:
        """Return ``1 - cdf(x)`` formed from the complementary count.

        Equation:
            [EQ-DIST-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        point = require_finite_real(x, name="x")
        values = self.values
        count = bisect.bisect_right(values, point)
        if self.method == "step":
            return (len(values) - count) / len(values)
        if count == 0:
            return 1.0
        if count >= len(values):
            return 0.0
        low, high = values[count - 1], values[count]
        above = (len(values) - 1 - count) + (high - point) / (high - low)
        return above / (len(values) - 1)

    def quantile(self, probability: float) -> float:
        """Return the type-7 (linear) or type-1 (step) sample quantile.

        Equation:
            [EQ-DIST-EMPIRICAL-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        p = require_probability(probability, name="probability")
        values = self.values
        n = len(values)
        if self.method == "step":
            if p == 0.0:
                return values[0]
            return values[min(max(math.ceil(n * p) - 1, 0), n - 1)]
        h = (n - 1) * p
        j = min(int(math.floor(h)), n - 2)
        return values[j] + (h - j) * (values[j + 1] - values[j])

    @property
    def mean(self) -> float:
        """Return the mean of the interpolated (or step) law.

        Linear: ``(sum x_i - (x_(1) + x_(n))/2)/(n - 1)`` (uniform on each
        gap); step: the sample mean.

        Equation:
            [EQ-DIST-EMPIRICAL-002].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        values = self.values
        n = len(values)
        if self.method == "step":
            return self._sum / n
        return (self._sum - 0.5 * (values[0] + values[-1])) / (n - 1)

    @property
    def variance(self) -> float:
        """Return the variance of the interpolated (or step) law.

        Linear: ``sum_i (a^2 + a b + b^2)/(3 (n-1)) - mean^2`` over gaps
        ``[a, b]``, evaluated about the mean; step: ``sum (x_i - xbar)^2/n``.

        Equation:
            [EQ-DIST-EMPIRICAL-002].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        values = self.values
        mean = self.mean
        n = len(values)
        if self.method == "step":
            return math.fsum((v - mean) ** 2 for v in values) / n
        total = math.fsum(
            ((a - mean) ** 2 + (a - mean) * (b - mean) + (b - mean) ** 2) / 3.0
            for a, b in zip(values, values[1:], strict=False)
        )
        return total / (n - 1)

    def sample(
        self, size: int = 1, *, seed: int | RandomStream | None = None
    ) -> tuple[float, ...]:
        """Resample (step) or draw by inversion of the interpolated CDF (linear).

        Args:
            size: Number of observations, with ``size >= 0``.
            seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
                ``None`` for fresh operating-system entropy.

        Returns:
            Tuple of pseudo-random observations.

        Equation:
            [EQ-DIST-EMPIRICAL-001].

        References:
            [REF-EFRON-TIBSHIRANI-1993, Chapter 6].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        if self.method == "step":
            return tuple(generator.choice(self.values) for _ in range(count))
        return tuple(self.quantile(open_unit(generator)) for _ in range(count))
