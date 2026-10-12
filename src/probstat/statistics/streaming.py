"""One-pass, constant-memory (streaming) accumulators for sample summaries.

Every accumulator here sees each observation once and stores a fixed number of
floats, independent of the stream length. The moment accumulators
(:class:`RunningMoments`, :class:`WeightedRunningMoments`,
:class:`RunningCovariance`) and :class:`ExceedanceCounter` are *mergeable*:
summaries of disjoint chunks combine exactly (up to rounding) into the summary
of the concatenated data, so chunks can be processed in parallel.

Numerical design: each moment accumulator works in coordinates shifted by its
first observation ``K`` (the "shifted data" device discussed by Chan, Golub,
and LeVeque). For data with a large common offset, such as ``1e9 + N(0, 1)``,
``x - K`` is computed exactly and the Welford/Pébay recurrences then run on
well-scaled numbers; the offset is added back only when the mean is reported.
Central moments are shift invariant, so no other correction is required.

:class:`P2Quantile` is a fixed-memory *approximation* (five markers) and is
not mergeable; :class:`ReservoirSampler` keeps an exact uniform random subset.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
)

from .order import quantile as _exact_quantile


def _check_ddof(ddof: int, count: float) -> int:
    """Validate a degrees-of-freedom offset against a sample size.

    Args:
        ddof: Delta degrees of freedom.
        count: Number of observations (or total frequency weight).

    Returns:
        The validated offset.

    Raises:
        TypeError: If ``ddof`` is not an integer.
        ValueError: If ``ddof`` is negative or not below ``count``.

    References:
        [REF-NIST-EDA, Section 1.3.5.2]. No equation is evaluated.

    """
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError("ddof must be an integer")
    if ddof < 0 or not ddof < count:
        raise ValueError("ddof must satisfy 0 <= ddof < number of observations")
    return ddof


class RunningMoments:
    """Mergeable one-pass count, mean, variance, skewness, and kurtosis.

    Stores the count ``n``, the mean, and the central sums
    ``M_p = sum (x_i - mean)^p`` for ``p = 2, 3, 4``. Single observations use
    Welford's recurrence extended to the third and fourth moments; two
    accumulators combine by the pairwise formulas of Chan, Golub, and LeVeque
    and of Pébay [EQ-STAT-STREAM-001], [EQ-STAT-STREAM-002],
    [EQ-STAT-STREAM-003].

    Args:
        values: Optional initial finite observations.

    References:
        [REF-WELFORD-1962], [REF-CHAN-GOLUB-LEVEQUE-1983], [REF-PEBAY-2008].

    """

    __slots__ = ("_count", "_shift", "_mean", "_m2", "_m3", "_m4")

    def __init__(self, values: Iterable[float] = ()) -> None:
        """Create an empty accumulator and optionally consume observations.

        Args:
            values: Optional initial finite observations.

        Raises:
            TypeError: If an observation is not real.
            ValueError: If an observation is not finite.

        References:
            [REF-WELFORD-1962]. No equation is evaluated.

        """
        self._count = 0
        self._shift = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._m3 = 0.0
        self._m4 = 0.0
        self.extend(values)

    def push(self, value: float) -> None:
        """Add one observation in constant time and memory.

        Args:
            value: Finite real observation.

        Raises:
            TypeError: If the value is not real.
            ValueError: If the value is not finite.

        Equation:
            [EQ-STAT-STREAM-001], [EQ-STAT-STREAM-003].

        References:
            [REF-WELFORD-1962], [REF-PEBAY-2008].

        """
        x = require_finite_real(value, name="value")
        if self._count == 0:
            self._shift = x
        n_a = self._count
        n = n_a + 1
        delta = (x - self._shift) - self._mean
        delta_n = delta / n
        term = delta * delta_n * n_a
        self._m4 += (
            term * delta_n * delta_n * (n * n - 3 * n + 3)
            + 6.0 * delta_n * delta_n * self._m2
            - 4.0 * delta_n * self._m3
        )
        self._m3 += term * delta_n * (n - 2) - 3.0 * delta_n * self._m2
        self._m2 += term
        self._mean += delta_n
        self._count = n

    def extend(self, values: Iterable[float]) -> None:
        """Add every observation of an iterable, one at a time.

        Args:
            values: Finite real observations; may be a generator.

        Raises:
            TypeError: If an observation is not real.
            ValueError: If an observation is not finite.

        References:
            [REF-WELFORD-1962]. Applies [EQ-STAT-STREAM-001] per value.

        """
        for value in values:
            self.push(value)

    def merge(self, other: RunningMoments) -> RunningMoments:
        """Return the accumulator of the concatenated data of two accumulators.

        Neither operand is modified. The result equals a single pass over both
        chunks up to rounding, so the order of chunk combination is irrelevant
        in exact arithmetic.

        Args:
            other: Accumulator summarizing a disjoint chunk.

        Returns:
            New combined accumulator.

        Raises:
            TypeError: If ``other`` is not a :class:`RunningMoments`.

        Equation:
            [EQ-STAT-STREAM-002], [EQ-STAT-STREAM-003].

        References:
            [REF-CHAN-GOLUB-LEVEQUE-1983], [REF-PEBAY-2008].

        """
        if not isinstance(other, RunningMoments):
            raise TypeError("other must be a RunningMoments")
        result = RunningMoments()
        if other._count == 0:
            result._copy_from(self)
            return result
        if self._count == 0:
            result._copy_from(other)
            return result
        n_a, n_b = self._count, other._count
        n = n_a + n_b
        other_mean = (other._shift - self._shift) + other._mean
        delta = other_mean - self._mean
        delta_n = delta / n
        m2a, m2b = self._m2, other._m2
        m3a, m3b = self._m3, other._m3
        result._count = n
        result._shift = self._shift
        result._mean = self._mean + delta_n * n_b
        result._m2 = m2a + m2b + delta * delta_n * n_a * n_b
        result._m3 = (
            m3a
            + m3b
            + delta * delta_n * delta_n * n_a * n_b * (n_a - n_b)
            + 3.0 * delta_n * (n_a * m2b - n_b * m2a)
        )
        result._m4 = (
            self._m4
            + other._m4
            + delta
            * delta_n
            * delta_n
            * delta_n
            * n_a
            * n_b
            * (n_a * n_a - n_a * n_b + n_b * n_b)
            + 6.0 * delta_n * delta_n * (n_a * n_a * m2b + n_b * n_b * m2a)
            + 4.0 * delta_n * (n_a * m3b - n_b * m3a)
        )
        return result

    def _copy_from(self, other: RunningMoments) -> None:
        """Copy all state from another accumulator into this one.

        Args:
            other: Source accumulator.

        References:
            [REF-WELFORD-1962]. No equation is evaluated.

        """
        self._count = other._count
        self._shift = other._shift
        self._mean = other._mean
        self._m2 = other._m2
        self._m3 = other._m3
        self._m4 = other._m4

    @property
    def count(self) -> int:
        """Return the number of observations consumed.

        Returns:
            Non-negative integer count.

        References:
            [REF-WELFORD-1962]. No equation is evaluated.

        """
        return self._count

    def _require(self, minimum: int) -> None:
        """Raise when fewer than ``minimum`` observations were consumed.

        Args:
            minimum: Required number of observations.

        Raises:
            ValueError: If the count is below ``minimum``.

        References:
            Project validation policy. No equation is evaluated.

        """
        if self._count < minimum:
            raise ValueError(
                f"at least {minimum} observation(s) required, have {self._count}"
            )

    @property
    def mean(self) -> float:
        """Return the arithmetic mean of the consumed observations.

        Returns:
            Sample mean.

        Raises:
            ValueError: If no observation was consumed.

        Equation:
            [EQ-STAT-STREAM-001].

        References:
            [REF-WELFORD-1962].

        """
        self._require(1)
        return self._shift + self._mean

    @property
    def sum_of_squared_deviations(self) -> float:
        """Return the central sum ``M_2 = sum (x_i - mean)^2``.

        Returns:
            Non-negative second central sum.

        References:
            [REF-WELFORD-1962], [EQ-STAT-STREAM-001].

        """
        return max(self._m2, 0.0)

    def variance(self, *, ddof: int = 0) -> float:
        """Return ``M_2 / (n - ddof)`` as in :func:`statistics.variance`.

        Args:
            ddof: Delta degrees of freedom; ``1`` gives the unbiased estimator.

        Returns:
            Non-negative variance.

        Raises:
            TypeError: If ``ddof`` is not an integer.
            ValueError: If ``ddof`` is outside ``[0, n)``.

        Equation:
            [EQ-STAT-002], [EQ-STAT-STREAM-001].

        References:
            [REF-WELFORD-1962], [REF-NIST-EDA, Section 1.3.5.2].

        """
        self._require(1)
        checked = _check_ddof(ddof, self._count)
        return self.sum_of_squared_deviations / (self._count - checked)

    def standard_deviation(self, *, ddof: int = 0) -> float:
        """Return the square root of :meth:`variance`.

        Args:
            ddof: Delta degrees of freedom.

        Returns:
            Non-negative standard deviation.

        Equation:
            [EQ-STAT-003].

        References:
            [REF-NIST-EDA, Section 1.3.5.2].

        """
        return math.sqrt(self.variance(ddof=ddof))

    def standard_error(self) -> float:
        """Return the Monte Carlo standard error ``s / sqrt(n)`` with ``ddof=1``.

        Returns:
            Estimated standard deviation of the sample mean.

        Raises:
            ValueError: If fewer than two observations were consumed.

        Equation:
            [EQ-MC-SE-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapters 3-4].

        """
        self._require(2)
        return math.sqrt(self.variance(ddof=1) / self._count)

    def relative_standard_error(self) -> float:
        """Return ``standard_error / |mean|`` (infinite when the mean is zero).

        Returns:
            Non-negative relative standard error or ``math.inf``.

        Equation:
            [EQ-STAT-STREAM-007].

        References:
            [REF-ROBERT-CASELLA-2004, Chapters 3-4].

        """
        center = abs(self.mean)
        error = self.standard_error()
        if center == 0.0:
            # A zero mean makes a relative error meaningless, even when every
            # observation is zero.
            return math.inf
        return error / center

    def skewness(self) -> float:
        """Return the moment skewness ``g1 = sqrt(n) M_3 / M_2^(3/2)``.

        Matches :func:`probstat.statistics.skewness` (biased moment ratio).

        Returns:
            Sample skewness.

        Raises:
            ValueError: If fewer than two values or zero variance.

        Equation:
            [EQ-STAT-SHAPE-001], [EQ-STAT-STREAM-003].

        References:
            [REF-JOANES-GILL-1998], [REF-PEBAY-2008].

        """
        self._require(2)
        if self._m2 <= 0.0:
            raise ValueError("skewness is undefined for zero variance")
        return float(math.sqrt(self._count) * self._m3 / self._m2**1.5)

    def excess_kurtosis(self) -> float:
        """Return the moment excess kurtosis ``g2 = n M_4 / M_2^2 - 3``.

        Matches :func:`probstat.statistics.excess_kurtosis`.

        Returns:
            Sample excess kurtosis.

        Raises:
            ValueError: If fewer than two values or zero variance.

        Equation:
            [EQ-STAT-SHAPE-001], [EQ-STAT-STREAM-003].

        References:
            [REF-JOANES-GILL-1998], [REF-PEBAY-2008].

        """
        self._require(2)
        if self._m2 <= 0.0:
            raise ValueError("kurtosis is undefined for zero variance")
        return self._count * self._m4 / (self._m2 * self._m2) - 3.0

    def __repr__(self) -> str:
        """Return a compact description of the accumulator state.

        Returns:
            Readable representation.

        References:
            [REF-PYTHON-DATAMODEL]. No equation is evaluated.

        """
        if self._count == 0:
            return "RunningMoments(count=0)"
        return f"RunningMoments(count={self._count}, mean={self.mean!r})"


class WeightedRunningMoments:
    """Mergeable one-pass weighted mean and variance (West's algorithm).

    Stores the weight total ``W = sum w_i``, the squared-weight total
    ``V = sum w_i^2``, the weighted mean, and
    ``S = sum w_i (x_i - mean)^2`` [EQ-STAT-STREAM-004].

    Args:
        pairs: Optional initial ``(value, weight)`` pairs.

    References:
        [REF-WEST-1979], [REF-CHAN-GOLUB-LEVEQUE-1983].

    """

    __slots__ = ("_count", "_shift", "_weight", "_weight_sq", "_mean", "_s")

    def __init__(self, pairs: Iterable[tuple[float, float]] = ()) -> None:
        """Create an empty accumulator and optionally consume weighted values.

        Args:
            pairs: Optional initial ``(value, weight)`` pairs.

        References:
            [REF-WEST-1979]. No equation is evaluated.

        """
        self._count = 0
        self._shift = 0.0
        self._weight = 0.0
        self._weight_sq = 0.0
        self._mean = 0.0
        self._s = 0.0
        for value, weight in pairs:
            self.push(value, weight)

    def push(self, value: float, weight: float = 1.0) -> None:
        """Add one observation with a non-negative weight.

        A zero weight increments :attr:`count` but leaves every moment unchanged.

        Args:
            value: Finite real observation.
            weight: Finite non-negative weight.

        Raises:
            TypeError: If an argument is not real.
            ValueError: If an argument is non-finite or the weight is negative.

        Equation:
            [EQ-STAT-STREAM-004].

        References:
            [REF-WEST-1979].

        """
        x = require_finite_real(value, name="value")
        w = require_non_negative_finite_real(weight, name="weight")
        self._count += 1
        if w == 0.0:
            return
        if self._weight == 0.0:
            self._shift = x
        old_weight = self._weight
        new_weight = old_weight + w
        delta = (x - self._shift) - self._mean
        r = delta * w / new_weight
        self._mean += r
        self._s += old_weight * delta * r
        self._weight = new_weight
        self._weight_sq += w * w

    def merge(self, other: WeightedRunningMoments) -> WeightedRunningMoments:
        """Return the accumulator of the concatenated weighted data.

        Args:
            other: Accumulator summarizing a disjoint chunk.

        Returns:
            New combined accumulator; operands are unchanged.

        Raises:
            TypeError: If ``other`` is not a :class:`WeightedRunningMoments`.

        Equation:
            [EQ-STAT-STREAM-004].

        References:
            [REF-WEST-1979], [REF-CHAN-GOLUB-LEVEQUE-1983].

        """
        if not isinstance(other, WeightedRunningMoments):
            raise TypeError("other must be a WeightedRunningMoments")
        result = WeightedRunningMoments()
        source = None
        if other._weight == 0.0:
            source = self
        elif self._weight == 0.0:
            source = other
        if source is not None:
            result._shift = source._shift
            result._weight = source._weight
            result._weight_sq = source._weight_sq
            result._mean = source._mean
            result._s = source._s
            result._count = self._count + other._count
            return result
        w_a, w_b = self._weight, other._weight
        total = w_a + w_b
        delta = ((other._shift - self._shift) + other._mean) - self._mean
        result._count = self._count + other._count
        result._shift = self._shift
        result._weight = total
        result._weight_sq = self._weight_sq + other._weight_sq
        result._mean = self._mean + delta * w_b / total
        result._s = self._s + other._s + delta * delta * w_a * w_b / total
        return result

    @property
    def count(self) -> int:
        """Return the number of pushed observations, zero weights included.

        Returns:
            Non-negative integer count.

        References:
            [REF-WEST-1979]. No equation is evaluated.

        """
        return self._count

    @property
    def total_weight(self) -> float:
        """Return ``W = sum w_i``.

        Returns:
            Non-negative weight total.

        References:
            [REF-WEST-1979], [EQ-STAT-STREAM-004].

        """
        return self._weight

    @property
    def effective_sample_size(self) -> float:
        """Return the weight-based effective sample size ``W^2 / sum w_i^2``.

        Returns:
            Effective number of equally weighted observations.

        Raises:
            ValueError: If the total weight is zero.

        Equation:
            [EQ-STAT-STREAM-004].

        References:
            [REF-WEST-1979].

        """
        if self._weight == 0.0:
            raise ValueError("total weight must be positive")
        return self._weight * self._weight / self._weight_sq

    @property
    def mean(self) -> float:
        """Return the weighted mean ``sum w_i x_i / W``.

        Returns:
            Weighted mean.

        Raises:
            ValueError: If the total weight is zero.

        Equation:
            [EQ-STAT-STREAM-004].

        References:
            [REF-WEST-1979].

        """
        if self._weight == 0.0:
            raise ValueError("total weight must be positive")
        return self._shift + self._mean

    def variance(self, *, kind: str = "population") -> float:
        """Return a weighted variance under an explicit weight convention.

        ``"population"`` divides ``S`` by ``W``; ``"frequency"`` treats weights
        as repeat counts and divides by ``W - 1``; ``"reliability"`` treats
        weights as relative precisions and divides by ``W - sum w^2 / W``,
        which is unbiased for independent values with a common variance.

        Args:
            kind: ``"population"``, ``"frequency"``, or ``"reliability"``.

        Returns:
            Non-negative weighted variance.

        Raises:
            ValueError: If ``kind`` is unknown or the denominator is not positive.

        Equation:
            [EQ-STAT-STREAM-004].

        References:
            [REF-WEST-1979].

        """
        if self._weight == 0.0:
            raise ValueError("total weight must be positive")
        if kind == "population":
            denominator = self._weight
        elif kind == "frequency":
            denominator = self._weight - 1.0
        elif kind == "reliability":
            denominator = self._weight - self._weight_sq / self._weight
        else:
            raise ValueError("kind must be 'population', 'frequency', or 'reliability'")
        if denominator <= 0.0:
            raise ValueError(f"{kind} variance denominator is not positive")
        return max(self._s, 0.0) / denominator


class RunningCovariance:
    """Mergeable one-pass mean vector, covariance, and correlation matrix.

    Stores the count, the mean vector, and the co-moment matrix
    ``C_jk = sum (x_ij - mean_j)(x_ik - mean_k)`` [EQ-STAT-STREAM-002].
    Use ``dimension=2`` and push ``(x, y)`` pairs for a single covariance.

    Args:
        dimension: Positive vector length.

    References:
        [REF-WELFORD-1962], [REF-CHAN-GOLUB-LEVEQUE-1983], [REF-PEBAY-2008].

    """

    __slots__ = ("_dimension", "_count", "_shift", "_mean", "_comoment")

    def __init__(self, dimension: int) -> None:
        """Create an empty accumulator for vectors of a fixed length.

        Args:
            dimension: Positive vector length.

        Raises:
            TypeError: If ``dimension`` is not an integer.
            ValueError: If ``dimension`` is not positive.

        References:
            [REF-PEBAY-2008]. No equation is evaluated.

        """
        checked = require_non_negative_integer(dimension, name="dimension")
        if checked == 0:
            raise ValueError("dimension must be positive")
        self._dimension = checked
        self._count = 0
        self._shift = [0.0] * checked
        self._mean = [0.0] * checked
        self._comoment = [[0.0] * checked for _ in range(checked)]

    def push(self, vector: Sequence[float]) -> None:
        """Add one observation vector.

        Args:
            vector: Finite real values of length :attr:`dimension`.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the length is wrong or an entry is not finite.

        Equation:
            [EQ-STAT-STREAM-001], [EQ-STAT-STREAM-002].

        References:
            [REF-WELFORD-1962], [REF-PEBAY-2008].

        """
        if len(vector) != self._dimension:
            raise ValueError(f"vector must have length {self._dimension}")
        x = [
            require_finite_real(value, name=f"vector[{index}]")
            for index, value in enumerate(vector)
        ]
        if self._count == 0:
            self._shift = list(x)
        n = self._count + 1
        delta = [x[j] - self._shift[j] - self._mean[j] for j in range(len(x))]
        factor = (n - 1) / n
        comoment = self._comoment
        for j, d_j in enumerate(delta):
            self._mean[j] += d_j / n
            for k in range(j, len(delta)):
                increment = factor * (d_j * delta[k])
                comoment[j][k] += increment
                if k != j:
                    comoment[k][j] += increment
        self._count = n

    def extend(self, vectors: Iterable[Sequence[float]]) -> None:
        """Add every vector of an iterable.

        Args:
            vectors: Observation vectors; may be a generator.

        References:
            [REF-WELFORD-1962]. Applies [EQ-STAT-STREAM-002] per vector.

        """
        for vector in vectors:
            self.push(vector)

    def merge(self, other: RunningCovariance) -> RunningCovariance:
        """Return the accumulator of the concatenated vector data.

        Args:
            other: Accumulator of the same dimension for a disjoint chunk.

        Returns:
            New combined accumulator; operands are unchanged.

        Raises:
            TypeError: If ``other`` is not a :class:`RunningCovariance`.
            ValueError: If the dimensions differ.

        Equation:
            [EQ-STAT-STREAM-002].

        References:
            [REF-CHAN-GOLUB-LEVEQUE-1983], [REF-PEBAY-2008].

        """
        if not isinstance(other, RunningCovariance):
            raise TypeError("other must be a RunningCovariance")
        if other._dimension != self._dimension:
            raise ValueError("dimensions differ")
        result = RunningCovariance(self._dimension)
        source = None
        if other._count == 0:
            source = self
        elif self._count == 0:
            source = other
        if source is not None:
            result._count = source._count
            result._shift = list(source._shift)
            result._mean = list(source._mean)
            result._comoment = [list(row) for row in source._comoment]
            return result
        n_a, n_b = self._count, other._count
        n = n_a + n_b
        delta = [
            (other._shift[j] - self._shift[j]) + other._mean[j] - self._mean[j]
            for j in range(self._dimension)
        ]
        weight = n_a * n_b / n
        result._count = n
        result._shift = list(self._shift)
        result._mean = [
            self._mean[j] + delta[j] * n_b / n for j in range(self._dimension)
        ]
        result._comoment = [
            [
                self._comoment[j][k]
                + other._comoment[j][k]
                + weight * (delta[j] * delta[k])
                for k in range(self._dimension)
            ]
            for j in range(self._dimension)
        ]
        return result

    @property
    def dimension(self) -> int:
        """Return the vector length.

        Returns:
            Positive integer dimension.

        References:
            [REF-PEBAY-2008]. No equation is evaluated.

        """
        return self._dimension

    @property
    def count(self) -> int:
        """Return the number of vectors consumed.

        Returns:
            Non-negative integer count.

        References:
            [REF-PEBAY-2008]. No equation is evaluated.

        """
        return self._count

    @property
    def mean(self) -> tuple[float, ...]:
        """Return the componentwise mean vector.

        Returns:
            Tuple of means.

        Raises:
            ValueError: If no vector was consumed.

        Equation:
            [EQ-STAT-STREAM-001].

        References:
            [REF-WELFORD-1962].

        """
        if self._count == 0:
            raise ValueError("at least 1 observation required, have 0")
        return tuple(s + m for s, m in zip(self._shift, self._mean, strict=True))

    def covariance(self, i: int = 0, j: int = 1, *, ddof: int = 0) -> float:
        """Return the covariance of components ``i`` and ``j``.

        The default ``ddof=0`` matches :func:`probstat.statistics.covariance`.

        Args:
            i: First component index.
            j: Second component index.
            ddof: Delta degrees of freedom.

        Returns:
            ``C_ij / (n - ddof)``.

        Raises:
            ValueError: If ``ddof`` is invalid for the count.
            IndexError: If an index is out of range.

        Equation:
            [EQ-STAT-006], [EQ-STAT-STREAM-002].

        References:
            [REF-PEBAY-2008], [REF-NIST-EDA, Section 1.3.5.1].

        """
        if self._count == 0:
            raise ValueError("at least 1 observation required, have 0")
        checked = _check_ddof(ddof, self._count)
        value = self._comoment[i][j]
        if i == j:
            value = max(value, 0.0)
        return value / (self._count - checked)

    def correlation(self, i: int = 0, j: int = 1) -> float:
        """Return the Pearson correlation of components ``i`` and ``j``.

        Args:
            i: First component index.
            j: Second component index.

        Returns:
            Correlation clipped to ``[-1, 1]``.

        Raises:
            ValueError: If a component has zero variance.

        Equation:
            [EQ-STAT-007].

        References:
            [REF-PEBAY-2008], [REF-NIST-EDA, Section 1.3.5.1].

        """
        if self._count == 0:
            raise ValueError("at least 1 observation required, have 0")
        c_ii = self._comoment[i][i]
        c_jj = self._comoment[j][j]
        if c_ii <= 0.0 or c_jj <= 0.0:
            raise ValueError("correlation is undefined for zero variance")
        value = self._comoment[i][j] / math.sqrt(c_ii * c_jj)
        return min(1.0, max(-1.0, value))

    def covariance_matrix(self, *, ddof: int = 0) -> tuple[tuple[float, ...], ...]:
        """Return the full covariance matrix.

        Args:
            ddof: Delta degrees of freedom.

        Returns:
            Symmetric matrix as a tuple of row tuples.

        Equation:
            [EQ-STAT-STREAM-002].

        References:
            [REF-PEBAY-2008].

        """
        return tuple(
            tuple(self.covariance(i, j, ddof=ddof) for j in range(self._dimension))
            for i in range(self._dimension)
        )

    def correlation_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return the full Pearson correlation matrix.

        Returns:
            Symmetric matrix with unit diagonal.

        Equation:
            [EQ-STAT-007].

        References:
            [REF-PEBAY-2008].

        """
        return tuple(
            tuple(
                1.0 if i == j else self.correlation(i, j)
                for j in range(self._dimension)
            )
            for i in range(self._dimension)
        )


class P2Quantile:
    """Fixed-memory streaming estimate of one quantile (the P² algorithm).

    Five markers track the minimum, the ``p/2``, ``p``, and ``(1+p)/2``
    quantiles, and the maximum. After each observation the three interior
    markers are moved by at most one rank position, using a piecewise-parabolic
    (P²) height prediction, falling back to linear interpolation when the
    parabola would break marker monotonicity [EQ-STAT-STREAM-005].

    Accuracy limits: the estimate is a heuristic, not a sample quantile. It
    has no worst-case error bound, depends on arrival order, and is least
    accurate for extreme ``p`` (few observations fall beyond the marker),
    heavy tails, discrete data with large atoms, and non-stationary streams.
    For ``n <= 5`` the exact type-7 sample quantile is returned. P² summaries
    cannot be merged; use :class:`ReservoirSampler` when chunks must combine.

    Args:
        probability: Quantile level strictly between zero and one.

    References:
        [REF-JAIN-CHLAMTAC-1985].

    """

    __slots__ = ("_p", "_count", "_heights", "_positions", "_desired", "_steps")

    def __init__(self, probability: float) -> None:
        """Create an empty estimator for one quantile level.

        Args:
            probability: Quantile level strictly between zero and one.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If ``probability`` is not in ``(0, 1)``.

        References:
            [REF-JAIN-CHLAMTAC-1985]. No equation is evaluated.

        """
        p = require_finite_real(probability, name="probability")
        if not 0.0 < p < 1.0:
            raise ValueError("probability must be strictly between 0 and 1")
        self._p = p
        self._count = 0
        self._heights: list[float] = []
        self._positions = [1, 2, 3, 4, 5]
        self._desired = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0]
        self._steps = (0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0)

    @property
    def probability(self) -> float:
        """Return the target quantile level.

        Returns:
            Level in ``(0, 1)``.

        References:
            [REF-JAIN-CHLAMTAC-1985]. No equation is evaluated.

        """
        return self._p

    @property
    def count(self) -> int:
        """Return the number of observations consumed.

        Returns:
            Non-negative integer count.

        References:
            [REF-JAIN-CHLAMTAC-1985]. No equation is evaluated.

        """
        return self._count

    def push(self, value: float) -> None:
        """Add one observation and adjust the markers.

        Args:
            value: Finite real observation.

        Raises:
            TypeError: If the value is not real.
            ValueError: If the value is not finite.

        Equation:
            [EQ-STAT-STREAM-005].

        References:
            [REF-JAIN-CHLAMTAC-1985].

        """
        x = require_finite_real(value, name="value")
        self._count += 1
        q = self._heights
        if self._count <= 5:
            q.append(x)
            q.sort()
            return
        n = self._positions
        if x < q[0]:
            q[0] = x
            cell = 0
        elif x >= q[4]:
            q[4] = max(q[4], x)
            cell = 3
        else:
            cell = 0
            while x >= q[cell + 1]:
                cell += 1
        for index in range(cell + 1, 5):
            n[index] += 1
        desired = self._desired
        for index in range(5):
            desired[index] += self._steps[index]
        for i in (1, 2, 3):
            d = desired[i] - n[i]
            if (d >= 1.0 and n[i + 1] - n[i] > 1) or (
                d <= -1.0 and n[i - 1] - n[i] < -1
            ):
                step = 1 if d > 0.0 else -1
                candidate = self._parabolic(i, step)
                if not q[i - 1] < candidate < q[i + 1]:
                    candidate = q[i] + step * (q[i + step] - q[i]) / (
                        n[i + step] - n[i]
                    )
                q[i] = candidate
                n[i] += step

    def _parabolic(self, i: int, step: int) -> float:
        """Return the piecewise-parabolic prediction for marker ``i``.

        Args:
            i: Interior marker index (1, 2, or 3).
            step: Rank move, ``+1`` or ``-1``.

        Returns:
            Predicted marker height.

        Equation:
            [EQ-STAT-STREAM-005].

        References:
            [REF-JAIN-CHLAMTAC-1985].

        """
        q = self._heights
        n = self._positions
        left = n[i] - n[i - 1]
        right = n[i + 1] - n[i]
        return q[i] + step / (n[i + 1] - n[i - 1]) * (
            (left + step) * (q[i + 1] - q[i]) / right
            + (right - step) * (q[i] - q[i - 1]) / left
        )

    def extend(self, values: Iterable[float]) -> None:
        """Add every observation of an iterable.

        Args:
            values: Finite real observations; may be a generator.

        References:
            [REF-JAIN-CHLAMTAC-1985]. Applies [EQ-STAT-STREAM-005] per value.

        """
        for value in values:
            self.push(value)

    @property
    def estimate(self) -> float:
        """Return the current quantile estimate.

        Returns:
            Middle-marker height, or the exact type-7 quantile when ``n <= 5``.

        Raises:
            ValueError: If no observation was consumed.

        Equation:
            [EQ-STAT-STREAM-005].

        References:
            [REF-JAIN-CHLAMTAC-1985], [REF-HYNDMAN-FAN-1996].

        """
        if self._count == 0:
            raise ValueError("at least 1 observation required, have 0")
        if self._count <= 5:
            return _exact_quantile(self._heights, self._p)
        return self._heights[2]


class ReservoirSampler:
    """Exact uniform random subset of fixed size from a stream (Algorithm R).

    After ``n`` observations every subset of size ``min(n, capacity)`` is
    equally likely, so each observation is retained with probability
    ``capacity / n`` [EQ-STAT-STREAM-006]. The retained values support any
    exact summary (quantiles, histograms) of a uniform subsample.

    Args:
        capacity: Positive reservoir size.
        seed: Integer seed, :class:`RandomStream`, or ``None`` for fresh entropy.

    References:
        [REF-VITTER-1985], [REF-STEELE-LEA-FLOOD-2014].

    """

    __slots__ = ("_capacity", "_count", "_items", "_rng")

    def __init__(
        self, capacity: int, *, seed: int | RandomStream | None = None
    ) -> None:
        """Create an empty reservoir.

        Args:
            capacity: Positive reservoir size.
            seed: Integer seed, :class:`RandomStream`, or ``None``.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If ``capacity`` is not positive.

        References:
            [REF-VITTER-1985]. No equation is evaluated.

        """
        checked = require_non_negative_integer(capacity, name="capacity")
        if checked == 0:
            raise ValueError("capacity must be positive")
        self._capacity = checked
        self._count = 0
        self._items: list[float] = []
        self._rng = resolve_stream(seed).generator()

    @property
    def capacity(self) -> int:
        """Return the reservoir size.

        Returns:
            Positive integer capacity.

        References:
            [REF-VITTER-1985]. No equation is evaluated.

        """
        return self._capacity

    @property
    def count(self) -> int:
        """Return the number of observations seen.

        Returns:
            Non-negative integer count.

        References:
            [REF-VITTER-1985]. No equation is evaluated.

        """
        return self._count

    def push(self, value: float) -> None:
        """Offer one observation to the reservoir.

        Args:
            value: Finite real observation.

        Raises:
            TypeError: If the value is not real.
            ValueError: If the value is not finite.

        Equation:
            [EQ-STAT-STREAM-006].

        References:
            [REF-VITTER-1985].

        """
        x = require_finite_real(value, name="value")
        if self._count < self._capacity:
            self._items.append(x)
        else:
            slot = self._rng.randrange(self._count + 1)
            if slot < self._capacity:
                self._items[slot] = x
        self._count += 1

    def extend(self, values: Iterable[float]) -> None:
        """Offer every observation of an iterable.

        Args:
            values: Finite real observations; may be a generator.

        References:
            [REF-VITTER-1985]. Applies [EQ-STAT-STREAM-006] per value.

        """
        for value in values:
            self.push(value)

    @property
    def sample(self) -> tuple[float, ...]:
        """Return the retained values in reservoir-slot order.

        Returns:
            Tuple of at most :attr:`capacity` values.

        References:
            [REF-VITTER-1985]. No equation is evaluated.

        """
        return tuple(self._items)


class ExceedanceCounter:
    """Mergeable streaming event or exceedance proportion with a Wilson interval.

    Without a threshold, :meth:`push` takes exact Boolean indicators. With a
    threshold ``t``, :meth:`push` takes real values and counts the strict
    exceedance ``x > t``.

    Args:
        threshold: Optional finite threshold.

    References:
        [REF-WILSON-1927], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """

    __slots__ = ("_threshold", "_trials", "_events")

    def __init__(self, threshold: float | None = None) -> None:
        """Create an empty counter.

        Args:
            threshold: Optional finite threshold.

        Raises:
            TypeError: If the threshold is not real.
            ValueError: If the threshold is not finite.

        References:
            [REF-WILSON-1927]. No equation is evaluated.

        """
        self._threshold = (
            None
            if threshold is None
            else require_finite_real(threshold, name="threshold")
        )
        self._trials = 0
        self._events = 0

    def push(self, value: bool | float) -> None:
        """Record one indicator, or one value compared with the threshold.

        Args:
            value: Boolean indicator (no threshold) or finite real value.

        Raises:
            TypeError: If an indicator is not Boolean or a value is not real.
            ValueError: If a value is not finite.

        Equation:
            [EQ-MC-PROB-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3].

        """
        if self._threshold is None:
            if not isinstance(value, bool):
                raise TypeError("indicators must be Boolean when no threshold is set")
            event = value
        else:
            event = require_finite_real(value, name="value") > self._threshold
        self._trials += 1
        self._events += int(event)

    def extend(self, values: Iterable[bool | float]) -> None:
        """Record every item of an iterable.

        Args:
            values: Indicators or values; may be a generator.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3]. No equation is evaluated.

        """
        for value in values:
            self.push(value)

    def merge(self, other: ExceedanceCounter) -> ExceedanceCounter:
        """Return the counter of the concatenated streams.

        Args:
            other: Counter with the same threshold.

        Returns:
            New combined counter.

        Raises:
            TypeError: If ``other`` is not an :class:`ExceedanceCounter`.
            ValueError: If the thresholds differ.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3]. No equation is evaluated.

        """
        if not isinstance(other, ExceedanceCounter):
            raise TypeError("other must be an ExceedanceCounter")
        if other._threshold != self._threshold:
            raise ValueError("thresholds differ")
        result = ExceedanceCounter(self._threshold)
        result._trials = self._trials + other._trials
        result._events = self._events + other._events
        return result

    @property
    def trials(self) -> int:
        """Return the number of recorded items.

        Returns:
            Non-negative integer count.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3]. No equation is evaluated.

        """
        return self._trials

    @property
    def events(self) -> int:
        """Return the number of events (exceedances).

        Returns:
            Non-negative integer count.

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3]. No equation is evaluated.

        """
        return self._events

    @property
    def proportion(self) -> float:
        """Return the empirical event proportion.

        Returns:
            ``events / trials``.

        Raises:
            ValueError: If nothing was recorded.

        Equation:
            [EQ-MC-PROB-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3].

        """
        if self._trials == 0:
            raise ValueError("at least one trial required")
        return self._events / self._trials

    def standard_error(self) -> float:
        """Return the plug-in Bernoulli standard error ``sqrt(p (1-p) / n)``.

        Returns:
            Non-negative standard error.

        Equation:
            [EQ-MC-PROB-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3].

        """
        p = self.proportion
        return math.sqrt(p * (1.0 - p) / self._trials)

    def wilson_interval(self, *, confidence: float = 0.95) -> tuple[float, float]:
        """Return the Wilson score interval for the event probability.

        The formula is the one of
        :func:`probstat.inference.wilson_score_interval`, evaluated locally
        because :mod:`probstat.statistics` sits below :mod:`probstat.inference`.

        Args:
            confidence: Nominal two-sided level strictly between zero and one.

        Returns:
            ``(lower, upper)`` clipped to ``[0, 1]``.

        Raises:
            ValueError: If nothing was recorded or ``confidence`` is invalid.

        Equation:
            [EQ-INF-CI-WILSON-001].

        References:
            [REF-WILSON-1927].

        """
        level = require_finite_real(confidence, name="confidence")
        if not 0.0 < level < 1.0:
            raise ValueError("confidence must be strictly between zero and one")
        n = self._trials
        estimate = self.proportion
        critical = NormalDist().inv_cdf(0.5 + 0.5 * level)
        squared = critical**2
        denominator = 1.0 + squared / n
        center = (estimate + squared / (2.0 * n)) / denominator
        half_width = (
            critical
            * math.sqrt(estimate * (1.0 - estimate) / n + squared / (4.0 * n**2))
            / denominator
        )
        lower = 0.0 if self._events == 0 else max(0.0, center - half_width)
        upper = 1.0 if self._events == n else min(1.0, center + half_width)
        return lower, upper


@dataclass(frozen=True, slots=True)
class StreamingRunResult:
    """Outcome of :func:`run_until`, a relative-error-targeted running mean.

    Args:
        mean: Running mean when sampling stopped.
        standard_error: Monte Carlo standard error ``s / sqrt(n)``.
        relative_error: ``standard_error / |mean|`` (``inf`` for a zero mean).
        sample_size: Number of samples drawn (the work count).
        converged: Whether ``relative_error <= target_relative_error``.
        target_relative_error: Requested relative standard error.
        moments: The accumulator, for further summaries or merging.
        trace: ``(n, mean, standard_error)`` at every check.

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4].

    """

    mean: float
    standard_error: float
    relative_error: float
    sample_size: int
    converged: bool
    target_relative_error: float
    moments: RunningMoments
    trace: tuple[tuple[int, float, float], ...]


def run_until(
    source: Iterable[float] | Callable[[], float],
    *,
    target_relative_error: float,
    max_samples: int,
    min_samples: int = 100,
    check_every: int = 100,
) -> StreamingRunResult:
    """Draw samples into a running mean until a relative error is reached.

    After at least ``min_samples`` draws, and then every ``check_every``
    draws, the relative standard error ``s / (sqrt(n) |mean|)`` is compared
    with the target; sampling stops at the first check that meets it, when
    ``max_samples`` is reached, or when an iterable source is exhausted.
    Memory is constant: samples are not stored.

    Data-dependent stopping makes the reported error slightly optimistic and
    the confidence of a normal interval slightly below nominal; a larger
    ``min_samples`` and a sparser ``check_every`` reduce this effect. A mean
    near zero makes a relative target unreachable (``converged`` is then
    ``False``); target an absolute error through :class:`RunningMoments`
    instead.

    Args:
        source: Zero-argument sampler or iterable of finite real samples.
        target_relative_error: Positive target for ``SE / |mean|``.
        max_samples: Maximum number of draws, at least ``min_samples``.
        min_samples: Draws before the first check, at least two.
        check_every: Positive number of draws between checks.

    Returns:
        Mean, standard error, relative error, work count, and convergence flag.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If arguments have invalid domains or fewer than two
            samples were available.

    Equation:
        [EQ-STAT-STREAM-007], [EQ-MC-SE-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-WELFORD-1962].

    """
    target = require_finite_real(target_relative_error, name="target_relative_error")
    if target <= 0.0:
        raise ValueError("target_relative_error must be positive")
    maximum = require_non_negative_integer(max_samples, name="max_samples")
    minimum = require_non_negative_integer(min_samples, name="min_samples")
    interval = require_non_negative_integer(check_every, name="check_every")
    if minimum < 2:
        raise ValueError("min_samples must be at least two")
    if maximum < minimum:
        raise ValueError("max_samples must be at least min_samples")
    if interval == 0:
        raise ValueError("check_every must be positive")
    if callable(source):
        sampler = source

        def draws() -> Iterable[float]:
            """Yield successive sampler outputs indefinitely.

            Yields:
                One sampler output per iteration.

            References:
                [REF-ROBERT-CASELLA-2004, Chapter 3]. No equation is evaluated.

            """
            while True:
                yield sampler()

        stream: Iterable[float] = draws()
    else:
        stream = source
    moments = RunningMoments()
    trace: list[tuple[int, float, float]] = []
    converged = False
    relative = math.inf
    next_check = minimum
    for value in stream:
        moments.push(value)
        n = moments.count
        if n == next_check or n == maximum:
            relative = moments.relative_standard_error()
            trace.append((n, moments.mean, moments.standard_error()))
            if relative <= target:
                converged = True
                break
            next_check += interval
        if n >= maximum:
            break
    if moments.count < 2:
        raise ValueError("source yielded fewer than two samples")
    if not trace or trace[-1][0] != moments.count:
        relative = moments.relative_standard_error()
        trace.append((moments.count, moments.mean, moments.standard_error()))
        # A stream that ended before min_samples never reached a check.
        converged = relative <= target and moments.count >= minimum
    return StreamingRunResult(
        mean=moments.mean,
        standard_error=moments.standard_error(),
        relative_error=relative,
        sample_size=moments.count,
        converged=converged,
        target_relative_error=target,
        moments=moments,
        trace=tuple(trace),
    )
