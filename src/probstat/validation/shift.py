"""Two-sample distribution-shift statistics and tests.

These answer "do the calibration data and the deployment data come from the
same distribution?", the question behind every exchangeability assumption
(conformal prediction, validation-domain transfer, surrogate reuse):

* :func:`energy_distance` and :func:`energy_distance_test` (Székely-Rizzo),
  with an ``O(N)``-per-permutation path for one-dimensional data;
* :func:`mmd_squared` and :func:`mmd_test`, the unbiased kernel maximum mean
  discrepancy with a Gaussian kernel (Gretton et al.);
* :func:`wasserstein_distance` between two samples (exact, from the merged
  empirical quantile functions), :func:`distribution_wasserstein_distance`
  between two distribution objects (quantile-scale quadrature), and the
  Gaussian closed form :func:`normal_wasserstein_distance`.

The two-sample Kolmogorov-Smirnov test lives with the one-sample version in
:func:`probstat.inference.kolmogorov_smirnov_two_sample_test`.

Permutation p-values are ``(1 + b) / (1 + B)`` with ``b`` the number of
relabelings whose statistic is at least the observed one; this is a valid
(slightly conservative) p-value for any number ``B`` of random relabelings.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import cast

from probstat.foundations import (
    QuantileFunction,
    RandomStream,
    coerce_finite_sample,
    require_finite_real,
    resolve_stream,
)
from probstat.foundations.validation import require_non_negative_integer
from probstat.numerics.double_exponential import (
    sign_change_points,
    tanh_sinh_pieces,
    warn_if_unreliable,
)
from probstat.statistics import median

Points = tuple[tuple[float, ...], ...]


def _points(values: Sequence[float] | Sequence[Sequence[float]], name: str) -> Points:
    """Normalize a 1-D sample or a sample of vectors to a tuple of points.

    References:
        Project validation policy. No equation is evaluated.

    """
    items = tuple(values)
    if not items:
        raise ValueError(f"{name} must not be empty")
    if all(isinstance(item, Real) and not isinstance(item, bool) for item in items):
        return tuple((value,) for value in coerce_finite_sample(items, name=name))
    rows = tuple(
        coerce_finite_sample(cast(Sequence[float], item), name=name) for item in items
    )
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"every point of {name} must have the same dimension")
    return rows


def _paired_points(
    first: Sequence[float] | Sequence[Sequence[float]],
    second: Sequence[float] | Sequence[Sequence[float]],
) -> tuple[Points, Points]:
    """Validate two samples of the same dimension.

    References:
        Project validation policy. No equation is evaluated.

    """
    a = _points(first, "first")
    b = _points(second, "second")
    if len(a[0]) != len(b[0]):
        raise ValueError("first and second must have the same dimension")
    return a, b


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Euclidean distance between two points.

    References:
        [REF-SZEKELY-RIZZO-2013].

    """
    if len(first) == 1:
        return abs(first[0] - second[0])
    return math.sqrt(
        math.fsum((a - b) ** 2 for a, b in zip(first, second, strict=True))
    )


def _sorted_pair_sum(values: Sequence[float]) -> float:
    """Return ``sum_(i<j) |z_i - z_j|`` for values already in ascending order.

    References:
        [EQ-VAL-SHIFT-ENERGY-001].

    """
    k = len(values)
    return math.fsum((2 * r - k - 1) * value for r, value in enumerate(values, 1))


# ---------------------------------------------------------------------------
# Energy distance
# ---------------------------------------------------------------------------


def _energy_from_sums(
    within_first: float, within_second: float, cross: float, m: int, n: int
) -> float:
    """Return the V-statistic energy distance from pair sums.

    ``within_*`` are sums over unordered within-sample pairs and ``cross``
    the sum over all ``m n`` cross pairs.

    References:
        [EQ-VAL-SHIFT-ENERGY-001].

    """
    return (
        2.0 * cross / (m * n)
        - 2.0 * within_first / (m * m)
        - 2.0 * within_second / (n * n)
    )


def _pooled_median(values: Sequence[float]) -> float:
    """Return the median of the pooled one-dimensional sample.

    References:
        [REF-SZEKELY-RIZZO-2013]. No equation is evaluated.

    """
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def energy_distance(
    first: Sequence[float] | Sequence[Sequence[float]],
    second: Sequence[float] | Sequence[Sequence[float]],
) -> float:
    """Return the sample energy distance ``2 E|X - Y| - E|X - X'| - E|Y - Y'|``.

    The expectations are averages over all pairs including ``i = j``
    (V-statistics), so the value is non-negative and zero only when the two
    empirical distributions coincide. For one-dimensional data it equals
    ``2 int (F_m - G_n)^2 dx`` and is computed in ``O(N log N)`` from sorted
    values; otherwise all ``O(N^2)`` Euclidean distances are summed.

    Args:
        first: Sample of numbers or of equal-length vectors.
        second: Sample of the same dimension.

    Returns:
        Non-negative energy distance.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a sample is empty, non-finite, or of another dimension.

    Equation:
        [EQ-VAL-SHIFT-ENERGY-001].

    References:
        [REF-SZEKELY-RIZZO-2013].

    """
    a, b = _paired_points(first, second)
    m, n = len(a), len(b)
    if len(a[0]) == 1:
        # The statistic is shift-invariant; centring on the pooled median keeps
        # the rank-weighted sums from losing digits to a large common offset.
        centre = _pooled_median([point[0] for point in a + b])
        xs = sorted(point[0] - centre for point in a)
        ys = sorted(point[0] - centre for point in b)
        total = _sorted_pair_sum(sorted(xs + ys))
        within_x, within_y = _sorted_pair_sum(xs), _sorted_pair_sum(ys)
        return max(
            0.0,
            _energy_from_sums(within_x, within_y, total - within_x - within_y, m, n),
        )
    within_x = math.fsum(
        _distance(a[i], a[j]) for i in range(m) for j in range(i + 1, m)
    )
    within_y = math.fsum(
        _distance(b[i], b[j]) for i in range(n) for j in range(i + 1, n)
    )
    cross = math.fsum(_distance(p, q) for p in a for q in b)
    return max(0.0, _energy_from_sums(within_x, within_y, cross, m, n))


@dataclass(frozen=True, slots=True)
class TwoSamplePermutationResult:
    """Two-sample permutation test of equal distributions.

    Args:
        statistic: Observed test statistic.
        p_value: ``(1 + b) / (1 + B)``; small values are evidence of a
            difference (shift); not the probability that the null is true.
        method: Statistic and calibration label.
        permutations: Number ``B`` of random relabelings.
        seed: Root seed of the relabelings.
        first_size: Size of the first sample.
        second_size: Size of the second sample.
        parameter: Kernel bandwidth for MMD, ``nan`` otherwise.

    References:
        [REF-GOOD-2005], [REF-PHIPSON-SMYTH-2010].

    """

    statistic: float
    p_value: float
    method: str
    permutations: int
    seed: int
    first_size: int
    second_size: int
    parameter: float


def _permutation_setup(
    permutations: int, seed: int | RandomStream | None, label: str
) -> tuple[int, RandomStream, random.Random]:
    """Validate the relabeling count and create the seeded generator.

    References:
        [REF-PHIPSON-SMYTH-2010]. No equation is evaluated.

    """
    draws = require_non_negative_integer(permutations, name="permutations")
    if draws < 1:
        raise ValueError("permutations must be positive")
    stream = resolve_stream(seed)
    return draws, stream, stream.child(label).generator()


def _p_value(observed: float, draws: Sequence[float]) -> float:
    """Return ``(1 + #{T_b >= T_obs - tol}) / (1 + B)``.

    The tolerance ``1e-12 max(1, |T_obs|)`` keeps exact ties counted as ties
    despite summation-order roundoff.

    References:
        [EQ-VAL-SHIFT-PERM-001], [REF-PHIPSON-SMYTH-2010].

    """
    tolerance = 1e-12 * max(1.0, abs(observed))
    hits = sum(1 for value in draws if value >= observed - tolerance)
    return (1 + hits) / (1 + len(draws))


def _block_sum_factory(
    matrix: Sequence[Sequence[float]],
) -> Callable[[Sequence[int]], float]:
    """Return a function summing a symmetric matrix over unordered index pairs.

    References:
        [EQ-VAL-SHIFT-PERM-001].

    """

    def block(indices: Sequence[int]) -> float:
        """Return ``sum_(i<j in indices) matrix[i][j]``.

        References:
            [EQ-VAL-SHIFT-PERM-001].

        """
        total = 0.0
        for position, i in enumerate(indices):
            row = matrix[i]
            for j in indices[position + 1 :]:
                total += row[j]
        return total

    return block


def _pair_sums(
    first: Sequence[int],
    row_sums: Sequence[float],
    grand: float,
    block: Callable[[Sequence[int]], float],
) -> tuple[float, float, float]:
    """Return within-first, within-second, and cross sums from one block sum.

    With off-diagonal row sums ``R_i`` and the unordered total ``T``:
    ``S_XY = sum_(i in X) R_i - 2 S_XX`` and ``S_YY = T - S_XX - S_XY``, so
    only the within-first block costs ``O(m^2)``.

    References:
        [EQ-VAL-SHIFT-PERM-001].

    """
    within_first = block(first)
    cross = math.fsum(row_sums[i] for i in first) - 2.0 * within_first
    return within_first, grand - within_first - cross, cross


def energy_distance_test(
    first: Sequence[float] | Sequence[Sequence[float]],
    second: Sequence[float] | Sequence[Sequence[float]],
    *,
    permutations: int = 999,
    seed: int | RandomStream | None = None,
) -> TwoSamplePermutationResult:
    """Test equal distributions with the energy statistic and permutations.

    The statistic is ``T = (m n / (m + n)) E_(m,n)`` with the V-statistic
    energy distance ``E``. Under the null of one common distribution the
    pooled sample is exchangeable, so relabelings give an exact-size test.
    For 1-D data each relabeling costs ``O(N)`` (pooled values are sorted
    once and the labelled subsequences stay sorted); for vectors the pooled
    distance matrix is formed once and each relabeling costs ``O(m^2)``.

    Args:
        first: Sample of numbers or of equal-length vectors.
        second: Sample of the same dimension.
        permutations: Number of random relabelings.
        seed: Integer seed or stream (child ``"energy-permutation"``).

    Returns:
        Statistic, permutation p-value, and metadata.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a sample is empty or ``permutations < 1``.

    Equation:
        [EQ-VAL-SHIFT-ENERGY-001], [EQ-VAL-SHIFT-PERM-001].

    References:
        [REF-SZEKELY-RIZZO-2013], [REF-PHIPSON-SMYTH-2010].

    """
    a, b = _paired_points(first, second)
    m, n = len(a), len(b)
    draws, stream, generator = _permutation_setup(
        permutations, seed, "energy-permutation"
    )
    factor = m * n / (m + n)
    pooled = a + b
    labels = [True] * m + [False] * n
    if len(a[0]) == 1:
        order = sorted(range(m + n), key=lambda index: pooled[index][0])
        centre = _pooled_median([point[0] for point in pooled])
        values = [pooled[index][0] - centre for index in order]
        total = _sorted_pair_sum(values)

        def statistic(mask: Sequence[bool]) -> float:
            """Return ``T`` for labels aligned with the sorted pooled values.

            References:
                [EQ-VAL-SHIFT-ENERGY-001].

            """
            xs = [v for v, flag in zip(values, mask, strict=True) if flag]
            ys = [v for v, flag in zip(values, mask, strict=True) if not flag]
            wx, wy = _sorted_pair_sum(xs), _sorted_pair_sum(ys)
            return factor * _energy_from_sums(wx, wy, total - wx - wy, m, n)

        sorted_labels = [labels[index] for index in order]
        observed = statistic(sorted_labels)
        results = []
        for _ in range(draws):
            generator.shuffle(sorted_labels)
            results.append(statistic(sorted_labels))
    else:
        size = m + n
        matrix = [
            [_distance(pooled[i], pooled[j]) for j in range(size)] for i in range(size)
        ]
        row_sums = [math.fsum(row) for row in matrix]
        grand = 0.5 * math.fsum(row_sums)
        block = _block_sum_factory(matrix)
        indices = list(range(size))

        def mv_statistic(first_indices: Sequence[int]) -> float:
            """Return ``T`` for one choice of first-sample indices.

            References:
                [EQ-VAL-SHIFT-ENERGY-001].

            """
            wx, wy, cross = _pair_sums(first_indices, row_sums, grand, block)
            return factor * _energy_from_sums(wx, wy, cross, m, n)

        observed = mv_statistic(indices[:m])
        results = []
        for _ in range(draws):
            generator.shuffle(indices)
            results.append(mv_statistic(indices[:m]))
    return TwoSamplePermutationResult(
        observed,
        _p_value(observed, results),
        "energy distance, permutation p-value",
        draws,
        stream.seed,
        m,
        n,
        math.nan,
    )


# ---------------------------------------------------------------------------
# Kernel maximum mean discrepancy
# ---------------------------------------------------------------------------


def median_heuristic_bandwidth(
    first: Sequence[float] | Sequence[Sequence[float]],
    second: Sequence[float] | Sequence[Sequence[float]],
) -> float:
    """Return the median pairwise Euclidean distance of the pooled sample.

    Args:
        first: Sample of numbers or of equal-length vectors.
        second: Sample of the same dimension.

    Returns:
        Positive bandwidth for the Gaussian kernel.

    Raises:
        ValueError: If fewer than two pooled points exist or the median
            distance is zero.

    References:
        [REF-GRETTON-ETAL-2012].

    """
    a, b = _paired_points(first, second)
    pooled = a + b
    distances = [
        _distance(pooled[i], pooled[j])
        for i in range(len(pooled))
        for j in range(i + 1, len(pooled))
    ]
    if not distances:
        raise ValueError("at least two pooled points are required")
    bandwidth = median(distances)
    if bandwidth <= 0.0:
        raise ValueError("median pairwise distance is zero; pass a bandwidth")
    return bandwidth


def _kernel_matrix(pooled: Points, bandwidth: float) -> list[list[float]]:
    """Return the Gaussian kernel matrix with a zeroed diagonal.

    References:
        [EQ-VAL-SHIFT-MMD-001].

    """
    size = len(pooled)
    scale = 2.0 * bandwidth * bandwidth
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            value = math.exp(-(_distance(pooled[i], pooled[j]) ** 2) / scale)
            matrix[i][j] = matrix[j][i] = value
    return matrix


def _mmd_from_sums(
    within_first: float, within_second: float, cross: float, m: int, n: int
) -> float:
    """Return the unbiased ``MMD_u^2`` from unordered pair sums.

    References:
        [EQ-VAL-SHIFT-MMD-001].

    """
    return (
        2.0 * within_first / (m * (m - 1))
        + 2.0 * within_second / (n * (n - 1))
        - 2.0 * cross / (m * n)
    )


def _bandwidth(
    bandwidth: float | None,
    first: Sequence[float] | Sequence[Sequence[float]],
    second: Sequence[float] | Sequence[Sequence[float]],
) -> float:
    """Return a validated bandwidth or the median heuristic.

    References:
        [REF-GRETTON-ETAL-2012]. No equation is evaluated.

    """
    if bandwidth is None:
        return median_heuristic_bandwidth(first, second)
    value = require_finite_real(bandwidth, name="bandwidth")
    if value <= 0.0:
        raise ValueError("bandwidth must be positive")
    return value


def mmd_squared(
    first: Sequence[float] | Sequence[Sequence[float]],
    second: Sequence[float] | Sequence[Sequence[float]],
    *,
    bandwidth: float | None = None,
) -> float:
    """Return the unbiased squared MMD with a Gaussian kernel.

    ``MMD_u^2 = (1/(m(m-1))) sum_(i!=j) k(x_i, x_j)
    + (1/(n(n-1))) sum_(i!=j) k(y_i, y_j) - (2/(m n)) sum_(i,j) k(x_i, y_j)``
    with ``k(a, b) = exp(-||a - b||^2 / (2 h^2))``. It is an unbiased
    estimate of the population MMD^2 and can be slightly negative.

    Args:
        first: Sample of at least two numbers or vectors.
        second: Sample of at least two points of the same dimension.
        bandwidth: Kernel bandwidth ``h``; median heuristic if omitted.

    Returns:
        Unbiased MMD^2 estimate.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a sample has fewer than two points or ``h <= 0``.

    Equation:
        [EQ-VAL-SHIFT-MMD-001].

    References:
        [REF-GRETTON-ETAL-2012].

    """
    a, b = _paired_points(first, second)
    m, n = len(a), len(b)
    if m < 2 or n < 2:
        raise ValueError("each sample needs at least two points")
    h = _bandwidth(bandwidth, first, second)
    matrix = _kernel_matrix(a + b, h)
    within_x = math.fsum(matrix[i][j] for i in range(m) for j in range(i + 1, m))
    within_y = math.fsum(
        matrix[i][j] for i in range(m, m + n) for j in range(i + 1, m + n)
    )
    cross = math.fsum(matrix[i][j] for i in range(m) for j in range(m, m + n))
    return _mmd_from_sums(within_x, within_y, cross, m, n)


def mmd_test(
    first: Sequence[float] | Sequence[Sequence[float]],
    second: Sequence[float] | Sequence[Sequence[float]],
    *,
    bandwidth: float | None = None,
    permutations: int = 999,
    seed: int | RandomStream | None = None,
) -> TwoSamplePermutationResult:
    """Test equal distributions with the Gaussian-kernel MMD and permutations.

    The bandwidth (median heuristic by default) is computed once from the
    pooled sample, which is invariant under relabeling, so the permutation
    test keeps its exact size. Each relabeling costs ``O(m^2)`` after the
    kernel matrix is formed.

    Args:
        first: Sample of at least two numbers or vectors.
        second: Sample of at least two points of the same dimension.
        bandwidth: Kernel bandwidth ``h``; median heuristic if omitted.
        permutations: Number of random relabelings.
        seed: Integer seed or stream (child ``"mmd-permutation"``).

    Returns:
        ``MMD_u^2``, permutation p-value, and metadata (``parameter`` = ``h``).

    Raises:
        TypeError: If a value is not real.
        ValueError: If a sample has fewer than two points, ``h <= 0``, or
            ``permutations < 1``.

    Equation:
        [EQ-VAL-SHIFT-MMD-001], [EQ-VAL-SHIFT-PERM-001].

    References:
        [REF-GRETTON-ETAL-2012], [REF-PHIPSON-SMYTH-2010].

    """
    a, b = _paired_points(first, second)
    m, n = len(a), len(b)
    if m < 2 or n < 2:
        raise ValueError("each sample needs at least two points")
    h = _bandwidth(bandwidth, first, second)
    draws, stream, generator = _permutation_setup(permutations, seed, "mmd-permutation")
    matrix = _kernel_matrix(a + b, h)
    row_sums = [math.fsum(row) for row in matrix]
    grand = 0.5 * math.fsum(row_sums)
    block = _block_sum_factory(matrix)
    indices = list(range(m + n))

    def statistic(first_indices: Sequence[int]) -> float:
        """Return ``MMD_u^2`` for one choice of first-sample indices.

        References:
            [EQ-VAL-SHIFT-MMD-001].

        """
        wx, wy, cross = _pair_sums(first_indices, row_sums, grand, block)
        return _mmd_from_sums(wx, wy, cross, m, n)

    observed = statistic(indices[:m])
    results = []
    for _ in range(draws):
        generator.shuffle(indices)
        results.append(statistic(indices[:m]))
    return TwoSamplePermutationResult(
        observed,
        _p_value(observed, results),
        "unbiased Gaussian-kernel MMD^2, permutation p-value",
        draws,
        stream.seed,
        m,
        n,
        h,
    )


# ---------------------------------------------------------------------------
# Wasserstein distances
# ---------------------------------------------------------------------------


def _order(value: object) -> float:
    """Validate a Wasserstein order ``p >= 1``.

    References:
        [REF-PANARETOS-ZEMEL-2019]. No equation is evaluated.

    """
    p = require_finite_real(value, name="order")
    if p < 1.0:
        raise ValueError("order must be at least 1")
    return p


def wasserstein_distance(
    first: Sequence[float], second: Sequence[float], *, order: float = 1.0
) -> float:
    """Return the exact ``W_p`` distance between two 1-D samples.

    In one dimension ``W_p^p = int_0^1 |F^-1(u) - G^-1(u)|^p du``. Both
    empirical quantile functions are step functions with jumps at ``k/m``
    and ``l/n``; the integral is summed exactly over the merged breakpoints
    (exact rational breakpoints, no discretization). For ``p = 1`` this
    equals ``int |F_m - G_n| dx`` (the area metric).

    Args:
        first: Non-empty finite sample.
        second: Non-empty finite sample.
        order: ``p >= 1``.

    Returns:
        Non-negative distance in the unit of the data.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a sample is empty or non-finite, or ``order < 1``.

    Equation:
        [EQ-VAL-SHIFT-W-001].

    References:
        [REF-PANARETOS-ZEMEL-2019].

    """
    xs = sorted(coerce_finite_sample(first, name="first"))
    ys = sorted(coerce_finite_sample(second, name="second"))
    p = _order(order)
    m, n = len(xs), len(ys)
    pieces: list[float] = []
    i = j = 0
    previous = 0
    # Breakpoints k/m and l/n compared exactly as integers k*n and l*m.
    while i < m and j < n:
        next_x, next_y = (i + 1) * n, (j + 1) * m
        current = min(next_x, next_y)
        width = (current - previous) / (m * n)
        if width > 0.0:
            pieces.append(width * abs(xs[i] - ys[j]) ** p)
        previous = current
        if next_x == current:
            i += 1
        if next_y == current:
            j += 1
    return float(math.fsum(pieces) ** (1.0 / p))


@dataclass(frozen=True, slots=True)
class WassersteinEstimate:
    """Quadrature estimate of a Wasserstein distance between two laws.

    Args:
        distance: ``W_p`` estimate.
        order: Order ``p``.
        error_estimate: Absolute change of ``W_p^p`` between the last two
            quadrature levels (an error indicator, not a bound).
        evaluations: Integrand evaluations.

    References:
        [REF-PANARETOS-ZEMEL-2019], [REF-TAKAHASI-MORI-1974].

    """

    distance: float
    order: float
    error_estimate: float
    evaluations: int


def distribution_wasserstein_distance(
    first: QuantileFunction,
    second: QuantileFunction,
    *,
    order: float = 1.0,
    tolerance: float = 1.0e-10,
) -> WassersteinEstimate:
    """Return ``W_p`` between two 1-D laws by quantile-scale quadrature.

    ``W_p^p = int_0^1 |Q_1(u) - Q_2(u)|^p du`` is integrated with the
    tanh-sinh rule, which tolerates the unbounded quantiles at both ends;
    upper-tail quantiles use ``isf`` where available. Nodes reach about
    ``1e-300`` from ``0`` and ``1``; closer than ``1e-15`` a failing or
    overflowing quantile counts as zero (see
    :func:`probstat.numerics.tanh_sinh`). Where the quantile functions cross
    (or where they start or stop coinciding), ``|.|^p`` has a kink; these
    points are located by :func:`sign_change_points` on the complement scale
    in the upper half and made piece boundaries, so ``p = 1`` converges at
    the double-exponential rate too. Crossings closer together than the
    search grid (``1/64``) can be missed, which shows as a larger error
    estimate and a warning.

    Args:
        first: Law exposing ``quantile`` (and optionally ``isf``).
        second: Law exposing ``quantile`` (and optionally ``isf``).
        order: ``p >= 1``.
        tolerance: Relative stopping tolerance on ``W_p^p``.

    Returns:
        Distance estimate with its quadrature evidence.

    Raises:
        TypeError: If a law lacks ``quantile``.
        ValueError: If ``order < 1`` or a quantile is non-finite.

    Equation:
        [EQ-VAL-SHIFT-W-002], [EQ-VAL-QUAD-DE-001].

    References:
        [REF-PANARETOS-ZEMEL-2019], [REF-TAKAHASI-MORI-1974].

    """
    if not isinstance(first, QuantileFunction) or not isinstance(
        second, QuantileFunction
    ):
        raise TypeError("both laws must expose quantile(p)")
    p = _order(order)

    def quantile_of(law: QuantileFunction) -> Callable[[float, float], float]:
        """Return ``(u, 1 - u) -> Q(u)`` preferring ``isf`` in the upper half.

        References:
            [EQ-VAL-SHIFT-W-002].

        """
        isf = getattr(law, "isf", None)

        def evaluate(u: float, complement: float) -> float:
            """Evaluate one quantile.

            References:
                [EQ-VAL-SHIFT-W-002].

            """
            if u > 0.5 and callable(isf):
                return float(isf(complement))
            return float(law.quantile(u))

        return evaluate

    q1, q2 = quantile_of(first), quantile_of(second)
    # Put the kinks of |Q1 - Q2|^p (crossing quantiles) at piece boundaries.
    breaks = sign_change_points(
        lambda u, uc: q1(u, uc) - q2(u, uc), complements=True
    )
    result = tanh_sinh_pieces(
        lambda u, uc: abs(q1(u, uc) - q2(u, uc)) ** p,
        breaks,
        tolerance=tolerance,
        max_level=10,
        absolute_tolerance=1e-30,
    )
    warn_if_unreliable(result, "Wasserstein distance")
    return WassersteinEstimate(
        float(max(0.0, result.value) ** (1.0 / p)),
        p,
        result.error_estimate,
        result.evaluations,
    )


def normal_wasserstein_distance(
    first_mu: float, first_sigma: float, second_mu: float, second_sigma: float
) -> float:
    """Return the closed-form ``W_2`` between two univariate normal laws.

    ``W_2^2 = (mu_1 - mu_2)^2 + (sigma_1 - sigma_2)^2``, because the
    quantile functions differ by ``(mu_1 - mu_2) + (sigma_1 - sigma_2) z``
    and ``E z = 0``, ``E z^2 = 1``.

    Args:
        first_mu: Mean of the first law.
        first_sigma: Positive standard deviation of the first law.
        second_mu: Mean of the second law.
        second_sigma: Positive standard deviation of the second law.

    Returns:
        Non-negative distance.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a value is non-finite or a sigma is not positive.

    Equation:
        [EQ-VAL-SHIFT-W-003].

    References:
        [REF-PANARETOS-ZEMEL-2019].

    """
    m1 = require_finite_real(first_mu, name="first_mu")
    m2 = require_finite_real(second_mu, name="second_mu")
    s1 = require_finite_real(first_sigma, name="first_sigma")
    s2 = require_finite_real(second_sigma, name="second_sigma")
    if s1 <= 0.0 or s2 <= 0.0:
        raise ValueError("standard deviations must be positive")
    return math.hypot(m1 - m2, s1 - s2)
