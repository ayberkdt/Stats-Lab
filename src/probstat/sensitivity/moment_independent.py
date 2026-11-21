"""Moment-independent indices from given data: Borgonovo's delta and PAWN.

Variance-based indices summarize the effect of an input on one moment of the
output. Moment-independent indices compare the whole conditional
distribution of ``Y`` given ``X_i`` with the unconditional one:

- Borgonovo's ``delta_i = (1/2) E_{X_i}[int |f_Y(y) - f_{Y | X_i}(y)| dy]``
  is the expected total-variation distance between the two densities
  ([EQ-SENS-DELTA-001]); it lies in ``[0, 1]``, is zero if and only if ``Y``
  and ``X_i`` are independent, and does not change under a strictly
  monotone transformation of ``Y``.
- The PAWN index of Pianosi and Wagener (2015) replaces the densities by
  distribution functions and the integral by the Kolmogorov-Smirnov
  distance ``KS(x_i) = sup_y |F_Y(y) - F_{Y | X_i = x_i}(y)|``, summarized
  over the conditioning values by its median, mean, or maximum
  ([EQ-SENS-PAWN-001]). No density has to be estimated.

Both are estimated here from one i.i.d. sample ``(x_k, y_k)``: the rows are
sorted by ``x_i`` and cut into ``M`` classes of equal size (equal-probability
classes of ``X_i``); the output values of a class stand for the conditional
law. PAWN was introduced with a tailored design of conditional samples; the
class-partition form used here applies the same index to a generic sample.

Bias. Both estimators are positive even for an input without any effect,
because the distance between two estimated distributions is positive by
sampling noise: of order ``sqrt(B / n_c)`` for a histogram with ``B`` bins
and class size ``n_c``, and ``sqrt(1 / n_c)`` for the Kolmogorov-Smirnov
distance. A kernel density estimate also smooths the conditional density
with a wider bandwidth than the unconditional one, which adds a positive
term for irrelevant inputs and blurs sharp conditional densities.
Averaging over classes of finite width replaces ``f_{Y | X_i = x}`` by a
mixture over the class, which biases the index downwards for influential
inputs. The functions therefore offer a permutation null level
([EQ-SENS-DELTA-002]): the index is recomputed after random reassignment
of the outputs to the classes, the mean of these values is reported as
``null_level`` together with a permutation p-value, and
``index - null_level`` is reported as ``corrected``. The corrected value
is a heuristic: the noise bias is largest when the true index is zero, so
subtracting it over-corrects influential inputs. For a strongly
influential input the downward bias of the class average dominates and
even the raw estimate is too low.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.statistics import gaussian_kde, kde_bandwidth

from .regression_based import _validated_sample

DENSITIES = ("histogram", "kde")
PAWN_STATISTICS = ("median", "mean", "max")


@dataclass(frozen=True, slots=True)
class BorgonovoResult:
    """Given-data estimates of Borgonovo's delta.

    Args:
        input_names: Input names.
        delta: Raw estimates (biased upwards for unimportant inputs).
        corrected: ``delta - null_level`` (``None`` without permutations);
            a heuristic, see the module docstring. It is about zero for an
            input without effect and too low for an influential one (0.42
            and 0.11 for exact values 0.51 and 0.16 at ``n = 4000``), so
            it is not an estimate of delta.
        null_level: Mean of the estimator over random class assignments
            (``nan`` without permutations).
        p_values: Permutation p-values of ``H0: X_i independent of Y``
            (``None`` without permutations).
        density: ``"histogram"`` or ``"kde"``.
        partitions: Number of classes ``M`` of each input.
        bins: Histogram bins (0 for ``"kde"``).
        permutations: Number of permutations.
        sample_size: Number of rows ``n``.

    References:
        [REF-BORGONOVO-2007], [REF-PLISCHKE-BORGONOVO-SMITH-2013].

    """

    input_names: tuple[str, ...]
    delta: tuple[float, ...]
    corrected: tuple[float, ...] | None
    null_level: float
    p_values: tuple[float, ...] | None
    density: str
    partitions: int
    bins: int
    permutations: int
    sample_size: int

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> raw delta``.

        References:
            [EQ-SENS-DELTA-001], [REF-BORGONOVO-2007].

        """
        return dict(zip(self.input_names, self.delta, strict=True))


@dataclass(frozen=True, slots=True)
class PawnResult:
    """Given-data PAWN indices.

    Args:
        input_names: Input names.
        index: The chosen summary of the class KS distances per input.
        statistic: ``"median"``, ``"mean"``, or ``"max"``.
        ks_distances: ``ks_distances[i][m]`` is the KS distance between the
            output of class ``m`` of input ``i`` and the whole sample.
        corrected: ``index - null_level`` (``None`` without permutations); a
            heuristic.
        null_level: Mean of the index over random class assignments
            (``nan`` without permutations).
        p_values: Permutation p-values (``None`` without permutations).
        partitions: Number of classes ``M``.
        permutations: Number of permutations.
        sample_size: Number of rows ``n``.

    References:
        [REF-PIANOSI-WAGENER-2015].

    """

    input_names: tuple[str, ...]
    index: tuple[float, ...]
    statistic: str
    ks_distances: tuple[tuple[float, ...], ...]
    corrected: tuple[float, ...] | None
    null_level: float
    p_values: tuple[float, ...] | None
    partitions: int
    permutations: int
    sample_size: int

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> PAWN index``.

        References:
            [EQ-SENS-PAWN-001], [REF-PIANOSI-WAGENER-2015].

        """
        return dict(zip(self.input_names, self.index, strict=True))


def _class_edges(n: int, classes: int) -> tuple[int, ...]:
    """Return the boundaries of ``classes`` consecutive classes of ``n`` rows.

    Class sizes differ by at most one.

    References:
        [EQ-SENS-GIVEN-001], [REF-PLISCHKE-BORGONOVO-SMITH-2013].

    """
    return tuple(m * n // classes for m in range(classes + 1))


def _check_partitions(partitions: int | None, n: int, minimum: int) -> int:
    """Return a validated number of classes.

    The default ``min(50, max(2, round(n^(1/3))))`` lets the class size grow
    like ``n^(2/3)``.

    Raises:
        ValueError: If fewer than two classes or classes smaller than
            ``minimum`` rows would result.

    References:
        [REF-PLISCHKE-BORGONOVO-SMITH-2013]. No equation is evaluated.

    """
    classes: int
    if partitions is None:
        classes = min(50, max(2, round(float(n) ** (1.0 / 3.0))))
        classes = max(2, min(classes, n // minimum))
    elif isinstance(partitions, bool) or not isinstance(partitions, int):
        raise ValueError("partitions must be an integer")
    else:
        classes = partitions
    if classes < 2 or n // classes < minimum:
        raise ValueError(
            f"partitions must be at least 2 with at least {minimum} rows per class"
        )
    return classes


def _check_permutations(permutations: int) -> int:
    """Return a validated permutation count.

    Raises:
        ValueError: If it is not a non-negative integer.

    References:
        [EQ-SENS-DELTA-002], [REF-PHIPSON-SMYTH-2010].

    """
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, int)
        or permutations < 0
    ):
        raise ValueError("permutations must be a non-negative integer")
    return permutations


class _HistogramDelta:
    """Delta estimator on equal-probability bins of the pooled output.

    References:
        [EQ-SENS-DELTA-001], [REF-BORGONOVO-2007].

    """

    __slots__ = ("_bin", "_bins", "_edges", "_pooled")

    def __init__(self, y: Sequence[float], edges: Sequence[int], bins: int) -> None:
        """Assign every output to a bin of (nearly) equal pooled frequency.

        The bin of ``y_k`` is ``floor(B r_k / n)`` with ``r_k`` the number
        of outputs smaller than ``y_k``; tied outputs share a bin, and the
        assignment depends on the outputs only through their order.

        References:
            [EQ-SENS-DELTA-001], [REF-BORGONOVO-2007].

        """
        n = len(y)
        ordered = sorted(y)
        self._bin = [bisect_left(ordered, v) * bins // n for v in y]
        self._bins = bins
        self._edges = edges
        pooled = [0] * bins
        for b in self._bin:
            pooled[b] += 1
        self._pooled = [count / n for count in pooled]

    def __call__(self, order: Sequence[int]) -> float:
        """Return ``(1/2) sum_m (n_m/n) sum_b |p_b - p_mb|`` for a row order.

        References:
            [EQ-SENS-DELTA-001], [REF-BORGONOVO-2007].

        """
        n = len(order)
        total = 0.0
        for start, stop in zip(self._edges, self._edges[1:], strict=False):
            counts = [0] * self._bins
            for k in order[start:stop]:
                counts[self._bin[k]] += 1
            size = stop - start
            distance = math.fsum(
                abs(p - c / size) for p, c in zip(self._pooled, counts, strict=True)
            )
            total += size / n * distance
        return 0.5 * total


class _KernelDelta:
    """Delta estimator with Gaussian kernel density estimates on a grid.

    References:
        [EQ-SENS-DELTA-001], [EQ-STAT-KDE-001], [REF-BORGONOVO-2007].

    """

    __slots__ = ("_edges", "_grid", "_pooled", "_step", "_y")

    def __init__(self, y: Sequence[float], edges: Sequence[int], grid: int) -> None:
        """Evaluate the unconditional density on an equally spaced grid.

        The grid covers the sample range extended by three unconditional
        bandwidths on both sides.

        Raises:
            ValueError: If the output is constant.

        References:
            [EQ-STAT-KDE-001], [EQ-STAT-KDE-003], [REF-SILVERMAN-1986,
            Section 3.4.2].

        """
        h = kde_bandwidth(y, rule="silverman")
        if not h > 0.0:
            raise ValueError("the output is constant; delta is undefined")
        low, high = min(y) - 3.0 * h, max(y) + 3.0 * h
        self._step = (high - low) / (grid - 1)
        self._grid = tuple(low + g * self._step for g in range(grid))
        self._y = y
        self._edges = edges
        density = gaussian_kde(y, bandwidth=h)
        self._pooled = tuple(density.pdf(point) for point in self._grid)

    def __call__(self, order: Sequence[int]) -> float:
        """Return ``(1/2) sum_m (n_m/n) int |f_Y - f_m| dy`` for a row order.

        Each class density uses its own Silverman bandwidth (the pooled
        bandwidth for a class with constant output); the integral is a
        trapezoidal sum on the grid.

        References:
            [EQ-SENS-DELTA-001], [EQ-STAT-KDE-001], [REF-BORGONOVO-2007].

        """
        n = len(order)
        total = 0.0
        for start, stop in zip(self._edges, self._edges[1:], strict=False):
            members = [self._y[k] for k in order[start:stop]]
            h = kde_bandwidth(members, rule="silverman")
            if not h > 0.0:
                h = self._step
            density = gaussian_kde(members, bandwidth=h)
            gaps = [
                abs(p - density.pdf(point))
                for p, point in zip(self._pooled, self._grid, strict=True)
            ]
            integral = self._step * (math.fsum(gaps) - 0.5 * (gaps[0] + gaps[-1]))
            total += (stop - start) / n * integral
        return 0.5 * total


def _null_summary(
    observed: Sequence[float], null: Sequence[float]
) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
    """Return corrected values, the null mean, and permutation p-values.

    References:
        [EQ-SENS-DELTA-002], [EQ-SENS-HSIC-004], [REF-PHIPSON-SMYTH-2010].

    """
    level = math.fsum(null) / len(null)
    p_values = tuple(
        (1 + sum(1 for value in null if value >= item)) / (len(null) + 1)
        for item in observed
    )
    return tuple(item - level for item in observed), level, p_values


def borgonovo_delta(
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    partitions: int | None = None,
    density: str = "histogram",
    bins: int | None = None,
    grid: int = 200,
    permutations: int = 0,
    seed: int | RandomStream | None = None,
) -> BorgonovoResult:
    """Estimate Borgonovo's delta of each input from an existing sample.

    For input ``i`` the rows are sorted by ``x_i`` and cut into ``M``
    classes of equal size; with ``f_m`` an estimate of the output density in
    class ``m`` and ``f`` of the unconditional density,
    ``delta_i = (1/2) sum_m (n_m / n) int |f(y) - f_m(y)| dy``.

    ``density="histogram"`` (default) uses ``B`` bins of equal pooled
    frequency, so the integral is ``sum_b |p_b - p_mb|``; the estimate then
    depends on the outputs only through their order and is exactly invariant
    under strictly increasing transformations of ``Y``. ``density="kde"``
    uses Gaussian kernel estimates with Silverman bandwidths (one for the
    whole sample, one per class) integrated on a grid; it is smoother, not
    invariant, and unsuitable for heavy-tailed outputs unless they are
    transformed first. Both estimators are biased (see the module
    docstring); ``permutations > 0`` reports the level the estimator takes
    for an input without effect. Tied input values are split between
    classes in the order of the stable sort.

    Args:
        inputs: ``n`` rows of ``d`` input values (i.i.d. rows).
        outputs: ``n`` output values.
        names: Input names (default ``x0, x1, ...``).
        partitions: Number of classes ``M`` (default
            ``min(50, max(2, round(n^(1/3))))``); every class needs at least
            ten rows.
        density: ``"histogram"`` or ``"kde"``.
        bins: Histogram bins ``B >= 2`` (default
            ``max(2, min(30, round(sqrt(n / M) / 2)))``).
        grid: Number of grid points of the ``"kde"`` integral (at least 16).
        permutations: Number of random class assignments for the null level
            and the p-values (0: none); the same null serves all inputs
            because all inputs have the same class sizes.
        seed: Seed or stream of the permutations.

    Returns:
        Raw and corrected estimates with the null level.

    Raises:
        ValueError: For ragged or non-finite data, classes smaller than ten
            rows, an unknown density, invalid ``bins``, ``grid`` or
            ``permutations``, or a constant output.

    Equation:
        [EQ-SENS-DELTA-001], [EQ-SENS-DELTA-002], [EQ-STAT-KDE-001].

    References:
        [REF-BORGONOVO-2007], [REF-PLISCHKE-BORGONOVO-SMITH-2013],
        [REF-PHIPSON-SMYTH-2010].

    """
    columns, y, labels = _validated_sample(inputs, outputs, names)
    n = len(y)
    classes = _check_partitions(partitions, n, 10)
    if density not in DENSITIES:
        raise ValueError(f"density must be one of {DENSITIES}")
    count = _check_permutations(permutations)
    if min(y) == max(y):
        raise ValueError("the output is constant; delta is undefined")
    edges = _class_edges(n, classes)
    used_bins = 0
    estimator: _HistogramDelta | _KernelDelta
    if density == "histogram":
        if bins is None:
            used_bins = max(2, min(30, round(math.sqrt(n / classes) / 2.0)))
        elif isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError("bins must be an integer of at least two")
        else:
            used_bins = bins
        estimator = _HistogramDelta(y, edges, used_bins)
    else:
        if isinstance(grid, bool) or not isinstance(grid, int) or grid < 16:
            raise ValueError("grid must be an integer of at least 16")
        estimator = _KernelDelta(y, edges, grid)
    delta = tuple(
        estimator(sorted(range(n), key=column.__getitem__)) for column in columns
    )
    corrected: tuple[float, ...] | None = None
    p_values: tuple[float, ...] | None = None
    level = math.nan
    if count:
        generator = resolve_stream(seed).child("delta-permutation").generator()
        order = list(range(n))
        null: list[float] = []
        for _ in range(count):
            generator.shuffle(order)
            null.append(estimator(order))
        corrected, level, p_values = _null_summary(delta, null)
    return BorgonovoResult(
        labels, delta, corrected, level, p_values, density, classes, used_bins, count, n
    )


def _summary(values: Sequence[float], statistic: str) -> float:
    """Return the median, mean, or maximum of the class KS distances.

    References:
        [EQ-SENS-PAWN-001], [REF-PIANOSI-WAGENER-2015].

    """
    if statistic == "max":
        return max(values)
    if statistic == "mean":
        return math.fsum(values) / len(values)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _class_ks(
    below: Sequence[int],
    upto: Sequence[int],
    order: Sequence[int],
    edges: Sequence[int],
) -> tuple[float, ...]:
    """Return the KS distance of every class from the whole sample.

    ``below[k]`` and ``upto[k]`` count the outputs smaller than, and not
    larger than, ``y_k``. Both empirical distribution functions are step
    functions, so the supremum of their difference is attained at an output
    of the class, either at the value or just before it.

    References:
        [EQ-SENS-PAWN-001], [REF-PIANOSI-WAGENER-2015].

    """
    n = len(order)
    distances: list[float] = []
    for start, stop in zip(edges, edges[1:], strict=False):
        members = sorted(order[start:stop], key=below.__getitem__)
        size = len(members)
        # Integer arithmetic: |a/n - b/size| = |a size - b n| / (n size), so
        # the maximum is exact and identical for y and for -y.
        largest = 0
        position = 0
        while position < size:
            run = position
            level = below[members[position]]
            while run < size and below[members[run]] == level:
                run += 1
            before = abs(level * size - position * n)
            after = abs(upto[members[position]] * size - run * n)
            largest = max(largest, before, after)
            position = run
        distances.append(largest / (n * size))
    return tuple(distances)


def pawn_indices(
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    partitions: int = 10,
    statistic: str = "median",
    permutations: int = 0,
    seed: int | RandomStream | None = None,
) -> PawnResult:
    """Estimate PAWN indices of each input from an existing sample.

    For input ``i`` the rows are sorted by ``x_i`` and cut into ``M``
    classes of equal size. ``KS_m`` is the largest absolute difference
    between the empirical distribution function of the outputs in class
    ``m`` and that of all ``n`` outputs; the index is the median (default),
    mean, or maximum of ``KS_1, ..., KS_M``. The distances depend on the
    outputs only through their order, so the index is exactly invariant
    under strictly monotone transformations of ``Y``. The maximum reacts to
    an effect confined to part of the input range but grows with ``M`` by
    sampling noise alone; compare it with its ``null_level``.

    Args:
        inputs: ``n`` rows of ``d`` input values (i.i.d. rows).
        outputs: ``n`` output values.
        names: Input names (default ``x0, x1, ...``).
        partitions: Number of classes ``M >= 2`` with at least five rows
            each.
        statistic: ``"median"``, ``"mean"``, or ``"max"``.
        permutations: Number of random class assignments for the null level
            and p-values (0: none).
        seed: Seed or stream of the permutations.

    Returns:
        Indices, class distances, and the optional null level.

    Raises:
        ValueError: For ragged or non-finite data, an invalid
            ``partitions``, ``statistic`` or ``permutations``, or a constant
            output.

    Equation:
        [EQ-SENS-PAWN-001], [EQ-SENS-DELTA-002].

    References:
        [REF-PIANOSI-WAGENER-2015], [REF-PHIPSON-SMYTH-2010].

    """
    columns, y, labels = _validated_sample(inputs, outputs, names)
    n = len(y)
    classes = _check_partitions(partitions, n, 5)
    if statistic not in PAWN_STATISTICS:
        raise ValueError(f"statistic must be one of {PAWN_STATISTICS}")
    count = _check_permutations(permutations)
    if min(y) == max(y):
        raise ValueError("the output is constant; PAWN indices are undefined")
    ordered = sorted(y)
    below = [bisect_left(ordered, v) for v in y]
    upto = [bisect_right(ordered, v) for v in y]
    edges = _class_edges(n, classes)
    distances = tuple(
        _class_ks(below, upto, sorted(range(n), key=column.__getitem__), edges)
        for column in columns
    )
    index = tuple(_summary(row, statistic) for row in distances)
    corrected: tuple[float, ...] | None = None
    p_values: tuple[float, ...] | None = None
    level = math.nan
    if count:
        generator = resolve_stream(seed).child("pawn-permutation").generator()
        order = list(range(n))
        null: list[float] = []
        for _ in range(count):
            generator.shuffle(order)
            null.append(_summary(_class_ks(below, upto, order, edges), statistic))
        corrected, level, p_values = _null_summary(index, null)
    return PawnResult(
        labels,
        index,
        statistic,
        distances,
        corrected,
        level,
        p_values,
        classes,
        count,
        n,
    )
