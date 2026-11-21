"""Dependence measures that detect non-linear and non-monotone association.

Pearson and Spearman correlations vanish for many dependent pairs (for
example ``Y = X^2`` with symmetric ``X``). This module adds two coefficients
whose population versions are zero exactly under independence:

* :func:`chatterjee_correlation`, a rank coefficient that measures how well
  ``Y`` is a (measurable) function of a scalar ``X``;
* :func:`distance_correlation`, built from pairwise distances, for vector
  observations of any dimensions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
)

_NULL_VARIANCE = 0.4
_DCOV_NORMAL_SCALE = 1.0 + math.pi / 3.0 - math.sqrt(3.0)


@dataclass(frozen=True, slots=True)
class ChatterjeeResult:
    """Chatterjee's rank correlation with its independence test.

    Args:
        statistic: ``xi_n``.
        observations: Number of pairs ``n``.
        z_score: ``sqrt(n) xi_n / sqrt(2/5)``.
        p_value: One-sided asymptotic p-value ``1 - Phi(z_score)`` for the
            null hypothesis of independence (large ``xi_n`` rejects).
        x_ties: Number of observations whose ``x`` value is shared with
            another observation; their order was drawn at random.

    References:
        [EQ-STAT-DEP-001], [EQ-STAT-DEP-002], [REF-CHATTERJEE-2021].

    """

    statistic: float
    observations: int
    z_score: float
    p_value: float
    x_ties: int


def chatterjee_correlation(
    x: Iterable[float],
    y: Iterable[float],
    *,
    seed: int | RandomStream | None = None,
) -> ChatterjeeResult:
    """Return Chatterjee's rank correlation ``xi_n`` of ``y`` on ``x``.

    The pairs are sorted by ``x``; with ``r_i`` the rank of the ``i``-th
    ``y`` in that order,
    ``xi_n = 1 - 3 sum_{i<n} |r_{i+1} - r_i| / (n^2 - 1)``. The coefficient
    is not symmetric: it measures how well ``y`` is determined by ``x``. Its
    population limit is 0 exactly under independence and 1 exactly when
    ``y`` is a measurable function of ``x``. In finite samples ``xi_n`` can
    be slightly negative, and for a strictly monotone relation it equals
    ``(n - 2)/(n + 1)``, not 1.

    Under independence with continuous ``y``, ``sqrt(n) xi_n`` is
    asymptotically ``N(0, 2/5)``; the reported p-value uses that limit and
    is therefore approximate for small ``n``.

    Ties in ``x`` are broken uniformly at random with the seeded stream (the
    result then depends on the seed). Ties in ``y`` are rejected: the
    formula and the null variance above hold only without them.

    Args:
        x: Finite real observations of the explanatory variable.
        y: Paired finite real observations without repeated values.
        seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
            ``None`` for fresh entropy; used only when ``x`` has ties.

    Returns:
        The coefficient, its z-score, and the asymptotic p-value.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the samples differ in length, have fewer than two
            pairs, or ``y`` has ties.

    Equation:
        [EQ-STAT-DEP-001], [EQ-STAT-DEP-002].

    References:
        [REF-CHATTERJEE-2021].

    """
    xs = coerce_finite_sample(x, name="x")
    ys = coerce_finite_sample(y, name="y")
    n = len(xs)
    if len(ys) != n:
        raise ValueError("x and y must have equal length")
    if n < 2:
        raise ValueError("at least two pairs are required")
    by_y = sorted(range(n), key=lambda index: ys[index])
    if any(ys[by_y[i]] == ys[by_y[i + 1]] for i in range(n - 1)):
        raise ValueError("y must not contain ties")
    rank = [0] * n
    for position, index in enumerate(by_y):
        rank[index] = position + 1
    counts: dict[float, int] = {}
    for value in xs:
        counts[value] = counts.get(value, 0) + 1
    tied = sum(count for count in counts.values() if count > 1)
    if tied:
        generator = resolve_stream(seed).child("chatterjee-ties").generator()
        keys = [generator.random() for _ in range(n)]
        by_x = sorted(range(n), key=lambda index: (xs[index], keys[index]))
    else:
        by_x = sorted(range(n), key=lambda index: xs[index])
    total = sum(abs(rank[by_x[i + 1]] - rank[by_x[i]]) for i in range(n - 1))
    statistic = 1.0 - 3.0 * total / (n * n - 1.0)
    z_score = math.sqrt(n / _NULL_VARIANCE) * statistic
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return ChatterjeeResult(
        statistic=statistic,
        observations=n,
        z_score=z_score,
        p_value=p_value,
        x_ties=tied,
    )


def _observations(values: Iterable[object], *, name: str) -> list[tuple[float, ...]]:
    """Validate scalar or vector observations of one common dimension.

    Args:
        values: Real numbers or iterables of real numbers.
        name: Argument name for error messages.

    Returns:
        One tuple of floats per observation.

    Raises:
        ValueError: If the sample is empty or the dimensions disagree.

    References:
        [REF-SZEKELY-RIZZO-BAKIROV-2007]. No equation is evaluated.

    """
    points: list[tuple[float, ...]] = []
    for index, value in enumerate(values):
        if isinstance(value, Iterable) and not isinstance(value, str | bytes):
            point = tuple(
                require_finite_real(component, name=f"{name}[{index}]")
                for component in value
            )
        else:
            point = (require_finite_real(value, name=f"{name}[{index}]"),)
        points.append(point)
    if not points:
        raise ValueError(f"{name} must not be empty")
    dimension = len(points[0])
    if dimension == 0 or any(len(point) != dimension for point in points):
        raise ValueError(f"{name} observations must share one positive dimension")
    return points


def _centred_distances(points: list[tuple[float, ...]]) -> list[list[float]]:
    """Return the double-centred Euclidean distance matrix.

    ``A_kl = a_kl - abar_k. - abar_.l + abar_..`` with ``a_kl = |x_k - x_l|``.

    Args:
        points: Observations of one common dimension.

    Returns:
        The ``n x n`` matrix ``A``; its rows and columns sum to zero.

    Equation:
        [EQ-STAT-DEP-003].

    References:
        [REF-SZEKELY-RIZZO-BAKIROV-2007].

    """
    n = len(points)
    distance = [[0.0] * n for _ in range(n)]
    for k in range(n):
        for index in range(k + 1, n):
            value = math.dist(points[k], points[index])
            distance[k][index] = value
            distance[index][k] = value
    row_means = [math.fsum(row) / n for row in distance]
    grand = math.fsum(row_means) / n
    return [
        [
            distance[k][index] - row_means[k] - row_means[index] + grand
            for index in range(n)
        ]
        for k in range(n)
    ]


@dataclass(frozen=True, slots=True)
class DistanceCorrelationResult:
    """Sample distance covariance and correlation with an optional test.

    Args:
        distance_correlation: ``dCor_n`` in ``[0, 1]`` (0 when a distance
            variance vanishes).
        distance_covariance: ``dCov_n``, the non-negative root of
            ``(1/n^2) sum_kl A_kl B_kl``.
        distance_variance_x: ``dVar_n(X)``, the root of
            ``(1/n^2) sum_kl A_kl^2``.
        distance_variance_y: ``dVar_n(Y)``.
        observations: Number of paired observations ``n``.
        permutations: Number of random permutations used for the test.
        p_value: Permutation p-value ``(1 + #{perm >= observed})/(1 + R)``
            for independence, or ``None`` without a test.

    References:
        [EQ-STAT-DEP-003], [EQ-STAT-DEP-004],
        [REF-SZEKELY-RIZZO-BAKIROV-2007].

    """

    distance_correlation: float
    distance_covariance: float
    distance_variance_x: float
    distance_variance_y: float
    observations: int
    permutations: int
    p_value: float | None


def distance_correlation(
    x: Iterable[object],
    y: Iterable[object],
    *,
    permutations: int = 0,
    seed: int | RandomStream | None = None,
) -> DistanceCorrelationResult:
    """Return the sample distance covariance and distance correlation.

    With Euclidean distance matrices ``a_kl = |x_k - x_l|`` and
    ``b_kl = |y_k - y_l|``, double-centred to ``A`` and ``B``,
    ``dCov_n^2 = (1/n^2) sum_kl A_kl B_kl``, ``dVar_n^2(X) = dCov_n^2(X, X)``
    and ``dCor_n = dCov_n / sqrt(dVar_n(X) dVar_n(Y))``. The population
    distance correlation of random vectors with finite first moments is zero
    if and only if they are independent. ``dCor_n`` is invariant to
    translations, orthogonal transformations, and a common positive
    rescaling of either sample, and equals 1 for ``y = a + b x`` with
    scalar ``b != 0``.

    These are the original (V-statistic) estimators: ``dCov_n^2`` is biased
    upwards, so ``dCor_n`` is positive for independent samples and must not
    be read as evidence of dependence without the test. With
    ``permutations = R > 0`` the observations of ``y`` are permuted ``R``
    times with the seeded stream and the p-value is the proportion of
    ``dCov_n^2`` values at least as large as the observed one, counting the
    observed one; it is valid for exchangeable pairs and has resolution
    ``1/(R + 1)``. Time and memory are ``O(n^2)`` (``O(R n^2)`` with the
    test).

    Args:
        x: ``n`` scalars or vectors of one common dimension.
        y: ``n`` paired scalars or vectors (the dimension may differ from
            that of ``x``).
        permutations: Number ``R >= 0`` of permutations for the test.
        seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
            ``None`` for fresh entropy (used only when ``R > 0``).

    Returns:
        Distance correlation, covariance, variances, and the p-value.

    Raises:
        ValueError: If inputs are invalid, the sample sizes differ, fewer
            than two observations are given, or ``permutations`` is
            negative.

    Equation:
        [EQ-STAT-DEP-003], [EQ-STAT-DEP-004].

    References:
        [REF-SZEKELY-RIZZO-BAKIROV-2007].

    """
    xs = _observations(x, name="x")
    ys = _observations(y, name="y")
    n = len(xs)
    if len(ys) != n:
        raise ValueError("x and y must have equal length")
    if n < 2:
        raise ValueError("at least two observations are required")
    if isinstance(permutations, bool) or not isinstance(permutations, int):
        raise ValueError("permutations must be a non-negative integer")
    if permutations < 0:
        raise ValueError("permutations must be a non-negative integer")
    a = _centred_distances(xs)
    b = _centred_distances(ys)
    scale = float(n * n)
    cross = math.fsum(
        math.fsum(u * v for u, v in zip(a[k], b[k], strict=True)) for k in range(n)
    )
    dcov2 = max(cross / scale, 0.0)
    dvar2_x = math.fsum(u * u for row in a for u in row) / scale
    dvar2_y = math.fsum(v * v for row in b for v in row) / scale
    product = dvar2_x * dvar2_y
    dcor = math.sqrt(dcov2 / math.sqrt(product)) if product > 0.0 else 0.0
    p_value: float | None = None
    if permutations > 0:
        generator = resolve_stream(seed).child("distance-permutations").generator()
        order = list(range(n))
        threshold = cross - 1e-12 * abs(cross)
        exceed = 0
        for _ in range(permutations):
            generator.shuffle(order)
            total = 0.0
            for k in range(n):
                row_a = a[k]
                row_b = b[order[k]]
                total += sum(row_a[index] * row_b[order[index]] for index in range(n))
            if total >= threshold:
                exceed += 1
        p_value = (1.0 + exceed) / (1.0 + permutations)
    return DistanceCorrelationResult(
        distance_correlation=min(dcor, 1.0),
        distance_covariance=math.sqrt(dcov2),
        distance_variance_x=math.sqrt(dvar2_x),
        distance_variance_y=math.sqrt(dvar2_y),
        observations=n,
        permutations=permutations,
        p_value=p_value,
    )


def distance_correlation_bivariate_normal(rho: float) -> float:
    """Return the population distance correlation of a bivariate normal pair.

    For standard normal ``(X, Y)`` with correlation ``rho``,
    ``dCor^2 = (rho asin(rho) + sqrt(1 - rho^2) - rho asin(rho/2)
    - sqrt(4 - rho^2) + 1) / (1 + pi/3 - sqrt(3))``. It follows from
    ``dCov^2 = E|X-X'||Y-Y'| + E|X-X'| E|Y-Y'| - 2 E|X-X'||Y-Y''|`` and
    ``E|UV| = (2/pi)(sqrt(1 - r^2) + r asin(r))`` for standard normal
    ``U, V`` with correlation ``r`` (here ``r = rho`` and ``r = rho/2``,
    each difference having variance 2). The result is even in ``rho``,
    increasing in ``|rho|``, at most ``|rho|``, and invariant to the means
    and scales of the pair.

    Args:
        rho: Correlation in ``[-1, 1]``.

    Returns:
        The distance correlation in ``[0, 1]``.

    Raises:
        ValueError: If ``rho`` is outside ``[-1, 1]``.

    Equation:
        [EQ-STAT-DEP-005].

    References:
        [REF-SZEKELY-RIZZO-BAKIROV-2007].

    """
    r = require_finite_real(rho, name="rho")
    if not -1.0 <= r <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    numerator = (
        r * math.asin(r)
        + math.sqrt(1.0 - r * r)
        - r * math.asin(0.5 * r)
        - math.sqrt(4.0 - r * r)
        + 1.0
    )
    return math.sqrt(min(max(numerator / _DCOV_NORMAL_SCALE, 0.0), 1.0))
