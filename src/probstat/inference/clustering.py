"""k-means clustering with k-means++ restarts and the silhouette coefficient.

:func:`kmeans` minimizes the within-cluster sum of squares by Lloyd's
alternation (EQ-MVA-008) from k-means++ seeds (EQ-GMM-003) and keeps the
best of several restarts; :func:`silhouette_samples` and
:func:`silhouette_score` measure how well each point sits in its cluster
(EQ-MVA-009).

References:
    [REF-LLOYD-1982], [REF-ARTHUR-VASSILVITSKII-2007], [REF-ROUSSEEUW-1987],
    [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 14.3].

"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Vector

from .mixtures import _as_points, _kmeans_plus_plus, _squared_distance

__all__ = [
    "KMeansResult",
    "kmeans",
    "silhouette_samples",
    "silhouette_score",
]


@dataclass(frozen=True, slots=True)
class KMeansResult:
    """Best k-means partition over the restarts.

    Args:
        centers: Cluster centres (means of their members).
        labels: Cluster index of each point.
        inertia: Within-cluster sum of squared distances.
        iterations: Lloyd iterations of the best restart.
        converged: Whether the best restart reached a fixed point.
        restart_inertias: Final inertia of every restart, in order.

    References:
        [EQ-MVA-008], [REF-LLOYD-1982].

    """

    centers: tuple[Vector, ...]
    labels: tuple[int, ...]
    inertia: float
    iterations: int
    converged: bool
    restart_inertias: tuple[float, ...]

    def predict(self, data: Iterable[object]) -> tuple[int, ...]:
        """Assign new points to the nearest centre.

        References:
            [EQ-MVA-008], [REF-LLOYD-1982].

        """
        points = _as_points(data)
        return tuple(_nearest(point, self.centers) for point in points)


def _nearest(point: Vector, centers: Sequence[Vector]) -> int:
    """Return the index of the nearest centre (lowest index on ties).

    References:
        [REF-LLOYD-1982]. Equation [EQ-MVA-008].

    """
    return min(range(len(centers)), key=lambda k: _squared_distance(point, centers[k]))


def _lloyd_run(
    points: Sequence[Vector], centers: list[Vector], max_iterations: int
) -> tuple[list[Vector], list[int], float, int, bool]:
    """Alternate assignment and centroid steps until the labels repeat.

    An empty cluster is re-seeded at the point that contributes most to the
    current inertia, which strictly lowers the objective.

    References:
        [REF-LLOYD-1982], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 14.3.6].
        Equation [EQ-MVA-008].

    """
    k, p = len(centers), len(points[0])
    labels = [_nearest(point, centers) for point in points]
    converged = False
    iterations = 0
    for step in range(1, max_iterations + 1):
        iterations = step
        sizes = [0] * k
        for label in labels:
            sizes[label] += 1
        for empty in (c for c in range(k) if sizes[c] == 0):
            far = max(
                range(len(points)),
                key=lambda i: (
                    _squared_distance(points[i], centers[labels[i]])
                    if sizes[labels[i]] > 1
                    else -1.0
                ),
            )
            sizes[labels[far]] -= 1
            labels[far] = empty
            sizes[empty] = 1
        centers = [
            tuple(
                math.fsum(
                    point[j]
                    for point, label in zip(points, labels, strict=True)
                    if label == c
                )
                / sizes[c]
                for j in range(p)
            )
            for c in range(k)
        ]
        new_labels = [_nearest(point, centers) for point in points]
        if new_labels == labels:
            converged = True
            break
        labels = new_labels
    inertia = math.fsum(
        _squared_distance(point, centers[label])
        for point, label in zip(points, labels, strict=True)
    )
    return centers, labels, inertia, iterations, converged


def kmeans(
    data: Iterable[object],
    clusters: int,
    *,
    restarts: int = 10,
    max_iterations: int = 300,
    seed: int | RandomStream | None = None,
) -> KMeansResult:
    """Partition points into ``K`` clusters by Lloyd's algorithm with restarts.

    Each restart draws k-means++ seeds (``D^2`` sampling, EQ-GMM-003) and
    alternates the assignment ``c(i) = argmin_k ||x_i - m_k||^2`` with the
    centroid update ``m_k = mean of cluster k`` until the assignment repeats
    (EQ-MVA-008). Every step lowers the within-cluster sum of squares
    ``W = sum_i ||x_i - m_c(i)||^2``, so each restart ends at a local
    minimum; the restart with the lowest ``W`` is returned.

    Args:
        data: Points (scalars or equal-length sequences).
        clusters: Number of clusters ``1 <= K <= n``.
        restarts: Number of independent seedings (at least one).
        max_iterations: Lloyd iterations per restart.
        seed: Integer seed or stream (``None`` draws fresh entropy).

    Returns:
        The best partition found.

    Raises:
        ValueError: For ``K`` outside ``[1, n]`` or no restarts.

    Equation:
        [EQ-MVA-008], [EQ-GMM-003].

    References:
        [REF-LLOYD-1982], [REF-ARTHUR-VASSILVITSKII-2007].

    """
    points = _as_points(data)
    k = require_non_negative_integer(clusters, name="clusters")
    if not 1 <= k <= len(points):
        raise ValueError("clusters must lie in [1, number of points]")
    runs = require_non_negative_integer(restarts, name="restarts")
    if runs < 1:
        raise ValueError("restarts must be at least 1")
    limit = require_non_negative_integer(max_iterations, name="max_iterations")
    if limit < 1:
        raise ValueError("max_iterations must be at least 1")
    generator = resolve_stream(seed).generator()
    best: tuple[list[Vector], list[int], float, int, bool] | None = None
    inertias: list[float] = []
    for _ in range(runs):
        seeds = _kmeans_plus_plus(points, k, generator)
        run = _lloyd_run(points, list(seeds), limit)
        inertias.append(run[2])
        if best is None or run[2] < best[2]:
            best = run
    assert best is not None
    centers, labels, inertia, iterations, converged = best
    return KMeansResult(
        centers=tuple(centers),
        labels=tuple(labels),
        inertia=inertia,
        iterations=iterations,
        converged=converged,
        restart_inertias=tuple(inertias),
    )


def silhouette_samples(data: Iterable[object], labels: Iterable[int]) -> Vector:
    """Return Rousseeuw's silhouette width of every point.

    ``s(i) = (b(i) - a(i)) / max(a(i), b(i))`` with ``a(i)`` the mean
    Euclidean distance from point ``i`` to the other members of its
    cluster and ``b(i)`` the smallest mean distance to the members of
    another cluster; ``s(i) = 0`` for a point alone in its cluster
    (EQ-MVA-009).

    Args:
        data: Points.
        labels: Cluster label of each point (at least two distinct labels,
            fewer than ``n``).

    Returns:
        Silhouette widths in ``[-1, 1]``.

    Raises:
        ValueError: For mismatched lengths or an invalid number of clusters.

    Equation:
        [EQ-MVA-009].

    References:
        [REF-ROUSSEEUW-1987].

    """
    points = _as_points(data)
    tags = tuple(labels)
    if len(tags) != len(points):
        raise ValueError("labels must have one entry per point")
    groups = sorted(set(tags))
    if not 2 <= len(groups) <= len(points) - 1:
        raise ValueError("the silhouette needs between 2 and n - 1 clusters")
    members = {g: [i for i, t in enumerate(tags) if t == g] for g in groups}
    widths: list[float] = []
    for i, point in enumerate(points):
        own = members[tags[i]]
        if len(own) == 1:
            widths.append(0.0)
            continue
        distance = [math.sqrt(_squared_distance(point, other)) for other in points]
        a = math.fsum(distance[j] for j in own if j != i) / (len(own) - 1)
        b = min(
            math.fsum(distance[j] for j in members[g]) / len(members[g])
            for g in groups
            if g != tags[i]
        )
        scale = max(a, b)
        widths.append(0.0 if scale == 0.0 else (b - a) / scale)
    return tuple(widths)


def silhouette_score(data: Iterable[object], labels: Iterable[int]) -> float:
    """Return the average silhouette width of a partition.

    Args:
        data: Points.
        labels: Cluster label of each point.

    Returns:
        Mean of :func:`silhouette_samples`; values near 1 indicate compact,
        well-separated clusters.

    Raises:
        ValueError: As for :func:`silhouette_samples`.

    Equation:
        [EQ-MVA-009].

    References:
        [REF-ROUSSEEUW-1987].

    """
    widths = silhouette_samples(data, labels)
    return math.fsum(widths) / len(widths)
