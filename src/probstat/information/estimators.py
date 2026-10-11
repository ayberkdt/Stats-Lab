"""Nearest-neighbour estimators of differential entropy and mutual information.

Both estimators are pure Python with an exhaustive ``O(n^2 d)`` neighbour
search: every pairwise distance is computed once per point. That is fast
enough for a few thousand points and keeps the code dependency-free; a k-d
tree would be needed for larger samples.

* :func:`kozachenko_leonenko_entropy` uses the distance to the ``k``-th
  neighbour as a local density estimate.
* :func:`ksg_mutual_information` is algorithm 1 of Kraskov, Stoegbauer and
  Grassberger: neighbour counts in the marginal spaces inside the joint
  max-norm ``k``-NN radius, whose biases largely cancel.

Samples must be continuous: tied points give zero radii and are rejected.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from typing import Literal

from probstat.foundations import require_finite_real
from probstat.numerics import digamma

from ._units import InformationUnit, unit_factor

Metric = Literal["euclidean", "chebyshev"]
Points = tuple[tuple[float, ...], ...]
_METRICS = ("euclidean", "chebyshev")


def _points(samples: Iterable[object], *, name: str) -> Points:
    """Coerce scalars or equal-length vectors into a tuple of points.

    Args:
        samples: Finite scalars, or finite vectors of one common length.
        name: Argument name.

    Returns:
        Tuple of coordinate tuples.

    Raises:
        TypeError: If an entry is neither real nor a sequence of reals.
        ValueError: If the sample is empty, ragged, or non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    points: list[tuple[float, ...]] = []
    for index, item in enumerate(samples):
        if isinstance(item, Sequence) and not isinstance(item, str):
            point = tuple(
                require_finite_real(value, name=f"{name}[{index}]") for value in item
            )
        else:
            point = (require_finite_real(item, name=f"{name}[{index}]"),)
        points.append(point)
    if not points:
        raise ValueError(f"{name} must not be empty")
    dimension = len(points[0])
    if dimension == 0 or any(len(point) != dimension for point in points):
        raise ValueError(f"{name} points must share one positive dimension")
    return tuple(points)


def _neighbour_count(k: int, size: int) -> int:
    """Validate the neighbour order ``1 <= k < n``.

    Args:
        k: Neighbour order.
        size: Sample size.

    Returns:
        ``k``.

    Raises:
        TypeError: If ``k`` is not an integer.
        ValueError: If ``k`` is out of range.

    References:
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004]. No equation is evaluated.

    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 1 <= k < size:
        raise ValueError("k must satisfy 1 <= k < sample size")
    return k


def _distance_row(points: Points, index: int, metric: str) -> list[float]:
    """Return distances from one point to every point (``inf`` to itself).

    Scalars use ``|x - y|`` (identical in both norms); vectors use
    :func:`math.dist` (Euclidean) or the max norm.

    Args:
        points: Sample points of one dimension.
        index: Index of the reference point.
        metric: ``"euclidean"`` or ``"chebyshev"``.

    Returns:
        List of ``n`` distances.

    References:
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004]. No equation is evaluated.

    """
    origin = points[index]
    if len(origin) == 1:
        value = origin[0]
        row = [abs(value - other[0]) for other in points]
    elif metric == "chebyshev":
        row = [
            max(abs(x - y) for x, y in zip(origin, other, strict=True))
            for other in points
        ]
    else:
        row = [math.dist(origin, other) for other in points]
    row[index] = math.inf
    return row


def _log_unit_ball_volume(dimension: int, metric: str) -> float:
    """Return ``log V_d`` of the unit-radius ball for the chosen norm.

    Args:
        dimension: Dimension ``d``.
        metric: ``"euclidean"`` (``pi^(d/2)/Gamma(d/2 + 1)``) or
            ``"chebyshev"`` (``2^d``).

    Returns:
        Log-volume.

    Equation:
        [EQ-INFO-KLE-001].

    References:
        [REF-KOZACHENKO-LEONENKO-1987],
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004].

    """
    if metric == "chebyshev":
        return dimension * math.log(2.0)
    return 0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1.0)


def kozachenko_leonenko_entropy(
    samples: Iterable[object],
    *,
    k: int = 3,
    metric: Metric = "euclidean",
    unit: InformationUnit = "nats",
) -> float:
    """Estimate differential entropy by the Kozachenko-Leonenko ``k``-NN rule.

    ``h = psi(n) - psi(k) + log V_d + (d/n) sum_i log r_i``, where ``r_i`` is
    the distance from point ``i`` to its ``k``-th nearest neighbour and
    ``V_d`` the unit-ball volume of the norm. The estimator is consistent;
    for smooth densities its bias decays with ``n``, and larger ``k``
    trades variance for bias. Cost ``O(n^2 d)``.

    Args:
        samples: At least two finite scalars or equal-length vectors, with no
            duplicated points.
        k: Neighbour order with ``1 <= k < n``.
        metric: ``"euclidean"`` or ``"chebyshev"`` (max norm).
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Entropy estimate.

    Raises:
        ValueError: If an argument is invalid or a ``k``-NN radius is zero.

    Equation:
        [EQ-INFO-KLE-001].

    References:
        [REF-KOZACHENKO-LEONENKO-1987],
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004].

    """
    factor = unit_factor(unit)
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    points = _points(samples, name="samples")
    size = len(points)
    order = _neighbour_count(k, size)
    dimension = len(points[0])
    log_radii: list[float] = []
    for i in range(size):
        radius = heapq.nsmallest(order, _distance_row(points, i, metric))[-1]
        if radius <= 0.0:
            raise ValueError("duplicated points give a zero k-NN radius")
        log_radii.append(math.log(radius))
    value = (
        digamma(size)
        - digamma(order)
        + _log_unit_ball_volume(dimension, metric)
        + dimension * math.fsum(log_radii) / size
    )
    return value * factor


def ksg_mutual_information(
    x: Iterable[object],
    y: Iterable[object],
    *,
    k: int = 3,
    unit: InformationUnit = "nats",
) -> float:
    """Estimate ``I(X; Y)`` by the Kraskov-Stoegbauer-Grassberger estimator.

    ``I = psi(k) + psi(n) - mean_i[psi(n_x(i) + 1) + psi(n_y(i) + 1)]``, where
    ``eps_i`` is the max-norm distance from ``(x_i, y_i)`` to its ``k``-th
    joint neighbour and ``n_x(i)`` (``n_y(i)``) counts the other points whose
    ``x`` (``y``) lies strictly within ``eps_i``. The estimate can be slightly
    negative under independence and is returned as computed. Cost
    ``O(n^2 (d_x + d_y))``.

    Args:
        x: Finite scalars or equal-length vectors.
        y: Paired finite scalars or vectors, same length as ``x``.
        k: Neighbour order with ``1 <= k < n``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Mutual-information estimate.

    Raises:
        ValueError: If an argument is invalid, the lengths differ, or a joint
            radius is zero (duplicated pairs).

    Equation:
        [EQ-INFO-KSG-001].

    References:
        [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004].

    """
    factor = unit_factor(unit)
    first = _points(x, name="x")
    second = _points(y, name="y")
    size = len(first)
    if len(second) != size:
        raise ValueError("x and y must have the same length")
    order = _neighbour_count(k, size)
    totals: list[float] = []
    for i in range(size):
        dx = _distance_row(first, i, "chebyshev")
        dy = _distance_row(second, i, "chebyshev")
        joint = (max(a, b) for a, b in zip(dx, dy, strict=True))
        radius = heapq.nsmallest(order, joint)[-1]
        if radius <= 0.0:
            raise ValueError("duplicated pairs give a zero joint k-NN radius")
        count_x = sum(1 for value in dx if value < radius)
        count_y = sum(1 for value in dy if value < radius)
        totals.append(digamma(count_x + 1) + digamma(count_y + 1))
    value = digamma(order) + digamma(size) - math.fsum(totals) / size
    return value * factor
