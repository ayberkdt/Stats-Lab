"""Sliced Wasserstein distance between empirical measures in ``R^d``.

The Wasserstein distance between point clouds needs a linear programme in
the number of points. The sliced distance replaces it by an average of
one-dimensional distances between projections, each of which is solved
exactly by sorting: ``SW_p^p = E_theta W_p^p(theta'X, theta'Y)`` with
``theta`` uniform on the unit sphere. The expectation is estimated by
seeded Monte Carlo over directions and reported with a standard error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import require_finite_real
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Vector

from .transport import _points, _weights

_MASS_TOLERANCE = 1e-9


def _exponent(p: object) -> float:
    """Validate a Wasserstein exponent ``p >= 1``.

    Args:
        p: Candidate exponent.

    Returns:
        The exponent as a float.

    Raises:
        ValueError: If ``p < 1`` or ``p`` is not finite.

    References:
        [REF-PEYRE-CUTURI-2019]. No equation is evaluated.

    """
    exponent = require_finite_real(p, name="p")
    if exponent < 1.0:
        raise ValueError("p must be at least 1")
    return exponent


def _normalized(
    weights: Iterable[object] | None, count: int, *, name: str
) -> tuple[float, ...]:
    """Return probability weights for ``count`` atoms.

    Args:
        weights: Non-negative weights, or ``None`` for ``1/count`` each.
        count: Number of atoms.
        name: Argument name for error messages.

    Returns:
        Weights that sum to one.

    Raises:
        ValueError: If the weights are invalid or their number is wrong.

    References:
        [REF-PEYRE-CUTURI-2019]. No equation is evaluated.

    """
    if weights is None:
        return (1.0 / count,) * count
    values = _weights(weights, name=name)
    if len(values) != count:
        raise ValueError(f"{name} must have one entry per point")
    total = math.fsum(values)
    return tuple(value / total for value in values)


def _quantile_coupling_cost(
    xs: Sequence[float],
    ys: Sequence[float],
    wx: Sequence[float] | None,
    wy: Sequence[float] | None,
    exponent: float,
) -> float:
    """Return ``int_0^1 |Q_x(u) - Q_y(u)|^p du`` for two discrete laws.

    Both quantile functions are step functions; the integral is the sum over
    the merged breakpoints of the cumulative weights. For equally many
    equally weighted atoms it reduces to
    ``(1/n) sum_i |x_(i) - y_(i)|^p``.

    Args:
        xs: Atoms of the first law.
        ys: Atoms of the second law.
        wx: Probability weights of ``xs`` or ``None`` for uniform.
        wy: Probability weights of ``ys`` or ``None`` for uniform.
        exponent: ``p >= 1``.

    Returns:
        ``W_p^p``.

    Equation:
        [EQ-OT-SW-001].

    References:
        [REF-PEYRE-CUTURI-2019].

    """
    n, m = len(xs), len(ys)
    if wx is None and wy is None and n == m:
        sx, sy = sorted(xs), sorted(ys)
        if exponent == 2.0:
            return math.fsum((u - v) ** 2 for u, v in zip(sx, sy, strict=True)) / n
        return (
            math.fsum(abs(u - v) ** exponent for u, v in zip(sx, sy, strict=True)) / n
        )
    order_x = sorted(range(n), key=lambda index: xs[index])
    order_y = sorted(range(m), key=lambda index: ys[index])
    weight_x = [1.0 / n if wx is None else wx[index] for index in order_x]
    weight_y = [1.0 / m if wy is None else wy[index] for index in order_y]
    i = j = 0
    left_x, left_y = weight_x[0], weight_y[0]
    terms: list[float] = []
    while True:
        step = min(left_x, left_y)
        if step > 0.0:
            gap = abs(xs[order_x[i]] - ys[order_y[j]])
            terms.append(step * gap**exponent)
        left_x -= step
        left_y -= step
        advance_x = left_x <= 0.0
        advance_y = left_y <= 0.0
        if advance_x:
            i += 1
            if i == n:
                break
            left_x = weight_x[i]
        if advance_y:
            j += 1
            if j == m:
                break
            left_y = weight_y[j]
    return math.fsum(terms)


def empirical_wasserstein_1d(
    source: Iterable[float],
    target: Iterable[float],
    *,
    p: float = 2.0,
    source_weights: Iterable[object] | None = None,
    target_weights: Iterable[object] | None = None,
) -> float:
    """Return ``W_p`` between two discrete laws on the line, exactly.

    On the line the monotone (quantile) coupling is optimal for every
    ``p >= 1``, so ``W_p^p = int_0^1 |Q_0(u) - Q_1(u)|^p du``. For discrete
    laws the quantile functions are step functions and the integral is a
    finite sum, evaluated here after sorting. This is the closed form that
    :func:`~probstat.information.quantile_wasserstein` approximates by
    quadrature for general laws.

    Args:
        source: Finite real atoms ``x_i``.
        target: Finite real atoms ``y_j``.
        p: Exponent ``p >= 1``.
        source_weights: Non-negative weights of the ``x_i`` (normalized to
            sum to one), or ``None`` for equal weights.
        target_weights: Non-negative weights of the ``y_j``, or ``None``.

    Returns:
        The Wasserstein-``p`` distance.

    Raises:
        ValueError: If inputs are invalid or ``p < 1``.

    Equation:
        [EQ-OT-W1D-001], [EQ-OT-SW-001].

    References:
        [REF-PEYRE-CUTURI-2019], [REF-VILLANI-2009, Chapter 6].

    """
    exponent = _exponent(p)
    xs = tuple(
        require_finite_real(value, name=f"source[{index}]")
        for index, value in enumerate(source)
    )
    ys = tuple(
        require_finite_real(value, name=f"target[{index}]")
        for index, value in enumerate(target)
    )
    if not xs or not ys:
        raise ValueError("source and target must not be empty")
    wx = (
        None
        if source_weights is None
        else _normalized(source_weights, len(xs), name="source_weights")
    )
    wy = (
        None
        if target_weights is None
        else _normalized(target_weights, len(ys), name="target_weights")
    )
    cost = _quantile_coupling_cost(xs, ys, wx, wy, exponent)
    return float(max(cost, 0.0) ** (1.0 / exponent))


@dataclass(frozen=True, slots=True)
class SlicedWassersteinResult:
    """Monte Carlo estimate of a sliced Wasserstein distance.

    Args:
        distance: ``SW_p``, the ``p``-th root of ``power``.
        power: Estimate of ``SW_p^p``, the mean over directions of
            ``W_p^p`` of the projections.
        standard_error: Standard error of ``power`` over directions (0 in
            one dimension, where the value is exact).
        distance_standard_error: Delta-method standard error of
            ``distance``, ``standard_error / (p distance^(p-1))``; ``inf``
            when ``distance`` is 0 but ``standard_error`` is not.
        directions: Number of projection directions used.
        p: Exponent.
        dimension: Dimension ``d`` of the points.
        exact: Whether the value is exact (``d = 1``).

    References:
        [EQ-OT-SW-001], [EQ-OT-SW-002], [REF-BONNEEL-ETAL-2015].

    """

    distance: float
    power: float
    standard_error: float
    distance_standard_error: float
    directions: int
    p: float
    dimension: int
    exact: bool


def sliced_wasserstein(
    source_points: Iterable[object],
    target_points: Iterable[object],
    *,
    p: float = 2.0,
    directions: int = 200,
    source_weights: Iterable[object] | None = None,
    target_weights: Iterable[object] | None = None,
    seed: int | RandomStream | None = None,
) -> SlicedWassersteinResult:
    """Estimate the sliced Wasserstein distance between two point clouds.

    ``SW_p^p(mu, nu) = E_theta[W_p^p(theta_# mu, theta_# nu)]`` with
    ``theta`` uniform on the unit sphere of ``R^d`` and ``theta_#`` the
    projection ``x -> theta'x``. Directions are normalized standard normal
    vectors drawn from the seeded stream; each projected problem is solved
    exactly by the quantile coupling. The reported standard error covers
    the Monte Carlo error over directions only: the point clouds are taken
    as the measures of interest, not as samples from underlying laws.

    In one dimension the two directions ``+1`` and ``-1`` give the same
    value, so the result is the exact ``W_p`` and no random number is used.
    ``SW_p`` is a metric that is bounded above by ``W_p`` (each projection
    is 1-Lipschitz) but it is not equal to ``W_p`` for ``d > 1``; for a pure
    translation by ``t``, ``SW_2^2 = |t|^2/d`` whereas ``W_2^2 = |t|^2``.

    Args:
        source_points: Scalars or vectors ``x_i``.
        target_points: Scalars or vectors ``y_j`` of the same dimension.
        p: Exponent ``p >= 1``.
        directions: Number of random directions (at least 2 for ``d > 1``).
        source_weights: Non-negative weights of the ``x_i``, normalized to
            sum to one, or ``None`` for equal weights.
        target_weights: Non-negative weights of the ``y_j``, or ``None``.
        seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
            ``None`` for fresh entropy.

    Returns:
        Distance, ``SW_p^p``, and standard errors.

    Raises:
        ValueError: If inputs are invalid, dimensions disagree, ``p < 1``,
            or ``directions < 2`` for ``d > 1``.

    Equation:
        [EQ-OT-SW-001], [EQ-OT-SW-002].

    References:
        [REF-BONNEEL-ETAL-2015], [REF-PEYRE-CUTURI-2019].

    """
    exponent = _exponent(p)
    xs = _points(source_points, name="source_points")
    ys = _points(target_points, name="target_points")
    dimension = len(xs[0])
    if len(ys[0]) != dimension:
        raise ValueError("source and target points must have the same dimension")
    if isinstance(directions, bool) or not isinstance(directions, int):
        raise ValueError("directions must be an integer")
    wx: Vector | None = (
        None
        if source_weights is None
        else _normalized(source_weights, len(xs), name="source_weights")
    )
    wy: Vector | None = (
        None
        if target_weights is None
        else _normalized(target_weights, len(ys), name="target_weights")
    )
    if dimension == 1:
        value = _quantile_coupling_cost(
            [point[0] for point in xs], [point[0] for point in ys], wx, wy, exponent
        )
        return SlicedWassersteinResult(
            distance=float(max(value, 0.0) ** (1.0 / exponent)),
            power=value,
            standard_error=0.0,
            distance_standard_error=0.0,
            directions=1,
            p=exponent,
            dimension=1,
            exact=True,
        )
    if directions < 2:
        raise ValueError("directions must be at least 2")
    generator = resolve_stream(seed).child("sliced-directions").generator()
    values: list[float] = []
    for _ in range(directions):
        while True:
            raw = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
            norm = math.hypot(*raw)
            if norm > 0.0:
                break
        theta = [component / norm for component in raw]
        projected_x = [
            math.fsum(t * c for t, c in zip(theta, point, strict=True)) for point in xs
        ]
        projected_y = [
            math.fsum(t * c for t, c in zip(theta, point, strict=True)) for point in ys
        ]
        values.append(
            _quantile_coupling_cost(projected_x, projected_y, wx, wy, exponent)
        )
    power = math.fsum(values) / directions
    spread = math.fsum((value - power) ** 2 for value in values) / (directions - 1)
    standard_error = math.sqrt(spread / directions)
    distance = float(max(power, 0.0) ** (1.0 / exponent))
    if standard_error == 0.0:
        distance_error = 0.0
    elif distance > 0.0:
        distance_error = standard_error / (exponent * distance ** (exponent - 1.0))
    else:
        distance_error = math.inf
    return SlicedWassersteinResult(
        distance=distance,
        power=power,
        standard_error=standard_error,
        distance_standard_error=distance_error,
        directions=directions,
        p=exponent,
        dimension=dimension,
        exact=False,
    )
