"""Local regression, isotonic regression, and kernel regression.

- :func:`loess` is Cleveland's locally weighted polynomial regression with
  tricube neighbourhood weights (EQ-SMOOTH-009) and optional bisquare
  robustness iterations (EQ-SMOOTH-010).
- :func:`isotonic_regression` is the weighted least-squares monotone fit by
  the pool-adjacent-violators algorithm (EQ-SMOOTH-011).
- :func:`kernel_regression` is the Nadaraya-Watson (local-constant) or
  local-linear kernel estimator (EQ-SMOOTH-012) whose bandwidth minimizes
  the leave-one-out cross-validation score (EQ-SMOOTH-013).

References:
    [REF-CLEVELAND-1979], [REF-AYER-ETAL-1955], [REF-DE-LEEUW-HORNIK-MAIR-2009],
    [REF-NADARAYA-1964], [REF-FAN-GIJBELS-1996],
    [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Vector
from probstat.statistics import kde_bandwidth

__all__ = [
    "IsotonicBlock",
    "IsotonicFit",
    "KernelRegressionFit",
    "LoessFit",
    "isotonic_regression",
    "kernel_regression",
    "loess",
]


def _pairs(x: Iterable[float], y: Iterable[float]) -> tuple[Vector, Vector]:
    """Validate two equal-length finite samples.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    xs = coerce_finite_sample(x, name="x")
    ys = coerce_finite_sample(y, name="y")
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    return xs, ys


def _local_polynomial(
    x0: float,
    xs: Sequence[float],
    ys: Sequence[float],
    weights: Sequence[float],
    degree: int,
    scale: float,
) -> float | None:
    """Return the weighted local polynomial fit at ``x0``, lowering a singular degree.

    The local design uses ``u = (x - x0)/scale``, so the intercept is the
    fitted value; a degree whose normal equations are numerically singular
    is reduced until they are not.

    Returns:
        The fitted value, or ``None`` when all weights vanish.

    References:
        [REF-CLEVELAND-1979], [REF-FAN-GIJBELS-1996]. Equation [EQ-SMOOTH-009].

    """
    active = [
        (w, (x - x0) / scale, y)
        for w, x, y in zip(weights, xs, ys, strict=True)
        if w > 0.0
    ]
    if not active:
        return None
    for d in range(degree, -1, -1):
        size = d + 1
        moments = [math.fsum(w * u**k for w, u, _ in active) for k in range(2 * d + 1)]
        system = [[moments[i + j] for j in range(size)] for i in range(size)]
        rhs = [math.fsum(w * u**i * y for w, u, y in active) for i in range(size)]
        solution = _solve_small(system, rhs)
        if solution is not None:
            return solution[0]
    return None


def _solve_small(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve a small system by Gaussian elimination with partial pivoting.

    Returns:
        The solution, or ``None`` when the matrix is numerically singular.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.4]. No named equation is needed.

    """
    n = len(rhs)
    a = [row[:] + [b] for row, b in zip(matrix, rhs, strict=True)]
    norm = max(abs(v) for row in matrix for v in row)
    if norm == 0.0:
        return None
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= 1e-12 * norm:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        out[r] = (a[r][n] - math.fsum(a[r][c] * out[c] for c in range(r + 1, n))) / a[
            r
        ][r]
    return out


# ---------------------------------------------------------------------------
# LOESS
# ---------------------------------------------------------------------------


def _tricube(u: float) -> float:
    """Return ``(1 - |u|^3)^3`` for ``|u| < 1`` and zero otherwise.

    References:
        [REF-CLEVELAND-1979]. Equation [EQ-SMOOTH-009].

    """
    a = abs(u)
    return (1.0 - a**3) ** 3 if a < 1.0 else 0.0


def _bisquare(u: float) -> float:
    """Return ``(1 - u^2)^2`` for ``|u| < 1`` and zero otherwise.

    References:
        [REF-CLEVELAND-1979]. Equation [EQ-SMOOTH-010].

    """
    return (1.0 - u * u) ** 2 if abs(u) < 1.0 else 0.0


def _median(values: Sequence[float]) -> float:
    """Return the sample median.

    References:
        [REF-CLEVELAND-1979]. No named equation is evaluated.

    """
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


@dataclass(frozen=True, slots=True)
class LoessFit:
    """A LOESS fit with its final robustness weights.

    Args:
        x: Predictor values (input order).
        y: Responses.
        span: Fraction ``f`` of points in each neighbourhood.
        degree: Local polynomial degree (0, 1, or 2).
        neighbours: Neighbourhood size ``q = round(f n)``.
        fitted: Fitted values at the data.
        residuals: ``y - fitted``.
        robustness_weights: Bisquare weights ``delta_k`` (all one without
            robustness iterations).
        iterations: Number of robustness iterations performed.

    References:
        [EQ-SMOOTH-009], [EQ-SMOOTH-010], [REF-CLEVELAND-1979].

    """

    x: Vector
    y: Vector
    span: float
    degree: int
    neighbours: int
    fitted: Vector
    residuals: Vector
    robustness_weights: Vector
    iterations: int

    def predict(self, points: Iterable[float]) -> Vector:
        """Evaluate the local fit at new points with the final robustness weights.

        Args:
            points: Evaluation points; outside ``[min x, max x]`` the local
                polynomial is extrapolated, which is rarely meaningful.

        Returns:
            Fitted values.

        References:
            [EQ-SMOOTH-009], [REF-CLEVELAND-1979].

        """
        return tuple(
            _loess_at(
                x0,
                self.x,
                self.y,
                self.robustness_weights,
                self.degree,
                self.neighbours,
            )
            for x0 in coerce_finite_sample(points, name="points")
        )


def _loess_at(
    x0: float,
    xs: Sequence[float],
    ys: Sequence[float],
    robustness: Sequence[float],
    degree: int,
    neighbours: int,
) -> float:
    """Return the LOESS estimate at one point.

    References:
        [REF-CLEVELAND-1979]. Equation [EQ-SMOOTH-009].

    """
    distances = sorted(abs(x - x0) for x in xs)
    h = distances[neighbours - 1]
    if h > 0.0:
        # Cleveland's definition: the q-th nearest point lies at |u| = 1 and
        # receives tricube weight zero.
        weights = [
            _tricube((x - x0) / h) * r for x, r in zip(xs, robustness, strict=True)
        ]
        scale = h
    else:
        weights = [
            (1.0 if x == x0 else 0.0) * r for x, r in zip(xs, robustness, strict=True)
        ]
        scale = 1.0
    value = _local_polynomial(x0, xs, ys, weights, degree, scale)
    if value is None:
        raise ValueError("every neighbour of a point has zero weight; increase span")
    return value


def loess(
    x: Iterable[float],
    y: Iterable[float],
    *,
    span: float = 0.75,
    degree: int = 1,
    robustness_iterations: int = 0,
) -> LoessFit:
    """Fit Cleveland's locally weighted regression (LOESS/LOWESS).

    At each point ``x0`` the ``q = round(f n)`` nearest neighbours receive
    tricube weights ``T(|x_k - x0| / h(x0))``, ``h(x0)`` the distance to the
    ``q``-th nearest, and a polynomial of degree ``d`` is fitted by weighted
    least squares (EQ-SMOOTH-009). Each robustness iteration recomputes the
    bisquare weights ``delta_k = B(e_k / (6 median|e|))`` from the current
    residuals and refits with weights ``delta_k T(.)`` (EQ-SMOOTH-010);
    Cleveland recommends two iterations. A zero median absolute residual
    keeps weight one only for the exactly fitted points.

    Args:
        x: Predictor values.
        y: Responses.
        span: Fraction ``0 < f <= 1`` of the data in each neighbourhood.
        degree: Local degree 0, 1 (default), or 2.
        robustness_iterations: Number of bisquare reweighting steps.

    Returns:
        The LOESS fit.

    Raises:
        ValueError: For mismatched lengths, an invalid span or degree, or a
            neighbourhood with fewer than ``degree + 1`` points.

    Equation:
        [EQ-SMOOTH-009], [EQ-SMOOTH-010].

    References:
        [REF-CLEVELAND-1979].

    """
    xs, ys = _pairs(x, y)
    f = require_finite_real(span, name="span")
    d = require_non_negative_integer(degree, name="degree")
    iterations = require_non_negative_integer(
        robustness_iterations, name="robustness_iterations"
    )
    if not 0.0 < f <= 1.0:
        raise ValueError("span must lie in (0, 1]")
    if d > 2:
        raise ValueError("degree must be 0, 1, or 2")
    n = len(xs)
    q = min(n, int(math.floor(f * n + 0.5)))
    if q < d + 1:
        raise ValueError(
            "span is too small: each neighbourhood needs degree + 1 points"
        )
    robustness: Vector = (1.0,) * n
    fitted = tuple(_loess_at(x0, xs, ys, robustness, d, q) for x0 in xs)
    for _ in range(iterations):
        residuals = [yv - m for yv, m in zip(ys, fitted, strict=True)]
        scale = 6.0 * _median([abs(e) for e in residuals])
        tiny = 1e-12 * max(1.0, max(abs(v) for v in ys))
        if scale <= tiny:
            robustness = tuple(1.0 if abs(e) <= tiny else 0.0 for e in residuals)
        else:
            robustness = tuple(_bisquare(e / scale) for e in residuals)
        fitted = tuple(_loess_at(x0, xs, ys, robustness, d, q) for x0 in xs)
    return LoessFit(
        x=xs,
        y=ys,
        span=f,
        degree=d,
        neighbours=q,
        fitted=fitted,
        residuals=tuple(yv - m for yv, m in zip(ys, fitted, strict=True)),
        robustness_weights=robustness,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IsotonicBlock:
    """A maximal run of observations sharing one fitted level.

    Args:
        start: First index (inclusive).
        stop: Last index plus one.
        level: Weighted mean of the responses in the run.
        weight: Total weight of the run.

    References:
        [EQ-SMOOTH-011], [REF-AYER-ETAL-1955].

    """

    start: int
    stop: int
    level: float
    weight: float


@dataclass(frozen=True, slots=True)
class IsotonicFit:
    """Weighted least-squares monotone fit.

    Args:
        fitted: Fitted values in input order.
        blocks: Level sets in index order.
        increasing: Direction of the constraint.
        weighted_sse: ``sum w_i (y_i - fitted_i)^2``.

    References:
        [EQ-SMOOTH-011], [REF-DE-LEEUW-HORNIK-MAIR-2009].

    """

    fitted: Vector
    blocks: tuple[IsotonicBlock, ...]
    increasing: bool
    weighted_sse: float


def isotonic_regression(
    y: Iterable[float],
    *,
    weights: Iterable[float] | None = None,
    increasing: bool = True,
) -> IsotonicFit:
    """Fit the monotone sequence closest to ``y`` in weighted least squares.

    The pool-adjacent-violators algorithm scans the data once, merging the
    newest block with its predecessor while their weighted means violate the
    order; each merge replaces two blocks by their weighted mean. The
    result equals the min-max formula
    ``f_i = max_{j <= i} min_{k >= i} Av_w(y_j, ..., y_k)`` (EQ-SMOOTH-011).

    Args:
        y: Responses in the order of the predictor.
        weights: Positive weights (default one each).
        increasing: ``True`` for non-decreasing, ``False`` for
            non-increasing.

    Returns:
        The isotonic fit.

    Raises:
        ValueError: For non-positive weights or mismatched lengths.

    Equation:
        [EQ-SMOOTH-011].

    References:
        [REF-AYER-ETAL-1955], [REF-DE-LEEUW-HORNIK-MAIR-2009].

    """
    ys = coerce_finite_sample(y, name="y")
    if weights is None:
        ws: Vector = (1.0,) * len(ys)
    else:
        ws = coerce_finite_sample(weights, name="weights")
        if len(ws) != len(ys):
            raise ValueError("weights must have the same length as y")
        if any(w <= 0.0 for w in ws):
            raise ValueError("weights must be positive")
    sign = 1.0 if increasing else -1.0
    # Each block: [start, stop, weighted sum, weight]; levels compared in sign * y.
    stack: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(ys, ws, strict=True)):
        stack.append([float(index), float(index + 1), sign * value * weight, weight])
        while (
            len(stack) > 1
            and stack[-2][2] / stack[-2][3] >= stack[-1][2] / stack[-1][3]
        ):
            top = stack.pop()
            stack[-1][1] = top[1]
            stack[-1][2] += top[2]
            stack[-1][3] += top[3]
    blocks = tuple(
        IsotonicBlock(
            start=int(start),
            stop=int(stop),
            level=sign * total / weight,
            weight=weight,
        )
        for start, stop, total, weight in stack
    )
    fitted: list[float] = []
    for block in blocks:
        fitted.extend([block.level] * (block.stop - block.start))
    sse = math.fsum(w * (v - f) ** 2 for w, v, f in zip(ws, ys, fitted, strict=True))
    return IsotonicFit(
        fitted=tuple(fitted),
        blocks=blocks,
        increasing=bool(increasing),
        weighted_sse=sse,
    )


# ---------------------------------------------------------------------------
# Kernel regression
# ---------------------------------------------------------------------------


def _gaussian(u: float) -> float:
    """Return the standard normal kernel up to its constant.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 6.1]. Equation [EQ-SMOOTH-012].

    """
    return math.exp(-0.5 * u * u)


def _epanechnikov(u: float) -> float:
    """Return the Epanechnikov kernel ``1 - u^2`` on ``|u| < 1`` up to its constant.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 6.1]. Equation [EQ-SMOOTH-012].

    """
    return 1.0 - u * u if abs(u) < 1.0 else 0.0


_KERNELS: dict[str, Callable[[float], float]] = {
    "gaussian": _gaussian,
    "epanechnikov": _epanechnikov,
}


def _kernel_estimate(
    x0: float,
    xs: Sequence[float],
    ys: Sequence[float],
    bandwidth: float,
    degree: int,
    kernel: Callable[[float], float],
    skip: int | None = None,
) -> float | None:
    """Return the local-constant or local-linear estimate at ``x0``.

    Args:
        x0: Evaluation point.
        xs: Predictor values.
        ys: Responses.
        bandwidth: Kernel bandwidth ``h``.
        degree: 0 (local constant) or 1 (local linear).
        kernel: Kernel function of ``(x - x0)/h``.
        skip: Index left out (for leave-one-out cross-validation).

    Returns:
        The estimate, or ``None`` when the local fit is undefined.

    References:
        [REF-NADARAYA-1964], [REF-FAN-GIJBELS-1996]. Equation [EQ-SMOOTH-012].

    """
    weights = [
        0.0 if index == skip else kernel((x - x0) / bandwidth)
        for index, x in enumerate(xs)
    ]
    if degree == 0:
        total = math.fsum(weights)
        if total <= 0.0:
            return None
        return math.fsum(w * y for w, y in zip(weights, ys, strict=True)) / total
    s0 = math.fsum(weights)
    if s0 <= 0.0:
        return None
    u = [(x - x0) / bandwidth for x in xs]
    s1 = math.fsum(w * v for w, v in zip(weights, u, strict=True))
    s2 = math.fsum(w * v * v for w, v in zip(weights, u, strict=True))
    t0 = math.fsum(w * y for w, y in zip(weights, ys, strict=True))
    t1 = math.fsum(w * v * y for w, v, y in zip(weights, u, ys, strict=True))
    determinant = s0 * s2 - s1 * s1
    if determinant <= 1e-12 * s0 * max(s2, 1e-300):
        return None
    return (s2 * t0 - s1 * t1) / determinant


@dataclass(frozen=True, slots=True)
class KernelRegressionFit:
    """A kernel regression fit with its cross-validation curve.

    Args:
        x: Predictor values.
        y: Responses.
        bandwidth: Selected or given bandwidth ``h``.
        degree: 0 (Nadaraya-Watson) or 1 (local linear).
        kernel: Kernel name.
        fitted: Fitted values at the data.
        cv_score: Leave-one-out mean squared prediction error at ``h``.
        cv_curve: ``(h, CV(h))`` for every candidate bandwidth (``inf``
            where a leave-one-out fit is undefined).

    References:
        [EQ-SMOOTH-012], [EQ-SMOOTH-013], [REF-FAN-GIJBELS-1996].

    """

    x: Vector
    y: Vector
    bandwidth: float
    degree: int
    kernel: str
    fitted: Vector
    cv_score: float
    cv_curve: tuple[tuple[float, float], ...]

    def predict(self, points: Iterable[float]) -> Vector:
        """Evaluate the kernel estimate at new points.

        Args:
            points: Evaluation points.

        Returns:
            Estimates (``nan`` where the local fit is undefined, for example
            far from the data with a compact kernel).

        References:
            [EQ-SMOOTH-012], [REF-NADARAYA-1964], [REF-FAN-GIJBELS-1996].

        """
        kernel = _KERNELS[self.kernel]
        out: list[float] = []
        for x0 in coerce_finite_sample(points, name="points"):
            value = _kernel_estimate(
                x0, self.x, self.y, self.bandwidth, self.degree, kernel
            )
            out.append(math.nan if value is None else value)
        return tuple(out)


def _loo_score(
    xs: Sequence[float],
    ys: Sequence[float],
    bandwidth: float,
    degree: int,
    kernel: Callable[[float], float],
) -> float:
    """Return the leave-one-out cross-validation score at one bandwidth.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 7.10]. Equation [EQ-SMOOTH-013].

    """
    errors: list[float] = []
    for index, (x0, y0) in enumerate(zip(xs, ys, strict=True)):
        value = _kernel_estimate(x0, xs, ys, bandwidth, degree, kernel, skip=index)
        if value is None:
            return math.inf
        errors.append((y0 - value) ** 2)
    return math.fsum(errors) / len(errors)


def kernel_regression(
    x: Iterable[float],
    y: Iterable[float],
    *,
    bandwidth: float | None = None,
    degree: int = 1,
    kernel: str = "gaussian",
    candidates: Iterable[float] | None = None,
) -> KernelRegressionFit:
    """Fit a Nadaraya-Watson or local-linear kernel regression.

    The estimate at ``x0`` is the intercept of a kernel-weighted polynomial
    fit of degree 0 (Nadaraya-Watson, ``sum K_i y_i / sum K_i``) or 1
    (local linear) with weights ``K((x_i - x0)/h)`` (EQ-SMOOTH-012). With
    ``bandwidth=None`` the bandwidth minimizes the leave-one-out score
    ``CV(h) = (1/n) sum (y_i - m_(-i)(x_i))^2`` (EQ-SMOOTH-013) over the
    candidates, by default 25 values ``h_rot 2^(k/4)``, ``k = -16..8``,
    around the Silverman rule-of-thumb bandwidth ``h_rot`` of ``x``.

    Args:
        x: Predictor values.
        y: Responses.
        bandwidth: Fixed ``h > 0``, or ``None`` to select it.
        degree: 0 (Nadaraya-Watson) or 1 (local linear, default).
        kernel: ``"gaussian"`` or ``"epanechnikov"``.
        candidates: Candidate bandwidths for the cross-validation.

    Returns:
        The kernel regression fit.

    Raises:
        ValueError: For mismatched lengths, invalid options, or (when the
            bandwidth is selected) no candidate with a finite score; a fixed
            bandwidth whose leave-one-out fits are undefined reports
            ``cv_score = inf``.

    Equation:
        [EQ-SMOOTH-012], [EQ-SMOOTH-013].

    References:
        [REF-NADARAYA-1964], [REF-FAN-GIJBELS-1996],
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Sections 6.1 and 7.10].

    """
    xs, ys = _pairs(x, y)
    if len(xs) < 3:
        raise ValueError("kernel regression needs at least three points")
    d = require_non_negative_integer(degree, name="degree")
    if d > 1:
        raise ValueError("degree must be 0 or 1")
    if kernel not in _KERNELS:
        raise ValueError("kernel must be 'gaussian' or 'epanechnikov'")
    function = _KERNELS[kernel]
    if bandwidth is not None:
        h = require_finite_real(bandwidth, name="bandwidth")
        if not h > 0.0:
            raise ValueError("bandwidth must be positive")
        grid: Vector = (h,)
    elif candidates is not None:
        grid = coerce_finite_sample(candidates, name="candidates")
        if any(v <= 0.0 for v in grid):
            raise ValueError("candidate bandwidths must be positive")
    else:
        base = kde_bandwidth(xs)
        if not base > 0.0:
            raise ValueError("x is constant; no bandwidth can be chosen")
        grid = tuple(base * 2.0 ** (k / 4.0) for k in range(-16, 9))
    curve = tuple((h, _loo_score(xs, ys, h, d, function)) for h in grid)
    best_h, best_score = min(curve, key=lambda item: item[1])
    if bandwidth is None and not math.isfinite(best_score):
        raise ValueError("no candidate bandwidth gives a defined leave-one-out fit")
    fitted: list[float] = []
    for x0 in xs:
        value = _kernel_estimate(x0, xs, ys, best_h, d, function)
        fitted.append(math.nan if value is None else value)
    return KernelRegressionFit(
        x=xs,
        y=ys,
        bandwidth=best_h,
        degree=d,
        kernel=kernel,
        fitted=tuple(fitted),
        cv_score=best_score,
        cv_curve=curve,
    )
