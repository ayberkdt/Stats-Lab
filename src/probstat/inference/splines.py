"""B-spline bases, P-splines, and cubic smoothing splines.

- :func:`bspline_basis` evaluates every B-spline of a knot vector at one
  point by the de Boor-Cox recursion (EQ-SMOOTH-001).
- :func:`fit_pspline` fits a penalized regression spline: a rich B-spline
  basis on equally spaced knots with a difference penalty on adjacent
  coefficients (Eilers and Marx 1996; EQ-SMOOTH-002).
- :func:`fit_smoothing_spline` fits the natural cubic spline that minimizes
  ``sum w_i (y_i - g(x_i))^2 + lambda int g''(x)^2 dx`` in ``O(n)`` by the
  Reinsch algorithm (EQ-SMOOTH-007), with traces and leverages from the
  Hutchinson-de Hoog band inverse (EQ-SMOOTH-008).

In both, ``lambda`` is fixed by the caller or chosen by minimizing GCV
(EQ-SMOOTH-004) or REML (EQ-SMOOTH-005); the fit reports its effective
degrees of freedom (EQ-SMOOTH-003) and Bayesian pointwise confidence bands
(EQ-SMOOTH-006).

References:
    [REF-DE-BOOR-1972], [REF-EILERS-MARX-1996], [REF-WOOD-2017],
    [REF-REINSCH-1967], [REF-GREEN-SILVERMAN-1994],
    [REF-HUTCHINSON-DEHOOG-1985], [REF-CRAVEN-WAHBA-1979], [REF-WOOD-2011],
    [REF-WAHBA-1983], [REF-NYCHKA-1988].

"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, dot, matvec

from ._penalized import (
    PenalizedProblem,
    check_method,
    minimize_log_scalar,
    select_smoothing,
)

__all__ = [
    "PSplineFit",
    "SmoothBand",
    "SmoothingSplineFit",
    "bspline_basis",
    "bspline_design_matrix",
    "difference_penalty",
    "fit_pspline",
    "fit_smoothing_spline",
    "uniform_knots",
]


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------


def _check_knots(
    knots: Iterable[object], degree: object
) -> tuple[tuple[float, ...], int]:
    """Validate a knot vector and a degree.

    References:
        [REF-DE-BOOR-1972]. No equation is evaluated.

    """
    p = require_non_negative_integer(degree, name="degree")
    t = coerce_finite_sample(knots, name="knots")
    if any(b < a for a, b in zip(t, t[1:], strict=False)):
        raise ValueError("knots must be non-decreasing")
    if len(t) < 2 * p + 2:
        raise ValueError("a degree-p spline needs at least 2p + 2 knots")
    if not t[p] < t[len(t) - p - 1]:
        raise ValueError("the knot domain [t_p, t_(m-p-1)] must have positive length")
    if any(t[i + p + 1] <= t[i] for i in range(len(t) - p - 1)):
        raise ValueError("no knot may be repeated more than degree + 1 times")
    return t, p


def _basis_values(x: float, t: Sequence[float], p: int) -> tuple[int, list[float]]:
    """Return the first nonzero index and the ``p + 1`` nonzero B-spline values.

    References:
        [REF-DE-BOOR-1972]. Equation [EQ-SMOOTH-001].

    """
    count = len(t) - p - 1
    if not t[p] <= x <= t[count]:
        raise ValueError(
            f"x = {x!r} lies outside the spline domain [{t[p]!r}, {t[count]!r}]; "
            "B-splines do not extrapolate"
        )
    span = min(max(bisect.bisect_right(t, x) - 1, p), count - 1)
    values = [1.0] + [0.0] * p
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    for j in range(1, p + 1):
        left[j] = x - t[span + 1 - j]
        right[j] = t[span + j] - x
        saved = 0.0
        for r in range(j):
            temp = values[r] / (right[r + 1] + left[j - r])
            values[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        values[j] = saved
    return span - p, values


def bspline_basis(x: float, knots: Iterable[float], degree: int = 3) -> Vector:
    """Evaluate every B-spline ``B_{i,p}`` of a knot vector at one point.

    The Cox-de Boor recursion
    ``B_{i,p}(x) = (x - t_i)/(t_{i+p} - t_i) B_{i,p-1}(x)
    + (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) B_{i+1,p-1}(x)`` (with
    ``0/0 = 0``) is run in de Boor's triangular form, which touches only the
    ``p + 1`` functions that do not vanish at ``x``. Intervals are
    half-open except the last, which includes the right end of the domain.

    Args:
        x: Point in the domain ``[t_p, t_(m-p-1)]``.
        knots: Non-decreasing knots ``t_0, ..., t_(m-1)``.
        degree: Polynomial degree ``p`` (``3`` for cubic).

    Returns:
        The ``m - p - 1`` basis values; they are non-negative and sum to one.

    Raises:
        ValueError: For an invalid knot vector or ``x`` outside the domain.

    Equation:
        [EQ-SMOOTH-001].

    References:
        [REF-DE-BOOR-1972].

    """
    t, p = _check_knots(knots, degree)
    point = require_finite_real(x, name="x")
    first, values = _basis_values(point, t, p)
    row = [0.0] * (len(t) - p - 1)
    row[first : first + p + 1] = values
    return tuple(row)


def bspline_design_matrix(
    points: Iterable[float], knots: Iterable[float], degree: int = 3
) -> Matrix:
    """Return the B-spline design matrix ``B_ij = B_{j,p}(x_i)``.

    Args:
        points: Evaluation points inside the knot domain.
        knots: Non-decreasing knots.
        degree: Polynomial degree.

    Returns:
        Matrix with one row per point and one column per basis function.

    Raises:
        ValueError: For an invalid knot vector or a point outside the domain.

    Equation:
        [EQ-SMOOTH-001].

    References:
        [REF-DE-BOOR-1972], [REF-EILERS-MARX-1996].

    """
    t, p = _check_knots(knots, degree)
    count = len(t) - p - 1
    rows: list[Vector] = []
    for x in coerce_finite_sample(points, name="points"):
        first, values = _basis_values(x, t, p)
        row = [0.0] * count
        row[first : first + p + 1] = values
        rows.append(tuple(row))
    return tuple(rows)


def uniform_knots(
    lower: float, upper: float, segments: int = 20, degree: int = 3
) -> tuple[float, ...]:
    """Return equally spaced knots whose domain is ``[lower, upper]``.

    The interval is cut into ``segments`` equal pieces and ``degree`` knots
    are added beyond each end at the same spacing, as Eilers and Marx
    recommend, giving ``segments + degree`` basis functions.

    Args:
        lower: Left end of the domain.
        upper: Right end, ``> lower``.
        segments: Number of equal intervals (at least one).
        degree: Polynomial degree.

    Returns:
        ``segments + 2 degree + 1`` knots.

    Raises:
        ValueError: If ``upper <= lower`` or ``segments < 1``.

    Equation:
        [EQ-SMOOTH-002].

    References:
        [REF-EILERS-MARX-1996].

    """
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    pieces = require_non_negative_integer(segments, name="segments")
    p = require_non_negative_integer(degree, name="degree")
    if not b > a:
        raise ValueError("upper must exceed lower")
    if pieces < 1:
        raise ValueError("segments must be at least 1")
    width = (b - a) / pieces
    knots = [a + width * index for index in range(-p, pieces + p + 1)]
    knots[p] = a
    knots[p + pieces] = b
    return tuple(knots)


def difference_penalty(size: int, order: int = 2) -> Matrix:
    """Return the difference penalty ``D_d' D_d`` on ``size`` coefficients.

    ``D_d`` is the ``(size - d) x size`` matrix of ``d``-th differences, so
    ``beta' D_d' D_d beta = sum_k (Delta^d beta_k)^2``. Its null space is
    the coefficient sequences that are polynomials of degree below ``d``;
    for a B-spline basis those coefficients give polynomials of degree
    below ``d``, which the penalty therefore leaves unshrunk.

    Args:
        size: Number of coefficients.
        order: Difference order ``0 <= d < size``.

    Returns:
        Symmetric positive-semidefinite ``size x size`` matrix of rank
        ``size - d``.

    Raises:
        ValueError: If ``order >= size``.

    Equation:
        [EQ-SMOOTH-002].

    References:
        [REF-EILERS-MARX-1996].

    """
    k = require_non_negative_integer(size, name="size")
    d = require_non_negative_integer(order, name="order")
    if d >= k:
        raise ValueError("order must be smaller than size")
    row = [1.0]
    for _ in range(d):
        row = [b - a for a, b in zip([0.0, *row], [*row, 0.0], strict=True)]
    differences = [[0.0] * k for _ in range(k - d)]
    for i in range(k - d):
        for j, value in enumerate(row):
            differences[i][i + j] = value
    return tuple(
        tuple(
            math.fsum(differences[r][i] * differences[r][j] for r in range(k - d))
            for j in range(k)
        )
        for i in range(k)
    )


# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------


def _critical_value(level: object) -> float:
    """Return the two-sided normal critical value of a confidence level.

    References:
        [EQ-SMOOTH-006], [REF-NYCHKA-1988]. No further equation is evaluated.

    """
    value = require_finite_real(level, name="level")
    if not 0.0 < value < 1.0:
        raise ValueError("level must lie in (0, 1)")
    return NormalDist().inv_cdf(0.5 + 0.5 * value)


@dataclass(frozen=True, slots=True)
class SmoothBand:
    """Pointwise Bayesian confidence band of a smooth.

    Args:
        points: Evaluation points.
        estimate: Fitted smooth at each point.
        standard_error: Posterior standard deviation at each point.
        lower: ``estimate - z standard_error``.
        upper: ``estimate + z standard_error``.
        level: Nominal pointwise coverage.

    References:
        [EQ-SMOOTH-006], [REF-WAHBA-1983], [REF-NYCHKA-1988].

    """

    points: Vector
    estimate: Vector
    standard_error: Vector
    lower: Vector
    upper: Vector
    level: float


def _band(points: Vector, estimate: Vector, errors: Vector, level: float) -> SmoothBand:
    """Assemble a band from estimates and standard errors.

    References:
        [EQ-SMOOTH-006], [REF-NYCHKA-1988].

    """
    z = _critical_value(level)
    return SmoothBand(
        points=points,
        estimate=estimate,
        standard_error=errors,
        lower=tuple(m - z * s for m, s in zip(estimate, errors, strict=True)),
        upper=tuple(m + z * s for m, s in zip(estimate, errors, strict=True)),
        level=float(level),
    )


def _xy_weights(
    x: Iterable[float], y: Iterable[float], weights: Iterable[float] | None
) -> tuple[Vector, Vector, Vector]:
    """Validate predictor, response, and positive weights of equal length.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    xs = coerce_finite_sample(x, name="x")
    ys = coerce_finite_sample(y, name="y")
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if weights is None:
        ws: Vector = (1.0,) * len(xs)
    else:
        ws = coerce_finite_sample(weights, name="weights")
        if len(ws) != len(xs):
            raise ValueError("weights must have the same length as x")
        if any(w <= 0.0 for w in ws):
            raise ValueError("weights must be positive")
    return xs, ys, ws


# ---------------------------------------------------------------------------
# P-splines
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PSplineFit:
    """A penalized B-spline regression fit (Eilers-Marx P-spline).

    Args:
        knots: Knot vector.
        degree: B-spline degree.
        penalty_order: Difference order of the penalty.
        coefficients: B-spline coefficients ``beta``.
        smoothing: Smoothing parameter ``lambda``.
        method: ``"gcv"``, ``"reml"``, or ``"fixed"``.
        criterion: GCV score or REML criterion at ``lambda`` (GCV for a
            fixed ``lambda``).
        edf: Effective degrees of freedom ``tr(H)``.
        scale: Residual variance ``phi = RSS / (n - edf)``.
        fitted: Fitted values at the data.
        residuals: ``y - fitted``.
        covariance: Bayesian posterior covariance ``phi A^-1`` of ``beta``.
        profile: Grid profile ``(log lambda, criterion)`` of the search.

    References:
        [EQ-SMOOTH-002], [EQ-SMOOTH-003], [EQ-SMOOTH-006],
        [REF-EILERS-MARX-1996], [REF-WOOD-2017, Chapter 5].

    """

    knots: tuple[float, ...]
    degree: int
    penalty_order: int
    coefficients: Vector
    smoothing: float
    method: str
    criterion: float
    edf: float
    scale: float
    fitted: Vector
    residuals: Vector
    covariance: Matrix
    profile: tuple[tuple[float, float], ...]

    def predict(self, points: Iterable[float]) -> Vector:
        """Evaluate the fitted spline ``sum_j beta_j B_j(x)``.

        Args:
            points: Points inside the knot domain.

        Returns:
            Fitted values.

        Raises:
            ValueError: For a point outside the knot domain.

        References:
            [EQ-SMOOTH-001], [EQ-SMOOTH-002].

        """
        design = bspline_design_matrix(points, self.knots, self.degree)
        return matvec(design, self.coefficients)

    def band(self, points: Iterable[float], level: float = 0.95) -> SmoothBand:
        """Return the pointwise Bayesian confidence band at ``points``.

        Args:
            points: Points inside the knot domain.
            level: Nominal pointwise coverage in ``(0, 1)``.

        Returns:
            Band with ``f(x) +/- z sqrt(b(x)' V b(x))``.

        Raises:
            ValueError: For a point outside the domain or an invalid level.

        References:
            [EQ-SMOOTH-006], [REF-WAHBA-1983], [REF-NYCHKA-1988].

        """
        xs = coerce_finite_sample(points, name="points")
        design = bspline_design_matrix(xs, self.knots, self.degree)
        estimate = matvec(design, self.coefficients)
        errors = tuple(
            math.sqrt(max(dot(row, matvec(self.covariance, row)), 0.0))
            for row in design
        )
        return _band(xs, estimate, errors, level)


def fit_pspline(
    x: Iterable[float],
    y: Iterable[float],
    *,
    segments: int = 20,
    degree: int = 3,
    penalty_order: int = 2,
    smoothing: float | None = None,
    method: str = "reml",
    weights: Iterable[float] | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> PSplineFit:
    """Fit a P-spline: B-spline regression with a difference penalty.

    Minimizes ``sum w_i (y_i - sum_j beta_j B_j(x_i))^2 + lambda
    sum_k (Delta^d beta_k)^2`` (EQ-SMOOTH-002). With ``smoothing=None``,
    ``lambda`` minimizes the REML criterion (default) or the GCV score over
    ``log lambda`` (a 37-point grid of 36 natural-log units centred on the
    scale ``tr(B'WB)/tr(D'D)``, then golden-section refinement).

    Args:
        x: Predictor values.
        y: Responses.
        segments: Number of equal knot intervals on ``[lower, upper]``.
        degree: B-spline degree.
        penalty_order: Difference order ``d`` (polynomials of degree below
            ``d`` are not penalized).
        smoothing: Fixed ``lambda >= 0``, or ``None`` to select it.
        method: ``"reml"`` or ``"gcv"`` when ``lambda`` is selected.
        weights: Positive prior weights (variance ``phi / w_i``).
        lower: Left end of the domain (default ``min x``).
        upper: Right end of the domain (default ``max x``).

    Returns:
        The fitted P-spline.

    Raises:
        ValueError: For mismatched lengths, invalid options, data outside
            ``[lower, upper]``, or too few observations for ``edf``.

    Equation:
        [EQ-SMOOTH-001], [EQ-SMOOTH-002], [EQ-SMOOTH-003], [EQ-SMOOTH-004],
        [EQ-SMOOTH-005], [EQ-SMOOTH-006].

    References:
        [REF-EILERS-MARX-1996], [REF-WOOD-2017, Chapters 5 and 6],
        [REF-WOOD-2011].

    """
    xs, ys, ws = _xy_weights(x, y, weights)
    chosen = check_method(method)
    a = min(xs) if lower is None else require_finite_real(lower, name="lower")
    b = max(xs) if upper is None else require_finite_real(upper, name="upper")
    knots = uniform_knots(a, b, segments, degree)
    p = require_non_negative_integer(degree, name="degree")
    count = len(knots) - p - 1
    penalty = difference_penalty(count, penalty_order)
    design = bspline_design_matrix(xs, knots, p)
    problem = PenalizedProblem(
        design=design,
        response=ys,
        weights=ws,
        penalties=(penalty,),
        ranks=(count - int(penalty_order),),
        blocks=(tuple(range(count)),),
    )
    if smoothing is None:
        lambdas, value, profile = select_smoothing(problem, chosen, (None,))
        label = chosen
    else:
        lam = require_finite_real(smoothing, name="smoothing")
        if lam < 0.0:
            raise ValueError("smoothing must be non-negative")
        lambdas, profile, label = (lam,), (), "fixed"
        value = math.nan
    solution = problem.solve(lambdas)
    if label == "fixed":
        value = problem.criterion("gcv", lambdas, solution)
    n = len(ys)
    if not n - solution.edf > 0.0:
        raise ValueError("the fit interpolates the data; increase smoothing or data")
    scale = solution.rss / (n - solution.edf)
    covariance = tuple(tuple(scale * v for v in row) for row in solution.inverse)
    return PSplineFit(
        knots=knots,
        degree=p,
        penalty_order=int(penalty_order),
        coefficients=solution.coefficients,
        smoothing=lambdas[0],
        method=label,
        criterion=value,
        edf=solution.edf,
        scale=scale,
        fitted=solution.fitted,
        residuals=tuple(yv - m for yv, m in zip(ys, solution.fitted, strict=True)),
        covariance=covariance,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Banded symmetric algebra for the Reinsch algorithm
# ---------------------------------------------------------------------------


class _Band:
    """Symmetric band matrix ``M`` with ``upper[k][i] = M[i][i + k]``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.3], [REF-HUTCHINSON-DEHOOG-1985].

    """

    def __init__(self, size: int, width: int) -> None:
        """Create a zero band matrix.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.3]. No equation is evaluated.

        """
        self.size = size
        self.width = width
        self.upper = [[0.0] * size for _ in range(width + 1)]

    def get(self, i: int, j: int) -> float:
        """Return ``M[i][j]`` (zero outside the band).

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.3]. No equation is evaluated.

        """
        if i > j:
            i, j = j, i
        k = j - i
        return self.upper[k][i] if k <= self.width else 0.0

    def factor(self) -> tuple[list[float], list[list[float]]]:
        """Return ``d`` and ``u`` of ``M = U' D U`` (``u[k][i] = U[i][i+k]``).

        Raises:
            ValueError: If ``M`` is not positive definite.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.3]. Equation [EQ-SMOOTH-008].

        """
        n, b = self.size, self.width
        d = [0.0] * n
        u = [[0.0] * n for _ in range(b + 1)]
        for i in range(n):
            pivot = self.upper[0][i] - math.fsum(
                u[k][i - k] ** 2 * d[i - k] for k in range(1, b + 1) if i - k >= 0
            )
            if not pivot > 0.0:
                raise ValueError("band matrix is not positive definite")
            d[i] = pivot
            for k in range(1, b + 1):
                if i + k >= n:
                    break
                total = self.upper[k][i] - math.fsum(
                    u[m][i - m] * u[m + k][i - m] * d[i - m]
                    for m in range(1, b - k + 1)
                    if i - m >= 0
                )
                u[k][i] = total / pivot
        return d, u

    @staticmethod
    def solve(
        d: list[float], u: list[list[float]], rhs: Sequence[float]
    ) -> list[float]:
        """Solve ``U' D U x = rhs`` from the factor.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.3]. No named equation is needed.

        """
        n, b = len(d), len(u) - 1
        z = list(rhs)
        for i in range(n):
            z[i] -= math.fsum(
                u[k][i - k] * z[i - k] for k in range(1, b + 1) if i - k >= 0
            )
        z = [value / pivot for value, pivot in zip(z, d, strict=True)]
        for i in range(n - 1, -1, -1):
            z[i] -= math.fsum(u[k][i] * z[i + k] for k in range(1, b + 1) if i + k < n)
        return z

    @staticmethod
    def inverse_band(d: list[float], u: list[list[float]]) -> list[list[float]]:
        """Return the band ``s[k][i] = (M^-1)[i][i+k]`` by Hutchinson-de Hoog.

        For ``j >= i``: ``Sigma_ij = delta_ij / d_i - sum_k U_{i,i+k}
        Sigma_{i+k,j}``, run from the last row upward.

        References:
            [REF-HUTCHINSON-DEHOOG-1985]. Equation [EQ-SMOOTH-008].

        """
        n, b = len(d), len(u) - 1
        s = [[0.0] * n for _ in range(b + 1)]

        def sigma(i: int, j: int) -> float:
            """Return a computed band entry of the inverse.

            References:
                [REF-HUTCHINSON-DEHOOG-1985]. No further equation is evaluated.

            """
            if i > j:
                i, j = j, i
            return s[j - i][i]

        for i in range(n - 1, -1, -1):
            for offset in range(min(b, n - 1 - i), -1, -1):
                j = i + offset
                total = (1.0 / d[i]) if offset == 0 else 0.0
                total -= math.fsum(
                    u[k][i] * sigma(i + k, j) for k in range(1, b + 1) if i + k < n
                )
                s[offset][i] = total
        return s


# ---------------------------------------------------------------------------
# Cubic smoothing spline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Reinsch:
    """Reinsch quantities of one smoothing parameter.

    References:
        [EQ-SMOOTH-007], [EQ-SMOOTH-008], [REF-GREEN-SILVERMAN-1994].

    """

    values: Vector
    second_derivatives: Vector
    rss: float
    penalty: float
    edf: float
    leverage: Vector
    log_det: float


class _ReinschSystem:
    """Knots, weights, and the ``Q`` and ``R`` matrices of Green and Silverman.

    References:
        [EQ-SMOOTH-007], [REF-REINSCH-1967], [REF-GREEN-SILVERMAN-1994].

    """

    def __init__(self, x: Vector, y: Vector, w: Vector) -> None:
        """Precompute ``Q``, ``R``, ``Q'W^-1Q``, ``Q'y``, and ``log det R``.

        References:
            [EQ-SMOOTH-007], [REF-GREEN-SILVERMAN-1994].

        """
        self.x, self.y, self.w = x, y, w
        n = len(x)
        h = [x[i + 1] - x[i] for i in range(n - 1)]
        self.h = h
        m = n - 2
        # Column j of Q has entries at rows j, j + 1, j + 2.
        self.q = [
            (1.0 / h[j], -1.0 / h[j] - 1.0 / h[j + 1], 1.0 / h[j + 1]) for j in range(m)
        ]
        r = _Band(m, 2)
        for j in range(m):
            r.upper[0][j] = (h[j] + h[j + 1]) / 3.0
            if j + 1 < m:
                r.upper[1][j] = h[j + 1] / 6.0
        self.r = r
        qwq = _Band(m, 2)
        for j in range(m):
            for k in range(j, min(j + 3, m)):
                total = 0.0
                for row in range(k, j + 3):
                    total += self.q[j][row - j] * self.q[k][row - k] / w[row]
                qwq.upper[k - j][j] = total
        self.qwq = qwq
        self.qty = [
            math.fsum(self.q[j][t] * y[j + t] for t in range(3)) for j in range(m)
        ]
        d_r, _ = r.factor()
        self.log_det_r = math.fsum(math.log(v) for v in d_r)

    def scale(self) -> float:
        """Return ``tr(W)/tr(K)``, the unit-free centre of the ``lambda`` search.

        References:
            [REF-WOOD-2017, Chapter 6]. Equation [EQ-SMOOTH-008].

        """
        m = len(self.qty)
        d, u = self.r.factor()
        inverse = _Band.inverse_band(d, u)
        trace_k = 0.0
        for j in range(m):
            for k in range(max(0, j - 2), min(m, j + 3)):
                lo, hi = min(j, k), max(j, k)
                rinv = inverse[hi - lo][lo]
                # (Q'Q)_{kj}
                qq = 0.0
                for row in range(max(j, k), min(j, k) + 3):
                    qq += self.q[j][row - j] * self.q[k][row - k]
                trace_k += rinv * qq
        return math.fsum(self.w) / trace_k if trace_k > 0.0 else 1.0

    def solve(self, lam: float) -> _Reinsch:
        """Return the fit at ``lambda`` in ``O(n)`` operations.

        References:
            [EQ-SMOOTH-007], [EQ-SMOOTH-008], [REF-REINSCH-1967],
            [REF-HUTCHINSON-DEHOOG-1985].

        """
        n, m = len(self.x), len(self.qty)
        system = _Band(m, 2)
        for k in range(3):
            for j in range(m):
                system.upper[k][j] = self.r.upper[k][j] + lam * self.qwq.upper[k][j]
        d, u = system.factor()
        gamma = _Band.solve(d, u, self.qty)
        correction = [0.0] * n
        for j in range(m):
            for t in range(3):
                correction[j + t] += self.q[j][t] * gamma[j]
        values = tuple(
            yv - lam * c / w
            for yv, c, w in zip(self.y, correction, self.w, strict=True)
        )
        rss = math.fsum(
            w * (yv - g) ** 2 for w, yv, g in zip(self.w, self.y, values, strict=True)
        )
        r_gamma = [
            math.fsum(
                self.r.get(j, k) * gamma[k] for k in range(max(0, j - 1), min(m, j + 2))
            )
            for j in range(m)
        ]
        penalty = lam * math.fsum(g * rg for g, rg in zip(gamma, r_gamma, strict=True))
        inverse = _Band.inverse_band(d, u)

        def sigma(j: int, k: int) -> float:
            """Return ``(M^-1)_jk`` inside the band.

            References:
                [REF-HUTCHINSON-DEHOOG-1985]. No further equation is evaluated.

            """
            lo, hi = min(j, k), max(j, k)
            return inverse[hi - lo][lo] if hi - lo <= 2 else 0.0

        trace = 0.0
        for j in range(m):
            for k in range(max(0, j - 2), min(m, j + 3)):
                trace += sigma(j, k) * self.qwq.get(k, j)
        leverage = []
        for i in range(n):
            columns = [j for j in range(max(0, i - 2), min(m, i + 1))]
            quad = 0.0
            for j in columns:
                for k in columns:
                    quad += self.q[j][i - j] * sigma(j, k) * self.q[k][i - k]
            leverage.append(1.0 - lam * quad / self.w[i])
        log_det = (
            math.fsum(math.log(w) for w in self.w)
            + math.fsum(math.log(v) for v in d)
            - self.log_det_r
        )
        return _Reinsch(
            values=values,
            second_derivatives=(0.0, *gamma, 0.0),
            rss=rss,
            penalty=penalty,
            edf=n - lam * trace,
            leverage=tuple(leverage),
            log_det=log_det,
        )

    def criterion(self, method: str, lam: float, fit: _Reinsch) -> float:
        """Return GCV or REML of a smoothing-spline fit.

        References:
            [EQ-SMOOTH-004], [EQ-SMOOTH-005], [REF-CRAVEN-WAHBA-1979],
            [REF-WOOD-2011].

        """
        n = len(self.x)
        if method == "gcv":
            denominator = n - fit.edf
            return n * fit.rss / denominator**2 if denominator > 0.0 else math.inf
        residual_df = n - 2
        total = fit.rss + fit.penalty
        if total <= 0.0 or lam <= 0.0:
            return math.inf
        phi = total / residual_df
        return (
            0.5 * residual_df * (1.0 + math.log(2.0 * math.pi * phi))
            + 0.5 * fit.log_det
            - 0.5 * residual_df * math.log(lam)
        )


@dataclass(frozen=True, slots=True)
class SmoothingSplineFit:
    """A cubic smoothing spline: the natural cubic spline through ``(x_i, g_i)``.

    Args:
        knots: Sorted distinct predictor values (the knots).
        values: Fitted values ``g_i`` at the knots.
        second_derivatives: ``g''`` at the knots (zero at both ends).
        smoothing: Smoothing parameter ``lambda``.
        method: ``"gcv"``, ``"reml"``, or ``"fixed"``.
        criterion: GCV score or REML criterion at ``lambda`` (GCV for a
            fixed ``lambda``).
        edf: Effective degrees of freedom ``tr(H)``.
        scale: Residual variance ``phi = RSS / (n - edf)``.
        leverage: Diagonal ``H_ii`` of the smoother matrix.
        weights: Prior weights, in knot order.
        profile: Grid profile ``(log lambda, criterion)`` of the search.

    References:
        [EQ-SMOOTH-007], [REF-GREEN-SILVERMAN-1994], [REF-REINSCH-1967].

    """

    knots: Vector
    values: Vector
    second_derivatives: Vector
    smoothing: float
    method: str
    criterion: float
    edf: float
    scale: float
    leverage: Vector
    weights: Vector
    profile: tuple[tuple[float, float], ...]

    def predict(self, points: Iterable[float]) -> Vector:
        """Evaluate the natural cubic spline; it is linear beyond the end knots.

        Inside ``[t_i, t_(i+1)]`` with ``h = t_(i+1) - t_i``:
        ``g(x) = ((x - t_i) g_(i+1) + (t_(i+1) - x) g_i)/h
        - (x - t_i)(t_(i+1) - x)((1 + (x - t_i)/h) gamma_(i+1)
        + (1 + (t_(i+1) - x)/h) gamma_i)/6``.

        Args:
            points: Evaluation points.

        Returns:
            Spline values.

        References:
            [EQ-SMOOTH-007], [REF-GREEN-SILVERMAN-1994].

        """
        t, g, c = self.knots, self.values, self.second_derivatives
        n = len(t)
        out: list[float] = []
        for x in coerce_finite_sample(points, name="points"):
            if x < t[0]:
                h = t[1] - t[0]
                slope = (g[1] - g[0]) / h - h * c[1] / 6.0
                out.append(g[0] + slope * (x - t[0]))
                continue
            if x > t[-1]:
                h = t[-1] - t[-2]
                slope = (g[-1] - g[-2]) / h + h * c[-2] / 6.0
                out.append(g[-1] + slope * (x - t[-1]))
                continue
            i = min(bisect.bisect_right(t, x) - 1, n - 2)
            h = t[i + 1] - t[i]
            a, b = x - t[i], t[i + 1] - x
            out.append(
                (a * g[i + 1] + b * g[i]) / h
                - a * b * ((1.0 + a / h) * c[i + 1] + (1.0 + b / h) * c[i]) / 6.0
            )
        return tuple(out)

    def band(self, level: float = 0.95) -> SmoothBand:
        """Return the Bayesian pointwise band at the knots.

        The posterior variance of ``g_i`` is ``phi (W + lambda K)^-1_ii =
        phi H_ii / w_i``; between knots no band is computed.

        Args:
            level: Nominal pointwise coverage in ``(0, 1)``.

        Returns:
            Band at the knots.

        References:
            [EQ-SMOOTH-006], [REF-WAHBA-1983], [REF-NYCHKA-1988].

        """
        errors = tuple(
            math.sqrt(max(self.scale * h / w, 0.0))
            for h, w in zip(self.leverage, self.weights, strict=True)
        )
        return _band(self.knots, self.values, errors, level)


def fit_smoothing_spline(
    x: Iterable[float],
    y: Iterable[float],
    *,
    smoothing: float | None = None,
    method: str = "reml",
    weights: Iterable[float] | None = None,
) -> SmoothingSplineFit:
    """Fit the cubic smoothing spline by the Reinsch algorithm.

    The minimizer of ``sum w_i (y_i - g(x_i))^2 + lambda int g''^2`` over
    twice-differentiable ``g`` is the natural cubic spline with knots at the
    ``x_i`` whose values solve ``(W + lambda Q R^-1 Q') g = W y``. The
    pentadiagonal system ``(R + lambda Q'W^-1 Q) gamma = Q'y`` gives the
    second derivatives ``gamma`` and ``g = y - lambda W^-1 Q gamma``
    (EQ-SMOOTH-007); the trace and the leverages come from the central band
    of the inverse (EQ-SMOOTH-008), so every evaluation is ``O(n)``.
    ``lambda`` is on the scale of the objective above, the convention of
    ``scipy.interpolate.make_smoothing_spline``.

    Args:
        x: Distinct predictor values (any order).
        y: Responses.
        smoothing: Fixed ``lambda > 0``, or ``None`` to select it.
        method: ``"reml"`` (default) or ``"gcv"`` when ``lambda`` is selected.
        weights: Positive prior weights.

    Returns:
        The fitted smoothing spline, with knots in ascending order.

    Raises:
        ValueError: For fewer than three points, repeated ``x`` values,
            mismatched lengths, or invalid options.

    Equation:
        [EQ-SMOOTH-003], [EQ-SMOOTH-004], [EQ-SMOOTH-005], [EQ-SMOOTH-007],
        [EQ-SMOOTH-008].

    References:
        [REF-REINSCH-1967], [REF-GREEN-SILVERMAN-1994],
        [REF-HUTCHINSON-DEHOOG-1985], [REF-CRAVEN-WAHBA-1979],
        [REF-WOOD-2011].

    """
    xs, ys, ws = _xy_weights(x, y, weights)
    chosen = check_method(method)
    if len(xs) < 3:
        raise ValueError("a smoothing spline needs at least three points")
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = tuple(xs[i] for i in order)
    ys = tuple(ys[i] for i in order)
    ws = tuple(ws[i] for i in order)
    if any(b <= a for a, b in zip(xs, xs[1:], strict=False)):
        raise ValueError("x values must be distinct; average tied responses first")
    system = _ReinschSystem(xs, ys, ws)
    profile: tuple[tuple[float, float], ...] = ()
    if smoothing is None:

        def objective(rho: float) -> float:
            """Return the criterion at ``lambda = exp(rho)``.

            References:
                [EQ-SMOOTH-004], [EQ-SMOOTH-005]. No further equation is evaluated.

            """
            lam = math.exp(rho)
            try:
                return system.criterion(chosen, lam, system.solve(lam))
            except ValueError:
                return math.inf

        rho, _, profile = minimize_log_scalar(objective, math.log(system.scale()))
        lam = math.exp(rho)
        label = chosen
    else:
        lam = require_finite_real(smoothing, name="smoothing")
        if not lam > 0.0:
            raise ValueError("smoothing must be positive")
        label = "fixed"
    fit = system.solve(lam)
    value = system.criterion("gcv" if label == "fixed" else label, lam, fit)
    n = len(xs)
    if not n - fit.edf > 0.0:
        raise ValueError("the fit interpolates the data; increase smoothing")
    return SmoothingSplineFit(
        knots=xs,
        values=fit.values,
        second_derivatives=fit.second_derivatives,
        smoothing=lam,
        method=label,
        criterion=value,
        edf=fit.edf,
        scale=fit.rss / (n - fit.edf),
        leverage=fit.leverage,
        weights=ws,
        profile=profile,
    )
