"""Robust M-estimation of location and regression, and the Theil-Sen slope.

An M-estimator replaces the squared-error loss of the mean and of least
squares with a loss ``rho`` that grows more slowly, so that a few gross
errors cannot drag the estimate arbitrarily far. Two classical choices are
provided:

* **Huber** (monotone ``psi``, convex ``rho``): quadratic near zero and linear
  beyond ``k``; ``k = 1.345`` gives 95% asymptotic efficiency at the normal
  law. The estimating equation has a unique root.
* **Tukey biweight (bisquare)** (redescending ``psi``): observations beyond
  ``c`` standardized residuals get weight zero; ``c = 4.685`` gives 95%
  efficiency. The objective is not convex, so the root found depends on the
  starting value; a robust start (median, least absolute deviations) is used.

Both are solved by iteratively reweighted least squares (IRLS) with the
residual scale estimated by the normalized median absolute deviation (MAD),
whose breakdown point is one half. Standard errors use the empirical
sandwich ``A^-1 B A^-1`` and are large-sample approximations that ignore the
variability of the estimated scale.

Breakdown point and efficiency pull in different directions. The location
M-estimators with MAD scale inherit the 50% breakdown point of the median.
Regression M-estimators resist outlying responses but have breakdown point
``1/n`` against outlying design points (bad leverage points). The Theil-Sen
slope (median of pairwise slopes) has breakdown point ``1 - 1/sqrt(2)``
(about 29%) in simple regression, design points included.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, as_vector, householder_qr, lu_solve
from probstat.statistics import median, median_absolute_deviation

from ._hypothesis import check_level, t_critical
from .confidence import ConfidenceInterval
from .linear_models import DesignMatrix, RankDeficientDesignError, _coerce_design
from .quantile_regression import fit_quantile_regression

PsiFunction = Literal["huber", "bisquare"]
RobustStart = Literal["auto", "ls", "lad"]

HUBER_TUNING = 1.345
"""Huber constant ``k`` with 95% asymptotic efficiency at the normal law."""

BISQUARE_TUNING = 4.685
"""Tukey biweight constant ``c`` with 95% asymptotic efficiency at the normal."""

_MAD_NORMAL = 1.482602218505602
"""``1 / Phi^-1(3/4)``: makes the MAD consistent for a normal standard deviation."""


def _check_psi(psi: object, tuning: object) -> tuple[PsiFunction, float]:
    """Validate the ``psi`` family and its tuning constant.

    Args:
        psi: ``"huber"`` or ``"bisquare"``.
        tuning: Positive tuning constant, or ``None`` for the 95%-efficiency
            default of the family.

    Returns:
        Validated family label and tuning constant.

    Raises:
        TypeError: If ``tuning`` is not real.
        ValueError: If the family is unknown or ``tuning`` is not positive.

    References:
        [EQ-INF-ROB-001], [REF-MARONNA-MARTIN-YOHAI-2006].

    """
    if psi == "huber":
        family: PsiFunction = "huber"
        default = HUBER_TUNING
    elif psi == "bisquare":
        family = "bisquare"
        default = BISQUARE_TUNING
    else:
        raise ValueError("psi must be 'huber' or 'bisquare'")
    constant = default if tuning is None else require_finite_real(tuning, name="tuning")
    if constant <= 0.0:
        raise ValueError("tuning must be positive")
    return family, constant


def psi_function(
    value: float, *, psi: PsiFunction = "huber", tuning: float | None = None
) -> float:
    """Return the M-estimation score ``psi(u)`` of the Huber or biweight family.

    Huber: ``psi(u) = max(-k, min(k, u))``. Tukey biweight:
    ``psi(u) = u (1 - (u/c)^2)^2`` for ``|u| < c`` and zero beyond.

    Args:
        value: Standardized residual ``u``.
        psi: ``"huber"`` or ``"bisquare"``.
        tuning: ``k`` or ``c``; ``None`` uses 1.345 or 4.685.

    Returns:
        The score at ``u``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If the family is unknown or ``tuning`` is not positive.

    Equation:
        [EQ-INF-ROB-001].

    References:
        [REF-HUBER-1964], [REF-BEATON-TUKEY-1974],
        [REF-MARONNA-MARTIN-YOHAI-2006].

    """
    family, constant = _check_psi(psi, tuning)
    return _psi(require_finite_real(value, name="value"), family, constant)


def _psi(u: float, family: PsiFunction, c: float) -> float:
    """Evaluate ``psi`` without validation.

    References:
        [EQ-INF-ROB-001].

    """
    if family == "huber":
        return max(-c, min(c, u))
    if abs(u) >= c:
        return 0.0
    ratio = u / c
    return u * (1.0 - ratio * ratio) ** 2


def _psi_derivative(u: float, family: PsiFunction, c: float) -> float:
    """Evaluate ``psi'(u)`` (zero at the Huber corners by convention).

    References:
        [EQ-INF-ROB-001].

    """
    if family == "huber":
        return 1.0 if abs(u) <= c else 0.0
    if abs(u) >= c:
        return 0.0
    ratio2 = (u / c) ** 2
    return (1.0 - ratio2) * (1.0 - 5.0 * ratio2)


def _weight(u: float, family: PsiFunction, c: float) -> float:
    """Evaluate the IRLS weight ``w(u) = psi(u)/u`` (``w(0) = 1``).

    References:
        [EQ-INF-ROB-001].

    """
    if family == "huber":
        magnitude = abs(u)
        return 1.0 if magnitude <= c else c / magnitude
    if abs(u) >= c:
        return 0.0
    ratio = u / c
    return (1.0 - ratio * ratio) ** 2


@dataclass(frozen=True, slots=True)
class RobustLocationResult:
    """M-estimate of location with a fixed MAD scale.

    Args:
        estimate: Root ``mu`` of ``sum psi((x_i - mu)/s) = 0``.
        scale: Scale ``s`` (normalized MAD unless supplied).
        standard_error: Sandwich standard error
            ``s sqrt(n/(n-1) sum psi^2) / |sum psi'|``.
        confidence_interval: ``mu +- t_{n-1} se`` (large-sample).
        psi: ``"huber"`` or ``"bisquare"``.
        tuning: Tuning constant used.
        weights: Final IRLS weights ``psi(u_i)/u_i`` in input order.
        iterations: IRLS iterations performed.
        converged: Whether the step fell below the tolerance.
        sample_size: Number of observations.

    References:
        [EQ-INF-ROB-002], [REF-HUBER-1964], [REF-HUBER-1981].

    """

    estimate: float
    scale: float
    standard_error: float
    confidence_interval: ConfidenceInterval
    psi: str
    tuning: float
    weights: tuple[float, ...]
    iterations: int
    converged: bool
    sample_size: int


def robust_location(
    values: Iterable[object],
    *,
    psi: PsiFunction = "huber",
    tuning: float | None = None,
    scale: float | None = None,
    confidence: float = 0.95,
    tolerance: float = 1e-12,
    max_iterations: int = 500,
) -> RobustLocationResult:
    """Estimate location by a Huber or biweight M-estimator with MAD scale.

    Starting from the median, IRLS iterates
    ``mu <- sum w_i x_i / sum w_i`` with ``w_i = w((x_i - mu)/s)`` and the
    scale ``s`` fixed at the normalized MAD (Huber's proposal with a
    high-breakdown auxiliary scale). For the Huber family the root is unique;
    for the biweight it is the root reached from the median.

    Args:
        values: At least two finite observations.
        psi: ``"huber"`` or ``"bisquare"``.
        tuning: ``k`` or ``c``; ``None`` uses 1.345 (Huber) or 4.685
            (biweight), both 95% efficient at the normal law.
        scale: Known positive scale; ``None`` uses the normalized MAD.
        confidence: Level of the reported interval.
        tolerance: Stop when ``|mu_new - mu| <= tolerance * s``.
        max_iterations: Iteration cap.

    Returns:
        :class:`RobustLocationResult`.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If fewer than two observations are given, the MAD is zero
            (more than half the observations coincide), or an option is
            invalid.

    Equation:
        [EQ-INF-ROB-001], [EQ-INF-ROB-002].

    References:
        [REF-HUBER-1964], [REF-HUBER-1981],
        [REF-MARONNA-MARTIN-YOHAI-2006].

    """
    sample = coerce_finite_sample(values, name="values")
    n = len(sample)
    if n < 2:
        raise ValueError("at least two observations are required")
    family, c = _check_psi(psi, tuning)
    level = check_level(confidence, name="confidence")
    s = (
        median_absolute_deviation(sample)
        if scale is None
        else require_finite_real(scale, name="scale")
    )
    if s <= 0.0:
        raise ValueError(
            "scale is zero (more than half of the observations coincide); "
            "supply a positive scale"
        )
    mu = median(sample)
    converged = False
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        weights = [_weight((x - mu) / s, family, c) for x in sample]
        total = math.fsum(weights)
        if total <= 0.0:
            raise ValueError("every observation received weight zero")
        updated = math.fsum(w * x for w, x in zip(weights, sample, strict=True)) / total
        step = abs(updated - mu)
        mu = updated
        if step <= tolerance * s:
            converged = True
            break
    u = [(x - mu) / s for x in sample]
    slope = math.fsum(_psi_derivative(value, family, c) for value in u)
    if slope <= 0.0:
        raise ValueError(
            "sum of psi' is not positive; the sandwich variance is undefined"
        )
    spread = math.fsum(_psi(value, family, c) ** 2 for value in u)
    standard_error = s * math.sqrt(n / (n - 1) * spread) / slope
    critical = t_critical(0.5 + 0.5 * level, n - 1)
    interval = ConfidenceInterval(
        mu - critical * standard_error,
        mu + critical * standard_error,
        level,
        f"{family} M-estimate, sandwich standard error, t({n - 1}) reference",
    )
    return RobustLocationResult(
        mu,
        s,
        standard_error,
        interval,
        family,
        c,
        tuple(_weight(value, family, c) for value in u),
        iterations,
        converged,
        n,
    )


@dataclass(frozen=True, slots=True)
class RobustRegressionResult:
    """Regression M-estimate fitted by IRLS with a MAD residual scale.

    Args:
        coefficients: ``beta`` solving ``sum psi(r_i/s) x_i = 0``.
        standard_errors: Square roots of the sandwich covariance diagonal.
        covariance: ``s^2 n/(n-p) A^-1 B A^-1`` with
            ``A = sum psi'(u_i) x_i x_i'`` and ``B = sum psi(u_i)^2 x_i x_i'``.
        confidence_intervals: ``beta_j +- t_{n-p} se_j`` (large-sample).
        column_names: Names of the design columns.
        scale: Residual scale ``s = 1.4826 median |r_i|`` at the solution.
        weights: Final IRLS weights in input order (zero means rejected).
        residuals: ``y_i - x_i' beta``.
        fitted_values: ``x_i' beta``.
        psi: ``"huber"`` or ``"bisquare"``.
        tuning: Tuning constant used.
        start: ``"ls"`` or ``"lad"``, the starting fit.
        iterations: IRLS iterations performed.
        converged: Whether the residual change fell below the tolerance.

    References:
        [EQ-INF-ROB-003], [REF-HUBER-1981],
        [REF-MARONNA-MARTIN-YOHAI-2006].

    """

    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    covariance: Matrix
    confidence_intervals: tuple[ConfidenceInterval, ...]
    column_names: tuple[str, ...]
    scale: float
    weights: tuple[float, ...]
    residuals: tuple[float, ...]
    fitted_values: tuple[float, ...]
    psi: str
    tuning: float
    start: str
    iterations: int
    converged: bool

    def coefficient(self, name: str) -> float:
        """Return one coefficient by column name.

        Args:
            name: Column name.

        Returns:
            The coefficient.

        Raises:
            KeyError: If the name is not a column.

        References:
            [EQ-INF-ROB-003]. No equation is evaluated.

        """
        if name not in self.column_names:
            raise KeyError(name)
        return self.coefficients[self.column_names.index(name)]


def _weighted_least_squares(
    rows: Matrix, response: Vector, weights: Sequence[float], names: tuple[str, ...]
) -> Vector:
    """Solve the weighted least-squares problem on the positive-weight rows.

    References:
        [EQ-LINALG-009], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    kept = [i for i, w in enumerate(weights) if w > 0.0]
    if len(kept) < len(names):
        raise ValueError("fewer positive-weight observations than coefficients")
    roots = {i: math.sqrt(weights[i]) for i in kept}
    factor = householder_qr(tuple(tuple(roots[i] * v for v in rows[i]) for i in kept))
    if factor.rank_deficient:
        culprits = ", ".join(names[j] for j in factor.rank_deficient)
        raise RankDeficientDesignError(
            f"weighted design is rank deficient; column(s) {culprits}"
        )
    return factor.solve(tuple(roots[i] * response[i] for i in kept))


def _residuals(rows: Matrix, response: Vector, beta: Vector) -> tuple[float, ...]:
    """Return ``y_i - x_i' beta``.

    References:
        [EQ-INF-ROB-003]. No new equation.

    """
    return tuple(
        y - math.fsum(b * x for b, x in zip(beta, row, strict=True))
        for row, y in zip(rows, response, strict=True)
    )


def _residual_scale(residuals: Sequence[float]) -> float:
    """Return the normalized MAD of residuals about zero, ``1.4826 med |r|``.

    References:
        [EQ-INF-ROB-003], [REF-MARONNA-MARTIN-YOHAI-2006].

    """
    return _MAD_NORMAL * median(abs(r) for r in residuals)


def _inverse(matrix: Matrix) -> Matrix:
    """Invert a small square matrix column by column with pivoted LU.

    References:
        [EQ-LINALG-004], [REF-GOLUB-VANLOAN-2013, Section 3.4].

    """
    size = len(matrix)
    columns = [
        lu_solve(matrix, tuple(1.0 if i == j else 0.0 for i in range(size)))
        for j in range(size)
    ]
    return tuple(tuple(columns[j][i] for j in range(size)) for i in range(size))


def fit_robust_linear_model(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    psi: PsiFunction = "huber",
    tuning: float | None = None,
    start: RobustStart = "auto",
    column_names: Sequence[str] | None = None,
    confidence: float = 0.95,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> RobustRegressionResult:
    """Fit a linear regression M-estimator by IRLS with a MAD residual scale.

    Each iteration sets ``s = 1.4826 median |r_i|``, the weights
    ``w_i = psi(r_i/s)/(r_i/s)``, and refits weighted least squares (the
    scheme of ``MASS::rlm`` with ``scale.est = "MAD"``). The fixed point
    solves ``sum psi(r_i/s) x_i = 0`` with ``s`` the MAD of its own
    residuals. The default start is least squares for the convex Huber loss
    and least absolute deviations (median regression) for the redescending
    biweight, whose root depends on the start.

    Args:
        design: :class:`DesignMatrix` or ``n x p`` rows; include a column of
            ones for an intercept.
        response: ``n`` finite responses.
        psi: ``"huber"`` or ``"bisquare"``.
        tuning: ``k`` or ``c``; ``None`` uses 1.345 or 4.685.
        start: ``"auto"``, ``"ls"``, or ``"lad"``.
        column_names: Names for a raw design; defaults to ``x0, x1, ...``.
        confidence: Level of the coefficient intervals.
        tolerance: Stop when ``||r_new - r_old|| <= tolerance ||r_old||``.
        max_iterations: Iteration cap.

    Returns:
        :class:`RobustRegressionResult`.

    Raises:
        RankDeficientDesignError: If the (weighted) design is rank deficient.
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``n <= p``, the residual scale is
            zero (an exact fit of more than half the data), the sandwich
            matrix ``A`` is singular, or an option is invalid. A residual
            scale below ``1e-12 max|y|`` counts as zero.

    Equation:
        [EQ-INF-ROB-001], [EQ-INF-ROB-003].

    References:
        [REF-HUBER-1981], [REF-BEATON-TUKEY-1974],
        [REF-MARONNA-MARTIN-YOHAI-2006].

    """
    rows, names, _ = _coerce_design(design, column_names)
    y = as_vector(response, name="response")
    n, p = len(rows), len(rows[0])
    if len(y) != n:
        raise ValueError("response length must equal the number of design rows")
    if n <= p:
        raise ValueError("robust regression needs more observations than coefficients")
    family, c = _check_psi(psi, tuning)
    level = check_level(confidence, name="confidence")
    if start == "auto":
        chosen = "ls" if family == "huber" else "lad"
    elif start in ("ls", "lad"):
        chosen = start
    else:
        raise ValueError("start must be 'auto', 'ls', or 'lad'")
    if chosen == "ls":
        beta = _weighted_least_squares(rows, y, [1.0] * n, names)
    else:
        beta = fit_quantile_regression(rows, y, quantile=0.5).coefficients
    residuals = _residuals(rows, y, beta)
    floor = 1e-12 * max(abs(v) for v in y)
    converged = False
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        s = _residual_scale(residuals)
        if s <= floor:
            raise ValueError(
                "residual scale is zero: more than half of the data are fitted exactly"
            )
        weights = [_weight(r / s, family, c) for r in residuals]
        beta = _weighted_least_squares(rows, y, weights, names)
        updated = _residuals(rows, y, beta)
        change = math.sqrt(
            math.fsum((a - b) ** 2 for a, b in zip(updated, residuals, strict=True))
        )
        size = math.sqrt(math.fsum(r * r for r in residuals))
        residuals = updated
        if change <= tolerance * max(size, 1e-300):
            converged = True
            break
    s = _residual_scale(residuals)
    if s <= floor:
        raise ValueError(
            "residual scale is zero: more than half of the data are fitted exactly"
        )
    u = [r / s for r in residuals]
    a_matrix = [[0.0] * p for _ in range(p)]
    b_matrix = [[0.0] * p for _ in range(p)]
    for row, value in zip(rows, u, strict=True):
        slope = _psi_derivative(value, family, c)
        score2 = _psi(value, family, c) ** 2
        for i in range(p):
            for j in range(p):
                a_matrix[i][j] += slope * row[i] * row[j]
                b_matrix[i][j] += score2 * row[i] * row[j]
    try:
        a_inverse = _inverse(tuple(tuple(r) for r in a_matrix))
    except ValueError as error:
        raise ValueError(
            "sum psi'(u_i) x_i x_i' is singular; the sandwich covariance is undefined"
        ) from error
    factor = s * s * n / (n - p)
    middle = [
        [
            math.fsum(b_matrix[i][k] * a_inverse[k][j] for k in range(p))
            for j in range(p)
        ]
        for i in range(p)
    ]
    covariance = tuple(
        tuple(
            factor * math.fsum(a_inverse[i][k] * middle[k][j] for k in range(p))
            for j in range(p)
        )
        for i in range(p)
    )
    covariance = tuple(
        tuple(0.5 * (covariance[i][j] + covariance[j][i]) for j in range(p))
        for i in range(p)
    )
    if any(covariance[j][j] <= 0.0 for j in range(p)):
        raise ValueError("sandwich covariance has a non-positive diagonal")
    errors = tuple(math.sqrt(covariance[j][j]) for j in range(p))
    critical = t_critical(0.5 + 0.5 * level, n - p)
    label = f"{family} M-regression, sandwich standard error, t({n - p}) reference"
    intervals = tuple(
        ConfidenceInterval(b - critical * e, b + critical * e, level, label)
        for b, e in zip(beta, errors, strict=True)
    )
    return RobustRegressionResult(
        tuple(beta),
        errors,
        covariance,
        intervals,
        names,
        s,
        tuple(_weight(value, family, c) for value in u),
        residuals,
        tuple(obs - r for obs, r in zip(y, residuals, strict=True)),
        family,
        c,
        chosen,
        iterations,
        converged,
    )


@dataclass(frozen=True, slots=True)
class TheilSenResult:
    """Theil-Sen slope, intercept, and Sen's distribution-free slope interval.

    Args:
        slope: Median of the pairwise slopes ``(y_j - y_i)/(x_j - x_i)``.
        intercept: ``median(y_i - slope x_i)``.
        confidence_interval: Sen's interval from order statistics of the
            pairwise slopes.
        pair_count: Number ``N`` of pairs with distinct ``x``.
        method: Label of the construction.

    References:
        [EQ-INF-ROB-004], [REF-SEN-1968],
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 9].

    """

    slope: float
    intercept: float
    confidence_interval: ConfidenceInterval
    pair_count: int
    method: str


def theil_sen_slope(
    x: Iterable[object],
    y: Iterable[object],
    *,
    confidence: float = 0.95,
) -> TheilSenResult:
    """Estimate a simple-regression slope by the median of pairwise slopes.

    The interval inverts Kendall's test of ``beta`` on ``(x, y - beta x)``:
    with ``N`` pairwise slopes sorted, ``C = z_{1-alpha/2} sigma`` where
    ``sigma^2 = [n(n-1)(2n+5) - sum_t t(t-1)(2t+5)]/18`` corrects for ties
    in ``x``, the limits are the slopes of 1-based ranks
    ``round((N - C)/2)`` and ``round((N + C)/2) + 1`` (clipped to
    ``[1, N]``). Ties in ``y`` are not corrected; for continuous responses
    they have probability zero.

    Args:
        x: Predictor values (at least two distinct).
        y: Responses of equal length.
        confidence: Level of the slope interval.

    Returns:
        :class:`TheilSenResult`.

    Raises:
        TypeError: If a value is not real.
        ValueError: If lengths differ, fewer than two observations are given,
            or all ``x`` are equal.

    Equation:
        [EQ-INF-ROB-004].

    References:
        [REF-SEN-1968], [REF-KENDALL-1938],
        [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 9].

    """
    xs = coerce_finite_sample(x, name="x")
    ys = coerce_finite_sample(y, name="y")
    if len(xs) != len(ys):
        raise ValueError("x and y must have equal lengths")
    n = len(xs)
    if n < 2:
        raise ValueError("at least two observations are required")
    level = check_level(confidence, name="confidence")
    slopes = sorted(
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(n)
        for j in range(i + 1, n)
        if xs[j] != xs[i]
    )
    count = len(slopes)
    if count == 0:
        raise ValueError("all x values are equal; the slope is undefined")
    slope = median(slopes)
    intercept = median(yv - slope * xv for xv, yv in zip(xs, ys, strict=True))
    tie_groups: dict[float, int] = {}
    for value in xs:
        tie_groups[value] = tie_groups.get(value, 0) + 1
    variance = (
        n * (n - 1) * (2 * n + 5)
        - sum(t * (t - 1) * (2 * t + 5) for t in tie_groups.values())
    ) / 18.0
    spread = NormalDist().inv_cdf(0.5 + 0.5 * level) * math.sqrt(variance)
    lower_index = min(max(round((count - spread) / 2.0) - 1, 0), count - 1)
    upper_index = min(max(round((count + spread) / 2.0), 0), count - 1)
    interval = ConfidenceInterval(
        slopes[lower_index],
        slopes[upper_index],
        level,
        "Sen interval from Kendall's tau (x-tie corrected)",
    )
    return TheilSenResult(slope, intercept, interval, count, "Theil-Sen slope")
