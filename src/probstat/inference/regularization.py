"""Ridge, lasso, and elastic-net regression with K-fold cross-validation.

Predictors are centred (when an intercept is fitted) and, by default, scaled
to unit population standard deviation before penalization, so the penalty
treats every predictor alike; coefficients are reported on the original scale.
The intercept is never penalized.

Two penalty scalings are used, each matching its standard source:

- Ridge (closed form, Hoerl-Kennard/ESL): minimize
  ``||y_c - X_s b||^2 + lambda ||b||^2``.
- Elastic net (coordinate descent, glmnet): minimize
  ``(1/2n) ||y_c - X_s b||^2 + lambda ((1-alpha)/2 ||b||^2 + alpha ||b||_1)``.

So the elastic net with ``alpha = 0`` and penalty ``lambda`` equals ridge with
penalty ``n lambda``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, as_matrix, as_vector, householder_qr

from .linear_models import DesignMatrix, RankDeficientDesignError


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Centred and scaled predictor columns with their transformation.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.4].

    """

    columns: tuple[Vector, ...]
    response: Vector
    means: Vector
    scales: Vector
    response_mean: float
    names: tuple[str, ...]


def _predictor_rows(
    predictors: DesignMatrix | Iterable[Iterable[object]],
    column_names: Sequence[str] | None,
) -> tuple[Matrix, tuple[str, ...]]:
    """Return predictor rows without any intercept column, and their names.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(predictors, DesignMatrix):
        rows = predictors.rows
        names = predictors.column_names
        if predictors.intercept:
            rows = tuple(row[1:] for row in rows)
            names = names[1:]
        if not rows or not rows[0]:
            raise ValueError("design has no predictors besides the intercept")
        return rows, names
    rows = as_matrix(predictors, name="predictors")
    width = len(rows[0])
    names = (
        tuple(column_names)
        if column_names is not None
        else tuple(f"x{j}" for j in range(width))
    )
    if len(names) != width:
        raise ValueError("column_names must match the number of predictors")
    return rows, names


def _prepare(
    rows: Matrix,
    names: tuple[str, ...],
    response: Vector,
    *,
    intercept: bool,
    standardize: bool,
) -> _Prepared:
    """Centre and scale predictors; centre the response when fitting an intercept.

    References:
        [EQ-PEN-STD-001], [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2].

    """
    n = len(rows)
    if len(response) != n:
        raise ValueError("response length must equal the number of rows")
    if n < 2:
        raise ValueError("at least two observations are required")
    columns: list[Vector] = []
    means: list[float] = []
    scales: list[float] = []
    for j, name in enumerate(names):
        column = tuple(row[j] for row in rows)
        mean = math.fsum(column) / n if intercept else 0.0
        centred = tuple(value - mean for value in column)
        spread = math.sqrt(math.fsum(value * value for value in centred) / n)
        if spread == 0.0:
            raise ValueError(f"predictor {name!r} has zero variation")
        scale = spread if standardize else 1.0
        columns.append(tuple(value / scale for value in centred))
        means.append(mean)
        scales.append(scale)
    response_mean = math.fsum(response) / n if intercept else 0.0
    return _Prepared(
        columns=tuple(columns),
        response=tuple(value - response_mean for value in response),
        means=tuple(means),
        scales=tuple(scales),
        response_mean=response_mean,
        names=names,
    )


def _non_negative(value: object, name: str) -> float:
    """Validate a finite non-negative penalty.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _original_scale(prepared: _Prepared, standardized: Vector) -> tuple[float, Vector]:
    """Map standardized coefficients back to the intercept and raw slopes.

    References:
        [EQ-PEN-STD-001], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.4].

    """
    slopes = tuple(b / s for b, s in zip(standardized, prepared.scales, strict=True))
    intercept = prepared.response_mean - math.fsum(
        b * m for b, m in zip(slopes, prepared.means, strict=True)
    )
    return intercept, slopes


def _linear_predict(
    intercept: float, slopes: Vector, rows: Iterable[Iterable[object]]
) -> Vector:
    """Evaluate ``b0 + x' b`` on new predictor rows.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.2]. No new equation.

    """
    checked = as_matrix(rows, name="rows")
    if len(checked[0]) != len(slopes):
        raise ValueError("new rows must have one entry per predictor")
    return tuple(
        intercept + math.fsum(b * x for b, x in zip(slopes, row, strict=True))
        for row in checked
    )


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RidgeResult:
    """Closed-form ridge fit.

    Attributes:
        intercept: Unpenalized intercept on the original scale.
        coefficients: Slopes on the original predictor scale.
        standardized_coefficients: Slopes on the penalized (centred/scaled)
            scale.
        column_names: Predictor names.
        penalty: ``lambda`` in ``||y_c - X_s b||^2 + lambda ||b||^2``.
        effective_degrees_of_freedom: ``tr H_lambda = sum d_j^2/(d_j^2 +
            lambda)`` (intercept excluded).
        residual_sum_squares: ``sum (y - yhat)^2``.
        fitted_values: In-sample fitted values.
        predictor_means: Centring constants.
        predictor_scales: Scaling constants.

    References:
        [REF-HOERL-KENNARD-1970], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009,
        Section 3.4.1].

    """

    intercept: float
    coefficients: Vector
    standardized_coefficients: Vector
    column_names: tuple[str, ...]
    penalty: float
    effective_degrees_of_freedom: float
    residual_sum_squares: float
    fitted_values: Vector
    predictor_means: Vector
    predictor_scales: Vector

    def predict(self, rows: Iterable[Iterable[object]]) -> Vector:
        """Return ``b0 + x' b`` at new predictor rows (no intercept column).

        Args:
            rows: Predictor rows on the original scale.

        Returns:
            Predictions.

        Raises:
            ValueError: If a row has the wrong length.

        Equation:
            [EQ-PEN-RIDGE-001].

        References:
            [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.4.1].

        """
        return _linear_predict(self.intercept, self.coefficients, rows)


def fit_ridge(
    predictors: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    penalty: float,
    intercept: bool = True,
    standardize: bool = True,
    column_names: Sequence[str] | None = None,
) -> RidgeResult:
    """Fit ridge regression in closed form through an augmented QR.

    The minimizer of ``||y_c - X_s b||^2 + lambda ||b||^2`` is the least-squares
    solution of the stacked system ``[X_s; sqrt(lambda) I] b = [y_c; 0]``,
    i.e. ``b = (X_s'X_s + lambda I)^-1 X_s'y_c``. With ``R`` from that QR,
    ``R'R = X_s'X_s + lambda I`` and the effective degrees of freedom are
    ``||X_s R^-1||_F^2 = sum_j d_j^2 / (d_j^2 + lambda)``.

    Args:
        predictors: Predictor rows (no intercept column) or a
            :class:`DesignMatrix`, whose intercept column is dropped.
        response: Finite responses.
        penalty: ``lambda >= 0``; zero reproduces ordinary least squares.
        intercept: Fit an unpenalized intercept (centre the data).
        standardize: Scale predictors to unit population standard deviation
            before penalizing.
        column_names: Names for raw predictor rows.

    Returns:
        :class:`RidgeResult`.

    Raises:
        RankDeficientDesignError: If ``penalty == 0`` and the centred
            predictors are rank deficient.
        ValueError: If inputs are invalid or a predictor is constant.

    Equation:
        [EQ-PEN-RIDGE-001], [EQ-PEN-STD-001], [EQ-LINALG-009].

    References:
        [REF-HOERL-KENNARD-1970],
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 3.4.1].

    """
    lam = _non_negative(penalty, "penalty")
    rows, names = _predictor_rows(predictors, column_names)
    y = as_vector(response, name="response")
    prepared = _prepare(rows, names, y, intercept=intercept, standardize=standardize)
    n, p = len(rows), len(names)
    scaled_rows = tuple(
        tuple(prepared.columns[j][i] for j in range(p)) for i in range(n)
    )
    root = math.sqrt(lam)
    if lam > 0.0:
        augmented = scaled_rows + tuple(
            tuple(root if j == k else 0.0 for j in range(p)) for k in range(p)
        )
        target = prepared.response + (0.0,) * p
    else:
        if n - (1 if intercept else 0) < p:
            raise RankDeficientDesignError(
                "ridge with zero penalty needs more observations than predictors"
            )
        augmented, target = scaled_rows, prepared.response
    factor = householder_qr(augmented)
    if factor.rank_deficient:
        culprits = ", ".join(names[j] for j in factor.rank_deficient)
        raise RankDeficientDesignError(
            f"predictors are rank deficient (column(s) {culprits}); "
            "use a positive ridge penalty"
        )
    standardized = factor.solve(target)
    inverse = factor.r_inverse()
    effective = math.fsum(
        math.fsum(row[k] * inverse[k][j] for k in range(p)) ** 2
        for row in scaled_rows
        for j in range(p)
    )
    intercept_value, slopes = _original_scale(prepared, standardized)
    fitted = tuple(
        intercept_value + math.fsum(b * x for b, x in zip(slopes, row, strict=True))
        for row in rows
    )
    rss = math.fsum((yi - fi) ** 2 for yi, fi in zip(y, fitted, strict=True))
    return RidgeResult(
        intercept=intercept_value,
        coefficients=slopes,
        standardized_coefficients=standardized,
        column_names=names,
        penalty=lam,
        effective_degrees_of_freedom=effective,
        residual_sum_squares=rss,
        fitted_values=fitted,
        predictor_means=prepared.means,
        predictor_scales=prepared.scales,
    )


# ---------------------------------------------------------------------------
# Lasso and elastic net by cyclic coordinate descent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ElasticNetResult:
    """Elastic-net (lasso when ``alpha = 1``) fit by coordinate descent.

    Attributes:
        intercept: Unpenalized intercept on the original scale.
        coefficients: Slopes on the original predictor scale.
        standardized_coefficients: Slopes on the penalized scale.
        column_names: Predictor names.
        penalty: ``lambda`` in the glmnet objective.
        alpha: Mixing weight in ``[0, 1]`` (one is the lasso).
        objective: Objective value on the penalized scale.
        iterations: Coordinate sweeps used.
        converged: Whether the largest scaled coefficient change fell below
            tolerance.
        predictor_means: Centring constants.
        predictor_scales: Scaling constants.

    References:
        [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010], [REF-ZOU-HASTIE-2005],
        [REF-TIBSHIRANI-1996].

    """

    intercept: float
    coefficients: Vector
    standardized_coefficients: Vector
    column_names: tuple[str, ...]
    penalty: float
    alpha: float
    objective: float
    iterations: int
    converged: bool
    predictor_means: Vector
    predictor_scales: Vector

    @property
    def active_set(self) -> tuple[str, ...]:
        """Return the names of predictors with non-zero coefficients.

        Returns:
            Names in column order.

        References:
            [REF-TIBSHIRANI-1996]. No equation is evaluated.

        """
        return tuple(
            name
            for name, value in zip(self.column_names, self.coefficients, strict=True)
            if value != 0.0
        )

    def predict(self, rows: Iterable[Iterable[object]]) -> Vector:
        """Return ``b0 + x' b`` at new predictor rows (no intercept column).

        Args:
            rows: Predictor rows on the original scale.

        Returns:
            Predictions.

        Raises:
            ValueError: If a row has the wrong length.

        Equation:
            [EQ-PEN-EN-001].

        References:
            [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010].

        """
        return _linear_predict(self.intercept, self.coefficients, rows)


def _soft_threshold(value: float, threshold: float) -> float:
    """Return ``S(z, gamma) = sign(z) max(|z| - gamma, 0)``.

    References:
        [EQ-PEN-EN-001], [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2.1].

    """
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _coordinate_descent(
    prepared: _Prepared,
    lam: float,
    alpha: float,
    start: Vector,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[Vector, int, bool, float]:
    """Run cyclic coordinate descent with naive residual updates.

    References:
        [EQ-PEN-EN-001], [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2].

    """
    columns = prepared.columns
    n = len(prepared.response)
    p = len(columns)
    curvature = [math.fsum(x * x for x in column) / n for column in columns]
    beta = list(start)
    residual = list(prepared.response)
    for j in range(p):
        if beta[j] != 0.0:
            column = columns[j]
            for i in range(n):
                residual[i] -= column[i] * beta[j]
    response_scale = math.sqrt(math.fsum(v * v for v in prepared.response) / n) or 1.0
    l1 = lam * alpha
    l2 = lam * (1.0 - alpha)
    converged = False
    sweeps = 0
    for sweep in range(1, max_iterations + 1):
        sweeps = sweep
        largest = 0.0
        for j in range(p):
            column = columns[j]
            old = beta[j]
            rho = math.fsum(column[i] * residual[i] for i in range(n)) / n
            rho += curvature[j] * old
            new = _soft_threshold(rho, l1) / (curvature[j] + l2)
            if new != old:
                delta = new - old
                for i in range(n):
                    residual[i] -= column[i] * delta
                beta[j] = new
                largest = max(largest, abs(delta) * math.sqrt(curvature[j]))
        if largest <= tolerance * response_scale:
            converged = True
            break
    objective = 0.5 * math.fsum(r * r for r in residual) / n + lam * (
        0.5 * (1.0 - alpha) * math.fsum(b * b for b in beta)
        + alpha * math.fsum(abs(b) for b in beta)
    )
    return tuple(beta), sweeps, converged, objective


def _validate_alpha(alpha: object) -> float:
    """Validate the elastic-net mixing parameter.

    References:
        [REF-ZOU-HASTIE-2005]. No equation is evaluated.

    """
    checked = require_finite_real(alpha, name="alpha")
    if not 0.0 <= checked <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return checked


def _validate_iterations(
    max_iterations: object, tolerance: object
) -> tuple[int, float]:
    """Validate the iteration limit and tolerance.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    tol = require_finite_real(tolerance, name="tolerance")
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie strictly between zero and one")
    return max_iterations, tol


def _penalty_max(prepared: _Prepared, alpha: float) -> float:
    """Return the smallest penalty at which every coefficient is zero.

    References:
        [EQ-PEN-KKT-001], [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2.5].

    """
    if alpha == 0.0:
        raise ValueError("alpha = 0 has no finite maximal penalty; pass penalties")
    n = len(prepared.response)
    return (
        max(
            abs(
                math.fsum(x * r for x, r in zip(column, prepared.response, strict=True))
            )
            / n
            for column in prepared.columns
        )
        / alpha
    )


def _result(
    prepared: _Prepared,
    standardized: Vector,
    lam: float,
    alpha: float,
    sweeps: int,
    converged: bool,
    objective: float,
) -> ElasticNetResult:
    """Package a coordinate-descent solution.

    References:
        [EQ-PEN-STD-001]. No new equation is evaluated.

    """
    intercept, slopes = _original_scale(prepared, standardized)
    return ElasticNetResult(
        intercept=intercept,
        coefficients=slopes,
        standardized_coefficients=standardized,
        column_names=prepared.names,
        penalty=lam,
        alpha=alpha,
        objective=objective,
        iterations=sweeps,
        converged=converged,
        predictor_means=prepared.means,
        predictor_scales=prepared.scales,
    )


def fit_elastic_net(
    predictors: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    penalty: float,
    alpha: float = 1.0,
    intercept: bool = True,
    standardize: bool = True,
    column_names: Sequence[str] | None = None,
    max_iterations: int = 100_000,
    tolerance: float = 1e-10,
) -> ElasticNetResult:
    """Fit the lasso (``alpha = 1``) or elastic net by cyclic coordinate descent.

    Minimizes ``(1/2n)||y_c - X_s b||^2 + lambda((1-alpha)/2 ||b||^2 +
    alpha ||b||_1)``. Each coordinate update is
    ``b_j <- S(rho_j, lambda alpha) / (v_j + lambda (1 - alpha))`` with
    ``rho_j = x_j'r_(j)/n``, the partial-residual correlation, and
    ``v_j = x_j'x_j/n`` (one for standardized predictors).

    Args:
        predictors: Predictor rows (no intercept column) or a
            :class:`DesignMatrix`, whose intercept column is dropped.
        response: Finite responses.
        penalty: ``lambda >= 0``.
        alpha: Mixing parameter in ``[0, 1]``.
        intercept: Fit an unpenalized intercept.
        standardize: Scale predictors to unit population standard deviation.
        column_names: Names for raw predictor rows.
        max_iterations: Maximum number of full coordinate sweeps.
        tolerance: Stop when every ``sqrt(v_j)|delta b_j|`` is below
            ``tolerance`` times the response scale.

    Returns:
        :class:`ElasticNetResult`.

    Raises:
        ValueError: If inputs are invalid or a predictor is constant.

    Equation:
        [EQ-PEN-EN-001], [EQ-PEN-STD-001], [EQ-PEN-KKT-001].

    References:
        [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010], [REF-TIBSHIRANI-1996],
        [REF-ZOU-HASTIE-2005].

    """
    lam = _non_negative(penalty, "penalty")
    mix = _validate_alpha(alpha)
    limit, tol = _validate_iterations(max_iterations, tolerance)
    rows, names = _predictor_rows(predictors, column_names)
    y = as_vector(response, name="response")
    prepared = _prepare(rows, names, y, intercept=intercept, standardize=standardize)
    if lam == 0.0 and mix > 0.0 and len(rows) <= len(names):
        raise ValueError("an unpenalized fit needs more observations than predictors")
    beta, sweeps, converged, objective = _coordinate_descent(
        prepared,
        lam,
        mix,
        (0.0,) * len(names),
        max_iterations=limit,
        tolerance=tol,
    )
    return _result(prepared, beta, lam, mix, sweeps, converged, objective)


def _penalty_grid(
    prepared: _Prepared,
    alpha: float,
    penalties: Iterable[object] | None,
    n_penalties: int,
    penalty_min_ratio: float,
) -> Vector:
    """Return a strictly decreasing penalty grid.

    References:
        [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2.5]. No new equation.

    """
    if penalties is not None:
        grid = tuple(sorted(set(as_vector(penalties, name="penalties")), reverse=True))
        if not grid:
            raise ValueError("penalties must be non-empty")
        if grid[-1] < 0.0:
            raise ValueError("penalties must be non-negative")
        return grid
    if isinstance(n_penalties, bool) or not isinstance(n_penalties, int):
        raise TypeError("n_penalties must be an integer")
    if n_penalties < 2:
        raise ValueError("n_penalties must be at least two")
    ratio = require_finite_real(penalty_min_ratio, name="penalty_min_ratio")
    if not 0.0 < ratio < 1.0:
        raise ValueError("penalty_min_ratio must lie strictly between zero and one")
    top = _penalty_max(prepared, alpha)
    if top == 0.0:
        raise ValueError("response is uncorrelated with every predictor")
    return tuple(top * ratio ** (k / (n_penalties - 1)) for k in range(n_penalties))


def _path(
    prepared: _Prepared,
    alpha: float,
    grid: Vector,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[ElasticNetResult, ...]:
    """Fit a warm-started path along a decreasing penalty grid.

    References:
        [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2.5].

    """
    start: Vector = (0.0,) * len(prepared.columns)
    fits: list[ElasticNetResult] = []
    for lam in grid:
        beta, sweeps, converged, objective = _coordinate_descent(
            prepared,
            lam,
            alpha,
            start,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        fits.append(_result(prepared, beta, lam, alpha, sweeps, converged, objective))
        start = beta
    return tuple(fits)


def elastic_net_path(
    predictors: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    alpha: float = 1.0,
    penalties: Iterable[object] | None = None,
    n_penalties: int = 50,
    penalty_min_ratio: float = 1e-3,
    intercept: bool = True,
    standardize: bool = True,
    column_names: Sequence[str] | None = None,
    max_iterations: int = 100_000,
    tolerance: float = 1e-10,
) -> tuple[ElasticNetResult, ...]:
    """Fit elastic-net solutions along a decreasing penalty path.

    The default grid is log-spaced from ``lambda_max = max_j |x_j'y_c|/(n
    alpha)``, the smallest penalty with an all-zero solution, down to
    ``penalty_min_ratio * lambda_max``. Each fit is warm-started at the
    previous solution.

    Args:
        predictors: Predictor rows or a :class:`DesignMatrix`.
        response: Finite responses.
        alpha: Mixing parameter in ``[0, 1]``; ``alpha = 0`` requires explicit
            ``penalties``.
        penalties: Optional explicit penalties (sorted decreasingly).
        n_penalties: Number of grid points when ``penalties`` is omitted.
        penalty_min_ratio: Smallest grid penalty relative to ``lambda_max``.
        intercept: Fit an unpenalized intercept.
        standardize: Scale predictors to unit population standard deviation.
        column_names: Names for raw predictor rows.
        max_iterations: Maximum coordinate sweeps per penalty.
        tolerance: Coordinate-descent tolerance.

    Returns:
        Fits in decreasing penalty order.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        [EQ-PEN-EN-001], [EQ-PEN-KKT-001].

    References:
        [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 2.5].

    """
    mix = _validate_alpha(alpha)
    limit, tol = _validate_iterations(max_iterations, tolerance)
    rows, names = _predictor_rows(predictors, column_names)
    y = as_vector(response, name="response")
    prepared = _prepare(rows, names, y, intercept=intercept, standardize=standardize)
    grid = _penalty_grid(prepared, mix, penalties, n_penalties, penalty_min_ratio)
    return _path(prepared, mix, grid, max_iterations=limit, tolerance=tol)


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """K-fold cross-validation of an elastic-net penalty path.

    Attributes:
        penalties: Decreasing penalty grid.
        mean_error: Mean held-out squared error per penalty (average of the
            fold mean squared errors).
        standard_error: Standard error of ``mean_error`` across folds.
        fold_errors: ``fold_errors[k][m]`` is the mean squared error of fold
            ``k`` at penalty ``m``.
        fold_assignment: Fold index of every observation.
        penalty_min: Penalty minimizing ``mean_error``.
        penalty_1se: Largest penalty within one standard error of the minimum.
        path: Full-data fits along ``penalties``.
        best: Full-data fit at ``penalty_min``.
        one_standard_error: Full-data fit at ``penalty_1se``.

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 7.10],
        [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010].

    """

    penalties: Vector
    mean_error: Vector
    standard_error: Vector
    fold_errors: tuple[Vector, ...]
    fold_assignment: tuple[int, ...]
    penalty_min: float
    penalty_1se: float
    path: tuple[ElasticNetResult, ...]
    best: ElasticNetResult
    one_standard_error: ElasticNetResult


def kfold_assignment(
    size: int, folds: int, seed: int | RandomStream | None = 0
) -> tuple[int, ...]:
    """Return balanced, randomly permuted fold labels ``0, ..., K-1``.

    Observation ``perm[i]`` of a seeded permutation receives fold ``i mod K``,
    so fold sizes differ by at most one.

    Args:
        size: Number of observations ``n >= 2``.
        folds: Number of folds ``2 <= K <= n``.
        seed: Integer, :class:`RandomStream`, or ``None`` (not reproducible).
            The permutation uses the stream's ``child("cv-folds")``.

    Returns:
        Fold label per observation.

    Raises:
        ValueError: If ``size`` or ``folds`` is invalid.

    Equation:
        [EQ-PEN-CV-001].

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 7.10],
        [REF-STEELE-LEA-FLOOD-2014].

    """
    for value, name in ((size, "size"), (folds, "folds")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if size < 2:
        raise ValueError("size must be at least two")
    if not 2 <= folds <= size:
        raise ValueError("folds must lie between 2 and the number of observations")
    generator = resolve_stream(seed).child("cv-folds").generator()
    order = list(range(size))
    generator.shuffle(order)
    labels = [0] * size
    for position, index in enumerate(order):
        labels[index] = position % folds
    return tuple(labels)


def cross_validate_elastic_net(
    predictors: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    alpha: float = 1.0,
    penalties: Iterable[object] | None = None,
    n_penalties: int = 30,
    penalty_min_ratio: float = 1e-3,
    folds: int = 5,
    seed: int | RandomStream | None = 0,
    intercept: bool = True,
    standardize: bool = True,
    column_names: Sequence[str] | None = None,
    max_iterations: int = 100_000,
    tolerance: float = 1e-9,
) -> CrossValidationResult:
    """Choose an elastic-net penalty by K-fold cross-validation.

    The penalty grid is computed once on the full data. For each fold, a
    warm-started path is fitted on the other folds (re-standardizing on the
    training part only) and scored by mean squared prediction error on the
    held-out fold. ``CV(lambda) = (1/K) sum_k MSE_k(lambda)`` with standard
    error ``sd_k(MSE_k)/sqrt(K)``; ``penalty_1se`` applies the one-standard-
    error rule.

    Args:
        predictors: Predictor rows or a :class:`DesignMatrix`.
        response: Finite responses.
        alpha: Mixing parameter in ``[0, 1]``; ``alpha = 0`` needs
            ``penalties``.
        penalties: Optional explicit penalty grid.
        n_penalties: Grid size when ``penalties`` is omitted.
        penalty_min_ratio: Smallest grid penalty relative to ``lambda_max``.
        folds: Number of folds ``K``.
        seed: Seed or :class:`RandomStream` for the fold permutation.
        intercept: Fit an unpenalized intercept.
        standardize: Scale predictors to unit population standard deviation.
        column_names: Names for raw predictor rows.
        max_iterations: Maximum coordinate sweeps per fit.
        tolerance: Coordinate-descent tolerance.

    Returns:
        :class:`CrossValidationResult`.

    Raises:
        ValueError: If inputs are invalid or a training fold has a constant
            predictor.

    Equation:
        [EQ-PEN-CV-001], [EQ-PEN-EN-001].

    References:
        [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 7.10],
        [REF-FRIEDMAN-HASTIE-TIBSHIRANI-2010, Section 3].

    """
    mix = _validate_alpha(alpha)
    limit, tol = _validate_iterations(max_iterations, tolerance)
    rows, names = _predictor_rows(predictors, column_names)
    y = as_vector(response, name="response")
    prepared = _prepare(rows, names, y, intercept=intercept, standardize=standardize)
    grid = _penalty_grid(prepared, mix, penalties, n_penalties, penalty_min_ratio)
    labels = kfold_assignment(len(rows), folds, seed)
    fold_errors: list[Vector] = []
    for fold in range(folds):
        train = [i for i, label in enumerate(labels) if label != fold]
        test = [i for i, label in enumerate(labels) if label == fold]
        train_prepared = _prepare(
            tuple(rows[i] for i in train),
            names,
            tuple(y[i] for i in train),
            intercept=intercept,
            standardize=standardize,
        )
        fits = _path(train_prepared, mix, grid, max_iterations=limit, tolerance=tol)
        test_rows = tuple(rows[i] for i in test)
        errors = []
        for fit in fits:
            predictions = fit.predict(test_rows)
            errors.append(
                math.fsum(
                    (y[i] - value) ** 2
                    for i, value in zip(test, predictions, strict=True)
                )
                / len(test)
            )
        fold_errors.append(tuple(errors))
    means: list[float] = []
    standard_errors: list[float] = []
    for m in range(len(grid)):
        values = [fold_errors[k][m] for k in range(folds)]
        mean = math.fsum(values) / folds
        variance = math.fsum((v - mean) ** 2 for v in values) / (folds - 1)
        means.append(mean)
        standard_errors.append(math.sqrt(variance / folds))
    best_index = min(range(len(grid)), key=lambda m: (means[m], m))
    threshold = means[best_index] + standard_errors[best_index]
    one_se_index = min(m for m in range(len(grid)) if means[m] <= threshold)
    path = _path(prepared, mix, grid, max_iterations=limit, tolerance=tol)
    return CrossValidationResult(
        penalties=grid,
        mean_error=tuple(means),
        standard_error=tuple(standard_errors),
        fold_errors=tuple(fold_errors),
        fold_assignment=labels,
        penalty_min=grid[best_index],
        penalty_1se=grid[one_se_index],
        path=path,
        best=path[best_index],
        one_standard_error=path[one_se_index],
    )
