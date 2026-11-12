"""Gaussian additive models and partial dependence.

``y_i = alpha + sum_j f_j(x_ij) + e_i`` with each ``f_j`` a P-spline
(EQ-SMOOTH-002) constrained to ``sum_i f_j(x_ij) = 0`` so that the intercept
is identified (EQ-GAM-001).

- :func:`fit_additive_model` solves the joint penalized least-squares
  problem with one smoothing parameter per term, chosen by REML or GCV
  (EQ-SMOOTH-004, EQ-SMOOTH-005).
- :func:`backfit_additive_model` runs the backfitting algorithm with fixed
  smoothing parameters (EQ-GAM-002); for these symmetric penalized
  smoothers it converges to the joint penalized solution.
- :func:`partial_dependence` averages any prediction function over the data
  with one feature held at each grid value (EQ-GAM-003).

References:
    [REF-WOOD-2017, Chapter 4], [REF-HASTIE-TIBSHIRANI-1990],
    [REF-BUJA-HASTIE-TIBSHIRANI-1989], [REF-FRIEDMAN-2001],
    [REF-EILERS-MARX-1996].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, dot, matvec

from ._penalized import PenalizedProblem, check_method, select_smoothing
from .splines import (
    SmoothBand,
    _band,
    bspline_design_matrix,
    difference_penalty,
    uniform_knots,
)

__all__ = [
    "AdditiveModelFit",
    "AdditiveTerm",
    "PartialDependence",
    "backfit_additive_model",
    "fit_additive_model",
    "partial_dependence",
]


@dataclass(frozen=True, slots=True)
class AdditiveTerm:
    """One smooth term ``f_j(x) = sum_k beta_jk B_jk(x)`` of an additive model.

    Args:
        name: Predictor name.
        knots: Knot vector of the B-spline basis.
        degree: B-spline degree.
        penalty_order: Difference order of the penalty.
        coefficients: Coefficients in the full B-spline basis; they satisfy
            ``sum_i f_j(x_ij) = 0`` over the fitting data.
        covariance: Bayesian covariance of ``coefficients``.
        smoothing: Smoothing parameter ``lambda_j``.
        edf: Effective degrees of freedom of the term.

    References:
        [EQ-GAM-001], [REF-WOOD-2017, Chapter 4].

    """

    name: str
    knots: tuple[float, ...]
    degree: int
    penalty_order: int
    coefficients: Vector
    covariance: Matrix
    smoothing: float
    edf: float

    def evaluate(self, points: Iterable[float]) -> Vector:
        """Return ``f_j`` at points inside the knot domain.

        Args:
            points: Predictor values.

        Returns:
            Term values (centred over the fitting data).

        Raises:
            ValueError: For a point outside the knot domain.

        References:
            [EQ-SMOOTH-001], [EQ-GAM-001].

        """
        return matvec(
            bspline_design_matrix(points, self.knots, self.degree), self.coefficients
        )

    def band(self, points: Iterable[float], level: float = 0.95) -> SmoothBand:
        """Return the pointwise Bayesian band of the centred term.

        Args:
            points: Predictor values inside the knot domain.
            level: Nominal pointwise coverage.

        Returns:
            Band of ``f_j`` (the intercept's uncertainty is excluded).

        References:
            [EQ-SMOOTH-006], [REF-WOOD-2017, Chapter 6], [REF-NYCHKA-1988].

        """
        xs = coerce_finite_sample(points, name="points")
        design = bspline_design_matrix(xs, self.knots, self.degree)
        estimate = matvec(design, self.coefficients)
        errors = tuple(
            math.sqrt(max(dot(row, matvec(self.covariance, row)), 0.0))
            for row in design
        )
        return _band(xs, estimate, errors, level)


@dataclass(frozen=True, slots=True)
class AdditiveModelFit:
    """A fitted Gaussian additive model.

    Args:
        intercept: ``alpha``, the mean response (for unit weights).
        intercept_standard_error: Posterior standard deviation of ``alpha``.
        terms: Smooth terms in input order.
        method: ``"reml"``, ``"gcv"``, ``"fixed"``, or ``"backfitting"``.
        criterion: Selection criterion at the final smoothing parameters
            (GCV for fixed or backfitted parameters).
        edf: Total effective degrees of freedom (including the intercept).
        scale: Residual variance ``RSS / (n - edf)``.
        fitted: Fitted values.
        residuals: ``y - fitted``.
        iterations: Backfitting sweeps (zero for the joint fit).

    References:
        [EQ-GAM-001], [EQ-GAM-002], [REF-WOOD-2017, Chapter 4].

    """

    intercept: float
    intercept_standard_error: float
    terms: tuple[AdditiveTerm, ...]
    method: str
    criterion: float
    edf: float
    scale: float
    fitted: Vector
    residuals: Vector
    iterations: int

    def term(self, name: str) -> AdditiveTerm:
        """Return the term of one predictor.

        Raises:
            KeyError: If no term has this name.

        References:
            [EQ-GAM-001]. No further equation is evaluated.

        """
        for item in self.terms:
            if item.name == name:
                return item
        raise KeyError(name)

    def predict(self, data: Mapping[str, Iterable[float]]) -> Vector:
        """Return ``alpha + sum_j f_j(x_j)`` for new rows.

        Args:
            data: One column per term name, of equal length, inside each
                term's knot domain.

        Returns:
            Predictions.

        Raises:
            ValueError: For missing columns, unequal lengths, or extrapolation.

        References:
            [EQ-GAM-001], [REF-HASTIE-TIBSHIRANI-1990].

        """
        columns = {item.name: item.evaluate(data[item.name]) for item in self.terms}
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError("all columns must have the same length")
        size = lengths.pop()
        return tuple(
            self.intercept + math.fsum(columns[item.name][i] for item in self.terms)
            for i in range(size)
        )


@dataclass(frozen=True, slots=True)
class PartialDependence:
    """Partial-dependence curve of one feature.

    Args:
        feature: Feature name.
        grid: Feature values.
        values: Average prediction with the feature held at each value.

    References:
        [EQ-GAM-003], [REF-FRIEDMAN-2001].

    """

    feature: str
    grid: Vector
    values: Vector


@dataclass(frozen=True, slots=True)
class _TermBasis:
    """Basis, constraint, and penalty of one term.

    References:
        [EQ-GAM-001], [REF-WOOD-2017, Chapter 4].

    """

    name: str
    knots: tuple[float, ...]
    full_design: Matrix
    design: Matrix
    constraint: Matrix
    penalty: Matrix
    rank: int


def _columns(
    data: Mapping[str, Iterable[float]],
    y: Iterable[float],
    weights: Iterable[float] | None,
) -> tuple[dict[str, Vector], Vector, Vector]:
    """Validate the predictor columns, response, and weights.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if not data:
        raise ValueError("data must contain at least one predictor")
    columns = {
        str(name): coerce_finite_sample(values, name=str(name))
        for name, values in data.items()
    }
    ys = coerce_finite_sample(y, name="y")
    if any(len(values) != len(ys) for values in columns.values()):
        raise ValueError("every column must have the same length as y")
    if weights is None:
        ws: Vector = (1.0,) * len(ys)
    else:
        ws = coerce_finite_sample(weights, name="weights")
        if len(ws) != len(ys) or any(w <= 0.0 for w in ws):
            raise ValueError("weights must be positive and match y in length")
    return columns, ys, ws


def _sum_to_zero_basis(design: Matrix) -> Matrix:
    """Return ``Z`` (``k x (k-1)``) with orthonormal columns orthogonal to ``1'B``.

    Uses the Householder reflection ``H = I - 2 v v'/(v'v)`` that maps the
    column sums ``c = B'1`` to a multiple of ``e_1``; the last ``k - 1``
    columns of ``H`` span the coefficient vectors with ``c' beta = 0``.

    References:
        [REF-WOOD-2017, Chapter 4], [REF-GOLUB-VANLOAN-2013, Section 5.1].
        Equation [EQ-GAM-001].

    """
    k = len(design[0])
    c = [math.fsum(row[j] for row in design) for j in range(k)]
    length = math.sqrt(math.fsum(v * v for v in c))
    v = list(c)
    v[0] += math.copysign(length, c[0])
    vv = math.fsum(value * value for value in v)
    return tuple(
        tuple((1.0 if i == j else 0.0) - 2.0 * v[i] * v[j] / vv for j in range(1, k))
        for i in range(k)
    )


def _bases(
    columns: Mapping[str, Vector],
    segments: int | Mapping[str, int],
    degree: int,
    penalty_order: int,
) -> list[_TermBasis]:
    """Build every term's basis, constraint, and constrained penalty.

    References:
        [EQ-GAM-001], [REF-EILERS-MARX-1996], [REF-WOOD-2017, Chapter 4].

    """
    d = require_non_negative_integer(penalty_order, name="penalty_order")
    if d < 1:
        raise ValueError(
            "penalty_order must be at least 1 so constants are unpenalized"
        )
    bases: list[_TermBasis] = []
    for name, xs in columns.items():
        pieces = segments[name] if isinstance(segments, Mapping) else segments
        if not max(xs) > min(xs):
            raise ValueError(f"predictor {name!r} is constant")
        knots = uniform_knots(min(xs), max(xs), pieces, degree)
        design = bspline_design_matrix(xs, knots, degree)
        k = len(design[0])
        z = _sum_to_zero_basis(design)
        s = difference_penalty(k, d)
        sz = tuple(
            tuple(math.fsum(s[i][m] * z[m][j] for m in range(k)) for j in range(k - 1))
            for i in range(k)
        )
        zsz = tuple(
            tuple(math.fsum(z[m][i] * sz[m][j] for m in range(k)) for j in range(k - 1))
            for i in range(k - 1)
        )
        constrained = tuple(
            tuple(math.fsum(row[m] * z[m][j] for m in range(k)) for j in range(k - 1))
            for row in design
        )
        bases.append(
            _TermBasis(
                name=name,
                knots=knots,
                full_design=design,
                design=constrained,
                constraint=z,
                penalty=zsz,
                rank=k - d,
            )
        )
    return bases


def _joint_problem(
    bases: Sequence[_TermBasis], ys: Vector, ws: Vector
) -> PenalizedProblem:
    """Assemble ``[1 | B_1 Z_1 | ...]`` and the zero-padded block penalties.

    References:
        [EQ-GAM-001], [REF-WOOD-2017, Chapter 4].

    """
    width = 1 + sum(len(item.penalty) for item in bases)
    rows = [[1.0] for _ in ys]
    penalties: list[Matrix] = []
    blocks: list[tuple[int, ...]] = []
    offset = 1
    for item in bases:
        size = len(item.penalty)
        for row, part in zip(rows, item.design, strict=True):
            row.extend(part)
        full = [[0.0] * width for _ in range(width)]
        for i in range(size):
            for j in range(size):
                full[offset + i][offset + j] = item.penalty[i][j]
        penalties.append(tuple(tuple(row) for row in full))
        blocks.append(tuple(range(offset, offset + size)))
        offset += size
    return PenalizedProblem(
        design=tuple(tuple(row) for row in rows),
        response=ys,
        weights=ws,
        penalties=tuple(penalties),
        ranks=tuple(item.rank for item in bases),
        blocks=tuple(blocks),
    )


def _assemble(
    bases: Sequence[_TermBasis],
    problem: PenalizedProblem,
    lambdas: Sequence[float],
    method: str,
    criterion: float,
    degree: int,
    penalty_order: int,
    iterations: int,
    coefficients: Sequence[Vector] | None = None,
    intercept: float | None = None,
) -> AdditiveModelFit:
    """Turn a joint solution (or backfitted coefficients) into a fit record.

    References:
        [EQ-GAM-001], [EQ-SMOOTH-003], [EQ-SMOOTH-006], [REF-WOOD-2017, Chapter 6].

    """
    solution = problem.solve(lambdas)
    ys = problem.response
    n = len(ys)
    if coefficients is None:
        beta = solution.coefficients
        alpha = beta[0]
        term_coefficients: list[Vector] = []
        for item, block in zip(bases, problem.blocks, strict=True):
            theta = tuple(beta[i] for i in block)
            term_coefficients.append(matvec(item.constraint, theta))
    else:
        term_coefficients = [tuple(c) for c in coefficients]
        alpha = float(intercept) if intercept is not None else 0.0
    fitted_terms = [
        matvec(item.full_design, c)
        for item, c in zip(bases, term_coefficients, strict=True)
    ]
    fitted = tuple(alpha + math.fsum(t[i] for t in fitted_terms) for i in range(n))
    rss = math.fsum(
        w * (y - m) ** 2 for w, y, m in zip(problem.weights, ys, fitted, strict=True)
    )
    if not n - solution.edf > 0.0:
        raise ValueError("the model interpolates the data; increase smoothing")
    scale = rss / (n - solution.edf)
    terms: list[AdditiveTerm] = []
    for item, block, c, lam, edf in zip(
        bases,
        problem.blocks,
        term_coefficients,
        lambdas,
        solution.term_edf,
        strict=True,
    ):
        block_cov = [[scale * solution.inverse[i][j] for j in block] for i in block]
        z = item.constraint
        zv = [
            [
                math.fsum(z[i][m] * block_cov[m][j] for m in range(len(block)))
                for j in range(len(block))
            ]
            for i in range(len(z))
        ]
        covariance = tuple(
            tuple(
                math.fsum(zv[i][m] * z[j][m] for m in range(len(block)))
                for j in range(len(z))
            )
            for i in range(len(z))
        )
        terms.append(
            AdditiveTerm(
                name=item.name,
                knots=item.knots,
                degree=degree,
                penalty_order=penalty_order,
                coefficients=c,
                covariance=covariance,
                smoothing=lam,
                edf=edf,
            )
        )
    return AdditiveModelFit(
        intercept=alpha,
        intercept_standard_error=math.sqrt(scale * solution.inverse[0][0]),
        terms=tuple(terms),
        method=method,
        criterion=criterion,
        edf=solution.edf,
        scale=scale,
        fitted=fitted,
        residuals=tuple(y - m for y, m in zip(ys, fitted, strict=True)),
        iterations=iterations,
    )


def _smoothing_list(
    names: Sequence[str], smoothing: float | Mapping[str, float | None] | None
) -> list[float | None]:
    """Expand the smoothing argument to one entry per term.

    References:
        [EQ-SMOOTH-002]. No further equation is evaluated.

    """
    if smoothing is None:
        return [None] * len(names)
    if isinstance(smoothing, Mapping):
        values: list[float | None] = []
        for name in names:
            value = smoothing.get(name)
            if value is not None:
                value = require_finite_real(value, name=f"smoothing[{name}]")
                if value < 0.0:
                    raise ValueError("smoothing parameters must be non-negative")
            values.append(value)
        return values
    value = require_finite_real(smoothing, name="smoothing")
    if value < 0.0:
        raise ValueError("smoothing parameters must be non-negative")
    return [value] * len(names)


def fit_additive_model(
    data: Mapping[str, Iterable[float]],
    y: Iterable[float],
    *,
    segments: int | Mapping[str, int] = 10,
    degree: int = 3,
    penalty_order: int = 2,
    smoothing: float | Mapping[str, float | None] | None = None,
    method: str = "reml",
    weights: Iterable[float] | None = None,
) -> AdditiveModelFit:
    """Fit a Gaussian additive model by joint penalized least squares.

    Each term is a P-spline on equally spaced knots over the range of its
    predictor, reparameterized by ``beta_j = Z_j theta_j`` so that
    ``sum_i f_j(x_ij) = 0`` (EQ-GAM-001). The coefficients minimize
    ``||y - alpha - sum_j B_j Z_j theta_j||_W^2 + sum_j lambda_j theta_j'
    Z_j' D'D Z_j theta_j``; free ``lambda_j`` minimize REML (default) or
    GCV by Nelder-Mead in ``log lambda`` (one term: grid and golden section).

    Args:
        data: Predictor columns by name.
        y: Responses.
        segments: Knot intervals per term (one value or one per name).
        degree: B-spline degree.
        penalty_order: Difference order ``d >= 1``.
        smoothing: ``None`` to select every ``lambda_j``, one fixed value
            for all, or a mapping with a fixed value or ``None`` per name.
        method: ``"reml"`` or ``"gcv"``.
        weights: Positive prior weights.

    Returns:
        The additive model fit.

    Raises:
        ValueError: For invalid inputs, a constant predictor, or a design
            whose unpenalized part is not identified.

    Equation:
        [EQ-GAM-001], [EQ-SMOOTH-002], [EQ-SMOOTH-003], [EQ-SMOOTH-004],
        [EQ-SMOOTH-005], [EQ-SMOOTH-006].

    References:
        [REF-WOOD-2017, Chapters 4 and 6], [REF-WOOD-2011],
        [REF-EILERS-MARX-1996].

    """
    columns, ys, ws = _columns(data, y, weights)
    chosen = check_method(method)
    p = require_non_negative_integer(degree, name="degree")
    bases = _bases(columns, segments, p, penalty_order)
    problem = _joint_problem(bases, ys, ws)
    fixed = _smoothing_list(list(columns), smoothing)
    lambdas, value, _ = select_smoothing(problem, chosen, fixed)
    label = chosen if any(v is None for v in fixed) else "fixed"
    if label == "fixed":
        value = problem.criterion("gcv", lambdas, problem.solve(lambdas))
    return _assemble(bases, problem, lambdas, label, value, p, int(penalty_order), 0)


def backfit_additive_model(
    data: Mapping[str, Iterable[float]],
    y: Iterable[float],
    *,
    smoothing: float | Mapping[str, float],
    segments: int | Mapping[str, int] = 10,
    degree: int = 3,
    penalty_order: int = 2,
    weights: Iterable[float] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> AdditiveModelFit:
    """Fit a Gaussian additive model by backfitting with fixed smoothing.

    Starting from ``alpha = mean_w(y)`` and ``f_j = 0``, each sweep updates
    ``f_j <- S_j (y - alpha - sum_(k != j) f_k)`` and centres it, where
    ``S_j = B_j (B_j' W B_j + lambda_j D'D)^-1 B_j' W`` is the P-spline
    smoother of term ``j`` (EQ-GAM-002); after each sweep ``alpha`` is reset
    to the weighted mean of ``y - sum_j f_j``. Sweeps stop when the largest change
    of a fitted term is below ``tolerance`` times the response scale. For
    these symmetric smoothers the limit solves the joint penalized normal
    equations, so the result equals :func:`fit_additive_model` with the same
    ``lambda_j``; the reported EDF and bands are those of that joint system.

    Args:
        data: Predictor columns by name.
        y: Responses.
        smoothing: One ``lambda > 0`` for every term or one per name.
        segments: Knot intervals per term.
        degree: B-spline degree.
        penalty_order: Difference order ``d >= 1``.
        weights: Positive prior weights.
        tolerance: Relative convergence tolerance.
        max_iterations: Maximum number of sweeps.

    Returns:
        The additive model fit with ``method="backfitting"``.

    Raises:
        ValueError: For invalid inputs or if the sweeps do not converge.

    Equation:
        [EQ-GAM-002], [EQ-GAM-001].

    References:
        [REF-BUJA-HASTIE-TIBSHIRANI-1989], [REF-HASTIE-TIBSHIRANI-1990].

    """
    columns, ys, ws = _columns(data, y, weights)
    p = require_non_negative_integer(degree, name="degree")
    d = require_non_negative_integer(penalty_order, name="penalty_order")
    names = list(columns)
    fixed = _smoothing_list(names, smoothing)
    lambdas = [float(v) for v in fixed if v is not None]
    if len(lambdas) != len(names) or any(lam <= 0.0 for lam in lambdas):
        raise ValueError("backfitting needs a positive smoothing parameter per term")
    limit = require_non_negative_integer(max_iterations, name="max_iterations")
    tol = require_finite_real(tolerance, name="tolerance")
    bases = _bases(columns, segments, p, d)
    n = len(ys)
    total_weight = math.fsum(ws)
    alpha = math.fsum(w * v for w, v in zip(ws, ys, strict=True)) / total_weight
    full_designs = [item.full_design for item in bases]
    smoothers: list[PenalizedProblem] = []
    for design in full_designs:
        k = len(design[0])
        smoothers.append(
            PenalizedProblem(
                design=design,
                response=ys,
                weights=ws,
                penalties=(difference_penalty(k, d),),
                ranks=(k - d,),
                blocks=(tuple(range(k)),),
            )
        )
    grams = [problem.gram for problem in smoothers]
    terms = [[0.0] * n for _ in names]
    coefficients: list[Vector] = [(0.0,) * len(design[0]) for design in full_designs]
    spread = max(ys) - min(ys) or 1.0
    iterations = 0
    for sweep in range(1, limit + 1):
        iterations = sweep
        change = 0.0
        for j, smoother in enumerate(smoothers):
            partial = tuple(
                ys[i]
                - alpha
                - math.fsum(terms[k][i] for k in range(len(names)) if k != j)
                for i in range(n)
            )
            local = PenalizedProblem(
                design=smoother.design,
                response=partial,
                weights=ws,
                penalties=smoother.penalties,
                ranks=smoother.ranks,
                blocks=smoother.blocks,
            )
            beta = local.solve((lambdas[j],), grams[j]).coefficients
            values = matvec(full_designs[j], beta)
            centre = math.fsum(values) / n
            beta = tuple(b - centre for b in beta)
            new = [v - centre for v in values]
            change = max(
                change, max(abs(a - b) for a, b in zip(new, terms[j], strict=True))
            )
            terms[j] = new
            coefficients[j] = beta
        alpha = (
            math.fsum(
                w * (ys[i] - math.fsum(t[i] for t in terms)) for i, w in enumerate(ws)
            )
            / total_weight
        )
        if change <= tol * spread:
            break
    else:
        raise ValueError("backfitting did not converge; raise max_iterations")
    problem = _joint_problem(bases, ys, ws)
    criterion = problem.criterion("gcv", lambdas, problem.solve(lambdas))
    return _assemble(
        bases,
        problem,
        lambdas,
        "backfitting",
        criterion,
        p,
        d,
        iterations,
        coefficients=coefficients,
        intercept=alpha,
    )


def partial_dependence(
    predict: Callable[[Mapping[str, Sequence[float]]], Sequence[float]],
    data: Mapping[str, Iterable[float]],
    feature: str,
    grid: Iterable[float],
) -> PartialDependence:
    """Return Friedman's partial dependence of a prediction function.

    ``PD_j(v) = (1/n) sum_i m(x_i with x_ij replaced by v)`` (EQ-GAM-003).
    For an additive model it equals ``alpha + f_j(v)`` exactly, because the
    other terms are centred over the data; for a model with interactions it
    is an average over them and can hide them.

    Args:
        predict: Function from a column mapping to predictions.
        data: Columns at which the average is taken.
        feature: Name of the varied column.
        grid: Values of the varied column.

    Returns:
        The partial-dependence curve.

    Raises:
        KeyError: If ``feature`` is not a column of ``data``.
        ValueError: For empty or unequal columns.

    Equation:
        [EQ-GAM-003].

    References:
        [REF-FRIEDMAN-2001], [REF-HASTIE-TIBSHIRANI-FRIEDMAN-2009, Section 10.13].

    """
    columns = {
        str(name): coerce_finite_sample(values, name=str(name))
        for name, values in data.items()
    }
    if feature not in columns:
        raise KeyError(feature)
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same length")
    size = lengths.pop()
    points = coerce_finite_sample(grid, name="grid")
    values: list[float] = []
    for value in points:
        modified = dict(columns)
        modified[feature] = (value,) * size
        predictions = predict(modified)
        values.append(math.fsum(predictions) / size)
    return PartialDependence(feature=feature, grid=points, values=tuple(values))
