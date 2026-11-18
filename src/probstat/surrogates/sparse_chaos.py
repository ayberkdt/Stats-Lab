"""Sparse polynomial chaos expansions by least-angle regression or OMP.

A full least-squares PCE needs more model runs than basis terms, and the
total-degree set grows as ``C(d + p, p)``. A sparse PCE keeps only the terms
that matter. Following Blatman and Sudret (2011):

1. Least-angle regression (LAR, Efron et al. 2004) on the centred and
   unit-norm basis columns ranks the non-constant terms: each step moves the
   fit along the direction equiangular to every active column until an
   inactive column is as correlated with the residual as the active ones.
   The constant term is always kept and is not ranked; centring the
   columns and the response removes it from the LAR problem.
2. The hybrid step refits ordinary least squares on each nested set
   ``{constant} + first k ranked terms`` and keeps the ``k`` with the
   smallest corrected leave-one-out (LOO) error
   ``T(P, N) epsilon_LOO``, ``T = N/(N-P) (1 + tr((Psi'Psi)^-1))``.
3. Orthogonal matching pursuit (OMP) is offered as a greedy alternative:
   each step adds the column most correlated with the current least-squares
   residual.
4. The whole selection runs over a range of degrees and hyperbolic norms and
   keeps the truncation set with the smallest corrected LOO error.

The nested refits share one Gram-Schmidt factorization that grows by a
column per step, so a whole path costs about one least-squares fit of the
largest set.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector, householder_qr
from probstat.modeling import JointInput, ProbabilisticModel
from probstat.propagation import make_design

from .polynomial_chaos import (
    MultiIndex,
    PolynomialChaosExpansion,
    SparseChaosCandidate,
    SparseChaosSelection,
    _check_basis,
    _corrected_loo_factor,
    _dot,
    _least_squares_errors,
    _standardize,
    hermite_orthonormal,
    legendre_orthonormal,
    multi_indices,
)

SPARSE_CHAOS_METHODS = ("lar", "omp")

# Computational choices (not from the literature): a corrected relative LOO
# error below ``_EXACT_FLOOR`` counts as an exact fit, so among exact nested
# sets the sparsest one wins instead of the one with the smallest rounding
# noise; ``_CHOLESKY_TOLERANCE`` is the smallest squared sine between a new
# standardized column and the span of the active ones; a column whose
# centred norm is below ``_CONSTANT_TOLERANCE`` times its raw norm is treated
# as constant on the design; ``_GRAM_SCHMIDT_TOLERANCE`` is the relative
# length below which an orthogonalized column counts as dependent.
_EXACT_FLOOR = 1e-20
_CHOLESKY_TOLERANCE = 1e-10
_CONSTANT_TOLERANCE = 1e-6
_GRAM_SCHMIDT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class LeastAngleRegressionPath:
    """Knots of a least-angle regression path on standardized columns.

    Knot ``k < K`` (``K = len(order)``) is where column ``order[k]`` joins
    the active set; knot ``K`` is the end of the path, the least-squares fit
    on the entered columns (the last segment is followed to its end when no
    further column may enter, because all usable columns entered or the step
    limit was reached).

    Args:
        order: Column indices in the order they entered.
        max_correlations: ``C_k = max_j |x_j' (y - mu_k)|`` at knot
            ``k = 0, ..., K``, non-increasing.
        coefficients: Coefficients ``beta_k`` on the standardized columns at
            each knot (``K + 1`` vectors of one entry per column; ``beta_0 =
            0``).
        excluded: Columns that never entered because they are constant on
            the design or numerically in the span of the active columns.
        means: Column means removed by the standardization.
        scales: Centred column norms divided out by the standardization.

    References:
        [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004].

    """

    order: tuple[int, ...]
    max_correlations: Vector
    coefficients: tuple[Vector, ...]
    excluded: tuple[int, ...]
    means: Vector
    scales: Vector


def _standardize_columns(
    columns: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], list[float], list[bool]]:
    """Centre every column and scale it to unit Euclidean norm.

    Args:
        columns: Raw columns of equal length ``N``.

    Returns:
        ``(standardized, means, scales, usable)``; a column that is constant
        on the design is left at zero and marked unusable.

    References:
        [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004, Section 2].

    """
    standardized: list[list[float]] = []
    means: list[float] = []
    scales: list[float] = []
    usable: list[bool] = []
    for column in columns:
        count = len(column)
        mean = math.fsum(column) / count
        centred = [value - mean for value in column]
        scale = math.sqrt(_dot(centred, centred))
        raw = math.sqrt(_dot(column, column))
        keep = scale > _CONSTANT_TOLERANCE * raw and scale > 0.0
        standardized.append([value / scale for value in centred] if keep else [])
        means.append(mean)
        scales.append(scale if keep else 0.0)
        usable.append(keep)
    return standardized, means, scales, usable


class _LeastAngle:
    """Incremental least-angle regression on standardized columns.

    The active-set Gram matrix ``G = X_A' X_A`` is kept as a Cholesky factor
    that grows by one row per entry. With active signs ``s`` the equiangular
    direction is ``u = A X_A G^-1 s`` with ``A = (s' G^-1 s)^(-1/2)``, and the
    step is ``gamma = min+ {(C - c_j)/(A - a_j), (C + c_j)/(A + a_j)}`` over
    inactive ``j``, ``a = X' u``.

    References:
        [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004, Section 2],
        [EQ-SUR-SPCE-001].

    """

    def __init__(
        self,
        columns: list[list[float]],
        response: Sequence[float],
        usable: Sequence[bool],
        max_active: int,
    ) -> None:
        """Start the path at ``mu = 0`` with correlations ``c = X' y``.

        Args:
            columns: Centred unit-norm columns (empty for unusable ones).
            response: Centred response.
            usable: Whether each column may enter.
            max_active: Largest number of active columns.

        References:
            [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004].

        """
        self.columns = columns
        self.count = len(response)
        self.correlations = [
            _dot(column, response) if keep else 0.0
            for column, keep in zip(columns, usable, strict=True)
        ]
        self.available = [bool(keep) for keep in usable]
        self.excluded = [not keep for keep in usable]
        self.active: list[int] = []
        self.active_rows: list[list[float]] = [[] for _ in range(self.count)]
        self.cholesky: list[list[float]] = []
        self.cholesky_columns: list[list[float]] = []
        self.beta = [0.0] * len(columns)
        self.max_active = max_active
        self.knot_correlations: list[float] = []
        self.knot_coefficients: list[Vector] = []
        initial = max(
            (
                abs(c)
                for c, ok in zip(self.correlations, self.available, strict=True)
                if ok
            ),
            default=0.0,
        )
        self.stop_level = 1e-12 * initial
        self.pending = self._argmax() if initial > 0.0 and max_active > 0 else None

    def _argmax(self) -> int | None:
        """Return the available column of largest absolute correlation.

        References:
            [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004].

        """
        best: int | None = None
        level = -1.0
        for j, ok in enumerate(self.available):
            if ok and abs(self.correlations[j]) > level:
                best, level = j, abs(self.correlations[j])
        return best

    def _try_add(self, j: int) -> bool:
        """Append column ``j`` to the Cholesky factor of the active Gram matrix.

        Returns:
            ``False`` when the column is numerically in the active span.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        column = self.columns[j]
        gram = [_dot(column, self.columns[a]) for a in self.active]
        solved: list[float] = []
        for i, row in enumerate(self.cholesky):
            solved.append((gram[i] - _dot(row[:i], solved)) / row[i])
        diagonal = 1.0 - _dot(solved, solved)
        if diagonal <= _CHOLESKY_TOLERANCE:
            return False
        for i, value in enumerate(solved):
            self.cholesky_columns[i].append(value)
        self.cholesky.append([*solved, math.sqrt(diagonal)])
        self.cholesky_columns.append([])
        for row, value in zip(self.active_rows, column, strict=True):
            row.append(value)
        self.active.append(j)
        self.available[j] = False
        return True

    def _gram_solve(self, right: Sequence[float]) -> list[float]:
        """Return ``G^-1 b`` from the Cholesky factor ``G = L L'``.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        size = len(self.cholesky)
        forward: list[float] = []
        for i in range(size):
            row = self.cholesky[i]
            forward.append((right[i] - _dot(row[:i], forward)) / row[i])
        solution = [0.0] * size
        for i in range(size - 1, -1, -1):
            below = _dot(self.cholesky_columns[i], solution[i + 1 :])
            solution[i] = (forward[i] - below) / self.cholesky[i][i]
        return solution

    def _advance(self) -> int | None:
        """Move along the equiangular direction to the next knot.

        Returns:
            The column that ties with the active set at the new knot, or
            ``None`` when the path ended (least-squares fit reached or no
            column may enter).

        Equation:
            [EQ-SUR-SPCE-001].

        References:
            [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004, Section 2].

        """
        level = max(abs(self.correlations[a]) for a in self.active)
        if level <= self.stop_level:
            return None
        signs = [1.0 if self.correlations[a] >= 0.0 else -1.0 for a in self.active]
        direction = self._gram_solve(signs)
        scale = 1.0 / math.sqrt(_dot(signs, direction))
        weights = [scale * value for value in direction]
        equiangular = [_dot(row, weights) for row in self.active_rows]
        inner = [
            _dot(column, equiangular) if column else 0.0 for column in self.columns
        ]
        gamma = level / scale
        entering: int | None = None
        if len(self.active) < self.max_active:
            threshold = 1e-12 * gamma
            for j, ok in enumerate(self.available):
                if not ok:
                    continue
                c = self.correlations[j]
                for numerator, denominator in (
                    (level - c, scale - inner[j]),
                    (level + c, scale + inner[j]),
                ):
                    if denominator > 0.0:
                        candidate = numerator / denominator
                        if threshold < candidate < gamma:
                            gamma, entering = candidate, j
        for weight, a in zip(weights, self.active, strict=True):
            self.beta[a] += gamma * weight
        for j, value in enumerate(inner):
            self.correlations[j] -= gamma * value
        return entering

    def step(self) -> int | None:
        """Activate the pending column and advance to the next knot.

        Returns:
            The column that entered, or ``None`` when the path has ended.

        Equation:
            [EQ-SUR-SPCE-001].

        References:
            [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004].

        """
        while self.pending is not None:
            j = self.pending
            correlation = abs(self.correlations[j])
            if not self._try_add(j):
                self.available[j] = False
                self.excluded[j] = True
                self.pending = self._advance() if self.active else self._argmax()
                continue
            self.knot_correlations.append(correlation)
            self.knot_coefficients.append(tuple(self.beta))
            self.pending = self._advance()
            return j
        return None


def least_angle_regression(
    matrix: Sequence[Sequence[float]],
    response: Sequence[float],
    *,
    max_steps: int | None = None,
) -> LeastAngleRegressionPath:
    """Return the least-angle regression path of a response on given columns.

    The columns of ``matrix`` and the response are centred and the columns
    scaled to unit norm, which absorbs an intercept; a column constant on
    the rows never enters. Starting from ``mu = 0``, the fit moves along
    ``u = A X_A G^-1 s`` (equiangular to all active columns) by
    ``gamma = min+ {(C - c_j)/(A - a_j), (C + c_j)/(A + a_j)}``, when the
    next column ties with the active correlations ``C``. Without lasso
    modification no column ever leaves the active set.

    Args:
        matrix: ``N`` rows of ``m`` regressors.
        response: ``N`` responses.
        max_steps: Largest number of entered columns; defaults to
            ``min(m, N - 1)``.

    Returns:
        Entry order, correlation levels, and coefficients at every knot.

    Raises:
        ValueError: If the shapes disagree, a value is not finite, fewer
            than two rows are given, or ``max_steps`` is negative.

    Equation:
        [EQ-SUR-SPCE-001].

    References:
        [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004].

    """
    rows = tuple(as_vector(row, name="row") for row in matrix)
    values = as_vector(response, name="response")
    count = len(rows)
    if count != len(values):
        raise ValueError("matrix and response must have equal length")
    if count < 2:
        raise ValueError("least-angle regression needs at least two rows")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("matrix rows must share one positive length")
    limit = min(width, count - 1)
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise ValueError("max_steps must be an integer")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        limit = min(limit, max_steps)
    columns = [list(column) for column in zip(*rows, strict=True)]
    standardized, means, scales, usable = _standardize_columns(columns)
    center = math.fsum(values) / count
    centred = [value - center for value in values]
    lars = _LeastAngle(standardized, centred, usable, limit)
    order: list[int] = []
    while (entered := lars.step()) is not None:
        order.append(entered)
    level = max((abs(lars.correlations[a]) for a in lars.active), default=0.0)
    return LeastAngleRegressionPath(
        tuple(order),
        (*lars.knot_correlations, level),
        (*lars.knot_coefficients, tuple(lars.beta)),
        tuple(j for j, flag in enumerate(lars.excluded) if flag),
        tuple(means),
        tuple(scales),
    )


class _NestedLeastSquares:
    """Least squares on a growing column set by Gram-Schmidt with reorthogonalization.

    ``Psi_k = Q_k R_k`` grows by one column per call. The fitted values, the
    hat-matrix diagonal ``h_i = sum_j q_ij^2``, and ``tr((Psi'Psi)^-1) =
    ||R^-1||_F^2`` are all updated in ``O(N k)``, so the plain and corrected
    LOO errors of every nested set come at the cost of one fit.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 5.2], [EQ-SUR-SPCE-003].

    """

    def __init__(self, response: Vector, variance: float) -> None:
        """Start from the constant column ``Psi_0 = 1``.

        Args:
            response: Responses ``y``.
            variance: Sample variance of ``y`` (divisor ``N - 1``).

        References:
            [EQ-SUR-PCE-006].

        """
        count = len(response)
        self.count = count
        self.variance = variance
        self.response = response
        base = 1.0 / math.sqrt(count)
        self.q: list[list[float]] = [[base] * count]
        self.q_rows: list[list[float]] = [[base] for _ in range(count)]
        self.inverse_columns: list[list[float]] = [[base]]
        self.trace = 1.0 / count
        center = math.fsum(response) / count
        self.residual = [y - center for y in response]
        self.leverage = [1.0 / count] * count

    def add(self, column: Sequence[float]) -> bool:
        """Append one column; return ``False`` if it is numerically dependent.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 5.2.8].

        """
        vector = list(column)
        norm = math.sqrt(_dot(vector, vector))
        projections = [0.0] * len(self.q)
        for _ in range(2):
            inner = [_dot(basis, vector) for basis in self.q]
            vector = [
                value - _dot(row, inner)
                for value, row in zip(vector, self.q_rows, strict=True)
            ]
            projections = [a + b for a, b in zip(projections, inner, strict=True)]
        length = math.sqrt(_dot(vector, vector))
        if norm == 0.0 or length <= _GRAM_SCHMIDT_TOLERANCE * norm:
            return False
        unit = [value / length for value in vector]
        size = len(self.q)
        top = [0.0] * size
        for k, inverse_column in enumerate(self.inverse_columns):
            weight = projections[k]
            for i, value in enumerate(inverse_column):
                top[i] += value * weight
        new_column = [-value / length for value in top] + [1.0 / length]
        self.trace += _dot(new_column, new_column)
        self.inverse_columns.append(new_column)
        self.q.append(unit)
        weight = _dot(unit, self.response)
        self.residual = [
            r - weight * u for r, u in zip(self.residual, unit, strict=True)
        ]
        self.leverage = [h + u * u for h, u in zip(self.leverage, unit, strict=True)]
        for row, value in zip(self.q_rows, unit, strict=True):
            row.append(value)
        return True

    @property
    def terms(self) -> int:
        """Return the number of columns ``P`` in the current set.

        References:
            [REF-BLATMAN-SUDRET-2011].

        """
        return len(self.q)

    def corrected_loo(self) -> float:
        """Return the relative corrected LOO error of the current set.

        Equation:
            [EQ-SUR-PCE-006], [EQ-SUR-SPCE-003].

        References:
            [REF-BLATMAN-SUDRET-2011], [REF-CHAPELLE-VAPNIK-BENGIO-2002].

        """
        squared: list[float] = []
        for residual, leverage in zip(self.residual, self.leverage, strict=True):
            if leverage >= 1.0 - 1e-12:
                return math.inf
            squared.append((residual / (1.0 - leverage)) ** 2)
        loo = math.fsum(squared) / self.count / self.variance
        return loo * _corrected_loo_factor(self.count, self.terms, self.trace)


def _select_path(
    indices: tuple[MultiIndex, ...],
    columns: Sequence[Sequence[float]],
    responses: Vector,
    variance: float,
    method: str,
    early_stop: bool,
) -> tuple[tuple[int, ...], tuple[float, ...], int]:
    """Rank the non-constant terms of one truncation set and score each prefix.

    Args:
        indices: Truncation set, constant first.
        columns: Basis columns ``Psi_alpha(xi_i)``, aligned with ``indices``.
        responses: Responses ``y``.
        variance: Sample variance of ``y``.
        method: ``"lar"`` or ``"omp"``.
        early_stop: Stop once the corrected LOO error has not improved for
            ``max(10, ceil(0.1 k_max))`` steps.

    Returns:
        ``(entered positions, corrected LOO per step, best step)``.

    Equation:
        [EQ-SUR-SPCE-001], [EQ-SUR-SPCE-002], [EQ-SUR-SPCE-003],
        [EQ-SUR-SPCE-004].

    References:
        [REF-BLATMAN-SUDRET-2011], [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004].

    """
    count = len(responses)
    raw = [columns[j] for j in range(1, len(indices))]
    standardized, _, scales, usable = _standardize_columns(raw)
    max_terms = min(sum(usable), count - 2)
    nested = _NestedLeastSquares(responses, variance)
    path = [nested.corrected_loo()]
    order: list[int] = []
    best_step = 0
    best_level = max(path[0], _EXACT_FLOOR)
    patience = max(10, math.ceil(0.1 * max_terms))
    since_best = 0
    lars: _LeastAngle | None = None
    if method == "lar":
        center = math.fsum(responses) / count
        lars = _LeastAngle(
            standardized, [y - center for y in responses], usable, max_terms
        )
    candidates = list(usable)
    while len(order) < max_terms:
        if lars is not None:
            entered = lars.step()
            if entered is None:
                break
        else:
            best_score = 0.0
            entered = None
            for j, ok in enumerate(candidates):
                if ok:
                    score = abs(_dot(raw[j], nested.residual)) / scales[j]
                    if score > best_score:
                        best_score, entered = score, j
            if entered is None or best_score <= 1e-14 * math.sqrt(variance * count):
                break
            candidates[entered] = False
        if not nested.add(raw[entered]):
            if lars is not None:
                break
            continue
        order.append(entered + 1)
        error = nested.corrected_loo()
        path.append(error)
        level = max(error, _EXACT_FLOOR)
        if level < best_level:
            best_level, best_step, since_best = level, len(order), 0
        else:
            since_best += 1
            if early_stop and since_best >= patience:
                break
    return tuple(order), tuple(path), best_step


def _univariate_tables(
    standardized: Sequence[Vector], basis: str, degree: int
) -> list[list[Vector]]:
    """Return ``psi_0..psi_p`` of every coordinate of every design point.

    References:
        [EQ-SUR-PCE-002], [EQ-SUR-PCE-003].

    """
    family = hermite_orthonormal if basis == "hermite" else legendre_orthonormal
    return [[family(value, degree) for value in point] for point in standardized]


def _basis_column(tables: Sequence[Sequence[Vector]], alpha: MultiIndex) -> list[float]:
    """Return ``Psi_alpha`` at every design point from univariate tables.

    References:
        [EQ-SUR-PCE-001].

    """
    active = [(i, a) for i, a in enumerate(alpha) if a]
    column: list[float] = []
    for table in tables:
        product = 1.0
        for i, a in active:
            product *= table[i][a]
        column.append(product)
    return column


def _validated_degrees(degrees: Iterable[int]) -> tuple[int, ...]:
    """Return the distinct requested degrees in increasing order.

    Raises:
        ValueError: If no degree is given or one is not a non-negative integer.

    References:
        [EQ-SUR-PCE-004].

    """
    values = tuple(degrees)
    if not values:
        raise ValueError("degrees must contain at least one degree")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("degrees must be non-negative integers")
    return tuple(sorted(set(values)))


def _validated_norms(q_norms: Sequence[float]) -> tuple[float, ...]:
    """Return the requested hyperbolic norms after range checks.

    Raises:
        ValueError: If no norm is given or one lies outside ``(0, 1]``.

    References:
        [EQ-SUR-PCE-004].

    """
    values = tuple(require_finite_real(q, name="q_norm") for q in q_norms)
    if not values:
        raise ValueError("q_norms must contain at least one norm")
    if any(not 0.0 < q <= 1.0 for q in values):
        raise ValueError("q_norm must lie in (0, 1]")
    return values


def sparse_polynomial_chaos_from_samples(
    inputs: JointInput,
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    degrees: Iterable[int],
    q_norms: Sequence[float] = (1.0,),
    basis: str = "hermite",
    method: str = "lar",
    quantity: str = "y",
    early_stop: bool = True,
) -> PolynomialChaosExpansion:
    """Fit a sparse PCE by hybrid LAR or OMP with degree and q-norm adaptivity.

    For each degree ``p`` (increasing) and each ``q`` the truncation set
    ``A_(p,q)`` is built, its non-constant terms are ranked by LAR or OMP,
    and every nested set ``{constant} + first k terms`` is refitted by least
    squares and scored by its corrected LOO error
    ``T(P, N) epsilon_LOO``. The set with the smallest error over all
    ``(p, q, k)`` is refitted by Householder QR and returned. Unlike the
    full least-squares fit, the truncation set may be larger than the design
    (``N < P``); at most ``N - 2`` terms are entered.

    Early stopping (``early_stop=True``): a path stops once the corrected
    LOO error has not improved for ``max(10, ceil(0.1 k_max))`` steps, and
    the degree loop stops once two consecutive degrees fail to improve the
    best error over all ``q``. Errors below ``1e-20`` count as an
    exact fit, so among exact sets the sparsest is kept. These are
    computational choices.

    Args:
        inputs: Joint input law of the samples.
        points: Physical input vectors.
        values: Model responses, one per point.
        degrees: Candidate maximal total degrees.
        q_norms: Candidate hyperbolic norms in ``(0, 1]``.
        basis: ``"hermite"`` or ``"legendre"``.
        method: ``"lar"`` (hybrid least-angle regression) or ``"omp"``
            (orthogonal matching pursuit, which tends to overfit more).
        quantity: Label of the approximated quantity.
        early_stop: Whether to cut the paths and the degree loop early;
            ``False`` ranks every term up to ``N - 2`` on every set.

    Returns:
        Sparse expansion with ``corrected_loo_error``, ``candidate_size``,
        and ``selection`` (the entry order and corrected LOO path of every
        visited truncation set).

    Raises:
        ValueError: If the basis or method is unknown or inapplicable, a
            degree or norm is out of range, there are fewer than three
            samples, shapes disagree, or the responses are constant.

    Equation:
        [EQ-SUR-SPCE-001], [EQ-SUR-SPCE-002], [EQ-SUR-SPCE-003],
        [EQ-SUR-SPCE-004], [EQ-SUR-PCE-004], [EQ-SUR-PCE-005].

    References:
        [REF-BLATMAN-SUDRET-2011],
        [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004],
        [REF-CHAPELLE-VAPNIK-BENGIO-2002].

    """
    _check_basis(inputs, basis)
    if method not in SPARSE_CHAOS_METHODS:
        raise ValueError(f"method must be one of {SPARSE_CHAOS_METHODS}")
    degree_list = _validated_degrees(degrees)
    norms = _validated_norms(q_norms)
    sets = {
        (p, q): multi_indices(inputs.dimension, p, q_norm=q)
        for p in degree_list
        for q in norms
    }
    design = tuple(as_vector(point, name="point") for point in points)
    responses = as_vector(values, name="values")
    if len(design) != len(responses):
        raise ValueError("points and values must have equal length")
    count = len(design)
    if count < 3:
        raise ValueError("a sparse PCE needs at least three samples")
    center = math.fsum(responses) / count
    variance = math.fsum((y - center) ** 2 for y in responses) / (count - 1)
    if variance <= 0.0:
        raise ValueError("responses are constant; nothing to approximate")
    standardized = [_standardize(inputs, basis, point) for point in design]
    tables = _univariate_tables(standardized, basis, degree_list[-1])
    cache: dict[MultiIndex, list[float]] = {}

    def column(alpha: MultiIndex) -> list[float]:
        """Return the cached basis column of one multi-index.

        References:
            [EQ-SUR-PCE-001].

        """
        if alpha not in cache:
            cache[alpha] = _basis_column(tables, alpha)
        return cache[alpha]

    candidates: list[SparseChaosCandidate] = []
    seen: set[tuple[MultiIndex, ...]] = set()
    best = -1
    best_level = math.inf
    strikes = 0
    for p in degree_list:
        visited = improved = False
        for q in norms:
            indices = sets[(p, q)]
            if indices in seen:
                continue
            seen.add(indices)
            visited = True
            order, path, step = _select_path(
                indices,
                [column(alpha) for alpha in indices],
                responses,
                variance,
                method,
                early_stop,
            )
            candidates.append(
                SparseChaosCandidate(
                    p,
                    q,
                    len(indices),
                    tuple(indices[j] for j in order),
                    path,
                    step,
                    path[step],
                )
            )
            level = max(path[step], _EXACT_FLOOR)
            if level < best_level:
                best_level, best, improved = level, len(candidates) - 1, True
        if not visited:
            continue
        strikes = 0 if improved else strikes + 1
        if early_stop and strikes >= 2:
            break
    chosen = candidates[best]
    full_set = sets[(chosen.degree, chosen.q_norm)]
    kept = set(chosen.entry_order[: chosen.selected_terms])
    selected = full_set[:1] + tuple(alpha for alpha in full_set[1:] if alpha in kept)
    rows = [
        list(row) for row in zip(*(column(alpha) for alpha in selected), strict=True)
    ]
    factor = householder_qr(tuple(tuple(row) for row in rows))
    if not factor.full_rank:
        raise ValueError(
            "the selected basis is numerically rank deficient on the design; "
            "increase the design size"
        )
    coefficients, loo, empirical, corrected = _least_squares_errors(
        rows, responses, factor
    )
    return PolynomialChaosExpansion(
        inputs,
        basis,
        selected,
        coefficients,
        quantity,
        design,
        responses,
        loo,
        empirical,
        chosen.degree,
        chosen.q_norm,
        corrected_loo_error=corrected,
        candidate_size=chosen.candidate_size,
        selection=SparseChaosSelection(method, tuple(candidates), best),
    )


def fit_sparse_polynomial_chaos(
    model: ProbabilisticModel,
    quantity: str | None = None,
    *,
    size: int,
    degrees: Iterable[int],
    q_norms: Sequence[float] = (1.0,),
    basis: str = "hermite",
    method: str = "lar",
    design: str = "lhs",
    seed: int | RandomStream | None = None,
    early_stop: bool = True,
) -> PolynomialChaosExpansion:
    """Build a sparse PCE surrogate of one QoI of a probabilistic model.

    The ``size`` design points are drawn on the unit hypercube
    (``make_design``), mapped through the model's joint input law, and
    evaluated once; :func:`sparse_polynomial_chaos_from_samples` then selects
    the basis.

    Args:
        model: Probabilistic model.
        quantity: QoI name; defaults to the first declared quantity.
        size: Number of model evaluations (at least three).
        degrees: Candidate maximal total degrees.
        q_norms: Candidate hyperbolic norms in ``(0, 1]``.
        basis: ``"hermite"`` or ``"legendre"``.
        method: ``"lar"`` or ``"omp"``.
        design: ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        seed: Integer seed or stream (child ``"sparse-pce-design"``).
        early_stop: Whether to cut the paths and the degree loop early.

    Returns:
        Sparse expansion; ``model.evaluations`` grows by ``size``.

    Raises:
        ValueError: If the basis or method is inapplicable, ``size`` is
            below three, or a degree or norm is out of range.

    Equation:
        [EQ-SUR-SPCE-004], [EQ-SAMP-INV-001].

    References:
        [REF-BLATMAN-SUDRET-2011],
        [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004].

    """
    name = model.quantity_names[0] if quantity is None else quantity
    model.quantity(name)
    _check_basis(model.inputs, basis)
    if method not in SPARSE_CHAOS_METHODS:
        raise ValueError(f"method must be one of {SPARSE_CHAOS_METHODS}")
    if isinstance(size, bool) or not isinstance(size, int) or size < 3:
        raise ValueError("size must be an integer of at least 3")
    degree_list = _validated_degrees(degrees)
    norms = _validated_norms(q_norms)
    stream = resolve_stream(seed).child("sparse-pce-design")
    unit = make_design(design, size, model.dimension, seed=stream)
    points = model.inputs.sample_design(unit)
    values = model.evaluate_many(points)[name]
    return sparse_polynomial_chaos_from_samples(
        model.inputs,
        points,
        values,
        degrees=degree_list,
        q_norms=norms,
        basis=basis,
        method=method,
        quantity=name,
        early_stop=early_stop,
    )
