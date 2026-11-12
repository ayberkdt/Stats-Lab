"""Linear quantile regression by an exact simplex method on elemental bases.

The ``tau``-th regression quantile minimizes the check loss

``R(beta) = sum_i rho_tau(y_i - x_i' beta)``,
``rho_tau(r) = r (tau - 1{r < 0})``,

a linear program whose solutions include a vertex that interpolates ``p``
observations exactly (an elemental basis ``h`` with
``beta = X_h^-1 y_h``). The solver walks from vertex to vertex in the manner
of Barrodale and Roberts: at the current basis it evaluates the ``2p``
directional derivatives of ``R`` along the edges that release one basic
observation in either direction, follows the steepest descending edge, and
moves along it by an exact weighted-median line search to the breakpoint
where the derivative changes sign, which may pass several vertices at once.
Every step strictly decreases ``R`` (the step length is positive), so the
walk terminates. At a degenerate vertex (more than ``p`` zero residuals) the
edges of every elemental basis of the zero set are examined before
optimality is declared, which certifies the subgradient condition
``0 in dR(beta)``.

Solutions need not be unique: when ``n tau`` is an integer the set of
minimizers can be an interval (the sample median of an even sample), and one
optimal vertex is returned. Intervals come from the pairs (``xy``) bootstrap
with percentile limits.
"""

from __future__ import annotations

import itertools
import math
import statistics
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real, require_probability
from probstat.linalg import Matrix, Vector, as_vector, householder_qr, lu_solve
from probstat.statistics import quantile as sample_quantile

from ._hypothesis import check_level
from .confidence import ConfidenceInterval
from .linear_models import DesignMatrix, RankDeficientDesignError, _coerce_design

IntervalMethod = Literal["none", "bootstrap"]
_DEGENERATE_BASIS_LIMIT = 20000
_DERIVATIVE_TOLERANCE = 1e-12


def check_loss(residual: float, quantile: float) -> float:
    """Return the quantile check loss ``rho_tau(r) = r (tau - 1{r < 0})``.

    Args:
        residual: Residual ``r``.
        quantile: Level ``tau`` strictly inside ``(0, 1)``.

    Returns:
        ``tau r`` for ``r >= 0`` and ``(tau - 1) r`` for ``r < 0``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``quantile`` is outside ``(0, 1)``.

    Equation:
        [EQ-INF-QR-001].

    References:
        [REF-KOENKER-BASSETT-1978], [REF-KOENKER-2005].

    """
    r = require_finite_real(residual, name="residual")
    tau = check_level(quantile, name="quantile")
    return r * (tau - (1.0 if r < 0.0 else 0.0))


@dataclass(frozen=True, slots=True)
class QuantileRegressionResult:
    """A linear regression quantile and its optional bootstrap intervals.

    Args:
        coefficients: Optimal ``beta`` (an elemental vertex).
        column_names: Names of the design columns.
        quantile: Level ``tau``.
        objective: ``sum_i rho_tau(r_i)`` at the solution.
        residuals: ``y_i - x_i' beta``, exactly zero on the basis.
        fitted_values: ``y_i - r_i``.
        basis: Indices of the ``p`` observations interpolated exactly.
        iterations: Simplex steps taken.
        optimality_certified: Whether the subgradient condition was verified
            (always, unless a degenerate vertex had too many elemental bases
            to enumerate).
        standard_errors: Bootstrap standard deviations of the coefficients,
            or ``None`` without bootstrap.
        confidence_intervals: Percentile bootstrap intervals, or ``None``.
        replicates: Bootstrap replicates used (zero without bootstrap).
        seed: Seed of the bootstrap stream, or ``None``.

    References:
        [EQ-INF-QR-001], [EQ-INF-QR-002], [REF-KOENKER-2005].

    """

    coefficients: tuple[float, ...]
    column_names: tuple[str, ...]
    quantile: float
    objective: float
    residuals: tuple[float, ...]
    fitted_values: tuple[float, ...]
    basis: tuple[int, ...]
    iterations: int
    optimality_certified: bool
    standard_errors: tuple[float, ...] | None
    confidence_intervals: tuple[ConfidenceInterval, ...] | None
    replicates: int
    seed: int | None

    def coefficient(self, name: str) -> float:
        """Return one coefficient by column name.

        Args:
            name: Column name.

        Returns:
            The coefficient.

        Raises:
            KeyError: If the name is not a column.

        References:
            [EQ-INF-QR-001]. No equation is evaluated.

        """
        if name not in self.column_names:
            raise KeyError(name)
        return self.coefficients[self.column_names.index(name)]


def _inverse(matrix: Matrix) -> Matrix:
    """Invert a small square matrix column by column with pivoted LU.

    Raises:
        ValueError: If the matrix is numerically singular.

    References:
        [EQ-LINALG-004], [REF-GOLUB-VANLOAN-2013, Section 3.4].

    """
    size = len(matrix)
    columns = [
        lu_solve(matrix, tuple(1.0 if i == j else 0.0 for i in range(size)))
        for j in range(size)
    ]
    return tuple(tuple(columns[j][i] for j in range(size)) for i in range(size))


def _initial_basis(rows: Matrix, response: Vector, names: tuple[str, ...]) -> list[int]:
    """Choose ``p`` independent rows with the smallest least-squares residuals.

    Raises:
        RankDeficientDesignError: If the design is rank deficient.

    References:
        [EQ-LINALG-009], [REF-KOENKER-2005].

    """
    factor = householder_qr(rows)
    if factor.rank_deficient:
        culprits = ", ".join(names[j] for j in factor.rank_deficient)
        raise RankDeficientDesignError(
            f"design matrix is rank deficient; column(s) {culprits}"
        )
    beta = factor.solve(response)
    residuals = [
        abs(y - math.fsum(b * x for b, x in zip(beta, row, strict=True)))
        for row, y in zip(rows, response, strict=True)
    ]
    p = len(rows[0])
    chosen: list[int] = []
    orthonormal: list[list[float]] = []
    for index in sorted(range(len(rows)), key=lambda i: (residuals[i], i)):
        vector = list(rows[index])
        norm0 = math.sqrt(math.fsum(v * v for v in vector))
        if norm0 == 0.0:
            continue
        for q in orthonormal:
            projection = math.fsum(a * b for a, b in zip(q, vector, strict=True))
            vector = [v - projection * a for v, a in zip(vector, q, strict=True)]
        norm = math.sqrt(math.fsum(v * v for v in vector))
        if norm > 1e-9 * norm0:
            orthonormal.append([v / norm for v in vector])
            chosen.append(index)
            if len(chosen) == p:
                return chosen
    raise RankDeficientDesignError("design matrix is rank deficient")


def _edge_derivatives(
    rows: Matrix,
    residuals: Sequence[float],
    basis: Sequence[int],
    inverse: Matrix,
    zero: Sequence[bool],
    tau: float,
) -> tuple[list[float], list[float], list[float], list[list[float]]]:
    """Return the ``2p`` edge derivatives at a basis and the edge rates.

    For edge ``(j, s)`` the residual of basic observation ``h_j`` becomes
    ``s t`` while the other basic residuals stay zero; every other residual
    moves at rate ``s c_ij`` with ``c_i = x_i' X_h^-1``.

    Returns:
        Derivatives for ``s = +1`` and ``s = -1``, the scales
        ``1 + sum_i |c_ij|`` used for the tolerance, and the rows ``c_i``.

    References:
        [EQ-INF-QR-001], [REF-BARRODALE-ROBERTS-1973],
        [REF-KOENKER-2005].

    """
    p = len(basis)
    in_basis = set(basis)
    plus = [tau] * p
    minus = [1.0 - tau] * p
    scales = [1.0] * p
    rates: list[list[float]] = []
    for i, row in enumerate(rows):
        c = [math.fsum(row[k] * inverse[k][j] for k in range(p)) for j in range(p)]
        rates.append(c)
        if i in in_basis:
            continue
        r = residuals[i]
        for j in range(p):
            value = c[j]
            scales[j] += abs(value)
            if zero[i]:
                plus[j] += max(tau * value, (tau - 1.0) * value)
                minus[j] += max(-tau * value, (1.0 - tau) * value)
            else:
                score = tau if r > 0.0 else tau - 1.0
                plus[j] += score * value
                minus[j] -= score * value
    return plus, minus, scales, rates


def _descent_edge(
    plus: Sequence[float], minus: Sequence[float], scales: Sequence[float]
) -> tuple[int, int, float] | None:
    """Return the steepest descending edge ``(j, sign, derivative)`` or ``None``.

    References:
        [EQ-INF-QR-001], [REF-BARRODALE-ROBERTS-1973].

    """
    best: tuple[int, int, float] | None = None
    best_value = -_DERIVATIVE_TOLERANCE
    for j in range(len(plus)):
        for sign, value in ((1, plus[j]), (-1, minus[j])):
            normalized = value / scales[j]
            if normalized < best_value:
                best_value = normalized
                best = (j, sign, value)
    return best


def _solve(
    rows: Matrix,
    response: Vector,
    tau: float,
    names: tuple[str, ...],
    max_iterations: int,
) -> tuple[Vector, tuple[int, ...], int, bool]:
    """Minimize the check loss by the elemental-basis simplex walk.

    Returns:
        Coefficients, the final basis, the step count, and whether
        optimality was certified.

    Raises:
        RankDeficientDesignError: If the design is rank deficient.
        RuntimeError: If the iteration cap is reached.

    References:
        [EQ-INF-QR-001], [REF-BARRODALE-ROBERTS-1973],
        [REF-KOENKER-BASSETT-1978], [REF-KOENKER-2005].

    """
    n, p = len(rows), len(rows[0])
    basis = _initial_basis(rows, response, names)
    tolerance = 1e-10 * max(1.0, max(abs(y) for y in response))
    certified = True
    for iteration in range(max_iterations + 1):
        matrix = tuple(rows[i] for i in basis)
        beta = lu_solve(matrix, tuple(response[i] for i in basis))
        inverse = _inverse(matrix)
        residuals = [
            response[i] - math.fsum(b * x for b, x in zip(beta, rows[i], strict=True))
            for i in range(n)
        ]
        for i in basis:
            residuals[i] = 0.0
        zero = [abs(r) <= tolerance for r in residuals]
        plus, minus, scales, rates = _edge_derivatives(
            rows, residuals, basis, inverse, zero, tau
        )
        edge = _descent_edge(plus, minus, scales)
        if edge is None and sum(zero) > p:
            found = _degenerate_descent(rows, residuals, basis, zero, tau)
            if isinstance(found, str):
                certified = False
            elif found is not None:
                basis, edge, scales, rates = found
        if edge is None:
            return beta, tuple(sorted(basis)), iteration, certified
        if iteration == max_iterations:
            break
        j, sign, slope = edge
        in_basis = set(basis)
        breakpoints: list[tuple[float, int, float]] = []
        for i in range(n):
            if i in in_basis or zero[i]:
                continue
            rate = sign * rates[i][j]
            if rate != 0.0 and residuals[i] * rate < 0.0:
                breakpoints.append((-residuals[i] / rate, i, abs(rate)))
        breakpoints.sort()
        entering = -1
        for _, i, increment in breakpoints:
            slope += increment
            if slope >= -_DERIVATIVE_TOLERANCE * scales[j]:
                entering = i
                break
        if entering < 0:
            raise RuntimeError("check-loss objective is unbounded along an edge")
        basis = list(basis)
        basis[j] = entering
    raise RuntimeError(
        f"quantile regression did not converge in {max_iterations} steps"
    )


def _degenerate_descent(
    rows: Matrix,
    residuals: Sequence[float],
    basis: Sequence[int],
    zero: Sequence[bool],
    tau: float,
) -> (
    tuple[list[int], tuple[int, int, float], list[float], list[list[float]]]
    | str
    | None
):
    """Search every elemental basis of the zero set for a descending edge.

    At a vertex with more than ``p`` zero residuals the directional
    derivative is linear on each cone cut out by the zero-residual
    hyperplanes, so it is non-negative everywhere exactly when it is
    non-negative on every edge, and every edge is an edge of some elemental
    basis drawn from the zero set.

    Returns:
        ``None`` when no descending edge exists (optimal), the new basis,
        the edge, the tolerance scales, and the rates otherwise, or ``"uncertified"``
        when the zero set has too many bases to enumerate.

    References:
        [EQ-INF-QR-001], [REF-KOENKER-2005].

    """
    p = len(basis)
    # Repeated design rows span the same hyperplane, so one copy of each
    # suffices to generate every edge (bootstrap resamples repeat rows).
    distinct: dict[tuple[float, ...], int] = {}
    for i in basis:
        distinct.setdefault(tuple(rows[i]), i)
    for i, flag in enumerate(zero):
        if flag:
            distinct.setdefault(tuple(rows[i]), i)
    zero_set = sorted(distinct.values())
    if math.comb(len(zero_set), p) > _DEGENERATE_BASIS_LIMIT:
        warnings.warn(
            "degenerate quantile-regression vertex has too many elemental bases; "
            "optimality is not certified",
            RuntimeWarning,
            stacklevel=4,
        )
        return "uncertified"
    current = sorted(basis)
    if len(zero_set) <= p:
        return None
    for candidate in itertools.combinations(zero_set, p):
        if list(candidate) == current:
            continue
        matrix = tuple(rows[i] for i in candidate)
        try:
            inverse = _inverse(matrix)
        except ValueError:
            continue
        if any(not math.isfinite(v) or abs(v) > 1e12 for row in inverse for v in row):
            continue
        plus, minus, scales, rates = _edge_derivatives(
            rows, residuals, candidate, inverse, zero, tau
        )
        edge = _descent_edge(plus, minus, scales)
        if edge is not None:
            return list(candidate), edge, scales, rates
    return None


def fit_quantile_regression(
    design: DesignMatrix | Iterable[Iterable[object]],
    response: Iterable[object],
    *,
    quantile: float = 0.5,
    column_names: Sequence[str] | None = None,
    interval: IntervalMethod = "none",
    replicates: int = 500,
    confidence: float = 0.95,
    seed: int | RandomStream | None = None,
    max_iterations: int = 10000,
) -> QuantileRegressionResult:
    """Fit a linear regression quantile exactly by an elemental simplex walk.

    Minimizes ``sum_i rho_tau(y_i - x_i' beta)`` over ``beta``; the result
    interpolates ``p`` observations (``basis``) and satisfies the subgradient
    optimality condition. With ``interval="bootstrap"``, ``replicates``
    pairs resamples ``(x_i, y_i)`` are refitted and the percentile limits and
    standard deviations of each coefficient are reported; resamples whose
    design is rank deficient are redrawn.

    Args:
        design: :class:`DesignMatrix` or ``n x p`` rows; include a column of
            ones for an intercept.
        response: ``n`` finite responses.
        quantile: Level ``tau`` strictly inside ``(0, 1)``; ``0.5`` gives
            least-absolute-deviations (median) regression.
        column_names: Names for a raw design; defaults to ``x0, x1, ...``.
        interval: ``"none"`` or ``"bootstrap"``.
        replicates: Bootstrap replicates (at least 50).
        confidence: Level of the bootstrap intervals.
        seed: Integer seed or stream for the bootstrap; ``None`` is not
            reproducible.
        max_iterations: Cap on simplex steps per fit.

    Returns:
        :class:`QuantileRegressionResult`.

    Raises:
        RankDeficientDesignError: If the design is rank deficient.
        RuntimeError: If the simplex walk exceeds ``max_iterations`` or most
            bootstrap resamples are rank deficient.
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, ``n < p``, or an option is invalid.

    Equation:
        [EQ-INF-QR-001], [EQ-INF-QR-002].

    References:
        [REF-KOENKER-BASSETT-1978], [REF-BARRODALE-ROBERTS-1973],
        [REF-KOENKER-2005],
        [REF-EFRON-TIBSHIRANI-1993, Chapter 9].

    """
    rows, names, _ = _coerce_design(design, column_names)
    y = as_vector(response, name="response")
    n, p = len(rows), len(rows[0])
    if len(y) != n:
        raise ValueError("response length must equal the number of design rows")
    if n < p:
        raise ValueError("quantile regression needs at least as many rows as columns")
    tau = check_level(quantile, name="quantile")
    if interval not in ("none", "bootstrap"):
        raise ValueError("interval must be 'none' or 'bootstrap'")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    beta, basis, iterations, certified = _solve(rows, y, tau, names, max_iterations)
    exact = set(basis)
    residuals = tuple(
        0.0
        if i in exact
        else value - math.fsum(b * x for b, x in zip(beta, row, strict=True))
        for i, (row, value) in enumerate(zip(rows, y, strict=True))
    )
    objective = math.fsum(r * (tau - (1.0 if r < 0.0 else 0.0)) for r in residuals)
    errors: tuple[float, ...] | None = None
    intervals: tuple[ConfidenceInterval, ...] | None = None
    used = 0
    stream_seed: int | None = None
    if interval == "bootstrap":
        if isinstance(replicates, bool) or not isinstance(replicates, int):
            raise TypeError("replicates must be an integer")
        if replicates < 50:
            raise ValueError("replicates must be at least 50")
        level = check_level(confidence, name="confidence")
        stream = resolve_stream(seed)
        stream_seed = stream.seed
        draws = _bootstrap(rows, y, tau, names, replicates, stream, max_iterations)
        used = len(draws)
        columns = list(zip(*draws, strict=True))
        lower_p = require_probability(0.5 - 0.5 * level, name="lower")
        upper_p = require_probability(0.5 + 0.5 * level, name="upper")
        errors = tuple(statistics.stdev(column) for column in columns)
        label = f"pairs bootstrap percentile ({used} replicates)"
        intervals = tuple(
            ConfidenceInterval(
                sample_quantile(column, lower_p),
                sample_quantile(column, upper_p),
                level,
                label,
            )
            for column in columns
        )
    return QuantileRegressionResult(
        tuple(beta),
        names,
        tau,
        objective,
        residuals,
        tuple(value - r for value, r in zip(y, residuals, strict=True)),
        basis,
        iterations,
        certified,
        errors,
        intervals,
        used,
        stream_seed,
    )


def _bootstrap(
    rows: Matrix,
    response: Vector,
    tau: float,
    names: tuple[str, ...],
    replicates: int,
    stream: RandomStream,
    max_iterations: int,
) -> list[Vector]:
    """Refit the regression quantile on ``replicates`` pairs resamples.

    Raises:
        RuntimeError: If more rank-deficient resamples than replicates occur.

    References:
        [EQ-INF-QR-002], [REF-KOENKER-2005],
        [REF-EFRON-TIBSHIRANI-1993, Chapter 9].

    """
    generator = stream.child("bootstrap").generator()
    n = len(rows)
    draws: list[Vector] = []
    failures = 0
    while len(draws) < replicates:
        picks = [generator.randrange(n) for _ in range(n)]
        sample_rows = tuple(rows[i] for i in picks)
        sample_y = tuple(response[i] for i in picks)
        try:
            beta, _, _, _ = _solve(sample_rows, sample_y, tau, names, max_iterations)
        except RankDeficientDesignError:
            failures += 1
            if failures > replicates:
                raise RuntimeError(
                    "most bootstrap resamples have a rank-deficient design"
                ) from None
            continue
        draws.append(beta)
    return draws
