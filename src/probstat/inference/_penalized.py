"""Penalized least squares with GCV or REML smoothing-parameter selection.

The shared engine behind :mod:`.splines` (P-splines) and
:mod:`.additive_models`. For a design ``X``, positive weights ``W``, and
penalties ``S_j`` with smoothing parameters ``lambda_j`` it solves
``(X'WX + sum_j lambda_j S_j) beta = X'Wy`` by Cholesky (EQ-SMOOTH-002),
reports the effective degrees of freedom ``tr(A^-1 X'WX)`` (EQ-SMOOTH-003),
and evaluates the GCV (EQ-SMOOTH-004) and REML (EQ-SMOOTH-005) criteria
whose minimizers choose the ``lambda_j``.

References:
    [REF-WOOD-2017, Chapters 4 and 6], [REF-WOOD-2011],
    [REF-CRAVEN-WAHBA-1979], [REF-EILERS-MARX-1996].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.linalg import (
    Matrix,
    Vector,
    cholesky,
    cholesky_solve,
    dot,
    inverse_spd,
    matvec,
)
from probstat.optimization import minimize_nelder_mead

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
METHODS = ("gcv", "reml")


@dataclass(frozen=True, slots=True)
class PenalizedSolution:
    """One penalized least-squares solve at fixed smoothing parameters.

    Args:
        coefficients: ``beta`` solving the penalized normal equations.
        fitted: ``X beta``.
        rss: Weighted residual sum of squares ``sum w_i (y_i - mu_i)^2``.
        penalty: ``sum_j lambda_j beta' S_j beta``.
        edf: Effective degrees of freedom ``tr(A^-1 X'WX)``.
        term_edf: Per-block effective degrees of freedom (diagonal sums of
            ``A^-1 X'WX`` over each penalty's columns).
        inverse: ``A^-1`` with ``A = X'WX + sum_j lambda_j S_j``.
        log_det: ``log det A``.

    References:
        [EQ-SMOOTH-002], [EQ-SMOOTH-003], [REF-WOOD-2017, Chapter 6].

    """

    coefficients: Vector
    fitted: Vector
    rss: float
    penalty: float
    edf: float
    term_edf: tuple[float, ...]
    inverse: Matrix
    log_det: float


@dataclass(frozen=True, slots=True)
class PenalizedProblem:
    """A weighted design, response, and block penalties.

    Args:
        design: ``n x p`` design matrix ``X``.
        response: Response ``y``.
        weights: Positive prior weights ``w``.
        penalties: Penalty matrices ``S_j`` (``p x p``).
        ranks: Rank of each ``S_j``.
        blocks: Column indices penalized by each ``S_j`` (for per-term EDF).

    References:
        [EQ-SMOOTH-002], [REF-WOOD-2017, Chapter 4].

    """

    design: Matrix
    response: Vector
    weights: Vector
    penalties: tuple[Matrix, ...]
    ranks: tuple[int, ...]
    blocks: tuple[tuple[int, ...], ...]

    @property
    def gram(self) -> Matrix:
        """Return ``X'WX``.

        References:
            [EQ-SMOOTH-002]. No further equation is evaluated.

        """
        columns = len(self.design[0])
        return tuple(
            tuple(
                math.fsum(
                    w * row[i] * row[j]
                    for w, row in zip(self.weights, self.design, strict=True)
                )
                for j in range(columns)
            )
            for i in range(columns)
        )

    def solve(
        self, smoothing: Sequence[float], gram: Matrix | None = None
    ) -> PenalizedSolution:
        """Solve the penalized normal equations at fixed ``lambda``.

        Args:
            smoothing: One non-negative ``lambda_j`` per penalty.
            gram: Precomputed ``X'WX`` (computed when omitted).

        Returns:
            The penalized solution.

        Raises:
            ValueError: If ``X'WX + S_lambda`` is not positive definite (the
                design does not identify the unpenalized directions).

        References:
            [EQ-SMOOTH-002], [EQ-SMOOTH-003], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        g = self.gram if gram is None else gram
        size = len(g)
        system = [list(row) for row in g]
        for lam, penalty in zip(smoothing, self.penalties, strict=True):
            for i in range(size):
                for j in range(size):
                    system[i][j] += lam * penalty[i][j]
        a: Matrix = tuple(tuple(row) for row in system)
        lower = cholesky(a)
        rhs = tuple(
            math.fsum(
                w * row[i] * y
                for w, row, y in zip(
                    self.weights, self.design, self.response, strict=True
                )
            )
            for i in range(size)
        )
        beta = cholesky_solve(lower, rhs)
        fitted = matvec(self.design, beta)
        rss = math.fsum(
            w * (y - m) ** 2
            for w, y, m in zip(self.weights, self.response, fitted, strict=True)
        )
        penalty_value = math.fsum(
            lam * dot(beta, matvec(penalty, beta))
            for lam, penalty in zip(smoothing, self.penalties, strict=True)
        )
        inverse = inverse_spd(a)
        influence_diagonal = [
            math.fsum(inverse[i][k] * g[k][i] for k in range(size)) for i in range(size)
        ]
        term_edf = tuple(
            math.fsum(influence_diagonal[i] for i in block) for block in self.blocks
        )
        log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(size))
        return PenalizedSolution(
            coefficients=beta,
            fitted=fitted,
            rss=rss,
            penalty=penalty_value,
            edf=math.fsum(influence_diagonal),
            term_edf=term_edf,
            inverse=inverse,
            log_det=log_det,
        )

    def criterion(
        self, method: str, smoothing: Sequence[float], solution: PenalizedSolution
    ) -> float:
        """Return the GCV score or the REML criterion of a solution.

        GCV is ``n RSS / (n - edf)^2``; REML is the negative restricted
        log-likelihood up to a constant free of ``lambda``,
        ``((n - M_p)/2)(1 + log(2 pi phi)) + (1/2) log|A|
        - (1/2) sum_j r_j log lambda_j`` with
        ``phi = (RSS + penalty)/(n - M_p)`` and ``M_p = p - sum_j r_j``.

        Args:
            method: ``"gcv"`` or ``"reml"``.
            smoothing: The ``lambda_j`` of the solution.
            solution: Result of :meth:`solve`.

        Returns:
            Criterion value (``inf`` where it is undefined).

        References:
            [EQ-SMOOTH-004], [EQ-SMOOTH-005], [REF-CRAVEN-WAHBA-1979],
            [REF-WOOD-2011].

        """
        n = len(self.response)
        if method == "gcv":
            denominator = n - solution.edf
            if denominator <= 0.0:
                return math.inf
            return n * solution.rss / (denominator * denominator)
        null_dimension = len(self.design[0]) - sum(self.ranks)
        residual_df = n - null_dimension
        total = solution.rss + solution.penalty
        if residual_df <= 0 or total <= 0.0 or any(lam <= 0.0 for lam in smoothing):
            return math.inf
        phi = total / residual_df
        return (
            0.5 * residual_df * (1.0 + math.log(2.0 * math.pi * phi))
            + 0.5 * solution.log_det
            - 0.5
            * math.fsum(
                r * math.log(lam) for r, lam in zip(self.ranks, smoothing, strict=True)
            )
        )


def check_method(method: object) -> str:
    """Validate a smoothing-parameter selection method.

    Args:
        method: ``"gcv"`` or ``"reml"``.

    Returns:
        The method name.

    Raises:
        ValueError: For any other value.

    References:
        [EQ-SMOOTH-004], [EQ-SMOOTH-005]. No further equation is evaluated.

    """
    if method not in METHODS:
        raise ValueError("method must be 'gcv' or 'reml'")
    return str(method)


def penalty_scale(gram: Matrix, penalty: Matrix) -> float:
    """Return ``tr(X'WX)/tr(S)``, the scale that makes ``lambda`` unit-free.

    Args:
        gram: ``X'WX``.
        penalty: ``S``.

    Returns:
        Positive scale (``1`` if a trace vanishes).

    References:
        [REF-WOOD-2017, Chapter 6]. No named equation is evaluated.

    """
    numerator = math.fsum(gram[i][i] for i in range(len(gram)))
    denominator = math.fsum(penalty[i][i] for i in range(len(penalty)))
    if numerator <= 0.0 or denominator <= 0.0:
        return 1.0
    return numerator / denominator


def minimize_log_scalar(
    objective: Callable[[float], float],
    centre: float,
    *,
    half_width: float = 18.0,
    grid_points: int = 37,
    tolerance: float = 1e-7,
) -> tuple[float, float, tuple[tuple[float, float], ...]]:
    """Minimize a criterion of ``rho = log lambda`` by grid search and golden section.

    The grid ``centre + linspace(-half_width, half_width, grid_points)``
    brackets the global minimum on that range (criteria in ``rho`` can have
    several local minima); golden-section search refines the bracket
    around the best grid point.

    Args:
        objective: Criterion as a function of ``rho``.
        centre: Centre of the grid.
        half_width: Half-width of the grid in ``rho``.
        grid_points: Number of grid points (odd).
        tolerance: Bracket width at which the refinement stops.

    Returns:
        ``(rho, value, profile)`` with the grid profile ``(rho, value)``.

    References:
        [REF-WOOD-2017, Chapter 6], [REF-NOCEDAL-WRIGHT-2006, Chapter 3].
        No named equation is evaluated.

    """
    step = 2.0 * half_width / (grid_points - 1)
    grid = [centre - half_width + step * index for index in range(grid_points)]
    values = [objective(rho) for rho in grid]
    best = min(range(grid_points), key=lambda index: values[index])
    profile = tuple(zip(grid, values, strict=True))
    if not math.isfinite(values[best]):
        raise ValueError("the smoothing criterion is not finite on the search grid")
    if best in (0, grid_points - 1):
        return grid[best], values[best], profile
    low, high = grid[best - 1], grid[best + 1]
    x1 = high - _GOLDEN * (high - low)
    x2 = low + _GOLDEN * (high - low)
    f1, f2 = objective(x1), objective(x2)
    while high - low > tolerance:
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - _GOLDEN * (high - low)
            f1 = objective(x1)
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + _GOLDEN * (high - low)
            f2 = objective(x2)
    rho, value = (x1, f1) if f1 <= f2 else (x2, f2)
    if values[best] < value:
        return grid[best], values[best], profile
    return rho, value, profile


def select_smoothing(
    problem: PenalizedProblem,
    method: str,
    fixed: Sequence[float | None],
) -> tuple[tuple[float, ...], float, tuple[tuple[float, float], ...]]:
    """Choose the free smoothing parameters by minimizing GCV or REML.

    One free parameter is found by :func:`minimize_log_scalar`; several by
    a common-offset grid search followed by Nelder-Mead in ``log lambda``
    within ``18`` natural-log units of each penalty scale.

    Args:
        problem: Penalized problem.
        method: ``"gcv"`` or ``"reml"``.
        fixed: Per penalty, a fixed ``lambda`` or ``None`` to select it.

    Returns:
        ``(lambdas, criterion, profile)``; ``profile`` is the grid profile
        of a single free parameter and empty otherwise.

    References:
        [EQ-SMOOTH-004], [EQ-SMOOTH-005], [REF-WOOD-2017, Chapter 6],
        [REF-NELDER-MEAD-1965].

    """
    gram = problem.gram
    free = [index for index, value in enumerate(fixed) if value is None]
    centres = [
        math.log(penalty_scale(gram, problem.penalties[index])) for index in free
    ]

    def full(rhos: Sequence[float]) -> list[float]:
        """Merge free ``log lambda`` values into the full parameter list.

        References:
            [EQ-SMOOTH-002]. No further equation is evaluated.

        """
        values = [0.0 if value is None else float(value) for value in fixed]
        for index, rho in zip(free, rhos, strict=True):
            values[index] = math.exp(rho)
        return values

    def evaluate(rhos: Sequence[float]) -> float:
        """Return the criterion at the given free ``log lambda`` values.

        References:
            [EQ-SMOOTH-004], [EQ-SMOOTH-005]. No further equation is evaluated.

        """
        lambdas = full(rhos)
        try:
            solution = problem.solve(lambdas, gram)
        except ValueError:
            return math.inf
        return problem.criterion(method, lambdas, solution)

    if not free:
        lambdas = full(())
        solution = problem.solve(lambdas, gram)
        return tuple(lambdas), problem.criterion(method, lambdas, solution), ()
    if len(free) == 1:
        rho, value, profile = minimize_log_scalar(lambda r: evaluate((r,)), centres[0])
        return tuple(full((rho,))), value, profile
    offsets = [-12.0 + 2.0 * index for index in range(13)]
    start_offset = min(offsets, key=lambda c: evaluate([m + c for m in centres]))
    start = [m + start_offset for m in centres]
    bounds = [(m - 18.0, m + 18.0) for m in centres]
    result = minimize_nelder_mead(
        evaluate, start, initial_step=0.2, tolerance=1e-10, bounds=bounds
    )
    rhos = result.x if result.value <= evaluate(start) else start
    return tuple(full(rhos)), evaluate(rhos), ()
