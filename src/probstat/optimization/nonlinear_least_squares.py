"""Levenberg-Marquardt nonlinear least squares and its Gauss-Newton covariance."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    dot,
    inverse_spd,
    matmul,
    matvec,
    norm,
    scale_matrix,
    solve_spd,
    transpose,
)

from ._common import VectorFunction
from .finite_differences import central_jacobian
from .results import LeastSquaresResult


def least_squares(
    residual: VectorFunction,
    x0: Sequence[float],
    *,
    jacobian: Callable[[Vector], Sequence[Sequence[float]]] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> LeastSquaresResult:
    """Minimize ``0.5 ||r(x)||^2`` by Levenberg-Marquardt with Marquardt scaling.

    Args:
        residual: Vector residual function with fixed output length.
        x0: Finite starting point.
        jacobian: Optional analytic Jacobian; central differences otherwise.
        tolerance: Relative cost-reduction and step tolerance.
        max_iterations: Maximum accepted-or-rejected iterations.

    Returns:
        Least-squares result with the final Jacobian for covariance estimates.

    Equation:
        [EQ-OPT-LM-001].

    References:
        [REF-MORE-1978], [REF-NOCEDAL-WRIGHT-2006, Section 10.3].

    """
    x = as_vector(x0, name="x0")
    if not x:
        raise ValueError("x0 must be non-empty")
    evaluations = 0

    def r(point: Vector) -> Vector:
        """Evaluate and count the residual vector.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Chapter 10]. No equation is evaluated.

        """
        nonlocal evaluations
        evaluations += 1
        return as_vector(residual(point), name="residual")

    def jac(point: Vector) -> Matrix:
        """Evaluate and count the Jacobian.

        References:
            [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        nonlocal evaluations
        if jacobian is not None:
            evaluations += 1
            return tuple(as_vector(row, name="jacobian row") for row in jacobian(point))
        matrix, used = central_jacobian(residual, point)
        evaluations += used
        return matrix

    residuals = r(x)
    if len(residuals) < len(x):
        raise ValueError("least squares needs at least as many residuals as parameters")
    cost = 0.5 * dot(residuals, residuals)
    j = jac(x)
    damping = 1e-3
    for iteration in range(max_iterations):
        jt = transpose(j)
        gram = matmul(jt, j)
        gradient = matvec(jt, residuals)
        if norm(gradient) <= tolerance * max(1.0, cost):
            return LeastSquaresResult(
                x,
                residuals,
                cost,
                j,
                iteration,
                evaluations,
                True,
                "gradient tolerance met",
            )
        diagonal = tuple(max(gram[i][i], 1e-12) for i in range(len(x)))
        while True:
            damped = tuple(
                tuple(
                    gram[a][b] + (damping * diagonal[a] if a == b else 0.0)
                    for b in range(len(x))
                )
                for a in range(len(x))
            )
            try:
                delta = solve_spd(damped, tuple(-g for g in gradient))
            except ValueError:
                damping *= 10.0
                continue
            candidate = tuple(a + b for a, b in zip(x, delta, strict=True))
            try:
                candidate_residuals = r(candidate)
                candidate_cost = 0.5 * dot(candidate_residuals, candidate_residuals)
            except ValueError:
                candidate_cost = math.inf
            if candidate_cost < cost:
                break
            damping *= 10.0
            if damping > 1e16:
                return LeastSquaresResult(
                    x,
                    residuals,
                    cost,
                    j,
                    iteration,
                    evaluations,
                    False,
                    "damping exceeded without reducing the cost",
                )
        reduction = cost - candidate_cost
        step_small = norm(delta) <= tolerance * (norm(x) + tolerance)
        x, residuals, cost = candidate, candidate_residuals, candidate_cost
        j = jac(x)
        damping = max(damping / 10.0, 1e-12)
        if reduction <= tolerance * max(cost, 1e-300) or step_small:
            return LeastSquaresResult(
                x,
                residuals,
                cost,
                j,
                iteration + 1,
                evaluations,
                True,
                "relative reduction or step tolerance met",
            )
    return LeastSquaresResult(
        x,
        residuals,
        cost,
        j,
        max_iterations,
        evaluations,
        False,
        "iteration limit reached",
    )


def gauss_newton_covariance(
    result: LeastSquaresResult, *, noise_variance: float | None = None
) -> Matrix:
    """Return the Gauss-Newton parameter covariance ``s^2 (J'J)^-1``.

    Args:
        result: Converged least-squares result.
        noise_variance: Known residual variance; if omitted it is estimated as
            ``||r||^2 / (m - n)``.

    Returns:
        Asymptotic parameter covariance under local linearity.

    Raises:
        ValueError: If ``m <= n`` without a known variance or ``J'J`` is singular.

    Equation:
        [EQ-OPT-LM-002].

    References:
        [REF-SEBER-WILD-2003, Section 2.1], [REF-NOCEDAL-WRIGHT-2006, Chapter 10].

    """
    m, n = len(result.residuals), len(result.x)
    if noise_variance is None:
        if m <= n:
            raise ValueError(
                "estimating noise variance requires more residuals than parameters"
            )
        variance = 2.0 * result.cost / (m - n)
    else:
        variance = require_finite_real(noise_variance, name="noise_variance")
        if variance <= 0.0:
            raise ValueError("noise_variance must be positive")
    gram = matmul(transpose(result.jacobian), result.jacobian)
    return scale_matrix(inverse_spd(gram), variance)
