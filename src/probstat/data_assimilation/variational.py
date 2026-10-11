"""Three-dimensional variational (3D-Var) analysis.

3D-Var finds the single state that best balances a background (prior)
estimate against one batch of observations by minimizing

``J(x) = 1/2 (x - x_b)' B^-1 (x - x_b) + 1/2 (y - h(x))' R^-1 (y - h(x))``.

For a linear ``h`` the minimizer is the Kalman/BLUE analysis mean, so 3D-Var
and one Kalman update agree; for nonlinear ``h`` it is the maximum a
posteriori state of the Gaussian prior/likelihood pair. Unlike the Kalman
filter it does not evolve ``B`` in time: the background covariance is a
modeling input, not a propagated quantity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_matrix,
    as_vector,
    dot,
    inverse_spd,
    matmul,
    matvec,
    symmetrize,
    transpose,
)
from probstat.optimization import central_jacobian, minimize_bfgs


@dataclass(frozen=True, slots=True)
class VariationalAnalysis:
    """Outcome of a 3D-Var minimization.

    Args:
        analysis: Minimizing state ``x_a``.
        cost: ``J(x_a)``.
        background_cost: Background term of ``J`` at ``x_a``.
        observation_cost: Observation term of ``J`` at ``x_a``.
        analysis_covariance: Gauss-Newton inverse Hessian
            ``(B^-1 + H' R^-1 H)^-1`` at ``x_a``; exact posterior covariance
            only for linear ``h``.
        converged: Whether the optimizer met its tolerance.
        iterations: Optimizer iterations.
        evaluations: Cost/gradient evaluations.

    References:
        [REF-LORENC-1986].

    """

    analysis: Vector
    cost: float
    background_cost: float
    observation_cost: float
    analysis_covariance: Matrix
    converged: bool
    iterations: int
    evaluations: int


def three_dimensional_variational(
    background: Sequence[float],
    background_covariance: Iterable[Iterable[object]],
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Iterable[Iterable[object]],
    *,
    observation_jacobian: Callable[[Vector], Iterable[Iterable[float]]] | None = None,
    gradient_tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> VariationalAnalysis:
    """Minimize the 3D-Var cost function by BFGS with an analytic gradient.

    ``grad J(x) = B^-1 (x - x_b) - H_x' R^-1 (y - h(x))`` with
    ``H_x = dh/dx`` (analytic when supplied, central differences otherwise).

    Args:
        background: Background state ``x_b`` (also the starting point).
        background_covariance: Positive-definite ``B``.
        observation: Observation ``y``.
        observation_function: Observation operator ``h``.
        observation_covariance: Positive-definite ``R``.
        observation_jacobian: Optional analytic ``dh/dx``.
        gradient_tolerance: BFGS relative gradient tolerance.
        max_iterations: BFGS iteration limit.

    Returns:
        Analysis state, cost decomposition, and analysis covariance.

    Raises:
        ValueError: If shapes disagree or a covariance is not positive
            definite.

    Equation:
        [EQ-DA-3DVAR-001], [EQ-OPT-BFGS-001].

    References:
        [REF-LORENC-1986], [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    x_b = as_vector(background, name="background")
    y = as_vector(observation, name="observation")
    b_matrix = symmetrize(background_covariance, name="background_covariance")
    r_matrix = symmetrize(observation_covariance, name="observation_covariance")
    if len(b_matrix) != len(x_b):
        raise ValueError("background_covariance must match the state dimension")
    if len(r_matrix) != len(y):
        raise ValueError("observation_covariance must match the observation length")
    try:
        b_inverse = inverse_spd(b_matrix)
    except ValueError as error:
        raise ValueError("background_covariance must be positive definite") from error
    try:
        r_inverse = inverse_spd(r_matrix)
    except ValueError as error:
        raise ValueError("observation_covariance must be positive definite") from error

    def observe(state: Vector) -> Vector:
        """Evaluate ``h`` with a shape check.

        References:
            [EQ-DA-3DVAR-001], [REF-LORENC-1986].

        """
        image = as_vector(observation_function(state), name="h(x)")
        if len(image) != len(y):
            raise ValueError("h(x) length must equal the observation length")
        return image

    def jacobian(state: Vector) -> Matrix:
        """Return ``dh/dx`` analytically or by central differences.

        References:
            [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        if observation_jacobian is not None:
            matrix = as_matrix(observation_jacobian(state), name="observation_jacobian")
        else:
            matrix, _ = central_jacobian(observe, state)
        if len(matrix) != len(y) or len(matrix[0]) != len(x_b):
            raise ValueError("observation Jacobian has the wrong shape")
        return matrix

    def terms(state: Vector) -> tuple[float, float, Vector, Vector]:
        """Return both cost terms and the whitened residuals.

        References:
            [EQ-DA-3DVAR-001], [REF-LORENC-1986].

        """
        departure = tuple(a - b for a, b in zip(state, x_b, strict=True))
        residual = tuple(a - b for a, b in zip(y, observe(state), strict=True))
        background_term = 0.5 * dot(departure, matvec(b_inverse, departure))
        observation_term = 0.5 * dot(residual, matvec(r_inverse, residual))
        return background_term, observation_term, departure, residual

    def cost(state: Vector) -> float:
        """Evaluate ``J(x)``.

        References:
            [EQ-DA-3DVAR-001], [REF-LORENC-1986].

        """
        background_term, observation_term, _, _ = terms(state)
        return background_term + observation_term

    def gradient(state: Vector) -> Vector:
        """Evaluate ``grad J(x)``.

        References:
            [EQ-DA-3DVAR-001], [REF-LORENC-1986].

        """
        _, _, departure, residual = terms(state)
        background_part = matvec(b_inverse, departure)
        observation_part = matvec(
            transpose(jacobian(state)), matvec(r_inverse, residual)
        )
        return tuple(
            a - b for a, b in zip(background_part, observation_part, strict=True)
        )

    result = minimize_bfgs(
        cost,
        x_b,
        gradient=gradient,
        gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations,
    )
    background_term, observation_term, _, _ = terms(result.x)
    linearized = jacobian(result.x)
    hessian = add(
        b_inverse, matmul(matmul(transpose(linearized), r_inverse), linearized)
    )
    symmetric = tuple(
        tuple(0.5 * (hessian[i][j] + hessian[j][i]) for j in range(len(hessian)))
        for i in range(len(hessian))
    )
    covariance = inverse_spd(symmetric)
    return VariationalAnalysis(
        result.x,
        math.fsum((background_term, observation_term)),
        background_term,
        observation_term,
        covariance,
        result.converged,
        result.iterations,
        result.evaluations,
    )
