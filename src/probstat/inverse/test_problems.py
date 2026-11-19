"""Discretized ill-posed test problems with known exact solutions.

Each constructor returns the operator ``A``, the exact solution ``x``, and the
consistent exact data ``b = A x`` of a linear inverse problem whose singular
values decay to zero. No noise is added: the caller perturbs ``b`` with its
own seeded stream, so the experiment states its noise model explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector


@dataclass(frozen=True, slots=True)
class DiscreteIllPosedProblem:
    """A discretized linear inverse problem ``A x = b`` with its exact solution.

    Attributes:
        name: Short identifier of the problem family.
        operator: ``n x n`` forward matrix ``A``.
        exact_solution: Exact solution ``x`` sampled on ``grid``.
        exact_data: Noise-free data ``b = A x`` (computed with correctly
            rounded sums).
        grid: Discretization points of the solution.
        singular_values: Exact singular values of ``A`` in descending order
            when they are known in closed form, otherwise ``None``.

    References:
        [REF-HANSEN-2008, Section 2.1 and Chapter 4].

    """

    name: str
    operator: Matrix
    exact_solution: Vector
    exact_data: Vector
    grid: Vector
    singular_values: Vector | None


def _require_size(size: int) -> int:
    """Validate a discretization size of at least two points.

    Args:
        size: Candidate number of grid points.

    Returns:
        The validated size.

    Raises:
        TypeError: If ``size`` is not an integer.
        ValueError: If ``size`` is below two.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 2:
        raise ValueError("size must be at least two")
    return size


def _apply(matrix: Matrix, vector: Vector) -> Vector:
    """Return ``A x`` with correctly rounded row sums.

    Args:
        matrix: Square matrix.
        vector: Vector of matching length.

    Returns:
        The product.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    return tuple(
        math.fsum(a * x for a, x in zip(row, vector, strict=True)) for row in matrix
    )


def backward_heat_problem(
    size: int = 32, *, time: float = 0.005
) -> DiscreteIllPosedProblem:
    """Return the discrete backward heat problem with a closed-form SVD.

    The forward operator maps an initial temperature on ``(0, 1)`` with zero
    boundary values to the temperature after ``time`` under the
    finite-difference heat equation ``u' = -D u``, where ``D`` is the
    second-difference matrix on ``size`` interior points ``s_j = j h``,
    ``h = 1 / (size + 1)``. With the orthonormal sine vectors
    ``V_jk = sqrt(2 h) sin(j k pi h)`` and the eigenvalues
    ``mu_k = (4 / h^2) sin^2(k pi h / 2)`` of ``D``,
    ``A = V diag(exp(-time mu_k)) V'``, so the singular values
    ``exp(-time mu_k)`` decay like ``exp(-time pi^2 k^2)``: a severely
    ill-posed problem. The exact initial state is
    ``x(s) = (27 / 4) s (1 - s)^2``, which vanishes at both ends and has
    its maximum ``1`` at ``s = 1/3``.

    Args:
        size: Number of interior grid points (at least two).
        time: Positive diffusion time; larger values are more ill-posed.

    Returns:
        :class:`DiscreteIllPosedProblem` with exact singular values.

    Raises:
        TypeError: If ``size`` is not an integer.
        ValueError: If ``size < 2`` or ``time`` is not positive and finite.

    Equation:
        [EQ-INV-REG-PROB-001].

    References:
        [REF-HANSEN-2008, Sections 2.1 and 2.3] (discrete ill-posed
        problems and their SVD). The sine eigenpairs of the
        second-difference matrix are verified directly in
        ``tests/test_regularization.py``.

    """
    n = _require_size(size)
    duration = require_finite_real(time, name="time")
    if duration <= 0.0:
        raise ValueError("time must be positive")
    h = 1.0 / (n + 1)
    grid = tuple((j + 1) * h for j in range(n))
    scale = math.sqrt(2.0 * h)
    vectors = tuple(
        tuple(scale * math.sin((j + 1) * (k + 1) * math.pi * h) for k in range(n))
        for j in range(n)
    )
    eigenvalues = tuple(
        4.0 / h**2 * math.sin((k + 1) * math.pi * h / 2.0) ** 2 for k in range(n)
    )
    values = tuple(math.exp(-duration * mu) for mu in eigenvalues)
    operator = tuple(
        tuple(
            math.fsum(vectors[i][k] * values[k] * vectors[j][k] for k in range(n))
            for j in range(n)
        )
        for i in range(n)
    )
    solution = tuple(6.75 * s * (1.0 - s) ** 2 for s in grid)
    return DiscreteIllPosedProblem(
        name="backward_heat",
        operator=operator,
        exact_solution=solution,
        exact_data=_apply(operator, solution),
        grid=grid,
        singular_values=values,
    )


def second_derivative_problem(size: int = 32) -> DiscreteIllPosedProblem:
    """Return the ``deriv2`` integral equation discretized by the midpoint rule.

    The first-kind Fredholm equation ``int_0^1 K(s, t) f(t) dt = g(s)`` with
    the Green's function of the second derivative,
    ``K(s, t) = s (t - 1)`` for ``s < t`` and ``t (s - 1)`` for ``s >= t``,
    is discretized at the midpoints ``s_i = t_i = (i - 1/2) h``, ``h = 1/n``,
    as ``A_ij = h K(s_i, t_j)``. Its singular values decay like ``k^-2``: a
    mildly ill-posed problem. The exact solution is ``f(t) = t``, whose
    continuous data are ``g(s) = (s^3 - s) / 6``; ``exact_data`` is the
    consistent discrete ``A x``, which differs from ``g(s_i)`` by the
    ``O(h^2)`` quadrature error. Regularization Tools discretizes the same
    equation by a Galerkin method, so its matrix differs from this one.

    Args:
        size: Number of grid points (at least two).

    Returns:
        :class:`DiscreteIllPosedProblem` without closed-form singular values.

    Raises:
        TypeError: If ``size`` is not an integer.
        ValueError: If ``size < 2``.

    Equation:
        [EQ-INV-REG-PROB-002].

    References:
        [REF-HANSEN-2008, Chapter 4, deriv2] (kernel and the case
        ``f(t) = t``).

    """
    n = _require_size(size)
    h = 1.0 / n
    grid = tuple((i + 0.5) * h for i in range(n))

    def kernel(s: float, t: float) -> float:
        """Evaluate the Green's function of the second derivative.

        References:
            [REF-HANSEN-2008, Chapter 4, deriv2]. No other equation is used.

        """
        return s * (t - 1.0) if s < t else t * (s - 1.0)

    operator = tuple(tuple(h * kernel(s, t) for t in grid) for s in grid)
    solution = grid
    return DiscreteIllPosedProblem(
        name="second_derivative",
        operator=operator,
        exact_solution=solution,
        exact_data=_apply(operator, solution),
        grid=grid,
        singular_values=None,
    )
