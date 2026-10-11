"""Dependency-free numerical optimization, root finding, and differentiation.

These solvers are deliberately small and transparent: each returns its
evaluation count and a convergence flag instead of silently returning the last
iterate. They serve reliability (design points), inverse problems (least
squares and MAP estimates), and surrogate hyperparameter fitting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_vector,
    dot,
    identity,
    inverse_spd,
    matmul,
    matvec,
    norm,
    outer,
    scale_matrix,
    solve_spd,
    transpose,
)

ScalarFunction = Callable[[Vector], float]
VectorFunction = Callable[[Vector], Sequence[float]]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Outcome of an unconstrained minimization.

    Args:
        x: Final iterate.
        value: Objective value at ``x``.
        gradient_norm: Euclidean gradient norm at ``x`` (``nan`` if unused).
        iterations: Completed iterations.
        evaluations: Objective evaluations, including finite differences.
        converged: Whether a stopping tolerance was met.
        message: Human-readable termination reason.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2].

    """

    x: Vector
    value: float
    gradient_norm: float
    iterations: int
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class LeastSquaresResult:
    """Outcome of a nonlinear least-squares fit.

    Args:
        x: Final parameter vector.
        residuals: Residual vector ``r(x)``.
        cost: Half the residual sum of squares ``0.5 ||r||^2``.
        jacobian: Residual Jacobian at ``x``.
        iterations: Completed Levenberg-Marquardt iterations.
        evaluations: Residual-function evaluations.
        converged: Whether a stopping tolerance was met.
        message: Termination reason.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 10], [REF-MORE-1978].

    """

    x: Vector
    residuals: Vector
    cost: float
    jacobian: Matrix
    iterations: int
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class RootResult:
    """Outcome of a bracketing scalar root search.

    Args:
        root: Approximate root.
        value: Function value at ``root``.
        iterations: Completed iterations.
        evaluations: Function evaluations.
        converged: Whether the bracket width or residual tolerance was met.

    References:
        [REF-BRENT-1973, Chapter 4].

    """

    root: float
    value: float
    iterations: int
    evaluations: int
    converged: bool


def _finite_value(value: object, name: str) -> float:
    """Validate one finite objective or residual value.

    References:
        Project validation policy. No equation is evaluated.

    """
    return require_finite_real(value, name=name)


def _step(x: float, relative_step: float) -> float:
    """Return a scale-aware finite-difference step.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

    """
    return relative_step * max(1.0, abs(x))


def central_gradient(
    function: ScalarFunction, x: Sequence[float], *, relative_step: float = 1e-6
) -> tuple[Vector, int]:
    """Approximate a gradient by central differences.

    Args:
        function: Scalar function of a vector.
        x: Finite evaluation point.
        relative_step: Step relative to ``max(1, |x_i|)``.

    Returns:
        Gradient approximation and the number of function evaluations.

    Equation:
        [EQ-OPT-FD-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    point = as_vector(x, name="x")
    gradient: list[float] = []
    for index, value in enumerate(point):
        h = _step(value, relative_step)
        forward = list(point)
        backward = list(point)
        forward[index] = value + h
        backward[index] = value - h
        upper = _finite_value(function(tuple(forward)), "f(x+h)")
        lower = _finite_value(function(tuple(backward)), "f(x-h)")
        gradient.append((upper - lower) / (2.0 * h))
    return tuple(gradient), 2 * len(point)


def central_jacobian(
    function: VectorFunction, x: Sequence[float], *, relative_step: float = 1e-6
) -> tuple[Matrix, int]:
    """Approximate the Jacobian of a vector function by central differences.

    Args:
        function: Vector-valued function of a vector.
        x: Finite evaluation point.
        relative_step: Step relative to ``max(1, |x_i|)``.

    Returns:
        ``m`` by ``n`` Jacobian approximation and evaluation count.

    Equation:
        [EQ-OPT-FD-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    point = as_vector(x, name="x")
    columns: list[Vector] = []
    for index, value in enumerate(point):
        h = _step(value, relative_step)
        forward = list(point)
        backward = list(point)
        forward[index] = value + h
        backward[index] = value - h
        upper = as_vector(function(tuple(forward)), name="F(x+h)")
        lower = as_vector(function(tuple(backward)), name="F(x-h)")
        if len(upper) != len(lower):
            raise ValueError("vector function changed output length")
        columns.append(
            tuple((u - v) / (2.0 * h) for u, v in zip(upper, lower, strict=True))
        )
    return transpose(tuple(columns)), 2 * len(point)


def central_hessian(
    function: ScalarFunction, x: Sequence[float], *, relative_step: float = 1e-4
) -> tuple[Matrix, int]:
    """Approximate a symmetric Hessian by second-order central differences.

    Args:
        function: Twice-differentiable scalar function.
        x: Finite evaluation point.
        relative_step: Step relative to ``max(1, |x_i|)``.

    Returns:
        Symmetric Hessian approximation and evaluation count.

    Equation:
        [EQ-OPT-FD-002].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    point = as_vector(x, name="x")
    size = len(point)
    steps = [_step(value, relative_step) for value in point]
    center = _finite_value(function(point), "f(x)")
    evaluations = 1
    hessian = [[0.0] * size for _ in range(size)]

    def shifted(i: int, si: float, j: int | None = None, sj: float = 0.0) -> float:
        """Evaluate the function at a point shifted along one or two axes.

        References:
            [EQ-OPT-FD-002], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        nonlocal evaluations
        moved = list(point)
        moved[i] += si * steps[i]
        if j is not None:
            moved[j] += sj * steps[j]
        evaluations += 1
        return _finite_value(function(tuple(moved)), "f(shifted x)")

    for i in range(size):
        plus = shifted(i, 1.0)
        minus = shifted(i, -1.0)
        hessian[i][i] = (plus - 2.0 * center + minus) / (steps[i] ** 2)
        for j in range(i + 1, size):
            value = (
                shifted(i, 1.0, j, 1.0)
                - shifted(i, 1.0, j, -1.0)
                - shifted(i, -1.0, j, 1.0)
                + shifted(i, -1.0, j, -1.0)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i][j] = hessian[j][i] = value
    return tuple(tuple(row) for row in hessian), evaluations


def minimize_bfgs(
    function: ScalarFunction,
    x0: Sequence[float],
    *,
    gradient: Callable[[Vector], Sequence[float]] | None = None,
    gradient_tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> OptimizationResult:
    """Minimize a smooth function by BFGS with an Armijo backtracking search.

    Args:
        function: Smooth scalar objective.
        x0: Finite starting point.
        gradient: Optional analytic gradient; central differences otherwise.
        gradient_tolerance: Stop when ``||grad|| <= tol * max(1, |f|)``.
        max_iterations: Maximum number of quasi-Newton iterations.

    Returns:
        Minimization result with evaluation accounting.

    Equation:
        [EQ-OPT-BFGS-001], [EQ-OPT-ARMIJO-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Sections 3.1 and 6.1].

    """
    x = as_vector(x0, name="x0")
    if not x:
        raise ValueError("x0 must be non-empty")
    evaluations = 0

    def objective(point: Vector) -> float:
        """Evaluate and count the objective.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Chapter 2]. No equation is evaluated.

        """
        nonlocal evaluations
        evaluations += 1
        return _finite_value(function(point), "f(x)")

    def grad(point: Vector) -> Vector:
        """Evaluate and count the gradient.

        References:
            [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        nonlocal evaluations
        if gradient is not None:
            evaluations += 1
            return as_vector(gradient(point), name="gradient")
        value, used = central_gradient(function, point)
        evaluations += used
        return value

    size = len(x)
    inverse_hessian = identity(size)
    value = objective(x)
    g = grad(x)
    for iteration in range(max_iterations):
        g_norm = norm(g)
        if g_norm <= gradient_tolerance * max(1.0, abs(value)):
            return OptimizationResult(
                x, value, g_norm, iteration, evaluations, True, "gradient tolerance met"
            )
        direction = tuple(-d for d in matvec(inverse_hessian, g))
        slope = dot(g, direction)
        if slope >= 0.0:
            inverse_hessian = identity(size)
            direction = tuple(-d for d in g)
            slope = -dot(g, g)
        step = 1.0
        while True:
            candidate = tuple(
                xi + step * di for xi, di in zip(x, direction, strict=True)
            )
            try:
                candidate_value = objective(candidate)
            except ValueError:
                candidate_value = math.inf
            if candidate_value <= value + 1e-4 * step * slope:
                break
            step *= 0.5
            if step < 1e-16:
                return OptimizationResult(
                    x,
                    value,
                    g_norm,
                    iteration,
                    evaluations,
                    False,
                    "line search failed to decrease the objective",
                )
        new_g = grad(candidate)
        s = tuple(a - b for a, b in zip(candidate, x, strict=True))
        y = tuple(a - b for a, b in zip(new_g, g, strict=True))
        sy = dot(s, y)
        if sy > 1e-12 * norm(s) * norm(y):
            rho = 1.0 / sy
            left = add(identity(size), outer(s, y), scale=-rho)
            inverse_hessian = add(
                matmul(matmul(left, inverse_hessian), transpose(left)),
                outer(s, s),
                scale=rho,
            )
        if abs(candidate_value - value) <= 1e-15 * max(1.0, abs(value)) and norm(
            s
        ) <= 1e-15 * max(1.0, norm(x)):
            x, value, g = candidate, candidate_value, new_g
            return OptimizationResult(
                x,
                value,
                norm(g),
                iteration + 1,
                evaluations,
                True,
                "step tolerance met",
            )
        x, value, g = candidate, candidate_value, new_g
    return OptimizationResult(
        x, value, norm(g), max_iterations, evaluations, False, "iteration limit reached"
    )


def minimize_nelder_mead(
    function: ScalarFunction,
    x0: Sequence[float],
    *,
    initial_step: float = 0.1,
    tolerance: float = 1e-10,
    max_evaluations: int = 20_000,
) -> OptimizationResult:
    """Minimize a function without derivatives by the Nelder-Mead simplex.

    Args:
        function: Scalar objective; non-finite values are treated as ``+inf``.
        x0: Finite starting point.
        initial_step: Relative edge length of the starting simplex.
        tolerance: Stop when the simplex value spread falls below this value.
        max_evaluations: Maximum objective evaluations.

    Returns:
        Minimization result; ``gradient_norm`` is ``nan`` (not computed).

    Equation:
        [EQ-OPT-NM-001].

    References:
        [REF-NELDER-MEAD-1965], [REF-NOCEDAL-WRIGHT-2006, Section 9.5].

    """
    start = as_vector(x0, name="x0")
    if not start:
        raise ValueError("x0 must be non-empty")
    evaluations = 0

    def f(point: Vector) -> float:
        """Evaluate the objective, mapping failures to ``+inf``.

        References:
            [REF-NELDER-MEAD-1965]. No equation is evaluated.

        """
        nonlocal evaluations
        evaluations += 1
        try:
            value = float(function(point))
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    size = len(start)
    simplex = [start]
    for index in range(size):
        vertex = list(start)
        vertex[index] += initial_step * max(1.0, abs(start[index]))
        simplex.append(tuple(vertex))
    values = [f(vertex) for vertex in simplex]
    iterations = 0
    while evaluations < max_evaluations:
        order = sorted(range(size + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) <= tolerance * max(1.0, abs(values[0])):
            spread = max(
                norm(tuple(a - b for a, b in zip(v, simplex[0], strict=True)))
                for v in simplex
            )
            if spread <= math.sqrt(tolerance) * max(1.0, norm(simplex[0])):
                return OptimizationResult(
                    simplex[0],
                    values[0],
                    math.nan,
                    iterations,
                    evaluations,
                    True,
                    "simplex tolerance met",
                )
        iterations += 1
        centroid = tuple(
            math.fsum(vertex[j] for vertex in simplex[:-1]) / size for j in range(size)
        )

        worst = simplex[-1]

        def along(
            coefficient: float,
            centroid: Vector = centroid,
            worst: Vector = worst,
        ) -> Vector:
            """Return ``centroid + coefficient (centroid - worst)``.

            References:
                [EQ-OPT-NM-001], [REF-NELDER-MEAD-1965].

            """
            return tuple(
                c + coefficient * (c - w) for c, w in zip(centroid, worst, strict=True)
            )

        reflected = along(1.0)
        reflected_value = f(reflected)
        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[0]:
            expanded = along(2.0)
            expanded_value = f(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        contracted = along(0.5 if reflected_value < values[-1] else -0.5)
        contracted_value = f(contracted)
        if contracted_value < min(reflected_value, values[-1]):
            simplex[-1], values[-1] = contracted, contracted_value
            continue
        best = simplex[0]
        simplex = [best] + [
            tuple(b + 0.5 * (v - b) for b, v in zip(best, vertex, strict=True))
            for vertex in simplex[1:]
        ]
        values = [values[0]] + [f(vertex) for vertex in simplex[1:]]
    best_index = min(range(size + 1), key=lambda i: values[i])
    return OptimizationResult(
        simplex[best_index],
        values[best_index],
        math.nan,
        iterations,
        evaluations,
        False,
        "evaluation limit reached",
    )


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


def brent_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> RootResult:
    """Find a root in a sign-changing bracket by Brent's method.

    Args:
        function: Continuous scalar function.
        lower: Bracket endpoint.
        upper: Bracket endpoint with ``f(lower) f(upper) <= 0``.
        tolerance: Absolute bracket-width tolerance.
        max_iterations: Maximum iterations.

    Returns:
        Root result.

    Raises:
        ValueError: If the bracket does not change sign.

    Equation:
        [EQ-OPT-BRENT-001].

    References:
        [REF-BRENT-1973, Chapter 4].

    """
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    fa = _finite_value(function(a), "f(lower)")
    fb = _finite_value(function(b), "f(upper)")
    evaluations = 2
    if fa * fb > 0.0:
        raise ValueError("root bracket must change sign")
    c, fc = b, fb
    d = e = b - a
    for iteration in range(max_iterations):
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tolerance_1 = 2.0 * 2.0**-52 * abs(b) + 0.5 * tolerance
        half_width = 0.5 * (c - b)
        if abs(half_width) <= tolerance_1 or fb == 0.0:
            return RootResult(b, fb, iteration, evaluations, True)
        if abs(e) >= tolerance_1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * half_width * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * half_width * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * half_width * q - abs(tolerance_1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = half_width
        else:
            d = e = half_width
        a, fa = b, fb
        b += d if abs(d) > tolerance_1 else math.copysign(tolerance_1, half_width)
        fb = _finite_value(function(b), "f(b)")
        evaluations += 1
    return RootResult(b, fb, max_iterations, evaluations, False)
