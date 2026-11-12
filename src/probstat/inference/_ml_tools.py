"""Shared maximum-likelihood machinery of the count and categorical regressions.

A damped Newton maximizer with a backtracking line search, the inverse
observed information as a covariance, Wald tables, and delta-method standard
errors of average marginal effects. The regression modules supply analytic
log-likelihoods and gradients; the Hessian is analytic where it is short
(ordinal and multinomial logit, negative binomial regression) and a central
difference of the analytic gradient otherwise (zero-inflated and hurdle
models).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.linalg import Matrix, Vector, as_vector, inverse_spd, solve_spd
from probstat.optimization import central_jacobian

from .linear_models import _validate_level
from .sampling_distributions import standard_normal_sf

Objective = Callable[[Vector], float]
GradientFunction = Callable[[Vector], Vector]
HessianFunction = Callable[[Vector], Matrix]

_ARMIJO = 1e-4
_MAX_HALVINGS = 60


@dataclass(frozen=True, slots=True)
class NewtonOutcome:
    """Result of :func:`newton_maximize`.

    Attributes:
        x: Final parameter vector.
        value: Objective (log-likelihood) at ``x``.
        gradient: Gradient at ``x``.
        hessian: Hessian at ``x`` (analytic or central differences).
        iterations: Newton iterations used.
        converged: Whether the stopping rule was met.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 3].

    """

    x: Vector
    value: float
    gradient: Vector
    hessian: Matrix
    iterations: int
    converged: bool


def numerical_hessian(gradient: GradientFunction, x: Vector) -> Matrix:
    """Return the symmetrized central-difference Jacobian of a gradient.

    Args:
        gradient: Analytic gradient of the log-likelihood.
        x: Evaluation point.

    Returns:
        ``(J + J') / 2`` with ``J`` the central-difference Jacobian.

    References:
        [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    jacobian, _ = central_jacobian(gradient, x)
    size = len(x)
    return tuple(
        tuple(0.5 * (jacobian[i][j] + jacobian[j][i]) for j in range(size))
        for i in range(size)
    )


def _damped_direction(gradient: Vector, hessian: Matrix) -> Vector:
    """Solve ``(-H + lambda I) d = g`` with the smallest working damping.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 3] (modified Newton with a
        multiple of the identity).

    """
    size = len(gradient)
    negative = tuple(tuple(-value for value in row) for row in hessian)
    scale = max((abs(negative[i][i]) for i in range(size)), default=1.0) or 1.0
    damping = 0.0
    for _ in range(40):
        system = tuple(
            tuple(negative[i][j] + (damping if i == j else 0.0) for j in range(size))
            for i in range(size)
        )
        try:
            return solve_spd(system, gradient)
        except (ValueError, ArithmeticError, ZeroDivisionError):
            damping = 1e-8 * scale if damping == 0.0 else 10.0 * damping
    return tuple(value / scale for value in gradient)


def newton_maximize(
    objective: Objective,
    gradient: GradientFunction,
    start: Sequence[float],
    *,
    hessian: HessianFunction | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> NewtonOutcome:
    """Maximize a smooth log-likelihood by damped Newton steps.

    Each step solves ``(-H + lambda I) d = g`` with ``lambda = 0`` when
    ``-H`` is positive definite and the smallest working multiple otherwise,
    then halves the step until the Armijo condition
    ``l(x + t d) >= l(x) + 1e-4 t g'd`` holds; an objective of ``-inf``
    marks an infeasible point. The run converges when the relative step
    ``max_j |t d_j| / (|x_j| + 0.1)`` falls below ``tolerance`` or the
    Newton decrement ``g'd`` is below ``1e-20 (1 + |l|)``.

    Args:
        objective: Log-likelihood; ``-inf`` outside the parameter space.
        gradient: Analytic gradient.
        start: Feasible starting point.
        hessian: Analytic Hessian; central differences of ``gradient`` when
            ``None``.
        max_iterations: Iteration limit.
        tolerance: Relative step tolerance.

    Returns:
        :class:`NewtonOutcome` with the Hessian at the final point.

    Raises:
        ValueError: If the objective is not finite at ``start``.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 3].

    """
    x = as_vector(start, name="start")
    value = objective(x)
    if not math.isfinite(value):
        raise ValueError("the log-likelihood is not finite at the starting point")
    hess = (
        hessian if hessian is not None else (lambda p: numerical_hessian(gradient, p))
    )
    converged = False
    iteration = 0
    for step_number in range(1, max_iterations + 1):
        iteration = step_number
        g = gradient(x)
        direction = _damped_direction(g, hess(x))
        decrement = math.fsum(a * b for a, b in zip(g, direction, strict=True))
        if decrement <= 1e-20 * (1.0 + abs(value)):
            converged = True
            break
        step = 1.0
        accepted = False
        for _ in range(_MAX_HALVINGS):
            trial = tuple(a + step * d for a, d in zip(x, direction, strict=True))
            trial_value = objective(trial)
            if math.isfinite(trial_value) and (
                trial_value >= value + _ARMIJO * step * decrement
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            converged = decrement <= 1e-12 * (1.0 + abs(value))
            break
        relative = max(
            abs(a - b) / (abs(b) + 0.1) for a, b in zip(trial, x, strict=True)
        )
        x, value = trial, trial_value
        if relative < tolerance:
            converged = True
            break
    return NewtonOutcome(
        x=x,
        value=value,
        gradient=gradient(x),
        hessian=hess(x),
        iterations=iteration,
        converged=converged,
    )


def inverse_information(hessian: Matrix) -> Matrix:
    """Return ``(-H)^-1``, the inverse observed information.

    Args:
        hessian: Hessian of the log-likelihood at the maximum.

    Returns:
        Asymptotic covariance of the maximum-likelihood estimate.

    Raises:
        ArithmeticError: If ``-H`` is not positive definite.

    References:
        [REF-CASELLA-BERGER-2002, Section 10.1].

    """
    negative = tuple(tuple(-value for value in row) for row in hessian)
    try:
        return inverse_spd(negative)
    except (ValueError, ArithmeticError, ZeroDivisionError) as error:
        raise ArithmeticError(
            "the observed information is not positive definite; the "
            "maximum-likelihood estimate is not identified"
        ) from error


def wald_table(estimates: Vector, covariance: Matrix) -> tuple[Vector, Vector, Vector]:
    """Return standard errors, Wald ratios, and two-sided normal p-values.

    Args:
        estimates: Parameter estimates.
        covariance: Their covariance.

    Returns:
        ``(se, z, p)`` with ``se_j = sqrt(V_jj)`` and ``p_j = 2 (1 - Phi(|z_j|))``.

    References:
        [EQ-GLM-WALD-001], [REF-CASELLA-BERGER-2002, Section 10.3].

    """
    errors = tuple(math.sqrt(max(covariance[j][j], 0.0)) for j in range(len(estimates)))
    ratios = tuple(
        b / s if s > 0.0 else math.copysign(math.inf, b)
        for b, s in zip(estimates, errors, strict=True)
    )
    p_values = tuple(
        2.0 * standard_normal_sf(abs(z)) if math.isfinite(z) else 0.0 for z in ratios
    )
    return errors, ratios, p_values


def normal_intervals(
    estimates: Vector, errors: Vector, level: float
) -> tuple[tuple[float, float], ...]:
    """Return Wald intervals ``b +- z_(1-alpha/2) se``.

    Args:
        estimates: Estimates.
        errors: Standard errors.
        level: Two-sided level in ``(0, 1)``.

    Returns:
        One ``(lower, upper)`` pair per estimate.

    Raises:
        ValueError: If ``level`` is invalid.

    References:
        [EQ-GLM-WALD-001], [REF-CASELLA-BERGER-2002, Section 10.4].

    """
    from statistics import NormalDist

    checked = _validate_level(level)
    quantile = NormalDist().inv_cdf(0.5 * (1.0 + checked))
    return tuple(
        (b - quantile * s, b + quantile * s)
        for b, s in zip(estimates, errors, strict=True)
    )


@dataclass(frozen=True, slots=True)
class MarginalEffects:
    """Average marginal effects of one design column with delta-method errors.

    Attributes:
        column: Design-column name.
        kind: ``"derivative"`` (mean of ``d E / d x_j``) or ``"discrete"``
            (mean of ``E(x_j = 1) - E(x_j = 0)``).
        outcomes: Outcome labels (categories, or ``("mean",)`` for counts).
        estimates: One average marginal effect per outcome.
        standard_errors: Delta-method standard errors ``sqrt(G V G')``.
        covariance: Delta-method covariance ``G V G'`` of the effects.

    References:
        [EQ-REG-AME-001], [REF-CASELLA-BERGER-2002, Section 5.5].

    """

    column: str
    kind: str
    outcomes: tuple[str, ...]
    estimates: Vector
    standard_errors: Vector
    covariance: Matrix

    def intervals(self, level: float = 0.95) -> tuple[tuple[float, float], ...]:
        """Return normal Wald intervals for every average marginal effect.

        Args:
            level: Two-sided level in ``(0, 1)``.

        Returns:
            One ``(lower, upper)`` pair per outcome.

        Raises:
            ValueError: If ``level`` is invalid.

        References:
            [EQ-REG-AME-001], [REF-CASELLA-BERGER-2002, Section 5.5].

        """
        return normal_intervals(self.estimates, self.standard_errors, level)


def delta_method_effects(
    effect: Callable[[Vector], Vector],
    parameters: Vector,
    covariance: Matrix,
    *,
    column: str,
    kind: str,
    outcomes: tuple[str, ...],
) -> MarginalEffects:
    """Return average marginal effects with delta-method covariance.

    Args:
        effect: Map from the parameter vector to the effects (analytic in the
            parameters); its Jacobian ``G`` is taken by central differences.
        parameters: Maximum-likelihood estimate.
        covariance: Its covariance ``V``.
        column: Column name.
        kind: ``"derivative"`` or ``"discrete"``.
        outcomes: Outcome labels.

    Returns:
        :class:`MarginalEffects` with covariance ``G V G'``.

    References:
        [EQ-REG-AME-001], [REF-CASELLA-BERGER-2002, Section 5.5].

    """
    estimates = effect(parameters)
    jacobian, _ = central_jacobian(effect, parameters)
    size = len(parameters)
    product = tuple(
        tuple(
            math.fsum(row[a] * covariance[a][b] for a in range(size))
            for b in range(size)
        )
        for row in jacobian
    )
    effect_covariance = tuple(
        tuple(math.fsum(left[b] * right[b] for b in range(size)) for right in jacobian)
        for left in product
    )
    errors = tuple(
        math.sqrt(max(effect_covariance[k][k], 0.0)) for k in range(len(estimates))
    )
    return MarginalEffects(
        column=column,
        kind=kind,
        outcomes=outcomes,
        estimates=estimates,
        standard_errors=errors,
        covariance=effect_covariance,
    )


def resolve_column(names: tuple[str, ...], column: str | int) -> int:
    """Return the index of a design column given by name or position.

    Args:
        names: Design-column names.
        column: Name or zero-based index.

    Returns:
        Column index.

    Raises:
        KeyError: If the name is unknown.
        IndexError: If the index is out of range.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(column, str):
        if column not in names:
            raise KeyError(f"unknown design column {column!r}")
        return names.index(column)
    if isinstance(column, bool) or not isinstance(column, int):
        raise TypeError("column must be a name or an integer index")
    if not 0 <= column < len(names):
        raise IndexError("column index out of range")
    return column


def set_column(rows: Matrix, index: int, value: float) -> Matrix:
    """Return a copy of the design with one column set to a constant.

    Args:
        rows: Design rows.
        index: Column index.
        value: New value.

    Returns:
        Modified rows.

    References:
        [EQ-REG-AME-001]. No other equation is evaluated.

    """
    return tuple(
        tuple(value if j == index else x for j, x in enumerate(row)) for row in rows
    )


def log1pexp(value: float) -> float:
    """Return ``log(1 + exp(value))`` without overflow.

    Args:
        value: Real number.

    Returns:
        Softplus of ``value``.

    References:
        [REF-HIGHAM-2002]. No named equation is needed.

    """
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def logistic(value: float) -> float:
    """Return ``1 / (1 + exp(-value))`` without overflow.

    Args:
        value: Real number (``+-inf`` allowed).

    Returns:
        Logistic distribution function at ``value``.

    References:
        [REF-AGRESTI-2013]. No named equation is needed.

    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
