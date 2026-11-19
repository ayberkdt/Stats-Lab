"""Reverse-mode traceable pieces of the calibration log-densities (ADR-0114).

The float log-densities of :class:`~probstat.inverse.CalibrationProblem`
validate with float-only helpers and call the float-only Nataf map of the
prior, so automatic differentiation cannot pass through them. The helpers
here evaluate the same quantities with operators and
:class:`~probstat.numerics.admath` so that a log-density built from them
accepts :class:`~probstat.numerics.ReverseVariable` coordinates:

* the prior map ``theta = T^-1(u)`` takes its value from the float map and
  its first derivatives from the inverse-function rule
  ``d x_i / d z_i = phi(z_i) / f_i(x_i)`` ([EQ-CAL-AD-001]); the quantile
  functions themselves are float-only (ADR-0090, Decision 4);
* the Cholesky factor and the forward substitution of a covariance whose
  entries are variables;
* the Gaussian and Student-t observation densities.

For float arguments every helper returns what the float code returns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from probstat.distributions._autodiff import admath, any_variable, primal, with_value
from probstat.linalg import Matrix, cholesky, inverse_spd, log_det_spd, matvec
from probstat.modeling import JointInput
from probstat.numerics import AutodiffUnsupportedError, Dual, ReverseVariable
from probstat.numerics.reverse_autodiff import NonDifferentiableError

_LOG_TWO_PI = math.log(2.0 * math.pi)


def require_first_order(values: Sequence[object], *, where: str) -> None:
    """Reject dual numbers and second-order tapes at a first-order-only step.

    The inverse-function derivative of the prior map is exact to first
    order only. A forward-over-reverse Hessian (tape values that are duals)
    or a forward-mode dual would silently get a wrong second derivative, so
    it is refused instead.

    Args:
        values: Coordinates entering the step.
        where: Step name for the message.

    Raises:
        AutodiffUnsupportedError: For a dual or a dual-valued variable.

    References:
        [EQ-CAL-AD-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    for value in values:
        if isinstance(value, Dual) or (
            isinstance(value, ReverseVariable) and not isinstance(value.value, float)
        ):
            raise AutodiffUnsupportedError(
                f"{where} supports first derivatives (reverse mode) only; "
                "second derivatives are taken by finite differences"
            )


def supports_prior_map(inputs: JointInput) -> str | None:
    """Return why the prior cannot be traced, or ``None`` if it can.

    References:
        [EQ-CAL-AD-001], [REF-LIU-DERKIUREGHIAN-1986].

    """
    if inputs.has_discrete:
        return (
            f"the prior has discrete inputs {list(inputs.discrete_names)}, "
            "whose standard-space map is a step function"
        )
    if inputs.uses_rosenblatt:
        return (
            f"the prior uses a {inputs.copula_family} copula (Rosenblatt map), "
            "which is not differentiated; independence and the Gaussian "
            "copula are supported"
        )
    return None


def prior_lower(inputs: JointInput) -> Matrix | None:
    """Return the Cholesky factor of the Gaussian-copula correlation, if any.

    References:
        [EQ-MOD-NATAF-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    if inputs.correlation is None:
        return None
    return cholesky(inputs.correlation)


def from_standard_normal(
    inputs: JointInput, lower: Matrix | None, u: Sequence[float]
) -> tuple[float, ...]:
    """Map standard-normal coordinates to physical parameters, traceably.

    The value is ``inputs.from_standard_normal(u)`` exactly. For variables
    ``x_i`` carries the derivative ``phi(z_i) / f_i(x_i) * dz_i/du`` with
    ``z = L u`` ([EQ-CAL-AD-001]).

    Args:
        inputs: Continuous prior with independence or a Gaussian copula.
        lower: Its correlation Cholesky factor (``None`` for independence).
        u: Coordinates (floats or first-order reverse-mode variables).

    Returns:
        Physical parameters.

    Raises:
        AutodiffUnsupportedError: For duals or second-order tapes.
        NonDifferentiableError: Where a marginal density is zero at the
            mapped point (the derivative is unbounded).

    Equation:
        [EQ-CAL-AD-001], [EQ-MOD-NATAF-001].

    References:
        [REF-LIU-DERKIUREGHIAN-1986], [REF-CASELLA-BERGER-2002, Section 2.1].

    """
    if not any_variable(*u):
        return inputs.from_standard_normal(u)
    require_first_order(u, where="the prior map T^-1(u)")
    point = tuple(primal(value) for value in u)
    physical = inputs.from_standard_normal(point)
    latent = point if lower is None else matvec(lower, point)
    mapped: list[float] = []
    for index, (item, x, z) in enumerate(
        zip(inputs.inputs, physical, latent, strict=True)
    ):
        density = item.log_pdf(x)
        slope = math.exp(-0.5 * z * z - 0.5 * _LOG_TWO_PI - density)
        if not math.isfinite(density) or not math.isfinite(slope):
            raise NonDifferentiableError(
                f"the prior map of {item.name!r} has an unbounded derivative at "
                f"u = {point[index]!r}"
            )
        if lower is None:
            expression = slope * u[index]
        else:
            expression = slope * admath.fsum(
                lower[index][k] * u[k] for k in range(index + 1)
            )
        mapped.append(with_value(x, expression))
    return tuple(mapped)


def log_prior(inputs: JointInput, theta: Sequence[float]) -> float:
    """Return the prior log-density ``log pi(theta)``, traceably.

    Independent marginals are summed through their own (differentiable)
    ``log_pdf``. A Gaussian copula adds ``-0.5 z'(R^-1 - I) z - 0.5 log det R``
    with ``z_i = Phi^-1(F_i(theta_i))`` carrying the first derivative
    ``f_i / phi(z_i)`` ([EQ-CAL-AD-001]).

    Args:
        inputs: Continuous prior with independence or a Gaussian copula.
        theta: Physical parameters (floats or reverse-mode variables).

    Returns:
        Log-density; ``-inf`` outside the support.

    Raises:
        AutodiffUnsupportedError: If a marginal law has no differentiable
            ``log_pdf``, or a Gaussian copula meets a second-order tape.

    Equation:
        [EQ-MOD-COPULA-001], [EQ-CAL-AD-001].

    References:
        [REF-NELSEN-2006, Section 2.3], [REF-LIU-DERKIUREGHIAN-1986].

    """
    terms: list[float] = []
    for item, value in zip(inputs.inputs, theta, strict=True):
        law: Any = item.distribution
        method = getattr(law, "log_pdf", None)
        if method is None:
            raise AutodiffUnsupportedError(
                f"the prior law of {item.name!r} has no log_pdf to differentiate"
            )
        try:
            term = method(value)
        except ValueError:
            return -math.inf
        if not math.isfinite(primal(term)):
            return -math.inf
        terms.append(term)
    marginal = admath.fsum(terms)
    if inputs.correlation is None:
        return marginal
    require_first_order(theta, where="the Gaussian-copula prior density")
    point = tuple(primal(value) for value in theta)
    try:
        latent = inputs.to_latent(point)
    except ValueError:
        return -math.inf
    scores: list[float] = []
    for item, x, z, value in zip(inputs.inputs, point, latent, theta, strict=True):
        slope = math.exp(item.log_pdf(x) + 0.5 * z * z + 0.5 * _LOG_TWO_PI)
        scores.append(with_value(z, slope * value))
    correlation = inputs.correlation
    precision = inverse_spd(correlation)
    size = len(scores)
    quadratic = admath.fsum(
        scores[i] * (precision[i][j] - (1.0 if i == j else 0.0)) * scores[j]
        for i in range(size)
        for j in range(size)
    )
    return marginal - 0.5 * quadratic - 0.5 * log_det_spd(correlation)


def cholesky_lower(matrix: Sequence[Sequence[float]]) -> list[list[float]] | None:
    """Return the Cholesky factor of a symmetric matrix of variables.

    Same recursion as :func:`probstat.linalg.cholesky` written with
    operators, ``admath.fsum`` and ``admath.sqrt``.

    Args:
        matrix: Symmetric matrix (floats or variables).

    Returns:
        Lower factor, or ``None`` if a pivot is not positive (the matrix is
        not numerically positive definite).

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(matrix)
    lower: list[list[float]] = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            remainder = matrix[row][column] - admath.fsum(
                lower[row][inner] * lower[column][inner] for inner in range(column)
            )
            if row == column:
                pivot = primal(remainder)
                if pivot <= 0.0 or not math.isfinite(pivot):
                    return None
                lower[row][column] = admath.sqrt(remainder)
            else:
                lower[row][column] = remainder / lower[column][column]
    return lower


def solve_lower(
    lower: Sequence[Sequence[float]], right_hand_side: Sequence[float]
) -> list[float]:
    """Solve ``L x = b`` by forward substitution with operators.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    solution: list[float] = []
    for row, target in enumerate(right_hand_side):
        residual = target - admath.fsum(
            lower[row][column] * solution[column] for column in range(row)
        )
        solution.append(residual / lower[row][row])
    return solution


def observation_log_density(
    residual: float,
    sigma: float,
    discrepancy_variance: float,
    degrees_of_freedom: float | None,
) -> float:
    """Return one Gaussian or Student-t observation log-density, traceably.

    Operation for operation the expression of
    :meth:`~probstat.inverse.NoiseModel.log_density`.

    Equation:
        [EQ-CAL-LIK-001].

    References:
        [REF-GELMAN-ETAL-2013, Section 17.2], [REF-LANGE-LITTLE-TAYLOR-1989].

    """
    total = sigma * sigma + discrepancy_variance
    if degrees_of_freedom is None:
        return -0.5 * (_LOG_TWO_PI + admath.log(total) + residual * residual / total)
    nu = degrees_of_freedom
    return (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * admath.log(nu * math.pi * total)
        - 0.5 * (nu + 1.0) * admath.log1p(residual * residual / (nu * total))
    )
