"""Quadratic-sum (first-order Taylor) importance factors.

Linearizing a scalar model around a representative point ``x0``,
``g(X) ~= g(x0) + sum_i dg/dx_i(x0) (X_i - x0_i)``, gives for independent
inputs the first-order Taylor variance
``V_T = sum_i (dg/dx_i(x0))^2 sigma_i^2`` and the importance factors
``eta_i^2 = (dg/dx_i(x0))^2 sigma_i^2 / V_T`` ([EQ-SENS-TAYLOR-001]). They
sum to one and cost ``2 d`` model runs with central differences (none with
a supplied gradient).

The factors are local: they depend on the expansion point and on the
derivatives there, and they describe the output variance only when the
model is close to linear over the range in which the inputs actually vary.
Variance-based Sobol indices (:func:`~probstat.sensitivity.sobol_analysis`)
average over the whole input law and remain valid for non-linear models; a
large gap between ``eta_i^2`` and the Sobol first-order index ``S_i`` points
to curvature or interactions that the linearization misses.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.validation import require_finite_real
from probstat.modeling import ProbabilisticModel

from .analysis import central_difference_gradient

GradientArgument = (
    Sequence[float] | Callable[[tuple[float, ...]], Sequence[float]] | None
)
"""A gradient at the point, a callable returning it, or ``None`` for
central differences."""


@dataclass(frozen=True, slots=True)
class TaylorImportanceResult:
    """Quadratic-sum importance factors of a scalar output.

    Args:
        input_names: Input names in order.
        point: Expansion point ``x0``.
        standard_deviations: Input standard deviations ``sigma_i``.
        gradient: ``dg/dx_i`` at ``x0``.
        contributions: ``(dg/dx_i)^2 sigma_i^2`` per input.
        taylor_variance: First-order Taylor variance ``V_T``, the sum of the
            contributions.
        importance_factors: ``eta_i^2 = contribution_i / V_T``.
        gradient_method: ``"central-difference"`` or ``"supplied"``.
        steps: Central-difference steps (``None`` for a supplied gradient).
        evaluations: Model evaluations spent on the gradient.
        quantity: QoI name, or ``None`` for a plain function.

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-CASELLA-BERGER-2002, Section
        5.5.4].

    """

    input_names: tuple[str, ...]
    point: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    gradient: tuple[float, ...]
    contributions: tuple[float, ...]
    taylor_variance: float
    importance_factors: tuple[float, ...]
    gradient_method: str
    steps: tuple[float, ...] | None
    evaluations: int
    quantity: str | None = None

    @property
    def taylor_standard_deviation(self) -> float:
        """Return ``sqrt(V_T)``, the linearized output standard deviation.

        References:
            [EQ-SENS-TAYLOR-001], [REF-CASELLA-BERGER-2002, Section 5.5.4].

        """
        return math.sqrt(self.taylor_variance)

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> eta_i^2``.

        References:
            [EQ-SENS-TAYLOR-001], [REF-SALTELLI-2008, Chapter 1].

        """
        return MappingProxyType(
            dict(zip(self.input_names, self.importance_factors, strict=True))
        )

    def ranking(self) -> tuple[str, ...]:
        """Return input names in descending importance-factor order.

        References:
            [EQ-SENS-TAYLOR-001], [REF-SALTELLI-2008, Chapter 1].

        """
        order = sorted(
            range(len(self.importance_factors)),
            key=lambda i: -self.importance_factors[i],
        )
        return tuple(self.input_names[i] for i in order)


def _vector(values: Sequence[object], name: str) -> tuple[float, ...]:
    """Return a non-empty tuple of finite reals.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers")
    result = tuple(
        require_finite_real(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _relative_steps(point: Sequence[float], relative_step: float) -> tuple[float, ...]:
    """Return ``h_i = c |x_i|``, or ``c`` itself where ``x_i = 0``.

    References:
        [EQ-SENS-LOCAL-001], [REF-FORNBERG-1988].

    """
    return tuple(relative_step * abs(x) if x != 0.0 else relative_step for x in point)


def quadratic_sum_importance(
    function: Callable[[tuple[float, ...]], float],
    point: Sequence[float],
    standard_deviations: Sequence[float],
    *,
    gradient: GradientArgument = None,
    relative_step: float = 0.01,
    names: Sequence[str] | None = None,
) -> TaylorImportanceResult:
    """Return quadratic-sum importance factors of a scalar function.

    ``eta_i^2 = (dg/dx_i(x0))^2 sigma_i^2 / sum_j (dg/dx_j(x0))^2 sigma_j^2``
    for independent inputs. Without ``gradient`` the derivatives are central
    differences with step ``h_i = relative_step |x0_i|`` (``relative_step``
    itself where ``x0_i = 0``), so the default reproduces a 1% perturbation.

    Args:
        function: Deterministic scalar function of a tuple.
        point: Expansion point ``x0``.
        standard_deviations: Non-negative input standard deviations.
        gradient: Gradient values at ``x0``, a callable returning them, or
            ``None`` for central differences.
        relative_step: Positive relative perturbation.
        names: Input names (default ``x1, x2, ...``).

    Returns:
        Gradient, contributions, Taylor variance, and importance factors.

    Raises:
        TypeError: If ``function`` or ``gradient`` has an invalid type.
        ValueError: If dimensions disagree, a standard deviation is negative,
            the step is not positive, or the Taylor variance is zero.

    Equation:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-LOCAL-001].

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-CASELLA-BERGER-2002, Section
        5.5.4], [REF-FORNBERG-1988].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    x0 = _vector(point, "point")
    sigma = _vector(standard_deviations, "standard_deviations")
    if len(sigma) != len(x0):
        raise ValueError("standard_deviations must match the point dimension")
    if any(value < 0.0 for value in sigma):
        raise ValueError("standard_deviations must be non-negative")
    step = require_finite_real(relative_step, name="relative_step")
    if step <= 0.0:
        raise ValueError("relative_step must be positive")
    labels = (
        tuple(f"x{i + 1}" for i in range(len(x0))) if names is None else tuple(names)
    )
    if len(labels) != len(x0) or len(set(labels)) != len(labels):
        raise ValueError("names must be distinct and match the point dimension")
    steps: tuple[float, ...] | None
    if gradient is None:
        steps = _relative_steps(x0, step)
        local = central_difference_gradient(function, x0, steps)
        slope, evaluations, method = (
            local.gradient,
            local.evaluations,
            ("central-difference"),
        )
    else:
        values = gradient(x0) if callable(gradient) else gradient
        slope = _vector(values, "gradient")
        if len(slope) != len(x0):
            raise ValueError("gradient must match the point dimension")
        steps, evaluations, method = None, 0, "supplied"
    contributions = tuple((g * s) ** 2 for g, s in zip(slope, sigma, strict=True))
    variance = math.fsum(contributions)
    if variance <= 0.0:
        raise ValueError("the first-order Taylor variance is zero")
    return TaylorImportanceResult(
        labels,
        x0,
        sigma,
        tuple(slope),
        contributions,
        variance,
        tuple(c / variance for c in contributions),
        method,
        steps,
        evaluations,
    )


def taylor_importance_factors(
    model: ProbabilisticModel,
    quantity: str,
    *,
    point: Sequence[float] | Mapping[str, float] | None = None,
    standard_deviations: Sequence[float] | Mapping[str, float] | None = None,
    gradient: GradientArgument = None,
    relative_step: float = 0.01,
) -> TaylorImportanceResult:
    """Return quadratic-sum importance factors of a model QoI.

    Defaults: ``point`` is the vector of input means (``model.nominal()``)
    and ``standard_deviations`` the marginal standard deviations of the
    input law. Either may be given as a vector in model order or as a
    mapping by input name (for example a set of nominal values). The result
    is local to ``point``; compare with :func:`sobol_analysis` before
    trusting it on a non-linear model.

    Args:
        model: Probabilistic model with mutually independent inputs.
        quantity: QoI name.
        point: Expansion point (default: input means).
        standard_deviations: Input standard deviations (default: those of
            the input law).
        gradient: Gradient values at ``point``, a callable returning them, or
            ``None`` for central differences.
        relative_step: Positive relative perturbation of the central
            differences (``0.01`` is a 1% perturbation).

    Returns:
        Importance factors with the Taylor variance and the model
        evaluations actually spent (cache hits are not counted).

    Raises:
        KeyError: If the QoI or an input name of a mapping is unknown.
        TypeError: If an argument has an invalid type.
        ValueError: If the inputs are dependent, dimensions disagree, or the
            Taylor variance is zero.

    Equation:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-LOCAL-001].

    References:
        [REF-SALTELLI-2008, Chapter 1], [REF-CASELLA-BERGER-2002, Section
        5.5.4], [REF-FORNBERG-1988].

    """
    model.quantity(quantity)
    if not model.inputs.is_independent:
        raise ValueError(
            "quadratic-sum factors assume independent inputs; with dependence "
            "use multivariate.delta_method with the full covariance"
        )

    def ordered(
        values: Sequence[float] | Mapping[str, float] | None,
        default: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Return ``values`` in model order, or ``default`` when ``None``.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        if values is None:
            return default
        if isinstance(values, Mapping):
            return model.inputs.as_vector(values)
        return _vector(values, "values")

    x0 = ordered(point, tuple(model.nominal()))
    sigma = ordered(standard_deviations, tuple(model.inputs.standard_deviations()))
    before = model.evaluations
    result = quadratic_sum_importance(
        model.scalar_function(quantity),
        x0,
        sigma,
        gradient=gradient,
        relative_step=relative_step,
        names=model.inputs.names,
    )
    return TaylorImportanceResult(
        result.input_names,
        result.point,
        result.standard_deviations,
        result.gradient,
        result.contributions,
        result.taylor_variance,
        result.importance_factors,
        result.gradient_method,
        result.steps,
        model.evaluations - before,
        quantity,
    )
