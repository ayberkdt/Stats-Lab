"""Derivative-based global sensitivity measures (DGSM) of a model QoI.

For a quantity ``f(X)`` with independent inputs, Sobol' and Kucherenko
(2009) define ``nu_i = E[(df/dx_i)^2]`` ([EQ-SENS-DGSM-001]). The Poincare
inequality of the input law ``mu_i``,
``Var(h(X_i)) <= C(mu_i) E[h'(X_i)^2]``, applied to the conditional
variance gives the upper bound on the total Sobol index

``S_Ti <= C(mu_i) nu_i / D``,  ``D = Var f(X)``     ([EQ-SENS-DGSM-002])

with the optimal Poincare constants ``C = (b - a)^2 / pi^2`` for
``Uniform(a, b)`` (so ``nu_i / (pi^2 D)`` on ``[0, 1]``, Sobol' and
Kucherenko 2009) and ``C = sigma^2`` for ``Normal(mu, sigma)`` (Lamboni,
Iooss, Popelin and Gamboa 2013). Other input laws get no bound here
(``None``) rather than a constant that might be wrong. The bound is tight for
``cos(pi x)`` on ``[0, 1]`` and for linear functions of normal inputs; a
small ``C nu_i / D`` certifies that an input is unimportant, while a large
value does not prove importance.

Gradients come from forward-mode automatic differentiation through the
forward model (one model call per sample point returns value and gradient)
when the model accepts :class:`~probstat.numerics.Dual` inputs, and from
central differences (``2 d + 1`` calls per point) otherwise; the result
records which.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from probstat.distributions import Normal, Uniform
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import ProbabilisticModel
from probstat.numerics import AutodiffUnsupportedError, Dual
from probstat.optimization import central_gradient
from probstat.propagation import DESIGNS, make_design


@dataclass(frozen=True, slots=True)
class DGSMResult:
    """Derivative-based sensitivity measures of one QoI.

    Args:
        quantity: QoI name.
        input_names: Input names in model order.
        nu: ``E[(df/dx_i)^2]`` estimates (physical units of ``f / x_i``,
            squared).
        mean_absolute_derivative: ``E|df/dx_i|`` estimates.
        output_variance: Sample variance ``D`` of ``f`` on the same points.
        poincare_constants: ``C(mu_i)`` of each input law (``None`` when the
            law has no constant implemented).
        total_index_upper_bounds: ``C(mu_i) nu_i / D`` (``None`` without a
            constant or for dependent inputs).
        nu_standard_errors: Monte Carlo standard errors of ``nu`` (only for
            ``design="mc"``, whose points are i.i.d.; ``None`` otherwise).
        sample_size: Number of sample points.
        evaluations: Model evaluations (one per point with autodiff,
            ``2 d + 1`` per point with finite differences).
        design: Unit-hypercube design used for the points.
        gradient_method: ``"autodiff"`` or ``"finite-difference"``.
        gradient_fallback_reason: Why autodiff was not used (``None`` if it
            was, or if finite differences were requested).

    References:
        [REF-SOBOL-KUCHERENKO-2009], [REF-LAMBONI-ETAL-2013].

    """

    quantity: str
    input_names: tuple[str, ...]
    nu: tuple[float, ...]
    mean_absolute_derivative: tuple[float, ...]
    output_variance: float
    poincare_constants: tuple[float | None, ...]
    total_index_upper_bounds: tuple[float | None, ...]
    nu_standard_errors: tuple[float, ...] | None
    sample_size: int
    evaluations: int
    design: str
    gradient_method: str
    gradient_fallback_reason: str | None = None

    def as_mapping(self) -> Mapping[str, tuple[float, float | None]]:
        """Return ``input -> (nu_i, upper bound on S_Ti)``.

        References:
            [EQ-SENS-DGSM-001], [EQ-SENS-DGSM-002].

        """
        return {
            name: (value, bound)
            for name, value, bound in zip(
                self.input_names, self.nu, self.total_index_upper_bounds, strict=True
            )
        }

    def negligible(self, threshold: float = 0.01) -> tuple[str, ...]:
        """Return inputs whose total-index upper bound is below ``threshold``.

        Inputs without a bound are never reported as negligible.

        Args:
            threshold: Screening threshold on ``S_Ti``.

        Returns:
            Names of inputs certified (up to Monte Carlo error) as negligible.

        Equation:
            [EQ-SENS-DGSM-002].

        References:
            [REF-SOBOL-KUCHERENKO-2009].

        """
        return tuple(
            name
            for name, bound in zip(
                self.input_names, self.total_index_upper_bounds, strict=True
            )
            if bound is not None and bound < threshold
        )


def poincare_constant(law: object) -> float | None:
    """Return the optimal Poincare constant of a supported input law.

    Args:
        law: Marginal distribution.

    Returns:
        ``(b - a)^2 / pi^2`` for :class:`~probstat.distributions.Uniform`,
        ``sigma^2`` for :class:`~probstat.distributions.Normal`, ``None``
        otherwise.

    Equation:
        [EQ-SENS-DGSM-002].

    References:
        [REF-SOBOL-KUCHERENKO-2009], [REF-LAMBONI-ETAL-2013].

    """
    if type(law) is Uniform:
        return (law.upper - law.lower) ** 2 / math.pi**2
    if type(law) is Normal:
        return law.sigma**2
    return None


def _dual_value_and_gradient(
    model: ProbabilisticModel, quantity: str, x: Sequence[float]
) -> tuple[float, tuple[float, ...]]:
    """Return ``f(x)`` and ``grad_x f`` from one dual-number model call.

    Raises:
        AutodiffUnsupportedError: If the model output is not a usable dual.
        TypeError: If the model rejects dual inputs.

    References:
        [EQ-AD-DUAL-001], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    size = len(x)
    values: dict[str, object] = {
        name: Dual(value, tuple(1.0 if j == i else 0.0 for j in range(size)))
        for i, (name, value) in enumerate(zip(model.inputs.names, x, strict=True))
    }
    values.update(model.fixed)
    outputs = model.model(values)  # type: ignore[arg-type]
    if not isinstance(outputs, Mapping) or quantity not in outputs:
        raise ValueError(f"model output is missing QoI {quantity!r}")
    output = outputs[quantity]
    if isinstance(output, Dual):
        if output.order != 1 or len(output.tangent) != size:
            raise AutodiffUnsupportedError("model returned a Dual of unexpected shape")
        value = float(output.value)
        gradient = tuple(float(t) for t in output.tangent)
    elif isinstance(output, int | float) and not isinstance(output, bool):
        value, gradient = float(output), (0.0,) * size
    else:
        raise TypeError("model output must be a number or a Dual")
    model.counter.add()
    if not math.isfinite(value) or not all(math.isfinite(g) for g in gradient):
        raise ValueError("model value or gradient is not finite")
    return value, gradient


def dgsm(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    design: str = "sobol",
    seed: int | RandomStream | None = None,
    gradient: str = "auto",
    relative_step: float = 1e-6,
) -> DGSMResult:
    """Estimate derivative-based global sensitivity measures of a QoI.

    ``size`` points ``x_k = F^-1(p_k)`` of a unit-hypercube design are
    mapped to the input law; at each point the value and the gradient with
    respect to the *physical* inputs are computed, and
    ``nu_i = mean_k (df/dx_i)(x_k)^2``, ``E|df/dx_i|`` and the sample
    variance ``D`` of ``f`` are formed. For independent inputs with a
    Poincare constant the bound ``S_Ti <= C nu_i / D`` is reported.

    Args:
        model: Probabilistic model (bounds need independent inputs).
        quantity: QoI name.
        size: Number of sample points (at least two).
        design: ``"sobol"``, ``"halton"``, ``"lhs"``, or ``"mc"``.
        seed: Root seed or stream (child ``"dgsm-design"``).
        gradient: ``"auto"`` (autodiff with finite-difference fallback),
            ``"autodiff"``, or ``"finite-difference"``.
        relative_step: Central-difference step relative to ``max(1, |x_i|)``;
            the perturbed points may leave the support of a bounded input, so
            the forward model must be defined there.

    Returns:
        DGSM estimates, upper bounds, and cost accounting.

    Raises:
        ValueError: For invalid settings or a QoI with zero sample variance.
        KeyError: For an unknown QoI.
        AutodiffUnsupportedError: With ``gradient="autodiff"`` for a model
            that does not accept dual inputs.

    Equation:
        [EQ-SENS-DGSM-001], [EQ-SENS-DGSM-002], [EQ-AD-DUAL-001],
        [EQ-OPT-FD-001].

    References:
        [REF-SOBOL-KUCHERENKO-2009], [REF-LAMBONI-ETAL-2013],
        [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """
    model.quantity(quantity)
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least two")
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    if gradient not in ("auto", "autodiff", "finite-difference"):
        raise ValueError("gradient must be 'auto', 'autodiff', or 'finite-difference'")
    if not (math.isfinite(relative_step) and relative_step > 0.0):
        raise ValueError("relative_step must be positive")
    dimension = model.dimension
    unit = make_design(
        design, size, dimension, seed=resolve_stream(seed).child("dgsm-design")
    )
    points = model.inputs.sample_design(unit)
    scalar: Callable[[Sequence[float]], float] = model.scalar_function(quantity)
    method = "finite-difference" if gradient == "finite-difference" else "autodiff"
    fallback: str | None = None
    before = model.evaluations
    values: list[float] = []
    squares: list[list[float]] = [[] for _ in range(dimension)]
    absolutes: list[list[float]] = [[] for _ in range(dimension)]
    for point in points:
        if method == "autodiff":
            try:
                value, grad = _dual_value_and_gradient(model, quantity, point)
            except TypeError as error:
                if gradient == "autodiff":
                    if isinstance(error, AutodiffUnsupportedError):
                        raise
                    raise AutodiffUnsupportedError(
                        f"model does not support Dual numbers: {error}"
                    ) from error
                method = "finite-difference"
                fallback = f"{type(error).__name__}: {error}"
        if method == "finite-difference":
            value = scalar(point)
            raw, _ = central_gradient(scalar, point, relative_step=relative_step)
            grad = tuple(raw)
        values.append(value)
        for i, g in enumerate(grad):
            squares[i].append(g * g)
            absolutes[i].append(abs(g))
    count = len(values)
    center = math.fsum(values) / count
    variance = math.fsum((v - center) ** 2 for v in values) / (count - 1)
    if not variance > 0.0:
        raise ValueError("DGSM bounds require a positive output variance")
    nu = tuple(math.fsum(column) / count for column in squares)
    mean_abs = tuple(math.fsum(column) / count for column in absolutes)
    constants = tuple(
        poincare_constant(item.distribution) for item in model.inputs.inputs
    )
    independent = model.inputs.is_independent
    bounds = tuple(
        None if constant is None or not independent else constant * value / variance
        for constant, value in zip(constants, nu, strict=True)
    )
    errors: tuple[float, ...] | None = None
    if design == "mc":
        errors = tuple(
            math.sqrt(
                math.fsum((s - mean) ** 2 for s in column) / (count - 1) / count
            )
            for column, mean in zip(squares, nu, strict=True)
        )
    return DGSMResult(
        quantity,
        model.inputs.names,
        nu,
        mean_abs,
        variance,
        constants,
        bounds,
        errors,
        count,
        model.evaluations - before,
        design,
        method,
        fallback,
    )
