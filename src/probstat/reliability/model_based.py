"""FORM, SORM, and inverse reliability on a shared :class:`ProbabilisticModel`.

A :class:`ModelLimitState` turns one quantity of interest of a probabilistic
model into a limit state ``g`` with the library-wide failure convention
``g <= 0``. Because the model owns its joint input law, the same limit state
is available in physical space ``x`` and in independent standard-normal space
``u = T(x)`` through the Nataf transformation of
:class:`~probstat.modeling.JointInput`. FORM, SORM, importance sampling,
subset simulation, system reliability, and inverse design all work on this one
object, so the joint law is never redefined per method.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    dot,
    matmul,
    norm,
    solve_upper,
    symmetric_eigen,
    transpose,
)
from probstat.modeling import (
    ProbabilisticModel,
    standard_normal_cdf,
    standard_normal_quantile,
)
from probstat.optimization import brent_root, central_gradient, central_hessian

FailureSide = Literal["above", "below"]
GradientFunction = Callable[[Vector], Sequence[float]]

_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)
_MAX_LINE_SEARCH = 30
_ARMIJO = 1e-4


@dataclass(frozen=True, slots=True)
class ModelLimitState:
    """A limit state built from one quantity of interest of a model.

    With output ``y = model(x)[quantity]`` and threshold ``t``::

        failure="above":  g(x) = t - y   (failure when y >= t)
        failure="below":  g(x) = y - t   (failure when y <= t)

    so that ``g <= 0`` is failure in every case.

    Args:
        model: Probabilistic model supplying the joint input law and outputs.
        quantity: Name of the quantity of interest that defines failure.
        threshold: Finite admissible limit ``t`` of the quantity.
        failure: Which side of the threshold is failure.
        name: Optional label; a descriptive label is generated when empty.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 2 and 3].

    """

    model: ProbabilisticModel
    quantity: str
    threshold: float
    failure: FailureSide = "above"
    name: str = ""

    def __post_init__(self) -> None:
        """Validate the model, quantity, threshold, and failure side.

        Raises:
            TypeError: If ``model`` is not a :class:`ProbabilisticModel` or the
                name is not text.
            KeyError: If ``quantity`` is not declared by the model.
            ValueError: If the threshold is not finite or ``failure`` is
                neither ``"above"`` nor ``"below"``.

        References:
            [EQ-REL-MLS-001], [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        if not isinstance(self.model, ProbabilisticModel):
            raise TypeError("model must be a ProbabilisticModel")
        self.model.quantity(self.quantity)
        threshold = require_finite_real(self.threshold, name="threshold")
        if self.failure not in ("above", "below"):
            raise ValueError("failure must be 'above' or 'below'")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        object.__setattr__(self, "threshold", threshold)

    @property
    def label(self) -> str:
        """Return the explicit name or a generated description of failure.

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3]. No equation is evaluated.

        """
        if self.name.strip():
            return self.name
        relation = ">=" if self.failure == "above" else "<="
        return f"{self.quantity} {relation} {self.threshold:g}"

    @property
    def dimension(self) -> int:
        """Return the number of random inputs of the underlying model.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.model.dimension

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return input names in model order.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.model.inputs.names

    def margin(self, output: float) -> float:
        """Convert a quantity value ``y`` into the limit-state value ``g``.

        Args:
            output: Finite value of the quantity of interest.

        Returns:
            ``t - y`` for ``failure="above"`` and ``y - t`` otherwise.

        Raises:
            ValueError: If ``output`` is not finite.

        Equation:
            [EQ-REL-MLS-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        value = require_finite_real(output, name="quantity value")
        if self.failure == "above":
            return self.threshold - value
        return value - self.threshold

    def output_from_margin(self, margin: float) -> float:
        """Invert :meth:`margin`: recover ``y`` from a limit-state value ``g``.

        Args:
            margin: Finite limit-state value.

        Returns:
            Quantity value ``y`` that produces ``margin``.

        Equation:
            [EQ-REL-MLS-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        if self.failure == "above":
            return self.threshold - margin
        return margin + self.threshold

    def margin_from_outputs(self, outputs: Mapping[str, float]) -> float:
        """Return ``g`` from an already evaluated model output mapping.

        Args:
            outputs: QoI-name mapping returned by ``model.evaluate``.

        Returns:
            Limit-state value for this limit state's quantity.

        Equation:
            [EQ-REL-MLS-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        return self.margin(outputs[self.quantity])

    def physical(self, x: Sequence[float]) -> float:
        """Evaluate ``g(x)`` at one physical input vector (one model call).

        Args:
            x: Physical input vector in model order.

        Returns:
            Limit-state value; ``g <= 0`` denotes failure.

        Equation:
            [EQ-REL-MLS-001], [EQ-REL-LIMIT-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        return self.margin_from_outputs(self.model.evaluate(x))

    def standard(self, u: Sequence[float]) -> float:
        """Evaluate ``G(u) = g(T^-1(u))`` in standard-normal space.

        Args:
            u: Independent standard-normal coordinates in model order.

        Returns:
            Limit-state value at the mapped physical point.

        Equation:
            [EQ-REL-MLS-001], [EQ-MOD-NATAF-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        return self.physical(self.model.inputs.from_standard_normal(u))

    def failed(self, x: Sequence[float]) -> bool:
        """Return whether a physical input lies in the closed failure domain.

        Args:
            x: Physical input vector.

        Returns:
            ``True`` exactly when ``g(x) <= 0``.

        Equation:
            [EQ-REL-LIMIT-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        return self.physical(x) <= 0.0

    def with_threshold(self, threshold: float) -> ModelLimitState:
        """Return the same limit state with a different threshold.

        Args:
            threshold: New finite threshold.

        Returns:
            New limit state sharing the model and its evaluation counter.

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3]. No equation is evaluated.

        """
        return replace(self, threshold=threshold)


@dataclass(frozen=True, slots=True)
class FORMResult:
    """First-order reliability result at a standard-normal design point.

    Args:
        limit_state: Label of the analyzed limit state.
        beta: Signed reliability index ``alpha . u*`` (negative when the
            standard-normal origin already lies in the failure domain).
        probability: First-order failure probability ``Phi(-beta)``.
        design_point_u: Design point ``u*`` in standard-normal space.
        design_point_x: Design point ``x* = T^-1(u*)`` in physical space.
        alpha: Unit vector ``-grad G / ||grad G||`` at ``u*``.
        importance_factors: ``alpha_i^2`` keyed by input name. Under
            correlation ``u_i`` is the i-th Cholesky coordinate, so these
            factors depend on input order; see ``gamma``.
        gamma: Der Kiureghian's correlated importance vector
            ``L^-T alpha / ||L^-T alpha||`` keyed by input name (equal to
            ``alpha`` for independent inputs).
        gradient_u: Gradient of ``G`` at ``u*``.
        limit_state_value: ``G(u*)``; near zero for a converged search.
        iterations: Completed HL-RF updates.
        evaluations: Model evaluations charged to the search.
        gradient_evaluations: Calls of a user-supplied analytic gradient.
        converged: Whether both convergence criteria were met.
        message: Termination reason.

    References:
        [REF-HASOFER-LIND-1974], [REF-RACKWITZ-FIESSLER-1978],
        [REF-DERKIUREGHIAN-2005].

    """

    limit_state: str
    beta: float
    probability: float
    design_point_u: Vector
    design_point_x: Vector
    alpha: Vector
    importance_factors: Mapping[str, float]
    gamma: Mapping[str, float]
    gradient_u: Vector
    limit_state_value: float
    iterations: int
    evaluations: int
    gradient_evaluations: int
    converged: bool
    message: str


def _require_limit_state(limit_state: object) -> ModelLimitState:
    """Return the argument when it is a :class:`ModelLimitState`.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    return limit_state


def _require_positive_int(value: object, name: str) -> int:
    """Validate a strictly positive integer option.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_positive(value: object, name: str) -> float:
    """Validate a strictly positive finite real option.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _gradient_at(
    limit_state: ModelLimitState,
    point: Vector,
    gradient: GradientFunction | None,
    relative_step: float,
) -> Vector:
    """Return ``grad G(point)`` from an analytic callable or central differences.

    References:
        [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    if gradient is None:
        values, _ = central_gradient(
            limit_state.standard, point, relative_step=relative_step
        )
        return values
    values = as_vector(gradient(point), name="gradient")
    if len(values) != len(point):
        raise ValueError("gradient length must equal the input dimension")
    return values


def correlated_importance(limit_state: ModelLimitState, alpha: Vector) -> Vector:
    """Return Der Kiureghian's importance vector for correlated Nataf inputs.

    For the Nataf map ``u = L^-1 z`` the equivalent-normal importance vector
    reduces to ``gamma = L^-T alpha / ||L^-T alpha||``; it equals ``alpha`` for
    independent inputs.

    Args:
        limit_state: Limit state whose joint input law defines ``L``.
        alpha: Unit FORM direction in standard-normal space.

    Returns:
        Signed unit importance vector in input order.

    Raises:
        ValueError: If ``alpha`` has the wrong length.

    Equation:
        [EQ-REL-FORM-002].

    References:
        [REF-DERKIUREGHIAN-2005].

    """
    vector = as_vector(alpha, name="alpha")
    if len(vector) != limit_state.dimension:
        raise ValueError("alpha length must equal the input dimension")
    correlation = limit_state.model.inputs.correlation
    if correlation is None:
        return vector
    lower = cholesky(correlation)
    raw = solve_upper(transpose(lower), vector)
    size = norm(raw)
    return tuple(value / size for value in raw)


HLRFState = tuple[Vector, float, Vector, int, bool, str]


def _hlrf_iterate(
    function: Callable[[Vector], float],
    start: Vector,
    gradient_of: Callable[[Vector], Vector],
    line_search: bool,
    iterations_limit: int,
    tol: float,
    scale: float | None = None,
) -> HLRFState:
    """Run (improved) HL-RF iterations on any standard-space function.

    This is the search loop of :func:`form_hlrf`, shared with the deflated
    (bulged) searches of the multiple-design-point module.

    Args:
        function: Limit-state function ``G(u)``.
        start: Starting point.
        gradient_of: Gradient callable of ``G``.
        line_search: Whether to use the iHL-RF merit line search.
        iterations_limit: Maximum number of updates.
        tol: Relative feasibility and direction tolerance.
        scale: Reference value for the feasibility test ``|G| <= tol scale``;
            ``|G(start)|`` (or one when that is zero) when omitted.

    Returns:
        ``(u, G(u), grad G(u), iterations, converged, message)``.

    Raises:
        ValueError: If the gradient vanishes.

    References:
        [EQ-REL-HLRF-001], [EQ-REL-HLRF-002], [REF-RACKWITZ-FIESSLER-1978],
        [REF-ZHANG-DERKIUREGHIAN-1995].

    """
    u = start
    value = function(u)
    grad = gradient_of(u)
    if scale is None:
        scale = abs(value) if value != 0.0 else 1.0
    iterations = 0
    converged = False
    message = "maximum iterations reached before convergence"
    while True:
        grad_norm = norm(grad)
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise ValueError(
                f"limit-state gradient vanished at u={u}; FORM direction undefined"
            )
        alpha = tuple(-component / grad_norm for component in grad)
        projection = dot(alpha, u)
        residual = norm(
            tuple(ui - projection * ai for ui, ai in zip(u, alpha, strict=True))
        )
        if abs(value) <= tol * scale and residual <= tol * max(1.0, norm(u)):
            converged = True
            message = "converged"
            break
        if iterations >= iterations_limit:
            break
        factor = (dot(grad, u) - value) / (grad_norm * grad_norm)
        target = tuple(factor * component for component in grad)
        direction = tuple(t - ui for t, ui in zip(target, u, strict=True))
        if line_search:
            penalty = 2.0 * max(norm(u), norm(target)) / grad_norm
            merit = 0.5 * dot(u, u) + penalty * abs(value)
            slope = dot(u, direction) - penalty * abs(value)
            step = 1.0
            trial: Vector = target
            trial_value = 0.0
            for _ in range(_MAX_LINE_SEARCH):
                trial = tuple(
                    ui + step * di for ui, di in zip(u, direction, strict=True)
                )
                trial_value = function(trial)
                trial_merit = 0.5 * dot(trial, trial) + penalty * abs(trial_value)
                if trial_merit <= merit + _ARMIJO * step * slope:
                    break
                step *= 0.5
            u, value = trial, trial_value
        else:
            u = target
            value = function(u)
        grad = gradient_of(u)
        iterations += 1
    return u, value, grad, iterations, converged, message


def form_hlrf(
    limit_state: ModelLimitState,
    *,
    start: Sequence[float] | None = None,
    gradient: GradientFunction | None = None,
    line_search: bool = True,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    relative_step: float = 1e-6,
) -> FORMResult:
    """Find the FORM design point by the (improved) HL-RF algorithm.

    Each iteration linearizes ``G`` at ``u_k`` and moves toward the HL-RF
    point ``u_hlrf = [(grad G . u_k - G) / ||grad G||^2] grad G``. With
    ``line_search=True`` the step length is chosen by Armijo backtracking on
    the iHL-RF merit function ``m(u) = ||u||^2 / 2 + c |G(u)|`` with
    ``c = 2 max(||u_k||, ||u_hlrf||) / ||grad G||``, which makes the HL-RF
    direction a descent direction and prevents the classical oscillation of
    plain HL-RF on strongly nonlinear limit states.

    Convergence requires both ``|G(u)| <= tolerance |G(u_0)|`` and
    ``||u - (alpha . u) alpha|| <= tolerance max(1, ||u||)``. A run that hits
    ``max_iterations`` returns ``converged=False`` instead of raising, so the
    caller sees the last iterate and decides.

    Args:
        limit_state: Model limit state with the ``g <= 0`` failure convention.
        start: Optional starting point in standard-normal space (default the
            origin, i.e. the median input).
        gradient: Optional analytic gradient of ``G`` in standard-normal space;
            central differences are used when omitted.
        line_search: Whether to use the iHL-RF merit line search.
        max_iterations: Maximum number of HL-RF updates.
        tolerance: Relative feasibility and direction tolerance.
        relative_step: Central-difference step relative to ``max(1, |u_i|)``.

    Returns:
        Signed index, ``Phi(-beta)``, design point in ``u`` and ``x`` space,
        ``alpha``, importance factors, iteration and evaluation counts, and the
        convergence flag.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If options are invalid, ``start`` has the wrong length, or
            the gradient vanishes so that no search direction exists.

    Equation:
        [EQ-REL-HLRF-001], [EQ-REL-HLRF-002], [EQ-REL-FORM-002].

    References:
        [REF-HASOFER-LIND-1974], [REF-RACKWITZ-FIESSLER-1978],
        [REF-ZHANG-DERKIUREGHIAN-1995], [REF-DERKIUREGHIAN-2005].

    """
    state = _require_limit_state(limit_state)
    iterations_limit = _require_positive_int(max_iterations, "max_iterations")
    tol = _require_positive(tolerance, "tolerance")
    step_size = _require_positive(relative_step, "relative_step")
    dimension = state.dimension
    if start is None:
        u: Vector = tuple(0.0 for _ in range(dimension))
    else:
        u = as_vector(start, name="start")
        if len(u) != dimension:
            raise ValueError("start length must equal the input dimension")
    before = state.model.evaluations
    gradient_calls = 0

    def gradient_of(point: Vector) -> Vector:
        """Evaluate the gradient and count analytic-gradient calls.

        References:
            [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        nonlocal gradient_calls
        if gradient is not None:
            gradient_calls += 1
        return _gradient_at(state, point, gradient, step_size)

    search = _hlrf_iterate(
        state.standard, u, gradient_of, line_search, iterations_limit, tol
    )
    return _form_result(
        state, search, state.model.evaluations - before, gradient_calls
    )


def _form_result(
    state: ModelLimitState,
    search: HLRFState,
    evaluations: int,
    gradient_evaluations: int,
) -> FORMResult:
    """Assemble a :class:`FORMResult` from a finished HL-RF search.

    References:
        [EQ-REL-FORM-002], [REF-DERKIUREGHIAN-2005].

    """
    u, value, grad, iterations, converged, message = search
    grad_norm = norm(grad)
    alpha = tuple(-component / grad_norm for component in grad)
    beta = dot(alpha, u)
    names = state.input_names
    gamma = correlated_importance(state, alpha)
    return FORMResult(
        limit_state=state.label,
        beta=beta,
        probability=standard_normal_cdf(-beta),
        design_point_u=u,
        design_point_x=state.model.inputs.from_standard_normal(u),
        alpha=alpha,
        importance_factors=MappingProxyType(
            {name: a * a for name, a in zip(names, alpha, strict=True)}
        ),
        gamma=MappingProxyType(dict(zip(names, gamma, strict=True))),
        gradient_u=grad,
        limit_state_value=value,
        iterations=iterations,
        evaluations=evaluations,
        gradient_evaluations=gradient_evaluations,
        converged=converged,
        message=message,
    )


@dataclass(frozen=True, slots=True)
class SORMResult:
    """Second-order correction of a FORM result by principal curvatures.

    Args:
        form: Converged FORM result at the design point.
        curvatures: Principal curvatures ``kappa_i`` in ascending order;
            positive values bend the limit surface away from the origin.
        breitung_probability: Breitung's asymptotic estimate.
        hohenbichler_rackwitz_probability: Hohenbichler-Rackwitz estimate.
        hessian_evaluations: Model evaluations used by the Hessian.
        evaluations: Total model evaluations (FORM plus Hessian).

    References:
        [REF-BREITUNG-1984], [REF-HOHENBICHLER-RACKWITZ-1988],
        [REF-DERKIUREGHIAN-2005].

    """

    form: FORMResult
    curvatures: Vector
    breitung_probability: float
    hohenbichler_rackwitz_probability: float
    hessian_evaluations: int
    evaluations: int

    @property
    def breitung_beta(self) -> float:
        """Return the generalized index ``-Phi^-1`` of Breitung's estimate.

        Equation:
            [EQ-REL-BETA-001].

        References:
            [REF-DITLEVSEN-MADSEN-1996].

        """
        return generalized_reliability_index(self.breitung_probability)

    @property
    def hohenbichler_rackwitz_beta(self) -> float:
        """Return the generalized index of the Hohenbichler-Rackwitz estimate.

        Equation:
            [EQ-REL-BETA-001].

        References:
            [REF-DITLEVSEN-MADSEN-1996].

        """
        return generalized_reliability_index(self.hohenbichler_rackwitz_probability)


def generalized_reliability_index(probability: float) -> float:
    """Return the equivalent index ``beta = -Phi^-1(p)`` of a probability.

    Args:
        probability: Failure probability in ``[0, 1]``.

    Returns:
        Equivalent reliability index; ``inf`` for zero and ``-inf`` for one.

    Raises:
        ValueError: If ``probability`` lies outside ``[0, 1]``.

    Equation:
        [EQ-REL-BETA-001].

    References:
        [REF-DITLEVSEN-MADSEN-1996].

    """
    value = require_finite_real(probability, name="probability")
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if value == 0.0:
        return math.inf
    if value == 1.0:
        return -math.inf
    return -standard_normal_quantile(value)


def second_order_probabilities(
    beta: float, curvatures: Sequence[float]
) -> tuple[float, float]:
    """Return Breitung and Hohenbichler-Rackwitz SORM probabilities.

    ``P_B = Phi(-beta) prod_i (1 + beta kappa_i)^(-1/2)`` and
    ``P_HR = Phi(-beta) prod_i (1 + psi kappa_i)^(-1/2)`` with
    ``psi = phi(beta) / Phi(-beta)``.

    Args:
        beta: Positive FORM reliability index.
        curvatures: Principal curvatures at the design point.

    Returns:
        ``(P_B, P_HR)``.

    Raises:
        ValueError: If ``beta <= 0`` (the asymptotic corrections assume a safe
            origin), a curvature is not finite, or ``1 + beta kappa_i <= 0`` or
            ``1 + psi kappa_i <= 0`` for some ``i`` (the paraboloid would not
            enclose a finite neighborhood of the design point and the formula
            is invalid).

    Equation:
        [EQ-REL-SORM-002], [EQ-REL-SORM-003].

    References:
        [REF-BREITUNG-1984], [REF-HOHENBICHLER-RACKWITZ-1988].

    """
    index = require_finite_real(beta, name="beta")
    if index <= 0.0:
        raise ValueError("SORM corrections require a positive reliability index")
    kappas = tuple(
        require_finite_real(value, name=f"curvatures[{i}]")
        for i, value in enumerate(curvatures)
    )
    first_order = standard_normal_cdf(-index)
    if first_order <= 0.0:
        raise ValueError("beta is too large for a representable Phi(-beta)")
    psi = _INV_SQRT_TWO_PI * math.exp(-0.5 * index * index) / first_order
    breitung = first_order
    improved = first_order
    for i, kappa in enumerate(kappas):
        breitung_factor = 1.0 + index * kappa
        improved_factor = 1.0 + psi * kappa
        if breitung_factor <= 0.0 or improved_factor <= 0.0:
            raise ValueError(
                f"curvature {i} = {kappa:g} gives 1 + beta*kappa <= 0; "
                "the SORM paraboloid approximation is invalid"
            )
        breitung /= math.sqrt(breitung_factor)
        improved /= math.sqrt(improved_factor)
    return breitung, improved


def _rotation_with_last_row(alpha: Vector) -> Matrix:
    """Return an orthonormal matrix whose last row is ``alpha``.

    Rows are built by modified Gram-Schmidt from ``alpha`` and the unit
    vectors, skipping the unit vector most aligned with ``alpha``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 5.2].

    """
    size = len(alpha)
    pivot = max(range(size), key=lambda i: abs(alpha[i]))
    basis: list[Vector] = [alpha]
    for i in range(size):
        if i == pivot:
            continue
        vector = [1.0 if j == i else 0.0 for j in range(size)]
        for _ in range(2):
            for row in basis:
                projection = math.fsum(v * r for v, r in zip(vector, row, strict=True))
                vector = [v - projection * r for v, r in zip(vector, row, strict=True)]
        length = math.sqrt(math.fsum(v * v for v in vector))
        basis.append(tuple(v / length for v in vector))
    return tuple(basis[1:]) + (basis[0],)


def principal_curvatures(gradient: Vector, hessian: Matrix) -> Vector:
    """Return principal curvatures of ``G = 0`` from its gradient and Hessian.

    With ``alpha = -grad G / ||grad G||`` and an orthonormal rotation ``R``
    whose last row is ``alpha``, the curvatures are the eigenvalues of the
    leading ``(n-1) x (n-1)`` block of ``R H R' / ||grad G||``.

    Args:
        gradient: Non-zero gradient of ``G`` at the design point.
        hessian: Symmetric Hessian of ``G`` at the design point.

    Returns:
        Ascending curvatures (empty in one dimension).

    Raises:
        ValueError: If shapes disagree or the gradient is zero.

    Equation:
        [EQ-REL-SORM-001].

    References:
        [REF-BREITUNG-1984], [REF-DERKIUREGHIAN-2005].

    """
    grad = as_vector(gradient, name="gradient")
    size = len(grad)
    if len(hessian) != size or any(len(row) != size for row in hessian):
        raise ValueError("hessian must be square with the gradient's dimension")
    grad_norm = norm(grad)
    if grad_norm == 0.0:
        raise ValueError("gradient must be non-zero")
    if size == 1:
        return ()
    alpha = tuple(-component / grad_norm for component in grad)
    tangent = _rotation_with_last_row(alpha)[:-1]
    block = matmul(matmul(tangent, hessian), transpose(tangent))
    scaled = tuple(
        tuple(0.5 * (block[i][j] + block[j][i]) / grad_norm for j in range(size - 1))
        for i in range(size - 1)
    )
    values, _ = symmetric_eigen(scaled)
    return values


def sorm(
    limit_state: ModelLimitState,
    *,
    form_result: FORMResult | None = None,
    relative_step: float = 1e-4,
) -> SORMResult:
    """Correct FORM with curvature-based SORM formulas at the design point.

    The Hessian of ``G`` in standard-normal space is approximated by central
    differences (``2n^2 + 1`` model calls), rotated so that the last axis is
    ``alpha``, and diagonalized on the tangent block.

    Args:
        limit_state: Model limit state.
        form_result: Optional converged FORM result for the same limit state;
            FORM is run (and charged) when omitted.
        relative_step: Hessian step relative to ``max(1, |u_i|)``.

    Returns:
        Curvatures, Breitung and Hohenbichler-Rackwitz probabilities, and
        evaluation counts.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If FORM did not converge, ``beta <= 0``, or a curvature
            violates ``1 + beta kappa > 0``.

    Equation:
        [EQ-REL-SORM-001], [EQ-REL-SORM-002], [EQ-REL-SORM-003].

    References:
        [REF-BREITUNG-1984], [REF-HOHENBICHLER-RACKWITZ-1988],
        [REF-DERKIUREGHIAN-2005].

    """
    state = _require_limit_state(limit_state)
    step_size = _require_positive(relative_step, "relative_step")
    if form_result is None:
        form_result = form_hlrf(state)
    elif not isinstance(form_result, FORMResult):
        raise TypeError("form_result must be a FORMResult")
    if not form_result.converged:
        raise ValueError("SORM requires a converged FORM design point")
    before = state.model.evaluations
    hessian, _ = central_hessian(
        state.standard, form_result.design_point_u, relative_step=step_size
    )
    hessian_evaluations = state.model.evaluations - before
    curvatures = principal_curvatures(form_result.gradient_u, hessian)
    breitung, improved = second_order_probabilities(form_result.beta, curvatures)
    return SORMResult(
        form=form_result,
        curvatures=curvatures,
        breitung_probability=breitung,
        hohenbichler_rackwitz_probability=improved,
        hessian_evaluations=hessian_evaluations,
        evaluations=form_result.evaluations + hessian_evaluations,
    )


@dataclass(frozen=True, slots=True)
class InverseFORMResult:
    """Threshold that a model limit state must have to reach a target index.

    Args:
        limit_state: Label of the limit state (with its original threshold).
        target_beta: Requested reliability index.
        threshold: Threshold ``t`` whose FORM index equals ``target_beta``.
        design_point_u: Most-probable point on the sphere ``||u|| = beta_t``.
        design_point_x: Physical image of the design point.
        iterations: Fixed-point iterations.
        evaluations: Model evaluations.
        converged: Whether successive sphere points agreed within tolerance.

    References:
        [REF-DERKIUREGHIAN-ZHANG-LI-1994], [REF-TU-CHOI-PARK-1999].

    """

    limit_state: str
    target_beta: float
    threshold: float
    design_point_u: Vector
    design_point_x: Vector
    iterations: int
    evaluations: int
    converged: bool


def inverse_form(
    limit_state: ModelLimitState,
    target_beta: float,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    relative_step: float = 1e-6,
) -> InverseFORMResult:
    """Find the threshold whose first-order reliability index is ``target_beta``.

    The threshold enters ``g`` additively, so the design point for the unknown
    threshold minimizes ``G`` on the sphere ``||u|| = beta_t``. The fixed-point
    iteration ``u_{k+1} = beta_t alpha(u_k)`` with
    ``alpha = -grad G / ||grad G||`` (the performance-measure / advanced mean
    value iteration) locates it, and the threshold is the quantity value there.

    Args:
        limit_state: Model limit state; its current threshold only fixes the
            failure side and does not influence the answer.
        target_beta: Positive target reliability index.
        max_iterations: Maximum number of fixed-point updates.
        tolerance: Absolute tolerance on successive sphere points, relative to
            ``beta_t``.
        relative_step: Central-difference step relative to ``max(1, |u_i|)``.

    Returns:
        Threshold, design point, iterations, evaluations, and convergence flag.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If ``target_beta`` is not positive or a gradient vanishes.

    Equation:
        [EQ-REL-INV-001].

    References:
        [REF-DERKIUREGHIAN-ZHANG-LI-1994], [REF-TU-CHOI-PARK-1999].

    """
    state = _require_limit_state(limit_state)
    radius = _require_positive(target_beta, "target_beta")
    iterations_limit = _require_positive_int(max_iterations, "max_iterations")
    tol = _require_positive(tolerance, "tolerance")
    step_size = _require_positive(relative_step, "relative_step")
    before = state.model.evaluations
    u: Vector = tuple(0.0 for _ in range(state.dimension))
    converged = False
    iterations = 0
    while iterations < iterations_limit:
        grad = _gradient_at(state, u, None, step_size)
        grad_norm = norm(grad)
        if not math.isfinite(grad_norm) or grad_norm == 0.0:
            raise ValueError(f"limit-state gradient vanished at u={u}")
        updated = tuple(-radius * component / grad_norm for component in grad)
        change = norm(tuple(a - b for a, b in zip(updated, u, strict=True)))
        u = updated
        iterations += 1
        if change <= tol * radius:
            converged = True
            break
    threshold = state.output_from_margin(state.standard(u))
    return InverseFORMResult(
        limit_state=state.label,
        target_beta=radius,
        threshold=threshold,
        design_point_u=u,
        design_point_x=state.model.inputs.from_standard_normal(u),
        iterations=iterations,
        evaluations=state.model.evaluations - before,
        converged=converged,
    )


@dataclass(frozen=True, slots=True)
class DesignParameterResult:
    """Deterministic design parameter value that achieves a target index.

    Args:
        parameter: Name of the parameter in ``model.fixed``.
        value: Parameter value with ``beta(value) = target_beta``.
        target_beta: Requested reliability index.
        achieved_beta: FORM index at ``value``.
        form: FORM result at ``value``.
        root_iterations: Brent iterations.
        evaluations: Total model evaluations over every FORM run.
        converged: Whether Brent's bracket tolerance was met.

    References:
        [REF-TU-CHOI-PARK-1999], [REF-BRENT-1973, Chapter 4].

    """

    parameter: str
    value: float
    target_beta: float
    achieved_beta: float
    form: FORMResult
    root_iterations: int
    evaluations: int
    converged: bool


def design_for_target_beta(
    limit_state: ModelLimitState,
    parameter: str,
    target_beta: float,
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-6,
) -> DesignParameterResult:
    """Solve ``beta_FORM(theta) = beta_t`` for a fixed model parameter ``theta``.

    Each trial value replaces ``model.fixed[parameter]``, runs FORM, and Brent's
    method brackets the root of ``beta(theta) - beta_t`` on ``[lower, upper]``.

    Args:
        limit_state: Model limit state whose model declares ``parameter`` in
            ``model.fixed``.
        parameter: Name of the deterministic design parameter.
        target_beta: Finite target reliability index.
        lower: Bracket endpoint.
        upper: Bracket endpoint; ``beta - beta_t`` must change sign.
        tolerance: Absolute tolerance on the parameter.

    Returns:
        Parameter value, achieved index, FORM result, and evaluation count.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        KeyError: If ``parameter`` is not a fixed model parameter.
        ValueError: If the bracket does not change sign or FORM fails to
            converge at a trial value.

    Equation:
        [EQ-REL-RBDO-001], [EQ-OPT-BRENT-001].

    References:
        [REF-TU-CHOI-PARK-1999], [REF-BRENT-1973, Chapter 4].

    """
    state = _require_limit_state(limit_state)
    if parameter not in state.model.fixed:
        raise KeyError(f"{parameter!r} is not a fixed parameter of the model")
    goal = require_finite_real(target_beta, name="target_beta")
    a = require_finite_real(lower, name="lower")
    b = require_finite_real(upper, name="upper")
    tol = _require_positive(tolerance, "tolerance")
    before = state.model.evaluations
    cache: dict[float, FORMResult] = {}

    def form_at(value: float) -> FORMResult:
        """Run FORM with the parameter set to ``value`` (memoized).

        References:
            [EQ-REL-HLRF-001], [REF-TU-CHOI-PARK-1999].

        """
        if value not in cache:
            fixed = dict(state.model.fixed)
            fixed[parameter] = value
            trial = replace(state, model=replace(state.model, fixed=fixed))
            result = form_hlrf(trial)
            if not result.converged:
                raise ValueError(f"FORM did not converge at {parameter}={value}")
            cache[value] = result
        return cache[value]

    def residual(value: float) -> float:
        """Return ``beta(value) - beta_t``.

        References:
            [EQ-REL-RBDO-001], [REF-TU-CHOI-PARK-1999].

        """
        return form_at(value).beta - goal

    try:
        root = brent_root(residual, a, b, tolerance=tol)
    except ValueError as error:
        if "change sign" in str(error):
            raise ValueError(
                f"beta - target does not change sign on [{a}, {b}] for {parameter!r}"
            ) from error
        raise
    final = form_at(root.root)
    return DesignParameterResult(
        parameter=parameter,
        value=root.root,
        target_beta=goal,
        achieved_beta=final.beta,
        form=final,
        root_iterations=root.iterations,
        evaluations=state.model.evaluations - before,
        converged=root.converged,
    )
