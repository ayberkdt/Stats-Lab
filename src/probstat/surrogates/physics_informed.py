"""Hybrid physics and machine-learning surrogates for one-dimensional problems.

Experimental (added in 0.28). Pure Python, tiny networks, scalar arithmetic:
the module is a reference implementation for problems with one independent
variable, tens of collocation points and tens of network parameters, not a
deep-learning framework. It is unit-agnostic: the caller's residual, right-hand
side and physics model fix the physical units (SI recommended); the test cases
shipped with the library are synthetic and say so.

Four methods share one training engine (an exact gradient from the tape of
:mod:`probstat.numerics.reverse_autodiff`, minimized by BFGS):

Physics-informed neural network (PINN)
    A network ``u(t; w)`` is trained on the residual of a differential
    equation at collocation points plus boundary/initial conditions plus an
    optional data misfit, with explicit loss weights [EQ-SUR-PIML-002]. The
    derivatives ``u'`` and ``u''`` with respect to the *input* are exact:
    they are propagated through the tanh layers by second-order Taylor
    (jet) recurrences [EQ-SUR-PIML-001], written with arithmetic operators and
    ``admath``, so the *weight* gradient of the whole loss is obtained by
    reverse-mode differentiation of the recorded jet program (reverse over
    forward-Taylor mode). Appending unknown physical parameters ``theta`` to
    the trainable vector gives the inverse PINN [EQ-SUR-PIML-003].

Hybrid (universal) differential equation
    ``y' = f_phys(t, y; theta) + E NN(y; w)`` with a known physics part and a
    small network added to selected components ``E``. The weights are trained
    by differentiating, in reverse mode, through the classical RK4
    discretization of :func:`probstat.numerics.integrate_fixed_step`, over
    one-step or multiple-shooting segments started at observed states
    [EQ-SUR-PIML-004]. The learned term can be distilled into a sparse
    polynomial by sequentially thresholded least squares [EQ-SUR-PIML-005].

Residual (discrepancy) learning
    ``y(x) ~ f_phys(x) + r(x)`` with ``r`` a Gaussian process or deep ensemble
    fitted to the training residuals, and split-conformal intervals from a
    separate calibration set [EQ-SUR-PIML-006, EQ-SUR-PIML-007]. The conformal
    layer (``probstat.conformal`` is layer 6, this package layer 5) is
    implemented locally; its guarantee is marginal for exchangeable data and
    is *not* claimed under extrapolation, where the comparison helper reports
    the measured coverage.

DeepONet
    A branch network on the sensor values of an input function and a trunk
    network on the query location, combined by an inner product
    [EQ-SUR-PIML-008], trained on generated pairs of functions and their
    images (the antiderivative operator has a closed form
    [EQ-SUR-PIML-009]).

Limits (stated rather than hidden): the PINN needs ``tanh`` hidden layers and
a scalar input and output; losses are non-convex, so BFGS may stop at a local
minimum, reported through ``converged`` and ``message``; the hybrid-ODE loss
starts every segment at an observed (possibly noisy) state, which is biased
for noisy data (an errors-in-variables effect), so use it with low observation
noise or longer segments.

References:
    [REF-RAISSI-PERDIKARIS-KARNIADAKIS-2019],
    [REF-RACKAUCKAS-ETAL-2020],
    [REF-CHEN-RUBANOVA-BETTENCOURT-DUVENAUD-2018],
    [REF-LU-JIN-PANG-ZHANG-KARNIADAKIS-2021],
    [REF-BRUNTON-PROCTOR-KUTZ-2016],
    [REF-KENNEDY-OHAGAN-2001],
    [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017],
    [REF-GRIEWANK-WALTHER-2008],
    [REF-NOCEDAL-WRIGHT-2006].

"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import householder_qr
from probstat.numerics import MLP, admath, integrate_fixed_step
from probstat.numerics import reverse_value_and_gradient as _reverse_value_and_gradient
from probstat.optimization import adam_minimize, minimize_bfgs

from .ensemble import DeepEnsemble, fit_deep_ensemble
from .gaussian_process import GaussianProcessRegression, fit_gaussian_process

__all__ = [
    "DeepONet",
    "DeepONetDataset",
    "DeepONetFit",
    "HybridComparison",
    "HybridODEResult",
    "ModelScore",
    "PINNCondition",
    "PINNLossWeights",
    "PINNResult",
    "ResidualHybridModel",
    "SparseDistillation",
    "antiderivative_dataset",
    "antiderivative_exact",
    "compare_physics_ml_hybrid",
    "deeponet_relative_error",
    "distill_sparse_regression",
    "fit_deeponet",
    "fit_hybrid_ode",
    "fit_pinn_1d",
    "fit_residual_hybrid",
    "network_jet",
    "split_conformal_halfwidth",
]

_RESIDUAL_LEARNERS = ("gp", "ensemble", "none")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer or raise.

    Args:
        value: Candidate integer (``bool`` is rejected).
        name: Argument name for the error message.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an ``int``.
        ValueError: If ``value`` is not positive.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str) -> float:
    """Return ``value`` as a finite float or raise.

    Args:
        value: Candidate real number (``bool`` is rejected).
        name: Argument name for the error message.

    Returns:
        The validated float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Return a finite float tuple or raise.

    Args:
        values: Sequence of real numbers.
        name: Argument name for the error message.

    Returns:
        The validated tuple.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If an entry is not finite.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    return tuple(
        _finite(value, f"{name}[{index}]") for index, value in enumerate(values)
    )


def _hidden_widths(hidden: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate hidden-layer widths.

    Args:
        hidden: Positive integer widths (may be empty).
        name: Argument name for the error message.

    Returns:
        Tuple of widths.

    Raises:
        TypeError: If an entry is not an ``int``.
        ValueError: If an entry is not positive.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    return tuple(_positive_int(width, f"{name}[{i}]") for i, width in enumerate(hidden))


class _CachedObjective:
    """Value and reverse-mode gradient of a tape program with one-point cache.

    BFGS asks for the value and the gradient at the same point in separate
    calls; the cache records the program once per point and counts the
    recordings, which is the training cost reported in the results.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """

    def __init__(self, program: Callable[[Sequence[Any]], Any]) -> None:
        """Store the program ``x -> scalar`` written with tape-safe operations.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        self._program = program
        self._point: tuple[float, ...] | None = None
        self._value = 0.0
        self._gradient: tuple[float, ...] = ()
        self.recordings = 0

    def _evaluate(self, x: Sequence[float]) -> None:
        """Record the program at ``x`` unless it is the cached point.

        Raises:
            ValueError: If the loss or its gradient is not finite.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        point = tuple(float(value) for value in x)
        if point == self._point:
            return
        value, gradient = _reverse_value_and_gradient(self._program, point)
        self.recordings += 1
        if not math.isfinite(value) or not all(math.isfinite(g) for g in gradient):
            raise ValueError("non-finite loss or gradient")
        self._point, self._value, self._gradient = point, float(value), gradient

    def value(self, x: Sequence[float]) -> float:
        """Return the loss at ``x``.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        self._evaluate(x)
        return self._value

    def gradient(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return the exact loss gradient at ``x``.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        self._evaluate(x)
        return self._gradient


@dataclass(frozen=True, slots=True)
class _Trained:
    """Outcome of one BFGS run on a cached tape objective.

    Attributes:
        x: Final parameter vector.
        value: Final loss.
        converged: BFGS convergence flag.
        message: BFGS termination reason.
        iterations: BFGS iterations.
        evaluations: Tape recordings (loss and gradient together).
        seconds: Wall-clock time (informational, not compared).

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6]. No equation is evaluated.

    """

    x: tuple[float, ...]
    value: float
    converged: bool
    message: str
    iterations: int
    evaluations: int
    seconds: float


def _train(
    program: Callable[[Sequence[Any]], Any],
    start: Sequence[float],
    *,
    max_iterations: int,
    gradient_tolerance: float,
) -> _Trained:
    """Minimize a tape program by BFGS with its exact reverse-mode gradient.

    Args:
        program: Scalar loss written with tape-safe operations.
        start: Initial parameters.
        max_iterations: BFGS iteration limit.
        gradient_tolerance: Absolute gradient-norm tolerance.

    Returns:
        :class:`_Trained` with the cost accounting.

    Raises:
        ValueError: If the loss is not finite at the start.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6], [REF-GRIEWANK-WALTHER-2008,
        Chapter 3]. No equation of its own.

    """
    objective = _CachedObjective(program)
    began = time.perf_counter()
    result = minimize_bfgs(
        objective.value,
        tuple(start),
        gradient=objective.gradient,
        gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations,
    )
    seconds = time.perf_counter() - began
    return _Trained(
        x=tuple(result.x),
        value=float(result.value),
        converged=bool(result.converged),
        message=str(result.message),
        iterations=int(result.iterations),
        evaluations=objective.recordings,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# 1. Taylor-mode network jets and the PINN
# ---------------------------------------------------------------------------


def _require_scalar_tanh(network: MLP) -> None:
    """Check that ``network`` supports input jets.

    Args:
        network: Candidate network.

    Raises:
        TypeError: If ``network`` is not an :class:`~probstat.numerics.MLP`.
        ValueError: If it is not scalar-in/scalar-out with ``tanh`` hidden
            layers and an identity output layer.

    References:
        [EQ-SUR-PIML-001].

    """
    if not isinstance(network, MLP):
        raise TypeError("network must be a probstat.numerics.MLP")
    if network.input_dimension != 1 or network.output_dimension != 1:
        raise ValueError("input jets need a scalar-input, scalar-output network")
    if network.activation != "tanh" or network.output_activation != "identity":
        raise ValueError("input jets need tanh hidden layers and an identity output")


def network_jet(
    network: MLP, parameters: Sequence[Any], t: Any
) -> tuple[Any, Any, Any]:
    """Return ``(u, du/dt, d2u/dt2)`` of a scalar tanh MLP exactly.

    Each layer maps the jet ``(z, z', z'')`` of the pre-activations affinely
    and a tanh unit with ``a = tanh z`` maps it to
    ``(a, (1 - a^2) z', (1 - a^2)(z'' - 2 a z'^2))``. The input jet is
    ``(t, 1, 0)``. The result is the exact derivative of the network function
    (no step size) up to rounding, at about three times the cost of a forward
    pass. All operations are arithmetic and ``admath``, so ``parameters`` and
    ``t`` may be floats or reverse-mode tape variables.

    Args:
        network: Scalar-input, scalar-output MLP with ``tanh`` hidden layers
            and identity output (:class:`probstat.numerics.MLP` layout).
        parameters: Flat parameter vector of length ``network.parameter_count``.
        t: Input location.

    Returns:
        The value and the first two input derivatives.

    Raises:
        TypeError: If ``network`` is not a supported MLP.
        ValueError: For a wrong-length parameter vector or an unsupported
            architecture.

    Equation:
        [EQ-SUR-PIML-001].

    References:
        [EQ-SUR-PIML-001], [REF-RAISSI-PERDIKARIS-KARNIADAKIS-2019],
        [REF-GRIEWANK-WALTHER-2008, Chapter 13].

    """
    _require_scalar_tanh(network)
    if len(parameters) != network.parameter_count:
        raise ValueError(
            f"expected {network.parameter_count} parameters, got {len(parameters)}"
        )
    z0: list[Any] = [t]
    z1: list[Any] = [1.0]
    z2: list[Any] = [0.0]
    offset = 0
    last = len(network.layer_sizes) - 2
    for layer, (fan_in, fan_out) in enumerate(
        zip(network.layer_sizes[:-1], network.layer_sizes[1:], strict=True)
    ):
        weights_end = offset + fan_in * fan_out
        n0: list[Any] = []
        n1: list[Any] = []
        n2: list[Any] = []
        for row in range(fan_out):
            start = offset + row * fan_in
            p0: Any = parameters[weights_end + row]
            p1: Any = 0.0
            p2: Any = 0.0
            for column in range(fan_in):
                weight = parameters[start + column]
                p0 = p0 + weight * z0[column]
                p1 = p1 + weight * z1[column]
                p2 = p2 + weight * z2[column]
            if layer < last:
                activation = admath.tanh(p0)
                slope = 1.0 - activation * activation
                n0.append(activation)
                n1.append(slope * p1)
                n2.append(slope * (p2 - 2.0 * activation * p1 * p1))
            else:
                n0.append(p0)
                n1.append(p1)
                n2.append(p2)
        z0, z1, z2 = n0, n1, n2
        offset = weights_end + fan_out
    return z0[0], z1[0], z2[0]


@dataclass(frozen=True, slots=True)
class PINNCondition:
    """A boundary or initial condition ``u^(order)(t) = value``.

    Attributes:
        t: Location of the condition.
        value: Prescribed value of the derivative of the given order.
        order: Derivative order, 0, 1 or 2.
        weight: Non-negative weight of this condition inside the condition
            term of the loss [EQ-SUR-PIML-002].

    References:
        [EQ-SUR-PIML-002], [REF-RAISSI-PERDIKARIS-KARNIADAKIS-2019].

    """

    t: float
    value: float
    order: int = 0
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PINNLossWeights:
    """Explicit weights of the three loss terms of a PINN.

    ``L = residual * L_res + condition * L_cond + data * L_data``
    [EQ-SUR-PIML-002]. The weights are not tuned silently: a term with weight
    zero is dropped, and the unweighted values of all three terms are
    reported in the result.

    Attributes:
        residual: Weight of the mean squared equation residual.
        condition: Weight of the summed squared condition misfits.
        data: Weight of the mean squared data misfit.

    References:
        [EQ-SUR-PIML-002], [REF-RAISSI-PERDIKARIS-KARNIADAKIS-2019].

    """

    residual: float = 1.0
    condition: float = 1.0
    data: float = 1.0

    def __post_init__(self) -> None:
        """Validate that the weights are finite and non-negative.

        Raises:
            ValueError: If a weight is negative or not finite.

        References:
            [REF-PROBSTAT-VALIDATION], [EQ-SUR-PIML-002].

        """
        for name in ("residual", "condition", "data"):
            value = _finite(getattr(self, name), f"weights.{name}")
            if value < 0.0:
                raise ValueError(f"weights.{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class PINNResult:
    """A trained one-dimensional PINN with its training cost.

    Attributes:
        network: Architecture ``(1, *hidden, 1)`` with tanh hidden layers.
        parameters: Trained network weights (flat, :class:`MLP` layout).
        physical: Estimated physical parameters ``theta`` (empty for a
            forward problem).
        loss: Final weighted loss.
        residual_loss: Final unweighted mean squared residual.
        condition_loss: Final unweighted summed squared condition misfit.
        data_loss: Final unweighted mean squared data misfit (0 without data).
        weights: The loss weights used.
        collocation: Collocation points of the residual term.
        converged: Whether BFGS met a stopping tolerance.
        message: BFGS termination reason.
        iterations: BFGS iterations.
        evaluations: Loss-and-gradient tape recordings (the training cost).
        seconds: Wall-clock training time (informational).

    References:
        [EQ-SUR-PIML-002], [EQ-SUR-PIML-003].

    """

    network: MLP
    parameters: tuple[float, ...]
    physical: tuple[float, ...]
    loss: float
    residual_loss: float
    condition_loss: float
    data_loss: float
    weights: PINNLossWeights
    collocation: tuple[float, ...]
    converged: bool
    message: str
    iterations: int
    evaluations: int
    seconds: float
    residual: Callable[..., Any]

    def predict(self, t: float) -> float:
        """Return the trained solution ``u(t)``.

        Args:
            t: Finite location.

        Returns:
            ``u(t; w)``.

        References:
            [EQ-SUR-PIML-001].

        """
        return float(network_jet(self.network, self.parameters, _finite(t, "t"))[0])

    def derivative(self, t: float, order: int = 1) -> float:
        """Return the exact input derivative of the trained network.

        Args:
            t: Finite location.
            order: 1 or 2.

        Returns:
            ``d^order u / dt^order`` at ``t``.

        Raises:
            ValueError: If ``order`` is not 1 or 2.

        References:
            [EQ-SUR-PIML-001].

        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        jet = network_jet(self.network, self.parameters, _finite(t, "t"))
        return float(jet[order])

    def residual_at(self, t: float) -> float:
        """Return the equation residual of the trained solution at ``t``.

        Args:
            t: Finite location.

        Returns:
            ``R(t, u, u', u''; theta)`` with the estimated ``theta``.

        References:
            [EQ-SUR-PIML-002].

        """
        point = _finite(t, "t")
        u, du, d2u = network_jet(self.network, self.parameters, point)
        return float(self.residual(point, u, du, d2u, self.physical))


def fit_pinn_1d(
    residual: Callable[[Any, Any, Any, Any, Sequence[Any]], Any],
    domain: tuple[float, float],
    *,
    conditions: Sequence[PINNCondition] = (),
    data_times: Sequence[float] = (),
    data_values: Sequence[float] = (),
    physical_initial: Sequence[float] = (),
    hidden: Sequence[int] = (10,),
    collocation_points: int = 21,
    weights: PINNLossWeights | None = None,
    max_iterations: int = 500,
    gradient_tolerance: float = 1e-9,
    seed: int | RandomStream = 0,
) -> PINNResult:
    """Train a PINN on a 1-D ODE or boundary-value problem.

    Minimizes ``L(w, theta) = a_r mean_i R(t_i)^2 + a_c sum_c omega_c
    (u^(k_c)(t_c) - v_c)^2 + a_d mean_j (u(t_j) - y_j)^2``
    [EQ-SUR-PIML-002] over the network weights ``w`` and, when
    ``physical_initial`` is given, over the unknown physical parameters
    ``theta`` too (inverse PINN, [EQ-SUR-PIML-003]), where
    ``R = residual(t, u, u', u'', theta)`` and the input derivatives are
    exact jets [EQ-SUR-PIML-001]. Collocation points are the uniform grid on
    ``domain`` (endpoints included). BFGS uses the exact reverse-mode
    gradient. Hard constraints are not imposed: conditions enter as penalties
    whose weights are explicit.

    Args:
        residual: ``R(t, u, du, d2u, theta)`` written with arithmetic
            operators and ``admath`` (so it can be recorded on the tape);
            ``theta`` is the tuple of physical parameters (empty if none).
            It must vanish on the solution, e.g. ``du + theta[0] * u`` for
            ``u' = -k u``.
        domain: ``(a, b)`` with ``a < b``.
        conditions: Boundary or initial conditions.
        data_times: Locations of observations (optional).
        data_values: Observed values of ``u`` at ``data_times``.
        physical_initial: Starting values of the unknown parameters ``theta``
            (empty for a forward problem).
        hidden: Hidden widths of the tanh MLP ``(1, *hidden, 1)``.
        collocation_points: Number of collocation points (at least 2).
        weights: :class:`PINNLossWeights` (default all ones).
        max_iterations: BFGS iteration limit.
        gradient_tolerance: Absolute BFGS gradient tolerance.
        seed: Seed or stream for the Glorot initialization.

    Returns:
        :class:`PINNResult` with the solution, parameters, loss terms and
        the training cost (``evaluations``, ``iterations``, ``seconds``).

    Raises:
        TypeError: If ``residual`` is not callable or an argument has the
            wrong type.
        ValueError: If the domain, conditions, data or sizes are invalid, the
            data lengths differ, the problem has neither equation, condition
            nor data weight, or the loss is not finite at the start.

    Equation:
        [EQ-SUR-PIML-001], [EQ-SUR-PIML-002], [EQ-SUR-PIML-003].

    References:
        [EQ-SUR-PIML-002], [REF-RAISSI-PERDIKARIS-KARNIADAKIS-2019],
        [REF-NOCEDAL-WRIGHT-2006, Chapter 6].

    """
    if not callable(residual):
        raise TypeError("residual must be callable")
    low, high = _finite(domain[0], "domain[0]"), _finite(domain[1], "domain[1]")
    if not low < high:
        raise ValueError("domain must satisfy a < b")
    points = _positive_int(collocation_points, "collocation_points")
    if points < 2:
        raise ValueError("collocation_points must be at least 2")
    conds = tuple(conditions)
    for index, condition in enumerate(conds):
        if not isinstance(condition, PINNCondition):
            raise TypeError(f"conditions[{index}] must be a PINNCondition")
        _finite(condition.t, "condition.t")
        _finite(condition.value, "condition.value")
        if condition.order not in (0, 1, 2):
            raise ValueError("condition order must be 0, 1 or 2")
        if _finite(condition.weight, "condition.weight") < 0.0:
            raise ValueError("condition weight must be non-negative")
    times = _finite_vector(data_times, "data_times")
    values = _finite_vector(data_values, "data_values")
    if len(times) != len(values):
        raise ValueError("data_times and data_values must have the same length")
    theta0 = _finite_vector(physical_initial, "physical_initial")
    loss_weights = weights if weights is not None else PINNLossWeights()
    if not isinstance(loss_weights, PINNLossWeights):
        raise TypeError("weights must be a PINNLossWeights")
    iterations = _positive_int(max_iterations, "max_iterations")
    tolerance = _finite(gradient_tolerance, "gradient_tolerance")
    if tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive")
    if not conds and not times and loss_weights.residual == 0.0:
        raise ValueError("the loss has no active term")

    network = MLP((1, *_hidden_widths(hidden, "hidden"), 1))
    grid = tuple(low + (high - low) * i / (points - 1) for i in range(points))
    count = network.parameter_count

    def terms(x: Sequence[Any]) -> tuple[Any, Any, Any]:
        """Return the unweighted residual, condition and data loss terms.

        References:
            [EQ-SUR-PIML-002].

        """
        weights_x = x[:count]
        theta = tuple(x[count:])
        residual_sum: Any = 0.0
        for t in grid:
            u, du, d2u = network_jet(network, weights_x, t)
            r = residual(t, u, du, d2u, theta)
            residual_sum = residual_sum + r * r
        condition_sum: Any = 0.0
        for condition in conds:
            jet = network_jet(network, weights_x, condition.t)
            miss = jet[condition.order] - condition.value
            condition_sum = condition_sum + condition.weight * miss * miss
        data_sum: Any = 0.0
        for t, y in zip(times, values, strict=True):
            miss = network_jet(network, weights_x, t)[0] - y
            data_sum = data_sum + miss * miss
        data_mean = data_sum / len(times) if times else data_sum
        return residual_sum / len(grid), condition_sum, data_mean

    def loss(x: Sequence[Any]) -> Any:
        """Return the weighted PINN loss [EQ-SUR-PIML-002].

        References:
            [EQ-SUR-PIML-002].

        """
        r, c, d = terms(x)
        return (
            loss_weights.residual * r
            + loss_weights.condition * c
            + loss_weights.data * d
        )

    start = (*network.initial_parameters(resolve_stream(seed)), *theta0)
    trained = _train(
        loss, start, max_iterations=iterations, gradient_tolerance=tolerance
    )
    r_end, c_end, d_end = terms(trained.x)
    return PINNResult(
        network=network,
        parameters=trained.x[:count],
        physical=trained.x[count:],
        loss=trained.value,
        residual_loss=float(r_end),
        condition_loss=float(c_end),
        data_loss=float(d_end),
        weights=loss_weights,
        collocation=grid,
        converged=trained.converged,
        message=trained.message,
        iterations=trained.iterations,
        evaluations=trained.evaluations,
        seconds=trained.seconds,
        residual=residual,
    )


# ---------------------------------------------------------------------------
# 2. Hybrid (universal) differential equation and symbolic distillation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HybridODEResult:
    """A trained hybrid ODE ``y' = f_phys + E NN(y; w)``.

    Attributes:
        network: Correction network ``(n_state, *hidden, n_correction)``.
        parameters: Trained network weights.
        physical: Trained physical parameters (empty if none were fitted).
        correction_components: State components receiving the network output.
        loss: Final regularized shooting loss.
        converged: Whether BFGS met a stopping tolerance.
        message: BFGS termination reason.
        iterations: BFGS iterations.
        evaluations: Loss-and-gradient recordings (training cost).
        seconds: Wall-clock training time (informational).
        physics: The known right-hand side ``f_phys(t, y, theta)``.

    References:
        [EQ-SUR-PIML-004], [REF-RACKAUCKAS-ETAL-2020].

    """

    network: MLP
    parameters: tuple[float, ...]
    physical: tuple[float, ...]
    correction_components: tuple[int, ...]
    loss: float
    converged: bool
    message: str
    iterations: int
    evaluations: int
    seconds: float
    physics: Callable[[Any, Sequence[Any], Sequence[Any]], Sequence[Any]]

    def correction(self, state: Sequence[float]) -> tuple[float, ...]:
        """Return the learned term ``NN(y; w)`` at a state.

        Args:
            state: Finite state vector of the network input width.

        Returns:
            The network outputs, one per corrected component.

        References:
            [EQ-SUR-PIML-004].

        """
        return tuple(
            float(v)
            for v in self.network.forward(
                self.parameters, _finite_vector(state, "state")
            )
        )

    def rhs(self, t: float, state: Sequence[float]) -> tuple[float, ...]:
        """Return the hybrid right-hand side ``f_phys + E NN`` at ``(t, y)``.

        Args:
            t: Finite time.
            state: Finite state.

        Returns:
            The state derivative.

        References:
            [EQ-SUR-PIML-004].

        """
        return tuple(
            float(v)
            for v in _hybrid_rhs(
                self.physics,
                self.network,
                self.correction_components,
                self.parameters,
                self.physical,
                t,
                _finite_vector(state, "state"),
            )
        )

    def simulate(
        self, initial_state: Sequence[float], *, step: float, steps: int
    ) -> tuple[tuple[float, ...], ...]:
        """Integrate the hybrid model with RK4 from ``t = 0``.

        Args:
            initial_state: Finite initial state.
            step: Positive step.
            steps: Number of steps.

        Returns:
            The ``steps + 1`` states on the grid ``t_n = n step``.

        References:
            [EQ-SUR-PIML-004], [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

        """
        trajectory = integrate_fixed_step(
            self.rhs,
            0.0,
            _finite_vector(initial_state, "initial_state"),
            step=step,
            steps=steps,
            method="rk4",
        )
        return trajectory.states


def _hybrid_rhs(
    physics: Callable[[Any, Sequence[Any], Sequence[Any]], Sequence[Any]],
    network: MLP,
    components: tuple[int, ...],
    weights: Sequence[Any],
    theta: Sequence[Any],
    t: Any,
    state: Sequence[Any],
) -> list[Any]:
    """Return ``f_phys(t, y, theta)`` plus the network on selected components.

    Args:
        physics: Known right-hand side.
        network: Correction network.
        components: Corrected state indices, one per network output.
        weights: Network weights.
        theta: Physical parameters.
        t: Time.
        state: State.

    Returns:
        State derivative list (floats or tape variables).

    Raises:
        ValueError: If ``physics`` returns a vector of the wrong length.

    References:
        [EQ-SUR-PIML-004].

    """
    slope = list(physics(t, state, theta))
    if len(slope) != len(state):
        raise ValueError("physics must return one derivative per state component")
    learned = network.forward(weights, state)
    for index, value in zip(components, learned, strict=True):
        slope[index] = slope[index] + value
    return slope


def fit_hybrid_ode(
    physics: Callable[[Any, Sequence[Any], Sequence[Any]], Sequence[Any]],
    times: Sequence[float],
    trajectories: Sequence[Sequence[Sequence[float]]],
    *,
    correction_components: Sequence[int],
    hidden: Sequence[int] = (6,),
    substeps: int = 1,
    segment_steps: int = 1,
    physical_initial: Sequence[float] = (),
    weight_decay: float = 0.0,
    output_scale: float = 0.1,
    max_iterations: int = 300,
    gradient_tolerance: float = 1e-9,
    seed: int | RandomStream = 0,
) -> HybridODEResult:
    """Fit a network correction to known ODE physics through RK4 shooting.

    The model is ``y' = f_phys(t, y; theta) + E NN(y; w)`` where ``E`` adds
    output ``j`` of the network to state component
    ``correction_components[j]``. Each trajectory is cut into non-overlapping
    segments of ``segment_steps`` observation intervals; a segment starts at
    the observed state, is integrated by ``substeps`` RK4 steps per interval
    ([EQ-NUM-ODE-001]) and the squared differences to the observed states at
    the interval ends are averaged
    ``L = mean ||y_hat - y_obs||^2 + lambda ||w||^2`` [EQ-SUR-PIML-004].
    Gradients are exact reverse-mode derivatives through every RK4 stage
    (discretize-then-differentiate; no sensitivity ODE is solved).

    Args:
        physics: ``f_phys(t, y, theta)`` returning one derivative per state
            component, written with arithmetic operators and ``admath``.
        times: Strictly increasing, uniformly spaced observation times
            (at least 2).
        trajectories: One or more trajectories; each holds one finite state
            vector per observation time.
        correction_components: Distinct state indices that receive the
            network output (the network has this many outputs).
        hidden: Hidden widths of the tanh network (input width = state size).
        substeps: RK4 steps per observation interval (at least 1).
        segment_steps: Observation intervals per shooting segment.
        physical_initial: Starting values of physical parameters that are
            fitted jointly with the weights (empty to keep none).
        weight_decay: Non-negative ``lambda`` on the squared network weights.
        output_scale: Scale of the initial last-layer weights (0 starts at
            the pure-physics model).
        max_iterations: BFGS iteration limit.
        gradient_tolerance: Absolute BFGS gradient tolerance.
        seed: Seed or stream for the initialization.

    Returns:
        :class:`HybridODEResult`.

    Raises:
        TypeError: If ``physics`` is not callable or an argument has the
            wrong type.
        ValueError: For non-uniform times, ragged or non-finite data,
            invalid component indices, or a non-finite initial loss.

    Equation:
        [EQ-SUR-PIML-004], [EQ-NUM-ODE-001].

    References:
        [EQ-SUR-PIML-004], [REF-RACKAUCKAS-ETAL-2020],
        [REF-CHEN-RUBANOVA-BETTENCOURT-DUVENAUD-2018],
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """
    if not callable(physics):
        raise TypeError("physics must be callable")
    grid = _finite_vector(times, "times")
    if len(grid) < 2:
        raise ValueError("times needs at least two entries")
    spacing = grid[1] - grid[0]
    if not spacing > 0.0 or any(
        abs((b - a) - spacing) > 1e-9 * spacing * len(grid)
        for a, b in zip(grid[:-1], grid[1:], strict=True)
    ):
        raise ValueError("times must be strictly increasing and uniformly spaced")
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    data = tuple(
        tuple(_finite_vector(state, "trajectory state") for state in trajectory)
        for trajectory in trajectories
    )
    size = len(data[0][0]) if data[0] else 0
    for trajectory in data:
        if len(trajectory) != len(grid):
            raise ValueError("each trajectory needs one state per observation time")
        if any(len(state) != size for state in trajectory):
            raise ValueError("all states must have the same length")
    if size < 1:
        raise ValueError("states must not be empty")
    components = tuple(correction_components)
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in components):
        raise ValueError("correction_components must be non-negative integers")
    if not components or len(set(components)) != len(components):
        raise ValueError("correction_components must be non-empty and distinct")
    if any(c >= size for c in components):
        raise ValueError("correction component index out of range")
    inner = _positive_int(substeps, "substeps")
    seg = _positive_int(segment_steps, "segment_steps")
    decay = _finite(weight_decay, "weight_decay")
    if decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    scale = _finite(output_scale, "output_scale")
    if scale < 0.0:
        raise ValueError("output_scale must be non-negative")
    iterations = _positive_int(max_iterations, "max_iterations")
    tolerance = _finite(gradient_tolerance, "gradient_tolerance")
    if tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive")
    theta0 = _finite_vector(physical_initial, "physical_initial")

    network = MLP((size, *_hidden_widths(hidden, "hidden"), len(components)))
    count = network.parameter_count
    step = spacing / inner
    intervals = len(grid) - 1

    def loss(x: Sequence[Any]) -> Any:
        """Return the RK4 multiple-shooting loss [EQ-SUR-PIML-004].

        References:
            [EQ-SUR-PIML-004], [EQ-NUM-ODE-001].

        """
        weights_x = x[:count]
        theta = tuple(x[count:])

        def rhs(t: Any, y: Sequence[Any]) -> list[Any]:
            """Return the hybrid right-hand side on the tape.

            References:
                [EQ-SUR-PIML-004].

            """
            return _hybrid_rhs(physics, network, components, weights_x, theta, t, y)

        total: Any = 0.0
        compared = 0
        for trajectory in data:
            for first in range(0, intervals, seg):
                length = min(seg, intervals - first)
                path = integrate_fixed_step(
                    rhs,
                    grid[first],
                    trajectory[first],
                    step=step,
                    steps=length * inner,
                    method="rk4",
                )
                for k in range(1, length + 1):
                    predicted = path.states[k * inner]
                    observed = trajectory[first + k]
                    for a, b in zip(predicted, observed, strict=True):
                        total = total + (a - b) * (a - b)
                        compared += 1
        value = total / compared
        if decay > 0.0:
            penalty: Any = 0.0
            for w in weights_x:
                penalty = penalty + w * w
            value = value + decay * penalty
        return value

    start = (
        *network.initial_parameters(resolve_stream(seed), output_scale=scale),
        *theta0,
    )
    trained = _train(
        loss, start, max_iterations=iterations, gradient_tolerance=tolerance
    )
    return HybridODEResult(
        network=network,
        parameters=trained.x[:count],
        physical=trained.x[count:],
        correction_components=components,
        loss=trained.value,
        converged=trained.converged,
        message=trained.message,
        iterations=trained.iterations,
        evaluations=trained.evaluations,
        seconds=trained.seconds,
        physics=physics,
    )


@dataclass(frozen=True, slots=True)
class SparseDistillation:
    """Sparse polynomial distilled from a learned term.

    Attributes:
        names: Names of all library terms (``"1"``, ``"x0"``, ``"x0*x1"``, ...).
        exponents: Exponent tuple of every library term.
        coefficients: Coefficient of every library term (zero if pruned).
        active: Indices of the retained terms.
        rms_residual: Root-mean-square misfit of the sparse model on the
            fitting data.
        rounds: Thresholding rounds performed.
        converged: Whether the active set stopped changing within the limit.

    References:
        [EQ-SUR-PIML-005], [REF-BRUNTON-PROCTOR-KUTZ-2016].

    """

    names: tuple[str, ...]
    exponents: tuple[tuple[int, ...], ...]
    coefficients: tuple[float, ...]
    active: tuple[int, ...]
    rms_residual: float
    rounds: int
    converged: bool

    def terms(self) -> dict[str, float]:
        """Return the retained terms as ``{name: coefficient}``.

        References:
            [EQ-SUR-PIML-005].

        """
        return {self.names[i]: self.coefficients[i] for i in self.active}

    def evaluate(self, x: Sequence[float]) -> float:
        """Evaluate the sparse polynomial at a point.

        Args:
            x: Finite point with one entry per input variable.

        Returns:
            ``sum_k c_k prod_i x_i^e_ki`` over the retained terms.

        Raises:
            ValueError: If ``x`` has the wrong length.

        References:
            [EQ-SUR-PIML-005].

        """
        point = _finite_vector(x, "x")
        if len(point) != len(self.exponents[0]):
            raise ValueError("x has the wrong length")
        return math.fsum(
            self.coefficients[k]
            * math.prod(v**e for v, e in zip(point, self.exponents[k], strict=True))
            for k in self.active
        )


def distill_sparse_regression(
    inputs: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    degree: int = 3,
    threshold: float = 0.05,
    max_rounds: int = 10,
    variable_names: Sequence[str] | None = None,
) -> SparseDistillation:
    """Distill a learned term into a sparse polynomial by thresholded LS.

    The library holds every monomial of total degree at most ``degree`` in the
    ``d`` inputs. Sequentially thresholded least squares fits all terms by
    Householder-QR least squares, removes the terms with ``|c| < threshold``
    and refits the rest until the active set stops changing
    [EQ-SUR-PIML-005]. The threshold is in the units of the target per unit
    of the monomial, so inputs should be of order one.

    Args:
        inputs: Sample points (rows), e.g. states visited by the trajectory.
        targets: Values of the learned term at the points.
        degree: Maximum total degree (at least 1; the constant is included).
        threshold: Positive pruning threshold on ``|c|``.
        max_rounds: Maximum number of refits.
        variable_names: Optional names for the inputs (default ``x0, x1, ...``).

    Returns:
        :class:`SparseDistillation`.

    Raises:
        TypeError: For invalid argument types.
        ValueError: If the data are ragged, non-finite, too few for the
            library, the library is numerically rank deficient on the active
            set, or every term is pruned.

    Equation:
        [EQ-SUR-PIML-005].

    References:
        [EQ-SUR-PIML-005], [REF-BRUNTON-PROCTOR-KUTZ-2016],
        [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    rows = tuple(_finite_vector(row, "inputs row") for row in inputs)
    y = _finite_vector(targets, "targets")
    if not rows or len(rows) != len(y):
        raise ValueError("inputs and targets must be non-empty and of equal length")
    dimension = len(rows[0])
    if dimension < 1 or any(len(row) != dimension for row in rows):
        raise ValueError("all input rows must have the same positive length")
    top = _positive_int(degree, "degree")
    cut = _finite(threshold, "threshold")
    if cut <= 0.0:
        raise ValueError("threshold must be positive")
    limit = _positive_int(max_rounds, "max_rounds")
    labels = (
        tuple(variable_names)
        if variable_names is not None
        else tuple(f"x{i}" for i in range(dimension))
    )
    if len(labels) != dimension:
        raise ValueError("variable_names must have one entry per input")

    exponents: list[tuple[int, ...]] = []
    names: list[str] = []
    for total in range(top + 1):
        for combo in combinations_with_replacement(range(dimension), total):
            exponent = tuple(combo.count(i) for i in range(dimension))
            exponents.append(exponent)
            parts = [
                labels[i] if e == 1 else f"{labels[i]}^{e}"
                for i, e in enumerate(exponent)
                if e > 0
            ]
            names.append("*".join(parts) if parts else "1")
    if len(rows) <= len(exponents):
        raise ValueError("need more samples than library terms")
    library = tuple(
        tuple(
            math.prod(v**e for v, e in zip(row, exponent, strict=True))
            for exponent in exponents
        )
        for row in rows
    )
    active = list(range(len(exponents)))
    converged = False
    rounds = 0

    def refit(columns: Sequence[int]) -> tuple[float, ...]:
        """Return the least-squares coefficients on the given library columns.

        References:
            [EQ-SUR-PIML-005].

        """
        sub = tuple(tuple(row[k] for k in columns) for row in library)
        factor = householder_qr(sub)
        if not factor.full_rank:
            raise ValueError("the monomial library is rank deficient on the data")
        return tuple(factor.solve(y))

    for _ in range(limit):
        rounds += 1
        fitted = refit(active)
        keep = [k for k, c in zip(active, fitted, strict=True) if abs(c) >= cut]
        if len(keep) == len(active):
            converged = True
            break
        if not keep:
            raise ValueError("every library term was pruned; lower the threshold")
        active = keep
    final = refit(active)
    coefficients = [0.0] * len(exponents)
    for k, c in zip(active, final, strict=True):
        coefficients[k] = c
    misfit = math.sqrt(
        math.fsum(
            (math.fsum(coefficients[k] * row[k] for k in active) - target) ** 2
            for row, target in zip(library, y, strict=True)
        )
        / len(y)
    )
    return SparseDistillation(
        names=tuple(names),
        exponents=tuple(exponents),
        coefficients=tuple(coefficients),
        active=tuple(active),
        rms_residual=misfit,
        rounds=rounds,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# 3. Residual (discrepancy) learning with split-conformal intervals
# ---------------------------------------------------------------------------


def split_conformal_halfwidth(scores: Sequence[float], alpha: float) -> float:
    """Return the finite-sample split-conformal quantile of calibration scores.

    ``q = s_(k)`` with ``k = ceil((1 - alpha)(n + 1))`` computed in exact
    rational arithmetic from the decimal text of ``alpha``, and ``+inf`` when
    ``k = n + 1`` [EQ-SUR-PIML-007]. This is the same rule as
    :func:`probstat.conformal.conformal_quantile`, kept local because the
    conformal package is a higher layer.

    Args:
        scores: Finite calibration scores (at least one), typically absolute
            residuals ``|y_i - mu(x_i)|``.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        The quantile, possibly ``inf``.

    Raises:
        TypeError: If ``alpha`` is not a real number.
        ValueError: If ``alpha`` is outside ``(0, 1)`` or the scores are
            empty or not finite.

    Equation:
        [EQ-SUR-PIML-007].

    References:
        [EQ-SUR-PIML-007], [REF-VOVK-GAMMERMAN-SHAFER-2005],
        [REF-LEI-ETAL-2018], [REF-ANGELOPOULOS-BATES-2023].

    """
    level = _finite(alpha, "alpha")
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    ordered = sorted(_finite_vector(scores, "scores"))
    if not ordered:
        raise ValueError("at least one calibration score is required")
    n = len(ordered)
    rank = math.ceil((1 - Fraction(repr(level))) * (n + 1))
    if rank > n:
        return math.inf
    return ordered[max(rank, 1) - 1]


@dataclass(frozen=True, slots=True)
class ResidualHybridModel:
    """Physics model plus a learned residual with conformal intervals.

    The prediction is ``mu(x) = f_phys(x) + r(x)`` (``f_phys = 0`` for an ML-only
    model, ``r = 0`` for a physics-only model) and the interval is
    ``mu(x) +/- q`` with the split-conformal half-width ``q`` of the absolute
    calibration residuals [EQ-SUR-PIML-006].

    Attributes:
        physics: Physics model ``x -> float`` or ``None``.
        learner: ``"gp"``, ``"ensemble"`` or ``"none"``.
        residual_model: Fitted residual learner (``None`` for ``"none"``).
        alpha: Miscoverage level.
        quantile: Conformal half-width ``q`` (may be ``inf``).
        calibration_scores: Absolute calibration residuals.

    References:
        [EQ-SUR-PIML-006], [REF-KENNEDY-OHAGAN-2001].

    """

    physics: Callable[[Sequence[float]], float] | None
    learner: str
    residual_model: GaussianProcessRegression | DeepEnsemble | None
    alpha: float
    quantile: float
    calibration_scores: tuple[float, ...]

    def predict(self, x: Sequence[float]) -> float:
        """Return the hybrid point prediction ``f_phys(x) + r(x)``.

        Args:
            x: Finite input vector.

        Returns:
            The prediction.

        References:
            [EQ-SUR-PIML-006].

        """
        point = _finite_vector(x, "x")
        base = float(self.physics(point)) if self.physics is not None else 0.0
        if self.residual_model is None:
            return base
        return base + float(self.residual_model.predict(point))

    def interval(self, x: Sequence[float]) -> tuple[float, float]:
        """Return the split-conformal interval at ``x``.

        Args:
            x: Finite input vector.

        Returns:
            ``(mu - q, mu + q)``; infinite when too few calibration points.

        References:
            [EQ-SUR-PIML-006], [EQ-SUR-PIML-007].

        """
        centre = self.predict(x)
        return centre - self.quantile, centre + self.quantile


def _points(
    points: Sequence[Sequence[float]], name: str
) -> tuple[tuple[float, ...], ...]:
    """Validate a list of equal-length finite input vectors.

    Args:
        points: Input rows.
        name: Argument name for the error message.

    Returns:
        Tuple of tuples.

    Raises:
        ValueError: If empty, ragged or not finite.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    rows = tuple(_finite_vector(row, f"{name} row") for row in points)
    if not rows or any(len(row) != len(rows[0]) or not row for row in rows):
        raise ValueError(f"{name} must be a non-empty list of equal-length vectors")
    return rows


def fit_residual_hybrid(
    physics: Callable[[Sequence[float]], float] | None,
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    *,
    learner: str = "gp",
    alpha: float = 0.1,
    seed: int | RandomStream = 0,
    kernel: str = "matern52",
    ensemble_members: int = 3,
    ensemble_hidden: Sequence[int] = (8,),
    ensemble_iterations: int = 400,
) -> ResidualHybridModel:
    """Fit a physics-plus-residual model and calibrate split-conformal intervals.

    The residual learner is trained on ``r_i = y_i - f_phys(x_i)`` of the
    *training* set only; the absolute residuals of the final model on the
    disjoint *calibration* set give the half-width ``q`` [EQ-SUR-PIML-006,
    EQ-SUR-PIML-007]. The three combinations are the physics-only model
    (``learner="none"``), the ML-only model (``physics=None``) and the hybrid.
    Note that the GP and the ensemble standardize their targets, so far from
    the data the residual model reverts to the mean of the training residuals
    (not to zero).

    Args:
        physics: Calibrated physics model ``x -> float`` or ``None``.
        x_train: Training inputs.
        y_train: Training outputs.
        x_calibration: Calibration inputs, independent of the training set.
        y_calibration: Calibration outputs.
        learner: ``"gp"`` (:func:`fit_gaussian_process`, noise estimated),
            ``"ensemble"`` (:func:`fit_deep_ensemble`) or ``"none"``.
        alpha: Miscoverage level in ``(0, 1)``.
        seed: Seed or stream for the learner.
        kernel: GP kernel name.
        ensemble_members: Ensemble size.
        ensemble_hidden: Ensemble hidden widths.
        ensemble_iterations: Ensemble Adam steps per member.

    Returns:
        :class:`ResidualHybridModel`.

    Raises:
        TypeError: If ``physics`` is neither callable nor ``None``.
        ValueError: For an unknown learner, ``learner="none"`` without
            physics, mismatched or empty data, or an invalid ``alpha``.

    Equation:
        [EQ-SUR-PIML-006], [EQ-SUR-PIML-007].

    References:
        [EQ-SUR-PIML-006], [REF-KENNEDY-OHAGAN-2001],
        [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017],
        [REF-VOVK-GAMMERMAN-SHAFER-2005], [REF-LEI-ETAL-2018].

    """
    if physics is not None and not callable(physics):
        raise TypeError("physics must be callable or None")
    if learner not in _RESIDUAL_LEARNERS:
        raise ValueError(f"learner must be one of {_RESIDUAL_LEARNERS}")
    if learner == "none" and physics is None:
        raise ValueError("learner='none' needs a physics model")
    xs, ys = _points(x_train, "x_train"), _finite_vector(y_train, "y_train")
    xc, yc = (
        _points(x_calibration, "x_calibration"),
        _finite_vector(y_calibration, "y_calibration"),
    )
    if len(xs) != len(ys) or len(xc) != len(yc):
        raise ValueError("inputs and outputs must have equal length")
    if len(xs[0]) != len(xc[0]):
        raise ValueError("training and calibration inputs differ in dimension")
    stream = resolve_stream(seed)

    def base(x: Sequence[float]) -> float:
        """Return the physics prediction (zero for an ML-only model).

        References:
            [EQ-SUR-PIML-006].

        """
        return float(physics(x)) if physics is not None else 0.0

    residual_model: GaussianProcessRegression | DeepEnsemble | None = None
    targets = [y - base(x) for x, y in zip(xs, ys, strict=True)]
    if learner == "gp":
        residual_model = fit_gaussian_process(
            xs, targets, kernel=kernel, estimate_noise=True, seed=stream.child("gp")
        )
    elif learner == "ensemble":
        residual_model = fit_deep_ensemble(
            xs,
            targets,
            members=_positive_int(ensemble_members, "ensemble_members"),
            hidden=_hidden_widths(ensemble_hidden, "ensemble_hidden"),
            iterations=_positive_int(ensemble_iterations, "ensemble_iterations"),
            batch_size=16,
            seed=stream.child("ensemble"),
        )
    scores = tuple(
        abs(
            y
            - base(x)
            - (float(residual_model.predict(x)) if residual_model is not None else 0.0)
        )
        for x, y in zip(xc, yc, strict=True)
    )
    return ResidualHybridModel(
        physics=physics,
        learner=learner,
        residual_model=residual_model,
        alpha=float(alpha),
        quantile=split_conformal_halfwidth(scores, alpha),
        calibration_scores=scores,
    )


@dataclass(frozen=True, slots=True)
class ModelScore:
    """Accuracy and interval quality of one model on one test set.

    Attributes:
        model: ``"physics"``, ``"ml"`` or ``"hybrid"``.
        test_set: Name of the test set.
        rmse: Root-mean-square error of the point prediction.
        coverage: Fraction of test outputs inside the conformal interval.
        mean_width: Mean interval width (``inf`` if the quantile is infinite).

    References:
        [EQ-SUR-PIML-006].

    """

    model: str
    test_set: str
    rmse: float
    coverage: float
    mean_width: float


@dataclass(frozen=True, slots=True)
class HybridComparison:
    """Measured comparison of physics-only, ML-only and hybrid models.

    Attributes:
        scores: One :class:`ModelScore` per model and test set.
        nominal_coverage: ``1 - alpha``.
        models: The three fitted :class:`ResidualHybridModel` objects keyed
            by ``"physics"``, ``"ml"`` and ``"hybrid"`` as ordered pairs.

    References:
        [EQ-SUR-PIML-006].

    """

    scores: tuple[ModelScore, ...]
    nominal_coverage: float
    models: tuple[tuple[str, ResidualHybridModel], ...]

    def get(self, model: str, test_set: str) -> ModelScore:
        """Return the score of ``model`` on ``test_set``.

        Args:
            model: ``"physics"``, ``"ml"`` or ``"hybrid"``.
            test_set: Test-set name.

        Returns:
            The matching :class:`ModelScore`.

        Raises:
            KeyError: If no such score exists.

        References:
            [EQ-SUR-PIML-006].

        """
        for score in self.scores:
            if score.model == model and score.test_set == test_set:
                return score
        raise KeyError((model, test_set))


def compare_physics_ml_hybrid(
    physics: Callable[[Sequence[float]], float],
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    test_sets: Mapping[str, tuple[Sequence[Sequence[float]], Sequence[float]]],
    *,
    learner: str = "gp",
    alpha: float = 0.1,
    seed: int | RandomStream = 0,
    kernel: str = "matern52",
    ensemble_members: int = 3,
    ensemble_hidden: Sequence[int] = (8,),
    ensemble_iterations: int = 400,
) -> HybridComparison:
    """Compare physics-only, ML-only and hybrid models on named test sets.

    All three models share the training set, the calibration set and the
    conformal level; the physics-only model is conformalized too. The test
    sets typically contain an in-range set (exchangeable with calibration,
    where the coverage guarantee applies) and an extrapolation set (where it
    does not, and the measured coverage is the honest answer). Nothing is
    tuned on the test sets.

    Args:
        physics: Calibrated physics model ``x -> float``.
        x_train: Training inputs.
        y_train: Training outputs.
        x_calibration: Calibration inputs.
        y_calibration: Calibration outputs.
        test_sets: ``{name: (inputs, outputs)}``.
        learner: ``"gp"`` or ``"ensemble"`` (used by the ML and hybrid models).
        alpha: Miscoverage level.
        seed: Seed or stream.
        kernel: GP kernel name.
        ensemble_members: Ensemble size.
        ensemble_hidden: Ensemble hidden widths.
        ensemble_iterations: Ensemble Adam steps per member.

    Returns:
        :class:`HybridComparison`.

    Raises:
        ValueError: If ``learner`` is ``"none"``, ``test_sets`` is empty or a
            test set is malformed.

    Equation:
        [EQ-SUR-PIML-006].

    References:
        [EQ-SUR-PIML-006], [REF-KENNEDY-OHAGAN-2001].

    """
    if learner == "none":
        raise ValueError("the comparison needs a learner ('gp' or 'ensemble')")
    if not test_sets:
        raise ValueError("test_sets must not be empty")
    stream = resolve_stream(seed)
    options: dict[str, Any] = {
        "alpha": alpha,
        "kernel": kernel,
        "ensemble_members": ensemble_members,
        "ensemble_hidden": ensemble_hidden,
        "ensemble_iterations": ensemble_iterations,
    }
    fitted = (
        (
            "physics",
            fit_residual_hybrid(
                physics,
                x_train,
                y_train,
                x_calibration,
                y_calibration,
                learner="none",
                seed=stream.child("physics"),
                **options,
            ),
        ),
        (
            "ml",
            fit_residual_hybrid(
                None,
                x_train,
                y_train,
                x_calibration,
                y_calibration,
                learner=learner,
                seed=stream.child("ml"),
                **options,
            ),
        ),
        (
            "hybrid",
            fit_residual_hybrid(
                physics,
                x_train,
                y_train,
                x_calibration,
                y_calibration,
                learner=learner,
                seed=stream.child("hybrid"),
                **options,
            ),
        ),
    )
    scores: list[ModelScore] = []
    for name, (xt, yt) in test_sets.items():
        rows, truth = (
            _points(xt, f"test_sets[{name!r}] inputs"),
            _finite_vector(yt, f"test_sets[{name!r}] outputs"),
        )
        if len(rows) != len(truth):
            raise ValueError(f"test set {name!r} has mismatched lengths")
        for label, model in fitted:
            predictions = [model.predict(x) for x in rows]
            rmse = math.sqrt(
                math.fsum((p - t) ** 2 for p, t in zip(predictions, truth, strict=True))
                / len(truth)
            )
            inside = sum(
                abs(p - t) <= model.quantile
                for p, t in zip(predictions, truth, strict=True)
            )
            scores.append(
                ModelScore(
                    model=label,
                    test_set=name,
                    rmse=rmse,
                    coverage=inside / len(truth),
                    mean_width=2.0 * model.quantile,
                )
            )
    return HybridComparison(
        scores=tuple(scores),
        nominal_coverage=1.0 - float(alpha),
        models=fitted,
    )


# ---------------------------------------------------------------------------
# 4. DeepONet
# ---------------------------------------------------------------------------


def _deeponet_output(
    branch: MLP,
    trunk: MLP,
    parameters: Sequence[Any],
    u_values: Sequence[Any],
    ys: Sequence[Any],
    trunk_features: Sequence[Sequence[Any]] | None = None,
) -> list[Any]:
    """Return ``G(u)(y_j)`` for several query points.

    Args:
        branch: Branch network (sensors -> latent).
        trunk: Trunk network (query -> latent).
        parameters: Flat vector ``[branch, trunk, b0]``.
        u_values: Input function sampled at the sensors.
        ys: Query locations.
        trunk_features: Optional precomputed trunk outputs for ``ys``.

    Returns:
        ``sum_k b_k(u) t_k(y_j) + b0`` for each query.

    References:
        [EQ-SUR-PIML-008].

    """
    nb = branch.parameter_count
    nt = trunk.parameter_count
    b = branch.forward(parameters[:nb], list(u_values))
    bias = parameters[nb + nt]
    outputs: list[Any] = []
    for j, y in enumerate(ys):
        t = (
            trunk_features[j]
            if trunk_features is not None
            else trunk.forward(parameters[nb : nb + nt], [y])
        )
        total: Any = bias
        for bk, tk in zip(b, t, strict=True):
            total = total + bk * tk
        outputs.append(total)
    return outputs


@dataclass(frozen=True, slots=True)
class DeepONet:
    """Deep operator network ``G(u)(y) = sum_k b_k(u) t_k(y) + b0``.

    Attributes:
        branch: Branch MLP ``(m, *hidden, p)`` on the sensor values.
        trunk: Trunk MLP ``(1, *hidden, p)`` on the query location.
        sensors: The ``m`` sensor locations at which input functions are
            sampled.
        parameters: Flat vector ``[branch, trunk, b0]``.

    References:
        [EQ-SUR-PIML-008], [REF-LU-JIN-PANG-ZHANG-KARNIADAKIS-2021].

    """

    branch: MLP
    trunk: MLP
    sensors: tuple[float, ...]
    parameters: tuple[float, ...]

    def predict(
        self, u_values: Sequence[float], queries: Sequence[float]
    ) -> tuple[float, ...]:
        """Evaluate the operator on one sampled function at several queries.

        Args:
            u_values: The input function at :attr:`sensors`.
            queries: Query locations ``y``.

        Returns:
            ``G(u)(y)`` for every query.

        Raises:
            ValueError: If ``u_values`` does not have one entry per sensor.

        References:
            [EQ-SUR-PIML-008].

        """
        values = _finite_vector(u_values, "u_values")
        if len(values) != len(self.sensors):
            raise ValueError("u_values needs one entry per sensor")
        return tuple(
            float(v)
            for v in _deeponet_output(
                self.branch,
                self.trunk,
                self.parameters,
                values,
                _finite_vector(queries, "queries"),
            )
        )


@dataclass(frozen=True, slots=True)
class DeepONetDataset:
    """Generated operator-learning data with a shared query grid.

    Attributes:
        sensors: Sensor locations.
        queries: Shared query locations.
        branch_inputs: ``u_i`` sampled at the sensors, one row per function.
        targets: ``G(u_i)(y_j)``, one row per function.
        coefficients: Generating coefficients of each function.

    References:
        [EQ-SUR-PIML-009].

    """

    sensors: tuple[float, ...]
    queries: tuple[float, ...]
    branch_inputs: tuple[tuple[float, ...], ...]
    targets: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, float, float], ...]


def _trig_function(coefficients: tuple[float, float, float], s: float) -> float:
    """Return ``a0 + a1 cos(pi s) + b1 sin(pi s)``.

    References:
        [EQ-SUR-PIML-009].

    """
    a0, a1, b1 = coefficients
    return a0 + a1 * math.cos(math.pi * s) + b1 * math.sin(math.pi * s)


def antiderivative_exact(coefficients: tuple[float, float, float], y: float) -> float:
    """Return ``int_0^y u(s) ds`` for ``u = a0 + a1 cos(pi s) + b1 sin(pi s)``.

    ``G(u)(y) = a0 y + a1 sin(pi y)/pi + b1 (1 - cos(pi y))/pi``
    [EQ-SUR-PIML-009].

    Args:
        coefficients: ``(a0, a1, b1)``.
        y: Upper limit.

    Returns:
        The exact integral.

    Equation:
        [EQ-SUR-PIML-009].

    References:
        [EQ-SUR-PIML-009].

    """
    a0, a1, b1 = coefficients
    return (
        a0 * y
        + a1 * math.sin(math.pi * y) / math.pi
        + b1 * (1.0 - math.cos(math.pi * y)) / math.pi
    )


def antiderivative_dataset(
    n_functions: int,
    *,
    n_sensors: int = 6,
    n_queries: int = 8,
    seed: int | RandomStream = 0,
) -> DeepONetDataset:
    """Generate input functions on ``[0, 1]`` and their exact antiderivatives.

    Functions are ``a0 + a1 cos(pi s) + b1 sin(pi s)`` with coefficients
    drawn independently uniform on ``[-1, 1]``; sensors and the shared query
    grid are uniform on ``[0, 1]`` (queries exclude 0, where ``G = 0``).
    The data are synthetic; they come from the closed form
    [EQ-SUR-PIML-009].

    Args:
        n_functions: Number of functions.
        n_sensors: Number of sensors (at least 2).
        n_queries: Number of shared query locations.
        seed: Seed or stream.

    Returns:
        :class:`DeepONetDataset`.

    Raises:
        ValueError: For non-positive counts or fewer than 2 sensors.

    Equation:
        [EQ-SUR-PIML-009].

    References:
        [EQ-SUR-PIML-009], [REF-LU-JIN-PANG-ZHANG-KARNIADAKIS-2021].

    """
    count = _positive_int(n_functions, "n_functions")
    sensors_n = _positive_int(n_sensors, "n_sensors")
    queries_n = _positive_int(n_queries, "n_queries")
    if sensors_n < 2:
        raise ValueError("n_sensors must be at least 2")
    generator = resolve_stream(seed).child("antiderivative").generator()
    sensors = tuple(i / (sensors_n - 1) for i in range(sensors_n))
    queries = tuple((j + 1) / queries_n for j in range(queries_n))
    coefficients = tuple(
        (
            generator.uniform(-1.0, 1.0),
            generator.uniform(-1.0, 1.0),
            generator.uniform(-1.0, 1.0),
        )
        for _ in range(count)
    )
    return DeepONetDataset(
        sensors=sensors,
        queries=queries,
        branch_inputs=tuple(
            tuple(_trig_function(c, s) for s in sensors) for c in coefficients
        ),
        targets=tuple(
            tuple(antiderivative_exact(c, y) for y in queries) for c in coefficients
        ),
        coefficients=coefficients,
    )


@dataclass(frozen=True, slots=True)
class DeepONetFit:
    """A trained DeepONet and its training cost.

    Attributes:
        model: The fitted :class:`DeepONet`.
        loss: Final mean squared training error (last Adam iterate).
        iterations: Adam steps.
        evaluations: Loss-and-gradient recordings (training cost).
        seconds: Wall-clock training time (informational).

    References:
        [EQ-SUR-PIML-008].

    """

    model: DeepONet
    loss: float
    iterations: int
    evaluations: int
    seconds: float


def fit_deeponet(
    sensors: Sequence[float],
    branch_inputs: Sequence[Sequence[float]],
    queries: Sequence[float],
    targets: Sequence[Sequence[float]],
    *,
    latent: int = 6,
    branch_hidden: Sequence[int] = (10,),
    trunk_hidden: Sequence[int] = (10,),
    iterations: int = 1000,
    learning_rate: float = 0.01,
    seed: int | RandomStream = 0,
) -> DeepONetFit:
    """Train a DeepONet on sampled functions and their images.

    Minimizes the mean squared error
    ``(1/(N Q)) sum_{i,j} (G_w(u_i)(y_j) - G(u_i)(y_j))^2`` over all branch,
    trunk and bias parameters by full-batch Adam
    (:func:`probstat.optimization.adam_minimize`) with the exact reverse-mode
    gradient [EQ-SUR-PIML-008]. BFGS is not used here: its dense
    ``O(n^3)`` pure-Python matrix products dominate once there are a few
    hundred parameters (measured: 1.4 s per iteration at 163 parameters).
    The trunk features are computed once per query because the query grid is
    shared by all functions.

    Args:
        sensors: Sensor locations (recorded in the model).
        branch_inputs: ``u_i`` at the sensors, one row per training function.
        queries: Shared query locations.
        targets: ``G(u_i)(y_j)``, one row per function.
        latent: Width ``p`` of the shared latent space.
        branch_hidden: Hidden widths of the branch network.
        trunk_hidden: Hidden widths of the trunk network.
        iterations: Adam steps (at least 1).
        learning_rate: Adam step size (positive).
        seed: Seed or stream.

    Returns:
        :class:`DeepONetFit`.

    Raises:
        ValueError: If shapes disagree or an entry is not finite.

    Equation:
        [EQ-SUR-PIML-008].

    References:
        [EQ-SUR-PIML-008], [REF-LU-JIN-PANG-ZHANG-KARNIADAKIS-2021],
        [REF-CHEN-CHEN-1995].

    """
    sensor_grid = _finite_vector(sensors, "sensors")
    grid = _finite_vector(queries, "queries")
    inputs = tuple(_finite_vector(row, "branch_inputs row") for row in branch_inputs)
    outputs = tuple(_finite_vector(row, "targets row") for row in targets)
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("branch_inputs and targets need one row per function")
    if any(len(row) != len(sensor_grid) for row in inputs):
        raise ValueError("each branch input needs one value per sensor")
    if any(len(row) != len(grid) for row in outputs):
        raise ValueError("each target row needs one value per query")
    width = _positive_int(latent, "latent")
    steps = _positive_int(iterations, "iterations")
    rate = _finite(learning_rate, "learning_rate")
    if rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    branch = MLP(
        (len(sensor_grid), *_hidden_widths(branch_hidden, "branch_hidden"), width)
    )
    trunk = MLP((1, *_hidden_widths(trunk_hidden, "trunk_hidden"), width))
    nb, nt = branch.parameter_count, trunk.parameter_count
    stream = resolve_stream(seed)

    def loss(x: Sequence[Any]) -> Any:
        """Return the mean squared operator-learning error.

        References:
            [EQ-SUR-PIML-008].

        """
        features = [trunk.forward(x[nb : nb + nt], [y]) for y in grid]
        total: Any = 0.0
        for row, target in zip(inputs, outputs, strict=True):
            predicted = _deeponet_output(branch, trunk, x, row, grid, features)
            for p, t in zip(predicted, target, strict=True):
                total = total + (p - t) * (p - t)
        return total / (len(inputs) * len(grid))

    start = (
        *branch.initial_parameters(stream.child("branch")),
        *trunk.initial_parameters(stream.child("trunk")),
        0.0,
    )
    objective = _CachedObjective(loss)

    def stochastic(
        x: tuple[float, ...], _stream: RandomStream
    ) -> tuple[float, tuple[float, ...]]:
        """Return the full-batch loss and gradient (no sampling).

        References:
            [EQ-SUR-PIML-008].

        """
        return objective.value(x), objective.gradient(x)

    began = time.perf_counter()
    trained = adam_minimize(
        stochastic, start, iterations=steps, learning_rate=rate, seed=stream
    )
    seconds = time.perf_counter() - began
    return DeepONetFit(
        model=DeepONet(branch, trunk, sensor_grid, trained.x),
        loss=objective.value(trained.x),
        iterations=steps,
        evaluations=objective.recordings,
        seconds=seconds,
    )


def deeponet_relative_error(model: DeepONet, dataset: DeepONetDataset) -> float:
    """Return the relative L2 error of a DeepONet on a dataset.

    ``sqrt(sum (G_w - G)^2 / sum G^2)`` over all functions and queries.

    Args:
        model: Fitted network.
        dataset: Data with the same sensors as the model.

    Returns:
        The relative error.

    Raises:
        ValueError: If the sensors differ or the targets are identically zero.

    Equation:
        [EQ-SUR-PIML-008].

    References:
        [EQ-SUR-PIML-008].

    """
    if dataset.sensors != model.sensors:
        raise ValueError("dataset and model use different sensors")
    numerator = 0.0
    denominator = 0.0
    for row, target in zip(dataset.branch_inputs, dataset.targets, strict=True):
        predicted = model.predict(row, dataset.queries)
        numerator += math.fsum(
            (p - t) ** 2 for p, t in zip(predicted, target, strict=True)
        )
        denominator += math.fsum(t * t for t in target)
    if denominator == 0.0:
        raise ValueError("the targets are identically zero")
    return math.sqrt(numerator / denominator)
