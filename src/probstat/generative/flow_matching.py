"""Conditional flow matching with the straight-line (optimal-transport) path.

A continuous normalizing flow transports a base law ``p_0 = N(0, I)`` to a
data law along the ordinary differential equation ``dx/dt = v(t, x)``.
Flow matching (Lipman et al. 2023) fits ``v`` by regression instead of by
maximum likelihood through an ODE solver. Fix a data point ``x_1`` and the
conditional path

``x_t = (1 - (1 - s) t) x_0 + t x_1``,  ``x_0 ~ N(0, I)``,

with a small ``s = sigma_min >= 0``. Differentiating in ``t`` at fixed
``(x_0, x_1)`` gives the conditional target velocity
``u = x_1 - (1 - s) x_0`` [EQ-GEN-FM-001]. The marginal field that generates
the law of ``x_t`` is the conditional expectation ``E[u | x_t = x]``, which
is the minimizer of ``E |v(t, x_t) - u|^2`` over all functions
[EQ-GEN-FM-002]; a network trained on that loss therefore approximates the
marginal field without ever evaluating it. At ``t = 1`` the path law is the
data law convolved with ``N(0, s^2 I)``.

For Gaussian data the marginal field is linear in ``x`` with closed-form
coefficients (:class:`GaussianMarginalVelocity`, [EQ-GEN-FM-003]), which
gives an oracle for the sampler (:func:`integrate_flow`) and for the
log-density by the instantaneous change of variables
(:func:`flow_log_density`, [EQ-GEN-FM-004]) that does not involve training.

The networks are one or two hidden layers of 8 to 32 ``tanh`` units in
dimension one or two, trained in scalar Python; large-scale training is out
of scope.

References:
    [REF-LIPMAN-ETAL-2023], [REF-KINGMA-BA-2015].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..foundations.random_streams import RandomStream
from ..numerics import MLP, Dual, integrate_fixed_step
from ..optimization import AdamResult, adam_minimize
from . import _backprop
from ._common import (
    Point,
    as_point,
    as_points,
    positive_integer,
    positive_real,
    resolve,
    standard_normal_log_density,
)

VelocityField = Callable[[float, Sequence[Any]], Sequence[Any]]
DivergenceFunction = Callable[[float, Sequence[float]], float]
FLOW_ODE_METHODS: tuple[str, ...] = ("euler", "heun", "rk4")
DIVERGENCE_MODES: tuple[str, ...] = ("autodiff", "finite-difference")


def _sigma_min(value: object) -> float:
    """Validate the terminal path width ``0 <= sigma_min < 1``.

    Raises:
        ValueError: If the value is not a real number in ``[0, 1)``.

    References:
        [EQ-GEN-FM-001]. Project validation policy
        [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("sigma_min must be a real number in [0, 1)")
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError("sigma_min must be a real number in [0, 1)")
    return result


def _unit_time(value: object) -> float:
    """Validate a time in ``[0, 1]``.

    Raises:
        ValueError: If the value is not a real number in ``[0, 1]``.

    References:
        [EQ-GEN-FM-001]. Project validation policy
        [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("t must be a real number in [0, 1]")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("t must be a real number in [0, 1]")
    return result


def conditional_flow_path(
    x0: object, x1: object, t: float, *, sigma_min: float = 0.0
) -> tuple[Point, Point]:
    """Return the path point ``x_t`` and the conditional target velocity.

    ``x_t = (1 - (1 - s) t) x_0 + t x_1`` and, differentiating in ``t`` with
    ``x_0`` and ``x_1`` fixed, ``u = x_1 - (1 - s) x_0``. Written as a field
    in ``x`` this is ``u_t(x | x_1) = (x_1 - (1 - s) x) / (1 - (1 - s) t)``
    because ``x_0 = (x - t x_1) / (1 - (1 - s) t)``.

    Args:
        x0: Base point (a number in one dimension).
        x1: Data point of the same dimension.
        t: Time in ``[0, 1]``.
        sigma_min: Path width ``s`` at ``t = 1``, in ``[0, 1)``.

    Returns:
        ``(x_t, u)``.

    Raises:
        TypeError: If a point is not a sequence of reals or a real.
        ValueError: If the dimensions differ, a coordinate is not finite,
            or ``t`` or ``sigma_min`` is out of range.

    References:
        [EQ-GEN-FM-001], [REF-LIPMAN-ETAL-2023].

    """
    start = as_points([x0], name="x0")[0]
    end = as_point(x1, len(start), name="x1")
    time = _unit_time(t)
    shrink = 1.0 - _sigma_min(sigma_min)
    keep = 1.0 - shrink * time
    point = tuple(keep * a + time * b for a, b in zip(start, end, strict=True))
    velocity = tuple(b - shrink * a for a, b in zip(start, end, strict=True))
    return point, velocity


@dataclass(frozen=True)
class GaussianMarginalVelocity:
    """Exact marginal velocity field for independent Gaussian data.

    For ``x_1 ~ N(m, sd^2)`` (coordinate by coordinate) and the path of
    [EQ-GEN-FM-001], ``x_t = a x_0 + t x_1`` with ``a = 1 - (1 - s) t`` is
    Gaussian with mean ``t m`` and variance ``V(t) = a^2 + t^2 sd^2``, and
    ``u = x_1 - (1 - s) x_0`` is jointly Gaussian with it. Linear regression
    of ``u`` on ``x_t`` gives

    ``v(t, x) = m + c(t) (x - t m)``,
    ``c(t) = (t sd^2 - (1 - s) a) / V(t) = V'(t) / (2 V(t))``.

    Integrating ``dx/dt = v`` from ``t = 0`` to ``1`` maps ``N(0, 1)`` to
    ``N(m, sd^2 + s^2)`` by ``x_1 = m + sqrt(sd^2 + s^2) x_0``. The call is
    written with arithmetic operators only, so forward-mode duals pass
    through it.

    Args:
        mean: Data mean ``m`` (number or one entry per coordinate).
        standard_deviation: Positive data standard deviation(s) ``sd``.
        sigma_min: Path width ``s`` in ``[0, 1)``.

    References:
        [EQ-GEN-FM-003]; derived from [EQ-GEN-FM-001] and [EQ-GEN-FM-002]
        (see the module docstring), [REF-LIPMAN-ETAL-2023] for the path.

    """

    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    sigma_min: float = 0.0

    def __post_init__(self) -> None:
        """Validate the moments and the path width.

        Raises:
            TypeError: If a moment is not a real or a sequence of reals.
            ValueError: If the lengths differ, a value is not finite, a
                standard deviation is not positive, or ``sigma_min`` is
                outside ``[0, 1)``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        mean = as_points([self.mean], name="mean")[0]
        scale = as_point(self.standard_deviation, len(mean), name="standard_deviation")
        if any(value <= 0.0 for value in scale):
            raise ValueError("standard_deviation must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", scale)
        object.__setattr__(self, "sigma_min", _sigma_min(self.sigma_min))

    @property
    def dimension(self) -> int:
        """Return the data dimension.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return len(self.mean)

    @property
    def terminal_variance(self) -> Point:
        """Return the variance ``sd^2 + s^2`` of the law reached at ``t = 1``.

        References:
            [EQ-GEN-FM-003].

        """
        return tuple(sd * sd + self.sigma_min**2 for sd in self.standard_deviation)

    def slope(self, t: float) -> Point:
        """Return ``c(t) = (t sd^2 - (1 - s) a(t)) / (a(t)^2 + t^2 sd^2)``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            One slope per coordinate.

        Raises:
            ValueError: If ``t`` is outside ``[0, 1]``.

        References:
            [EQ-GEN-FM-003].

        """
        time = _unit_time(t)
        shrink = 1.0 - self.sigma_min
        keep = 1.0 - shrink * time
        return tuple(
            (time * sd * sd - shrink * keep) / (keep * keep + time * time * sd * sd)
            for sd in self.standard_deviation
        )

    def __call__(self, t: float, x: Sequence[Any]) -> list[Any]:
        """Return ``v(t, x) = m + c(t) (x - t m)``.

        Args:
            t: Time in ``[0, 1]``.
            x: Point of the data dimension (floats or duals).

        Returns:
            The velocity, of the numeric kind of ``x``.

        Raises:
            ValueError: If ``t`` is out of range or ``x`` has the wrong
                length.

        References:
            [EQ-GEN-FM-003].

        """
        if len(x) != self.dimension:
            raise ValueError(f"x must have dimension {self.dimension}")
        slopes = self.slope(t)
        return [
            m + c * (value - t * m)
            for value, m, c in zip(x, self.mean, slopes, strict=True)
        ]

    def divergence(self, t: float, x: Sequence[float]) -> float:
        """Return ``div v(t, x) = sum_k c_k(t)`` (it does not depend on ``x``).

        Args:
            t: Time in ``[0, 1]``.
            x: Point of the data dimension (unused except for its length).

        Returns:
            The divergence.

        Raises:
            ValueError: If ``t`` is out of range or ``x`` has the wrong
                length.

        References:
            [EQ-GEN-FM-003], [EQ-GEN-FM-004].

        """
        if len(x) != self.dimension:
            raise ValueError(f"x must have dimension {self.dimension}")
        return math.fsum(self.slope(t))


def _ode_settings(steps: object, method: object) -> tuple[int, str]:
    """Validate the number of steps and the method of a flow integration.

    Raises:
        ValueError: If ``steps`` is not a positive integer or ``method`` is
            not one of :data:`FLOW_ODE_METHODS`.

    References:
        [EQ-NUM-ODE-001]. Project validation policy
        [REF-PROBSTAT-VALIDATION].

    """
    count = positive_integer(steps, name="steps")
    if method not in FLOW_ODE_METHODS:
        raise ValueError(f"method must be one of {FLOW_ODE_METHODS}, got {method!r}")
    return count, str(method)


def integrate_flow(
    velocity: VelocityField,
    x0: object,
    *,
    steps: int = 20,
    method: str = "rk4",
) -> Point:
    """Transport a base point by ``dx/dt = v(t, x)`` from ``t = 0`` to ``1``.

    The equation is integrated by
    :func:`probstat.numerics.integrate_fixed_step` with ``steps`` equal
    steps; there is no error control, so compare two step counts when the
    accuracy matters.

    Args:
        velocity: ``(t, x) -> v`` returning one value per coordinate.
        x0: Base point (a number in one dimension).
        steps: Positive number of steps.
        method: ``"euler"``, ``"heun"``, or ``"rk4"``.

    Returns:
        The transported point ``x(1)``.

    Raises:
        TypeError: If ``velocity`` is not callable or ``x0`` is not a point.
        ValueError: For invalid settings, a velocity of the wrong length,
            or a non-finite state.

    References:
        [EQ-GEN-FM-002], [EQ-NUM-ODE-001], [REF-LIPMAN-ETAL-2023],
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """
    if not callable(velocity):
        raise TypeError("velocity must be callable")
    count, name = _ode_settings(steps, method)
    start = as_points([x0], name="x0")[0]
    trajectory = integrate_fixed_step(
        velocity, 0.0, start, step=1.0 / count, steps=count, method=name
    )
    return tuple(float(value) for value in trajectory.final_state)


def _autodiff_divergence(velocity: VelocityField) -> DivergenceFunction:
    """Return ``(t, x) -> div v`` computed with forward-mode duals.

    Coordinate ``k`` of ``x`` is seeded with the ``k``-th unit tangent; the
    divergence is the sum of the matching tangent entries of the outputs.

    References:
        [EQ-GEN-FM-004], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

    """

    def divergence(t: float, x: Sequence[float]) -> float:
        """Return the trace of the Jacobian of ``v(t, .)`` at ``x``.

        References:
            [EQ-GEN-FM-004].

        """
        size = len(x)
        seeded = [
            Dual(value, tuple(1.0 if j == k else 0.0 for j in range(size)))
            for k, value in enumerate(x)
        ]
        outputs = velocity(t, seeded)
        if len(outputs) != size:
            raise ValueError("velocity must return one value per coordinate")
        total = 0.0
        for k, output in enumerate(outputs):
            if isinstance(output, Dual):
                total += float(output.tangent[k])
        return total

    return divergence


def _finite_difference_divergence(
    velocity: VelocityField, width: float
) -> DivergenceFunction:
    """Return ``(t, x) -> div v`` by central differences of half-width ``width``.

    The truncation error is ``O(width^2)`` and the rounding error
    ``O(eps / width)``.

    References:
        [EQ-GEN-FM-004], [EQ-OPT-FD-001].

    """

    def divergence(t: float, x: Sequence[float]) -> float:
        """Return the central-difference trace of the Jacobian at ``x``.

        References:
            [EQ-GEN-FM-004], [EQ-OPT-FD-001].

        """
        total = 0.0
        for k in range(len(x)):
            upper = list(x)
            lower = list(x)
            upper[k] += width
            lower[k] -= width
            total += (float(velocity(t, upper)[k]) - float(velocity(t, lower)[k])) / (
                2.0 * width
            )
        return total

    return divergence


def flow_log_density(
    velocity: VelocityField,
    x: object,
    *,
    steps: int = 20,
    method: str = "rk4",
    divergence: str | DivergenceFunction = "autodiff",
    difference_step: float = 1e-5,
) -> float:
    """Return ``log p_1(x)`` of the law transported from ``N(0, I)``.

    Along a solution of ``dx/dt = v(t, x)`` the density obeys
    ``d log p_t(x(t)) / dt = -div v(t, x(t))`` (instantaneous change of
    variables). The point is carried backward from ``t = 1`` to ``0`` while
    the divergence is accumulated, and

    ``log p_1(x) = log N(x(0); 0, I) - int_0^1 div v(t, x(t)) dt``.

    Both integrals use the same fixed-step method in the reversed time
    ``tau = 1 - t``; there is no error control.

    Args:
        velocity: ``(t, x) -> v``.
        x: Point at which the density is wanted.
        steps: Positive number of steps.
        method: ``"euler"``, ``"heun"``, or ``"rk4"``.
        divergence: ``"autodiff"`` (exact, forward-mode duals; the velocity
            must then be written with arithmetic operators or
            ``probstat.numerics.admath``), ``"finite-difference"`` (central
            differences), or a function ``(t, x) -> div v``.
        difference_step: Positive half-width of the central differences.

    Returns:
        The log-density at ``x``.

    Raises:
        TypeError: If ``velocity`` is not callable or ``x`` is not a point.
        ValueError: For invalid settings or a non-finite state.

    References:
        [EQ-GEN-FM-004] (derived there from the continuity equation),
        [REF-LIPMAN-ETAL-2023], [EQ-NUM-ODE-001].

    """
    if not callable(velocity):
        raise TypeError("velocity must be callable")
    count, name = _ode_settings(steps, method)
    point = as_points([x], name="x")[0]
    size = len(point)
    if callable(divergence):
        trace = divergence
    elif divergence == "autodiff":
        trace = _autodiff_divergence(velocity)
    elif divergence == "finite-difference":
        width = positive_real(difference_step, name="difference_step")
        trace = _finite_difference_divergence(velocity, width)
    else:
        raise ValueError(
            f"divergence must be callable or one of {DIVERGENCE_MODES}, "
            f"got {divergence!r}"
        )

    def reversed_rhs(tau: float, state: Sequence[float]) -> list[float]:
        """Return ``(-v, div v)`` at time ``t = 1 - tau``.

        References:
            [EQ-GEN-FM-004].

        """
        t = min(1.0, max(0.0, 1.0 - tau))
        position = state[:size]
        slope = [-float(value) for value in velocity(t, position)]
        if len(slope) != size:
            raise ValueError("velocity must return one value per coordinate")
        slope.append(float(trace(t, position)))
        return slope

    trajectory = integrate_fixed_step(
        reversed_rhs, 0.0, [*point, 0.0], step=1.0 / count, steps=count, method=name
    )
    final = trajectory.final_state
    return float(standard_normal_log_density(final[:size])) - float(final[size])


@dataclass(frozen=True)
class VelocityNetwork:
    """Velocity model ``v_theta(t, x, c)`` built on a ``tanh`` MLP.

    The network input is ``(x_1, ..., x_d, 2 t - 1, c_1, ..., c_k)`` and its
    output has ``d`` entries; ``c`` is an optional conditioning vector (the
    observation in posterior estimation).

    Args:
        dimension: Dimension ``d`` of the transported variable.
        condition_dimension: Length ``k >= 0`` of the conditioning vector.
        hidden: Hidden-layer widths (default two layers of 16).

    References:
        [EQ-GEN-FM-002], [EQ-NN-MLP-001], [REF-LIPMAN-ETAL-2023].

    """

    dimension: int = 1
    condition_dimension: int = 0
    hidden: tuple[int, ...] = (16, 16)
    network: MLP = field(init=False)

    def __post_init__(self) -> None:
        """Validate the sizes and build the network.

        Raises:
            ValueError: For a non-positive dimension, a negative condition
                dimension, or invalid hidden widths.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        dimension = positive_integer(self.dimension, name="dimension")
        extra = self.condition_dimension
        if isinstance(extra, bool) or not isinstance(extra, int) or extra < 0:
            raise ValueError("condition_dimension must be a non-negative integer")
        widths = tuple(positive_integer(w, name="hidden widths") for w in self.hidden)
        if not widths:
            raise ValueError("hidden needs at least one layer")
        object.__setattr__(self, "hidden", widths)
        object.__setattr__(
            self, "network", MLP((dimension + 1 + extra, *widths, dimension))
        )

    @property
    def parameter_count(self) -> int:
        """Return the number of network parameters.

        References:
            [EQ-NN-MLP-001].

        """
        return self.network.parameter_count

    def initial_parameters(self, seed: int | RandomStream) -> tuple[float, ...]:
        """Return Glorot-uniform initial parameters.

        Args:
            seed: Integer seed or stream.

        Returns:
            Flat parameter vector.

        References:
            [EQ-NN-GLOROT-001].

        """
        return self.network.initial_parameters(seed)

    def inputs(
        self, t: Any, x: Sequence[Any], condition: Sequence[Any] = ()
    ) -> list[Any]:
        """Return the network input ``(x, 2 t - 1, c)``.

        Raises:
            ValueError: If ``x`` or ``condition`` has the wrong length.

        References:
            [EQ-GEN-FM-002]. No new equation is introduced.

        """
        if len(x) != self.dimension or len(condition) != self.condition_dimension:
            raise ValueError("x or condition has the wrong length")
        return [*x, 2.0 * t - 1.0, *condition]

    def forward(
        self,
        parameters: Sequence[Any],
        t: Any,
        x: Sequence[Any],
        condition: Sequence[Any] = (),
    ) -> list[Any]:
        """Evaluate the model on any numeric kind (floats, duals, tape).

        Args:
            parameters: Flat parameter vector.
            t: Time in ``[0, 1]``.
            x: Point of dimension ``d``.
            condition: Conditioning vector of length ``k``.

        Returns:
            ``v_theta(t, x, c)``.

        Raises:
            ValueError: If a length does not match the architecture.

        References:
            [EQ-GEN-FM-002], [EQ-NN-MLP-001].

        """
        return self.network.forward(parameters, self.inputs(t, x, condition))

    def velocity(
        self,
        parameters: Sequence[float],
        t: float,
        x: Sequence[float],
        condition: Sequence[float] = (),
    ) -> list[float]:
        """Evaluate the model on floats (fast path).

        Args:
            parameters: Flat parameter vector.
            t: Time in ``[0, 1]``.
            x: Point of dimension ``d``.
            condition: Conditioning vector of length ``k``.

        Returns:
            ``v_theta(t, x, c)``.

        Raises:
            ValueError: If a length does not match the architecture.

        References:
            [EQ-GEN-FM-002], [EQ-NN-MLP-001].

        """
        layers = _backprop.split_layers(self.network.layer_sizes, parameters)
        return _backprop.forward(layers, self.inputs(t, x, condition))

    def divergence(
        self,
        parameters: Sequence[float],
        t: float,
        x: Sequence[float],
        condition: Sequence[float] = (),
    ) -> float:
        """Return the exact divergence ``sum_k d v_k / d x_k``.

        The Jacobian diagonal is propagated forward through the layers
        (forward-mode differentiation), so the value is exact up to
        rounding; no finite differences are involved.

        Args:
            parameters: Flat parameter vector.
            t: Time in ``[0, 1]``.
            x: Point of dimension ``d``.
            condition: Conditioning vector of length ``k``.

        Returns:
            The divergence with respect to ``x``.

        Raises:
            ValueError: If a length does not match the architecture.

        References:
            [EQ-GEN-FM-004], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        layers = _backprop.split_layers(self.network.layer_sizes, parameters)
        return _backprop.forward_with_divergence(
            layers, self.inputs(t, x, condition), self.dimension
        )[1]


def flow_matching_loss_and_gradient(
    network: VelocityNetwork,
    parameters: Sequence[float],
    cases: Sequence[tuple[float, Sequence[float], Sequence[float], Sequence[float]]],
) -> tuple[float, list[float]]:
    """Return the flow-matching loss of a fixed batch and its gradient.

    Each case is ``(t, x_t, u, c)``; the loss is the batch mean of
    ``|v_theta(t, x_t, c) - u|^2`` and the gradient is computed by
    backpropagation on floats.

    Args:
        network: Velocity model.
        parameters: Flat parameter vector.
        cases: Non-empty batch of ``(t, x_t, u, c)`` tuples.

    Returns:
        ``(loss, gradient)``.

    Raises:
        ValueError: If the batch is empty or a length is wrong.

    References:
        [EQ-GEN-FM-002], [EQ-GEN-FM-005], [REF-LIPMAN-ETAL-2023].

    """
    if not cases:
        raise ValueError("cases must not be empty")
    batch = [network.inputs(t, x, c) for t, x, _, c in cases]

    def squared_error(case: int, output: list[float]) -> tuple[float, list[float]]:
        """Return ``|v - u|^2`` and its gradient ``2 (v - u)``.

        References:
            [EQ-GEN-FM-002].

        """
        residual = [v - u for v, u in zip(output, cases[case][2], strict=True)]
        return sum(r * r for r in residual), [2.0 * r for r in residual]

    return _backprop.batch_loss_and_gradient(
        network.network.layer_sizes, parameters, batch, squared_error
    )


def train_velocity_network(
    network: VelocityNetwork,
    targets: Sequence[Sequence[float]],
    conditions: Sequence[Sequence[float]] | None,
    *,
    sigma_min: float,
    iterations: int,
    batch_size: int,
    learning_rate: float,
    tail_fraction: float,
    stream: RandomStream,
) -> AdamResult:
    """Minimize the conditional flow-matching loss by Adam.

    Every iteration draws ``batch_size`` rows without replacement (the whole
    set when it is smaller), times ``t ~ U(0, 1)``, and base points
    ``x_0 ~ N(0, I)``, forms ``x_t`` and ``u`` by [EQ-GEN-FM-001], and takes
    one Adam step on the mean squared error [EQ-GEN-FM-002]. Shared by
    :func:`fit_flow_matching` and
    :func:`probstat.generative.fit_flow_matching_posterior`; the inputs are
    assumed validated and standardized by the caller.

    Args:
        network: Velocity model.
        targets: Standardized end points ``x_1`` (data or parameters).
        conditions: One conditioning vector per target, or ``None``.
        sigma_min: Path width ``s``.
        iterations: Adam iterations.
        batch_size: Mini-batch size.
        learning_rate: Adam step size.
        tail_fraction: Fraction of final iterates averaged.
        stream: Random stream (network initialization and batches).

    Returns:
        The Adam result with the loss trace.

    Raises:
        ValueError: For invalid optimizer settings.

    References:
        [EQ-GEN-FM-001], [EQ-GEN-FM-002], [EQ-OPT-ADAM-002],
        [REF-LIPMAN-ETAL-2023], [REF-KINGMA-BA-2015].

    """
    count = len(targets)
    take = min(batch_size, count)
    dimension = network.dimension
    shrink = 1.0 - sigma_min
    empty: tuple[float, ...] = ()

    def objective(
        parameters: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, Sequence[float]]:
        """Return the mini-batch flow-matching loss and its gradient.

        References:
            [EQ-GEN-FM-002], [EQ-GEN-FM-005].

        """
        generator = batch_stream.generator()
        cases = []
        for row in generator.sample(range(count), take):
            t = generator.random()
            keep = 1.0 - shrink * t
            end = targets[row]
            base = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
            cases.append(
                (
                    t,
                    [keep * a + t * b for a, b in zip(base, end, strict=True)],
                    [b - shrink * a for a, b in zip(base, end, strict=True)],
                    empty if conditions is None else conditions[row],
                )
            )
        return flow_matching_loss_and_gradient(network, parameters, cases)

    return adam_minimize(
        objective,
        network.initial_parameters(stream.child("init")),
        iterations=iterations,
        learning_rate=learning_rate,
        tail_fraction=tail_fraction,
        seed=stream.child("adam"),
    )


def loss_window_means(losses: Sequence[float]) -> tuple[float, float]:
    """Return the mean loss over the first and the last tenth of a trace.

    Args:
        losses: Non-empty loss trace.

    Returns:
        ``(initial mean, final mean)``; windows hold at least one value.

    References:
        Project reporting policy [REF-PROBSTAT-REPORTING].
        No equation is evaluated.

    """
    window = max(1, len(losses) // 10)
    return (
        math.fsum(losses[:window]) / window,
        math.fsum(losses[-window:]) / window,
    )


def standardize_columns(
    rows: Sequence[Sequence[float]],
) -> tuple[Point, Point, list[tuple[float, ...]]]:
    """Return column means, standard deviations, and the standardized rows.

    A constant column gets scale 1.

    Args:
        rows: Non-empty rows of one width.

    Returns:
        ``(location, scale, standardized rows)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    columns = list(zip(*rows, strict=True))
    count = len(rows)
    location = tuple(math.fsum(column) / count for column in columns)
    scale = tuple(
        math.sqrt(math.fsum((v - m) ** 2 for v in column) / count) or 1.0
        for column, m in zip(columns, location, strict=True)
    )
    standardized = [
        tuple((v - m) / s for v, m, s in zip(row, location, scale, strict=True))
        for row in rows
    ]
    return location, scale, standardized


@dataclass(frozen=True)
class FlowMatchingFit:
    """Velocity network fitted by conditional flow matching.

    The flow acts on standardized data ``z = (x - location) / scale``; the
    methods map back to the original coordinates. The loss does not tend to
    zero even for a perfect fit, because ``u`` keeps its conditional
    variance given ``x_t``. Adam has no convergence test: ``loss_decreased``
    is a necessary sanity check, and the fit must be judged by comparing
    samples or held-out log-densities with the data.

    Args:
        network: Velocity model.
        parameters: Fitted flat parameters.
        location: Training means of the coordinates.
        scale: Training standard deviations of the coordinates.
        sigma_min: Path width ``s``; the model targets the standardized data
            convolved with ``N(0, s^2 I)``.
        losses: Mini-batch loss at every Adam iteration.
        initial_loss: Mean loss over the first tenth of the iterations.
        final_loss: Mean loss over the last tenth of the iterations.
        iterations: Adam iterations (one batch gradient each).
        batch_size: Mini-batch size used.
        training_points: Number of data points.

    References:
        [EQ-GEN-FM-001], [EQ-GEN-FM-002], [REF-LIPMAN-ETAL-2023].

    """

    network: VelocityNetwork
    parameters: tuple[float, ...]
    location: Point
    scale: Point
    sigma_min: float
    losses: tuple[float, ...]
    initial_loss: float
    final_loss: float
    iterations: int
    batch_size: int
    training_points: int

    @property
    def dimension(self) -> int:
        """Return the data dimension.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.network.dimension

    @property
    def loss_decreased(self) -> bool:
        """Return whether the final loss window is below the initial one.

        References:
            Project reporting policy [REF-PROBSTAT-REPORTING].
            No equation is evaluated.

        """
        return self.final_loss < self.initial_loss

    def velocity(self, t: float, z: Sequence[float]) -> list[float]:
        """Return the fitted velocity at a standardized point.

        Args:
            t: Time in ``[0, 1]``.
            z: Standardized point.

        Returns:
            ``v_theta(t, z)``.

        Raises:
            ValueError: If ``z`` has the wrong length.

        References:
            [EQ-GEN-FM-002].

        """
        return self.network.velocity(self.parameters, t, z)

    def transport(self, base: object, *, steps: int = 20, method: str = "rk4") -> Point:
        """Map one base point ``z_0`` to data coordinates.

        Args:
            base: Point of ``N(0, I)`` (a number in one dimension).
            steps: Positive number of ODE steps.
            method: ``"euler"``, ``"heun"``, or ``"rk4"``.

        Returns:
            ``location + scale * z(1)``.

        Raises:
            ValueError: If the point or a setting is invalid.

        References:
            [EQ-GEN-FM-002], [EQ-NUM-ODE-001].

        """
        start = as_point(base, self.dimension, name="base")
        end = integrate_flow(self.velocity, start, steps=steps, method=method)
        return tuple(
            m + s * z for m, s, z in zip(self.location, self.scale, end, strict=True)
        )

    def sample(
        self,
        n: int,
        *,
        steps: int = 20,
        method: str = "rk4",
        seed: int | RandomStream = 0,
    ) -> tuple[Point, ...]:
        """Draw ``n`` points by integrating the fitted flow.

        Args:
            n: Positive number of draws.
            steps: Positive number of ODE steps per draw.
            method: ``"euler"``, ``"heun"``, or ``"rk4"``.
            seed: Integer seed or stream for the base draws.

        Returns:
            ``n`` points in data coordinates.

        Raises:
            ValueError: If ``n`` or a setting is invalid.

        References:
            [EQ-GEN-FM-002], [EQ-NUM-ODE-001], [REF-LIPMAN-ETAL-2023].

        """
        count = positive_integer(n, name="n")
        generator = resolve(seed).child("flow-matching-sample").generator()
        return tuple(
            self.transport(
                [generator.gauss(0.0, 1.0) for _ in range(self.dimension)],
                steps=steps,
                method=method,
            )
            for _ in range(count)
        )

    def log_density(self, x: object, *, steps: int = 20, method: str = "rk4") -> float:
        """Return the model log-density at a data point.

        The standardized point is carried back to ``t = 0`` with the exact
        network divergence (forward-mode differentiation, no finite
        differences) and ``sum_k log scale_k`` is subtracted for the
        standardization.

        Args:
            x: Data point.
            steps: Positive number of ODE steps.
            method: ``"euler"``, ``"heun"``, or ``"rk4"``.

        Returns:
            ``log p(x)`` of the fitted flow, up to the ODE discretization.

        Raises:
            ValueError: If the point or a setting is invalid.

        References:
            [EQ-GEN-FM-004], [REF-LIPMAN-ETAL-2023].

        """
        point = as_point(x, self.dimension)
        z = [
            (v - m) / s
            for v, m, s in zip(point, self.location, self.scale, strict=True)
        ]

        def trace(t: float, position: Sequence[float]) -> float:
            """Return the exact divergence of the fitted field.

            References:
                [EQ-GEN-FM-004].

            """
            return self.network.divergence(self.parameters, t, position)

        value = flow_log_density(
            self.velocity, z, steps=steps, method=method, divergence=trace
        )
        return value - math.fsum(math.log(s) for s in self.scale)


def fit_flow_matching(
    data: object,
    *,
    hidden: Sequence[int] = (16, 16),
    sigma_min: float = 0.0,
    iterations: int = 1500,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    tail_fraction: float = 0.2,
    seed: int | RandomStream = 0,
) -> FlowMatchingFit:
    """Fit a velocity network to data by conditional flow matching.

    The data are standardized coordinate by coordinate, and Adam minimizes
    the mini-batch mean of ``|v_theta(t, x_t) - (x_1 - (1 - s) x_0)|^2``
    with ``x_t = (1 - (1 - s) t) x_0 + t x_1``, ``t ~ U(0, 1)``,
    ``x_0 ~ N(0, I)``, and ``x_1`` drawn from the data without replacement.
    Base points and data are paired independently (no minibatch optimal
    transport).

    Args:
        data: At least two points (numbers in one dimension).
        hidden: Hidden-layer widths of the ``tanh`` network.
        sigma_min: Path width ``s`` in ``[0, 1)`` (in standardized units).
        iterations: Positive number of Adam iterations.
        batch_size: Positive mini-batch size (capped at the data size).
        learning_rate: Positive Adam step size.
        tail_fraction: Fraction in ``[0, 1)`` of final iterates averaged.
        seed: Integer seed or stream (initialization and batches).

    Returns:
        :class:`FlowMatchingFit` with the loss trace.

    Raises:
        TypeError: If the data are not points or the seed is invalid.
        ValueError: For fewer than two points, non-finite data, or invalid
            settings.

    References:
        [EQ-GEN-FM-001], [EQ-GEN-FM-002], [REF-LIPMAN-ETAL-2023],
        [REF-KINGMA-BA-2015].

    """
    points = as_points(data, minimum=2)
    steps = positive_integer(iterations, name="iterations")
    batch = positive_integer(batch_size, name="batch_size")
    rate = positive_real(learning_rate, name="learning_rate")
    width = _sigma_min(sigma_min)
    network = VelocityNetwork(len(points[0]), 0, tuple(hidden))
    location, scale, standardized = standardize_columns(points)
    result = train_velocity_network(
        network,
        standardized,
        None,
        sigma_min=width,
        iterations=steps,
        batch_size=batch,
        learning_rate=rate,
        tail_fraction=tail_fraction,
        stream=resolve(seed).child("flow-matching"),
    )
    initial_loss, final_loss = loss_window_means(result.losses)
    return FlowMatchingFit(
        network=network,
        parameters=result.x,
        location=location,
        scale=scale,
        sigma_min=width,
        losses=result.losses,
        initial_loss=initial_loss,
        final_loss=final_loss,
        iterations=steps,
        batch_size=min(batch, len(points)),
        training_points=len(points),
    )
