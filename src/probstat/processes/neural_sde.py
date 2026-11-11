"""Neural stochastic differential equations fitted by Euler pseudo-likelihood.

A :class:`NeuralSDE` is the autonomous diagonal-noise Ito equation
``dX_j = a_j(X) dt + b_j(X) dW_j`` whose drift ``a`` is a
:class:`~probstat.numerics.MLP` and whose diffusion ``b`` is either a
constant vector or a second MLP with a softplus output [EQ-PROC-NSDE-002].
Over a short step ``h`` the Euler-Maruyama scheme makes each transition
Gaussian, ``X_(k+1) | X_k ~ N(X_k + a(X_k) h, diag(b(X_k)^2) h)``; the sum
of these log-densities over observed transitions is the Euler
pseudo-likelihood [EQ-PROC-NSDE-001]. :func:`fit_neural_sde` maximizes it
with reverse-mode gradients and Adam. The pseudo-likelihood is biased by
``O(h)``, so it is meant for densely observed trajectories.

A fitted one-dimensional model converts to a :class:`ScalarSDE` and is
simulated with the existing Euler-Maruyama and Milstein schemes; higher
dimensions use the componentwise Euler-Maruyama step [EQ-PROC-SDE-001].

References:
    [REF-KLOEDEN-PLATEN-1992, Section 10.2], [REF-IACUS-2008, Chapter 3],
    [REF-GOODFELLOW-BENGIO-COURVILLE-2016, Chapter 6].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.numerics import MLP, admath, derivative, reverse_value_and_gradient
from probstat.optimization import AdamResult, adam_minimize

from .continuous import SamplePath, _positive, _positive_integer, uniform_grid
from .sde import SCHEMES, ScalarSDE, solve_sde

DIFFUSION_MODELS: tuple[str, ...] = ("constant", "mlp")

State = tuple[float, ...]
Transition = tuple[State, State, float]


@dataclass(frozen=True, slots=True)
class VectorSamplePath:
    """One realization of a vector-valued process on a time grid.

    Args:
        times: Increasing times starting at the initial time.
        states: States at those times, each a tuple of coordinates.

    References:
        [REF-KLOEDEN-PLATEN-1992, Section 10.2].

    """

    times: tuple[float, ...]
    states: tuple[State, ...]

    @property
    def terminal(self) -> State:
        """Return the state at the final time.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.states[-1]

    def coordinate(self, index: int) -> SamplePath:
        """Return one coordinate as a scalar :class:`SamplePath`.

        Args:
            index: Coordinate index.

        Returns:
            Scalar path of that coordinate.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return SamplePath(self.times, tuple(state[index] for state in self.states))


@dataclass(frozen=True)
class NeuralSDE:
    """Diagonal-noise SDE with an MLP drift and a constant or MLP diffusion.

    With input standardization ``u = (x - shift) / scale`` the coefficients
    are ``a_j(x) = scale_j N_j(u)`` and either ``b_j = exp(rho_j)``
    (``diffusion="constant"``) or ``b_j(x) = b_min + softplus(M_j(u))``
    (``diffusion="mlp"``) [EQ-PROC-NSDE-002]. Parameters form one flat vector:
    drift network, then ``rho`` or the diffusion network.

    Args:
        dimension: State dimension ``d``.
        hidden: Hidden widths of the drift network.
        activation: Hidden activation of both networks.
        diffusion: ``"constant"`` or ``"mlp"``.
        diffusion_hidden: Hidden widths of the diffusion network.
        minimum_diffusion: Positive floor ``b_min`` of an MLP diffusion.
        shift: Input centring, one value per coordinate (``None``: zeros).
        scale: Positive input scaling, one value per coordinate (``None``:
            ones).

    References:
        [EQ-PROC-NSDE-002], [EQ-NN-MLP-001].

    """

    dimension: int = 1
    hidden: tuple[int, ...] = (16,)
    activation: str = "tanh"
    diffusion: str = "constant"
    diffusion_hidden: tuple[int, ...] = (8,)
    minimum_diffusion: float = 1e-4
    shift: tuple[float, ...] | None = None
    scale: tuple[float, ...] | None = None
    drift_network: MLP = field(init=False, repr=False)
    diffusion_network: MLP | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the specification and build the networks.

        Raises:
            ValueError: For an invalid dimension, diffusion model, floor,
                shift, or scale.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        d = _positive_integer(self.dimension, name="dimension")
        if self.diffusion not in DIFFUSION_MODELS:
            raise ValueError(f"diffusion must be one of {DIFFUSION_MODELS}")
        _positive(self.minimum_diffusion, name="minimum_diffusion")
        shift = (0.0,) * d if self.shift is None else tuple(
            require_finite_real(v, name="shift") for v in self.shift
        )
        scale = (1.0,) * d if self.scale is None else tuple(
            _positive(v, name="scale") for v in self.scale
        )
        if len(shift) != d or len(scale) != d:
            raise ValueError("shift and scale must have one entry per coordinate")
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "hidden", tuple(self.hidden))
        object.__setattr__(self, "diffusion_hidden", tuple(self.diffusion_hidden))
        object.__setattr__(
            self, "drift_network", MLP((d, *self.hidden, d), activation=self.activation)
        )
        object.__setattr__(
            self,
            "diffusion_network",
            MLP(
                (d, *self.diffusion_hidden, d),
                activation=self.activation,
                output_activation="softplus",
            )
            if self.diffusion == "mlp"
            else None,
        )

    @property
    def parameter_count(self) -> int:
        """Return the total number of drift and diffusion parameters.

        References:
            [EQ-PROC-NSDE-002].

        """
        extra = (
            self.dimension
            if self.diffusion_network is None
            else self.diffusion_network.parameter_count
        )
        return self.drift_network.parameter_count + extra

    def _standardized(self, x: Sequence[Any]) -> list[Any]:
        """Return ``(x - shift) / scale``.

        Args:
            x: State (floats or differentiable variables).

        Returns:
            Standardized network input.

        Raises:
            ValueError: If the state has the wrong length.

        References:
            [EQ-PROC-NSDE-002].

        """
        if len(x) != self.dimension:
            raise ValueError(f"expected a state of dimension {self.dimension}")
        assert self.shift is not None and self.scale is not None
        return [(v - c) / s for v, c, s in zip(x, self.shift, self.scale, strict=True)]

    def initial_parameters(
        self,
        seed: int | RandomStream,
        *,
        diffusion_level: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """Return initial parameters with a prescribed starting diffusion.

        The drift network starts from Glorot weights; the diffusion starts at
        ``diffusion_level`` (ones by default): ``rho = log level`` for a
        constant diffusion, or a near-zero MLP whose output bias gives
        ``b_min + softplus(bias) = level``.

        Args:
            seed: Integer seed or stream.
            diffusion_level: Positive starting diffusion per coordinate.

        Returns:
            Flat parameter vector.

        Raises:
            ValueError: If a level is not above ``minimum_diffusion``.

        References:
            [EQ-NN-GLOROT-001], [EQ-PROC-NSDE-002].

        """
        stream = resolve_stream(seed)
        levels = (
            (1.0,) * self.dimension
            if diffusion_level is None
            else tuple(_positive(v, name="diffusion_level") for v in diffusion_level)
        )
        if len(levels) != self.dimension:
            raise ValueError("diffusion_level needs one entry per coordinate")
        drift = self.drift_network.initial_parameters(stream.child("drift"))
        if self.diffusion_network is None:
            return drift + tuple(math.log(v) for v in levels)
        if any(v <= self.minimum_diffusion for v in levels):
            raise ValueError("diffusion_level must exceed minimum_diffusion")
        network = list(
            self.diffusion_network.initial_parameters(
                stream.child("diffusion"), output_scale=0.1
            )
        )
        for j, level in enumerate(levels):
            excess = level - self.minimum_diffusion
            bias = excess + math.log(-math.expm1(-excess))
            network[len(network) - self.dimension + j] = bias
        return drift + tuple(network)

    def _split(self, parameters: Sequence[Any]) -> tuple[Sequence[Any], Sequence[Any]]:
        """Split the flat vector into drift and diffusion parts.

        Args:
            parameters: Flat parameter vector.

        Returns:
            ``(drift parameters, diffusion parameters)``.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-PROC-NSDE-002].

        """
        if len(parameters) != self.parameter_count:
            raise ValueError(
                f"expected {self.parameter_count} parameters, got {len(parameters)}"
            )
        cut = self.drift_network.parameter_count
        return parameters[:cut], parameters[cut:]

    def drift(self, parameters: Sequence[Any], x: Sequence[Any]) -> list[Any]:
        """Return the drift ``a(x) = scale * N(u)``.

        Args:
            parameters: Flat parameter vector (floats or tape variables).
            x: State.

        Returns:
            Drift vector.

        References:
            [EQ-PROC-NSDE-002].

        """
        drift_parameters, _ = self._split(parameters)
        output = self.drift_network.forward(drift_parameters, self._standardized(x))
        assert self.scale is not None
        return [s * v for s, v in zip(self.scale, output, strict=True)]

    def diffusion_coefficients(
        self, parameters: Sequence[Any], x: Sequence[Any]
    ) -> list[Any]:
        """Return the diagonal diffusion ``b(x)``.

        Args:
            parameters: Flat parameter vector.
            x: State.

        Returns:
            Positive diffusion coefficients.

        References:
            [EQ-PROC-NSDE-002].

        """
        _, rest = self._split(parameters)
        if self.diffusion_network is None:
            if len(x) != self.dimension:
                raise ValueError(f"expected a state of dimension {self.dimension}")
            return [admath.exp(rho) for rho in rest]
        output = self.diffusion_network.forward(rest, self._standardized(x))
        return [self.minimum_diffusion + v for v in output]

    def transition_negative_log_likelihood(
        self,
        parameters: Sequence[Any],
        x: Sequence[Any],
        x_next: Sequence[Any],
        step: float,
    ) -> Any:
        """Return ``-log N(x_next; x + a(x) h, diag(b(x)^2) h)``.

        Args:
            parameters: Flat parameter vector.
            x: State at the start of the step.
            x_next: Observed state after ``step``.
            step: Positive time step ``h``.

        Returns:
            Negative Euler log-density of one transition.

        References:
            [EQ-PROC-NSDE-001], [REF-IACUS-2008, Chapter 3].

        """
        a = self.drift(parameters, x)
        b = self.diffusion_coefficients(parameters, x)
        total: Any = 0.0
        for xj, yj, aj, bj in zip(x, x_next, a, b, strict=True):
            variance = bj * bj * step
            residual = yj - xj - aj * step
            total = total + 0.5 * (
                admath.log(2.0 * math.pi * variance) + residual * residual / variance
            )
        return total


def _path_transitions(
    path: SamplePath | VectorSamplePath | Sequence[float | Sequence[float]],
    step: float | None,
    dimension: int,
) -> list[Transition]:
    """Return the consecutive transitions ``(x_k, x_(k+1), h_k)`` of one path.

    Args:
        path: :class:`SamplePath`, :class:`VectorSamplePath`, or a sequence of
            states observed every ``step``.
        step: Observation interval for plain sequences.
        dimension: Expected state dimension.

    Returns:
        List of transitions.

    Raises:
        ValueError: For a missing step, a wrong dimension, non-increasing
            times, or non-finite values.

    References:
        [EQ-PROC-NSDE-001]. No further equation is evaluated.

    """
    if isinstance(path, SamplePath):
        times: Sequence[float] = path.times
        states: list[State] = [
            (require_finite_real(v, name="path value"),) for v in path.values
        ]
    elif isinstance(path, VectorSamplePath):
        times = path.times
        states = [
            tuple(require_finite_real(v, name="path value") for v in s)
            for s in path.states
        ]
    else:
        if step is None:
            raise ValueError("step is required for paths given as plain sequences")
        h = _positive(step, name="step")
        states = [
            (require_finite_real(s, name="path value"),)
            if isinstance(s, int | float)
            else tuple(require_finite_real(v, name="path value") for v in s)
            for s in path
        ]
        times = [k * h for k in range(len(states))]
    if any(len(s) != dimension for s in states):
        raise ValueError(f"every state must have dimension {dimension}")
    transitions: list[Transition] = []
    for k in range(len(states) - 1):
        h_k = times[k + 1] - times[k]
        if not h_k > 0.0:
            raise ValueError("observation times must be strictly increasing")
        transitions.append((states[k], states[k + 1], h_k))
    return transitions


@dataclass(frozen=True)
class NeuralSDEFit:
    """A :class:`NeuralSDE` with parameters fitted by pseudo-likelihood.

    Args:
        model: Fitted architecture (with the input standardization used).
        parameters: Fitted flat parameters.
        optimization: Adam trace of the mean negative pseudo-log-likelihood.
        transitions: Number of observed transitions.
        mean_negative_log_likelihood: Mean negative Euler log-density per
            transition at the fitted parameters over all data.

    References:
        [EQ-PROC-NSDE-001], [REF-IACUS-2008, Chapter 3].

    """

    model: NeuralSDE
    parameters: tuple[float, ...]
    optimization: AdamResult
    transitions: int
    mean_negative_log_likelihood: float

    def drift(self, x: float | Sequence[float]) -> State:
        """Return the fitted drift at one state.

        Args:
            x: State (a number in one dimension).

        Returns:
            Drift vector.

        References:
            [EQ-PROC-NSDE-002].

        """
        state = (x,) if isinstance(x, int | float) else tuple(x)
        return tuple(float(v) for v in self.model.drift(self.parameters, state))

    def diffusion(self, x: float | Sequence[float]) -> State:
        """Return the fitted diagonal diffusion at one state.

        Args:
            x: State (a number in one dimension).

        Returns:
            Diffusion vector.

        References:
            [EQ-PROC-NSDE-002].

        """
        state = (x,) if isinstance(x, int | float) else tuple(x)
        return tuple(
            float(v) for v in self.model.diffusion_coefficients(self.parameters, state)
        )

    def drift_on_grid(
        self, grid: Sequence[float | Sequence[float]]
    ) -> tuple[tuple[State, State], ...]:
        """Report the fitted drift at every grid point.

        Args:
            grid: States (numbers in one dimension).

        Returns:
            Pairs ``(state, drift)``.

        References:
            [EQ-PROC-NSDE-002].

        """
        rows: list[tuple[State, State]] = []
        for point in grid:
            state = (float(point),) if isinstance(point, int | float) else tuple(point)
            rows.append((state, self.drift(state)))
        return tuple(rows)

    def to_scalar_sde(self) -> ScalarSDE:
        """Return the fitted one-dimensional model as a :class:`ScalarSDE`.

        The diffusion derivative (needed by Milstein) is zero for a constant
        diffusion and is computed by forward-mode differentiation otherwise.

        Returns:
            Autonomous scalar SDE.

        Raises:
            ValueError: If the model is not one-dimensional.

        References:
            [EQ-PROC-SDE-001], [EQ-PROC-SDE-002], [EQ-AD-DUAL-001].

        """
        if self.model.dimension != 1:
            raise ValueError("to_scalar_sde requires a one-dimensional model")
        model, theta = self.model, self.parameters

        def a(t: float, x: float) -> float:
            """Return the fitted drift.

            References:
                [EQ-PROC-NSDE-002].

            """
            return float(model.drift(theta, (x,))[0])

        def b(t: float, x: float) -> float:
            """Return the fitted diffusion.

            References:
                [EQ-PROC-NSDE-002].

            """
            return float(model.diffusion_coefficients(theta, (x,))[0])

        def b_x(t: float, x: float) -> float:
            """Return the derivative of the fitted diffusion.

            References:
                [EQ-AD-DUAL-001].

            """
            if model.diffusion_network is None:
                return 0.0
            return derivative(
                lambda value: model.diffusion_coefficients(theta, (value,))[0], x
            )

        return ScalarSDE(a, b, b_x)

    def simulate(
        self,
        initial: float | Sequence[float],
        horizon: float,
        steps: int,
        *,
        scheme: str = "euler_maruyama",
        seed: int | RandomStream | None = None,
    ) -> VectorSamplePath:
        """Simulate the fitted SDE on a uniform grid.

        One-dimensional models run through :func:`solve_sde` (Euler-Maruyama
        or Milstein); higher dimensions use the componentwise Euler-Maruyama
        step with independent Brownian increments.

        Args:
            initial: Initial state.
            horizon: Positive final time.
            steps: Positive number of steps.
            scheme: ``"euler_maruyama"`` or (one dimension) ``"milstein"``.
            seed: Integer seed or stream.

        Returns:
            :class:`VectorSamplePath`.

        Raises:
            ValueError: For invalid settings or a diverging path.

        Equation:
            [EQ-PROC-SDE-001], [EQ-PROC-SDE-002].

        References:
            [REF-KLOEDEN-PLATEN-1992, Sections 10.2-10.3].

        """
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        start = (
            (require_finite_real(initial, name="initial"),)
            if isinstance(initial, int | float)
            else tuple(require_finite_real(v, name="initial") for v in initial)
        )
        if len(start) != self.model.dimension:
            raise ValueError(f"initial must have dimension {self.model.dimension}")
        if self.model.dimension == 1:
            path = solve_sde(
                self.to_scalar_sde(), start[0], horizon, steps, scheme=scheme, seed=seed
            )
            return VectorSamplePath(path.times, tuple((v,) for v in path.values))
        if scheme != "euler_maruyama":
            raise ValueError("only euler_maruyama is available in several dimensions")
        end = _positive(horizon, name="horizon")
        count = _positive_integer(steps, name="steps")
        h = end / count
        root = math.sqrt(h)
        generator = resolve_stream(seed).child("neural_sde").generator()
        states = [start]
        x = start
        for _ in range(count):
            a = self.drift(x)
            b = self.diffusion(x)
            x = tuple(
                xj + aj * h + bj * root * generator.gauss(0.0, 1.0)
                for xj, aj, bj in zip(x, a, b, strict=True)
            )
            if not all(math.isfinite(v) for v in x):
                raise ValueError("simulation produced a non-finite value")
            states.append(x)
        return VectorSamplePath(uniform_grid(end, count), tuple(states))


def fit_neural_sde(
    model: NeuralSDE,
    paths: Sequence[SamplePath | VectorSamplePath | Sequence[float | Sequence[float]]],
    *,
    step: float | None = None,
    iterations: int = 600,
    batch_size: int = 128,
    learning_rate: float = 0.02,
    tail_fraction: float = 0.25,
    standardize: bool = True,
    seed: int | RandomStream = 0,
) -> NeuralSDEFit:
    """Fit a neural SDE to observed trajectories by Euler pseudo-likelihood.

    Every consecutive pair of observations is one transition. Each Adam
    iteration draws a mini-batch of transitions (with replacement) and
    descends the mean negative Euler log-density [EQ-PROC-NSDE-001]. With
    ``standardize`` the network inputs are centred and scaled by the mean
    and standard deviation of the observed states, and the diffusion starts
    at the quadratic-variation level ``sqrt(mean(dx^2 / h))`` per coordinate.

    Args:
        model: Architecture; its ``shift`` and ``scale`` are replaced when
            ``standardize`` is true.
        paths: Observed trajectories: :class:`SamplePath`,
            :class:`VectorSamplePath`, or sequences of states sampled every
            ``step``.
        step: Observation interval for plain sequences.
        iterations: Positive number of Adam updates.
        batch_size: Positive mini-batch size.
        learning_rate: Adam step size.
        tail_fraction: Fraction of final iterates averaged by Adam.
        standardize: Whether to standardize inputs from the data.
        seed: Integer seed or stream.

    Returns:
        :class:`NeuralSDEFit`.

    Raises:
        ValueError: If there are no transitions or a setting is invalid.

    Equation:
        [EQ-PROC-NSDE-001].

    References:
        [REF-IACUS-2008, Chapter 3], [REF-KLOEDEN-PLATEN-1992, Section 10.2],
        [REF-KINGMA-BA-2015].

    """
    d = model.dimension
    transitions: list[Transition] = []
    for path in paths:
        transitions.extend(_path_transitions(path, step, d))
    if not transitions:
        raise ValueError("paths must contain at least one transition")
    steps = _positive_integer(iterations, name="iterations")
    batch = _positive_integer(batch_size, name="batch_size")
    count = len(transitions)
    if standardize:
        states = [x for x, _, _ in transitions] + [transitions[-1][1]]
        means = [math.fsum(s[j] for s in states) / len(states) for j in range(d)]
        spreads = [
            math.sqrt(math.fsum((s[j] - means[j]) ** 2 for s in states) / len(states))
            for j in range(d)
        ]
        model = replace(
            model,
            shift=tuple(means),
            scale=tuple(v if v > 0.0 else 1.0 for v in spreads),
        )
    levels = [
        math.sqrt(math.fsum((y[j] - x[j]) ** 2 / h for x, y, h in transitions) / count)
        for j in range(d)
    ]
    floor = 2.0 * model.minimum_diffusion
    levels = [v if v > floor else floor for v in levels]
    stream = resolve_stream(seed)
    x0 = model.initial_parameters(stream.child("init"), diffusion_level=levels)
    fitted = model

    def objective(
        theta: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, Sequence[float]]:
        """Return the mini-batch mean negative pseudo-log-likelihood and gradient.

        Args:
            theta: Current parameters.
            batch_stream: Stream for this iteration's batch.

        Returns:
            ``(loss, gradient)``.

        References:
            [EQ-PROC-NSDE-001], [EQ-AD-REVERSE-001].

        """
        generator = batch_stream.generator()
        chosen = [transitions[generator.randrange(count)] for _ in range(batch)]

        def loss(parameters: Sequence[Any]) -> Any:
            """Return the mean negative Euler log-density of the batch.

            References:
                [EQ-PROC-NSDE-001].

            """
            total: Any = 0.0
            for x, y, h in chosen:
                total = total + fitted.transition_negative_log_likelihood(
                    parameters, x, y, h
                )
            return total / batch

        return reverse_value_and_gradient(loss, theta)

    result = adam_minimize(
        objective,
        x0,
        iterations=steps,
        learning_rate=learning_rate,
        tail_fraction=tail_fraction,
        seed=stream.child("adam"),
    )
    mean_nll = (
        math.fsum(
            float(fitted.transition_negative_log_likelihood(result.x, x, y, h))
            for x, y, h in transitions
        )
        / count
    )
    return NeuralSDEFit(fitted, result.x, result, count, mean_nll)
