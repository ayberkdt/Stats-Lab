"""Discrete-time state-space models with additive Gaussian noise.

Two model types share one structural interface (``transition``, ``observe``,
``process_covariance``, ``observation_covariance``, ``initial_mean``,
``initial_covariance``) so every filter, smoother, ensemble method, and
particle method can consume either of them:

* :class:`LinearGaussianStateSpace` -- ``x_k = F x_(k-1) + B u_k + w_k`` and
  ``y_k = H x_k + v_k``;
* :class:`NonlinearStateSpace` -- ``x_k = f(x_(k-1)) + w_k`` and
  ``y_k = h(x_k) + v_k``, with optional analytic Jacobians.

Time convention: ``x_0 ~ N(m_0, P_0)`` is the prior state before the first
observation, and observations ``y_1, ..., y_T`` follow transitions.
Process noise ``w_k ~ N(0, Q)`` expresses model (dynamics) uncertainty;
measurement noise ``v_k ~ N(0, R)`` expresses sensor uncertainty. The two play
different roles: ``Q`` inflates the prediction covariance, ``R`` limits how
strongly one observation can pull the estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    matvec,
    symmetrize,
)
from probstat.optimization import central_jacobian

VectorMap = Callable[[Vector], Sequence[float]]
JacobianMap = Callable[[Vector], Iterable[Iterable[float]]]


def _covariance(
    values: Iterable[Iterable[object]],
    *,
    name: str,
    size: int,
    definite: bool,
) -> tuple[Matrix, Matrix]:
    """Validate a covariance matrix and return it with a lower square root.

    Args:
        values: Candidate square covariance.
        name: Field name used in error messages.
        size: Required dimension.
        definite: Require positive definiteness instead of semidefiniteness.

    Returns:
        Symmetrized covariance and a lower factor ``L`` with ``C = L L'``.

    Raises:
        ValueError: If the matrix is misshapen, asymmetric, or not
            positive (semi)definite.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    matrix = symmetrize(values, name=name)
    if len(matrix) != size:
        raise ValueError(f"{name} must be {size} by {size}")
    try:
        factor = cholesky(matrix) if definite else cholesky_semidefinite(matrix)
    except ValueError as error:
        kind = "positive definite" if definite else "positive semidefinite"
        raise ValueError(f"{name} must be {kind}") from error
    return matrix, factor


def _require_length(vector: Vector, size: int, name: str) -> Vector:
    """Require a vector of a given length.

    Args:
        vector: Candidate vector.
        size: Required length.
        name: Field name used in error messages.

    Returns:
        The unchanged vector.

    Raises:
        ValueError: If the length differs.

    References:
        Project validation policy. No equation is evaluated.

    """
    if len(vector) != size:
        raise ValueError(f"{name} must have length {size}")
    return vector


def _fast_vector(values: Sequence[float], size: int, name: str) -> Vector:
    """Convert a model-map image to floats with a cheap shape/finiteness check.

    Used in per-particle and per-member inner loops where the full indexed
    validation of :func:`probstat.linalg.as_vector` would dominate run time.

    Args:
        values: Image returned by a model map.
        size: Required length.
        name: Name used in error messages.

    Returns:
        Tuple of finite floats.

    Raises:
        ValueError: If the length differs or a component is non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    vector = tuple(map(float, values))
    if len(vector) != size:
        raise ValueError(f"{name} must have length {size}")
    if not all(map(math.isfinite, vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _affine(matrix: Matrix, vector: Vector, shift: Vector | None = None) -> Vector:
    """Return ``A x (+ b)`` with plain summation for inner sampling loops.

    Args:
        matrix: Rectangular matrix.
        vector: Vector with one entry per matrix column.
        shift: Optional additive vector.

    Returns:
        Affine image.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    image = [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]
    if shift is not None:
        image = [a + b for a, b in zip(image, shift, strict=True)]
    return tuple(image)


def _gaussian_draw(
    generator_gauss: Callable[[], float], mean: Vector, lower: Matrix
) -> Vector:
    """Draw ``mean + L z`` with independent standard-normal ``z``.

    Args:
        generator_gauss: Zero-argument standard-normal sampler.
        mean: Mean vector.
        lower: Lower square root of the covariance.

    Returns:
        One Gaussian draw.

    Equation:
        [EQ-FILT-SSM-001].

    References:
        [REF-DEVROYE-1986, Chapter XI].

    """
    noise = matvec(lower, tuple(generator_gauss() for _ in mean))
    return tuple(m + e for m, e in zip(mean, noise, strict=True))


@dataclass(frozen=True)
class LinearGaussianStateSpace:
    """Linear-Gaussian state-space model.

    ``x_k = F x_(k-1) + B u_k + w_k``, ``w_k ~ N(0, Q)``;
    ``y_k = H x_k + v_k``, ``v_k ~ N(0, R)``; ``x_0 ~ N(m_0, P_0)``.

    Args:
        transition_matrix: ``n`` by ``n`` state transition ``F``.
        process_covariance: ``n`` by ``n`` positive-semidefinite ``Q``.
        observation_matrix: ``m`` by ``n`` observation operator ``H``.
        observation_covariance: ``m`` by ``m`` positive-definite ``R``.
        initial_mean: Prior mean ``m_0`` of ``x_0``.
        initial_covariance: Positive-semidefinite prior covariance ``P_0``.
        control_matrix: Optional ``n`` by ``p`` control-input matrix ``B``.

    Equation:
        [EQ-FILT-SSM-001].

    References:
        [REF-KALMAN-1960], [REF-SARKKA-2013, Chapter 4].

    """

    transition_matrix: Matrix
    process_covariance: Matrix
    observation_matrix: Matrix
    observation_covariance: Matrix
    initial_mean: Vector
    initial_covariance: Matrix
    control_matrix: Matrix | None = None
    _process_factor: Matrix = field(init=False, repr=False, compare=False)
    _observation_factor: Matrix = field(init=False, repr=False, compare=False)
    _initial_factor: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate dimensions and covariance definiteness once.

        Raises:
            TypeError: If entries are not real.
            ValueError: If shapes disagree or covariances are invalid.

        References:
            [REF-SARKKA-2013, Chapter 4]. No equation is evaluated.

        """
        transition = as_matrix(
            self.transition_matrix, name="transition_matrix", square=True
        )
        size = len(transition)
        observation = as_matrix(self.observation_matrix, name="observation_matrix")
        if len(observation[0]) != size:
            raise ValueError("observation_matrix must have one column per state")
        process, process_factor = _covariance(
            self.process_covariance,
            name="process_covariance",
            size=size,
            definite=False,
        )
        noise, noise_factor = _covariance(
            self.observation_covariance,
            name="observation_covariance",
            size=len(observation),
            definite=True,
        )
        mean = _require_length(
            as_vector(self.initial_mean, name="initial_mean"), size, "initial_mean"
        )
        initial, initial_factor = _covariance(
            self.initial_covariance,
            name="initial_covariance",
            size=size,
            definite=False,
        )
        control: Matrix | None = None
        if self.control_matrix is not None:
            control = as_matrix(self.control_matrix, name="control_matrix")
            if len(control) != size:
                raise ValueError("control_matrix must have one row per state")
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "observation_matrix", observation)
        object.__setattr__(self, "process_covariance", process)
        object.__setattr__(self, "observation_covariance", noise)
        object.__setattr__(self, "initial_mean", mean)
        object.__setattr__(self, "initial_covariance", initial)
        object.__setattr__(self, "control_matrix", control)
        object.__setattr__(self, "_process_factor", process_factor)
        object.__setattr__(self, "_observation_factor", noise_factor)
        object.__setattr__(self, "_initial_factor", initial_factor)

    @property
    def state_dimension(self) -> int:
        """Return the state dimension ``n``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        return len(self.transition_matrix)

    @property
    def observation_dimension(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        return len(self.observation_matrix)

    def control_effect(self, control: Sequence[float] | None) -> Vector:
        """Return ``B u`` (a zero vector when there is no control).

        Args:
            control: Control input ``u_k`` or ``None``.

        Returns:
            Additive state shift.

        Raises:
            ValueError: If a control is given without a control matrix or has
                the wrong length.

        Equation:
            [EQ-FILT-SSM-001].

        References:
            [REF-SIMON-2006, Chapter 5].

        """
        if control is None:
            return tuple(0.0 for _ in range(self.state_dimension))
        if self.control_matrix is None:
            raise ValueError("model has no control_matrix")
        vector = as_vector(control, name="control")
        if len(vector) != len(self.control_matrix[0]):
            raise ValueError("control length must equal control_matrix columns")
        return matvec(self.control_matrix, vector)

    def transition(
        self, state: Sequence[float], control: Sequence[float] | None = None
    ) -> Vector:
        """Return the noise-free transition ``F x + B u``.

        Args:
            state: State vector ``x_(k-1)``.
            control: Optional control input ``u_k``.

        Returns:
            Predicted state mean before process noise.

        Equation:
            [EQ-FILT-SSM-001].

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        vector = _require_length(
            as_vector(state, name="state"), self.state_dimension, "state"
        )
        shift = self.control_effect(control)
        return tuple(
            a + b
            for a, b in zip(matvec(self.transition_matrix, vector), shift, strict=True)
        )

    def observe(self, state: Sequence[float]) -> Vector:
        """Return the noise-free observation ``H x``.

        Args:
            state: State vector ``x_k``.

        Returns:
            Predicted observation before measurement noise.

        Equation:
            [EQ-FILT-SSM-001].

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        vector = _require_length(
            as_vector(state, name="state"), self.state_dimension, "state"
        )
        return matvec(self.observation_matrix, vector)

    def _propagate(self, state: Vector, control: Sequence[float] | None) -> Vector:
        """Return ``F x + B u`` without re-validating an internal state.

        Args:
            state: Internally generated state of correct length.
            control: Optional control ``u_k``.

        Returns:
            Noise-free transition image.

        Equation:
            [EQ-FILT-SSM-001].

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        shift = None if control is None else self.control_effect(control)
        return _affine(self.transition_matrix, state, shift)

    def _measure(self, state: Vector) -> Vector:
        """Return ``H x`` without re-validating an internal state.

        Args:
            state: Internally generated state of correct length.

        Returns:
            Noise-free observation image.

        Equation:
            [EQ-FILT-SSM-001].

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        return _affine(self.observation_matrix, state)

    def transition_jacobian(self, state: Sequence[float]) -> Matrix:
        """Return the exact transition Jacobian ``F``.

        Args:
            state: Ignored linearization point (kept for interface parity).

        Returns:
            Transition matrix.

        References:
            [REF-SARKKA-2013, Chapter 5]. No equation is evaluated.

        """
        return self.transition_matrix

    def observation_jacobian(self, state: Sequence[float]) -> Matrix:
        """Return the exact observation Jacobian ``H``.

        Args:
            state: Ignored linearization point (kept for interface parity).

        Returns:
            Observation matrix.

        References:
            [REF-SARKKA-2013, Chapter 5]. No equation is evaluated.

        """
        return self.observation_matrix

    def as_nonlinear(self) -> NonlinearStateSpace:
        """Return the same model expressed as a :class:`NonlinearStateSpace`.

        Controls are not carried over; the conversion is for uncontrolled
        models used to cross-check extended, unscented, ensemble, and
        particle methods against the exact Kalman filter.

        Returns:
            Nonlinear model with exact linear maps and Jacobians.

        Raises:
            ValueError: If the model has a control matrix.

        Equation:
            [EQ-FILT-SSM-001], [EQ-FILT-SSM-002].

        References:
            [REF-SARKKA-2013, Chapters 4-5].

        """
        if self.control_matrix is not None:
            raise ValueError("controlled models cannot be converted")
        transition_matrix = self.transition_matrix
        observation_matrix = self.observation_matrix

        def transition(state: Vector) -> Vector:
            """Apply ``F x``.

            References:
                [EQ-FILT-SSM-001], [REF-SARKKA-2013, Chapter 4].

            """
            return matvec(transition_matrix, state)

        def observation(state: Vector) -> Vector:
            """Apply ``H x``.

            References:
                [EQ-FILT-SSM-001], [REF-SARKKA-2013, Chapter 4].

            """
            return matvec(observation_matrix, state)

        def transition_jacobian(state: Vector) -> Matrix:
            """Return ``F``.

            References:
                [REF-SARKKA-2013, Chapter 5]. No equation is evaluated.

            """
            return transition_matrix

        def observation_jacobian(state: Vector) -> Matrix:
            """Return ``H``.

            References:
                [REF-SARKKA-2013, Chapter 5]. No equation is evaluated.

            """
            return observation_matrix

        return NonlinearStateSpace(
            transition,
            observation,
            self.process_covariance,
            self.observation_covariance,
            self.initial_mean,
            self.initial_covariance,
            state_dimension=self.state_dimension,
            observation_dimension=self.observation_dimension,
            transition_jacobian_function=transition_jacobian,
            observation_jacobian_function=observation_jacobian,
        )


@dataclass(frozen=True)
class NonlinearStateSpace:
    """Nonlinear state-space model with additive Gaussian noise.

    ``x_k = f(x_(k-1)) + w_k``, ``w_k ~ N(0, Q)``;
    ``y_k = h(x_k) + v_k``, ``v_k ~ N(0, R)``; ``x_0 ~ N(m_0, P_0)``.

    Args:
        transition_function: Deterministic dynamics ``f``.
        observation_function: Deterministic measurement map ``h``.
        process_covariance: Positive-semidefinite ``Q``.
        observation_covariance: Positive-definite ``R``.
        initial_mean: Prior mean ``m_0``.
        initial_covariance: Positive-semidefinite ``P_0``.
        state_dimension: State dimension ``n``.
        observation_dimension: Observation dimension ``m``.
        transition_jacobian_function: Optional analytic ``df/dx``; central
            differences are used when omitted.
        observation_jacobian_function: Optional analytic ``dh/dx``; central
            differences are used when omitted.
        jacobian_step: Relative central-difference step.

    Equation:
        [EQ-FILT-SSM-002].

    References:
        [REF-SARKKA-2013, Chapters 4-5].

    """

    transition_function: VectorMap
    observation_function: VectorMap
    process_covariance: Matrix
    observation_covariance: Matrix
    initial_mean: Vector
    initial_covariance: Matrix
    state_dimension: int
    observation_dimension: int
    transition_jacobian_function: JacobianMap | None = None
    observation_jacobian_function: JacobianMap | None = None
    jacobian_step: float = 1e-6
    _process_factor: Matrix = field(init=False, repr=False, compare=False)
    _observation_factor: Matrix = field(init=False, repr=False, compare=False)
    _initial_factor: Matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate callables, dimensions, and covariances once.

        Raises:
            TypeError: If maps are not callable.
            ValueError: If dimensions or covariances are invalid.

        References:
            [REF-SARKKA-2013, Chapter 5]. No equation is evaluated.

        """
        if not callable(self.transition_function):
            raise TypeError("transition_function must be callable")
        if not callable(self.observation_function):
            raise TypeError("observation_function must be callable")
        for value, name in (
            (self.state_dimension, "state_dimension"),
            (self.observation_dimension, "observation_dimension"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.jacobian_step > 0.0:
            raise ValueError("jacobian_step must be positive")
        size = self.state_dimension
        process, process_factor = _covariance(
            self.process_covariance,
            name="process_covariance",
            size=size,
            definite=False,
        )
        noise, noise_factor = _covariance(
            self.observation_covariance,
            name="observation_covariance",
            size=self.observation_dimension,
            definite=True,
        )
        mean = _require_length(
            as_vector(self.initial_mean, name="initial_mean"), size, "initial_mean"
        )
        initial, initial_factor = _covariance(
            self.initial_covariance,
            name="initial_covariance",
            size=size,
            definite=False,
        )
        object.__setattr__(self, "process_covariance", process)
        object.__setattr__(self, "observation_covariance", noise)
        object.__setattr__(self, "initial_mean", mean)
        object.__setattr__(self, "initial_covariance", initial)
        object.__setattr__(self, "_process_factor", process_factor)
        object.__setattr__(self, "_observation_factor", noise_factor)
        object.__setattr__(self, "_initial_factor", initial_factor)

    def transition(
        self, state: Sequence[float], control: Sequence[float] | None = None
    ) -> Vector:
        """Return the noise-free transition ``f(x)``.

        Args:
            state: State vector ``x_(k-1)``.
            control: Must be ``None``; nonlinear controls belong inside ``f``.

        Returns:
            Validated image ``f(x)``.

        Raises:
            ValueError: If a control is supplied or ``f`` returns a vector of
                the wrong length.

        Equation:
            [EQ-FILT-SSM-002].

        References:
            [REF-SARKKA-2013, Chapter 5].

        """
        if control is not None:
            raise ValueError("NonlinearStateSpace does not accept controls")
        vector = _require_length(
            as_vector(state, name="state"), self.state_dimension, "state"
        )
        return _require_length(
            as_vector(self.transition_function(vector), name="f(x)"),
            self.state_dimension,
            "f(x)",
        )

    def observe(self, state: Sequence[float]) -> Vector:
        """Return the noise-free observation ``h(x)``.

        Args:
            state: State vector ``x_k``.

        Returns:
            Validated image ``h(x)``.

        Raises:
            ValueError: If ``h`` returns a vector of the wrong length.

        Equation:
            [EQ-FILT-SSM-002].

        References:
            [REF-SARKKA-2013, Chapter 5].

        """
        vector = _require_length(
            as_vector(state, name="state"), self.state_dimension, "state"
        )
        return _require_length(
            as_vector(self.observation_function(vector), name="h(x)"),
            self.observation_dimension,
            "h(x)",
        )

    def _propagate(self, state: Vector, control: Sequence[float] | None) -> Vector:
        """Return ``f(x)`` with a cheap output check for inner loops.

        Args:
            state: Internally generated state of correct length.
            control: Must be ``None``.

        Returns:
            Noise-free transition image.

        Raises:
            ValueError: If a control is supplied or the image is invalid.

        Equation:
            [EQ-FILT-SSM-002].

        References:
            [REF-SARKKA-2013, Chapter 5].

        """
        if control is not None:
            raise ValueError("NonlinearStateSpace does not accept controls")
        return _fast_vector(
            self.transition_function(state), self.state_dimension, "f(x)"
        )

    def _measure(self, state: Vector) -> Vector:
        """Return ``h(x)`` with a cheap output check for inner loops.

        Args:
            state: Internally generated state of correct length.

        Returns:
            Noise-free observation image.

        Raises:
            ValueError: If the image is invalid.

        Equation:
            [EQ-FILT-SSM-002].

        References:
            [REF-SARKKA-2013, Chapter 5].

        """
        return _fast_vector(
            self.observation_function(state), self.observation_dimension, "h(x)"
        )

    def transition_jacobian(self, state: Sequence[float]) -> Matrix:
        """Return ``df/dx`` analytically or by central differences.

        Args:
            state: Linearization point.

        Returns:
            ``n`` by ``n`` Jacobian.

        Raises:
            ValueError: If an analytic Jacobian has the wrong shape.

        Equation:
            [EQ-FILT-EKF-001], [EQ-OPT-FD-001].

        References:
            [REF-SARKKA-2013, Chapter 5], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        point = as_vector(state, name="state")
        if self.transition_jacobian_function is not None:
            matrix = as_matrix(
                self.transition_jacobian_function(point), name="transition_jacobian"
            )
        else:
            matrix, _ = central_jacobian(
                self.transition, point, relative_step=self.jacobian_step
            )
        if len(matrix) != self.state_dimension or len(matrix[0]) != len(point):
            raise ValueError("transition Jacobian has the wrong shape")
        return matrix

    def observation_jacobian(self, state: Sequence[float]) -> Matrix:
        """Return ``dh/dx`` analytically or by central differences.

        Args:
            state: Linearization point.

        Returns:
            ``m`` by ``n`` Jacobian.

        Raises:
            ValueError: If an analytic Jacobian has the wrong shape.

        Equation:
            [EQ-FILT-EKF-001], [EQ-OPT-FD-001].

        References:
            [REF-SARKKA-2013, Chapter 5], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        point = as_vector(state, name="state")
        if self.observation_jacobian_function is not None:
            matrix = as_matrix(
                self.observation_jacobian_function(point),
                name="observation_jacobian",
            )
        else:
            matrix, _ = central_jacobian(
                self.observe, point, relative_step=self.jacobian_step
            )
        if len(matrix) != self.observation_dimension or len(matrix[0]) != len(point):
            raise ValueError("observation Jacobian has the wrong shape")
        return matrix


StateSpaceModel = LinearGaussianStateSpace | NonlinearStateSpace


@dataclass(frozen=True, slots=True)
class StateSpaceSimulation:
    """One simulated trajectory of a state-space model.

    Args:
        states: True states ``x_0, ..., x_T``.
        observations: Noisy observations ``y_1, ..., y_T``.

    References:
        [REF-SARKKA-2013, Chapter 4].

    """

    states: tuple[Vector, ...]
    observations: tuple[Vector, ...]


def simulate(
    model: StateSpaceModel,
    steps: int,
    *,
    seed: int | RandomStream | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
) -> StateSpaceSimulation:
    """Simulate states and observations from a state-space model.

    Args:
        model: Linear or nonlinear additive-Gaussian model.
        steps: Positive number of transitions/observations ``T``.
        seed: Integer seed or stream (child streams ``"initial"``,
            ``"process"``, and ``"measurement"`` keep noise sources separate).
        controls: Optional controls ``u_1, ..., u_T`` for linear models.

    Returns:
        True states and observations.

    Raises:
        ValueError: If ``steps`` is not positive or controls have the wrong
            length.

    Equation:
        [EQ-FILT-SSM-001], [EQ-FILT-SSM-002].

    References:
        [REF-SARKKA-2013, Chapter 4], [REF-DEVROYE-1986, Chapter XI].

    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    plan = _control_plan(controls, steps)
    stream = resolve_stream(seed)
    initial_gauss = stream.child("initial").generator().gauss
    process_gauss = stream.child("process").generator().gauss
    measurement_gauss = stream.child("measurement").generator().gauss
    zero_state = tuple(0.0 for _ in range(model.state_dimension))
    zero_observation = tuple(0.0 for _ in range(model.observation_dimension))
    state = _gaussian_draw(initial_gauss, model.initial_mean, model._initial_factor)
    states = [state]
    observations: list[Vector] = []
    for control in plan:
        noise = _gaussian_draw(process_gauss, zero_state, model._process_factor)
        mean = model.transition(state, control)
        state = tuple(a + b for a, b in zip(mean, noise, strict=True))
        error = _gaussian_draw(
            measurement_gauss, zero_observation, model._observation_factor
        )
        observation = tuple(
            a + b for a, b in zip(model.observe(state), error, strict=True)
        )
        states.append(state)
        observations.append(observation)
    return StateSpaceSimulation(tuple(states), tuple(observations))


def _control_plan(
    controls: Sequence[Sequence[float] | None] | None, steps: int
) -> tuple[Sequence[float] | None, ...]:
    """Return one control entry (possibly ``None``) per time step.

    Args:
        controls: Optional control sequence.
        steps: Number of time steps.

    Returns:
        Tuple of length ``steps``.

    Raises:
        ValueError: If the control sequence has the wrong length.

    References:
        [REF-SIMON-2006, Chapter 5]. No equation is evaluated.

    """
    if controls is None:
        return tuple(None for _ in range(steps))
    plan = tuple(controls)
    if len(plan) != steps:
        raise ValueError("controls must contain one entry per time step")
    return plan
