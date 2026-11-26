"""ODE tangent/adjoint models, a simulator adapter and time-dependent inversion.

Experimental (added in 0.28).

This module closes the dynamic part of the adjoint roadmap with four pieces.

1. :class:`ODEForwardModel`: a :class:`~probstat.inverse.gauss_newton.
   DifferentiableForward` map from the parameters ``theta`` and (optionally) the
   initial state ``x0`` of ``x' = f(t, x, theta)`` to linear observations
   ``H x(t_j)`` at the given times. The state is advanced by a FIXED-step
   classical RK4 scheme with a declared number of steps per output interval
   [EQ-INV-ODE-001]. The tangent is the forward sensitivity of THE RK4 SCHEME
   ITSELF (the variational equation pushed through the same stages), so ``jvp``
   is the exact derivative of the discrete map, not an ``O(h^4)`` approximation
   of the derivative of the continuous flow [EQ-INV-ODE-002]. ``vjp`` is the
   exact transpose of that tangent, propagated backwards through the RK4
   stages [EQ-INV-ODE-003]; ``w'(J v) = (J'w)'v`` therefore holds to rounding
   and the gradient costs one forward and one reverse sweep for any number of
   parameters. Optional checkpointing trades recomputed RK4 steps for stored
   states and gives the identical gradient [EQ-INV-ODE-004].
2. :class:`SimulatorAdapter`: wraps a user callable (an external solver) as a
   :class:`~probstat.inverse.gauss_newton.DifferentiableForward` with input
   validation, evaluation counts, an explicit opt-in cache, structured
   failures (exceptions, non-finite or malformed outputs are recorded as
   :class:`SimulatorFailure`, never swallowed) and optional central
   differences with a DECLARED step [EQ-INV-ODE-005].
3. :class:`HeatCoefficientModel`: nonlinear coefficient inversion for
   ``u_t = (kappa(x) u_x)_x`` with ``kappa = exp(a)`` and ``a`` a basis
   expansion, discretized by conservative finite differences in space and
   implicit Euler in time; the tangent and the exact discrete adjoint run
   through the time steps with one reusable tridiagonal factorization
   [EQ-INV-ODE-006], [EQ-INV-ODE-007]. :func:`heat_coefficient_benchmark`
   generates its data on a finer space-time grid
   (``same_discretization=False``).
4. :func:`damped_oscillator_benchmark` (catalog case N03): a damped
   oscillator ``x'' + 2 zeta w x' + w^2 x = s(t)`` with unknown initial state
   and piecewise-constant source, truth from a finer time step, a separate
   noise stream, whitened singular values and the linear-Gaussian posterior
   of the initial state [EQ-INV-ODE-008], [EQ-INV-ODE-009].

Scope and caveats:
    - Fixed steps only. Adaptive step-size control, events and discontinuous
      right-hand sides are NOT supported; a right-hand side that is
      discontinuous inside an RK4 step gives a discrete map whose derivative
      is still exact for the discrete scheme but does not approximate the
      derivative of the flow. Piecewise-constant forcing must therefore be
      aligned with the output intervals, which is why every callback receives
      the index of the output interval that contains the step (``segment``).
    - A derivative that is exact for the discrete scheme is what an optimizer
      needs; it converges to the continuous derivative only as the step goes
      to zero.
    - ``value`` may return non-finite entries when the integration blows up;
      ``jvp``/``vjp`` raise ``ValueError`` instead of returning non-finite
      derivatives.

References:
    [REF-HAIRER-NORSETT-WANNER-1993], [REF-GRIEWANK-WALTHER-2008],
    [REF-NOCEDAL-WRIGHT-2006], [REF-KAIPIO-SOMERSALO-2007],
    [REF-TARANTOLA-2005].

"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inverse.benchmark_extended import NonlinearBenchmark
from probstat.inverse.gauss_newton import ModelCounts
from probstat.inverse.linear_gaussian import (
    GaussianLinearPosterior,
    linear_gaussian_posterior,
)
from probstat.inverse.pde_models import TridiagonalFactorization
from probstat.linalg import (
    DenseCovariance,
    DenseOperator,
    Matrix,
    ScaledIdentityCovariance,
    Vector,
    singular_value_decomposition,
)

# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _int_at_least(value: object, name: str, minimum: int = 1) -> int:
    """Validate an integer of at least ``minimum`` (``bool`` is rejected).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real(value: object, name: str, *, positive: bool = False) -> float:
    """Validate a finite real number, optionally strictly positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _vector(values: object, size: int, name: str, *, finite: bool = True) -> Vector:
    """Validate a real sequence of a given length and return it as a tuple.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    if len(values) != size:
        raise ValueError(f"{name} must have length {size}, got {len(values)}")
    out: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        number = float(item)
        if finite and not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        out.append(number)
    return tuple(out)


def _matrix(values: object, rows: int, cols: int, name: str) -> list[list[float]]:
    """Validate a ``rows x cols`` real matrix given as nested sequences.

    Non-finite entries are allowed (a blown-up right-hand side); the caller
    checks derivative outputs for finiteness.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a matrix (sequence of rows)")
    if len(values) != rows:
        raise ValueError(f"{name} must have {rows} rows, got {len(values)}")
    return [list(_vector(row, cols, f"{name} row", finite=False)) for row in values]


def _light_matrix(
    values: Sequence[Sequence[float]], rows: int, cols: int, name: str
) -> list[list[float]]:
    """Shape-check a callback matrix on the hot path (lengths only).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if len(values) != rows or any(len(row) != cols for row in values):
        raise ValueError(f"{name} must be {rows} x {cols}")
    return [[float(v) for v in row] for row in values]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean inner product.

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    return sum(x * y for x, y in zip(a, b, strict=True))


def _is_finite(values: Sequence[float]) -> bool:
    """Return whether every entry is finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    return all(math.isfinite(v) for v in values)


def _matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    """Return ``A v`` for a row-major matrix.

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    return [sum(x * y for x, y in zip(row, v, strict=True)) for row in a]


def _rmatvec(
    a: Sequence[Sequence[float]], v: Sequence[float], cols: int
) -> list[float]:
    """Return ``A' v`` for a row-major matrix with ``cols`` columns.

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    out = [0.0] * cols
    for row, weight in zip(a, v, strict=True):
        if weight != 0.0:
            for j in range(cols):
                out[j] += row[j] * weight
    return out


# ---------------------------------------------------------------------------
# ODE model
# ---------------------------------------------------------------------------

RightHandSide = Callable[[float, Vector, Vector, int], Sequence[float]]
Jacobian = Callable[[float, Vector, Vector, int], Sequence[Sequence[float]]]


@dataclass(frozen=True, slots=True)
class ODESystem:
    """An ODE ``x' = f(t, x, theta)`` with its analytic Jacobians.

    Every callback receives ``(t, x, theta, segment)`` where ``segment`` is the
    zero-based index of the OUTPUT interval ``[t_(j-1), t_j]`` that contains
    the RK4 step, so a piecewise-constant forcing aligned with the output
    intervals is evaluated without ambiguity at stage times that fall on a
    breakpoint. Autonomous or smooth systems ignore ``segment``.

    Attributes:
        state_dimension: ``d >= 1``.
        parameter_dimension: ``p >= 0``.
        rhs: ``(t, x, theta, segment) -> f`` (length ``d``).
        state_jacobian: ``-> f_x`` (``d x d``).
        parameter_jacobian: ``-> f_theta`` (``d x p``); required iff
            ``p >= 1``.

    References:
        [EQ-INV-ODE-001], [EQ-INV-ODE-002].

    """

    state_dimension: int
    parameter_dimension: int
    rhs: RightHandSide
    state_jacobian: Jacobian
    parameter_jacobian: Jacobian | None = None


@dataclass(frozen=True, slots=True)
class ODEWorkCounts:
    """Work counters of an :class:`ODEForwardModel`.

    Attributes:
        forward_solves: Calls of ``value``/``states``.
        tangent_solves: Calls of ``jvp``.
        adjoint_solves: Calls of ``vjp``.
        rk4_steps: RK4 steps executed, including tangent and recomputed ones.
        recomputed_steps: RK4 steps repeated only because of checkpointing.
        rhs_evaluations: Evaluations of ``f``.
        jacobian_evaluations: Evaluations of ``f_x`` (each paired with one
            ``f_theta`` when ``p >= 1``).
        peak_stored_states: Largest number of state vectors held at once by
            the reverse sweep (checkpoints plus one recomputed block).

    References:
        [EQ-INV-ODE-004].

    """

    forward_solves: int
    tangent_solves: int
    adjoint_solves: int
    rk4_steps: int
    recomputed_steps: int
    rhs_evaluations: int
    jacobian_evaluations: int
    peak_stored_states: int


def exponential_decay_system() -> ODESystem:
    """Return ``x' = -theta x`` (``d = 1``, ``p = 1``), the analytic test system.

    Returns:
        The :class:`ODESystem`.

    References:
        [EQ-INV-ODE-001], [REF-HAIRER-NORSETT-WANNER-1993].

    """

    def rhs(t: float, x: Vector, theta: Vector, segment: int) -> Sequence[float]:
        """Return ``-theta x``.

        References:
            [EQ-INV-ODE-001].

        """
        return (-theta[0] * x[0],)

    def state_jacobian(
        t: float, x: Vector, theta: Vector, segment: int
    ) -> Sequence[Sequence[float]]:
        """Return ``-theta``.

        References:
            [EQ-INV-ODE-002].

        """
        return ((-theta[0],),)

    def parameter_jacobian(
        t: float, x: Vector, theta: Vector, segment: int
    ) -> Sequence[Sequence[float]]:
        """Return ``-x``.

        References:
            [EQ-INV-ODE-002].

        """
        return ((-x[0],),)

    return ODESystem(1, 1, rhs, state_jacobian, parameter_jacobian)


def damped_oscillator_system(
    omega: float, zeta: float, *, pieces: int, intervals_per_piece: int
) -> ODESystem:
    """Return ``x'' + 2 zeta w x' + w^2 x = s(t)`` as a first-order system.

    The state is ``(x, x')`` and the parameters are the ``pieces`` source
    amplitudes; the source equals ``theta[segment // intervals_per_piece]``,
    i.e. it is constant on groups of ``intervals_per_piece`` output intervals
    [EQ-INV-ODE-008]. The system is linear in the state and the parameters.

    Args:
        omega: Natural frequency ``w > 0``.
        zeta: Damping ratio ``zeta >= 0`` (``zeta < 1`` oscillates).
        pieces: Number of constant source pieces (``>= 1``).
        intervals_per_piece: Output intervals per piece (``>= 1``).

    Returns:
        The :class:`ODESystem` with ``d = 2`` and ``p = pieces``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-ODE-008].

    """
    w = _real(omega, "omega", positive=True)
    z = _real(zeta, "zeta")
    if z < 0.0:
        raise ValueError("zeta must be non-negative")
    k = _int_at_least(pieces, "pieces")
    q = _int_at_least(intervals_per_piece, "intervals_per_piece")
    stiffness = w * w
    damping = 2.0 * z * w

    def piece(segment: int) -> int:
        """Return the source piece that contains an output interval.

        References:
            [EQ-INV-ODE-008].

        """
        return min(segment // q, k - 1)

    def rhs(t: float, x: Vector, theta: Vector, segment: int) -> Sequence[float]:
        """Return ``(x', -w^2 x - 2 zeta w x' + s)``.

        References:
            [EQ-INV-ODE-008].

        """
        return (x[1], -stiffness * x[0] - damping * x[1] + theta[piece(segment)])

    def state_jacobian(
        t: float, x: Vector, theta: Vector, segment: int
    ) -> Sequence[Sequence[float]]:
        """Return the constant state Jacobian.

        References:
            [EQ-INV-ODE-008].

        """
        return ((0.0, 1.0), (-stiffness, -damping))

    def parameter_jacobian(
        t: float, x: Vector, theta: Vector, segment: int
    ) -> Sequence[Sequence[float]]:
        """Return the indicator of the active source piece.

        References:
            [EQ-INV-ODE-008].

        """
        row = [0.0] * k
        row[piece(segment)] = 1.0
        return ((0.0,) * k, tuple(row))

    return ODESystem(2, k, rhs, state_jacobian, parameter_jacobian)


class ODEForwardModel:
    """Fixed-step RK4 forward model with exact discrete tangent and adjoint.

    The parameter vector is ``z = (x0, theta)`` when ``estimate_initial_state``
    is true and ``z = theta`` otherwise (``x0`` then fixed to
    ``initial_state``). The data are ``y = (H x(t_1), ..., H x(t_J))`` stacked
    in time order for output times ``t_1 < ... < t_J`` (``t_1 >= t0``; a time
    equal to ``t0`` observes the initial state). Between consecutive output
    times (starting at ``t0``) exactly ``steps_per_interval`` RK4 steps of equal
    size are taken, so every output time is hit exactly and no interpolation is
    involved [EQ-INV-ODE-001].

    ``jvp`` pushes the variational equation through the same RK4 stages
    [EQ-INV-ODE-002]; ``vjp`` is its exact transpose [EQ-INV-ODE-003]. With
    ``checkpoint_every = c > 1`` only every ``c``-th state is stored and the
    steps of one block are recomputed during the reverse sweep; the result is
    the same vector as with full storage up to rounding [EQ-INV-ODE-004].
    ``max_stored_states`` instead derives the smallest ``c`` that keeps the
    peak number of stored states within the budget (``ValueError`` if no ``c``
    does).

    References:
        [EQ-INV-ODE-001], [EQ-INV-ODE-002], [EQ-INV-ODE-003],
        [EQ-INV-ODE-004], [REF-HAIRER-NORSETT-WANNER-1993],
        [REF-GRIEWANK-WALTHER-2008].

    """

    __slots__ = (
        "_checkpoint",
        "_counts",
        "_estimate",
        "_obs_at",
        "_obs_matrix",
        "_obs_rows",
        "_schedule",
        "_system",
        "_x0_fixed",
        "_peak",
        "_total_steps",
    )

    def __init__(
        self,
        system: ODESystem,
        times: Sequence[float],
        observation_matrix: Sequence[Sequence[float]],
        *,
        initial_state: Sequence[float] | None = None,
        estimate_initial_state: bool = True,
        t0: float = 0.0,
        steps_per_interval: int = 8,
        checkpoint_every: int | None = None,
        max_stored_states: int | None = None,
    ) -> None:
        """Validate and store the model definition.

        Args:
            system: The :class:`ODESystem`.
            times: Strictly increasing output times, each ``>= t0``.
            observation_matrix: ``H`` (``q x d``) with ``q >= 1`` rows.
            initial_state: Fixed ``x0``; required iff
                ``estimate_initial_state`` is false.
            estimate_initial_state: Whether ``x0`` is part of ``z``.
            t0: Start time.
            steps_per_interval: RK4 steps per output interval (``>= 1``).
            checkpoint_every: Store every ``c``-th state (default 1, store
                all). Exclusive with ``max_stored_states``.
            max_stored_states: Budget of simultaneously stored states.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range or inconsistent.

        References:
            [EQ-INV-ODE-001], [EQ-INV-ODE-004].

        """
        if not isinstance(system, ODESystem):
            raise TypeError("system must be an ODESystem")
        d = _int_at_least(system.state_dimension, "state_dimension")
        p = _int_at_least(system.parameter_dimension, "parameter_dimension", 0)
        if not callable(system.rhs) or not callable(system.state_jacobian):
            raise TypeError("rhs and state_jacobian must be callable")
        if p >= 1 and not callable(system.parameter_jacobian):
            raise ValueError("parameter_jacobian is required when p >= 1")
        start = _real(t0, "t0")
        if isinstance(times, (str, bytes)) or not isinstance(times, Sequence):
            raise TypeError("times must be a sequence of real numbers")
        stamps = _vector(times, len(times), "times")
        if len(stamps) < 1:
            raise ValueError("times must be non-empty")
        previous = start
        for index, stamp in enumerate(stamps):
            if stamp < previous or (index > 0 and stamp == previous):
                raise ValueError("times must be strictly increasing and >= t0")
            previous = stamp
        if isinstance(observation_matrix, (str, bytes)) or not isinstance(
            observation_matrix, Sequence
        ):
            raise TypeError("observation_matrix must be a matrix")
        rows = _matrix(
            observation_matrix, len(observation_matrix), d, "observation_matrix"
        )
        if not rows:
            raise ValueError("observation_matrix needs at least one row")
        if not all(_is_finite(r) for r in rows):
            raise ValueError("observation_matrix must be finite")
        n_steps = _int_at_least(steps_per_interval, "steps_per_interval")
        self._estimate = bool(estimate_initial_state)
        if self._estimate:
            if initial_state is not None:
                raise ValueError("initial_state is given but estimate_initial_state")
            self._x0_fixed: Vector = (0.0,) * d
        else:
            if initial_state is None:
                raise ValueError("initial_state is required when not estimating it")
            self._x0_fixed = _vector(initial_state, d, "initial_state")
        if d * int(self._estimate) + p < 1:
            raise ValueError("the model has no parameters")
        schedule: list[tuple[float, float, int]] = []
        obs_at: dict[int, list[int]] = {}
        left = start
        for j, right in enumerate(stamps):
            if right > left:
                h = (right - left) / n_steps
                for i in range(n_steps):
                    schedule.append((left + i * h, h, j))
            obs_at.setdefault(len(schedule), []).append(j)
            left = right
        self._schedule = schedule
        self._total_steps = len(schedule)
        self._obs_at = obs_at
        self._obs_matrix = rows
        self._obs_rows = len(rows)
        self._system = system
        if checkpoint_every is not None and max_stored_states is not None:
            raise ValueError("give checkpoint_every or max_stored_states, not both")
        total = self._total_steps
        if max_stored_states is not None:
            budget = _int_at_least(max_stored_states, "max_stored_states", 2)
            choice = next(
                (
                    c
                    for c in range(1, max(total, 1) + 1)
                    if -(-total // c) + 1 + c <= budget
                ),
                None,
            )
            if choice is None:
                raise ValueError("max_stored_states is too small for this horizon")
            self._checkpoint = choice
        else:
            self._checkpoint = (
                1
                if checkpoint_every is None
                else _int_at_least(checkpoint_every, "checkpoint_every")
            )
        self._counts = [0, 0, 0, 0, 0, 0, 0, 0]
        self._peak = 0

    # -- dimensions and counters -------------------------------------------

    @property
    def state_dimension(self) -> int:
        """Return the state dimension ``d``.

        References:
            [EQ-INV-ODE-001].

        """
        return self._system.state_dimension

    @property
    def input_dimension(self) -> int:
        """Return ``n = d + p`` (initial state estimated) or ``p``.

        References:
            [EQ-INV-ODE-001].

        """
        d = self._system.state_dimension if self._estimate else 0
        return d + self._system.parameter_dimension

    @property
    def output_dimension(self) -> int:
        """Return ``m = q J`` (rows of ``H`` times output times).

        References:
            [EQ-INV-ODE-001].

        """
        return self._obs_rows * sum(len(v) for v in self._obs_at.values())

    @property
    def total_steps(self) -> int:
        """Return the number of RK4 steps of one trajectory.

        References:
            [EQ-INV-ODE-001].

        """
        return self._total_steps

    @property
    def checkpoint_every(self) -> int:
        """Return the checkpoint spacing ``c`` used by the reverse sweep.

        References:
            [EQ-INV-ODE-004].

        """
        return self._checkpoint

    @property
    def counts(self) -> ModelCounts:
        """Return the model-level evaluation counts (``value``, ``jvp``, ``vjp``).

        References:
            [EQ-INV-ODE-004].

        """
        return ModelCounts(
            value=self._counts[0],
            jvp=self._counts[1],
            vjp=self._counts[2],
            derivative_value=0,
        )

    @property
    def work(self) -> ODEWorkCounts:
        """Return the detailed work counters.

        References:
            [EQ-INV-ODE-004].

        """
        c = self._counts
        return ODEWorkCounts(
            forward_solves=c[0],
            tangent_solves=c[1],
            adjoint_solves=c[2],
            rk4_steps=c[3],
            recomputed_steps=c[4],
            rhs_evaluations=c[5],
            jacobian_evaluations=c[6],
            peak_stored_states=self._peak,
        )

    def reset_counts(self) -> None:
        """Reset all counters to zero.

        References:
            [EQ-INV-ODE-004].

        """
        self._counts = [0] * 8
        self._peak = 0

    # -- callbacks ----------------------------------------------------------

    def _split(self, z: object) -> tuple[Vector, Vector]:
        """Validate ``z`` and split it into ``(x0, theta)``.

        References:
            [EQ-INV-ODE-001].

        """
        point = _vector(z, self.input_dimension, "x")
        if not self._estimate:
            return self._x0_fixed, point
        d = self._system.state_dimension
        return point[:d], point[d:]

    def _f(self, t: float, x: Vector, theta: Vector, seg: int) -> list[float]:
        """Evaluate and validate the right-hand side.

        References:
            [EQ-INV-ODE-001].

        """
        self._counts[5] += 1
        d = self._system.state_dimension
        raw = self._system.rhs(t, x, theta, seg)
        if len(raw) != d:
            raise ValueError(f"rhs output must have length {d}, got {len(raw)}")
        return [float(v) for v in raw]

    def _jac(
        self, t: float, x: Vector, theta: Vector, seg: int
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Evaluate and validate ``(f_x, f_theta)``.

        References:
            [EQ-INV-ODE-002].

        """
        self._counts[6] += 1
        d = self._system.state_dimension
        p = self._system.parameter_dimension
        a = _light_matrix(
            self._system.state_jacobian(t, x, theta, seg), d, d, "state_jacobian"
        )
        if p == 0 or self._system.parameter_jacobian is None:
            return a, [[] for _ in range(d)]
        b = _light_matrix(
            self._system.parameter_jacobian(t, x, theta, seg),
            d,
            p,
            "parameter_jacobian",
        )
        return a, b

    # -- one RK4 step: value, tangent, adjoint -----------------------------

    def _step(self, g: int, x: Vector, theta: Vector) -> Vector:
        """Advance one classical RK4 step.

        References:
            [EQ-INV-ODE-001].

        """
        t, h, seg = self._schedule[g]
        self._counts[3] += 1
        f = self._f
        k1 = f(t, x, theta, seg)
        x2 = tuple(a + 0.5 * h * b for a, b in zip(x, k1, strict=True))
        k2 = f(t + 0.5 * h, x2, theta, seg)
        x3 = tuple(a + 0.5 * h * b for a, b in zip(x, k2, strict=True))
        k3 = f(t + 0.5 * h, x3, theta, seg)
        x4 = tuple(a + h * b for a, b in zip(x, k3, strict=True))
        k4 = f(t + h, x4, theta, seg)
        return tuple(
            a + h / 6.0 * (b + 2.0 * c + 2.0 * e + r)
            for a, b, c, e, r in zip(x, k1, k2, k3, k4, strict=True)
        )

    def _tangent_step(
        self, g: int, x: Vector, s: Vector, theta: Vector, v_theta: Vector
    ) -> tuple[Vector, Vector]:
        """Advance ``(x, s)`` by one RK4 step of the variational equation.

        References:
            [EQ-INV-ODE-002].

        """
        t, h, seg = self._schedule[g]
        self._counts[3] += 1
        times = (t, t + 0.5 * h, t + 0.5 * h, t + h)
        weights = (0.0, 0.5 * h, 0.5 * h, h)
        states: Vector = x
        tang: Vector = s
        ks: list[list[float]] = []
        dks: list[list[float]] = []
        for i in range(4):
            if i > 0:
                states = tuple(
                    a + weights[i] * b for a, b in zip(x, ks[-1], strict=True)
                )
                tang = tuple(
                    a + weights[i] * b for a, b in zip(s, dks[-1], strict=True)
                )
            k = self._f(times[i], states, theta, seg)
            a_mat, b_mat = self._jac(times[i], states, theta, seg)
            dk = _matvec(a_mat, tang)
            if v_theta:
                dk = [u + w for u, w in zip(dk, _matvec(b_mat, v_theta), strict=True)]
            ks.append(k)
            dks.append(dk)
        x_next = tuple(
            a + h / 6.0 * (b + 2.0 * c + 2.0 * e + r)
            for a, b, c, e, r in zip(x, *ks, strict=True)
        )
        s_next = tuple(
            a + h / 6.0 * (b + 2.0 * c + 2.0 * e + r)
            for a, b, c, e, r in zip(s, *dks, strict=True)
        )
        return x_next, s_next

    def _adjoint_step(
        self, g: int, x: Vector, theta: Vector, a: list[float], theta_bar: list[float]
    ) -> list[float]:
        """Return the adjoint of the step input and accumulate ``theta_bar``.

        This is the transpose of :meth:`_tangent_step`: the stages are rebuilt
        from ``x`` and the stage adjoints are swept from the fourth to the
        first.

        References:
            [EQ-INV-ODE-003].

        """
        t, h, seg = self._schedule[g]
        d = self._system.state_dimension
        p = self._system.parameter_dimension
        times = (t, t + 0.5 * h, t + 0.5 * h, t + h)
        stage_x: list[Vector] = [x]
        for i in range(1, 4):
            k_prev = self._f(times[i - 1], stage_x[-1], theta, seg)
            scale = h if i == 3 else 0.5 * h
            stage_x.append(tuple(u + scale * w for u, w in zip(x, k_prev, strict=True)))
        k_bar = [
            [h / 6.0 * u for u in a],
            [h / 3.0 * u for u in a],
            [h / 3.0 * u for u in a],
            [h / 6.0 * u for u in a],
        ]
        x_bar = list(a)
        feed = (0.0, 0.5 * h, 0.5 * h, h)
        for i in (3, 2, 1, 0):
            a_mat, b_mat = self._jac(times[i], stage_x[i], theta, seg)
            xb = _rmatvec(a_mat, k_bar[i], d)
            if p:
                gb = _rmatvec(b_mat, k_bar[i], p)
                for j in range(p):
                    theta_bar[j] += gb[j]
            for j in range(d):
                x_bar[j] += xb[j]
            if i > 0:
                for j in range(d):
                    k_bar[i - 1][j] += feed[i] * xb[j]
        return x_bar

    # -- public maps ---------------------------------------------------------

    def _observe(self, x: Vector) -> list[float]:
        """Return ``H x``.

        References:
            [EQ-INV-ODE-001].

        """
        return _matvec(self._obs_matrix, x)

    def _trajectory(self, x0: Vector, theta: Vector) -> list[Vector]:
        """Return the states at every observation, integrating once.

        References:
            [EQ-INV-ODE-001].

        """
        out: list[Vector] = []
        x = x0
        for k in range(self._total_steps + 1):
            if k > 0:
                x = self._step(k - 1, x, theta)
            for _ in self._obs_at.get(k, ()):
                out.append(x)
        return out

    def states(self, x: Sequence[float]) -> tuple[Vector, ...]:
        """Return the full states ``x(t_j)`` at the output times.

        Args:
            x: Parameters ``z`` (length ``n``).

        Returns:
            One state tuple per output time.

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length or is not finite.

        References:
            [EQ-INV-ODE-001].

        """
        x0, theta = self._split(x)
        self._counts[0] += 1
        return tuple(self._trajectory(x0, theta))

    def value(self, x: Sequence[float]) -> Vector:
        """Return the stacked observations ``G(z)``.

        Args:
            x: Parameters ``z`` (length ``n``).

        Returns:
            ``(H x(t_1), ..., H x(t_J))``; entries are non-finite if the
            integration blew up.

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length or is not finite.

        References:
            [EQ-INV-ODE-001].

        """
        out: list[float] = []
        for state in self.states(x):
            out.extend(self._observe(state))
        return tuple(out)

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return the discrete tangent ``J(z) v`` (forward sensitivity).

        Args:
            x: Base point ``z``.
            v: Direction (length ``n``).

        Returns:
            The exact directional derivative of :meth:`value`.

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, an input is not finite or the
                derivative is not finite.

        References:
            [EQ-INV-ODE-002].

        """
        x0, theta = self._split(x)
        direction = _vector(v, self.input_dimension, "v")
        d = self._system.state_dimension
        if self._estimate:
            s: Vector = direction[:d]
            v_theta = direction[d:]
        else:
            s = (0.0,) * d
            v_theta = direction
        self._counts[1] += 1
        state = x0
        out: list[float] = []
        for k in range(self._total_steps + 1):
            if k > 0:
                state, s = self._tangent_step(k - 1, state, s, theta, v_theta)
            for _ in self._obs_at.get(k, ()):
                out.extend(self._observe(s))
        if not _is_finite(out):
            raise ValueError("the tangent is not finite (integration blew up)")
        return tuple(out)

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return the exact discrete adjoint ``J(z)' w``.

        A forward sweep stores the states (every ``checkpoint_every``-th one),
        then a reverse sweep transposes the RK4 steps, recomputing the states
        of each checkpoint block when ``checkpoint_every > 1``.

        Args:
            x: Base point ``z``.
            w: Adjoint direction (length ``m``).

        Returns:
            ``J' w`` (length ``n``).

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, an input is not finite or the
                result is not finite.

        References:
            [EQ-INV-ODE-003], [EQ-INV-ODE-004].

        """
        x0, theta = self._split(x)
        weights = _vector(w, self.output_dimension, "w")
        d = self._system.state_dimension
        p = self._system.parameter_dimension
        total = self._total_steps
        c = self._checkpoint
        self._counts[2] += 1
        stored: dict[int, Vector] = {0: x0}
        state = x0
        for k in range(1, total + 1):
            state = self._step(k - 1, state, theta)
            if k % c == 0:
                stored[k] = state
        q = self._obs_rows

        def observation_adjoint(k: int, a: list[float]) -> None:
            """Add ``H' w_j`` for every output time observed at state ``k``.

            References:
                [EQ-INV-ODE-003].

            """
            for j in self._obs_at.get(k, ()):
                chunk = weights[j * q : (j + 1) * q]
                back = _rmatvec(self._obs_matrix, chunk, d)
                for i in range(d):
                    a[i] += back[i]

        a = [0.0] * d
        theta_bar = [0.0] * p
        block_starts = list(range(0, total, c))
        for b in reversed(block_starts):
            e = min(b + c, total)
            block = [stored[b]]
            for k in range(b, e - 1):
                block.append(self._step(k, block[-1], theta))
                self._counts[4] += 1
            self._peak = max(self._peak, len(stored) + len(block))
            for k in range(e, b, -1):
                observation_adjoint(k, a)
                a = self._adjoint_step(k - 1, block[k - 1 - b], theta, a, theta_bar)
        observation_adjoint(0, a)
        result = (*a, *theta_bar) if self._estimate else tuple(theta_bar)
        if not _is_finite(result):
            raise ValueError("the adjoint is not finite (integration blew up)")
        return tuple(result)


# ---------------------------------------------------------------------------
# external simulator adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimulatorFailure:
    """A recorded failure of an external simulator call.

    Attributes:
        operation: ``"value"``, ``"jvp"`` or ``"vjp"``.
        kind: ``"exception"``, ``"non_finite_output"`` or
            ``"invalid_output"`` (wrong length or non-real entries).
        message: Human-readable description.
        exception_type: Class name of the captured exception, else ``None``.
        call_index: One-based number of the simulator call (all operations
            counted together) that failed.
        point: The input point ``x`` of the failed call.

    References:
        [EQ-INV-ODE-005].

    """

    operation: str
    kind: str
    message: str
    exception_type: str | None
    call_index: int
    point: Vector


class SimulatorFailureError(RuntimeError):
    """Raised when a simulator call fails and ``on_failure`` is ``"raise"``.

    Attributes:
        failure: The recorded :class:`SimulatorFailure`.

    References:
        [EQ-INV-ODE-005].

    """

    def __init__(self, failure: SimulatorFailure) -> None:
        """Store the failure and build the message.

        Args:
            failure: The failure to report.

        References:
            [EQ-INV-ODE-005].

        """
        super().__init__(
            f"simulator {failure.operation} failed ({failure.kind}): {failure.message}"
        )
        self.failure = failure


@dataclass(frozen=True, slots=True)
class SimulatorDiagnostics:
    """Bookkeeping of a :class:`SimulatorAdapter`.

    Attributes:
        value_calls: Executions of the user ``forward`` callable made on
            behalf of ``value``.
        jvp_calls: Executions of the user ``jvp`` callable.
        vjp_calls: Executions of the user ``vjp`` callable.
        fd_value_calls: Executions of ``forward`` spent inside central
            differences.
        cache_hits: Requests answered from the cache.
        failures: Number of recorded failures.
        elapsed: Wall seconds spent inside user callables (not deterministic).

    References:
        [EQ-INV-ODE-005].

    """

    value_calls: int
    jvp_calls: int
    vjp_calls: int
    fd_value_calls: int
    cache_hits: int
    failures: int
    elapsed: float


class SimulatorAdapter:
    """Wrap an external simulator as a :class:`DifferentiableForward` model.

    Args are validated and every call is guarded: the callable receives a
    plain tuple, its output is checked for type, length and finiteness, and a
    problem is recorded as a :class:`SimulatorFailure` (``failures`` property)
    and either raised as :class:`SimulatorFailureError` (``on_failure="raise"``,
    the default) or, for ``value`` only, returned as an all-``nan`` vector
    (``on_failure="record"``) so that a line search can reject the trial.
    ``jvp``/``vjp`` failures always raise. Nothing is swallowed silently.

    Derivatives come from the user callbacks when given. A derivative without
    a callback is available only if ``finite_difference_step`` is declared:
    ``J v`` by the central difference with the scale-aware size
    ``e = step (1 + ||x||) / ||v||`` (two ``forward`` calls) and, only with
    ``allow_dense_fd_vjp=True``, ``J' w`` from the ``n`` Jacobian columns
    (``2 n`` calls) [EQ-INV-ODE-005]. Otherwise the call raises ``ValueError``;
    there is no silent fallback. ``cache_size > 0`` enables an LRU cache of
    successful results keyed by the exact inputs; it is OFF by default because
    a stochastic or stateful simulator must not be cached.

    References:
        [EQ-INV-ODE-005], [REF-NOCEDAL-WRIGHT-2006].

    """

    __slots__ = (
        "_allow_dense",
        "_cache",
        "_cache_size",
        "_calls",
        "_failures",
        "_forward",
        "_jvp",
        "_m",
        "_n",
        "_on_failure",
        "_step",
        "_vjp",
        "_elapsed",
        "_fd_calls",
        "_hits",
        "_serial",
    )

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        forward: Callable[[Vector], Sequence[float]],
        *,
        jvp: Callable[[Vector, Vector], Sequence[float]] | None = None,
        vjp: Callable[[Vector, Vector], Sequence[float]] | None = None,
        finite_difference_step: float | None = None,
        allow_dense_fd_vjp: bool = False,
        on_failure: Literal["raise", "record"] = "raise",
        cache_size: int = 0,
    ) -> None:
        """Store the callables and the policy.

        Args:
            input_dimension: ``n >= 1``.
            output_dimension: ``m >= 1``.
            forward: ``x -> G(x)`` (tuple in, sequence of ``m`` reals out).
            jvp: Optional ``(x, v) -> J(x) v``.
            vjp: Optional ``(x, w) -> J(x)' w``.
            finite_difference_step: Declared relative central-difference step
                (``> 0``); enables FD for the derivatives without a callback.
            allow_dense_fd_vjp: Permit the dense ``2 n``-call FD ``vjp``
                (needs ``finite_difference_step``).
            on_failure: ``"raise"`` or ``"record"`` (``value`` only).
            cache_size: ``0`` disables the cache, else its capacity.

        Raises:
            TypeError: If a callable or dimension has the wrong type.
            ValueError: If an option is out of range or inconsistent.

        References:
            [EQ-INV-ODE-005].

        """
        self._n = _int_at_least(input_dimension, "input_dimension")
        self._m = _int_at_least(output_dimension, "output_dimension")
        for name, fn in (("forward", forward), ("jvp", jvp), ("vjp", vjp)):
            if fn is not None and not callable(fn):
                raise TypeError(f"{name} must be callable")
        if not callable(forward):
            raise TypeError("forward must be callable")
        if on_failure not in ("raise", "record"):
            raise ValueError("on_failure must be 'raise' or 'record'")
        self._step = (
            None
            if finite_difference_step is None
            else _real(finite_difference_step, "finite_difference_step", positive=True)
        )
        if allow_dense_fd_vjp and self._step is None:
            raise ValueError("allow_dense_fd_vjp needs finite_difference_step")
        self._allow_dense = bool(allow_dense_fd_vjp)
        self._cache_size = _int_at_least(cache_size, "cache_size", 0)
        self._cache: OrderedDict[tuple[object, ...], Vector] = OrderedDict()
        self._forward = forward
        self._jvp = jvp
        self._vjp = vjp
        self._on_failure = on_failure
        self._failures: list[SimulatorFailure] = []
        self._calls = [0, 0, 0]
        self._fd_calls = 0
        self._hits = 0
        self._elapsed = 0.0
        self._serial = 0

    @property
    def input_dimension(self) -> int:
        """Return the parameter dimension ``n``.

        References:
            [EQ-INV-ODE-005].

        """
        return self._n

    @property
    def output_dimension(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [EQ-INV-ODE-005].

        """
        return self._m

    @property
    def failures(self) -> tuple[SimulatorFailure, ...]:
        """Return all recorded failures in call order.

        References:
            [EQ-INV-ODE-005].

        """
        return tuple(self._failures)

    @property
    def counts(self) -> ModelCounts:
        """Return the evaluation counts (``derivative_value`` = FD ``forward`` calls).

        References:
            [EQ-INV-ODE-005].

        """
        return ModelCounts(
            value=self._calls[0],
            jvp=self._calls[1],
            vjp=self._calls[2],
            derivative_value=self._fd_calls,
        )

    @property
    def diagnostics(self) -> SimulatorDiagnostics:
        """Return counts, cache hits, failures and callable wall time.

        References:
            [EQ-INV-ODE-005].

        """
        return SimulatorDiagnostics(
            value_calls=self._calls[0],
            jvp_calls=self._calls[1],
            vjp_calls=self._calls[2],
            fd_value_calls=self._fd_calls,
            cache_hits=self._hits,
            failures=len(self._failures),
            elapsed=self._elapsed,
        )

    @property
    def derivative_modes(self) -> dict[str, str]:
        """Return ``"analytic"``, ``"finite_difference"`` or ``"unavailable"``.

        References:
            [EQ-INV-ODE-005].

        """
        jvp = (
            "analytic"
            if self._jvp is not None
            else ("finite_difference" if self._step is not None else "unavailable")
        )
        vjp = (
            "analytic"
            if self._vjp is not None
            else ("finite_difference" if self._allow_dense else "unavailable")
        )
        return {"jvp": jvp, "vjp": vjp}

    def clear_cache(self) -> None:
        """Empty the cache.

        References:
            [EQ-INV-ODE-005].

        """
        self._cache.clear()

    def _fail(
        self,
        operation: str,
        kind: str,
        message: str,
        exc: str | None,
        point: Vector,
    ) -> SimulatorFailure:
        """Record and return a failure.

        References:
            [EQ-INV-ODE-005].

        """
        failure = SimulatorFailure(operation, kind, message, exc, self._serial, point)
        self._failures.append(failure)
        return failure

    def _call(
        self,
        operation: str,
        fn: Callable[..., Sequence[float]],
        args: tuple[Vector, ...],
        size: int,
    ) -> Vector:
        """Run a user callable under guard and validate its output.

        References:
            [EQ-INV-ODE-005].

        """
        self._serial += 1
        started = time.perf_counter()
        try:
            raw = fn(*args)
        except Exception as error:  # noqa: BLE001 - recorded, then re-raised
            self._elapsed += time.perf_counter() - started
            raise SimulatorFailureError(
                self._fail(
                    operation, "exception", str(error), type(error).__name__, args[0]
                )
            ) from error
        self._elapsed += time.perf_counter() - started
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise SimulatorFailureError(
                self._fail(
                    operation,
                    "invalid_output",
                    "output is not a sequence",
                    None,
                    args[0],
                )
            )
        if len(raw) != size:
            raise SimulatorFailureError(
                self._fail(
                    operation,
                    "invalid_output",
                    f"output has length {len(raw)}, expected {size}",
                    None,
                    args[0],
                )
            )
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
            raise SimulatorFailureError(
                self._fail(
                    operation,
                    "invalid_output",
                    "output has non-real entries",
                    None,
                    args[0],
                )
            )
        out = tuple(float(v) for v in raw)
        if not _is_finite(out):
            raise SimulatorFailureError(
                self._fail(
                    operation,
                    "non_finite_output",
                    "output has non-finite entries",
                    None,
                    args[0],
                )
            )
        return out

    def _cached(
        self,
        key: tuple[object, ...],
        compute: Callable[[], Vector],
    ) -> Vector:
        """Return a cached result or compute and store it.

        References:
            [EQ-INV-ODE-005].

        """
        if self._cache_size == 0:
            return compute()
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        result = compute()
        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    def _forward_call(self, point: Vector, *, inside_fd: bool) -> Vector:
        """Run ``forward`` and count it.

        References:
            [EQ-INV-ODE-005].

        """
        if inside_fd:
            self._fd_calls += 1
        else:
            self._calls[0] += 1
        return self._call("value", self._forward, (point,), self._m)

    def value(self, x: Sequence[float]) -> Vector:
        """Return ``G(x)`` from the simulator.

        Args:
            x: Parameters (length ``n``, finite reals).

        Returns:
            The validated output, or an all-``nan`` vector when the call
            failed and ``on_failure="record"``.

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length or is not finite.
            SimulatorFailureError: If the call failed and ``on_failure`` is
                ``"raise"``.

        References:
            [EQ-INV-ODE-005].

        """
        point = _vector(x, self._n, "x")
        try:
            return self._cached(
                ("value", point), lambda: self._forward_call(point, inside_fd=False)
            )
        except SimulatorFailureError:
            if self._on_failure == "record":
                return (math.nan,) * self._m
            raise

    def _fd_jvp(self, point: Vector, direction: Vector) -> Vector:
        """Return the central-difference ``J v`` with the declared step.

        References:
            [EQ-INV-ODE-005].

        """
        assert self._step is not None
        norm_v = math.hypot(*direction)
        if norm_v == 0.0:
            return (0.0,) * self._m
        e = self._step * (1.0 + math.hypot(*point)) / norm_v
        plus = tuple(a + e * b for a, b in zip(point, direction, strict=True))
        minus = tuple(a - e * b for a, b in zip(point, direction, strict=True))
        fp = self._forward_call(plus, inside_fd=True)
        fm = self._forward_call(minus, inside_fd=True)
        return tuple((a - b) / (2.0 * e) for a, b in zip(fp, fm, strict=True))

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return ``J(x) v`` (analytic callback or declared central difference).

        Args:
            x: Base point.
            v: Direction (length ``n``).

        Returns:
            ``J v`` (length ``m``).

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, an input is not finite or no
                derivative mode is available.
            SimulatorFailureError: If a simulator call failed.

        References:
            [EQ-INV-ODE-005].

        """
        point = _vector(x, self._n, "x")
        direction = _vector(v, self._n, "v")
        if self._jvp is not None:
            user = self._jvp
            self._calls[1] += 1
            return self._cached(
                ("jvp", point, direction),
                lambda: self._call("jvp", user, (point, direction), self._m),
            )
        if self._step is None:
            raise ValueError("no jvp callback and no finite_difference_step declared")
        self._calls[1] += 1
        return self._cached(
            ("fdjvp", point, direction), lambda: self._fd_jvp(point, direction)
        )

    def _fd_vjp(self, point: Vector, weights: Vector) -> Vector:
        """Return ``J' w`` from ``n`` central-difference Jacobian columns.

        References:
            [EQ-INV-ODE-005].

        """
        out: list[float] = []
        for j in range(self._n):
            unit = tuple(1.0 if i == j else 0.0 for i in range(self._n))
            column = self._fd_jvp(point, unit)
            out.append(_dot(column, weights))
        return tuple(out)

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return ``J(x)' w`` (analytic callback or dense finite differences).

        Args:
            x: Base point.
            w: Adjoint direction (length ``m``).

        Returns:
            ``J' w`` (length ``n``).

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, an input is not finite or no
                derivative mode is available.
            SimulatorFailureError: If a simulator call failed.

        References:
            [EQ-INV-ODE-005].

        """
        point = _vector(x, self._n, "x")
        weights = _vector(w, self._m, "w")
        if self._vjp is not None:
            user = self._vjp
            self._calls[2] += 1
            return self._cached(
                ("vjp", point, weights),
                lambda: self._call("vjp", user, (point, weights), self._n),
            )
        if not self._allow_dense:
            raise ValueError(
                "no vjp callback; a dense finite-difference vjp needs "
                "finite_difference_step and allow_dense_fd_vjp=True"
            )
        self._calls[2] += 1
        return self._cached(
            ("fdvjp", point, weights), lambda: self._fd_vjp(point, weights)
        )


# ---------------------------------------------------------------------------
# heat equation with unknown log-diffusivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeSteppingCounts:
    """Work counters of a :class:`HeatCoefficientModel`.

    Attributes:
        factorizations: Tridiagonal factorizations (one per evaluation that
            needs the operator).
        forward_solves: Tridiagonal solves spent on forward time steps
            (including those repeated inside ``jvp``/``vjp``).
        tangent_solves: Tridiagonal solves of the tangent recursion.
        adjoint_solves: Tridiagonal solves of the adjoint recursion.

    References:
        [EQ-INV-ODE-007].

    """

    factorizations: int
    forward_solves: int
    tangent_solves: int
    adjoint_solves: int


def _cosine_basis(size: int) -> list[Callable[[float], float]]:
    """Return ``cos(k pi s)`` for ``k = 0..size-1``.

    References:
        [EQ-INV-ODE-006].

    """

    def make(k: int) -> Callable[[float], float]:
        """Return the ``k``-th cosine.

        References:
            [EQ-INV-ODE-006].

        """

        def phi(s: float) -> float:
            """Return ``cos(k pi s)``.

            References:
                [EQ-INV-ODE-006].

            """
            return math.cos(k * math.pi * s)

        return phi

    return [make(k) for k in range(size)]


class HeatCoefficientModel:
    """Heat equation ``u_t = (kappa u_s)_s`` with unknown log-diffusivity.

    On ``(0, 1)`` with ``u = 0`` at both ends, ``N`` interior nodes
    (``h = 1/(N+1)``) and ``kappa_f = exp(sum_k c_k phi_k(s_f))`` at the
    ``N + 1`` face midpoints, the semi-discrete operator is the symmetric
    tridiagonal conservative stencil ``A(kappa)`` and the time step is
    implicit Euler ``(I + dt A) u^(j+1) = u^j`` [EQ-INV-ODE-006]. The unknown
    is the coefficient vector ``c``; the data are ``u`` interpolated linearly
    at the sensors for the chosen steps. ``A`` is linear in ``kappa``, so the
    tangent is ``(I + dt A) du^(j+1) = du^j - dt A(dkappa) u^(j+1)`` and the
    exact discrete adjoint reuses the same symmetric factorization at every
    step [EQ-INV-ODE-007]. Implicit Euler is unconditionally stable, so there
    is no time-step restriction to violate; its first-order time error is
    part of the discretization error against a finer truth.

    A coefficient whose exponential overflows or underflows to zero makes the
    forward evaluation FAIL: ``value`` returns an all-``nan`` vector (counted
    in :attr:`failed_evaluations`) and ``jvp``/``vjp`` raise ``ValueError``.

    References:
        [EQ-INV-ODE-006], [EQ-INV-ODE-007], [REF-KAIPIO-SOMERSALO-2007],
        [REF-GRIEWANK-WALTHER-2008].

    """

    __slots__ = (
        "_basis",
        "_counts",
        "_dt",
        "_failed",
        "_h",
        "_n",
        "_nodes",
        "_obs_steps",
        "_phi",
        "_sensors",
        "_steps",
        "_u0",
    )

    def __init__(
        self,
        interior_nodes: int,
        time_steps: int,
        final_time: float,
        sensors: Sequence[float],
        observation_steps: Sequence[int],
        initial_condition: Callable[[float], float],
        *,
        basis_size: int = 3,
        basis: Sequence[Callable[[float], float]] | None = None,
    ) -> None:
        """Validate and assemble the fixed model data.

        Args:
            interior_nodes: ``N >= 2``.
            time_steps: Number of implicit Euler steps ``>= 1``.
            final_time: Horizon ``T > 0``; ``dt = T / time_steps``.
            sensors: Positions strictly inside ``(0, 1)`` (at least one).
            observation_steps: Strictly increasing step indices in
                ``1..time_steps`` at which the sensors are read.
            initial_condition: ``s -> u(0, s)`` evaluated at the nodes.
            basis_size: Number of cosine basis functions when ``basis`` is
                ``None``.
            basis: Optional functions ``phi_k(s)`` of the log-diffusivity.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range or a callable returns
                a non-finite value.

        References:
            [EQ-INV-ODE-006].

        """
        self._n = _int_at_least(interior_nodes, "interior_nodes", 2)
        self._steps = _int_at_least(time_steps, "time_steps")
        horizon = _real(final_time, "final_time", positive=True)
        self._dt = horizon / self._steps
        self._h = 1.0 / (self._n + 1)
        if not callable(initial_condition):
            raise TypeError("initial_condition must be callable")
        self._u0 = tuple(
            _real(initial_condition((i + 1) * self._h), "initial_condition value")
            for i in range(self._n)
        )
        if isinstance(sensors, (str, bytes)) or not isinstance(sensors, Sequence):
            raise TypeError("sensors must be a sequence of real numbers")
        pos = _vector(sensors, len(sensors), "sensors")
        if not pos or not all(0.0 < s < 1.0 for s in pos):
            raise ValueError("sensors must be non-empty and lie strictly inside (0, 1)")
        self._sensors = pos
        steps = tuple(observation_steps)
        previous = 0
        for s in steps:
            _int_at_least(s, "observation step")
            if s <= previous or s > self._steps:
                raise ValueError(
                    "observation_steps must be strictly increasing within 1..time_steps"
                )
            previous = s
        if not steps:
            raise ValueError("observation_steps must be non-empty")
        self._obs_steps = steps
        functions = (
            _cosine_basis(_int_at_least(basis_size, "basis_size"))
            if basis is None
            else list(basis)
        )
        if not functions or not all(callable(f) for f in functions):
            raise TypeError("basis must be a non-empty sequence of callables")
        self._basis = functions
        self._phi = [
            [_real(f((j + 0.5) * self._h), "basis value") for f in functions]
            for j in range(self._n + 1)
        ]
        self._nodes = tuple(self._locate(s) for s in pos)
        self._counts = [0, 0, 0, 0, 0, 0, 0]
        self._failed = 0

    def _locate(self, s: float) -> tuple[int, float, float]:
        """Return ``(left node, weight_left, weight_right)`` of a sensor.

        References:
            [EQ-INV-ODE-006].

        """
        i = min(int(s / self._h), self._n)
        t = s / self._h - i
        return i, 1.0 - t, t

    @property
    def input_dimension(self) -> int:
        """Return the number of basis coefficients.

        References:
            [EQ-INV-ODE-006].

        """
        return len(self._basis)

    @property
    def output_dimension(self) -> int:
        """Return ``(#observation steps) x (#sensors)``.

        References:
            [EQ-INV-ODE-006].

        """
        return len(self._obs_steps) * len(self._sensors)

    @property
    def failed_evaluations(self) -> int:
        """Return how many ``value`` calls failed (overflow or ``kappa = 0``).

        References:
            [EQ-INV-ODE-006].

        """
        return self._failed

    @property
    def counts(self) -> ModelCounts:
        """Return the evaluation counts (``value``, ``jvp``, ``vjp``).

        References:
            [EQ-INV-ODE-007].

        """
        return ModelCounts(
            value=self._counts[0],
            jvp=self._counts[1],
            vjp=self._counts[2],
            derivative_value=0,
        )

    @property
    def work(self) -> TimeSteppingCounts:
        """Return the tridiagonal factorization and solve counters.

        References:
            [EQ-INV-ODE-007].

        """
        c = self._counts
        return TimeSteppingCounts(c[3], c[4], c[5], c[6])

    def diffusivity(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return ``kappa`` at the ``N + 1`` face midpoints.

        Args:
            x: Coefficients ``c`` (length ``K``).

        Returns:
            ``exp(sum_k c_k phi_k)`` per face.

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length, is not finite, or the
                exponential overflows or underflows to zero.

        References:
            [EQ-INV-ODE-006].

        """
        c = _vector(x, self.input_dimension, "x")
        out: list[float] = []
        for row in self._phi:
            try:
                kappa = math.exp(sum(a * b for a, b in zip(c, row, strict=True)))
            except OverflowError as error:
                raise ValueError("diffusivity overflows") from error
            if not (0.0 < kappa < math.inf):
                raise ValueError("diffusivity is not positive and finite")
            out.append(kappa)
        return tuple(out)

    def _apply_a(self, kappa: Sequence[float], u: Sequence[float]) -> list[float]:
        """Return ``A(kappa) u`` with zero boundary values.

        References:
            [EQ-INV-ODE-006].

        """
        n = self._n
        inv = 1.0 / (self._h * self._h)
        out = [0.0] * n
        for i in range(n):
            left = u[i - 1] if i > 0 else 0.0
            right = u[i + 1] if i < n - 1 else 0.0
            out[i] = inv * (kappa[i] * (u[i] - left) + kappa[i + 1] * (u[i] - right))
        return out

    def _factorize(self, kappa: Sequence[float]) -> TridiagonalFactorization:
        """Factorize ``M = I + dt A(kappa)``.

        References:
            [EQ-INV-ODE-006].

        """
        scale = self._dt / (self._h * self._h)
        diagonal = [1.0 + scale * (kappa[i] + kappa[i + 1]) for i in range(self._n)]
        off = [-scale * kappa[i + 1] for i in range(self._n - 1)]
        self._counts[3] += 1
        return TridiagonalFactorization(off, diagonal, off)

    def _observe(self, u: Sequence[float]) -> list[float]:
        """Return the linearly interpolated sensor readings.

        References:
            [EQ-INV-ODE-006].

        """
        out: list[float] = []
        for i, w0, w1 in self._nodes:
            left = u[i - 1] if i >= 1 else 0.0
            right = u[i] if i < self._n else 0.0
            out.append(w0 * left + w1 * right)
        return out

    def _observe_transpose(self, w: Sequence[float], out: list[float]) -> None:
        """Add ``H' w`` into ``out``.

        References:
            [EQ-INV-ODE-007].

        """
        for (i, w0, w1), value in zip(self._nodes, w, strict=True):
            if i >= 1:
                out[i - 1] += w0 * value
            if i < self._n:
                out[i] += w1 * value

    def _forward(
        self, kappa: Sequence[float], fact: TridiagonalFactorization
    ) -> list[list[float]]:
        """Return ``[u^0, ..., u^N]``.

        References:
            [EQ-INV-ODE-006].

        """
        states = [list(self._u0)]
        for _ in range(self._steps):
            states.append(fact.solve(states[-1]))
            self._counts[4] += 1
        return states

    def value(self, x: Sequence[float]) -> Vector:
        """Return the stacked sensor readings (step-major, sensor-minor).

        Args:
            x: Coefficients ``c``.

        Returns:
            The data vector, or all ``nan`` if the diffusivity is invalid
            (counted in :attr:`failed_evaluations`).

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length or is not finite.

        References:
            [EQ-INV-ODE-006].

        """
        c = _vector(x, self.input_dimension, "x")
        self._counts[0] += 1
        try:
            kappa = self.diffusivity(c)
        except ValueError:
            self._failed += 1
            return (math.nan,) * self.output_dimension
        states = self._forward(kappa, self._factorize(kappa))
        out: list[float] = []
        for step in self._obs_steps:
            out.extend(self._observe(states[step]))
        return tuple(out)

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return the discrete tangent ``J(c) v``.

        Args:
            x: Base coefficients.
            v: Direction (length ``K``).

        Returns:
            The exact directional derivative of :meth:`value`.

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, an input is not finite or the
                diffusivity is invalid.

        References:
            [EQ-INV-ODE-007].

        """
        c = _vector(x, self.input_dimension, "x")
        direction = _vector(v, self.input_dimension, "v")
        kappa = self.diffusivity(c)
        dkappa = [
            k * sum(a * b for a, b in zip(direction, row, strict=True))
            for k, row in zip(kappa, self._phi, strict=True)
        ]
        self._counts[1] += 1
        fact = self._factorize(kappa)
        states = self._forward(kappa, fact)
        du = [0.0] * self._n
        wanted = set(self._obs_steps)
        out: list[float] = []
        for j in range(1, self._steps + 1):
            source = self._apply_a(dkappa, states[j])
            rhs = [a - self._dt * b for a, b in zip(du, source, strict=True)]
            du = fact.solve(rhs)
            self._counts[5] += 1
            if j in wanted:
                out.extend(self._observe(du))
        return tuple(out)

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return the exact discrete adjoint ``J(c)' w``.

        Args:
            x: Base coefficients.
            w: Adjoint direction (length ``m``).

        Returns:
            ``J' w`` (length ``K``).

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, an input is not finite or the
                diffusivity is invalid.

        References:
            [EQ-INV-ODE-007].

        """
        c = _vector(x, self.input_dimension, "x")
        weights = _vector(w, self.output_dimension, "w")
        kappa = self.diffusivity(c)
        self._counts[2] += 1
        fact = self._factorize(kappa)
        states = self._forward(kappa, fact)
        sensors = len(self._sensors)
        index = {s: i for i, s in enumerate(self._obs_steps)}
        lam = [0.0] * self._n
        kappa_bar = [0.0] * (self._n + 1)
        inv = 1.0 / (self._h * self._h)
        for j in range(self._steps, 0, -1):
            if j in index:
                i = index[j]
                self._observe_transpose(weights[i * sensors : (i + 1) * sensors], lam)
            p = fact.solve(lam)
            self._counts[6] += 1
            u = states[j]
            for f in range(self._n + 1):
                dp = (p[f] if f < self._n else 0.0) - (p[f - 1] if f > 0 else 0.0)
                du = (u[f] if f < self._n else 0.0) - (u[f - 1] if f > 0 else 0.0)
                kappa_bar[f] -= self._dt * inv * dp * du
            lam = p
        weights_c = [k * kb for k, kb in zip(kappa, kappa_bar, strict=True)]
        return tuple(
            sum(row[k] * wc for row, wc in zip(self._phi, weights_c, strict=True))
            for k in range(self.input_dimension)
        )


def _default_initial_condition(s: float) -> float:
    """Return ``sin(pi s) + 0.5 sin(3 pi s)``, the benchmark initial state.

    References:
        [EQ-INV-ODE-006].

    """
    return math.sin(math.pi * s) + 0.5 * math.sin(3.0 * math.pi * s)


def heat_coefficient_benchmark(
    *,
    interior_nodes: int = 15,
    time_steps: int = 20,
    final_time: float = 0.1,
    sensors: Sequence[float] = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9),
    observation_steps: Sequence[int] = (5, 10, 15, 20),
    truth: Sequence[float] = (0.3, -0.4, 0.2),
    noise_sd: float = 1e-3,
    space_refinement: int = 4,
    time_refinement: int = 4,
    seed: int | RandomStream | None = 11,
) -> NonlinearBenchmark:
    """Return a coefficient-inversion case (identifier ``"P02C"``).

    The solver model is a coarse :class:`HeatCoefficientModel`; the data come
    from the SAME continuous problem on a grid refined by ``space_refinement``
    in space (``(N+1) r - 1`` interior nodes, same sensors) and
    ``time_refinement`` in time, so the data carry a genuine discretization
    error and ``same_discretization`` is ``False``; with both refinements
    equal to 1 the data are generated by the solver's own model (inverse
    crime, algebraic verification only) and the flag is ``True``
    [EQ-INV-ODE-006]. Noise is iid Gaussian with the absolute standard
    deviation ``noise_sd`` from its own stream, not rescaled.

    Args:
        interior_nodes: Coarse interior nodes ``N >= 2``.
        time_steps: Coarse implicit Euler steps.
        final_time: Horizon.
        sensors: Sensor positions in ``(0, 1)``.
        observation_steps: Coarse steps at which the sensors are read.
        truth: True coefficients ``c`` of the cosine basis.
        noise_sd: Noise standard deviation (``> 0``).
        space_refinement: Spatial refinement factor of the data grid.
        time_refinement: Temporal refinement factor of the data grid.
        seed: Root seed (only the noise is random).

    Returns:
        :class:`~probstat.inverse.benchmark_extended.NonlinearBenchmark`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-ODE-006], [REF-KAIPIO-SOMERSALO-2007].

    """
    sigma = _real(noise_sd, "noise_sd", positive=True)
    rs = _int_at_least(space_refinement, "space_refinement")
    rt = _int_at_least(time_refinement, "time_refinement")
    coefficients = _vector(truth, len(truth), "truth")
    model = HeatCoefficientModel(
        interior_nodes,
        time_steps,
        final_time,
        sensors,
        observation_steps,
        _default_initial_condition,
        basis_size=len(coefficients),
    )
    generator = HeatCoefficientModel(
        (interior_nodes + 1) * rs - 1,
        time_steps * rt,
        final_time,
        sensors,
        tuple(s * rt for s in observation_steps),
        _default_initial_condition,
        basis_size=len(coefficients),
    )
    exact = generator.value(coefficients)
    rng = resolve_stream(seed).child("P02C").child("noise").generator()
    draws = [rng.gauss(0.0, 1.0) for _ in exact]
    data = tuple(b + sigma * e for b, e in zip(exact, draws, strict=True))
    return NonlinearBenchmark(
        identifier="P02C",
        name=(
            f"heat equation log-diffusivity inversion ({len(coefficients)} cosine "
            f"coefficients, {len(data)} readings)"
        ),
        model=model,
        data=data,
        noise=ScaledIdentityCovariance(len(data), sigma * sigma),
        noise_level=sigma,
        realized_noise_norm=sigma * math.hypot(*draws),
        truth=coefficients,
        exact_data=exact,
        same_discretization=rs == 1 and rt == 1,
        prior_mean=None,
        prior_covariance=None,
        pathologies=("nonlinear", "discretization_error"),
        seeds=MappingProxyType({"noise": "P02C/noise"}),
    )


# ---------------------------------------------------------------------------
# N03: damped oscillator with unknown initial state and source
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OscillatorBenchmark(NonlinearBenchmark):
    """Catalog case N03 with its linear-Gaussian reference analysis.

    Inherits the fields of
    :class:`~probstat.inverse.benchmark_extended.NonlinearBenchmark`
    (``model`` is the :class:`ODEForwardModel`, parameters ``(x0, v0,
    s_1..s_K)``) and adds the quantities below. The map is affine in the
    unknowns (here linear), so the posterior is exactly Gaussian.

    Attributes:
        whitened_singular_values: Singular values of ``J / sigma`` (noise
            whitened, no prior), descending.
        prior_whitened_singular_values: Singular values of ``J P / sigma``
            (``P`` the prior standard deviations), descending; the ones above
            1 count data-informed directions.
        posterior: The exact
            :class:`~probstat.inverse.linear_gaussian.GaussianLinearPosterior`.
        initial_state_posterior_mean: Posterior mean of the initial state.
        initial_state_covariance: Posterior covariance of the initial state
            (``2 x 2``).
        initial_state_prior_sd: Prior standard deviations of the initial state.
        initial_state_posterior_sd: Posterior standard deviations.
        jacobian: The dense ``m x n`` Jacobian of the (linear) model.

    References:
        [EQ-INV-ODE-008], [EQ-INV-ODE-009].

    """

    whitened_singular_values: Vector = ()
    prior_whitened_singular_values: Vector = ()
    posterior: GaussianLinearPosterior | None = None
    initial_state_posterior_mean: Vector = ()
    initial_state_covariance: Matrix = ()
    initial_state_prior_sd: Vector = ()
    initial_state_posterior_sd: Vector = ()
    jacobian: Matrix = ()


def damped_oscillator_benchmark(
    *,
    omega: float = 3.0,
    zeta: float = 0.1,
    horizon: float = 4.0,
    pieces: int = 4,
    intervals_per_piece: int = 3,
    noise_sd: float = 0.05,
    prior_sd: float = 1.0,
    steps_per_interval: int = 4,
    data_refinement: int = 4,
    seed: int | RandomStream | None = 11,
) -> OscillatorBenchmark:
    """Return N03: damped oscillator, unknown initial state and source.

    The unknown is ``z = (x(0), x'(0), s_1, ..., s_K)`` with iid
    ``N(0, prior_sd^2)`` prior; the truth is a draw from that prior taken from
    the stream child ``"truth"``, the noise from the DISTINCT child
    ``"noise"``, so changing one seed path never changes the other. The data
    are the noisy positions at ``K * intervals_per_piece`` equally spaced
    times; the solver model uses ``steps_per_interval`` RK4 steps per
    interval, the noise-free data come from ``data_refinement`` times as many
    (``same_discretization=False``; ``data_refinement=1`` is the inverse
    crime). Because the map is linear, the exact posterior, the initial-state
    marginal and the whitened singular values are returned as well
    [EQ-INV-ODE-008], [EQ-INV-ODE-009].

    Args:
        omega: Natural frequency.
        zeta: Damping ratio.
        horizon: Final time (``> 0``).
        pieces: Number of constant source pieces.
        intervals_per_piece: Observation intervals per piece.
        noise_sd: Noise standard deviation (absolute, ``> 0``).
        prior_sd: Prior standard deviation of every unknown (``> 0``).
        steps_per_interval: Solver-model RK4 steps per interval.
        data_refinement: Data generator step multiplier (``>= 1``).
        seed: Root seed.

    Returns:
        :class:`OscillatorBenchmark` with identifier ``"N03"``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-ODE-008], [EQ-INV-ODE-009], [REF-KAIPIO-SOMERSALO-2007].

    """
    sigma = _real(noise_sd, "noise_sd", positive=True)
    tau = _real(prior_sd, "prior_sd", positive=True)
    t_end = _real(horizon, "horizon", positive=True)
    refinement = _int_at_least(data_refinement, "data_refinement")
    steps = _int_at_least(steps_per_interval, "steps_per_interval")
    system = damped_oscillator_system(
        omega, zeta, pieces=pieces, intervals_per_piece=intervals_per_piece
    )
    count = pieces * intervals_per_piece
    times = tuple(t_end * (j + 1) / count for j in range(count))
    observation = ((1.0, 0.0),)
    model = ODEForwardModel(system, times, observation, steps_per_interval=steps)
    generator = ODEForwardModel(
        system, times, observation, steps_per_interval=steps * refinement
    )
    n = model.input_dimension
    stream = resolve_stream(seed).child("N03")
    truth_rng = stream.child("truth").generator()
    truth = tuple(tau * truth_rng.gauss(0.0, 1.0) for _ in range(n))
    exact = generator.value(truth)
    noise_rng = stream.child("noise").generator()
    draws = [noise_rng.gauss(0.0, 1.0) for _ in exact]
    data = tuple(b + sigma * e for b, e in zip(exact, draws, strict=True))
    m = len(data)
    columns = [
        model.jvp((0.0,) * n, tuple(1.0 if i == j else 0.0 for i in range(n)))
        for j in range(n)
    ]
    jac: Matrix = tuple(tuple(columns[j][i] for j in range(n)) for i in range(m))
    noise = ScaledIdentityCovariance(m, sigma * sigma)
    prior = DenseCovariance(
        tuple(tuple(tau * tau if i == j else 0.0 for j in range(n)) for i in range(n))
    )
    posterior = linear_gaussian_posterior(
        DenseOperator(jac),
        data,
        noise=noise,
        prior_mean=(0.0,) * n,
        prior_covariance=prior,
    )
    whitened = singular_value_decomposition(
        tuple(tuple(v / sigma for v in row) for row in jac)
    ).singular_values
    prior_whitened = singular_value_decomposition(
        tuple(tuple(v * tau / sigma for v in row) for row in jac)
    ).singular_values
    d = system.state_dimension
    cov_columns = [
        posterior.covariance_action(tuple(1.0 if i == j else 0.0 for i in range(n)))
        for j in range(d)
    ]
    block: Matrix = tuple(tuple(cov_columns[j][i] for j in range(d)) for i in range(d))
    return OscillatorBenchmark(
        identifier="N03",
        name=(
            f"damped oscillator (w={omega:g}, zeta={zeta:g}), initial state and "
            f"{pieces} source pieces, {m} position readings"
        ),
        model=model,
        data=data,
        noise=noise,
        noise_level=sigma,
        realized_noise_norm=sigma * math.hypot(*draws),
        truth=truth,
        exact_data=exact,
        same_discretization=refinement == 1,
        prior_mean=(0.0,) * n,
        prior_covariance=prior,
        pathologies=("source_state_trade_off", "weak_observability"),
        seeds=MappingProxyType({"truth": "N03/truth", "noise": "N03/noise"}),
        whitened_singular_values=tuple(whitened),
        prior_whitened_singular_values=tuple(prior_whitened),
        posterior=posterior,
        initial_state_posterior_mean=tuple(posterior.mean[:d]),
        initial_state_covariance=block,
        initial_state_prior_sd=(tau,) * d,
        initial_state_posterior_sd=tuple(math.sqrt(block[i][i]) for i in range(d)),
        jacobian=jac,
    )
