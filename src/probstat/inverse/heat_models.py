"""One-dimensional heat equation as a matrix-free linear forward model.

Experimental (added in 0.28). A small, verified linear time-dependent
forward model for inverse problems, not a PDE package.

The model is ``u_t = kappa u_ss`` on ``(0, L)`` with zero Dirichlet values.
The ``n`` interior nodes ``s_j = j h`` (``h = L / (n + 1)``) carry the
second-difference matrix ``D = (kappa / h^2) tridiag(-1, 2, -1)``, so the
semi-discrete system is ``u' = -D u`` [EQ-INV-HEAT-001]. With
``dt = T / N`` the two supported one-step maps are

* implicit Euler: ``(I + dt D) u^(k+1) = u^k``;
* Crank-Nicolson: ``(I + dt D / 2) u^(k+1) = (I - dt D / 2) u^k``.

Both are unconditionally stable for this problem (their amplification
factors on every eigenvalue of ``D`` have modulus at most one,
[EQ-INV-HEAT-007]); the explicit Euler method is deliberately not offered
because it needs ``dt <= h^2 / (2 kappa)``. Crank-Nicolson is second order
in ``dt`` but its amplification factor tends to ``-1`` for stiff modes, so
nonsmooth data produce non-decaying sign-alternating oscillations; implicit
Euler damps them at first-order accuracy.

Two operators are provided. :class:`HeatEquation1D` maps the initial
temperature ``u^0`` to the observations; :class:`HeatSource1D` maps a
time-constant source ``q`` (with ``u^0 = 0``) to the observations. Both are
:class:`probstat.linalg.LinearOperator` objects: ``matvec`` is forward time
stepping with one tridiagonal solve per step, and ``rmatvec`` is the EXACT
transpose of the discrete map [EQ-INV-HEAT-002], obtained by stepping
backwards with the transposed step maps and the transposed observation
operator [EQ-INV-HEAT-004]. Neither ever forms a dense matrix; the cost is
``O(N n)`` per application. The tridiagonal matrix is factorized once at
construction (:class:`probstat.inverse.pde_models.TridiagonalFactorization`).

Observations are either the final state at all nodes or the values at given
sensor positions and times. Sensor positions are interpolated linearly in
space (through the zero boundary values) and times linearly between time
steps; the interpolation is part of the discrete model, so the adjoint
remains exact.

References:
    [REF-HANSEN-2008], [REF-KAIPIO-SOMERSALO-2007],
    [REF-GOLUB-VANLOAN-2013]. No numerical-PDE textbook key exists in
    ``docs/references.md``; the amplification-factor facts are elementary
    and verified in ``tests/test_heat_models.py``.

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from operator import add

from probstat.inverse.pde_models import TridiagonalFactorization
from probstat.linalg import OperatorBase

_SCHEMES = ("implicit_euler", "crank_nicolson")


@dataclass(frozen=True, slots=True)
class HeatSolveCounts:
    """Work counters of a heat-equation operator.

    Attributes:
        forward_applications: Number of ``matvec`` calls.
        transpose_applications: Number of ``rmatvec`` calls.
        forward_solves: Tridiagonal solves spent in ``matvec``.
        transpose_solves: Tridiagonal solves spent in ``rmatvec``.

    References:
        [EQ-INV-HEAT-006].

    """

    forward_applications: int
    transpose_applications: int
    forward_solves: int
    transpose_solves: int


def _positive_int(name: str, value: object, minimum: int) -> int:
    """Validate an integer lower bound.

    Args:
        name: Argument name for messages.
        value: Candidate integer.
        minimum: Smallest admissible value.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < minimum``.

    References:
        [EQ-INV-HEAT-001]. Input validation only.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _positive_real(name: str, value: object) -> float:
    """Validate a positive finite real.

    Args:
        name: Argument name for messages.
        value: Candidate real.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not positive and finite.

    References:
        [EQ-INV-HEAT-001]. Input validation only.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return number


def _scheme(value: object) -> str:
    """Validate the time-stepping scheme name.

    Args:
        value: Candidate name.

    Returns:
        The scheme name.

    Raises:
        ValueError: If the name is not ``"implicit_euler"`` or
            ``"crank_nicolson"``.

    References:
        [EQ-INV-HEAT-001], [EQ-INV-HEAT-007].

    """
    if value not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {value!r}")
    return str(value)


def _real_list(name: str, values: Sequence[float]) -> list[float]:
    """Validate a non-empty sequence of finite reals.

    Args:
        name: Argument name for messages.
        values: Candidate sequence.

    Returns:
        The values as floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If it is empty or has a non-finite entry.

    References:
        [EQ-INV-HEAT-004]. Input validation only.

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        out.append(number)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def heat_amplification(scheme: str, x: float) -> float:
    """Return the per-step amplification factor ``g(x)``, ``x = mu dt``.

    ``g(x) = 1 / (1 + x)`` for implicit Euler and
    ``g(x) = (1 - x / 2) / (1 + x / 2)`` for Crank-Nicolson. Both satisfy
    ``|g(x)| <= 1`` for ``x >= 0`` (unconditional stability); the
    Crank-Nicolson factor tends to ``-1`` as ``x -> infinity``.

    Args:
        scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.
        x: Non-negative finite ``mu dt``.

    Returns:
        ``g(x)``.

    Raises:
        TypeError: If ``x`` is not a real number.
        ValueError: If the scheme is unknown or ``x`` is negative or not
            finite.

    References:
        [EQ-INV-HEAT-007].

    """
    name = _scheme(scheme)
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError("x must be a real number")
    value = float(x)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError("x must be non-negative and finite")
    if name == "implicit_euler":
        return 1.0 / (1.0 + value)
    return (1.0 - 0.5 * value) / (1.0 + 0.5 * value)


def heat_semigroup_singular_values(
    cells: int,
    *,
    length: float = 1.0,
    diffusivity: float = 1.0,
    final_time: float = 1.0,
    steps: int = 1,
    scheme: str = "implicit_euler",
) -> tuple[float, ...]:
    """Closed-form singular values of the final-time discrete heat map.

    The step matrices are functions of ``D``, whose orthonormal eigenvectors
    are sine vectors with eigenvalues
    ``mu_k = 4 kappa / h^2 sin^2(k pi h / (2 L))``, ``h = L / (n + 1)``.
    The final-time map ``S^N`` therefore has the singular values
    ``sigma_k = |g(mu_k dt)|^N`` [EQ-INV-HEAT-003]; they are returned in
    descending order (for Crank-Nicolson the order in ``k`` is not
    monotone, so the values are sorted).

    Args:
        cells: Number ``n >= 2`` of interior nodes.
        length: Domain length ``L > 0``.
        diffusivity: ``kappa > 0``.
        final_time: ``T > 0``.
        steps: Number of time steps ``N >= 1``.
        scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.

    Returns:
        The ``n`` singular values, largest first.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-HEAT-003], [REF-HANSEN-2008].

    """
    n = _positive_int("cells", cells, 2)
    big_l = _positive_real("length", length)
    kappa = _positive_real("diffusivity", diffusivity)
    horizon = _positive_real("final_time", final_time)
    count = _positive_int("steps", steps, 1)
    name = _scheme(scheme)
    h = big_l / (n + 1)
    dt = horizon / count
    values = []
    for k in range(1, n + 1):
        mu = 4.0 * kappa / h**2 * math.sin(k * math.pi * h / (2.0 * big_l)) ** 2
        amplification = abs(heat_amplification(name, mu * dt))
        values.append(amplification**count)
    return tuple(sorted(values, reverse=True))


class _HeatOperator(OperatorBase):
    """Shared time-stepping machinery of the heat operators (internal).

    Subclasses decide whether the unknown is the initial state or the
    source; this class owns the grid, the factorization, the observation
    operator, the stepping loops and the work counters.

    References:
        [EQ-INV-HEAT-001], [EQ-INV-HEAT-002], [EQ-INV-HEAT-004].

    """

    __slots__ = (
        "_c",
        "_counts",
        "_dt",
        "_factorization",
        "_last_step",
        "_observations",
        "_points",
        "_scheme_name",
        "_theta",
        "cells",
        "diffusivity",
        "final_time",
        "length",
        "scheme",
        "steps",
    )

    def __init__(
        self,
        cells: int,
        *,
        length: float,
        diffusivity: float,
        final_time: float,
        steps: int,
        scheme: str,
        observation: str,
        sensors: Sequence[float] | None,
        times: Sequence[float] | None,
        name: str,
    ) -> None:
        """Validate the configuration, factorize ``I + theta dt D``.

        Args:
            cells: Number ``n >= 2`` of interior nodes.
            length: Domain length ``L > 0``.
            diffusivity: ``kappa > 0``.
            final_time: ``T > 0``.
            steps: Number of time steps ``N >= 1``.
            scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.
            observation: ``"final"`` or ``"sensors"``.
            sensors: Sensor positions in ``(0, L)`` (``"sensors"`` only).
            times: Observation times in ``[0, T]`` (``"sensors"`` only).
            name: Operator name for messages.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range, or sensors or times
                are given with ``observation="final"`` or missing otherwise.

        References:
            [EQ-INV-HEAT-001], [EQ-INV-HEAT-004].

        """
        n = _positive_int("cells", cells, 2)
        big_l = _positive_real("length", length)
        kappa = _positive_real("diffusivity", diffusivity)
        horizon = _positive_real("final_time", final_time)
        count = _positive_int("steps", steps, 1)
        scheme_name = _scheme(scheme)
        self.cells = n
        self.length = big_l
        self.diffusivity = kappa
        self.final_time = horizon
        self.steps = count
        self.scheme = scheme_name
        self._scheme_name = scheme_name
        self._dt = horizon / count
        h = big_l / (n + 1)
        self._c = kappa / h**2
        self._theta = 1.0 if scheme_name == "implicit_euler" else 0.5
        off = -self._theta * self._dt * self._c
        diagonal = 1.0 + 2.0 * self._theta * self._dt * self._c
        self._factorization = TridiagonalFactorization(
            [off] * (n - 1), [diagonal] * n, [off] * (n - 1)
        )
        # per step: list of (row, coefficient, node a, weight a, node b, weight b)
        self._observations: dict[int, list[tuple[int, float, int, float, int, float]]]
        self._observations = {}
        if observation == "final":
            if sensors is not None or times is not None:
                raise ValueError("sensors and times require observation='sensors'")
            self._points: tuple[tuple[float, float], ...] = ()
            self._last_step = count
            rows = n
            self._observations[count] = [(i, 1.0, i, 1.0, -1, 0.0) for i in range(n)]
        elif observation == "sensors":
            if sensors is None or times is None:
                raise ValueError("observation='sensors' needs sensors and times")
            positions = _real_list("sensors", sensors)
            instants = _real_list("times", times)
            for s in positions:
                if not 0.0 < s < big_l:
                    raise ValueError(f"sensor position {s} must lie in (0, {big_l})")
            for t in instants:
                if not 0.0 <= t <= horizon:
                    raise ValueError(f"observation time {t} must lie in [0, {horizon}]")
            points = []
            last = 0
            row = 0
            for t in instants:
                position = t / self._dt
                step = min(int(math.floor(position)), count - 1)
                theta = min(max(position - step, 0.0), 1.0)
                for s in positions:
                    points.append((t, s))
                    x = s / h
                    node = min(int(math.floor(x)), n)
                    w_right = x - node
                    a, b = node - 1, node
                    pieces = []
                    for time_step, time_weight in (
                        (step, 1.0 - theta),
                        (step + 1, theta),
                    ):
                        if time_weight == 0.0:
                            continue
                        pieces.append((time_step, time_weight))
                        last = max(last, time_step)
                    for time_step, time_weight in pieces:
                        self._observations.setdefault(time_step, []).append(
                            (
                                row,
                                time_weight,
                                a if a >= 0 else -1,
                                (1.0 - w_right) if a >= 0 else 0.0,
                                b if b < n else -1,
                                w_right if b < n else 0.0,
                            )
                        )
                    row += 1
            self._points = tuple(points)
            self._last_step = last
            rows = row
        else:
            raise ValueError("observation must be 'final' or 'sensors'")
        super().__init__(rows, n, name=name)
        self._counts = [0, 0, 0, 0]

    @property
    def observation_points(self) -> tuple[tuple[float, float], ...]:
        """Return the ``(time, position)`` pair of every sensor observation.

        Returns:
            The pairs in row order (time-major); empty for ``"final"``.

        References:
            [EQ-INV-HEAT-004].

        """
        return self._points

    @property
    def time_step(self) -> float:
        """Return ``dt = T / N``.

        Returns:
            The step size.

        References:
            [EQ-INV-HEAT-001].

        """
        return self._dt

    @property
    def counts(self) -> HeatSolveCounts:
        """Return a snapshot of the application and tridiagonal-solve counts.

        Returns:
            The counters since construction or the last :meth:`reset_counts`.

        References:
            [EQ-INV-HEAT-006].

        """
        c = self._counts
        return HeatSolveCounts(c[0], c[1], c[2], c[3])

    def reset_counts(self) -> None:
        """Set all counters to zero.

        References:
            [EQ-INV-HEAT-006].

        """
        self._counts = [0, 0, 0, 0]

    def _apply_b(self, u: list[float]) -> list[float]:
        """Apply the explicit half ``B`` of the step (``I`` or ``I - dt D / 2``).

        Args:
            u: Interior nodal values.

        Returns:
            ``B u``.

        References:
            [EQ-INV-HEAT-001].

        """
        if self._scheme_name == "implicit_euler":
            return list(u)
        k = 0.5 * self._dt * self._c
        padded = [0.0, *u, 0.0]
        return [
            centre - k * (2.0 * centre - left - right)
            for centre, left, right in zip(
                u, padded, islice(padded, 2, None), strict=False
            )
        ]

    def _observe_add(self, step: int, u: Sequence[float], out: list[float]) -> None:
        """Add the contribution of the state at ``step`` to the observations.

        Args:
            step: Time-step index.
            u: State at that step.
            out: Observation vector, updated in place.

        References:
            [EQ-INV-HEAT-004].

        """
        for row, coefficient, a, wa, b, wb in self._observations.get(step, ()):
            value = 0.0
            if a >= 0:
                value += wa * u[a]
            if b >= 0:
                value += wb * u[b]
            out[row] += coefficient * value

    def _observe_transpose_add(
        self, step: int, y: Sequence[float], out: list[float]
    ) -> None:
        """Add ``H_k' y`` for the state at ``step`` to ``out``.

        Args:
            step: Time-step index.
            y: Observation-space vector.
            out: State-space vector, updated in place.

        References:
            [EQ-INV-HEAT-004].

        """
        for row, coefficient, a, wa, b, wb in self._observations.get(step, ()):
            if a >= 0:
                out[a] += coefficient * wa * y[row]
            if b >= 0:
                out[b] += coefficient * wb * y[row]

    def _run_forward(
        self, initial: Sequence[float] | None, source: Sequence[float] | None
    ) -> list[float]:
        """Step forward from ``initial`` with an optional constant source.

        Args:
            initial: Initial state, or ``None`` for zero.
            source: Source ``q`` (per step ``dt q`` is added), or ``None``.

        Returns:
            The observations; one tridiagonal solve is spent per step.

        References:
            [EQ-INV-HEAT-001], [EQ-INV-HEAT-002], [EQ-INV-HEAT-005].

        """
        n = self.cells
        u = list(initial) if initial is not None else [0.0] * n
        out = [0.0] * self.shape[0]
        self._observe_add(0, u, out)
        scaled = None if source is None else [self._dt * q for q in source]
        for step in range(1, self._last_step + 1):
            rhs = self._apply_b(u)
            if scaled is not None:
                rhs = list(map(add, rhs, scaled))
            u = self._factorization.solve(rhs)
            self._counts[2] += 1
            self._observe_add(step, u, out)
        self._counts[0] += 1
        return out

    def _run_transpose(self, y: Sequence[float], *, source: bool) -> list[float]:
        """Apply the exact transpose of :meth:`_run_forward`.

        For an initial-state unknown this is the adjoint recursion
        ``a_k = H_k' y + S' a_(k+1)`` returning ``a_0``. For a source unknown
        the states ``a_1 .. a_K`` are summed and one final solve with
        ``dt M^-1`` is applied (``u^k = S u^(k-1) + dt M^-1 q``).

        Args:
            y: Observation-space vector.
            source: Whether the unknown is the source.

        Returns:
            ``A' y``.

        References:
            [EQ-INV-HEAT-002], [EQ-INV-HEAT-005].

        """
        n = self.cells
        a = [0.0] * n
        total = [0.0] * n
        stop = 1 if source else 0
        for step in range(self._last_step, stop - 1, -1):
            if step < self._last_step:
                # S' a = B (M^-1 a); both factors are symmetric tridiagonal.
                a = self._apply_b(self._factorization.solve(a))
                self._counts[3] += 1
            self._observe_transpose_add(step, y, a)
            if source:
                total = list(map(add, total, a))
        self._counts[1] += 1
        if not source:
            return a
        if self._last_step == 0:
            return total
        solved = self._factorization.solve(total)
        self._counts[3] += 1
        return [self._dt * v for v in solved]


class HeatEquation1D(_HeatOperator):
    """Linear map from the initial temperature to heat-equation observations.

    ``u_t = kappa u_ss`` on ``(0, L)`` with zero Dirichlet values, ``n``
    interior nodes and ``N`` steps of implicit Euler or Crank-Nicolson. The
    operator is ``A = H S^N``-style: ``matvec`` steps forward (one
    tridiagonal solve per step) and accumulates the observations,
    ``rmatvec`` is the exact discrete adjoint [EQ-INV-HEAT-002]. One
    ``matvec`` of the final-time operator costs exactly ``steps`` tridiagonal
    solves and one ``rmatvec`` the same (:attr:`counts`).

    Attributes:
        cells: Number ``n`` of interior nodes.
        length: Domain length ``L``.
        diffusivity: ``kappa``.
        final_time: Horizon ``T``.
        steps: Number ``N`` of time steps.
        scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.

    References:
        [EQ-INV-HEAT-001], [EQ-INV-HEAT-002], [REF-KAIPIO-SOMERSALO-2007],
        [REF-HANSEN-2008].

    """

    __slots__ = ()

    def __init__(
        self,
        cells: int,
        *,
        length: float = 1.0,
        diffusivity: float = 1.0,
        final_time: float = 1.0,
        steps: int = 10,
        scheme: str = "implicit_euler",
        observation: str = "final",
        sensors: Sequence[float] | None = None,
        times: Sequence[float] | None = None,
    ) -> None:
        """Build the operator.

        Args:
            cells: Number ``n >= 2`` of interior nodes (``h = L / (n + 1)``).
            length: Domain length ``L > 0``.
            diffusivity: ``kappa > 0``.
            final_time: ``T > 0``.
            steps: Number of time steps ``N >= 1``.
            scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.
            observation: ``"final"`` (state at all nodes at ``T``, a square
                operator) or ``"sensors"`` (values at ``sensors`` x
                ``times``, time-major rows, linear interpolation).
            sensors: Positions in ``(0, L)``; only with ``"sensors"``.
            times: Instants in ``[0, T]``; only with ``"sensors"``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``kappa``, ``T`` or ``N`` is not positive, a
                sensor lies outside ``(0, L)``, a time lies outside
                ``[0, T]``, or sensors and times do not match
                ``observation``.

        References:
            [EQ-INV-HEAT-001], [EQ-INV-HEAT-004].

        """
        super().__init__(
            cells,
            length=length,
            diffusivity=diffusivity,
            final_time=final_time,
            steps=steps,
            scheme=scheme,
            observation=observation,
            sensors=sensors,
            times=times,
            name="heat initial-condition operator",
        )

    def _forward(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply ``A`` by forward time stepping.

        Args:
            vector: Initial temperature at the interior nodes.

        Returns:
            The observations.

        References:
            [EQ-INV-HEAT-001], [EQ-INV-HEAT-002].

        """
        return self._run_forward(vector, None)

    def _transpose(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply ``A'`` by adjoint time stepping.

        Args:
            vector: Observation-space vector.

        Returns:
            ``A' y`` at the interior nodes.

        References:
            [EQ-INV-HEAT-002].

        """
        return self._run_transpose(vector, source=False)


class HeatSource1D(_HeatOperator):
    """Linear map from a time-constant source to heat-equation observations.

    ``u_t = kappa u_ss + q(s)`` with ``u(., 0) = 0`` and zero Dirichlet
    values; the unknown is the vector of source values at the ``n`` interior
    nodes. Each step is ``M u^k = B u^(k-1) + dt q`` (the same ``M``, ``B`` as
    the initial-condition problem), so the observations are linear in ``q``
    [EQ-INV-HEAT-005]. ``rmatvec`` accumulates the adjoint states and spends
    one final solve, i.e. exactly as many solves as ``matvec``. The inverse
    problem is the linear counterpart of the nonlinear coefficient inversion.

    Attributes:
        cells: Number ``n`` of interior nodes.
        length: Domain length ``L``.
        diffusivity: ``kappa``.
        final_time: Horizon ``T``.
        steps: Number ``N`` of time steps.
        scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.

    References:
        [EQ-INV-HEAT-005], [REF-KAIPIO-SOMERSALO-2007], [REF-HANSEN-2008].

    """

    __slots__ = ()

    def __init__(
        self,
        cells: int,
        *,
        length: float = 1.0,
        diffusivity: float = 1.0,
        final_time: float = 1.0,
        steps: int = 10,
        scheme: str = "implicit_euler",
        observation: str = "final",
        sensors: Sequence[float] | None = None,
        times: Sequence[float] | None = None,
    ) -> None:
        """Build the operator.

        Args:
            cells: Number ``n >= 2`` of interior nodes.
            length: Domain length ``L > 0``.
            diffusivity: ``kappa > 0``.
            final_time: ``T > 0``.
            steps: Number of time steps ``N >= 1``.
            scheme: ``"implicit_euler"`` or ``"crank_nicolson"``.
            observation: ``"final"`` or ``"sensors"`` (see
                :class:`HeatEquation1D`). Rows at time ``0`` are zero.
            sensors: Positions in ``(0, L)``; only with ``"sensors"``.
            times: Instants in ``[0, T]``; only with ``"sensors"``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: As for :class:`HeatEquation1D`.

        References:
            [EQ-INV-HEAT-005].

        """
        super().__init__(
            cells,
            length=length,
            diffusivity=diffusivity,
            final_time=final_time,
            steps=steps,
            scheme=scheme,
            observation=observation,
            sensors=sensors,
            times=times,
            name="heat source operator",
        )

    def _forward(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply ``A`` by forward time stepping from zero with source ``q``.

        Args:
            vector: Source values at the interior nodes.

        Returns:
            The observations.

        References:
            [EQ-INV-HEAT-005].

        """
        return self._run_forward(None, vector)

    def _transpose(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply ``A'`` by adjoint time stepping and one final solve.

        Args:
            vector: Observation-space vector.

        Returns:
            ``A' y`` at the interior nodes.

        References:
            [EQ-INV-HEAT-005].

        """
        return self._run_transpose(vector, source=True)
