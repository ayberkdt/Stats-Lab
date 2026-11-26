"""One-dimensional elliptic diffusion with tangent and adjoint derivatives.

Experimental (added in 0.28). A small, verified prototype of a PDE-constrained
forward model for inverse problems, not a finite-element package.

The model is ``-(k u')' = q`` on ``(0, L)`` with Dirichlet values
``u(0) = u_L`` and ``u(L) = u_R``. The domain is split into ``n`` cells of
width ``h = L / n``; the parameter is ``x_i = log k_i`` on cell ``i`` (so
``k = exp(x)`` is positive by construction). Face conductances are
``c_f = k_f / h`` with the harmonic mean ``k_f = 2 k_a k_b / (k_a + k_b)`` at an
interior face and ``c_f = 2 k / h`` (adjacent cell, half-cell distance) at a
boundary face. With the face jump ``w_f = u_f - u_(f-1)`` (``u_(-1) = u_L``,
``u_n = u_R``) the cell equations are ``c_i w_i - c_(i+1) w_(i+1) = h q_i``,
a symmetric positive definite tridiagonal system ``K(x) u = f(x)`` that is
solved exactly in ``O(n)`` by the Thomas algorithm. Observations are the
piecewise-linear interpolation of the profile (through the boundary values) at
the sensor positions.

Derivatives are exact derivatives of the discrete model. The tangent solves
``K du = -(dr/dx)[v]`` and the adjoint solves ``K' p = H' w``; both use the
single stored factorization, never form the Jacobian, and cost one
tridiagonal solve plus ``O(n)`` assembly each, so the gradient of a scalar
misfit needs one forward and one adjoint solve whatever the number of
parameters.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

_LOG_K_LIMIT = 345.0


def _finite_vector(
    name: str, values: Sequence[float], length: int | None
) -> list[float]:
    """Validate a vector of finite reals.

    Args:
        name: Argument name for messages.
        values: Candidate sequence.
        length: Required length, or ``None`` for any positive length.

    Returns:
        The values as a list of floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length is wrong or an entry is not finite.

    References:
        [EQ-INV-PDE-001]. Input validation only.

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
    if length is not None and len(out) != length:
        raise ValueError(f"{name} must have length {length}, got {len(out)}")
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _check_tridiagonal(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
) -> int:
    """Check the band lengths of a tridiagonal matrix.

    Args:
        lower: Sub-diagonal.
        diagonal: Main diagonal.
        upper: Super-diagonal.

    Returns:
        The order ``n`` of the matrix.

    Raises:
        ValueError: If the lengths are not ``(n - 1, n, n - 1)``.

    References:
        [EQ-INV-PDE-005]. Input validation only.

    """
    n = len(diagonal)
    if n < 1 or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("tridiagonal bands must have lengths (n-1, n, n-1), n >= 1")
    return n


class TridiagonalFactorization:
    """LU factorization ``A = L U`` of a tridiagonal matrix without pivoting.

    The Thomas algorithm is stable for diagonally dominant and symmetric
    positive definite matrices, which covers the diffusion system of this
    module; no pivoting is attempted, and an exactly zero or non-finite pivot
    raises. The factors are stored once and reused for solves with ``A`` and
    with ``A'`` (Euclidean transpose).

    Attributes:
        order: Matrix order ``n``.

    References:
        [EQ-INV-PDE-005], [REF-GOLUB-VANLOAN-2013].

    """

    __slots__ = ("_multipliers", "_pivots", "_upper", "order")

    def __init__(
        self,
        lower: Sequence[float],
        diagonal: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        """Factorize a tridiagonal matrix in ``O(n)``.

        Args:
            lower: Sub-diagonal ``A[i+1, i]`` (length ``n - 1``).
            diagonal: Main diagonal (length ``n``).
            upper: Super-diagonal ``A[i, i+1]`` (length ``n - 1``).

        Raises:
            ValueError: If band lengths disagree, entries are not finite, or a
                pivot is zero.

        References:
            [EQ-INV-PDE-005], [REF-GOLUB-VANLOAN-2013].

        """
        n = _check_tridiagonal(lower, diagonal, upper)
        lo = _finite_vector("lower", lower, n - 1) if n > 1 else []
        di = _finite_vector("diagonal", diagonal, n)
        up = _finite_vector("upper", upper, n - 1) if n > 1 else []
        pivots = [di[0]]
        multipliers: list[float] = []
        for i in range(n - 1):
            if pivots[i] == 0.0:
                raise ValueError(f"zero pivot at row {i}; matrix is singular")
            m = lo[i] / pivots[i]
            multipliers.append(m)
            pivots.append(di[i + 1] - m * up[i])
        if pivots[n - 1] == 0.0 or not all(math.isfinite(p) for p in pivots):
            raise ValueError(f"zero or non-finite pivot at row {n - 1}")
        self._pivots = pivots
        self._multipliers = multipliers
        self._upper = up
        self.order = n

    def solve(self, rhs: Sequence[float]) -> list[float]:
        """Solve ``A x = rhs``.

        Args:
            rhs: Right-hand side of length ``n``.

        Returns:
            The solution ``x``.

        Raises:
            ValueError: If ``rhs`` has the wrong length or is not finite.

        References:
            [EQ-INV-PDE-005], [REF-GOLUB-VANLOAN-2013].

        """
        n = self.order
        y = _finite_vector("rhs", rhs, n)
        for i in range(n - 1):
            y[i + 1] -= self._multipliers[i] * y[i]
        y[n - 1] /= self._pivots[n - 1]
        for i in range(n - 2, -1, -1):
            y[i] = (y[i] - self._upper[i] * y[i + 1]) / self._pivots[i]
        return y

    def solve_transpose(self, rhs: Sequence[float]) -> list[float]:
        """Solve ``A' x = rhs`` with the same factors (``A' = U' L'``).

        Args:
            rhs: Right-hand side of length ``n``.

        Returns:
            The solution ``x``.

        Raises:
            ValueError: If ``rhs`` has the wrong length or is not finite.

        References:
            [EQ-INV-PDE-005], [REF-GOLUB-VANLOAN-2013].

        """
        n = self.order
        z = _finite_vector("rhs", rhs, n)
        z[0] /= self._pivots[0]
        for i in range(1, n):
            z[i] = (z[i] - self._upper[i - 1] * z[i - 1]) / self._pivots[i]
        for i in range(n - 2, -1, -1):
            z[i] -= self._multipliers[i] * z[i + 1]
        return z


def solve_tridiagonal(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> tuple[float, ...]:
    """Solve a tridiagonal system by the Thomas algorithm in ``O(n)``.

    Args:
        lower: Sub-diagonal ``A[i+1, i]`` (length ``n - 1``).
        diagonal: Main diagonal (length ``n``).
        upper: Super-diagonal ``A[i, i+1]`` (length ``n - 1``).
        rhs: Right-hand side (length ``n``).

    Returns:
        The solution of ``A x = rhs``.

    Raises:
        ValueError: On inconsistent lengths, non-finite input, or a zero
            pivot (no pivoting is performed; use a dense pivoted solver for
            matrices that are neither diagonally dominant nor SPD).

    References:
        [EQ-INV-PDE-005], [REF-GOLUB-VANLOAN-2013].

    """
    return tuple(TridiagonalFactorization(lower, diagonal, upper).solve(rhs))


def solve_tridiagonal_transpose(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> tuple[float, ...]:
    """Solve ``A' x = rhs`` for a tridiagonal ``A`` given by its bands.

    Args:
        lower: Sub-diagonal of ``A``.
        diagonal: Main diagonal of ``A``.
        upper: Super-diagonal of ``A``.
        rhs: Right-hand side.

    Returns:
        The solution of ``A' x = rhs``.

    Raises:
        ValueError: As for :func:`solve_tridiagonal`.

    References:
        [EQ-INV-PDE-005], [REF-GOLUB-VANLOAN-2013].

    """
    return tuple(TridiagonalFactorization(lower, diagonal, upper).solve_transpose(rhs))


@dataclass(frozen=True, slots=True)
class PDESolveCounts:
    """Work counters of a PDE model.

    Attributes:
        forward: Number of forward state solves ``K u = f``.
        tangent: Number of tangent solves ``K du = ...``.
        adjoint: Number of adjoint solves ``K' p = ...``.

    References:
        [EQ-INV-PDE-004].

    """

    forward: int
    tangent: int
    adjoint: int


@dataclass(frozen=True, slots=True)
class FluxBalance:
    """Discrete conservation diagnostic of the finite-volume solution.

    Attributes:
        cell_residual: ``max_i |F_(i+1) - F_i - h q_i|`` over cells.
        global_imbalance: ``|F_n - F_0 - h sum_i q_i|``.
        total_source: ``h sum_i q_i``.
        left_flux: Flux ``F_0`` through the left boundary face.
        right_flux: Flux ``F_n`` through the right boundary face.

    References:
        [EQ-INV-PDE-006].

    """

    cell_residual: float
    global_imbalance: float
    total_source: float
    left_flux: float
    right_flux: float


@dataclass(frozen=True, slots=True)
class AdjointGradient:
    """Misfit, residual and gradient from one forward and one adjoint solve.

    Attributes:
        objective: ``1/2 ||W (G(x) - y)||^2``.
        gradient: ``J' W' W (G(x) - y)`` with respect to ``x``.
        residual: ``G(x) - y``.

    References:
        [EQ-INV-PDE-008].

    """

    objective: float
    gradient: tuple[float, ...]
    residual: tuple[float, ...]


class _ForwardLike(Protocol):
    """Structural subset of the differentiable-forward protocol used here."""

    @property
    def output_dimension(self) -> int:
        """Return the number of observations.

        References:
            [EQ-INV-PDE-008].

        """
        ...

    def value(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return the forward map.

        References:
            [EQ-INV-PDE-008].

        """
        ...

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> tuple[float, ...]:
        """Return the transposed Jacobian applied to ``w``.

        References:
            [EQ-INV-PDE-008].

        """
        ...


@dataclass(slots=True)
class _Evaluation:
    """Cached state of one parameter vector (internal)."""

    x: tuple[float, ...]
    conductance: list[float]
    left_weight: list[float]
    right_weight: list[float]
    state: list[float]
    jump: list[float]
    factorization: TridiagonalFactorization


class EllipticDiffusion1D:
    """Finite-volume diffusion model ``-(k u')' = q`` with log-conductivity input.

    The map ``x -> G(x) = H u(x) + b`` (``x = log k`` per cell, ``u`` the cell
    averages, ``H`` the sensor interpolation, ``b`` the contribution of the
    boundary values) is a differentiable forward model with
    ``input_dimension = n`` and ``output_dimension = len(sensors)``. The last
    evaluated parameter vector is cached together with its factorization, so
    ``value`` followed by ``jvp`` or ``vjp`` at the same ``x`` performs one
    forward solve in total.

    Attributes:
        cells: Number ``n`` of cells.
        length: Domain length ``L``.
        sensors: Sensor positions in ``(0, L)``.
        boundary: Dirichlet values ``(u_L, u_R)``.

    References:
        [EQ-INV-PDE-001], [EQ-INV-PDE-002], [REF-KAIPIO-SOMERSALO-2007].

    """

    def __init__(
        self,
        cells: int,
        *,
        sensors: Sequence[float],
        length: float = 1.0,
        source: float | Sequence[float] | Callable[[float], float] = 0.0,
        boundary: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Build the model.

        Args:
            cells: Number of cells ``n >= 2``.
            sensors: Strictly interior sensor positions in ``(0, length)``.
            length: Domain length ``L > 0``.
            source: Constant, callable ``q(s)`` evaluated at cell centres, or
                a vector of ``n`` cell-centre values.
            boundary: Dirichlet values ``(u_L, u_R)``.

        Raises:
            TypeError: On wrongly typed arguments.
            ValueError: If ``cells < 2``, ``length`` is not positive and
                finite, a sensor lies outside ``(0, length)``, or a source or
                boundary value is not finite.

        References:
            [EQ-INV-PDE-001], [EQ-INV-PDE-007].

        """
        if isinstance(cells, bool) or not isinstance(cells, int):
            raise TypeError("cells must be an integer")
        if cells < 2:
            raise ValueError("cells must be at least 2")
        length = float(length)
        if not (math.isfinite(length) and length > 0.0):
            raise ValueError("length must be positive and finite")
        self.cells = cells
        self.length = length
        self._h = length / cells
        centres = [(i + 0.5) * self._h for i in range(cells)]
        if callable(source):
            q = [float(source(s)) for s in centres]
            if not all(math.isfinite(v) for v in q):
                raise ValueError("source values must be finite")
        elif isinstance(source, (int, float)) and not isinstance(source, bool):
            if not math.isfinite(float(source)):
                raise ValueError("source must be finite")
            q = [float(source)] * cells
        elif isinstance(source, Sequence):
            q = _finite_vector("source", source, cells)
        else:
            raise TypeError("source must be a number, a callable, or a sequence")
        self._q = q
        left, right = _finite_vector("boundary", boundary, 2)
        self.boundary = (left, right)
        sensor_list = _finite_vector("sensors", sensors, None)
        for s in sensor_list:
            if not 0.0 < s < length:
                raise ValueError(f"sensor position {s} must lie in (0, {length})")
        self.sensors = tuple(sensor_list)
        nodes = [0.0, *centres, length]
        self._observation: list[tuple[int, float, int, float]] = []
        for s in sensor_list:
            m = min(max(bisect_right(nodes, s) - 1, 0), cells)
            t = (s - nodes[m]) / (nodes[m + 1] - nodes[m])
            self._observation.append((m - 1, 1.0 - t, m, t))
        self._offset = [
            (a_w * left if a == -1 else 0.0) + (b_w * right if b == cells else 0.0)
            for a, a_w, b, b_w in self._observation
        ]
        self._cache: _Evaluation | None = None
        self._forward_solves = 0
        self._tangent_solves = 0
        self._adjoint_solves = 0

    @property
    def input_dimension(self) -> int:
        """Return the number of parameters ``n`` (one log-conductivity per cell).

        Returns:
            The input dimension.

        References:
            [EQ-INV-PDE-001].

        """
        return self.cells

    @property
    def output_dimension(self) -> int:
        """Return the number of sensors.

        Returns:
            The output dimension.

        References:
            [EQ-INV-PDE-007].

        """
        return len(self.sensors)

    @property
    def counts(self) -> PDESolveCounts:
        """Return a snapshot of the forward, tangent and adjoint solve counts.

        Returns:
            The counters since construction or the last :meth:`reset_counts`.

        References:
            [EQ-INV-PDE-004].

        """
        return PDESolveCounts(
            self._forward_solves, self._tangent_solves, self._adjoint_solves
        )

    def reset_counts(self) -> None:
        """Set all solve counters to zero (the factorization cache is kept).

        References:
            [EQ-INV-PDE-004].

        """
        self._forward_solves = 0
        self._tangent_solves = 0
        self._adjoint_solves = 0

    def _conductivity(self, x: Sequence[float]) -> list[float]:
        """Return ``k = exp(x)`` after range checks.

        Args:
            x: Log-conductivities.

        Returns:
            Cell conductivities.

        Raises:
            ValueError: If ``x`` is invalid or ``|x_i|`` exceeds the supported
                range (``exp`` would overflow or underflow).

        References:
            [EQ-INV-PDE-002].

        """
        values = _finite_vector("x", x, self.cells)
        for i, v in enumerate(values):
            if abs(v) > _LOG_K_LIMIT:
                raise ValueError(
                    f"x[{i}] = {v} gives exp(x) outside [1e-150, 1e150]; "
                    "conductivity overflow or underflow"
                )
        return [math.exp(v) for v in values]

    def _evaluate(self, x: Sequence[float]) -> _Evaluation:
        """Assemble and solve the state equation, using the cache when possible.

        Args:
            x: Log-conductivities.

        Returns:
            The cached evaluation for ``x``.

        References:
            [EQ-INV-PDE-001], [EQ-INV-PDE-002].

        """
        key = tuple(_finite_vector("x", x, self.cells))
        if self._cache is not None and self._cache.x == key:
            return self._cache
        n, h = self.cells, self._h
        k = self._conductivity(key)
        conductance = [2.0 * k[0] / h]
        left_w = [0.0]
        right_w = [1.0]
        for f in range(1, n):
            ka, kb = k[f - 1], k[f]
            kf = 2.0 / (1.0 / ka + 1.0 / kb)
            conductance.append(kf / h)
            left_w.append(kb / (ka + kb))
            right_w.append(ka / (ka + kb))
        conductance.append(2.0 * k[n - 1] / h)
        left_w.append(1.0)
        right_w.append(0.0)
        diagonal = [conductance[i] + conductance[i + 1] for i in range(n)]
        offdiag = [-conductance[i] for i in range(1, n)]
        rhs = [h * q for q in self._q]
        rhs[0] += conductance[0] * self.boundary[0]
        rhs[n - 1] += conductance[n] * self.boundary[1]
        factorization = TridiagonalFactorization(offdiag, diagonal, offdiag)
        state = factorization.solve(rhs)
        extended = [self.boundary[0], *state, self.boundary[1]]
        jump = [extended[f + 1] - extended[f] for f in range(n + 1)]
        self._forward_solves += 1
        self._cache = _Evaluation(
            key, conductance, left_w, right_w, state, jump, factorization
        )
        return self._cache

    def system(
        self, x: Sequence[float]
    ) -> tuple[
        tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]
    ]:
        """Return the bands and right-hand side of ``K(x) u = f(x)``.

        Args:
            x: Log-conductivities (length ``n``).

        Returns:
            ``(lower, diagonal, upper, rhs)``; ``K`` is symmetric so
            ``lower == upper``. Not counted as a solve.

        Raises:
            ValueError: If ``x`` is invalid or out of the supported range.

        References:
            [EQ-INV-PDE-001], [EQ-INV-PDE-002].

        """
        ev = self._evaluate(x)
        n, h = self.cells, self._h
        c = ev.conductance
        diagonal = tuple(c[i] + c[i + 1] for i in range(n))
        off = tuple(-c[i] for i in range(1, n))
        rhs = [h * q for q in self._q]
        rhs[0] += c[0] * self.boundary[0]
        rhs[n - 1] += c[n] * self.boundary[1]
        return off, diagonal, off, tuple(rhs)

    def state(self, x: Sequence[float]) -> tuple[float, ...]:
        """Solve ``K(x) u = f(x)`` for the cell-average profile.

        Args:
            x: Log-conductivities (length ``n``).

        Returns:
            The cell values ``u``.

        Raises:
            ValueError: If ``x`` is invalid or out of the supported range.

        References:
            [EQ-INV-PDE-001].

        """
        return tuple(self._evaluate(x).state)

    def _observe(self, u: Sequence[float]) -> list[float]:
        """Apply the linear part ``H`` of the sensor interpolation.

        Args:
            u: Cell values.

        Returns:
            ``H u`` (without the boundary offset).

        References:
            [EQ-INV-PDE-007].

        """
        n = self.cells
        out = []
        for a, a_w, b, b_w in self._observation:
            total = 0.0
            if a >= 0:
                total += a_w * u[a]
            if b < n:
                total += b_w * u[b]
            out.append(total)
        return out

    def _observe_transpose(self, w: Sequence[float]) -> list[float]:
        """Apply ``H'`` to a vector of sensor weights.

        Args:
            w: Vector of length ``output_dimension``.

        Returns:
            ``H' w`` of length ``n``.

        References:
            [EQ-INV-PDE-007].

        """
        n = self.cells
        out = [0.0] * n
        for (a, a_w, b, b_w), wi in zip(self._observation, w, strict=True):
            if a >= 0:
                out[a] += a_w * wi
            if b < n:
                out[b] += b_w * wi
        return out

    def value(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return the sensor readings ``G(x) = H u(x) + b``.

        Args:
            x: Log-conductivities (length ``n``).

        Returns:
            Predicted observations.

        Raises:
            ValueError: If ``x`` is invalid or out of the supported range.

        References:
            [EQ-INV-PDE-007].

        """
        ev = self._evaluate(x)
        hu = self._observe(ev.state)
        return tuple(a + b for a, b in zip(hu, self._offset, strict=True))

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> tuple[float, ...]:
        """Apply the Jacobian: one tangent solve ``K du = -(dr/dx)[v]``.

        Args:
            x: Log-conductivities.
            v: Direction in parameter space (length ``n``).

        Returns:
            ``J(x) v = H du``.

        Raises:
            ValueError: If an argument is invalid or out of range.

        References:
            [EQ-INV-PDE-003].

        """
        ev = self._evaluate(x)
        dirn = _finite_vector("v", v, self.cells)
        n = self.cells
        energy = [0.0] * (n + 1)
        for f in range(n + 1):
            dk = 0.0
            if f >= 1:
                dk += ev.left_weight[f] * dirn[f - 1]
            if f < n:
                dk += ev.right_weight[f] * dirn[f]
            energy[f] = ev.conductance[f] * dk * ev.jump[f]
        rhs = [energy[i + 1] - energy[i] for i in range(n)]
        du = ev.factorization.solve(rhs)
        self._tangent_solves += 1
        return tuple(self._observe(du))

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> tuple[float, ...]:
        """Apply the transposed Jacobian: one adjoint solve ``K' p = H' w``.

        Args:
            x: Log-conductivities.
            w: Covector in observation space (length ``output_dimension``).

        Returns:
            ``J(x)' w`` with component ``i`` equal to ``-p' (dK/dx_i) u``
            (including the dependence of the boundary right-hand side).

        Raises:
            ValueError: If an argument is invalid or out of range.

        References:
            [EQ-INV-PDE-004].

        """
        ev = self._evaluate(x)
        weights = _finite_vector("w", w, self.output_dimension)
        n = self.cells
        p = ev.factorization.solve_transpose(self._observe_transpose(weights))
        self._adjoint_solves += 1
        padded = [0.0, *p, 0.0]
        gradient = [0.0] * n
        for f in range(n + 1):
            t = ev.conductance[f] * ev.jump[f] * (padded[f + 1] - padded[f])
            if f >= 1:
                gradient[f - 1] -= t * ev.left_weight[f]
            if f < n:
                gradient[f] -= t * ev.right_weight[f]
        return tuple(gradient)

    def face_fluxes(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return the fluxes ``F_f = -c_f (u_f - u_(f-1))`` at the ``n + 1`` faces.

        Args:
            x: Log-conductivities.

        Returns:
            Face fluxes ``-k u'`` from the left boundary to the right one.

        Raises:
            ValueError: If ``x`` is invalid or out of range.

        References:
            [EQ-INV-PDE-006].

        """
        ev = self._evaluate(x)
        return tuple(-c * j for c, j in zip(ev.conductance, ev.jump, strict=True))

    def flux_balance(self, x: Sequence[float]) -> FluxBalance:
        """Measure conservation of the computed solution.

        Args:
            x: Log-conductivities.

        Returns:
            The cell-wise and global flux residuals.

        Raises:
            ValueError: If ``x`` is invalid or out of range.

        References:
            [EQ-INV-PDE-006].

        """
        flux = self.face_fluxes(x)
        h = self._h
        cell = max(
            abs(flux[i + 1] - flux[i] - h * self._q[i]) for i in range(self.cells)
        )
        total = h * math.fsum(self._q)
        return FluxBalance(
            cell_residual=cell,
            global_imbalance=abs(flux[-1] - flux[0] - total),
            total_source=total,
            left_flux=flux[0],
            right_flux=flux[-1],
        )


def adjoint_gradient(
    model: _ForwardLike,
    x: Sequence[float],
    data: Sequence[float],
    noise: float | Sequence[float],
) -> AdjointGradient:
    """Gradient of ``1/2 ||W (G(x) - y)||^2`` by one forward and one adjoint solve.

    ``W = diag(1 / sigma)`` with ``sigma`` the noise standard deviations, so
    the gradient is ``J' diag(sigma^-2) (G(x) - y)``; the Jacobian ``J`` is
    never formed.

    Args:
        model: Object with ``output_dimension``, ``value`` and ``vjp``.
        x: Parameters.
        data: Observations ``y``.
        noise: Positive standard deviation, scalar or one per observation.

    Returns:
        Objective, gradient and residual.

    Raises:
        ValueError: If the data length does not match, or a standard
            deviation is not positive and finite.

    References:
        [EQ-INV-PDE-008], [REF-KAIPIO-SOMERSALO-2007].

    """
    m = model.output_dimension
    y = _finite_vector("data", data, m)
    if isinstance(noise, (int, float)) and not isinstance(noise, bool):
        sigma = [float(noise)] * m
    elif isinstance(noise, Sequence):
        sigma = _finite_vector("noise", noise, m)
    else:
        raise TypeError("noise must be a number or a sequence")
    if not all(s > 0.0 for s in sigma) or not all(math.isfinite(s) for s in sigma):
        raise ValueError("noise standard deviations must be positive and finite")
    predicted = model.value(x)
    residual = tuple(g - d for g, d in zip(predicted, y, strict=True))
    objective = 0.5 * math.fsum(
        (r / s) ** 2 for r, s in zip(residual, sigma, strict=True)
    )
    weighted = [r / (s * s) for r, s in zip(residual, sigma, strict=True)]
    return AdjointGradient(objective, tuple(model.vjp(x, weighted)), residual)
