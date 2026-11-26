"""Two-dimensional elliptic inversion and the explicit-Euler stability case.

Experimental (added in 0.28). Two small, verified PDE-constrained forward
models for inverse problems; not a finite-element package.

**Two-dimensional elliptic diffusion.** :class:`EllipticDiffusion2D` solves
``-div(k grad u) = q`` on the rectangle ``(0, Lx) x (0, Ly)`` with
homogeneous Dirichlet values on the whole boundary. The domain is split into
``nx * ny`` rectangular cells (flat index ``p = j nx + i``, ``i`` along ``x``)
and the parameter is ``x_p = log k_p`` per cell, so ``k = exp(x) > 0`` by
construction. The scheme is the cell-centred finite-volume (five-point)
method [EQ-INV-PDE2-001]: the flux through a face between cells ``a`` and
``b`` is ``c_f (u_a - u_b)`` with

* interior ``x``-face: ``c_f = k_f hy / hx``, ``k_f = 2 k_a k_b / (k_a + k_b)``
  (harmonic mean) and the ``y``-face analogue with ``hx / hy``;
* boundary face: ``c_f = 2 k_a hy / hx`` (half-cell distance to the Dirichlet
  value, which is zero) [EQ-INV-PDE2-002].

Conservation in every cell gives ``K(x) u = f``, ``f_p = hx hy q_p``, with
``K`` symmetric and, thanks to the boundary faces, positive definite.

The state equation is solved either by Jacobi-preconditioned conjugate
gradients without forming ``K`` (``solver="pcg"``, default relative residual
tolerance ``1e-12``, iteration cap ``max(200, 4 N)``) or by a banded Cholesky
factorization (``solver="banded_cholesky"``, bandwidth ``nx``, exact up to
roundoff, factor reused for the tangent and adjoint solves)
[EQ-INV-PDE2-005]. The TRUE residual ``||f - K u|| / ||f||`` is recomputed
after every solve and recorded in a :class:`LinearSolveRecord`. A solve that
does not reach the tolerance is never hidden: ``value`` then returns a vector
of ``nan`` (a failed forward evaluation, which the Gauss-Newton solvers treat
as a rejected trial), ``jvp`` / ``vjp`` / ``state`` raise
:class:`LinearSolverError`, and ``linear_failures`` counts the failures.

The model implements the ``DifferentiableForward`` protocol of
:mod:`probstat.inverse.gauss_newton`. The derivatives are exact derivatives of
the DISCRETE model: the tangent solves ``K du = -(dK/dx)[v] u`` [EQ-INV-PDE2-003]
and, because ``K`` is symmetric, the adjoint solves with the same matrix,
``K p = H' w``, and returns ``-p' (dK/dx_i) u`` per cell [EQ-INV-PDE2-004]. A
scalar misfit gradient therefore costs one forward and one adjoint solve,
whatever the number of parameters. Observations are bilinear interpolations of
the cell-centre values (through the zero boundary values) at point sensors, or
averages of the cell centres inside a square window around each sensor
[EQ-INV-PDE2-006].

The tolerance of the linear solver is part of the inference error: the
derivatives are exact for the discrete model only up to the solver tolerance
(and exact to roundoff with the Cholesky solver). The discretization error is
part of it too: :func:`synthetic_fine_grid_data` generates data on a refined
grid so that an inversion on the coarse model does not commit the inverse
crime, and flags the case ``refinement = 1`` that does.

**Explicit Euler for the heat equation.** The semi-discrete heat equation
``u' = -D u`` of :mod:`probstat.inverse.heat_models` has the explicit step
``u^(k+1) = (I - dt D) u^k`` with Fourier (sine) modes as eigenvectors and
amplification factors ``g_j = 1 - 4 r sin^2(j pi / (2 (n + 1)))``,
``r = kappa dt / h^2`` [EQ-INV-PDE2-008]. The scheme is stable iff
``|g_j| <= 1`` for all modes, guaranteed for ``dt <= h^2 / (2 kappa)`` in 1D
(``h^2 / (4 kappa)`` in 2D, ``h^2 / (2 d kappa)`` for ``d`` dimensions)
[EQ-INV-PDE2-009]. Above the limit the highest-frequency modes are amplified
by ``|g_j|^N``. :class:`HeatExplicitEuler1D` is the matrix-free map from the
initial state to observations; its ``rmatvec`` is the exact transpose
(backward sweep with ``I - dt D``, which is symmetric) [EQ-INV-PDE2-010].
Policy for an unstable step: the operator REFUSES to be constructed
(:class:`UnstableTimeStepError`) unless ``allow_unstable=True`` is passed, in
which case ``stable`` is ``False`` and the growth bound is reported;
:func:`explicit_heat_inverse_problem` refuses such an operator a second time
unless ``allow_unstable=True`` is passed to it too [EQ-INV-PDE2-011].

References:
    [REF-GOLUB-VANLOAN-2013], [REF-HESTENES-STIEFEL-1952],
    [REF-KAIPIO-SOMERSALO-2007]. No numerical-PDE textbook key exists in
    ``docs/references.md``; the finite-volume and amplification-factor
    statements are elementary and verified in
    ``tests/test_pde_models_2d.py``.

"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.foundations.random_streams import resolve_stream
from probstat.inverse.linear_problem import LinearInverseProblem
from probstat.inverse.pde_models import PDESolveCounts
from probstat.linalg import OperatorBase

_LOG_K_LIMIT = 345.0
_SOLVERS = ("pcg", "banded_cholesky")
_DRIFT_FACTOR = 10.0


class LinearSolverError(RuntimeError):
    """Raised when a linear solve of a PDE model did not reach its tolerance.

    References:
        [EQ-INV-PDE2-005].

    """


class UnstableTimeStepError(ValueError):
    """Raised when an explicit time step exceeds the stability limit.

    References:
        [EQ-INV-PDE2-011].

    """


@dataclass(frozen=True, slots=True)
class LinearSolveRecord:
    """Outcome of one linear solve ``K u = b``.

    Attributes:
        solver: ``"pcg"`` or ``"banded_cholesky"``.
        iterations: CG iterations (``0`` for the Cholesky solver).
        relative_residual: TRUE ``||b - K u|| / ||b||`` (``0`` for ``b = 0``).
        converged: Whether ``relative_residual`` met the criterion (PCG:
            ``<= 10 tolerance``, the factor allowing for recurrence drift;
            Cholesky: ``<= 1e-8``, a roundoff sanity check) and the
            factorization had positive pivots.

    References:
        [EQ-INV-PDE2-005].

    """

    solver: str
    iterations: int
    relative_residual: float
    converged: bool


@dataclass(frozen=True, slots=True)
class FluxBalance2D:
    """Global conservation diagnostic of the 2D finite-volume solution.

    Attributes:
        boundary_flux: Total outward flux ``sum c_f u_a`` over boundary faces.
        total_source: ``sum_p hx hy q_p``.
        imbalance: ``|boundary_flux - total_source|``.

    References:
        [EQ-INV-PDE2-007].

    """

    boundary_flux: float
    total_source: float
    imbalance: float


@dataclass(frozen=True, slots=True)
class FineGridData:
    """Synthetic data generated on a refined grid (no inverse crime).

    Attributes:
        data: Noisy observations from the refined model.
        clean: Noise-free observations of the refined model.
        noise_sd: Noise standard deviation.
        truth_fine: Log-conductivity of the truth on the refined cells.
        truth_coarse: The same function sampled at the coarse cell centres.
        refinement: Refinement factor of the data-generating grid.
        inverse_crime: ``True`` iff ``refinement == 1`` (data and inversion
            share one discretization, so discretization error is hidden).

    References:
        [EQ-INV-PDE2-001].

    """

    data: tuple[float, ...]
    clean: tuple[float, ...]
    noise_sd: float
    truth_fine: tuple[float, ...]
    truth_coarse: tuple[float, ...]
    refinement: int
    inverse_crime: bool


def _int_at_least(name: str, value: object, minimum: int) -> int:
    """Validate an integer lower bound.

    Args:
        name: Argument name.
        value: Candidate.
        minimum: Smallest allowed value.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < minimum``.

    References:
        [EQ-INV-PDE2-001]. Input validation only.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _positive(name: str, value: object) -> float:
    """Validate a positive finite real.

    Args:
        name: Argument name.
        value: Candidate.

    Returns:
        The float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not positive and finite.

    References:
        [EQ-INV-PDE2-001]. Input validation only.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return number


def _vector(name: str, values: Sequence[float], length: int | None) -> list[float]:
    """Validate a vector of finite reals.

    Args:
        name: Argument name.
        values: Candidate sequence.
        length: Required length or ``None`` for any positive length.

    Returns:
        The values as floats.

    Raises:
        TypeError: If ``values`` is not a sequence of reals.
        ValueError: If a length or finiteness requirement fails.

    References:
        [EQ-INV-PDE2-001]. Input validation only.

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


def _norm(values: Sequence[float]) -> float:
    """Return the Euclidean norm.

    Args:
        values: Vector.

    Returns:
        ``||values||_2``.

    References:
        [EQ-INV-PDE2-005].

    """
    return math.sqrt(math.fsum(v * v for v in values))


@dataclass(slots=True)
class _Evaluation:
    """Cached state of one parameter vector (internal)."""

    x: tuple[float, ...]
    cx: list[float]
    cy: list[float]
    lx: list[float]
    rx: list[float]
    ly: list[float]
    ry: list[float]
    diagonal: list[float]
    state: list[float]
    record: LinearSolveRecord
    cholesky: list[list[float]] | None


class EllipticDiffusion2D:
    """Cell-centred finite-volume model ``-div(k grad u) = q`` on a rectangle.

    The map ``x -> G(x) = H u(x)`` (``x = log k`` per cell, ``u`` the cell
    averages, ``H`` the sensor operator) is a differentiable forward model
    with ``input_dimension = nx ny`` and ``output_dimension = len(sensors)``.
    The last evaluated parameter vector is cached with its (factorized)
    system, so ``value`` followed by ``jvp`` or ``vjp`` at the same ``x``
    costs one forward solve in total.

    Attributes:
        nx: Cells in the ``x`` direction.
        ny: Cells in the ``y`` direction.
        lengths: Domain side lengths ``(Lx, Ly)``.
        sensors: Sensor positions ``(sx, sy)`` strictly inside the domain.
        average_radius: ``None`` for point sensors, else the half-width of
            the square averaging window.
        solver: ``"pcg"`` or ``"banded_cholesky"``.
        tolerance: Relative residual tolerance of PCG.

    References:
        [EQ-INV-PDE2-001], [EQ-INV-PDE2-002], [REF-KAIPIO-SOMERSALO-2007].

    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        sensors: Sequence[tuple[float, float]],
        lengths: tuple[float, float] = (1.0, 1.0),
        source: float | Sequence[float] | Callable[[float, float], float] = 0.0,
        average_radius: float | None = None,
        solver: str = "pcg",
        tolerance: float = 1e-12,
        max_iterations: int | None = None,
    ) -> None:
        """Build the model.

        Args:
            nx: Cells along ``x`` (``>= 2``).
            ny: Cells along ``y`` (``>= 2``).
            sensors: Sensor positions strictly inside ``(0, Lx) x (0, Ly)``.
            lengths: Side lengths ``(Lx, Ly)``.
            source: Constant, callable ``q(x, y)`` sampled at cell centres, or
                a flat vector of ``nx ny`` cell values (index ``j nx + i``).
            average_radius: ``None`` for bilinear point observations, or a
                positive half-width: each observation is the mean of the cell
                centres inside the square window around the sensor.
            solver: ``"pcg"`` (matrix-free, Jacobi preconditioner) or
                ``"banded_cholesky"``.
            tolerance: Relative residual tolerance of PCG (``> 0``).
            max_iterations: PCG iteration cap (default ``max(200, 4 N)``).

        Raises:
            TypeError: On wrongly typed arguments.
            ValueError: If a size, length, sensor, source, tolerance or
                solver name is invalid, or a sensor window contains no cell
                centre.

        References:
            [EQ-INV-PDE2-001], [EQ-INV-PDE2-006].

        """
        self.nx = _int_at_least("nx", nx, 2)
        self.ny = _int_at_least("ny", ny, 2)
        lx, ly = _vector("lengths", lengths, 2)
        self.lengths = (_positive("lengths[0]", lx), _positive("lengths[1]", ly))
        self._hx = self.lengths[0] / self.nx
        self._hy = self.lengths[1] / self.ny
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}")
        self.solver = solver
        self.tolerance = _positive("tolerance", tolerance)
        total = self.nx * self.ny
        self._max_iterations = (
            max(200, 4 * total)
            if max_iterations is None
            else _int_at_least("max_iterations", max_iterations, 1)
        )
        self._source_spec = source
        centres = self.cell_centres
        if callable(source):
            q = [float(source(cx, cy)) for cx, cy in centres]
            if not all(math.isfinite(v) for v in q):
                raise ValueError("source values must be finite")
        elif isinstance(source, (int, float)) and not isinstance(source, bool):
            if not math.isfinite(float(source)):
                raise ValueError("source must be finite")
            q = [float(source)] * total
        elif isinstance(source, Sequence):
            q = _vector("source", source, total)
        else:
            raise TypeError("source must be a number, a callable, or a sequence")
        self._rhs = [self._hx * self._hy * v for v in q]
        self._q = q
        if isinstance(sensors, (str, bytes)) or not isinstance(sensors, Sequence):
            raise TypeError("sensors must be a sequence of (x, y) pairs")
        if len(sensors) == 0:
            raise ValueError("sensors must not be empty")
        points: list[tuple[float, float]] = []
        for pair in sensors:
            sx, sy = _vector("sensor", pair, 2)
            if not (0.0 < sx < self.lengths[0] and 0.0 < sy < self.lengths[1]):
                raise ValueError(f"sensor ({sx}, {sy}) must lie inside the domain")
            points.append((sx, sy))
        self.sensors = tuple(points)
        self.average_radius = (
            None
            if average_radius is None
            else _positive("average_radius", average_radius)
        )
        self._observation = self._build_observation()
        self._cache: _Evaluation | None = None
        self._forward_solves = 0
        self._tangent_solves = 0
        self._adjoint_solves = 0
        self._linear_iterations = 0
        self._linear_failures = 0
        self._last_record: LinearSolveRecord | None = None
        self._worst_residual = 0.0

    @property
    def cell_centres(self) -> tuple[tuple[float, float], ...]:
        """Return the cell centres in flat order ``p = j nx + i``.

        Returns:
            Tuple of ``(x, y)`` centres.

        References:
            [EQ-INV-PDE2-001].

        """
        return tuple(
            ((i + 0.5) * self._hx, (j + 0.5) * self._hy)
            for j in range(self.ny)
            for i in range(self.nx)
        )

    @property
    def input_dimension(self) -> int:
        """Return the number of parameters ``nx ny``.

        Returns:
            The input dimension.

        References:
            [EQ-INV-PDE2-002].

        """
        return self.nx * self.ny

    @property
    def output_dimension(self) -> int:
        """Return the number of observations.

        Returns:
            The output dimension (number of sensors).

        References:
            [EQ-INV-PDE2-006].

        """
        return len(self.sensors)

    @property
    def counts(self) -> PDESolveCounts:
        """Return the numbers of forward, tangent and adjoint solves.

        Returns:
            The work counters (each a solve with the same matrix).

        References:
            [EQ-INV-PDE2-004].

        """
        return PDESolveCounts(
            self._forward_solves, self._tangent_solves, self._adjoint_solves
        )

    @property
    def linear_iterations(self) -> int:
        """Return the total PCG iterations of all solves so far.

        Returns:
            Iteration count (zero for the Cholesky solver).

        References:
            [EQ-INV-PDE2-005].

        """
        return self._linear_iterations

    @property
    def linear_failures(self) -> int:
        """Return the number of linear solves that did not converge.

        Returns:
            Failure count; failures are also raised or returned as ``nan``.

        References:
            [EQ-INV-PDE2-005].

        """
        return self._linear_failures

    @property
    def last_solve(self) -> LinearSolveRecord | None:
        """Return the record of the most recent linear solve.

        Returns:
            The record, or ``None`` before the first solve.

        References:
            [EQ-INV-PDE2-005].

        """
        return self._last_record

    @property
    def worst_relative_residual(self) -> float:
        """Return the largest TRUE relative residual of any solve so far.

        Returns:
            ``max ||b - K u|| / ||b||`` over all solves.

        References:
            [EQ-INV-PDE2-005].

        """
        return self._worst_residual

    def reset_counts(self) -> None:
        """Zero the solve and iteration counters (not the cache).

        References:
            [EQ-INV-PDE2-004].

        """
        self._forward_solves = 0
        self._tangent_solves = 0
        self._adjoint_solves = 0
        self._linear_iterations = 0
        self._linear_failures = 0
        self._worst_residual = 0.0

    def refined(self, factor: int = 2) -> EllipticDiffusion2D:
        """Return the same problem on a grid refined by ``factor`` per direction.

        Args:
            factor: Integer ``>= 1``.

        Returns:
            A new model with ``factor nx`` by ``factor ny`` cells, the same
            sensors, source function, solver and tolerance.

        Raises:
            ValueError: If ``factor < 1`` or the source was given as a cell
                vector (it cannot be transferred to another grid).

        References:
            [EQ-INV-PDE2-001].

        """
        f = _int_at_least("factor", factor, 1)
        if isinstance(self._source_spec, Sequence):
            raise ValueError("a model with a cell-vector source cannot be refined")
        return EllipticDiffusion2D(
            self.nx * f,
            self.ny * f,
            sensors=self.sensors,
            lengths=self.lengths,
            source=self._source_spec,
            average_radius=self.average_radius,
            solver=self.solver,
            tolerance=self.tolerance,
            max_iterations=self._max_iterations,
        )

    def _build_observation(self) -> list[tuple[list[int], list[float]]]:
        """Assemble the sparse rows of the sensor operator ``H``.

        Returns:
            For each sensor the cell indices and weights.

        Raises:
            ValueError: If an averaging window contains no cell centre.

        References:
            [EQ-INV-PDE2-006].

        """
        hx, hy, nx, ny = self._hx, self._hy, self.nx, self.ny
        rows: list[tuple[list[int], list[float]]] = []
        if self.average_radius is not None:
            r = self.average_radius
            slack = 1e-12 * max(self.lengths)
            for sx, sy in self.sensors:
                idx = [
                    j * nx + i
                    for j in range(ny)
                    for i in range(nx)
                    if abs((i + 0.5) * hx - sx) <= r + slack
                    and abs((j + 0.5) * hy - sy) <= r + slack
                ]
                if not idx:
                    raise ValueError(
                        f"averaging window of sensor ({sx}, {sy}) holds no cell"
                    )
                rows.append((idx, [1.0 / len(idx)] * len(idx)))
            return rows
        nodes_x = [0.0, *((i + 0.5) * hx for i in range(nx)), self.lengths[0]]
        nodes_y = [0.0, *((j + 0.5) * hy for j in range(ny)), self.lengths[1]]

        def axis(
            nodes: list[float], position: float, cells: int
        ) -> list[tuple[int, float]]:
            """Return ``(cell, weight)`` pairs of the 1D hat interpolation.

            References:
                [EQ-INV-PDE2-006].

            """
            m = min(max(bisect_right(nodes, position) - 1, 0), cells)
            t = (position - nodes[m]) / (nodes[m + 1] - nodes[m])
            pairs = [(m - 1, 1.0 - t), (m, t)]
            return [(c, w) for c, w in pairs if 0 <= c < cells]

        for sx, sy in self.sensors:
            idx = []
            wts = []
            for cj, wj in axis(nodes_y, sy, ny):
                for ci, wi in axis(nodes_x, sx, nx):
                    idx.append(cj * nx + ci)
                    wts.append(wi * wj)
            rows.append((idx, wts))
        return rows

    def _observe(self, u: Sequence[float]) -> list[float]:
        """Apply ``H`` to a cell vector.

        Args:
            u: Cell values.

        Returns:
            The sensor readings.

        References:
            [EQ-INV-PDE2-006].

        """
        return [
            math.fsum(w * u[i] for i, w in zip(idx, wts, strict=True))
            for idx, wts in self._observation
        ]

    def _observe_transpose(self, w: Sequence[float]) -> list[float]:
        """Apply ``H'`` to an observation vector.

        Args:
            w: Observation-space vector.

        Returns:
            The cell vector ``H' w``.

        References:
            [EQ-INV-PDE2-006].

        """
        out = [0.0] * (self.nx * self.ny)
        for (idx, wts), value in zip(self._observation, w, strict=True):
            for i, weight in zip(idx, wts, strict=True):
                out[i] += weight * value
        return out

    def _apply(self, ev: _Evaluation, u: Sequence[float]) -> list[float]:
        """Apply the stencil matrix ``K(x)`` without forming it.

        Args:
            ev: Assembled face data.
            u: Cell vector.

        Returns:
            ``K u``.

        References:
            [EQ-INV-PDE2-001].

        """
        nx, ny = self.nx, self.ny
        cx, cy = ev.cx, ev.cy
        out = [0.0] * (nx * ny)
        for j in range(ny):
            row = j * nx
            for i in range(nx):
                p = row + i
                value = ev.diagonal[p] * u[p]
                if i > 0:
                    value -= cx[j * (nx + 1) + i] * u[p - 1]
                if i < nx - 1:
                    value -= cx[j * (nx + 1) + i + 1] * u[p + 1]
                if j > 0:
                    value -= cy[j * nx + i] * u[p - nx]
                if j < ny - 1:
                    value -= cy[(j + 1) * nx + i] * u[p + nx]
                out[p] = value
        return out

    def _factorize(self, ev: _Evaluation) -> list[list[float]] | None:
        """Banded Cholesky factor ``K = L L'`` (lower band, bandwidth ``nx``).

        Args:
            ev: Assembled face data.

        Returns:
            Row ``p`` holds ``L[p, p - d]`` at index ``d`` (``0 <= d <= nx``),
            or ``None`` if a pivot is not positive.

        References:
            [EQ-INV-PDE2-005], [REF-GOLUB-VANLOAN-2013].

        """
        nx, n = self.nx, self.nx * self.ny
        band = nx
        factor: list[list[float]] = []
        for p in range(n):
            i, j = p % nx, p // nx
            lp = [0.0] * (band + 1)
            for d in range(min(band, p), 0, -1):
                q = p - d
                if d == 1 and i > 0:
                    s = -ev.cx[j * (nx + 1) + i]
                elif d == nx and j > 0:
                    s = -ev.cy[j * nx + i]
                else:
                    s = 0.0
                lq = factor[q]
                for r in range(max(p - band, 0), q):
                    s -= lp[p - r] * lq[q - r]
                lp[d] = s / lq[0]
            pivot = ev.diagonal[p] - math.fsum(v * v for v in lp[1:])
            if not pivot > 0.0:
                return None
            lp[0] = math.sqrt(pivot)
            factor.append(lp)
        return factor

    def _cholesky_solve(
        self, factor: list[list[float]], rhs: Sequence[float]
    ) -> list[float]:
        """Solve ``L L' u = rhs`` by forward and backward substitution.

        Args:
            factor: Banded factor from :meth:`_factorize`.
            rhs: Right-hand side.

        Returns:
            The solution.

        References:
            [EQ-INV-PDE2-005].

        """
        n, band = len(factor), self.nx
        y = [0.0] * n
        for p in range(n):
            s = rhs[p]
            for d in range(1, min(band, p) + 1):
                s -= factor[p][d] * y[p - d]
            y[p] = s / factor[p][0]
        u = [0.0] * n
        for p in range(n - 1, -1, -1):
            s = y[p]
            for d in range(1, min(band, n - 1 - p) + 1):
                s -= factor[p + d][d] * u[p + d]
            u[p] = s / factor[p][0]
        return u

    def _solve(self, ev: _Evaluation, rhs: Sequence[float]) -> list[float]:
        """Solve ``K u = rhs`` with the configured solver and record the outcome.

        Args:
            ev: Assembled face data (and factor, if any).
            rhs: Right-hand side.

        Returns:
            The (possibly unconverged) solution; the record is stored in
            ``last_solve`` and failures are counted.

        References:
            [EQ-INV-PDE2-005], [REF-HESTENES-STIEFEL-1952].

        """
        b_norm = _norm(rhs)
        iterations = 0
        ok = True
        if b_norm == 0.0:
            u = [0.0] * len(rhs)
        elif self.solver == "banded_cholesky":
            if ev.cholesky is None:
                u = [math.nan] * len(rhs)
                ok = False
            else:
                u = self._cholesky_solve(ev.cholesky, rhs)
        else:
            u, iterations = self._pcg(ev, rhs, b_norm)
        if b_norm == 0.0:
            rel = 0.0
        else:
            applied = self._apply(ev, u)
            rel = _norm([a - b for a, b in zip(rhs, applied, strict=True)]) / b_norm
        limit = _DRIFT_FACTOR * self.tolerance if self.solver == "pcg" else 1e-8
        converged = ok and rel <= limit
        if math.isnan(rel):
            rel = math.inf
        record = LinearSolveRecord(self.solver, iterations, rel, converged)
        self._last_record = record
        self._linear_iterations += iterations
        self._worst_residual = max(self._worst_residual, rel)
        if not converged:
            self._linear_failures += 1
        return u

    def _pcg(
        self, ev: _Evaluation, rhs: Sequence[float], b_norm: float
    ) -> tuple[list[float], int]:
        """Jacobi-preconditioned conjugate gradients from a zero start.

        Args:
            ev: Assembled face data.
            rhs: Right-hand side.
            b_norm: ``||rhs||``.

        Returns:
            The iterate and the iteration count.

        References:
            [EQ-INV-PDE2-005], [REF-HESTENES-STIEFEL-1952].

        """
        diag = ev.diagonal
        u = [0.0] * len(rhs)
        r = list(rhs)
        z = [a / d for a, d in zip(r, diag, strict=True)]
        p = list(z)
        rz = math.fsum(a * b for a, b in zip(r, z, strict=True))
        iterations = 0
        while iterations < self._max_iterations:
            if _norm(r) <= self.tolerance * b_norm:
                break
            kp = self._apply(ev, p)
            curvature = math.fsum(a * b for a, b in zip(p, kp, strict=True))
            if not curvature > 0.0:
                break
            alpha = rz / curvature
            u = [a + alpha * b for a, b in zip(u, p, strict=True)]
            r = [a - alpha * b for a, b in zip(r, kp, strict=True)]
            z = [a / d for a, d in zip(r, diag, strict=True)]
            rz_new = math.fsum(a * b for a, b in zip(r, z, strict=True))
            p = [a + (rz_new / rz) * b for a, b in zip(z, p, strict=True)]
            rz = rz_new
            iterations += 1
        return u, iterations

    def _check_x(self, x: Sequence[float]) -> tuple[float, ...]:
        """Validate a parameter vector including the exponent range.

        Args:
            x: Log-conductivities.

        Returns:
            The validated vector.

        Raises:
            ValueError: If ``x`` is invalid or ``|x_i|`` is beyond the range
                where ``exp`` is representable with headroom.

        References:
            [EQ-INV-PDE2-002].

        """
        values = _vector("x", x, self.nx * self.ny)
        for i, v in enumerate(values):
            if abs(v) > _LOG_K_LIMIT:
                raise ValueError(
                    f"x[{i}] = {v} gives exp(x) outside [1e-150, 1e150]; "
                    "conductivity overflow or underflow"
                )
        return tuple(values)

    def _evaluate(self, x: Sequence[float]) -> _Evaluation:
        """Assemble and solve the state equation, using the cache when possible.

        Args:
            x: Log-conductivities.

        Returns:
            The cached evaluation for ``x`` (it may carry a failed record).

        References:
            [EQ-INV-PDE2-001], [EQ-INV-PDE2-002].

        """
        key = self._check_x(x)
        if self._cache is not None and self._cache.x == key:
            return self._cache
        nx, ny = self.nx, self.ny
        k = [math.exp(v) for v in key]
        ax, ay = self._hy / self._hx, self._hx / self._hy
        cx = [0.0] * ((nx + 1) * ny)
        lx = [0.0] * ((nx + 1) * ny)
        rx = [0.0] * ((nx + 1) * ny)
        for j in range(ny):
            for f in range(nx + 1):
                at = j * (nx + 1) + f
                if f == 0:
                    cx[at] = 2.0 * k[j * nx] * ax
                    rx[at] = 1.0
                elif f == nx:
                    cx[at] = 2.0 * k[j * nx + nx - 1] * ax
                    lx[at] = 1.0
                else:
                    ka, kb = k[j * nx + f - 1], k[j * nx + f]
                    cx[at] = 2.0 / (1.0 / ka + 1.0 / kb) * ax
                    lx[at] = kb / (ka + kb)
                    rx[at] = ka / (ka + kb)
        cy = [0.0] * (nx * (ny + 1))
        ly = [0.0] * (nx * (ny + 1))
        ry = [0.0] * (nx * (ny + 1))
        for f in range(ny + 1):
            for i in range(nx):
                at = f * nx + i
                if f == 0:
                    cy[at] = 2.0 * k[i] * ay
                    ry[at] = 1.0
                elif f == ny:
                    cy[at] = 2.0 * k[(ny - 1) * nx + i] * ay
                    ly[at] = 1.0
                else:
                    ka, kb = k[(f - 1) * nx + i], k[f * nx + i]
                    cy[at] = 2.0 / (1.0 / ka + 1.0 / kb) * ay
                    ly[at] = kb / (ka + kb)
                    ry[at] = ka / (ka + kb)
        diagonal = [
            cx[j * (nx + 1) + i]
            + cx[j * (nx + 1) + i + 1]
            + cy[j * nx + i]
            + cy[(j + 1) * nx + i]
            for j in range(ny)
            for i in range(nx)
        ]
        ev = _Evaluation(
            key,
            cx,
            cy,
            lx,
            rx,
            ly,
            ry,
            diagonal,
            [],
            LinearSolveRecord(self.solver, 0, 0.0, True),
            None,
        )
        if self.solver == "banded_cholesky":
            ev.cholesky = self._factorize(ev)
        ev.state = self._solve(ev, self._rhs)
        assert self._last_record is not None
        ev.record = self._last_record
        self._forward_solves += 1
        self._cache = ev
        return ev

    def _require_converged(self, ev: _Evaluation) -> None:
        """Raise if the forward solve of an evaluation failed.

        Args:
            ev: The evaluation.

        Raises:
            LinearSolverError: If the stored forward solve did not converge.

        References:
            [EQ-INV-PDE2-005].

        """
        if not ev.record.converged:
            raise LinearSolverError(
                f"forward {ev.record.solver} solve failed: relative residual "
                f"{ev.record.relative_residual:.3e} after "
                f"{ev.record.iterations} iterations"
            )

    def _checked_solve(self, ev: _Evaluation, rhs: Sequence[float]) -> list[float]:
        """Solve an auxiliary system and raise if it did not converge.

        Args:
            ev: The evaluation.
            rhs: Right-hand side.

        Returns:
            The solution.

        Raises:
            LinearSolverError: If the solve did not converge.

        References:
            [EQ-INV-PDE2-005].

        """
        u = self._solve(ev, rhs)
        record = self._last_record
        assert record is not None
        if not record.converged:
            raise LinearSolverError(
                f"{record.solver} solve failed: relative residual "
                f"{record.relative_residual:.3e} after {record.iterations} iterations"
            )
        return u

    def state(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return the cell-average state ``u(x)``.

        Args:
            x: Log-conductivities (length ``nx ny``).

        Returns:
            The solution on the cells (flat order ``j nx + i``).

        Raises:
            ValueError: If ``x`` is invalid or out of range.
            LinearSolverError: If the linear solve did not converge.

        References:
            [EQ-INV-PDE2-001].

        """
        ev = self._evaluate(x)
        self._require_converged(ev)
        return tuple(ev.state)

    def value(self, x: Sequence[float]) -> tuple[float, ...]:
        """Return the sensor readings ``G(x) = H u(x)``.

        A linear solve that did not converge yields a vector of ``nan`` (a
        failed forward evaluation) and increments ``linear_failures``.

        Args:
            x: Log-conductivities.

        Returns:
            Predicted observations, or ``nan`` entries on solver failure.

        Raises:
            ValueError: If ``x`` is invalid or out of the supported range.

        References:
            [EQ-INV-PDE2-006].

        """
        ev = self._evaluate(x)
        if not ev.record.converged:
            return (math.nan,) * self.output_dimension
        return tuple(self._observe(ev.state))

    def _face_energy(
        self, ev: _Evaluation, v: Sequence[float], u: Sequence[float]
    ) -> list[float]:
        """Return ``(dK/dx)[v] u`` for a parameter direction ``v``.

        Args:
            ev: Assembled face data.
            v: Direction in parameter space.
            u: State.

        Returns:
            The cell vector ``(dK[v]) u``.

        References:
            [EQ-INV-PDE2-003].

        """
        nx, ny = self.nx, self.ny
        out = [0.0] * (nx * ny)
        for j in range(ny):
            for f in range(nx + 1):
                at = j * (nx + 1) + f
                dk = 0.0
                ua = ub = 0.0
                if f >= 1:
                    dk += ev.lx[at] * v[j * nx + f - 1]
                    ua = u[j * nx + f - 1]
                if f < nx:
                    dk += ev.rx[at] * v[j * nx + f]
                    ub = u[j * nx + f]
                flux = ev.cx[at] * dk * (ua - ub)
                if f >= 1:
                    out[j * nx + f - 1] += flux
                if f < nx:
                    out[j * nx + f] -= flux
        for f in range(ny + 1):
            for i in range(nx):
                at = f * nx + i
                dk = 0.0
                ua = ub = 0.0
                if f >= 1:
                    dk += ev.ly[at] * v[(f - 1) * nx + i]
                    ua = u[(f - 1) * nx + i]
                if f < ny:
                    dk += ev.ry[at] * v[f * nx + i]
                    ub = u[f * nx + i]
                flux = ev.cy[at] * dk * (ua - ub)
                if f >= 1:
                    out[(f - 1) * nx + i] += flux
                if f < ny:
                    out[f * nx + i] -= flux
        return out

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> tuple[float, ...]:
        """Apply the Jacobian: one tangent solve ``K du = -(dK/dx)[v] u``.

        Args:
            x: Log-conductivities.
            v: Direction in parameter space (length ``nx ny``).

        Returns:
            ``J(x) v = H du``.

        Raises:
            ValueError: If an argument is invalid or out of range.
            LinearSolverError: If the forward or tangent solve failed.

        References:
            [EQ-INV-PDE2-003].

        """
        ev = self._evaluate(x)
        self._require_converged(ev)
        direction = _vector("v", v, self.nx * self.ny)
        rhs = [-a for a in self._face_energy(ev, direction, ev.state)]
        du = self._checked_solve(ev, rhs)
        self._tangent_solves += 1
        return tuple(self._observe(du))

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> tuple[float, ...]:
        """Apply the transposed Jacobian: one adjoint solve ``K p = H' w``.

        ``K`` is symmetric, so the adjoint uses the same matrix (and the same
        factor, if the Cholesky solver is selected).

        Args:
            x: Log-conductivities.
            w: Covector in observation space (length ``output_dimension``).

        Returns:
            ``J(x)' w`` with component ``p`` equal to ``-p' (dK/dx_p) u``.

        Raises:
            ValueError: If an argument is invalid or out of range.
            LinearSolverError: If the forward or adjoint solve failed.

        References:
            [EQ-INV-PDE2-004].

        """
        ev = self._evaluate(x)
        self._require_converged(ev)
        weights = _vector("w", w, self.output_dimension)
        p = self._checked_solve(ev, self._observe_transpose(weights))
        self._adjoint_solves += 1
        u = ev.state
        nx, ny = self.nx, self.ny
        gradient = [0.0] * (nx * ny)
        for j in range(ny):
            for f in range(nx + 1):
                at = j * (nx + 1) + f
                ua = pa = ub = pb = 0.0
                if f >= 1:
                    ua, pa = u[j * nx + f - 1], p[j * nx + f - 1]
                if f < nx:
                    ub, pb = u[j * nx + f], p[j * nx + f]
                t = ev.cx[at] * (pa - pb) * (ua - ub)
                if f >= 1:
                    gradient[j * nx + f - 1] -= t * ev.lx[at]
                if f < nx:
                    gradient[j * nx + f] -= t * ev.rx[at]
        for f in range(ny + 1):
            for i in range(nx):
                at = f * nx + i
                ua = pa = ub = pb = 0.0
                if f >= 1:
                    ua, pa = u[(f - 1) * nx + i], p[(f - 1) * nx + i]
                if f < ny:
                    ub, pb = u[f * nx + i], p[f * nx + i]
                t = ev.cy[at] * (pa - pb) * (ua - ub)
                if f >= 1:
                    gradient[(f - 1) * nx + i] -= t * ev.ly[at]
                if f < ny:
                    gradient[f * nx + i] -= t * ev.ry[at]
        return tuple(gradient)

    def flux_balance(self, x: Sequence[float]) -> FluxBalance2D:
        """Measure global conservation of the computed solution.

        Summing the cell balances, the outward flux through the boundary
        faces equals the total source, ``sum_f c_f u_a = sum_p hx hy q_p``.

        Args:
            x: Log-conductivities.

        Returns:
            The boundary flux, the total source and their difference.

        Raises:
            ValueError: If ``x`` is invalid or out of range.
            LinearSolverError: If the forward solve failed.

        References:
            [EQ-INV-PDE2-007].

        """
        ev = self._evaluate(x)
        self._require_converged(ev)
        nx, ny, u = self.nx, self.ny, ev.state
        outward = 0.0
        terms: list[float] = []
        for j in range(ny):
            terms.append(ev.cx[j * (nx + 1)] * u[j * nx])
            terms.append(ev.cx[j * (nx + 1) + nx] * u[j * nx + nx - 1])
        for i in range(nx):
            terms.append(ev.cy[i] * u[i])
            terms.append(ev.cy[ny * nx + i] * u[(ny - 1) * nx + i])
        outward = math.fsum(terms)
        total = math.fsum(self._rhs)
        return FluxBalance2D(outward, total, abs(outward - total))


def synthetic_fine_grid_data(
    model: EllipticDiffusion2D,
    log_conductivity: Callable[[float, float], float],
    *,
    noise_sd: float,
    seed: int | None,
    refinement: int = 2,
) -> FineGridData:
    """Generate noisy data on a grid refined by ``refinement`` (no inverse crime).

    The truth is the function ``log k(x, y)`` sampled at the cell centres of
    the refined model; data are its sensor readings plus independent
    ``N(0, noise_sd^2)`` noise. Inverting these data with ``model`` (coarse)
    includes the discretization error in the inference error; with
    ``refinement = 1`` data and inversion coincide and the result is flagged
    ``inverse_crime = True``.

    Args:
        model: The inversion (coarse) model.
        log_conductivity: Truth ``log k(x, y)``.
        noise_sd: Noise standard deviation (``> 0``).
        seed: Seed for :func:`probstat.foundations.random_streams.resolve_stream`.
        refinement: Integer ``>= 1``.

    Returns:
        :class:`FineGridData`.

    Raises:
        TypeError: On wrongly typed arguments.
        ValueError: If ``noise_sd`` or ``refinement`` is invalid or the truth
            is not finite.
        LinearSolverError: If the data-generating solve fails.

    References:
        [EQ-INV-PDE2-001], [REF-KAIPIO-SOMERSALO-2007].

    """
    sd = _positive("noise_sd", noise_sd)
    fine = model.refined(refinement)
    truth_fine = tuple(float(log_conductivity(a, b)) for a, b in fine.cell_centres)
    truth_coarse = tuple(float(log_conductivity(a, b)) for a, b in model.cell_centres)
    clean = fine.value(truth_fine)
    if not all(math.isfinite(v) for v in clean):
        raise LinearSolverError("the data-generating solve failed")
    rng = resolve_stream(seed).generator()
    data = tuple(c + rng.gauss(0.0, sd) for c in clean)
    return FineGridData(
        data, clean, sd, truth_fine, truth_coarse, refinement, refinement == 1
    )


# ---------------------------------------------------------------------------
# Explicit Euler for the heat equation: stability and exact transpose.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExplicitStability:
    """Stability report of the explicit Euler heat step.

    Attributes:
        dimension: Spatial dimension (1 or 2).
        time_step: The step ``dt``.
        limit: ``h^2 / (2 d kappa)``, the classical bound.
        ratio: ``dt / limit``.
        stable: ``dt <= limit`` (conservative; see :func:`explicit_stability`).
        factors: Amplification factor of every Fourier mode.
        max_amplification: ``max_j |g_j|``.
        worst_mode: Index (``0``-based, flat) of the mode attaining it.

    References:
        [EQ-INV-PDE2-008], [EQ-INV-PDE2-009].

    """

    dimension: int
    time_step: float
    limit: float
    ratio: float
    stable: bool
    factors: tuple[float, ...]
    max_amplification: float
    worst_mode: int


def explicit_stability_limit(
    diffusivity: float, spacing: float, *, dimension: int = 1
) -> float:
    """Return the explicit-Euler stability limit ``h^2 / (2 d kappa)``.

    Args:
        diffusivity: ``kappa > 0``.
        spacing: Grid spacing ``h > 0``.
        dimension: Spatial dimension ``d`` in ``1, 2, 3``.

    Returns:
        The largest ``dt`` for which ``|g| <= 1`` on every grid size, i.e.
        ``h^2 / (2 kappa)`` in 1D and ``h^2 / (4 kappa)`` in 2D.

    Raises:
        TypeError: On wrongly typed arguments.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-PDE2-009].

    """
    kappa = _positive("diffusivity", diffusivity)
    h = _positive("spacing", spacing)
    d = _int_at_least("dimension", dimension, 1)
    if d > 3:
        raise ValueError("dimension must be 1, 2 or 3")
    return h * h / (2.0 * d * kappa)


def explicit_amplification_factors(
    diffusivity: float,
    spacing: float,
    time_step: float,
    cells: int,
    *,
    dimension: int = 1,
) -> tuple[float, ...]:
    """Return the amplification factor of every Fourier mode.

    With zero Dirichlet values the eigenvectors of ``D`` are sine modes and
    ``g_j = 1 - 4 r sin^2(j pi / (2 (cells + 1)))``, ``r = kappa dt / h^2``,
    for ``j = 1..cells``. In 2D (``cells x cells`` interior nodes) the factor
    of mode ``(j, k)`` is ``1 - 4 r (sin^2_j + sin^2_k)``, listed row by row
    (``k`` slowest).

    Args:
        diffusivity: ``kappa > 0``.
        spacing: ``h > 0``.
        time_step: ``dt > 0``.
        cells: Interior nodes per direction (``>= 1``).
        dimension: 1 or 2.

    Returns:
        The factors; the last of the 1D list is the highest-frequency mode.

    Raises:
        TypeError: On wrongly typed arguments.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-PDE2-009].

    """
    kappa = _positive("diffusivity", diffusivity)
    h = _positive("spacing", spacing)
    dt = _positive("time_step", time_step)
    n = _int_at_least("cells", cells, 1)
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    r = kappa * dt / (h * h)
    s2 = [math.sin(j * math.pi / (2.0 * (n + 1))) ** 2 for j in range(1, n + 1)]
    if dimension == 1:
        return tuple(1.0 - 4.0 * r * s for s in s2)
    return tuple(1.0 - 4.0 * r * (a + b) for b in s2 for a in s2)


def explicit_stability(
    diffusivity: float,
    spacing: float,
    time_step: float,
    cells: int,
    *,
    dimension: int = 1,
) -> ExplicitStability:
    """Return the stability report of an explicit Euler step.

    ``stable`` is the classical, grid-independent criterion
    ``dt <= h^2 / (2 d kappa)``. It is conservative for a finite grid: the
    largest discrete eigenvalue stays below ``4 kappa d / h^2``, so a step a
    hair above the limit can still have ``max |g| <= 1``; ``max_amplification``
    reports the exact value.

    Args:
        diffusivity: ``kappa > 0``.
        spacing: ``h > 0``.
        time_step: ``dt > 0``.
        cells: Interior nodes per direction.
        dimension: 1 or 2.

    Returns:
        :class:`ExplicitStability`.

    Raises:
        TypeError: On wrongly typed arguments.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-PDE2-008], [EQ-INV-PDE2-009].

    """
    factors = explicit_amplification_factors(
        diffusivity, spacing, time_step, cells, dimension=dimension
    )
    limit = explicit_stability_limit(diffusivity, spacing, dimension=dimension)
    magnitudes = [abs(g) for g in factors]
    worst = max(range(len(magnitudes)), key=magnitudes.__getitem__)
    return ExplicitStability(
        dimension=dimension,
        time_step=float(time_step),
        limit=limit,
        ratio=float(time_step) / limit,
        stable=float(time_step) <= limit * (1.0 + 1e-12),
        factors=factors,
        max_amplification=magnitudes[worst],
        worst_mode=worst,
    )


class HeatExplicitEuler1D(OperatorBase):
    """Matrix-free map from the initial state to explicit-Euler observations.

    ``u_t = kappa u_ss`` on ``(0, L)`` with zero Dirichlet values, ``n``
    interior nodes (``h = L / (n + 1)``) and the explicit step
    ``u^(k+1) = u^k + r (u^k_(j-1) - 2 u^k_j + u^k_(j+1))``,
    ``r = kappa dt / h^2``, i.e. ``u^(k+1) = T u^k`` with the symmetric matrix
    ``T = I - dt D`` [EQ-INV-PDE2-008]. The observations are the nodal values
    ``u^k_j`` at the chosen steps ``k`` and nodes ``j``. ``rmatvec`` is the
    EXACT transpose of the discrete map: a backward sweep with ``T'``
    accumulating ``sum_k (T')^k P_k' y_k`` [EQ-INV-PDE2-010].

    Construction REFUSES a step above the stability limit
    (:class:`UnstableTimeStepError`) unless ``allow_unstable=True``; then
    ``stable`` is ``False`` and ``growth_bound`` reports ``max|g|^N``
    [EQ-INV-PDE2-011]. Overflow of an unstable run is not hidden: the
    inherited ``matvec`` rejects a non-finite result with ``ValueError``.

    Attributes:
        cells: Interior nodes ``n``.
        length: Domain length.
        diffusivity: ``kappa``.
        time_step: ``dt = final_time / steps``.
        steps: Number of steps ``N``.
        spacing: ``h``.
        stability: The :class:`ExplicitStability` report.
        stable: ``stability.stable``.
        growth_bound: ``max_j |g_j|^N``, the largest per-mode amplification
            of the whole run.

    References:
        [EQ-INV-PDE2-008], [EQ-INV-PDE2-010], [EQ-INV-PDE2-011].

    """

    __slots__ = (
        "_nodes",
        "_steps_observed",
        "cells",
        "diffusivity",
        "growth_bound",
        "length",
        "spacing",
        "stability",
        "stable",
        "steps",
        "time_step",
    )

    def __init__(
        self,
        cells: int,
        *,
        length: float = 1.0,
        diffusivity: float = 1.0,
        final_time: float,
        steps: int,
        observation_steps: Sequence[int] | None = None,
        observation_nodes: Sequence[int] | None = None,
        allow_unstable: bool = False,
    ) -> None:
        """Validate the configuration and the stability condition.

        Args:
            cells: Interior nodes ``n >= 2``.
            length: Domain length ``L > 0``.
            diffusivity: ``kappa > 0``.
            final_time: ``T > 0``.
            steps: Number of steps ``N >= 1``.
            observation_steps: Steps in ``1..N`` to observe (default: the last).
            observation_nodes: Node indices in ``0..n-1`` to observe (default:
                all).
            allow_unstable: Permit ``dt`` above the limit (flagged, see class).

        Raises:
            TypeError: On wrongly typed arguments.
            ValueError: If an argument is out of range.
            UnstableTimeStepError: If ``dt > h^2 / (2 kappa)`` and
                ``allow_unstable`` is false.

        References:
            [EQ-INV-PDE2-008], [EQ-INV-PDE2-011].

        """
        n = _int_at_least("cells", cells, 2)
        big_n = _int_at_least("steps", steps, 1)
        big_l = _positive("length", length)
        kappa = _positive("diffusivity", diffusivity)
        horizon = _positive("final_time", final_time)
        if not isinstance(allow_unstable, bool):
            raise TypeError("allow_unstable must be a bool")
        spacing = big_l / (n + 1)
        dt = horizon / big_n
        report = explicit_stability(kappa, spacing, dt, n, dimension=1)
        if not report.stable and not allow_unstable:
            raise UnstableTimeStepError(
                f"explicit Euler step dt = {dt:.6g} exceeds the stability limit "
                f"h^2/(2 kappa) = {report.limit:.6g} (ratio {report.ratio:.4g}); "
                "use more steps, an implicit scheme, or allow_unstable=True"
            )
        picks = [big_n] if observation_steps is None else list(observation_steps)
        if not picks:
            raise ValueError("observation_steps must not be empty")
        for k in picks:
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= big_n:
                raise ValueError("observation_steps must be integers in 1..steps")
        nodes = list(range(n)) if observation_nodes is None else list(observation_nodes)
        if not nodes:
            raise ValueError("observation_nodes must not be empty")
        for j in nodes:
            if isinstance(j, bool) or not isinstance(j, int) or not 0 <= j < n:
                raise ValueError("observation_nodes must be integers in 0..cells-1")
        super().__init__(len(set(picks)) * len(nodes), n, name="HeatExplicitEuler1D")
        self.cells = n
        self.length = big_l
        self.diffusivity = kappa
        self.spacing = spacing
        self.steps = big_n
        self.time_step = dt
        self._steps_observed = tuple(sorted(set(picks)))
        self._nodes = tuple(nodes)
        self.stability = report
        self.stable = report.stable
        try:
            self.growth_bound = report.max_amplification**big_n
        except OverflowError:
            self.growth_bound = math.inf

    def _step(self, u: Sequence[float]) -> list[float]:
        """Apply ``T = I - dt D`` (symmetric, so also ``T'``).

        Args:
            u: Interior nodal values.

        Returns:
            The next time level.

        References:
            [EQ-INV-PDE2-008].

        """
        r = self.diffusivity * self.time_step / (self.spacing * self.spacing)
        n = self.cells
        out = [0.0] * n
        for j in range(n):
            left = u[j - 1] if j > 0 else 0.0
            right = u[j + 1] if j < n - 1 else 0.0
            out[j] = u[j] + r * (left - 2.0 * u[j] + right)
        return out

    def final_state(self, u0: Sequence[float]) -> tuple[float, ...]:
        """Return ``T^N u0``, the state after all steps (no observation).

        Args:
            u0: Initial interior nodal values (length ``cells``).

        Returns:
            The final state (may overflow for a flagged unstable run).

        Raises:
            TypeError: If ``u0`` is not a sequence of reals.
            ValueError: If it has the wrong length or is not finite.

        References:
            [EQ-INV-PDE2-008].

        """
        u = _vector("u0", u0, self.cells)
        for _ in range(self.steps):
            u = self._step(u)
        return tuple(u)

    def _forward(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply ``A`` by forward stepping.

        Args:
            vector: Initial interior state.

        Returns:
            The observations (observed steps in ascending order, nodes in the
            given order within each step).

        References:
            [EQ-INV-PDE2-008].

        """
        u = list(vector)
        wanted = set(self._steps_observed)
        out: list[float] = []
        for k in range(1, self.steps + 1):
            u = self._step(u)
            if k in wanted:
                out.extend(u[j] for j in self._nodes)
        return out

    def _transpose(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply ``A'`` by the backward sweep ``mu <- P_k' y_k + T' mu``.

        Args:
            vector: Observation-space vector.

        Returns:
            ``A' y`` at the interior nodes.

        References:
            [EQ-INV-PDE2-010].

        """
        per = len(self._nodes)
        blocks = {
            k: vector[b * per : (b + 1) * per]
            for b, k in enumerate(self._steps_observed)
        }
        mu = [0.0] * self.cells
        for k in range(self.steps, 0, -1):
            if k in blocks:
                for value, j in zip(blocks[k], self._nodes, strict=True):
                    mu[j] += value
            mu = self._step(mu)
        return mu


def explicit_mode_growth(operator: HeatExplicitEuler1D, mode: int) -> float:
    """Measure the growth of one sine mode over the whole run.

    Args:
        operator: The explicit operator.
        mode: Mode number ``j`` in ``1..cells``.

    Returns:
        ``||T^N s_j|| / ||s_j||`` for ``s_j(node m) = sin(j pi (m + 1) /
        (cells + 1))``, which equals ``|g_j|^N`` up to roundoff.

    Raises:
        TypeError: If ``mode`` is not an integer.
        ValueError: If ``mode`` is out of range.

    References:
        [EQ-INV-PDE2-009].

    """
    j = _int_at_least("mode", mode, 1)
    n = operator.cells
    if j > n:
        raise ValueError("mode must not exceed cells")
    s = [math.sin(j * math.pi * (m + 1) / (n + 1)) for m in range(n)]
    return _norm(operator.final_state(s)) / _norm(s)


def explicit_heat_inverse_problem(
    operator: HeatExplicitEuler1D,
    data: Sequence[float],
    *,
    allow_unstable: bool = False,
    **options: Any,
) -> LinearInverseProblem:
    """Build a linear inverse problem on an explicit-Euler forward map.

    Policy: refuse by default. An operator that was itself created with
    ``allow_unstable=True`` is refused again here unless ``allow_unstable``
    is also passed to this function, so an unstable forward step cannot
    reach an inversion by accident.

    Args:
        operator: The explicit-Euler operator.
        data: Observations (length ``shape[0]``).
        allow_unstable: Accept an operator whose step is above the limit.
        **options: Forwarded to
            :meth:`probstat.inverse.linear_problem.LinearInverseProblem.create`
            (``noise``, ``penalty``, ``reference``).

    Returns:
        The validated :class:`LinearInverseProblem`.

    Raises:
        TypeError: On wrongly typed arguments.
        UnstableTimeStepError: If the operator is unstable and
            ``allow_unstable`` is false.
        ValueError: From the problem validation.

    References:
        [EQ-INV-PDE2-011].

    """
    if not isinstance(operator, HeatExplicitEuler1D):
        raise TypeError("operator must be a HeatExplicitEuler1D")
    if not isinstance(allow_unstable, bool):
        raise TypeError("allow_unstable must be a bool")
    if not operator.stable and not allow_unstable:
        raise UnstableTimeStepError(
            f"refusing an inverse problem on an unstable explicit step "
            f"(dt/limit = {operator.stability.ratio:.4g}, growth bound "
            f"{operator.growth_bound:.3e}); pass allow_unstable=True to override"
        )
    return LinearInverseProblem.create(operator, data, **options)
