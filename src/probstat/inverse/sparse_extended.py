"""Extended sparse and total-variation reconstruction: IRLS, Huber-TV, TGV, MCP/SCAD.

Four additions to :mod:`probstat.inverse.sparse`, all for the data term
``1/2 ||A x - y||^2``. In every penalty ``alpha`` (or ``lam``) multiplies an
UNSQUARED functional, unlike the Tikhonov ``lambda^2``:

- :func:`irls_regularized_least_squares` -- iteratively reweighted norm
  (IRN/IRLS) for the analysis penalties ``alpha ||L x||_1`` (``L = I`` gives
  the lasso, ``L = D`` anisotropic total variation). The absolute value is
  replaced by ``sqrt(t^2 + eps^2)`` with an EXPLICIT ``eps > 0``; each outer
  step is a majorise-minimise step that solves the weighted stacked least
  squares ``[A; sqrt(alpha W) L]`` by CGLS [EQ-INV-SPX-001], [EQ-INV-SPX-002].
  The result is the minimiser of the SMOOTHED problem, within
  ``alpha p eps`` of the true objective's optimum, not an exact L1/TV
  solution [EQ-INV-SPX-003].
- :func:`huber_total_variation_least_squares` -- Huber-smoothed total
  variation solved by the Chambolle-Pock method, whose dual prox is the
  shrink-then-project map of the Huber conjugate [EQ-INV-SPX-004],
  [EQ-INV-SPX-005]. The objective tends to the TV objective as
  ``delta -> 0`` and, for large ``delta``, the problem becomes the quadratic
  ``alpha/(2 delta) ||D x||^2`` (Tikhonov) problem [EQ-INV-SPX-006].
- :func:`tgv2_least_squares_1d` -- second-order total generalized variation
  in one dimension, ``min_(x, v) 1/2||A x - y||^2 + alpha1 ||D x - v||_1 +
  alpha0 ||E v||_1``, by primal-dual iterations with steps satisfying
  ``tau sigma L^2 < 1`` for a CERTIFIED (not estimated) bound ``L`` of the
  operator norm [EQ-INV-SPX-007], [EQ-INV-SPX-008]. TGV favours
  piecewise-affine reconstructions where TV produces staircases.
- :func:`tgv2_least_squares_2d` -- the same TGV^2 model on a ``rows x columns``
  grid with a vector field ``v = (v_1, v_2)``, ``K z = (A x, M (G x - v),
  E v)``, ``G`` the Neumann forward-difference gradient, ``E`` the
  symmetrised gradient of ``v`` and ``M`` a mask that drops the zero-padded
  boundary differences of ``G`` so affine images have exactly zero penalty;
  steps use the PROVEN bound ``||K||^2 <= ||A||_F^2 + 2 (4/h_x^2 + 4/h_y^2) +
  1`` [EQ-INV-SPX-012], [EQ-INV-SPX-013].
- :func:`prox_mcp`, :func:`prox_scad` and
  :func:`nonconvex_sparse_least_squares` -- the exact scalar proximal maps of
  the non-convex minimax concave and SCAD penalties and proximal gradient with
  them [EQ-INV-SPX-009] to [EQ-INV-SPX-011]. The proximal gradient method
  finds a STATIONARY point only; the result depends on the start point and is
  not a global minimiser.

Two cross-cutting additions. :func:`irls_regularized_least_squares` also
solves the ISOTROPIC total variation on a 2D grid (``penalty="tv2d"`` or the
``components`` argument): the weight of a pixel comes from the gradient
magnitude, ``w_i = 1 / sqrt(|(G x)_i|^2 + eps^2)``, with the objective-gap
bound ``alpha p eps`` [EQ-INV-SPX-014]. IRLS, Huber-TV, TGV^2 (1D and 2D) and
the non-convex solver accept ``noise=``, a covariance factor of
:mod:`probstat.linalg`; the data term is then ``1/2 ||W (A x - y)||^2`` with
``W`` the whitening operator and ``noise=None`` reproduces the unwhitened
results exactly [EQ-INV-SPX-015].

Experimental (added in 0.28): the API may change before it is declared stable.

References:
    [REF-RODRIGUEZ-WOHLBERG-2009], [REF-DAUBECHIES-DEVORE-FORNASIER-GUNTURK-2010],
    [REF-HUBER-1964], [REF-BREDIES-KUNISCH-POCK-2010], [REF-ZHANG-2010],
    [REF-FAN-LI-2001], [REF-CHAMBOLLE-POCK-2011], [REF-BECK-TEBOULLE-2009].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.inverse.sparse import (
    _default_components,
    _gradient_bound,
    _nonnegative,
    _positive,
    _real,
    _vector,
    frobenius_norm_bound,
    total_variation_value,
)
from probstat.linalg import (
    CallbackOperator,
    ComposedOperator,
    CountingOperator,
    DiagonalOperator,
    FirstDifferenceOperator,
    GridGradientOperator,
    IdentityOperator,
    LinearOperator,
    OperatorCounts,
    Vector,
    VStackOperator,
    cgls,
    is_covariance_factor,
    require_operator,
)
from probstat.optimization.proximal import (
    PDHGResult,
    ProximalGradientResult,
    ista,
    primal_dual_hybrid_gradient,
    project_linf_ball,
    project_pointwise_l2_ball,
)


def _norm2(vector: Sequence[float]) -> float:
    """Return the Euclidean norm of a vector.

    Args:
        vector: Entries.

    Returns:
        ``sqrt(sum v_i^2)``.

    References:
        [EQ-INV-SPX-001].

    """
    return math.hypot(*vector) if vector else 0.0


def _half_misfit(operator: LinearOperator, data: Vector, point: Vector) -> float:
    """Return ``1/2 ||A x - y||^2``.

    Args:
        operator: ``A``.
        data: ``y``.
        point: ``x``.

    Returns:
        The data misfit.

    References:
        [EQ-INV-SPX-001].

    """
    image = operator.matvec(point)
    return 0.5 * math.fsum((a - b) ** 2 for a, b in zip(image, data, strict=True))


def _whitened_problem(
    operator: LinearOperator, data: Vector, noise: object
) -> tuple[LinearOperator, Vector]:
    """Return ``(W A, W y)`` for a covariance factor, or the inputs unchanged.

    With ``noise=None`` the SAME objects are returned, so the unwhitened code
    paths are untouched. Otherwise ``W = C^-1`` is the whitening operator of
    the covariance factor and the data term ``1/2 ||W (A x - y)||^2`` becomes
    the plain misfit of the pair ``(W A, W y)``.

    Args:
        operator: ``A`` (``m x n``).
        data: ``y`` of length ``m``.
        noise: ``None`` or a covariance factor of dimension ``m``.

    Returns:
        ``(A, y)`` for ``None``, else ``(W A, W y)``.

    Raises:
        TypeError: If ``noise`` is neither ``None`` nor a covariance factor.
        ValueError: If the covariance dimension differs from ``m``.

    References:
        [EQ-INV-SPX-015].

    """
    if noise is None:
        return operator, data
    if not is_covariance_factor(noise):
        raise TypeError("noise must be None or a covariance factor")
    if noise.dimension != operator.shape[0]:  # type: ignore[attr-defined]
        raise ValueError("noise dimension must equal the number of data")
    whitening = noise.whitening_operator()  # type: ignore[attr-defined]
    return (
        ComposedOperator([whitening, operator]),
        tuple(noise.whiten(data)),  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# 1. Iteratively reweighted norm
# ---------------------------------------------------------------------------


def _grid_shape(grid: tuple[int, int]) -> tuple[int, int]:
    """Validate ``(rows, columns)`` of an image grid.

    Args:
        grid: Candidate pair.

    Returns:
        ``(rows, columns)``.

    Raises:
        TypeError: If an entry is not an integer.
        ValueError: If the pair does not have two entries or an entry is
            smaller than two.

    References:
        [EQ-INV-SPX-012].

    """
    entries = tuple(grid)
    if len(entries) != 2:
        raise ValueError("grid must be (rows, columns)")
    for value in entries:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("grid entries must be integers")
        if value < 2:
            raise ValueError("grid needs at least two rows and two columns")
    return entries[0], entries[1]


def _point_norms(values: Sequence[float], components: int) -> list[float]:
    """Return the Euclidean norms of block-stacked gradient points.

    Args:
        values: ``[g^1; ...; g^c]`` with ``c = components`` blocks of ``N``.
        components: Number of blocks ``c``.

    Returns:
        ``[|g_i|]_i`` with ``|g_i| = sqrt(sum_k (g^k_i)^2)``.

    References:
        [EQ-INV-SPX-014].

    """
    count = len(values) // components
    return [
        math.hypot(*(values[k * count + i] for k in range(components)))
        for i in range(count)
    ]


@dataclass(frozen=True, slots=True)
class IRLSResult:
    """Result of :func:`irls_regularized_least_squares`.

    Attributes:
        solution: Final iterate ``x`` (the minimiser of the smoothed problem
            when ``converged``).
        objective: TRUE objective ``1/2||A x - y||^2 + alpha ||L x||_1``.
        smoothed_objective: ``1/2||A x - y||^2 + alpha sum sqrt((L x)_i^2 +
            eps^2)``, the function the iteration decreases.
        data_misfit: ``1/2||A x - y||^2``.
        penalty: ``alpha ||L x||_1``.
        alpha: Penalty coefficient (an unsquared multiplier).
        epsilon: Smoothing parameter ``eps``.
        penalty_name: ``"l1"``, ``"tv"`` or ``"custom"``.
        outer_iterations: Completed reweighting steps.
        inner_iterations: Total CGLS iterations over all outer steps.
        inner_iteration_history: CGLS iterations of each outer step.
        inner_failures: Number of weighted solves that missed their tolerance.
        objective_history: Smoothed objective at the start and after every
            outer step.
        true_objective_history: True objective at the same points.
        max_objective_increase: Largest rise of the smoothed objective between
            two consecutive outer steps (``0`` when monotone; positive only
            through inexact inner solves).
        operator_counts: Applications of ``A`` (including objective
            evaluations).
        penalty_counts: Applications of ``L``.
        converged: ``True`` only when the step tolerance was met and no inner
            solve failed.
        stop_reason: ``"tolerance"``, ``"max_iterations"``,
            ``"inner_solver_failed"`` or ``"non_finite"``.
        components: Gradient blocks per penalty point (1: entrywise L1 norm;
            2: isotropic grid TV).
        gap_bound: ``alpha p eps`` with ``p`` the number of penalty points:
            at the exact minimiser of the smoothed problem the true objective
            exceeds its minimum by at most this amount [EQ-INV-SPX-003],
            [EQ-INV-SPX-014].

    References:
        [EQ-INV-SPX-001], [EQ-INV-SPX-002], [EQ-INV-SPX-003],
        [EQ-INV-SPX-014].

    """

    solution: Vector
    objective: float
    smoothed_objective: float
    data_misfit: float
    penalty: float
    alpha: float
    epsilon: float
    penalty_name: str
    outer_iterations: int
    inner_iterations: int
    inner_iteration_history: tuple[int, ...]
    inner_failures: int
    objective_history: Vector
    true_objective_history: Vector
    max_objective_increase: float
    operator_counts: OperatorCounts
    penalty_counts: OperatorCounts
    converged: bool
    stop_reason: str
    components: int = 1
    gap_bound: float = math.nan

    @property
    def estimate(self) -> Vector:
        """Return ``solution`` (the name used by the other inverse results).

        References:
            [EQ-INV-LIN-001].

        """
        return self.solution


def irls_regularized_least_squares(
    operator: LinearOperator,
    data: Sequence[float],
    alpha: float,
    *,
    penalty: str = "l1",
    penalty_operator: LinearOperator | None = None,
    epsilon: float = 1e-3,
    initial: Sequence[float] | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
    inner_tolerance: float = 1e-10,
    inner_max_iterations: int | None = None,
    noise: object = None,
    grid: tuple[int, int] | None = None,
    spacing: tuple[float, float] = (1.0, 1.0),
    components: int = 1,
) -> IRLSResult:
    """Minimise ``1/2||A x - y||^2 + alpha sum_i sqrt((L x)_i^2 + eps^2)``.

    With ``t = L x`` the smoothed absolute value ``sqrt(t^2 + eps^2)`` is
    majorised at ``t_0`` by ``1/2 w_0 (t^2 + eps^2) + 1/(2 w_0)`` with
    ``w_0 = 1 / sqrt(t_0^2 + eps^2)``. Each outer step therefore minimises
    ``1/2||A x - y||^2 + alpha/2 sum w_i (L x)_i^2``, i.e. the stacked
    least-squares problem ``min ||[A; sqrt(alpha W) L] x - [y; 0]||`` by
    CGLS warm-started at the current iterate [EQ-INV-SPX-001]. With exact
    inner solves the smoothed objective is non-increasing
    [EQ-INV-SPX-002]; ``max_objective_increase`` reports any rise caused by
    the finite inner tolerance. ``eps`` is explicit and fixed (no
    continuation): at the optimum the TRUE objective ``F`` satisfies
    ``F(x_eps) <= min F + alpha p eps`` with ``p`` the number of penalty rows
    [EQ-INV-SPX-003], and the smoothed minimiser is not sparse in the exact
    sense (small entries are of order ``eps``). No claim is made about an
    exact L0 or L1 solution. ``penalty="l1"`` uses ``L = I`` (lasso),
    ``"tv"`` the 1D forward difference (anisotropic total variation); any
    other ``L`` is passed through ``penalty_operator``.

    ISOTROPIC grid TV. ``penalty="tv2d"`` (with ``grid=(rows, columns)``)
    uses the Neumann forward-difference
    :class:`~probstat.linalg.GridGradientOperator` and the pointwise norm
    ``|g_i| = sqrt(g_i1^2 + g_i2^2)``; the same pointwise grouping is
    available for any block-stacked ``L`` through ``components = c > 1``. The
    smoothed penalty is ``alpha sum_i sqrt(|g_i|^2 + eps^2)``, majorised at
    ``g_0`` by ``alpha/2 sum_i w_i (|g_i|^2 + eps^2) + alpha/(2 w_i)`` with
    ONE weight per pixel ``w_i = 1 / sqrt(|g_0,i|^2 + eps^2)`` (the map
    ``s -> sqrt(s + eps^2)`` is concave, so its tangent majorises it), and
    the gap bound is the same ``alpha p eps`` with ``p`` the number of pixels
    [EQ-INV-SPX-014]; the zero-padded last row and column contribute the
    constant ``alpha eps`` per pixel to the smoothed objective. The true
    objective can be compared with
    :func:`~probstat.inverse.total_variation_regularized_least_squares`
    up to this bound.

    ``noise`` (a covariance factor) replaces the data term by
    ``1/2 ||W (A x - y)||^2`` [EQ-INV-SPX-015]; ``None`` is exactly the
    unwhitened solver.

    Args:
        operator: ``A`` (``m x n``) or a dense matrix.
        data: ``y`` of length ``m``.
        alpha: Positive penalty coefficient.
        penalty: ``"l1"`` or ``"tv"``; ignored when ``penalty_operator`` is
            given.
        penalty_operator: Explicit ``L`` acting on the unknowns.
        epsilon: Positive smoothing parameter ``eps``.
        initial: Start ``x_0`` (zero by default, which gives a ridge first
            step with ``lambda^2 = alpha / eps``).
        tolerance: Relative step tolerance ``||x+ - x|| <= tol (1 + ||x+||)``.
        max_iterations: Outer iteration cap.
        inner_tolerance: Relative normal-residual tolerance of each CGLS
            solve (relative to the warm start).
        inner_max_iterations: CGLS cap (default ``10 n``: the weights
            ``1/eps`` make the stacked system ill conditioned).
        noise: ``None`` or a covariance factor of dimension ``m``; the data
            term becomes ``1/2 ||W (A x - y)||^2``.
        grid: ``(rows, columns)`` of the image (required by, and only allowed
            for, ``penalty="tv2d"``; ``rows * columns`` must equal ``n``).
        spacing: ``(h_y, h_x)`` of the ``"tv2d"`` grid.
        components: Gradient blocks per penalty point for a custom
            ``penalty_operator`` (1 keeps the entrywise L1 norm); must be 1
            for ``"l1"`` and ``"tv"``; ``"tv2d"`` sets it to 2.

    Returns:
        :class:`IRLSResult` with outer and inner iteration counts, the
        objective histories and the stop reason.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``alpha`` or ``epsilon`` is not positive and finite,
            ``penalty`` is unknown, a dimension is inconsistent, ``data`` or
            ``initial`` is not finite, a tolerance or cap is invalid,
            ``grid`` is missing for ``"tv2d"`` or given otherwise, or
            ``components`` does not divide the penalty rows.

    References:
        [EQ-INV-SPX-001], [EQ-INV-SPX-002], [EQ-INV-SPX-003],
        [EQ-INV-SPX-014], [EQ-INV-SPX-015], [REF-RODRIGUEZ-WOHLBERG-2009],
        [REF-DAUBECHIES-DEVORE-FORNASIER-GUNTURK-2010].

    """
    raw = require_operator(operator, name="operator")
    m, n = raw.shape
    y = _vector(data, m, "data")
    raw, y = _whitened_problem(raw, y, noise)
    scale = _positive(alpha, "alpha")
    eps = _positive(epsilon, "epsilon")
    tol = _nonnegative(tolerance, "tolerance")
    inner_tol = _positive(inner_tolerance, "inner_tolerance")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if isinstance(components, bool) or not isinstance(components, int):
        raise TypeError("components must be an integer")
    if components < 1:
        raise ValueError("components must be positive")
    if grid is not None and (penalty != "tv2d" or penalty_operator is not None):
        raise ValueError("grid is only used by penalty='tv2d'")
    block = components
    if penalty_operator is not None:
        l_raw = require_operator(penalty_operator, name="penalty_operator")
        name = "custom"
    elif penalty == "l1":
        l_raw = IdentityOperator(n)
        name = "l1"
    elif penalty == "tv":
        l_raw = FirstDifferenceOperator(n)
        name = "tv"
    elif penalty == "tv2d":
        if grid is None:
            raise ValueError("penalty='tv2d' requires grid=(rows, columns)")
        grid_rows, grid_columns = _grid_shape(grid)
        if grid_rows * grid_columns != n:
            raise ValueError("grid must satisfy rows * columns == number of unknowns")
        l_raw = GridGradientOperator(grid_rows, grid_columns, spacing=spacing)
        name = "tv2d"
        block = 2
    else:
        raise ValueError("penalty must be 'l1', 'tv' or 'tv2d'")
    if name in ("l1", "tv") and components != 1:
        raise ValueError("components applies to a custom penalty_operator only")
    if l_raw.shape[1] != n:
        raise ValueError("penalty_operator must act on the same unknowns")
    rows = l_raw.shape[0]
    if rows % block != 0:
        raise ValueError("penalty rows must be a multiple of components")
    points = rows // block
    x = (0.0,) * n if initial is None else _vector(initial, n, "initial")

    a_op = CountingOperator(raw)
    l_op = CountingOperator(l_raw)
    zeros = (0.0,) * rows
    rhs = y + zeros

    def objectives(point: Vector) -> tuple[float, float, float]:
        """Return ``(smoothed, true, misfit)`` at a point.

        References:
            [EQ-INV-SPX-001].

        """
        misfit = _half_misfit(a_op, y, point)
        t = l_op.matvec(point)
        if block == 1:
            smooth = misfit + scale * math.fsum(math.hypot(v, eps) for v in t)
            exact = misfit + scale * math.fsum(abs(v) for v in t)
            return smooth, exact, misfit
        sizes = _point_norms(t, block)
        smooth = misfit + scale * math.fsum(math.hypot(v, eps) for v in sizes)
        exact = misfit + scale * math.fsum(sizes)
        return smooth, exact, misfit

    smooth0, true0, _ = objectives(x)
    if not math.isfinite(smooth0):
        raise ValueError("the objective is not finite at the initial point")
    history = [smooth0]
    true_history = [true0]
    inner_hist: list[int] = []
    inner_cap = 10 * n if inner_max_iterations is None else inner_max_iterations
    inner_failures = 0
    biggest_rise = 0.0
    stop = "max_iterations"
    for _ in range(max_iterations):
        t = l_op.matvec(x)
        if block == 1:
            sqrt_weights = tuple(math.sqrt(scale / math.hypot(v, eps)) for v in t)
        else:
            per_point = tuple(
                math.sqrt(scale / math.hypot(v, eps)) for v in _point_norms(t, block)
            )
            sqrt_weights = per_point * block
        stacked = VStackOperator(
            [a_op, ComposedOperator([DiagonalOperator(sqrt_weights), l_op])]
        )
        solve = cgls(
            stacked,
            rhs,
            initial=x,
            residual_tolerance=0.0,
            normal_tolerance=inner_tol,
            max_iterations=inner_cap,
        )
        inner_hist.append(solve.iterations)
        if not solve.converged:
            inner_failures += 1
        new = tuple(solve.solution)
        if not all(math.isfinite(v) for v in new):
            stop = "non_finite"
            break
        smooth, exact, _ = objectives(new)
        if not math.isfinite(smooth):
            stop = "non_finite"
            break
        biggest_rise = max(biggest_rise, smooth - history[-1])
        history.append(smooth)
        true_history.append(exact)
        step = _norm2(tuple(a - b for a, b in zip(new, x, strict=True)))
        x = new
        if step <= tol * (1.0 + _norm2(x)):
            stop = "tolerance"
            break
    outer = len(inner_hist) if stop != "non_finite" else len(history) - 1
    smooth_f, true_f, misfit_f = objectives(x)
    converged = stop == "tolerance" and inner_failures == 0
    if stop == "tolerance" and inner_failures > 0:
        stop = "inner_solver_failed"
    return IRLSResult(
        solution=x,
        objective=true_f,
        smoothed_objective=smooth_f,
        data_misfit=misfit_f,
        penalty=true_f - misfit_f,
        alpha=scale,
        epsilon=eps,
        penalty_name=name,
        outer_iterations=outer,
        inner_iterations=sum(inner_hist),
        inner_iteration_history=tuple(inner_hist),
        inner_failures=inner_failures,
        objective_history=tuple(history),
        true_objective_history=tuple(true_history),
        max_objective_increase=biggest_rise,
        operator_counts=a_op.counts,
        penalty_counts=l_op.counts,
        converged=converged,
        stop_reason=stop,
        components=block,
        gap_bound=scale * points * eps,
    )


# ---------------------------------------------------------------------------
# 2. Huber total variation
# ---------------------------------------------------------------------------


def _huber_points(
    values: Sequence[float], delta: float, components: int, isotropic: bool
) -> float:
    """Return ``sum_i huber_delta(|g_i|)`` for a block-stacked gradient.

    Args:
        values: Block-stacked gradient entries.
        delta: Huber parameter.
        components: Blocks per point.
        isotropic: Euclidean norm per point (``True``) or entrywise.

    Returns:
        The summed Huber function.

    References:
        [EQ-INV-SPX-004].

    """

    def huber(s: float) -> float:
        """Return ``s^2 / (2 delta)`` or ``s - delta/2``.

        References:
            [EQ-INV-SPX-004].

        """
        return s * s / (2.0 * delta) if s <= delta else s - 0.5 * delta

    count = len(values) // components
    if not isotropic or components == 1:
        return math.fsum(huber(abs(v)) for v in values)
    return math.fsum(
        huber(math.hypot(*(values[k * count + i] for k in range(components))))
        for i in range(count)
    )


def huber_total_variation_value(
    values: Sequence[float],
    delta: float,
    gradient: LinearOperator | None = None,
    *,
    isotropic: bool = True,
    components: int | None = None,
) -> float:
    """Return the Huber-smoothed total variation ``sum_i h_delta(|(D x)_i|)``.

    ``h_delta(s) = s^2/(2 delta)`` for ``s <= delta`` and ``s - delta/2``
    otherwise [EQ-INV-SPX-004]. It satisfies ``|s| - delta/2 <= h_delta(s) <=
    |s|``, so the value is within ``p delta / 2`` of the total variation.

    Args:
        values: ``x``.
        delta: Positive Huber parameter.
        gradient: ``D`` (default: 1D forward difference).
        isotropic: Euclidean (``True``) or entrywise norm of a multi-component
            gradient.
        components: Blocks per point (default 2 for a grid gradient, else 1).

    Returns:
        The summed Huber function of the gradient magnitudes.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If ``delta`` is not positive, ``x`` is not finite or the
            dimensions are inconsistent.

    References:
        [EQ-INV-SPX-004], [REF-HUBER-1964].

    """
    d = _positive(delta, "delta")
    x = tuple(values)
    op = (
        FirstDifferenceOperator(len(x))
        if gradient is None
        else require_operator(gradient, name="gradient")
    )
    vec = _vector(x, op.shape[1], "values")
    c = _default_components(op) if components is None else components
    if isinstance(c, bool) or not isinstance(c, int) or c < 1:
        raise ValueError("components must be a positive integer")
    if op.shape[0] % c != 0:
        raise ValueError("gradient rows must be a multiple of components")
    return _huber_points(op.matvec(vec), d, c, isotropic)


@dataclass(frozen=True, slots=True)
class HuberTVResult:
    """Result of :func:`huber_total_variation_least_squares`.

    Attributes:
        solution: Estimate ``x``.
        objective: ``1/2||A x - y||^2 + alpha sum h_delta(|(D x)_i|)``,
            recomputed from the solution.
        data_misfit: ``1/2||A x - y||^2``.
        huber_tv: ``sum h_delta(|(D x)_i|)`` (unscaled).
        total_variation: ``TV(x)`` of the same solution (unscaled).
        tv_objective: ``1/2||A x - y||^2 + alpha TV(x)`` at the solution.
        gradient_norm: ``||A'(A x - y) + alpha D' h'_delta(D x)||``, the
            norm of the exact gradient of the smooth objective (an optimality
            measure independent of the PDHG residuals).
        alpha: Penalty coefficient.
        delta: Huber parameter.
        optimization: The underlying :class:`~probstat.optimization.PDHGResult`.
        norm_bound: Certified bound ``L >= ||[A; D]||_2`` used for the steps.
        operator_counts: Applications of ``A``.
        gradient_counts: Applications of ``D``.
        converged: Convergence flag of PDHG.
        stop_reason: Stop reason of PDHG.

    References:
        [EQ-INV-SPX-004], [EQ-INV-SPX-005], [EQ-INV-SPX-006].

    """

    solution: Vector
    objective: float
    data_misfit: float
    huber_tv: float
    total_variation: float
    tv_objective: float
    gradient_norm: float
    alpha: float
    delta: float
    optimization: PDHGResult
    norm_bound: float
    operator_counts: OperatorCounts
    gradient_counts: OperatorCounts
    converged: bool
    stop_reason: str

    @property
    def estimate(self) -> Vector:
        """Return ``solution`` (the name used by the other inverse results).

        References:
            [EQ-INV-LIN-001].

        """
        return self.solution


def huber_total_variation_least_squares(
    operator: LinearOperator,
    data: Sequence[float],
    alpha: float,
    delta: float,
    *,
    gradient: LinearOperator | None = None,
    isotropic: bool = True,
    components: int | None = None,
    forward_norm_bound: float | None = None,
    gradient_norm_bound: float | None = None,
    initial: Sequence[float] | None = None,
    tolerance_absolute: float = 1e-9,
    tolerance_relative: float = 1e-8,
    max_iterations: int = 50000,
    step_safety: float = 0.99,
    noise: object = None,
) -> HuberTVResult:
    """Minimise ``1/2||A x - y||^2 + alpha sum_i h_delta(|(D x)_i|)``.

    The Huber function is ``h_delta(s) = max_(|q| <= 1) q s - delta q^2 / 2``,
    so with ``g(v) = alpha sum h_delta(|v_i|)`` the conjugate is
    ``g*(q) = delta/(2 alpha) |q|^2 + indicator(|q_i| <= alpha)`` and
    ``prox_(sigma g*)(p) = P_(|q| <= alpha)(p / (1 + sigma delta / alpha))``
    [EQ-INV-SPX-005]. The problem ``min_x f(x) + g(K x)`` with ``K = [A; D]``
    and ``g`` including the data term is solved by
    :func:`~probstat.optimization.primal_dual_hybrid_gradient`, so the steps
    satisfy ``tau sigma ||K||^2 < 1`` for the CERTIFIED bound
    ``||K||^2 <= ||A||_F^2 + ||D||^2`` (``forward_norm_bound`` defaults to the
    Frobenius norm; ``gradient_norm_bound`` defaults to the closed form for
    the library stencils, otherwise it is mandatory).

    As ``delta -> 0`` the objective tends to the TV objective: for the
    respective minimisers ``|F_delta(x_delta) - F_TV(x_TV)| <= alpha p delta
    / 2``. For large ``delta`` (every ``|(D x)_i| <= delta`` at the optimum)
    the penalty is ``alpha/(2 delta) ||D x||^2``, i.e. Tikhonov with
    ``lambda^2 = alpha / delta`` [EQ-INV-SPX-006]. Because the objective is
    smooth the returned ``gradient_norm`` is an independent optimality check.

    Args:
        operator: ``A`` (``m x n``) or a dense matrix.
        data: ``y`` of length ``m``.
        alpha: Positive penalty coefficient.
        delta: Positive Huber parameter.
        gradient: ``D`` (default: 1D forward difference).
        isotropic: Euclidean (``True``) or entrywise norm of the gradient.
        components: Blocks per point (default 2 for a grid gradient, else 1).
        forward_norm_bound: Certified bound on ``||A||_2``.
        gradient_norm_bound: Certified bound on ``||D||_2``.
        initial: Start point (zero by default).
        tolerance_absolute: PDHG absolute residual tolerance.
        tolerance_relative: PDHG relative residual tolerance.
        max_iterations: PDHG iteration cap.
        step_safety: PDHG step factor in ``(0, 1)``.
        noise: ``None`` or a covariance factor of dimension ``m``; the data
            term becomes ``1/2 ||W (A x - y)||^2`` [EQ-INV-SPX-015] (the
            reported misfits and ``forward_norm_bound`` refer to ``W A``).

    Returns:
        :class:`HuberTVResult`; ``converged`` is ``False`` at the cap.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``alpha`` or ``delta`` is not positive and finite, a
            dimension is inconsistent, data are not finite, or a norm bound is
            missing for an unrecognised ``D``.

    References:
        [EQ-INV-SPX-004], [EQ-INV-SPX-005], [EQ-INV-SPX-006],
        [EQ-INV-SPX-015], [REF-HUBER-1964], [REF-CHAMBOLLE-POCK-2011].

    """
    raw = require_operator(operator, name="operator")
    m, n = raw.shape
    y = _vector(data, m, "data")
    raw, y = _whitened_problem(raw, y, noise)
    scale = _positive(alpha, "alpha")
    d = _positive(delta, "delta")
    d_raw = (
        FirstDifferenceOperator(n)
        if gradient is None
        else require_operator(gradient, name="gradient")
    )
    if d_raw.shape[1] != n:
        raise ValueError("gradient must act on the same unknowns as operator")
    c = _default_components(d_raw) if components is None else components
    if isinstance(c, bool) or not isinstance(c, int) or c < 1:
        raise ValueError("components must be a positive integer")
    if d_raw.shape[0] % c != 0:
        raise ValueError("gradient rows must be a multiple of components")
    start = (0.0,) * n if initial is None else _vector(initial, n, "initial")
    fwd = (
        frobenius_norm_bound(raw)
        if forward_norm_bound is None
        else _positive(forward_norm_bound, "forward_norm_bound")
    )
    if gradient_norm_bound is None:
        known = _gradient_bound(d_raw)
        if known is None:
            raise ValueError("gradient_norm_bound is required for this operator")
        grad_bound = known
    else:
        grad_bound = _positive(gradient_norm_bound, "gradient_norm_bound")
    norm_bound = math.hypot(fwd, grad_bound)

    a_count = CountingOperator(raw)
    d_count = CountingOperator(d_raw)
    stacked = VStackOperator([a_count, d_count])
    shrink_rate = d / scale

    def prox_primal(point: Vector, step: float) -> Vector:
        """Return ``point`` (the primal term is zero).

        References:
            [EQ-INV-SPX-005].

        """
        return point

    def prox_dual(point: Vector, step: float) -> Vector:
        """Return the prox of ``step g*`` (data block and Huber block).

        References:
            [EQ-INV-SPX-005].

        """
        head = tuple(
            (a - step * b) / (1.0 + step) for a, b in zip(point[:m], y, strict=True)
        )
        factor = 1.0 / (1.0 + step * shrink_rate)
        tail = tuple(factor * v for v in point[m:])
        if isotropic and c > 1:
            return head + project_pointwise_l2_ball(tail, scale, components=c)
        return head + project_linf_ball(tail, scale)

    def objective(point: Vector, image: Vector) -> float:
        """Return the primal objective from ``K x``.

        References:
            [EQ-INV-SPX-004].

        """
        fit = 0.5 * math.fsum((a - b) ** 2 for a, b in zip(image[:m], y, strict=True))
        return fit + scale * _huber_points(image[m:], d, c, isotropic)

    optimization = primal_dual_hybrid_gradient(
        stacked,
        prox_primal,
        prox_dual,
        start,
        operator_norm_bound=norm_bound,
        step_safety=step_safety,
        objective=objective,
        tolerance_absolute=tolerance_absolute,
        tolerance_relative=tolerance_relative,
        max_iterations=max_iterations,
    )
    x = optimization.solution
    residual = tuple(a - b for a, b in zip(raw.matvec(x), y, strict=True))
    misfit = 0.5 * math.fsum(r * r for r in residual)
    g = d_raw.matvec(x)
    count = len(g) // c
    slope = [0.0] * len(g)
    for i in range(count):
        point = tuple(g[k * count + i] for k in range(c))
        if isotropic and c > 1:
            denom = max(d, math.hypot(*point))
            for k in range(c):
                slope[k * count + i] = point[k] / denom
        else:
            for k in range(c):
                slope[k * count + i] = point[k] / max(d, abs(point[k]))
    back = d_raw.rmatvec(tuple(slope))
    grad = tuple(
        a + scale * b for a, b in zip(raw.rmatvec(residual), back, strict=True)
    )
    huber_tv = _huber_points(g, d, c, isotropic)
    tv = total_variation_value(x, d_raw, isotropic=isotropic, components=c)
    return HuberTVResult(
        solution=x,
        objective=misfit + scale * huber_tv,
        data_misfit=misfit,
        huber_tv=huber_tv,
        total_variation=tv,
        tv_objective=misfit + scale * tv,
        gradient_norm=_norm2(grad),
        alpha=scale,
        delta=d,
        optimization=optimization,
        norm_bound=norm_bound,
        operator_counts=a_count.counts,
        gradient_counts=d_count.counts,
        converged=optimization.converged,
        stop_reason=optimization.stop_reason,
    )


# ---------------------------------------------------------------------------
# 3. Second-order TGV in one dimension
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TGVResult:
    """Result of :func:`tgv2_least_squares_1d`.

    Attributes:
        solution: Estimate ``x`` (length ``n``).
        auxiliary: The minimising ``v`` (length ``n - 1``), an estimate of the
            derivative of ``x``.
        objective: ``1/2||A x - y||^2 + alpha1 ||D x - v||_1 + alpha0 ||E v||_1``
            at ``(x, v)``; an UPPER bound of the TGV^2 objective at ``x``
            (equal at the joint minimiser).
        data_misfit: ``1/2||A x - y||^2``.
        first_order_term: ``alpha1 ||D x - v||_1``.
        second_order_term: ``alpha0 ||E v||_1``.
        alpha1: First-order weight.
        alpha0: Second-order weight.
        optimization: The underlying :class:`~probstat.optimization.PDHGResult`.
        norm_bound: The certified bound ``L >= ||K||_2`` used for the steps.
        norm_bound_kind: Description of how ``L`` was obtained.
        operator_counts: Applications of ``A``.
        converged: Convergence flag of PDHG.
        stop_reason: Stop reason of PDHG.

    References:
        [EQ-INV-SPX-007], [EQ-INV-SPX-008].

    """

    solution: Vector
    auxiliary: Vector
    objective: float
    data_misfit: float
    first_order_term: float
    second_order_term: float
    alpha1: float
    alpha0: float
    optimization: PDHGResult
    norm_bound: float
    norm_bound_kind: str
    operator_counts: OperatorCounts
    converged: bool
    stop_reason: str

    @property
    def estimate(self) -> Vector:
        """Return ``solution`` (the name used by the other inverse results).

        References:
            [EQ-INV-LIN-001].

        """
        return self.solution


def tgv2_least_squares_1d(
    operator: LinearOperator,
    data: Sequence[float],
    alpha1: float,
    alpha0: float,
    *,
    spacing: float = 1.0,
    forward_norm_bound: float | None = None,
    initial: Sequence[float] | None = None,
    tolerance_absolute: float = 1e-9,
    tolerance_relative: float = 1e-8,
    max_iterations: int = 50000,
    step_safety: float = 0.99,
    noise: object = None,
) -> TGVResult:
    """Minimise ``1/2||A x - y||^2 + TGV^2(x)`` on a one-dimensional grid.

    The second-order total generalized variation is
    ``TGV^2(x) = min_v alpha1 ||D x - v||_1 + alpha0 ||E v||_1`` with ``D``
    and ``E`` forward differences of spacing ``h`` (``D``: ``n - 1`` rows,
    ``E``: ``n - 2`` rows); it vanishes exactly on affine signals, so ramps
    are not penalised and are not turned into staircases [EQ-INV-SPX-007].
    The joint problem in ``z = (x, v)`` is ``min g(K z)`` with
    ``K z = (A x, D x - v, E v)`` and ``g`` the sum of the data term and two
    L1 norms, solved by
    :func:`~probstat.optimization.primal_dual_hybrid_gradient`; the dual prox
    is the data shrink and two clippings [EQ-INV-SPX-008].

    Step sizes. The iteration needs ``tau sigma ||K||^2 < 1``; the code uses
    the PROVEN bound ``||K||^2 <= ||A||_F^2 + (||D||^2 + 1) + ||E||^2`` with
    ``||D||, ||E|| <= 2/h`` (from stacking and ``[D, -I][D, -I]' = D D' +
    I``), not a power-iteration estimate (which is only a lower bound). The
    bound is conservative, so convergence can be slow.

    Args:
        operator: ``A`` (``m x n``, ``n >= 3``) or a dense matrix.
        data: ``y`` of length ``m``.
        alpha1: Positive weight of ``||D x - v||_1``.
        alpha0: Positive weight of ``||E v||_1``.
        spacing: Positive grid spacing ``h``.
        forward_norm_bound: Certified bound on ``||A||_2`` (default the
            Frobenius norm, which needs a materialisable operator).
        initial: Start ``x_0`` (zero by default); ``v_0 = D x_0``.
        tolerance_absolute: PDHG absolute residual tolerance.
        tolerance_relative: PDHG relative residual tolerance.
        max_iterations: PDHG iteration cap.
        step_safety: PDHG step factor in ``(0, 1)``.
        noise: ``None`` or a covariance factor of dimension ``m``; the data
            term becomes ``1/2 ||W (A x - y)||^2`` [EQ-INV-SPX-015] (the
            misfit and ``forward_norm_bound`` refer to ``W A``).

    Returns:
        :class:`TGVResult`; ``converged`` is ``False`` at the cap.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a weight or the spacing is not positive and finite,
            ``n < 3``, data are not finite, a dimension is inconsistent, or a
            step size violates ``tau sigma L^2 < 1`` (``step_safety`` outside
            ``(0, 1)``).

    References:
        [EQ-INV-SPX-007], [EQ-INV-SPX-008], [EQ-INV-SPX-015],
        [REF-BREDIES-KUNISCH-POCK-2010], [REF-CHAMBOLLE-POCK-2011].

    """
    raw = require_operator(operator, name="operator")
    m, n = raw.shape
    if n < 3:
        raise ValueError("TGV^2 needs at least three unknowns")
    y = _vector(data, m, "data")
    raw, y = _whitened_problem(raw, y, noise)
    a1 = _positive(alpha1, "alpha1")
    a0 = _positive(alpha0, "alpha0")
    h = _positive(spacing, "spacing")
    x0 = (0.0,) * n if initial is None else _vector(initial, n, "initial")
    fwd = (
        frobenius_norm_bound(raw)
        if forward_norm_bound is None
        else _positive(forward_norm_bound, "forward_norm_bound")
    )
    first = FirstDifferenceOperator(n, spacing=h)
    second = FirstDifferenceOperator(n - 1, spacing=h)
    diff_bound = 2.0 / h
    norm_bound = math.sqrt(fwd**2 + (diff_bound**2 + 1.0) + diff_bound**2)
    kind = "certified: sqrt(||A||_F^2 + (||D||^2 + 1) + ||E||^2), ||D||,||E|| <= 2/h"
    a_count = CountingOperator(raw)
    k1 = n - 1
    k2 = n - 2

    def forward(z: Vector) -> Vector:
        """Return ``(A x, D x - v, E v)``.

        References:
            [EQ-INV-SPX-008].

        """
        x, v = z[:n], z[n:]
        dx = first.matvec(x)
        return (
            tuple(a_count.matvec(x))
            + tuple(a - b for a, b in zip(dx, v, strict=True))
            + tuple(second.matvec(v))
        )

    def adjoint(w: Vector) -> Vector:
        """Return the transpose of :func:`forward`.

        References:
            [EQ-INV-SPX-008].

        """
        w0, w1, w2 = w[:m], w[m : m + k1], w[m + k1 :]
        gx = tuple(
            a + b for a, b in zip(a_count.rmatvec(w0), first.rmatvec(w1), strict=True)
        )
        gv = tuple(-a + b for a, b in zip(w1, second.rmatvec(w2), strict=True))
        return gx + gv

    stacked = CallbackOperator((m + k1 + k2, n + k1), forward, adjoint, name="tgv2")

    def prox_primal(point: Vector, step: float) -> Vector:
        """Return ``point`` (the primal term is zero).

        References:
            [EQ-INV-SPX-008].

        """
        return point

    def prox_dual(point: Vector, step: float) -> Vector:
        """Return the prox of ``step g*``: data shrink and two clippings.

        References:
            [EQ-INV-SPX-008].

        """
        head = tuple(
            (a - step * b) / (1.0 + step) for a, b in zip(point[:m], y, strict=True)
        )
        return (
            head
            + project_linf_ball(point[m : m + k1], a1)
            + project_linf_ball(point[m + k1 :], a0)
        )

    def objective(point: Vector, image: Vector) -> float:
        """Return the primal objective from ``K z``.

        References:
            [EQ-INV-SPX-007].

        """
        fit = 0.5 * math.fsum((a - b) ** 2 for a, b in zip(image[:m], y, strict=True))
        return (
            fit
            + a1 * math.fsum(abs(v) for v in image[m : m + k1])
            + a0 * math.fsum(abs(v) for v in image[m + k1 :])
        )

    start = x0 + tuple(first.matvec(x0))
    optimization = primal_dual_hybrid_gradient(
        stacked,
        prox_primal,
        prox_dual,
        start,
        operator_norm_bound=norm_bound,
        step_safety=step_safety,
        objective=objective,
        tolerance_absolute=tolerance_absolute,
        tolerance_relative=tolerance_relative,
        max_iterations=max_iterations,
    )
    z = optimization.solution
    x, v = z[:n], z[n:]
    misfit = _half_misfit(raw, y, x)
    term1 = a1 * math.fsum(abs(a - b) for a, b in zip(first.matvec(x), v, strict=True))
    term0 = a0 * math.fsum(abs(e) for e in second.matvec(v))
    return TGVResult(
        solution=x,
        auxiliary=v,
        objective=misfit + term1 + term0,
        data_misfit=misfit,
        first_order_term=term1,
        second_order_term=term0,
        alpha1=a1,
        alpha0=a0,
        optimization=optimization,
        norm_bound=norm_bound,
        norm_bound_kind=kind,
        operator_counts=a_count.counts,
        converged=optimization.converged,
        stop_reason=optimization.stop_reason,
    )


# ---------------------------------------------------------------------------
# 3b. Second-order TGV on a two-dimensional grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TGV2DResult:
    """Result of :func:`tgv2_least_squares_2d`.

    Attributes:
        solution: Estimate ``x`` (row-major ``rows * columns`` values).
        auxiliary: The minimising vector field ``v = (v_1; v_2)`` stacked as
            two blocks of ``rows * columns`` values (an estimate of the
            gradient ``(D_x x; D_y x)``).
        objective: ``1/2||W(A x - y)||^2 + alpha1 sum_i |M (G x - v)_i| +
            alpha0 sum_i |(E v)_i|`` at ``(x, v)``; an UPPER bound of the
            TGV^2 objective at ``x`` (equal at the joint minimiser).
        data_misfit: ``1/2||W(A x - y)||^2``.
        first_order_term: ``alpha1 sum_i |M (G x - v)_i|``.
        second_order_term: ``alpha0 sum_i |(E v)_i|``.
        alpha1: First-order weight.
        alpha0: Second-order weight.
        rows: Grid rows.
        columns: Grid columns.
        spacing: ``(h_y, h_x)``.
        optimization: The underlying :class:`~probstat.optimization.PDHGResult`.
        norm_bound: The certified bound ``L >= ||K||_2`` used for the steps.
        norm_bound_kind: Description of how ``L`` was obtained.
        operator_counts: Applications of ``A`` (or ``W A`` with ``noise``).
        converged: Convergence flag of PDHG.
        stop_reason: Stop reason of PDHG.

    References:
        [EQ-INV-SPX-012], [EQ-INV-SPX-013].

    """

    solution: Vector
    auxiliary: Vector
    objective: float
    data_misfit: float
    first_order_term: float
    second_order_term: float
    alpha1: float
    alpha0: float
    rows: int
    columns: int
    spacing: tuple[float, float]
    optimization: PDHGResult
    norm_bound: float
    norm_bound_kind: str
    operator_counts: OperatorCounts
    converged: bool
    stop_reason: str

    @property
    def estimate(self) -> Vector:
        """Return ``solution`` (the name used by the other inverse results).

        References:
            [EQ-INV-LIN-001].

        """
        return self.solution


def _tgv2d_mask(rows: int, columns: int) -> Vector:
    """Return the 0/1 mask of the valid forward differences of the grid gradient.

    ``G = GridGradientOperator`` sets ``D_x`` to zero on the last column and
    ``D_y`` to zero on the last row (Neumann padding); the mask is zero
    exactly there, in the block layout ``[D_x; D_y]``.

    Args:
        rows: Grid rows.
        columns: Grid columns.

    Returns:
        The mask of length ``2 * rows * columns``.

    References:
        [EQ-INV-SPX-012].

    """
    horizontal = [
        0.0 if j == columns - 1 else 1.0 for _ in range(rows) for j in range(columns)
    ]
    vertical = [
        0.0 if i == rows - 1 else 1.0 for i in range(rows) for _ in range(columns)
    ]
    return tuple(horizontal + vertical)


def _grid_gradient(
    values: Sequence[float], rows: int, columns: int, hy: float, hx: float
) -> list[float]:
    """Return ``[D_x u; D_y u]``, the Neumann forward differences of a grid.

    Entry-for-entry the same map as
    :class:`~probstat.linalg.GridGradientOperator` (tested), without the
    per-call operator validation, because the primal-dual loop applies it
    thousands of times.

    Args:
        values: Row-major image of ``rows * columns`` entries.
        rows: Grid rows.
        columns: Grid columns.
        hy: Vertical spacing.
        hx: Horizontal spacing.

    Returns:
        The ``2 rows columns`` differences (zero on the last column of
        ``D_x`` and the last row of ``D_y``).

    References:
        [EQ-INV-SPX-012].

    """
    size = rows * columns
    out = [0.0] * (2 * size)
    for i in range(rows):
        base = i * columns
        row = values[base : base + columns]
        out[base : base + columns - 1] = [
            (b - a) / hx for a, b in zip(row, row[1:], strict=False)
        ]
        if i + 1 < rows:
            nxt = values[base + columns : base + 2 * columns]
            out[size + base : size + base + columns] = [
                (b - a) / hy for a, b in zip(row, nxt, strict=True)
            ]
    return out


def _grid_gradient_transpose(
    values: Sequence[float], rows: int, columns: int, hy: float, hx: float
) -> list[float]:
    """Return ``D_x' p + D_y' q`` for the stacked input ``(p; q)``.

    The transpose of :func:`_grid_gradient`.

    Args:
        values: ``(p; q)`` of length ``2 rows columns``.
        rows: Grid rows.
        columns: Grid columns.
        hy: Vertical spacing.
        hx: Horizontal spacing.

    Returns:
        The image-sized vector ``D_x' p + D_y' q``.

    References:
        [EQ-INV-SPX-012].

    """
    size = rows * columns
    out = [0.0] * size
    for i in range(rows):
        base = i * columns
        for j in range(columns - 1):
            weight = values[base + j] / hx
            out[base + j] -= weight
            out[base + j + 1] += weight
        if i + 1 < rows:
            for j in range(columns):
                weight = values[size + base + j] / hy
                out[base + j] -= weight
                out[base + columns + j] += weight
    return out


def _point_sum(values: Sequence[float], components: int) -> float:
    """Return ``sum_i |g_i|`` of block-stacked points (Euclidean per point).

    Args:
        values: Block-stacked entries.
        components: Blocks per point.

    Returns:
        The summed point norms.

    References:
        [EQ-INV-SPX-012].

    """
    return math.fsum(_point_norms(values, components))


def tgv2_least_squares_2d(
    operator: LinearOperator,
    data: Sequence[float],
    alpha1: float,
    alpha0: float,
    rows: int,
    columns: int,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    noise: object = None,
    forward_norm_bound: float | None = None,
    initial: Sequence[float] | None = None,
    tolerance_absolute: float = 1e-9,
    tolerance_relative: float = 1e-8,
    max_iterations: int = 50000,
    step_safety: float = 0.99,
) -> TGV2DResult:
    """Minimise ``1/2||W(A x - y)||^2 + TGV^2(x)`` on a ``rows x columns`` grid.

    The unknown image ``x`` is stored row-major. With the Neumann
    forward-difference gradient ``G = (D_x; D_y)``
    (the map of :class:`~probstat.linalg.GridGradientOperator`; ``D_x`` is
    zero on the last column, ``D_y`` on the last row) and a vector field
    ``v = (v_1, v_2)`` on the same grid, the symmetrised gradient (with the same Neumann
    differences) is ``E v = (D_x v_1, D_y v_2, (D_y v_1 + D_x v_2)/sqrt 2)``,
    whose Euclidean point norm equals the Frobenius norm of the symmetric
    ``2 x 2`` tensor. The model is
    ``TGV^2(x) = min_v alpha1 sum_i |M (G x - v)_i| + alpha0 sum_i |(E v)_i|``
    with ``M`` the mask that removes the zero-padded entries of ``G`` from the
    first-order term; the free entries of ``v`` on the last row and column
    are then only controlled by ``E``, so every affine image has TGV^2 EXACTLY
    zero (take ``v`` constant) while ``G x`` alone has an artificial boundary
    jump [EQ-INV-SPX-012]. In ``z = (x, v)`` the problem is ``min g(K z)``,
    ``K z = (A x, M (G x - v), E v)``, with ``g`` the data term plus two
    pointwise Euclidean norms; it is solved by
    :func:`~probstat.optimization.primal_dual_hybrid_gradient`, whose dual prox
    is the data shrink and two pointwise ball projections (radii ``alpha1``,
    ``alpha0``).

    Step sizes (PROVEN, not estimated) [EQ-INV-SPX-013]. For forward
    differences ``||D_x||^2 <= 4/h_x^2`` and ``||D_y||^2 <= 4/h_y^2`` (from
    ``(a - b)^2 <= 2 a^2 + 2 b^2``), so ``||G||^2 <= 4/h_x^2 + 4/h_y^2``
    (``8/h^2`` on a square grid). Since ``(a + b)^2/2 <= a^2 + b^2``,
    ``||E v||^2 <= ||G v_1||^2 + ||G v_2||^2``, i.e. ``||E||^2 <= ||G||^2``;
    ``||M (G x - v)||^2 <= (||G||^2 + 1)(||x||^2 + ||v||^2)`` by
    Cauchy-Schwarz. Adding the three blocks gives
    ``||K||^2 <= ||A||_F^2 + 2 ||G||^2 + 1``, and the code uses
    ``||G||^2 = 4/h_x^2 + 4/h_y^2``. The bound is conservative; convergence
    can be slow, and pure-Python iterations cost ``O(n)`` each.

    ``noise`` replaces the data term by ``1/2 ||W (A x - y)||^2``
    [EQ-INV-SPX-015].

    Args:
        operator: ``A`` (``m x rows*columns``) or a dense matrix.
        data: ``y`` of length ``m``.
        alpha1: Positive weight of the first-order term.
        alpha0: Positive weight of the second-order term.
        rows: Number of image rows (at least 2).
        columns: Number of image columns (at least 2).
        spacing: Positive ``(h_y, h_x)``.
        noise: ``None`` or a covariance factor of dimension ``m``.
        forward_norm_bound: Certified bound on ``||A||_2`` (``||W A||_2`` with
            ``noise``); default the Frobenius norm, which needs a
            materialisable operator.
        initial: Start ``x_0`` (zero by default); ``v_0 = G x_0``.
        tolerance_absolute: PDHG absolute residual tolerance.
        tolerance_relative: PDHG relative residual tolerance.
        max_iterations: PDHG iteration cap.
        step_safety: PDHG step factor in ``(0, 1)``.

    Returns:
        :class:`TGV2DResult`; ``converged`` is ``False`` at the cap.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a weight or a spacing entry is not positive and finite,
            a grid side is below two, ``rows * columns`` differs from the
            number of unknowns, data are not finite, a dimension is
            inconsistent, or ``step_safety`` lies outside ``(0, 1)``.

    References:
        [EQ-INV-SPX-012], [EQ-INV-SPX-013], [EQ-INV-SPX-015],
        [REF-BREDIES-KUNISCH-POCK-2010], [REF-CHAMBOLLE-POCK-2011].

    """
    raw = require_operator(operator, name="operator")
    m, n = raw.shape
    r, c = _grid_shape((rows, columns))
    if r * c != n:
        raise ValueError("rows * columns must equal the number of unknowns")
    y = _vector(data, m, "data")
    raw, y = _whitened_problem(raw, y, noise)
    a1 = _positive(alpha1, "alpha1")
    a0 = _positive(alpha0, "alpha0")
    if len(tuple(spacing)) != 2:
        raise ValueError("spacing must be (h_y, h_x)")
    hy = _positive(spacing[0], "spacing")
    hx = _positive(spacing[1], "spacing")
    x0 = (0.0,) * n if initial is None else _vector(initial, n, "initial")
    fwd = (
        frobenius_norm_bound(raw)
        if forward_norm_bound is None
        else _positive(forward_norm_bound, "forward_norm_bound")
    )
    mask = _tgv2d_mask(r, c)
    grad_sq = 4.0 / (hx * hx) + 4.0 / (hy * hy)
    norm_bound = math.sqrt(fwd**2 + 2.0 * grad_sq + 1.0)
    kind = "certified: sqrt(||A||_F^2 + 2 ||G||^2 + 1), ||G||^2 <= 4/hx^2 + 4/hy^2"
    a_count = CountingOperator(raw)
    root_half = math.sqrt(0.5)
    b1 = 2 * n
    b2 = 3 * n

    def forward(z: Vector) -> Vector:
        """Return ``(A x, M (G x - v), E v)``.

        References:
            [EQ-INV-SPX-012].

        """
        x, v1, v2 = z[:n], z[n : 2 * n], z[2 * n :]
        gx = _grid_gradient(x, r, c, hy, hx)
        first = tuple(k * (a - b) for k, a, b in zip(mask, gx, z[n:], strict=True))
        g1 = _grid_gradient(v1, r, c, hy, hx)
        g2 = _grid_gradient(v2, r, c, hy, hx)
        shear = tuple(root_half * (a + b) for a, b in zip(g1[n:], g2[:n], strict=True))
        return tuple(a_count.matvec(x)) + first + tuple(g1[:n]) + tuple(g2[n:]) + shear

    def adjoint(w: Vector) -> Vector:
        """Return the transpose of :func:`forward`.

        References:
            [EQ-INV-SPX-012].

        """
        w0, w1, w2 = w[:m], w[m : m + b1], w[m + b1 :]
        masked = tuple(k * a for k, a in zip(mask, w1, strict=True))
        gx = tuple(
            a + b
            for a, b in zip(
                a_count.rmatvec(w0),
                _grid_gradient_transpose(masked, r, c, hy, hx),
                strict=True,
            )
        )
        p11, p22 = w2[:n], w2[n:b1]
        shear = tuple(root_half * a for a in w2[b1:b2])
        t1 = _grid_gradient_transpose(tuple(p11) + shear, r, c, hy, hx)
        t2 = _grid_gradient_transpose(shear + tuple(p22), r, c, hy, hx)
        return (
            gx
            + tuple(-a + b for a, b in zip(masked[:n], t1, strict=True))
            + tuple(-a + b for a, b in zip(masked[n:], t2, strict=True))
        )

    stacked = CallbackOperator((m + b1 + b2, 3 * n), forward, adjoint, name="tgv2-2d")

    def prox_primal(point: Vector, step: float) -> Vector:
        """Return ``point`` (the primal term is zero).

        References:
            [EQ-INV-SPX-012].

        """
        return point

    def prox_dual(point: Vector, step: float) -> Vector:
        """Return the prox of ``step g*``: data shrink and two ball projections.

        References:
            [EQ-INV-SPX-012].

        """
        head = tuple(
            (a - step * b) / (1.0 + step) for a, b in zip(point[:m], y, strict=True)
        )
        return (
            head
            + project_pointwise_l2_ball(point[m : m + b1], a1, components=2)
            + project_pointwise_l2_ball(point[m + b1 :], a0, components=3)
        )

    def objective(point: Vector, image: Vector) -> float:
        """Return the primal objective from ``K z``.

        References:
            [EQ-INV-SPX-012].

        """
        fit = 0.5 * math.fsum((a - b) ** 2 for a, b in zip(image[:m], y, strict=True))
        return (
            fit
            + a1 * _point_sum(image[m : m + b1], 2)
            + a0 * _point_sum(image[m + b1 :], 3)
        )

    start = x0 + tuple(_grid_gradient(x0, r, c, hy, hx))
    optimization = primal_dual_hybrid_gradient(
        stacked,
        prox_primal,
        prox_dual,
        start,
        operator_norm_bound=norm_bound,
        step_safety=step_safety,
        objective=objective,
        tolerance_absolute=tolerance_absolute,
        tolerance_relative=tolerance_relative,
        max_iterations=max_iterations,
    )
    z = optimization.solution
    image = forward(z)
    misfit = _half_misfit(raw, y, z[:n])
    term1 = a1 * _point_sum(image[m : m + b1], 2)
    term0 = a0 * _point_sum(image[m + b1 :], 3)
    return TGV2DResult(
        solution=z[:n],
        auxiliary=z[n:],
        objective=misfit + term1 + term0,
        data_misfit=misfit,
        first_order_term=term1,
        second_order_term=term0,
        alpha1=a1,
        alpha0=a0,
        rows=r,
        columns=c,
        spacing=(hy, hx),
        optimization=optimization,
        norm_bound=norm_bound,
        norm_bound_kind=kind,
        operator_counts=a_count.counts,
        converged=optimization.converged,
        stop_reason=optimization.stop_reason,
    )


# ---------------------------------------------------------------------------
# 4. Non-convex MCP and SCAD
# ---------------------------------------------------------------------------


def _penalty_parameters(
    lam: float, shape: float, floor: float, name: str
) -> tuple[float, float]:
    """Validate ``(lam, shape)`` with ``shape > floor``.

    Args:
        lam: Penalty level (non-negative).
        shape: Concavity parameter.
        floor: Strict lower limit of ``shape``.
        name: Name of the shape parameter in messages.

    Returns:
        ``(lam, shape)`` as floats.

    Raises:
        TypeError: If a value is not real.
        ValueError: If ``lam < 0``, a value is not finite or
            ``shape <= floor``.

    References:
        [EQ-INV-SPX-009], [EQ-INV-SPX-010].

    """
    level = _nonnegative(lam, "lam")
    concavity = _real(shape, name)
    if concavity <= floor:
        raise ValueError(f"{name} must exceed {floor}")
    return level, concavity


def mcp_penalty(values: Sequence[float], lam: float, gamma: float = 3.0) -> float:
    """Return ``sum_i p(x_i)`` for the minimax concave penalty.

    ``p(t) = lam |t| - t^2/(2 gamma)`` for ``|t| <= gamma lam`` and
    ``gamma lam^2 / 2`` beyond [EQ-INV-SPX-009].

    Args:
        values: Entries ``x``.
        lam: Non-negative level.
        gamma: Concavity parameter ``> 1``.

    Returns:
        The summed penalty.

    Raises:
        TypeError: If an input is not real.
        ValueError: If ``lam < 0``, ``gamma <= 1`` or values are not finite.

    References:
        [EQ-INV-SPX-009], [REF-ZHANG-2010].

    """
    level, g = _penalty_parameters(lam, gamma, 1.0, "gamma")
    x = _vector(values, len(tuple(values)), "values")
    return math.fsum(
        level * abs(t) - t * t / (2.0 * g)
        if abs(t) <= g * level
        else 0.5 * g * level * level
        for t in x
    )


def scad_penalty(values: Sequence[float], lam: float, a: float = 3.7) -> float:
    """Return ``sum_i p(x_i)`` for the SCAD penalty.

    ``p(t) = lam |t|`` for ``|t| <= lam``,
    ``(2 a lam |t| - t^2 - lam^2) / (2 (a - 1))`` for ``lam < |t| <= a lam``
    and ``(a + 1) lam^2 / 2`` beyond [EQ-INV-SPX-010].

    Args:
        values: Entries ``x``.
        lam: Non-negative level.
        a: Concavity parameter ``> 2``.

    Returns:
        The summed penalty.

    Raises:
        TypeError: If an input is not real.
        ValueError: If ``lam < 0``, ``a <= 2`` or values are not finite.

    References:
        [EQ-INV-SPX-010], [REF-FAN-LI-2001].

    """
    level, shape = _penalty_parameters(lam, a, 2.0, "a")
    x = _vector(values, len(tuple(values)), "values")

    def one(t: float) -> float:
        """Return the scalar SCAD penalty.

        References:
            [EQ-INV-SPX-010].

        """
        s = abs(t)
        if s <= level:
            return level * s
        if s <= shape * level:
            return (2.0 * shape * level * s - s * s - level * level) / (
                2.0 * (shape - 1.0)
            )
        return 0.5 * (shape + 1.0) * level * level

    return math.fsum(one(t) for t in x)


def _checked_step(step: float, limit: float, name: str) -> float:
    """Return a step that keeps the scalar prox problem strongly convex.

    Args:
        step: Proposed step ``s``.
        limit: Strict upper limit (``gamma`` or ``a - 1``).
        name: Name of the concavity parameter.

    Returns:
        The step as a float.

    Raises:
        TypeError: If the step is not real.
        ValueError: If ``step`` is not positive and finite or ``step >=
            limit``.

    References:
        [EQ-INV-SPX-009], [EQ-INV-SPX-010].

    """
    s = _positive(step, "step")
    if s >= limit:
        raise ValueError(
            f"step must be below {limit} ({name}-dependent): the prox of a "
            "non-convex penalty is single-valued only below this limit"
        )
    return s


def prox_mcp(
    vector: Sequence[float], step: float, lam: float, gamma: float = 3.0
) -> Vector:
    """Return the exact prox of ``step * MCP`` (firm thresholding).

    The minimiser of ``1/2 (x - v)^2 + step p(x)`` is
    ``0`` for ``|v| <= step lam``,
    ``sign(v) (|v| - step lam) / (1 - step/gamma)`` for
    ``step lam < |v| <= gamma lam`` and ``v`` for ``|v| > gamma lam``
    [EQ-INV-SPX-009]. It is the GLOBAL minimiser and unique only when
    ``step < gamma`` (the quadratic part then dominates the concavity), which
    is enforced.

    Args:
        vector: Input ``v``.
        step: Step ``s`` with ``0 < s < gamma``.
        lam: Non-negative level.
        gamma: Concavity parameter ``> 1``.

    Returns:
        The proximal point entrywise.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a parameter is out of range, ``step >= gamma`` or
            ``vector`` is not finite.

    References:
        [EQ-INV-SPX-009], [REF-ZHANG-2010].

    """
    level, g = _penalty_parameters(lam, gamma, 1.0, "gamma")
    s = _checked_step(step, g, "gamma")
    v = _vector(vector, len(tuple(vector)), "vector")
    out = []
    for t in v:
        a = abs(t)
        if a <= s * level:
            out.append(0.0)
        elif a <= g * level:
            out.append(math.copysign((a - s * level) / (1.0 - s / g), t))
        else:
            out.append(t)
    return tuple(out)


def prox_scad(
    vector: Sequence[float], step: float, lam: float, a: float = 3.7
) -> Vector:
    """Return the exact prox of ``step * SCAD``.

    The minimiser of ``1/2 (x - v)^2 + step p(x)`` is
    ``sign(v) max(|v| - step lam, 0)`` for ``|v| <= (1 + step) lam``,
    ``((a - 1) v - sign(v) a step lam) / (a - 1 - step)`` for
    ``(1 + step) lam < |v| <= a lam`` and ``v`` for ``|v| > a lam``
    [EQ-INV-SPX-010]. It is the global minimiser and unique when
    ``step < a - 1``, which is enforced.

    Args:
        vector: Input ``v``.
        step: Step ``s`` with ``0 < s < a - 1``.
        lam: Non-negative level.
        a: Concavity parameter ``> 2``.

    Returns:
        The proximal point entrywise.

    Raises:
        TypeError: If an input is not real.
        ValueError: If a parameter is out of range, ``step >= a - 1`` or
            ``vector`` is not finite.

    References:
        [EQ-INV-SPX-010], [REF-FAN-LI-2001].

    """
    level, shape = _penalty_parameters(lam, a, 2.0, "a")
    s = _checked_step(step, shape - 1.0, "a")
    v = _vector(vector, len(tuple(vector)), "vector")
    out = []
    for t in v:
        mag = abs(t)
        if mag <= (1.0 + s) * level:
            out.append(math.copysign(max(mag - s * level, 0.0), t))
        elif mag <= shape * level:
            out.append(
                math.copysign(
                    ((shape - 1.0) * mag - shape * s * level) / (shape - 1.0 - s),
                    t,
                )
            )
        else:
            out.append(t)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class NonconvexResult:
    """Result of :func:`nonconvex_sparse_least_squares`.

    Attributes:
        solution: Final iterate (a stationary point when ``converged``).
        objective: ``1/2||A x - y||^2 + sum_i p(x_i)``.
        data_misfit: ``1/2||A x - y||^2``.
        penalty: ``sum_i p(x_i)``.
        support: Sorted indices with ``x_i != 0``.
        gradient_mapping_norm: Gradient-mapping norm at the solution (zero
            exactly at a fixed point of the proximal gradient map).
        penalty_name: ``"mcp"`` or ``"scad"``.
        lam: Penalty level.
        shape: Concavity parameter (``gamma`` or ``a``).
        optimization: The underlying
            :class:`~probstat.optimization.ProximalGradientResult`.
        operator_counts: Applications of ``A``.
        guarantee: Fixed statement of what is certified: a stationary point
            only, dependent on the start, not a global minimiser.
        converged: Convergence flag of the iteration.
        stop_reason: Stop reason of the iteration.

    References:
        [EQ-INV-SPX-011].

    """

    solution: Vector
    objective: float
    data_misfit: float
    penalty: float
    support: tuple[int, ...]
    gradient_mapping_norm: float
    penalty_name: str
    lam: float
    shape: float
    optimization: ProximalGradientResult
    operator_counts: OperatorCounts
    guarantee: str
    converged: bool
    stop_reason: str

    @property
    def estimate(self) -> Vector:
        """Return ``solution`` (the name used by the other inverse results).

        References:
            [EQ-INV-LIN-001].

        """
        return self.solution


def nonconvex_sparse_least_squares(
    operator: LinearOperator,
    data: Sequence[float],
    lam: float,
    *,
    penalty: str = "mcp",
    shape: float | None = None,
    initial: Sequence[float] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
    noise: object = None,
) -> NonconvexResult:
    """Minimise ``1/2||A x - y||^2 + sum_i p(x_i)`` for MCP or SCAD.

    Proximal gradient (:func:`~probstat.optimization.ista`) with the exact
    prox maps :func:`prox_mcp` / :func:`prox_scad` and Beck-Teboulle
    backtracking for the smooth part [EQ-INV-SPX-011]. The prox is a global
    minimiser, so every step decreases the objective, and a point with zero
    gradient mapping is a STATIONARY point (a first-order condition). The
    objective is NOT convex: the iteration is NOT guaranteed to find the
    global minimiser, and different start points can end at different
    stationary points with different objective values. Nothing here
    certifies global optimality. The first trial step is
    ``min(1, 0.9 * limit)`` with ``limit = gamma`` (MCP) or ``a - 1`` (SCAD),
    so the prox stays single-valued; backtracking only shrinks it.

    Args:
        operator: ``A`` (``m x n``) or a dense matrix.
        data: ``y`` of length ``m``.
        lam: Non-negative level ``lam`` (unsquared, like ``alpha``).
        penalty: ``"mcp"`` or ``"scad"``.
        shape: ``gamma > 1`` for MCP (default 3) or ``a > 2`` for SCAD
            (default 3.7).
        initial: Start point (zero by default); the result depends on it.
        tolerance: Absolute gradient-mapping tolerance.
        max_iterations: Iteration cap.
        noise: ``None`` or a covariance factor of dimension ``m``; the data
            term becomes ``1/2 ||W (A x - y)||^2`` [EQ-INV-SPX-015] (the
            reported misfit refers to ``W A``).

    Returns:
        :class:`NonconvexResult`; ``converged`` is ``False`` at the cap.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``penalty`` is unknown, a parameter is out of range,
            or ``data`` or ``initial`` is not finite.

    References:
        [EQ-INV-SPX-009], [EQ-INV-SPX-010], [EQ-INV-SPX-011],
        [EQ-INV-SPX-015], [REF-ZHANG-2010], [REF-FAN-LI-2001],
        [REF-BECK-TEBOULLE-2009].

    """
    raw = require_operator(operator, name="operator")
    m, n = raw.shape
    y = _vector(data, m, "data")
    raw, y = _whitened_problem(raw, y, noise)
    if penalty == "mcp":
        s_value = 3.0 if shape is None else shape
        level, concavity = _penalty_parameters(lam, s_value, 1.0, "gamma")
        limit = concavity
    elif penalty == "scad":
        s_value = 3.7 if shape is None else shape
        level, concavity = _penalty_parameters(lam, s_value, 2.0, "a")
        limit = concavity - 1.0
    else:
        raise ValueError("penalty must be 'mcp' or 'scad'")
    x0 = (0.0,) * n if initial is None else _vector(initial, n, "initial")
    counted = CountingOperator(raw)

    def smooth_value(point: Vector) -> float:
        """Return ``1/2||A x - y||^2``.

        References:
            [EQ-INV-SPX-011].

        """
        return _half_misfit(counted, y, point)

    def smooth_gradient(point: Vector) -> Vector:
        """Return ``A'(A x - y)``.

        References:
            [EQ-INV-SPX-011].

        """
        resid = tuple(a - b for a, b in zip(counted.matvec(point), y, strict=True))
        return tuple(counted.rmatvec(resid))

    def penalty_value(point: Vector) -> float:
        """Return the penalty ``sum p(x_i)``.

        References:
            [EQ-INV-SPX-009].

        """
        if penalty == "mcp":
            return mcp_penalty(point, level, concavity)
        return scad_penalty(point, level, concavity)

    def prox(point: Vector, step: float) -> Vector:
        """Return the exact prox of ``step * p``.

        References:
            [EQ-INV-SPX-009].

        """
        if penalty == "mcp":
            return prox_mcp(point, step, level, concavity)
        return prox_scad(point, step, level, concavity)

    run = ista(
        smooth_value,
        smooth_gradient,
        penalty_value,
        prox,
        x0,
        initial_step=min(1.0, 0.9 * limit),
        tolerance=_nonnegative(tolerance, "tolerance"),
        max_iterations=max_iterations,
    )
    x = run.solution
    misfit = _half_misfit(raw, y, x)
    pen = penalty_value(x)
    return NonconvexResult(
        solution=x,
        objective=misfit + pen,
        data_misfit=misfit,
        penalty=pen,
        support=tuple(i for i, v in enumerate(x) if v != 0.0),
        gradient_mapping_norm=run.gradient_mapping_norm,
        penalty_name=penalty,
        lam=level,
        shape=concavity,
        optimization=run,
        operator_counts=counted.counts,
        guarantee=(
            "stationary point of a non-convex objective; depends on the start "
            "point; not a global minimiser"
        ),
        converged=run.converged,
        stop_reason=run.stop_reason,
    )


__all__ = [
    "HuberTVResult",
    "IRLSResult",
    "NonconvexResult",
    "TGV2DResult",
    "TGVResult",
    "huber_total_variation_least_squares",
    "huber_total_variation_value",
    "irls_regularized_least_squares",
    "mcp_penalty",
    "nonconvex_sparse_least_squares",
    "prox_mcp",
    "prox_scad",
    "scad_penalty",
    "tgv2_least_squares_1d",
    "tgv2_least_squares_2d",
]
