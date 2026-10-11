"""Discrete optimal transport: exact plans, entropic Sinkhorn, and divergences.

The Kantorovich problem between weights ``a`` (length ``m``) and ``b``
(length ``n``) with equal total mass and a cost matrix ``C`` is

``min_P sum_ij C_ij P_ij`` subject to ``P >= 0``, ``P 1 = a``, ``P' 1 = b``,

with dual ``max sum_i a_i u_i + sum_j b_j v_j`` subject to
``u_i + v_j <= C_ij``. :func:`exact_transport` solves it with the
transportation simplex (north-west corner start, tree potentials, cycle
pivots, Bland's rule after repeated degenerate pivots) and returns dual
potentials, so :func:`transport_certificate` can verify optimality through
primal and dual feasibility and complementary slackness
``P_ij (C_ij - u_i - v_j) = 0``. :func:`sinkhorn` solves the entropic problem
in the log domain, and :func:`sinkhorn_divergence` debiases it.

Everything is pure Python and dense, so the exact solver is meant for
problems with up to a few hundred support points.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from probstat.foundations import require_finite_real
from probstat.linalg import Matrix, Vector, as_matrix

_BALANCE_TOLERANCE = 1e-9
_STAGE_TOLERANCE = 1e-3

Point = tuple[float, ...]


def _weights(values: Iterable[object], *, name: str) -> Vector:
    """Validate a non-negative weight vector with positive total mass.

    Args:
        values: Candidate weights.
        name: Argument name for error messages.

    Returns:
        Weights as floats.

    Raises:
        ValueError: If the vector is empty, has a negative or non-finite
            entry, or has zero total mass.

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.1]. No equation is evaluated.

    """
    weights = tuple(
        require_finite_real(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not weights:
        raise ValueError(f"{name} must not be empty")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"{name} entries must be non-negative")
    if not math.fsum(weights) > 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return weights


def _problem(
    source_weights: Iterable[object],
    target_weights: Iterable[object],
    cost: Iterable[Iterable[object]],
) -> tuple[Vector, Vector, Matrix]:
    """Validate a balanced transport problem.

    Args:
        source_weights: Source weights ``a``.
        target_weights: Target weights ``b``.
        cost: Cost matrix with ``len(a)`` rows and ``len(b)`` columns.

    Returns:
        Validated ``(a, b, C)``.

    Raises:
        ValueError: If the masses differ by more than ``1e-9`` relatively or
            the cost matrix has the wrong shape.

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.3]. No equation is evaluated.

    """
    a = _weights(source_weights, name="source_weights")
    b = _weights(target_weights, name="target_weights")
    matrix = as_matrix(cost, name="cost")
    if len(matrix) != len(a) or len(matrix[0]) != len(b):
        raise ValueError("cost must have len(source) rows and len(target) columns")
    mass_a, mass_b = math.fsum(a), math.fsum(b)
    if abs(mass_a - mass_b) > _BALANCE_TOLERANCE * max(1.0, mass_a, mass_b):
        raise ValueError("source and target weights must have equal total mass")
    return a, b, matrix


def _points(values: Iterable[object], *, name: str) -> tuple[Point, ...]:
    """Validate a point cloud of scalars or equal-length vectors.

    Args:
        values: Real numbers (1-D points) or iterables of reals.
        name: Argument name for error messages.

    Returns:
        Points as tuples of floats with a common dimension.

    Raises:
        ValueError: If the cloud is empty or dimensions disagree.

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.1]. No equation is evaluated.

    """
    points: list[Point] = []
    for index, value in enumerate(values):
        if isinstance(value, Iterable) and not isinstance(value, str | bytes):
            point = tuple(
                require_finite_real(component, name=f"{name}[{index}]")
                for component in value
            )
        else:
            point = (require_finite_real(value, name=f"{name}[{index}]"),)
        points.append(point)
    if not points:
        raise ValueError(f"{name} must not be empty")
    dimension = len(points[0])
    if dimension == 0 or any(len(point) != dimension for point in points):
        raise ValueError(f"{name} points must share one positive dimension")
    return tuple(points)


def power_cost_matrix(
    source_points: Iterable[object], target_points: Iterable[object], *, p: float = 1.0
) -> Matrix:
    """Return ``C_ij = |x_i - y_j|^p`` with the Euclidean norm.

    Args:
        source_points: Scalars or equal-length vectors ``x_i``.
        target_points: Scalars or vectors ``y_j`` of the same dimension.
        p: Positive exponent; ``p >= 1`` makes ``(OT cost)^(1/p)`` the
            Wasserstein-``p`` distance.

    Returns:
        Dense ``m x n`` cost matrix.

    Raises:
        ValueError: If the clouds are invalid, dimensions differ, or ``p`` is
            not positive.

    Equation:
        [EQ-OT-COST-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.4], [REF-VILLANI-2009, Chapter 6].

    """
    exponent = require_finite_real(p, name="p")
    if not exponent > 0.0:
        raise ValueError("p must be positive")
    xs = _points(source_points, name="source_points")
    ys = _points(target_points, name="target_points")
    if len(xs[0]) != len(ys[0]):
        raise ValueError("source and target points must share one dimension")
    rows: list[Vector] = []
    for x in xs:
        row: list[float] = []
        for y in ys:
            squared = math.fsum((xi - yi) ** 2 for xi, yi in zip(x, y, strict=True))
            row.append(squared if exponent == 2.0 else squared ** (0.5 * exponent))
        rows.append(tuple(row))
    return tuple(rows)


def squared_euclidean_cost_matrix(
    source_points: Iterable[object], target_points: Iterable[object]
) -> Matrix:
    """Return ``C_ij = |x_i - y_j|^2``, the cost behind ``W_2``.

    Args:
        source_points: Scalars or equal-length vectors ``x_i``.
        target_points: Scalars or vectors ``y_j`` of the same dimension.

    Returns:
        Dense ``m x n`` cost matrix.

    Raises:
        ValueError: If the clouds are invalid or dimensions differ.

    Equation:
        [EQ-OT-COST-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.4].

    """
    return power_cost_matrix(source_points, target_points, p=2.0)


@dataclass(frozen=True, slots=True)
class OptimalityCertificate:
    """Primal-dual optimality evidence for a transport plan.

    Args:
        primal_cost: ``sum_ij C_ij P_ij``.
        dual_cost: ``sum_i a_i u_i + sum_j b_j v_j``.
        duality_gap: ``|primal_cost - dual_cost|``.
        marginal_error: Largest absolute row or column marginal violation.
        negativity: Largest negative plan entry magnitude (zero if none).
        dual_infeasibility: ``max(0, max_ij u_i + v_j - C_ij)``.
        complementary_slackness: ``max_ij P_ij |C_ij - u_i - v_j|``.
        tolerance: Absolute tolerance used for ``optimal``.
        optimal: Whether every residual is within ``tolerance``.

    References:
        [REF-PEYRE-CUTURI-2019, Section 3.3].

    """

    primal_cost: float
    dual_cost: float
    duality_gap: float
    marginal_error: float
    negativity: float
    dual_infeasibility: float
    complementary_slackness: float
    tolerance: float
    optimal: bool


def transport_certificate(
    plan: Iterable[Iterable[object]],
    source_potential: Iterable[object],
    target_potential: Iterable[object],
    source_weights: Iterable[object],
    target_weights: Iterable[object],
    cost: Iterable[Iterable[object]],
    *,
    tolerance: float = 1e-9,
) -> OptimalityCertificate:
    """Check a plan and dual potentials for optimality.

    A primal-feasible plan and dual-feasible potentials that satisfy
    complementary slackness are both optimal, and the duality gap is then
    zero. Residuals are compared with ``tolerance * max(1, max|C|, mass)``.

    Args:
        plan: Candidate ``m x n`` transport plan.
        source_potential: Dual potential ``u`` of length ``m``.
        target_potential: Dual potential ``v`` of length ``n``.
        source_weights: Source weights ``a``.
        target_weights: Target weights ``b``.
        cost: Cost matrix ``C``.
        tolerance: Positive relative tolerance.

    Returns:
        Residuals and an ``optimal`` verdict.

    Raises:
        ValueError: If shapes disagree or the tolerance is not positive.

    Equation:
        [EQ-OT-LP-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 3.3].

    """
    a, b, matrix = _problem(source_weights, target_weights, cost)
    transport = as_matrix(plan, name="plan")
    u = tuple(require_finite_real(x, name="source_potential") for x in source_potential)
    v = tuple(require_finite_real(x, name="target_potential") for x in target_potential)
    m, n = len(a), len(b)
    if len(transport) != m or len(transport[0]) != n or len(u) != m or len(v) != n:
        raise ValueError("plan and potentials must match the weight lengths")
    relative = require_finite_real(tolerance, name="tolerance")
    if not relative > 0.0:
        raise ValueError("tolerance must be positive")
    primal = math.fsum(
        matrix[i][j] * transport[i][j] for i in range(m) for j in range(n)
    )
    dual = math.fsum(a[i] * u[i] for i in range(m)) + math.fsum(
        b[j] * v[j] for j in range(n)
    )
    rows = max(abs(math.fsum(transport[i]) - a[i]) for i in range(m))
    columns = max(
        abs(math.fsum(transport[i][j] for i in range(m)) - b[j]) for j in range(n)
    )
    negativity = max(0.0, -min(min(row) for row in transport))
    infeasibility = max(
        0.0, max(u[i] + v[j] - matrix[i][j] for i in range(m) for j in range(n))
    )
    slackness = max(
        abs(transport[i][j]) * abs(matrix[i][j] - u[i] - v[j])
        for i in range(m)
        for j in range(n)
    )
    scale = max(1.0, max(abs(value) for row in matrix for value in row), math.fsum(a))
    limit = relative * scale
    gap = abs(primal - dual)
    marginal = max(rows, columns)
    optimal = all(
        value <= limit
        for value in (gap, marginal, negativity, infeasibility, slackness)
    )
    return OptimalityCertificate(
        primal_cost=primal,
        dual_cost=dual,
        duality_gap=gap,
        marginal_error=marginal,
        negativity=negativity,
        dual_infeasibility=infeasibility,
        complementary_slackness=slackness,
        tolerance=limit,
        optimal=optimal,
    )


@dataclass(frozen=True, slots=True)
class TransportPlan:
    """An exact optimal transport plan with dual potentials.

    Args:
        plan: Optimal ``m x n`` coupling ``P``.
        cost: Optimal cost ``sum_ij C_ij P_ij``.
        source_potential: Dual potential ``u`` (normalized by ``u_0 = 0``).
        target_potential: Dual potential ``v``.
        basis: The ``m + n - 1`` basic cells of the final spanning tree.
        iterations: Number of simplex pivots.
        source_weights: Source weights ``a``.
        target_weights: Target weights ``b``.
        cost_matrix: Cost matrix ``C``.

    References:
        [REF-PEYRE-CUTURI-2019, Section 3.5], [REF-DANTZIG-1951].

    """

    plan: Matrix
    cost: float
    source_potential: Vector
    target_potential: Vector
    basis: tuple[tuple[int, int], ...]
    iterations: int
    source_weights: Vector = field(repr=False)
    target_weights: Vector = field(repr=False)
    cost_matrix: Matrix = field(repr=False)

    def certificate(self, *, tolerance: float = 1e-9) -> OptimalityCertificate:
        """Return the complementary-slackness certificate of this plan.

        Args:
            tolerance: Positive relative tolerance.

        Returns:
            Residuals and the ``optimal`` verdict.

        Equation:
            [EQ-OT-LP-001].

        References:
            [REF-PEYRE-CUTURI-2019, Section 3.3].

        """
        return transport_certificate(
            self.plan,
            self.source_potential,
            self.target_potential,
            self.source_weights,
            self.target_weights,
            self.cost_matrix,
            tolerance=tolerance,
        )


def _north_west_corner(
    a: Vector, b: Vector
) -> tuple[list[list[float]], list[tuple[int, int]]]:
    """Return a basic feasible plan with exactly ``m + n - 1`` basic cells.

    Args:
        a: Source weights.
        b: Target weights with the same total mass.

    Returns:
        Plan and basic cells, which form a spanning tree of the bipartite
        row-column graph (degenerate cells carry zero mass).

    Equation:
        [EQ-OT-SIMPLEX-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 3.4].

    """
    m, n = len(a), len(b)
    remaining_a, remaining_b = list(a), list(b)
    plan = [[0.0] * n for _ in range(m)]
    basis: list[tuple[int, int]] = []
    i = j = 0
    while True:
        basis.append((i, j))
        if i == m - 1:
            amount = max(0.0, remaining_b[j])
            plan[i][j] = amount
            remaining_a[i] -= amount
            if j == n - 1:
                break
            j += 1
        elif j == n - 1 or remaining_a[i] <= remaining_b[j]:
            amount = max(0.0, remaining_a[i])
            plan[i][j] = amount
            remaining_b[j] -= amount
            remaining_a[i] = 0.0
            i += 1
        else:
            amount = max(0.0, remaining_b[j])
            plan[i][j] = amount
            remaining_a[i] -= amount
            remaining_b[j] = 0.0
            j += 1
    return plan, basis


def _potentials(
    matrix: Matrix, basis: Sequence[tuple[int, int]], m: int, n: int
) -> tuple[list[float], list[float]]:
    """Solve ``u_i + v_j = C_ij`` on the basic tree with ``u_0 = 0``.

    Args:
        matrix: Cost matrix.
        basis: Basic cells forming a spanning tree.
        m: Number of rows.
        n: Number of columns.

    Returns:
        Row potentials ``u`` and column potentials ``v``.

    Raises:
        ArithmeticError: If the basis does not span every row and column.

    Equation:
        [EQ-OT-SIMPLEX-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 3.5].

    """
    row_cells: list[list[int]] = [[] for _ in range(m)]
    column_cells: list[list[int]] = [[] for _ in range(n)]
    for i, j in basis:
        row_cells[i].append(j)
        column_cells[j].append(i)
    u = [math.nan] * m
    v = [math.nan] * n
    u[0] = 0.0
    stack: list[tuple[bool, int]] = [(True, 0)]
    while stack:
        is_row, index = stack.pop()
        if is_row:
            for j in row_cells[index]:
                if math.isnan(v[j]):
                    v[j] = matrix[index][j] - u[index]
                    stack.append((False, j))
        else:
            for i in column_cells[index]:
                if math.isnan(u[i]):
                    u[i] = matrix[i][index] - v[index]
                    stack.append((True, i))
    if any(math.isnan(value) for value in u) or any(math.isnan(x) for x in v):
        raise ArithmeticError("transport basis is not a spanning tree")
    return u, v


def _tree_path(
    basis: Sequence[tuple[int, int]], m: int, n: int, row: int, column: int
) -> list[tuple[int, int]]:
    """Return the basic cells on the tree path from ``row`` to ``column``.

    Args:
        basis: Basic cells forming a spanning tree.
        m: Number of rows.
        n: Number of columns.
        row: Start row index.
        column: End column index.

    Returns:
        Cells ordered from the start row to the end column.

    Raises:
        ArithmeticError: If the column is unreachable.

    Equation:
        [EQ-OT-SIMPLEX-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 3.5].

    """
    adjacency: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in range(m + n)]
    for i, j in basis:
        adjacency[i].append((m + j, (i, j)))
        adjacency[m + j].append((i, (i, j)))
    parent: dict[int, tuple[int, tuple[int, int]]] = {}
    target = m + column
    stack = [row]
    seen = {row}
    while stack:
        node = stack.pop()
        if node == target:
            break
        for neighbour, cell in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                parent[neighbour] = (node, cell)
                stack.append(neighbour)
    if target not in seen:
        raise ArithmeticError("entering cell is not connected to the basis")
    path: list[tuple[int, int]] = []
    node = target
    while node != row:
        previous, cell = parent[node]
        path.append(cell)
        node = previous
    path.reverse()
    return path


def exact_transport(
    source_weights: Iterable[object],
    target_weights: Iterable[object],
    cost: Iterable[Iterable[object]],
    *,
    tolerance: float = 1e-12,
    max_iterations: int | None = None,
) -> TransportPlan:
    """Solve the discrete Kantorovich problem exactly by the transportation simplex.

    The north-west corner rule gives a basic feasible plan whose
    ``m + n - 1`` basic cells form a spanning tree. Each iteration computes
    potentials with ``u_i + v_j = C_ij`` on the tree; if every reduced cost
    ``C_ij - u_i - v_j`` is at least ``-tolerance * max(1, max|C|)`` the plan
    is optimal. Otherwise the most negative cell enters, mass ``theta`` is
    shifted around the unique cycle, and a minimizing cell leaves. After
    ``m + n`` consecutive degenerate pivots the solver switches to Bland's
    smallest-index rule, which cannot cycle.

    Args:
        source_weights: Non-negative source weights ``a``.
        target_weights: Non-negative target weights ``b`` with the same total
            mass (relative tolerance ``1e-9``).
        cost: Finite ``m x n`` cost matrix.
        tolerance: Relative reduced-cost tolerance.
        max_iterations: Pivot limit; defaults to ``50 m n + 100``.

    Returns:
        Optimal plan, cost, dual potentials, and final basis.

    Raises:
        ValueError: If inputs are invalid.
        ArithmeticError: If optimality is not reached within the pivot limit.

    Equation:
        [EQ-OT-LP-001], [EQ-OT-SIMPLEX-001].

    References:
        [REF-DANTZIG-1951], [REF-BLAND-1977],
        [REF-PEYRE-CUTURI-2019, Sections 3.3-3.5].

    """
    a, b, matrix = _problem(source_weights, target_weights, cost)
    relative = require_finite_real(tolerance, name="tolerance")
    if not relative > 0.0:
        raise ValueError("tolerance must be positive")
    m, n = len(a), len(b)
    limit = 50 * m * n + 100 if max_iterations is None else max_iterations
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("max_iterations must be a non-negative integer")
    threshold = relative * max(1.0, max(abs(x) for row in matrix for x in row))
    flow, basis = _north_west_corner(a, b)
    in_basis = set(basis)
    degenerate_run = 0
    bland = False
    iterations = 0
    while True:
        u, v = _potentials(matrix, basis, m, n)
        entering: tuple[int, int] | None = None
        best = -threshold
        for i in range(m):
            for j in range(n):
                if (i, j) in in_basis:
                    continue
                reduced = matrix[i][j] - u[i] - v[j]
                if reduced < best:
                    entering = (i, j)
                    if bland:
                        break
                    best = reduced
            if bland and entering is not None:
                break
        if entering is None:
            break
        if iterations >= limit:
            raise ArithmeticError("transportation simplex did not reach optimality")
        iterations += 1
        path = _tree_path(basis, m, n, entering[0], entering[1])
        minus = path[0::2]
        plus = path[1::2]
        theta = min(flow[i][j] for i, j in minus)
        candidates = [cell for cell in minus if flow[cell[0]][cell[1]] <= theta]
        leaving = min(candidates) if bland else candidates[0]
        for i, j in minus:
            flow[i][j] -= theta
        for i, j in plus:
            flow[i][j] += theta
        flow[entering[0]][entering[1]] += theta
        flow[leaving[0]][leaving[1]] = 0.0
        basis[basis.index(leaving)] = entering
        in_basis.discard(leaving)
        in_basis.add(entering)
        if theta <= 0.0:
            degenerate_run += 1
            if degenerate_run > m + n:
                bland = True
        else:
            degenerate_run = 0
    plan = tuple(tuple(max(0.0, x) for x in row) for row in flow)
    total = math.fsum(matrix[i][j] * plan[i][j] for i in range(m) for j in range(n))
    return TransportPlan(
        plan=plan,
        cost=total,
        source_potential=tuple(u),
        target_potential=tuple(v),
        basis=tuple(sorted(basis)),
        iterations=iterations,
        source_weights=a,
        target_weights=b,
        cost_matrix=matrix,
    )


def discrete_wasserstein(
    source_points: Iterable[object],
    source_weights: Iterable[object],
    target_points: Iterable[object],
    target_weights: Iterable[object],
    *,
    p: float = 1.0,
) -> float:
    """Return ``W_p = (min_P sum_ij |x_i - y_j|^p P_ij)^(1/p)`` exactly.

    Args:
        source_points: Scalars or vectors ``x_i``.
        source_weights: Non-negative weights of the ``x_i``.
        target_points: Scalars or vectors ``y_j`` of the same dimension.
        target_weights: Non-negative weights with the same total mass.
        p: Exponent ``p >= 1``.

    Returns:
        The Wasserstein-``p`` distance between the two discrete measures.

    Raises:
        ValueError: If inputs are invalid or ``p < 1``.

    Equation:
        [EQ-OT-COST-001], [EQ-OT-LP-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 2.4], [REF-VILLANI-2009, Chapter 6].

    """
    exponent = require_finite_real(p, name="p")
    if exponent < 1.0:
        raise ValueError("p must be at least 1")
    cost = power_cost_matrix(source_points, target_points, p=exponent)
    result = exact_transport(source_weights, target_weights, cost)
    return float(max(result.cost, 0.0) ** (1.0 / exponent))


@dataclass(frozen=True, slots=True)
class SinkhornResult:
    """Output of log-domain Sinkhorn iterations.

    Args:
        plan: Entropic coupling ``P_ij = a_i b_j exp((f_i + g_j - C_ij)/eps)``.
        transport_cost: ``sum_ij C_ij P_ij``.
        regularized_cost: Dual value ``<a, f> + <b, g> - eps <a b',
            exp((f + g - C)/eps) - 1>``, equal at convergence to
            ``min_P <P, C> + eps KL(P || a b')``.
        source_potential: Dual potential ``f``.
        target_potential: Dual potential ``g``.
        epsilon: Regularization strength.
        iterations: Number of ``(g, f)`` update pairs over all stages.
        converged: Whether the marginal error reached the tolerance.
        marginal_error: Final ``sum_i |(P 1)_i - a_i|`` (columns are exact).
        trace: Marginal error after every iteration.

    References:
        [REF-CUTURI-2013], [REF-PEYRE-CUTURI-2019, Section 4.4].

    """

    plan: Matrix
    transport_cost: float
    regularized_cost: float
    source_potential: Vector
    target_potential: Vector
    epsilon: float
    iterations: int
    converged: bool
    marginal_error: float
    trace: Vector


def _logsumexp(values: Sequence[float]) -> float:
    """Return ``log sum exp(values)`` stably, ``-inf`` for all ``-inf``.

    Args:
        values: Finite numbers or ``-inf``.

    Returns:
        The log-sum-exp.

    Equation:
        [EQ-OT-SINK-001].

    References:
        [REF-PEYRE-CUTURI-2019, Section 4.4].

    """
    largest = max(values)
    if largest == -math.inf:
        return -math.inf
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def sinkhorn(
    source_weights: Iterable[object],
    target_weights: Iterable[object],
    cost: Iterable[Iterable[object]],
    *,
    epsilon: float,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
    initial_target_potential: Iterable[object] | None = None,
    epsilon_scaling: float | None = None,
) -> SinkhornResult:
    """Solve entropic optimal transport by log-domain Sinkhorn iterations.

    With unit-mass weights the updates are the soft c-transforms
    ``f_i = -eps log sum_j b_j exp((g_j - C_ij)/eps)`` and
    ``g_j = -eps log sum_i a_i exp((f_i - C_ij)/eps)``, evaluated with
    log-sum-exp so no kernel ``exp(-C/eps)`` underflows. After each ``g``
    update the column marginals are exact; the row marginal error
    ``sum_i a_i |exp((f_i - f_i^new)/eps) - 1|`` is the stopping criterion.
    Weights of any common total mass ``M`` are normalized internally; the plan
    and costs are scaled back by ``M``. Zero-mass atoms receive zero plan mass
    and the natural extension of the potentials.

    Args:
        source_weights: Non-negative source weights ``a``.
        target_weights: Non-negative target weights ``b`` of equal mass.
        cost: Finite ``m x n`` cost matrix.
        epsilon: Positive regularization in cost units.
        tolerance: Positive tolerance on the relative ``L1`` marginal error.
        max_iterations: Maximum number of ``(g, f)`` update pairs.
        initial_target_potential: Optional warm start for ``g``.
        epsilon_scaling: Optional factor in ``(0, 1)``. When given, the
            problem is first solved loosely (marginal error ``1e-3``) at
            ``max|C|, max|C| * factor, ...`` and each stage warm-starts the
            next, ending at ``epsilon``; the trace spans all stages.

    Returns:
        Plan, costs, potentials, and convergence trace.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        [EQ-OT-SINK-001].

    References:
        [REF-CUTURI-2013], [REF-PEYRE-CUTURI-2019, Sections 4.2 and 4.4].

    """
    a, b, matrix = _problem(source_weights, target_weights, cost)
    eps = require_finite_real(epsilon, name="epsilon")
    if not eps > 0.0:
        raise ValueError("epsilon must be positive")
    stop = require_finite_real(tolerance, name="tolerance")
    if not stop > 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    m, n = len(a), len(b)
    mass = math.fsum(a)
    pa = tuple(x / mass for x in a)
    pb = tuple(x / math.fsum(b) for x in b)
    log_a = tuple(math.log(x) if x > 0.0 else -math.inf for x in pa)
    log_b = tuple(math.log(x) if x > 0.0 else -math.inf for x in pb)
    if initial_target_potential is None:
        g = [0.0] * n
    else:
        g = [
            require_finite_real(x, name="initial_target_potential")
            for x in initial_target_potential
        ]
        if len(g) != n:
            raise ValueError("initial_target_potential must have len(target) entries")

    def update_f(potential: Sequence[float], level: float) -> list[float]:
        """Return the soft c-transform of ``g`` at regularization ``level``.

        References:
            [EQ-OT-SINK-001], [REF-PEYRE-CUTURI-2019, Section 4.4].

        """
        return [
            -level
            * _logsumexp(
                [log_b[j] + (potential[j] - matrix[i][j]) / level for j in range(n)]
            )
            for i in range(m)
        ]

    def update_g(potential: Sequence[float], level: float) -> list[float]:
        """Return the soft c-transform of ``f`` at regularization ``level``.

        References:
            [EQ-OT-SINK-001], [REF-PEYRE-CUTURI-2019, Section 4.4].

        """
        return [
            -level
            * _logsumexp(
                [log_a[i] + (potential[i] - matrix[i][j]) / level for i in range(m)]
            )
            for j in range(n)
        ]

    levels: list[float] = []
    if epsilon_scaling is not None:
        factor = require_finite_real(epsilon_scaling, name="epsilon_scaling")
        if not 0.0 < factor < 1.0:
            raise ValueError("epsilon_scaling must lie strictly between 0 and 1")
        level = max(abs(x) for row in matrix for x in row)
        while level > eps:
            levels.append(level)
            level *= factor
    levels.append(eps)
    trace: list[float] = []
    converged = False
    error = math.inf
    iterations = 0
    used = levels[0]
    f: list[float] = []
    for stage, level in enumerate(levels):
        used = level
        final = stage == len(levels) - 1
        target = stop if final else max(stop, _STAGE_TOLERANCE)
        f = update_f(g, level)
        while iterations < max_iterations:
            iterations += 1
            g = update_g(f, level)
            f_next = update_f(g, level)
            error = math.fsum(
                pa[i] * abs(math.expm1((f[i] - f_next[i]) / level))
                for i in range(m)
                if pa[i] > 0.0
            )
            trace.append(error)
            if error <= target:
                converged = final
                break
            f = f_next
        if not error <= target:
            break
    plan = tuple(
        tuple(
            mass * pa[i] * pb[j] * math.exp((f[i] + g[j] - matrix[i][j]) / used)
            if pa[i] > 0.0 and pb[j] > 0.0
            else 0.0
            for j in range(n)
        )
        for i in range(m)
    )
    transport_cost = math.fsum(
        matrix[i][j] * plan[i][j] for i in range(m) for j in range(n)
    )
    plan_mass = math.fsum(value for row in plan for value in row) / mass
    dual = (
        math.fsum(pa[i] * f[i] for i in range(m))
        + math.fsum(pb[j] * g[j] for j in range(n))
        - used * (plan_mass - 1.0)
    )
    return SinkhornResult(
        plan=plan,
        transport_cost=transport_cost,
        regularized_cost=mass * dual,
        source_potential=tuple(f),
        target_potential=tuple(g),
        epsilon=eps,
        iterations=iterations,
        converged=converged,
        marginal_error=mass * error,
        trace=tuple(trace),
    )


def _self_transport_cost(
    weights: Vector,
    cost: Matrix,
    *,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Return ``OT_eps(alpha, alpha)`` by the averaged symmetric Sinkhorn map.

    For a symmetric problem the optimal potentials coincide, ``f = g``, and
    the damped update ``f <- (f + T(f))/2`` with the soft c-transform
    ``T(f)_i = -eps log sum_j a_j exp((f_j - C_ij)/eps)`` avoids the slow
    oscillation of plain alternating updates. At the fixed point
    ``OT_eps(alpha, alpha) = 2 <a, f>`` for unit mass.

    Args:
        weights: Non-negative weights with positive mass.
        cost: Symmetric square cost matrix.
        epsilon: Positive regularization.
        tolerance: Stopping tolerance on ``max_i |f_i - T(f)_i| / eps``.
        max_iterations: Iteration limit.

    Returns:
        The regularized self-transport cost, scaled by the total mass.

    Raises:
        ArithmeticError: If the iteration does not converge.

    Equation:
        [EQ-OT-SINK-002].

    References:
        [REF-FEYDY-ETAL-2019].

    """
    mass = math.fsum(weights)
    size = len(weights)
    probabilities = tuple(x / mass for x in weights)
    log_a = tuple(math.log(x) if x > 0.0 else -math.inf for x in probabilities)

    def transform(potential: Sequence[float]) -> list[float]:
        """Return the symmetric soft c-transform.

        References:
            [EQ-OT-SINK-002], [REF-FEYDY-ETAL-2019].

        """
        return [
            -epsilon
            * _logsumexp(
                [log_a[j] + (potential[j] - cost[i][j]) / epsilon for j in range(size)]
            )
            for i in range(size)
        ]

    f = transform([0.0] * size)
    for _ in range(max_iterations):
        image = transform(f)
        change = max(
            abs(f[i] - image[i]) for i in range(size) if probabilities[i] > 0.0
        )
        if change <= tolerance * epsilon:
            f = image
            return (
                mass
                * 2.0
                * math.fsum(
                    probabilities[i] * f[i] for i in range(size) if probabilities[i] > 0
                )
            )
        f = [0.5 * (f[i] + image[i]) for i in range(size)]
    raise ArithmeticError("symmetric Sinkhorn iteration did not converge")


def sinkhorn_divergence(
    source_points: Iterable[object],
    source_weights: Iterable[object],
    target_points: Iterable[object],
    target_weights: Iterable[object],
    *,
    epsilon: float,
    p: float = 2.0,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
) -> float:
    """Return the debiased Sinkhorn divergence between two point clouds.

    ``S_eps(alpha, beta) = OT_eps(alpha, beta) - OT_eps(alpha, alpha)/2
    - OT_eps(beta, beta)/2``, where ``OT_eps`` is the regularized cost of
    :func:`sinkhorn` with cost ``|x - y|^p``. The debiasing removes the
    entropic bias, so ``S_eps(alpha, alpha) = 0``; it tends to the exact
    transport cost as ``eps -> 0`` and to an MMD as ``eps -> infinity``.

    Args:
        source_points: Scalars or vectors of the first measure.
        source_weights: Non-negative weights of the first measure.
        target_points: Scalars or vectors of the second measure.
        target_weights: Non-negative weights with the same total mass.
        epsilon: Positive regularization in cost units.
        p: Positive cost exponent.
        tolerance: Sinkhorn marginal tolerance.
        max_iterations: Sinkhorn iteration limit per subproblem.

    Returns:
        The Sinkhorn divergence.

    Raises:
        ValueError: If inputs are invalid.
        ArithmeticError: If a Sinkhorn subproblem does not converge.

    Equation:
        [EQ-OT-SINK-002].

    References:
        [REF-FEYDY-ETAL-2019], [REF-PEYRE-CUTURI-2019, Section 4.5].

    """
    xs = _points(source_points, name="source_points")
    ys = _points(target_points, name="target_points")
    a = _weights(source_weights, name="source_weights")
    b = _weights(target_weights, name="target_weights")
    if len(xs) != len(a) or len(ys) != len(b):
        raise ValueError("each point cloud needs one weight per point")
    cross = sinkhorn(
        a,
        b,
        power_cost_matrix(xs, ys, p=p),
        epsilon=epsilon,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not cross.converged:
        raise ArithmeticError("the cross Sinkhorn problem did not converge")
    eps = cross.epsilon
    self_a = _self_transport_cost(
        a,
        power_cost_matrix(xs, xs, p=p),
        epsilon=eps,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    self_b = _self_transport_cost(
        b,
        power_cost_matrix(ys, ys, p=p),
        epsilon=eps,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return cross.regularized_cost - 0.5 * self_a - 0.5 * self_b
