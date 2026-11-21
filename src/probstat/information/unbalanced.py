"""Unbalanced entropic optimal transport with KL-relaxed marginals.

Balanced transport requires equal total masses and forces every unit of
mass to move. The unbalanced problem replaces the two marginal constraints
by Kullback-Leibler penalties, so mass may be created or destroyed at a
price ``rho`` and the two measures may have different totals:

``min_{P >= 0} <C, P> + eps KL(P | a b') + rho KL(P 1 | a) + rho KL(P' 1 | b)``

with the generalized divergence
``KL(p | q) = sum_i (p_i log(p_i / q_i) - p_i + q_i)``. The minimizer has
the Gibbs form ``P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps)`` and the
potentials solve a damped Sinkhorn fixed point, iterated here in the log
domain like :func:`~probstat.information.sinkhorn`.

Convention: the entropy term is taken relative to the product ``a b'``, as
in :func:`~probstat.information.sinkhorn`. The scaling iteration of the
cited source is usually written for a fixed reference measure on the product
space (the counting measure for discrete problems),
``u = (a / (K v))^lambda``. The two objectives are not equivalent: because
``KL(P | 1) = KL(P | a b') + sum_ij P_ij log(a_i b_j) + const``, the
counting-measure problem is this problem with the cost
``C_ij + eps log(a_i b_j)``. In balanced transport that extra term is
constant on the feasible set; with relaxed marginals it is not, so the
optimal plan, its mass, and the objective value differ from those of
software that uses the counting-measure convention (masses 2.009 against
1.969 on a three-by-two example). To reproduce such a result pass the cost
``C_ij + eps log(a_i b_j)`` (positive weights only); the test suite checks
that equivalence against the scaling iteration.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations import require_finite_real
from probstat.linalg import Matrix, Vector, as_matrix

from .transport import _logsumexp, _weights


def _generalized_kl(p: Sequence[float], q: Sequence[float]) -> float:
    """Return ``sum_i (p_i log(p_i/q_i) - p_i + q_i)`` with ``0 log 0 = 0``.

    Args:
        p: Non-negative vector.
        q: Non-negative reference vector with ``q_i > 0`` where ``p_i > 0``.

    Returns:
        The non-negative generalized Kullback-Leibler divergence, ``inf``
        when ``p`` has mass where ``q`` has none.

    Equation:
        [EQ-OT-UB-001].

    References:
        [REF-CHIZAT-ETAL-2018].

    """
    terms: list[float] = []
    for u, v in zip(p, q, strict=True):
        if u > 0.0:
            if not v > 0.0:
                return math.inf
            terms.append(u * math.log(u / v) - u + v)
        else:
            terms.append(v)
    return math.fsum(terms)


@dataclass(frozen=True, slots=True)
class UnbalancedSinkhornResult:
    """Output of the unbalanced Sinkhorn (scaling) iterations.

    Args:
        plan: ``P_ij = a_i b_j exp((f_i + g_j - C_ij)/eps)``.
        source_marginal: Row sums ``P 1``.
        target_marginal: Column sums ``P' 1``.
        transported_mass: ``sum_ij P_ij``.
        transport_cost: ``<C, P>``.
        primal_objective: ``<C, P> + eps KL(P | a b') + rho KL(P1 | a) +
            rho KL(P'1 | b)`` at the returned plan.
        dual_objective: ``rho sum_i a_i (1 - exp(-f_i/rho)) + rho sum_j b_j
            (1 - exp(-g_j/rho)) - eps sum_ij a_i b_j (exp((f_i + g_j -
            C_ij)/eps) - 1)`` at the returned potentials; a lower bound on
            the minimum, equal to ``primal_objective`` at the solution.
        source_potential: Dual potential ``f``.
        target_potential: Dual potential ``g``.
        epsilon: Entropic regularization.
        marginal_penalty: Marginal relaxation ``rho``.
        iterations: Number of ``(g, f)`` update pairs.
        converged: Whether the largest change of a potential in the last
            iteration was at most the tolerance.
        fixed_point_residual: ``max(|f - lambda T(g)|, |g - lambda T(f)|)``
            at the returned potentials (cost units).
        optimality_residual: ``max(|f_i + rho log((P1)_i/a_i)|,
            |g_j + rho log((P'1)_j/b_j)|)`` over atoms with positive weight,
            the violation of the first-order conditions of the primal
            objective; it equals ``(1 + rho/eps)`` times the per-equation
            fixed-point residual, so it is large relative to
            ``fixed_point_residual`` when ``rho >> eps``.
        trace: Largest potential change after every iteration.

    References:
        [EQ-OT-UB-001], [EQ-OT-UB-002], [EQ-OT-UB-003],
        [REF-CHIZAT-ETAL-2018].

    """

    plan: Matrix
    source_marginal: Vector
    target_marginal: Vector
    transported_mass: float
    transport_cost: float
    primal_objective: float
    dual_objective: float
    source_potential: Vector
    target_potential: Vector
    epsilon: float
    marginal_penalty: float
    iterations: int
    converged: bool
    fixed_point_residual: float
    optimality_residual: float
    trace: Vector

    @property
    def duality_gap(self) -> float:
        """Return ``primal_objective - dual_objective`` (non-negative in theory).

        References:
            [EQ-OT-UB-003], [REF-CHIZAT-ETAL-2018].

        """
        return self.primal_objective - self.dual_objective


def unbalanced_sinkhorn(
    source_weights: Iterable[object],
    target_weights: Iterable[object],
    cost: Iterable[Iterable[object]],
    *,
    epsilon: float,
    marginal_penalty: float,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
    translation_step: bool = True,
) -> UnbalancedSinkhornResult:
    """Solve unbalanced entropic transport by damped log-domain Sinkhorn.

    Objective ([EQ-OT-UB-001]), with ``rho = marginal_penalty``:
    ``min_{P >= 0} <C, P> + eps KL(P | a b') + rho KL(P1 | a) +
    rho KL(P'1 | b)``, ``KL(p | q) = sum (p log(p/q) - p + q)``. It is
    strictly convex, so the minimizer is unique.

    Derivation of the update ([EQ-OT-UB-002]). Setting the derivative with
    respect to ``P_ij`` to zero gives
    ``C_ij + eps log(P_ij/(a_i b_j)) + rho log((P1)_i/a_i)
    + rho log((P'1)_j/b_j) = 0``, hence
    ``P_ij = a_i b_j exp((f_i + g_j - C_ij)/eps)`` with
    ``f_i = -rho log((P1)_i/a_i)`` and ``g_j = -rho log((P'1)_j/b_j)``.
    Inserting the row sum
    ``(P1)_i = a_i exp(f_i/eps) sum_j b_j exp((g_j - C_ij)/eps)`` into the
    equation for ``f_i`` and solving gives
    ``f_i = lambda T(g)_i`` with ``lambda = rho/(rho + eps)`` and the
    balanced soft c-transform
    ``T(g)_i = -eps log sum_j b_j exp((g_j - C_ij)/eps)``; symmetrically
    ``g_j = lambda T(f)_j``. In scaling variables ``u = exp(f/eps)``,
    ``v = exp(g/eps)`` and ``K_ij = exp(-C_ij/eps)`` this is
    ``u = (1 / K(b v))^lambda``, ``v = (1 / K'(a u))^lambda``; if the
    entropy is taken relative to the counting measure instead of ``a b'``
    the same steps give the form ``u = (a / (K v))^lambda``. As
    ``rho -> inf``, ``lambda -> 1`` and the iteration becomes balanced
    Sinkhorn.

    Each half-step is ``lambda`` times a map that is non-expansive in the
    maximum norm, so the iteration contracts with factor at most
    ``lambda^2`` per pair of updates. For ``rho >> eps`` that factor is
    close to one because of the direction ``(f + c, g - c)``, which leaves
    the plan unchanged. With ``translation_step=True`` (default) each
    iteration ends with the exact maximization of the dual objective along
    that direction, ``c = (rho/2) log(sum_i a_i exp(-f_i/rho) /
    sum_j b_j exp(-g_j/rho))`` (set the derivative of the dual in ``c`` to
    zero); a fixed point has ``c = 0``, so the solution is unchanged. This
    step is this library's addition to the plain scaling iteration.

    The iteration stops when the largest change of any potential in one
    iteration is at most ``tolerance`` (cost units). The exact fixed-point
    and first-order residuals of the returned potentials are reported
    separately and should be checked when ``rho/eps`` is large. Zero-weight
    atoms get zero plan mass.

    Args:
        source_weights: Non-negative source weights ``a`` (any positive
            total mass).
        target_weights: Non-negative target weights ``b`` (any positive
            total mass, not necessarily that of ``a``).
        cost: Finite ``m x n`` cost matrix.
        epsilon: Positive entropic regularization in cost units.
        marginal_penalty: Positive ``rho`` in cost units; large values
            enforce the marginals, small values allow mass to vanish.
        tolerance: Positive tolerance on the potential change.
        max_iterations: Maximum number of ``(g, f)`` update pairs.
        translation_step: Apply the exact dual line search along
            ``(f + c, g - c)`` after each iteration.

    Returns:
        Plan, marginals, mass, objectives, potentials, and diagnostics.

    Raises:
        TypeError: If ``max_iterations`` is not an integer.
        ValueError: If weights, cost, or parameters are invalid.

    Equation:
        [EQ-OT-UB-001], [EQ-OT-UB-002], [EQ-OT-UB-003].

    References:
        [REF-CHIZAT-ETAL-2018], [REF-PEYRE-CUTURI-2019].

    """
    a = _weights(source_weights, name="source_weights")
    b = _weights(target_weights, name="target_weights")
    matrix = as_matrix(cost, name="cost")
    m, n = len(a), len(b)
    if len(matrix) != m or len(matrix[0]) != n:
        raise ValueError("cost must have len(source) rows and len(target) columns")
    eps = require_finite_real(epsilon, name="epsilon")
    if not eps > 0.0:
        raise ValueError("epsilon must be positive")
    rho = require_finite_real(marginal_penalty, name="marginal_penalty")
    if not rho > 0.0:
        raise ValueError("marginal_penalty must be positive")
    stop = require_finite_real(tolerance, name="tolerance")
    if not stop > 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    damping = rho / (rho + eps)
    log_a = tuple(math.log(x) if x > 0.0 else -math.inf for x in a)
    log_b = tuple(math.log(x) if x > 0.0 else -math.inf for x in b)

    def transform_g(potential: Sequence[float]) -> list[float]:
        """Return ``T(g)_i = -eps log sum_j b_j exp((g_j - C_ij)/eps)``.

        References:
            [EQ-OT-UB-002], [REF-CHIZAT-ETAL-2018].

        """
        return [
            -eps
            * _logsumexp(
                [log_b[j] + (potential[j] - matrix[i][j]) / eps for j in range(n)]
            )
            for i in range(m)
        ]

    def transform_f(potential: Sequence[float]) -> list[float]:
        """Return ``T(f)_j = -eps log sum_i a_i exp((f_i - C_ij)/eps)``.

        References:
            [EQ-OT-UB-002], [REF-CHIZAT-ETAL-2018].

        """
        return [
            -eps
            * _logsumexp(
                [log_a[i] + (potential[i] - matrix[i][j]) / eps for i in range(m)]
            )
            for j in range(n)
        ]

    def shift(f_values: Sequence[float], g_values: Sequence[float]) -> float:
        """Return the dual-optimal translation ``c`` along ``(f+c, g-c)``.

        References:
            [EQ-OT-UB-003], [REF-CHIZAT-ETAL-2018].

        """
        left = _logsumexp([log_a[i] - f_values[i] / rho for i in range(m)])
        right = _logsumexp([log_b[j] - g_values[j] / rho for j in range(n)])
        return 0.5 * rho * (left - right)

    f = [0.0] * m
    g = [0.0] * n
    trace: list[float] = []
    converged = False
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        g_new = [damping * value for value in transform_f(f)]
        f_new = [damping * value for value in transform_g(g_new)]
        if translation_step:
            c = shift(f_new, g_new)
            f_new = [value + c for value in f_new]
            g_new = [value - c for value in g_new]
        change = max(
            max(abs(x - y) for x, y in zip(f_new, f, strict=True)),
            max(abs(x - y) for x, y in zip(g_new, g, strict=True)),
        )
        f, g = f_new, g_new
        trace.append(change)
        if change <= stop:
            converged = True
            break
    tf = transform_g(g)
    tg = transform_f(f)
    fixed_point = max(
        max(abs(f[i] - damping * tf[i]) for i in range(m)),
        max(abs(g[j] - damping * tg[j]) for j in range(n)),
    )
    plan = tuple(
        tuple(
            a[i] * b[j] * math.exp((f[i] + g[j] - matrix[i][j]) / eps)
            if a[i] > 0.0 and b[j] > 0.0
            else 0.0
            for j in range(n)
        )
        for i in range(m)
    )
    rows = tuple(math.fsum(row) for row in plan)
    columns = tuple(math.fsum(plan[i][j] for i in range(m)) for j in range(n))
    optimality = max(
        max(
            (
                abs(f[i] + rho * math.log(rows[i] / a[i]))
                for i in range(m)
                if a[i] > 0.0
            ),
            default=0.0,
        ),
        max(
            (
                abs(g[j] + rho * math.log(columns[j] / b[j]))
                for j in range(n)
                if b[j] > 0.0
            ),
            default=0.0,
        ),
    )
    transport_cost = math.fsum(
        matrix[i][j] * plan[i][j] for i in range(m) for j in range(n)
    )
    mass = math.fsum(rows)
    reference_mass = math.fsum(a) * math.fsum(b)
    entropy = (
        math.fsum(
            plan[i][j] * (f[i] + g[j] - matrix[i][j]) / eps
            for i in range(m)
            for j in range(n)
            if plan[i][j] > 0.0
        )
        - mass
        + reference_mass
    )
    primal = (
        transport_cost
        + eps * entropy
        + rho * _generalized_kl(rows, a)
        + rho * _generalized_kl(columns, b)
    )
    dual = (
        -rho * math.fsum(a[i] * math.expm1(-f[i] / rho) for i in range(m))
        - rho * math.fsum(b[j] * math.expm1(-g[j] / rho) for j in range(n))
        - eps * (mass - reference_mass)
    )
    return UnbalancedSinkhornResult(
        plan=plan,
        source_marginal=rows,
        target_marginal=columns,
        transported_mass=mass,
        transport_cost=transport_cost,
        primal_objective=primal,
        dual_objective=dual,
        source_potential=tuple(f),
        target_potential=tuple(g),
        epsilon=eps,
        marginal_penalty=rho,
        iterations=iterations,
        converged=converged,
        fixed_point_residual=fixed_point,
        optimality_residual=optimality,
        trace=tuple(trace),
    )
