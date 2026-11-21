"""Multilevel best linear unbiased estimators and a multifidelity comparison.

Models ``f_0, ..., f_(L-1)`` of one random input have unknown means ``mu`` and
a known (or pilot) covariance ``C``. A *model group* ``S_k`` is a subset of the
models that is evaluated together on ``n_k`` shared inputs; different groups
use independent inputs. Every group sample is an observation
``Y_k = P_k mu + noise`` with ``Cov = C_k`` (the rows and columns ``S_k`` of
``C``), so generalized least squares gives the best linear unbiased estimator
of ``mu`` (Schaden and Ullmann):

``Psi = sum_k n_k P_k' C_k^-1 P_k``,
``y = sum_k P_k' C_k^-1 sum_i Y_k^(i)``, ``mu_hat = Psi^-1 y``,
``Cov(mu_hat) = Psi^-1``.

The module provides the variance prediction, the estimator, a sample
allocation for a cost budget, and a table that compares plain Monte Carlo,
MFMC, ACV-MF, and MLBLUE for a given covariance, cost vector, and budget.

The allocation problem ``min [Psi(n)^-1]_00`` subject to
``sum_k n_k W_k <= p``, ``n >= 0`` is the one stated by Croci, Willcox and
Wright, who solve it as a semidefinite program. No SDP solver is available in
this library: the continuous relaxation is solved here as a smooth nonlinear
program on the budget simplex with BFGS, its optimality is checked with a
lower bound derived from the generalized-least-squares form of the variance,
and the result is rounded heuristically. Model 0 is the high-fidelity model.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, inverse_spd
from probstat.optimization import minimize_bfgs

from .approximate_control_variates import acv_allocation
from .multifidelity import (
    InputSampler,
    ModelFunction,
    MultifidelityEstimate,
    _as_budget,
    _as_costs,
    _as_count,
    _as_covariance,
    _draw_inputs,
    _evaluate,
    _mean,
    _solve_spd,
    select_mfmc_models,
)

Groups = tuple[tuple[int, ...], ...]

_MAX_GROUPS = 4095


@dataclass(frozen=True, slots=True)
class MLBLUEAllocation:
    """Sample allocation of a multilevel best linear unbiased estimator.

    Sequences are aligned with ``groups``.

    Args:
        groups: Model groups ``S_k`` (sorted tuples of model indices).
        group_costs: Cost ``W_k = sum_(i in S_k) w_i`` of one group sample.
        shares: Budget shares ``theta_k = n_k W_k / p`` of the continuous
            solution (they sum to one).
        continuous_samples: Real sample counts ``n_k = p theta_k / W_k``.
        continuous_variance: ``[Psi(n)^-1]_00`` at ``continuous_samples``.
        lower_bound: Certified lower bound on the optimal continuous
            variance; ``continuous_variance - lower_bound`` bounds the
            suboptimality of the relaxation.
        samples: Rounded integer sample counts within the budget.
        variance: ``[Psi(n)^-1]_00`` at the integer ``samples``.
        mc_variance: Variance ``C_00 w_0 / p`` of plain Monte Carlo with the
            whole budget (real-valued sample size).
        cost: Cost ``sum_k n_k W_k`` of the integer allocation.
        budget: Cost budget ``p``.
        evaluations: Evaluations of every model under ``samples``.
        converged: Whether the relative gap
            ``1 - lower_bound / continuous_variance`` is at most the
            requested tolerance.
        iterations: BFGS iterations over all restarts.
        message: Description of how the solver stopped.

    References:
        [REF-SCHADEN-ULLMANN-2020], [REF-CROCI-WILLCOX-WRIGHT-2023].

    """

    groups: Groups
    group_costs: Vector
    shares: Vector
    continuous_samples: Vector
    continuous_variance: float
    lower_bound: float
    samples: tuple[int, ...]
    variance: float
    mc_variance: float
    cost: float
    budget: float
    evaluations: tuple[int, ...]
    converged: bool
    iterations: int
    message: str

    @property
    def relative_gap(self) -> float:
        """Return ``1 - lower_bound / continuous_variance``.

        Equation:
            [EQ-MC-MF-010].

        References:
            [REF-SCHADEN-ULLMANN-2020].

        """
        return 1.0 - self.lower_bound / self.continuous_variance

    @property
    def variance_ratio(self) -> float:
        """Return ``variance / mc_variance`` (below one favors MLBLUE).

        Equation:
            [EQ-MC-MF-011].

        References:
            [REF-SCHADEN-ULLMANN-2020].

        """
        return self.variance / self.mc_variance


@dataclass(frozen=True, slots=True)
class MultifidelityComparisonRow:
    """Predicted performance of one estimator for a budget.

    Args:
        method: ``"MC"``, ``"MFMC"``, ``"ACV-MF"``, or ``"MLBLUE"``.
        variance: Predicted variance of the integer allocation.
        continuous_variance: Predicted variance of the real-valued
            allocation the integers were rounded from.
        variance_ratio: ``variance`` divided by the plain Monte Carlo
            variance of the same table.
        cost: Cost of the integer allocation (at most the budget).
        evaluations: Evaluations of every model.
        converged: Convergence flag of the allocation (always ``True`` for
            the closed-form MC and MFMC rows).

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016],
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020], [REF-SCHADEN-ULLMANN-2020].

    """

    method: str
    variance: float
    continuous_variance: float
    variance_ratio: float
    cost: float
    evaluations: tuple[int, ...]
    converged: bool


@dataclass(frozen=True, slots=True)
class MultifidelityComparison:
    """Predicted variances of several estimators for one budget.

    Args:
        budget: Cost budget ``p``.
        rows: One row per method in the order MC, MFMC, ACV-MF, MLBLUE.
        mfmc_models: Models selected for MFMC, in estimator order.
        mlblue_groups: Groups that receive samples in the integer MLBLUE
            allocation.

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016],
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020], [REF-SCHADEN-ULLMANN-2020].

    """

    budget: float
    rows: tuple[MultifidelityComparisonRow, ...]
    mfmc_models: tuple[int, ...]
    mlblue_groups: Groups

    def row(self, method: str) -> MultifidelityComparisonRow:
        """Return the row of one method.

        Args:
            method: ``"MC"``, ``"MFMC"``, ``"ACV-MF"``, or ``"MLBLUE"``.

        Returns:
            The matching row.

        Raises:
            ValueError: If the method is not in the table.

        References:
            [EQ-MC-MF-011], [REF-SCHADEN-ULLMANN-2020].

        """
        for item in self.rows:
            if item.method == method:
                return item
        raise ValueError(f"unknown method {method!r}")

    @property
    def best(self) -> MultifidelityComparisonRow:
        """Return the row with the smallest integer-allocation variance.

        References:
            [EQ-MC-MF-011], [REF-SCHADEN-ULLMANN-2020].

        """
        return min(self.rows, key=lambda item: item.variance)

    def format(self) -> str:
        """Return the table as aligned plain text.

        Returns:
            One header line and one line per method with the variance, the
            ratio to plain Monte Carlo, the cost, and the evaluations.

        References:
            [EQ-MC-MF-011], [REF-SCHADEN-ULLMANN-2020].

        """
        lines = [f"{'method':<8}{'variance':>13}{'ratio':>9}{'cost':>12}  evaluations"]
        for item in self.rows:
            flag = "" if item.converged else "  (not converged)"
            lines.append(
                f"{item.method:<8}{item.variance:>13.5e}{item.variance_ratio:>9.4f}"
                f"{item.cost:>12.6g}  {item.evaluations}{flag}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Groups and the normal matrix
# --------------------------------------------------------------------------


def mlblue_groups(models: int, *, max_size: int | None = None) -> Groups:
    """Return all non-empty model groups up to a size limit.

    Args:
        models: Number of models ``L`` (at least 1).
        max_size: Largest group size; default ``L`` (all ``2^L - 1`` groups).

    Returns:
        Groups as sorted index tuples, ordered by size and then
        lexicographically.

    Raises:
        ValueError: If an argument is not a positive integer or more than
            4095 groups would be produced.

    References:
        [REF-SCHADEN-ULLMANN-2020]. No equation is evaluated.

    """
    count = _as_count(models, name="models", minimum=1)
    limit = (
        count if max_size is None else _as_count(max_size, name="max_size", minimum=1)
    )
    limit = min(limit, count)
    total = sum(math.comb(count, size) for size in range(1, limit + 1))
    if total > _MAX_GROUPS:
        raise ValueError(
            f"{total} groups requested; at most {_MAX_GROUPS} are supported "
            "(lower max_size)"
        )
    return tuple(
        group
        for size in range(1, limit + 1)
        for group in itertools.combinations(range(count), size)
    )


def _as_groups(groups: Iterable[Iterable[int]], size: int) -> Groups:
    """Return validated model groups as sorted index tuples.

    Raises:
        ValueError: If there is no group, a group is empty or repeats a
            model, an index is out of range, or a group occurs twice.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    result: list[tuple[int, ...]] = []
    for raw in groups:
        members = tuple(raw)
        for index in members:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("groups must hold integer model indices")
            if not 0 <= index < size:
                raise ValueError(f"group index {index} is outside 0..{size - 1}")
        if not members or len(set(members)) != len(members):
            raise ValueError("every group must be non-empty without repeats")
        result.append(tuple(sorted(members)))
    if not result:
        raise ValueError("groups must name at least one model group")
    if len(set(result)) != len(result):
        raise ValueError("groups must be distinct")
    if len(result) > _MAX_GROUPS:
        raise ValueError(f"at most {_MAX_GROUPS} groups are supported")
    return tuple(result)


def _group_inverses(matrix: Matrix, groups: Groups) -> list[Matrix]:
    """Return ``C_k^-1`` for every group.

    References:
        [EQ-MC-MF-009], [REF-SCHADEN-ULLMANN-2020].

    """
    return [
        inverse_spd(tuple(tuple(matrix[i][j] for j in group) for i in group))
        for group in groups
    ]


def _blue_solve(
    size: int,
    groups: Groups,
    blocks: Sequence[Matrix],
    counts: Sequence[float],
) -> list[float]:
    """Return ``v = Psi^+ e_0``, the first column of the estimator covariance.

    ``Psi = sum_k n_k P_k' B_k P_k`` is assembled on the models covered by a
    group with ``n_k > 0`` (on those it is positive definite); the entries of
    uncovered models are zero. ``v[0]`` is the variance of the BLUE of
    ``mu_0``.

    Raises:
        ValueError: If no sampled group contains model 0 or the normal
            matrix is numerically singular.

    References:
        [EQ-MC-MF-009], [REF-SCHADEN-ULLMANN-2020].

    """
    covered = sorted(
        {i for group, n in zip(groups, counts, strict=True) if n > 0.0 for i in group}
    )
    if 0 not in covered:
        raise ValueError("no sampled group contains the high-fidelity model 0")
    position = {index: place for place, index in enumerate(covered)}
    psi = [[0.0] * len(covered) for _ in covered]
    for group, block, n in zip(groups, blocks, counts, strict=True):
        if n <= 0.0:
            continue
        places = [position[i] for i in group]
        for a, row in zip(places, block, strict=True):
            target = psi[a]
            for b, value in zip(places, row, strict=True):
                target[b] += n * value
    unit = [0.0] * len(covered)
    unit[0] = 1.0
    solution = _solve_spd(psi, unit)
    full = [0.0] * size
    for index, value in zip(covered, solution, strict=True):
        full[index] = value
    return full


def _as_group_counts(samples: Sequence[float], groups: Groups) -> list[float]:
    """Return validated non-negative group sample counts.

    Raises:
        ValueError: If the length is wrong or a count is negative.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    counts = [require_finite_real(value, name="samples") for value in samples]
    if len(counts) != len(groups):
        raise ValueError("samples must have one entry per group")
    if any(value < 0.0 for value in counts):
        raise ValueError("samples must be non-negative")
    return counts


def mlblue_variance(
    covariance: Iterable[Iterable[object]],
    groups: Iterable[Iterable[int]],
    samples: Sequence[float],
) -> float:
    """Return the variance of the MLBLUE of the high-fidelity mean.

    ``Var(mu_hat_0) = [Psi^-1]_00`` with
    ``Psi = sum_k n_k P_k' C_k^-1 P_k``, where ``P_k`` selects the models of
    group ``S_k`` and ``C_k = P_k C P_k'``. Models that no sampled group
    contains are left out of ``Psi``. The value is exact when ``C`` is the
    true covariance; no linear unbiased estimator that uses the same group
    samples has a smaller variance.

    Args:
        covariance: Model covariance matrix (high-fidelity model first).
        groups: Model groups ``S_k`` as index collections.
        samples: Non-negative sample counts ``n_k`` (real values are allowed
            for planning).

    Returns:
        ``[Psi^-1]_00``.

    Raises:
        ValueError: If no group with ``n_k > 0`` contains model 0, a group
            or count is invalid, or the covariance is not positive definite.

    Equation:
        [EQ-MC-MF-009].

    References:
        [REF-SCHADEN-ULLMANN-2020].

    """
    matrix = _as_covariance(covariance)
    sets = _as_groups(groups, len(matrix))
    counts = _as_group_counts(samples, sets)
    return _blue_solve(len(matrix), sets, _group_inverses(matrix, sets), counts)[0]


def mlblue_estimate(
    models: Sequence[ModelFunction],
    input_sampler: InputSampler,
    covariance: Iterable[Iterable[object]],
    groups: Iterable[Iterable[int]],
    samples: Sequence[int],
    *,
    costs: Sequence[float] | None = None,
    seed: int | RandomStream | None = None,
) -> MultifidelityEstimate:
    """Estimate the high-fidelity mean by a multilevel BLUE.

    Group ``k`` draws ``n_k`` inputs from ``root.child("mlblue-group", k)``
    and evaluates its models on them. With the group sums
    ``t_k = sum_i Y_k^(i)``,

    ``y = sum_k P_k' C_k^-1 t_k``, ``mu_hat = Psi^-1 y``,

    and the estimate is ``mu_hat_0``. The estimator is linear in the model
    outputs and unbiased for every fixed positive-definite ``covariance``;
    it is the *best* linear unbiased estimator, with variance
    ``[Psi^-1]_00``, when ``covariance`` is the true model covariance. A
    pilot covariance must come from a sample that is independent of the
    inputs drawn here.

    Args:
        models: Model callables, high-fidelity first.
        input_sampler: ``input_sampler(n, stream)`` returning ``n``
            independent inputs.
        covariance: Model covariance matrix.
        groups: Model groups ``S_k``.
        samples: Integer sample counts ``n_k >= 0``; at least one sampled
            group must contain model 0.
        costs: Optional costs of every model for the cost report.
        seed: Root seed or stream.

    Returns:
        Estimate with the predicted variance ``[Psi^-1]_00``.

    Raises:
        ValueError: If the inputs are inconsistent, model 0 is not sampled,
            the sampler returns the wrong number of inputs, or an output is
            not finite.

    Equation:
        [EQ-MC-MF-009].

    References:
        [REF-SCHADEN-ULLMANN-2020].

    """
    functions = tuple(models)
    matrix = _as_covariance(covariance)
    size = len(matrix)
    if len(functions) != size:
        raise ValueError("covariance must have one row per model")
    sets = _as_groups(groups, size)
    counts = tuple(samples)
    if len(counts) != len(sets):
        raise ValueError("samples must have one entry per group")
    for value in counts:
        _as_count(value, name="samples", minimum=0)
    weights = None if costs is None else _as_costs(costs, size)
    blocks = _group_inverses(matrix, sets)
    first_column = _blue_solve(size, sets, blocks, counts)
    root = resolve_stream(seed)
    right = [0.0] * size
    evaluations = [0] * size
    high_fidelity: list[float] = []
    for place, (group, block, count) in enumerate(
        zip(sets, blocks, counts, strict=True)
    ):
        if count == 0:
            continue
        inputs = _draw_inputs(input_sampler, count, root.child("mlblue-group", place))
        sums = []
        for index in group:
            values = _evaluate(functions[index], inputs, index)
            evaluations[index] += count
            sums.append(math.fsum(values))
            if index == 0:
                high_fidelity.extend(values)
        for index, row in zip(group, block, strict=True):
            right[index] += math.fsum(a * b for a, b in zip(row, sums, strict=True))
    # mu_hat_0 = e_0' Psi^-1 y = (Psi^-1 e_0)' y because Psi is symmetric.
    estimate = math.fsum(v * r for v, r in zip(first_column, right, strict=True))
    return MultifidelityEstimate(
        method="MLBLUE",
        estimate=estimate,
        variance=first_column[0],
        high_fidelity_mean=_mean(high_fidelity),
        high_fidelity_samples=len(high_fidelity),
        evaluations=tuple(evaluations),
        cost=None
        if weights is None
        else math.fsum(w * n for w, n in zip(weights, evaluations, strict=True)),
    )


# --------------------------------------------------------------------------
# Sample allocation
# --------------------------------------------------------------------------


class _ShareProblem:
    """Normalized MLBLUE variance as a function of budget shares.

    With ``theta`` on the simplex and ``n_k = p theta_k / W_k``,
    ``Var = (C_00 w_0 / p) g(theta)``, ``g(theta) = [Phi(theta)^-1]_00``,
    ``Phi = sum_k theta_k A_k`` and
    ``A_k = (w_0 C_00 / W_k) P_k' C_k^-1 P_k``. The shares are parametrized
    as ``theta_k = (1 - eps) z_k^2 / |z|^2 + eps / K`` so that every group
    keeps the share ``eps / K`` and ``Phi`` stays positive definite.

    References:
        [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020],
        [REF-CROCI-WILLCOX-WRIGHT-2023].

    """

    __slots__ = ("size", "groups", "blocks", "floor", "last")

    def __init__(
        self, matrix: Matrix, groups: Groups, group_costs: Vector, floor: float
    ) -> None:
        """Precompute the scaled group information matrices ``A_k``.

        References:
            [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020].

        """
        self.size = len(matrix)
        self.groups = groups
        self.floor = floor
        self.last: tuple[Vector, tuple[float, list[float]]] | None = None
        scale = matrix[0][0]
        self.blocks: list[Matrix] = [
            tuple(tuple(scale * value / cost for value in row) for row in block)
            for block, cost in zip(
                _group_inverses(matrix, groups), group_costs, strict=True
            )
        ]

    def evaluate(self, shares: Sequence[float]) -> tuple[float, list[float]]:
        """Return ``g(theta)`` and the sensitivities ``d_k = v' A_k v``.

        ``v = Phi^-1 e_0``, ``g = v_0``, and ``dg / dtheta_k = -d_k``; by
        construction ``sum_k theta_k d_k = g``.

        Raises:
            ValueError: If model 0 is not covered or ``Phi`` is singular.

        References:
            [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020].

        """
        column = _blue_solve(self.size, self.groups, self.blocks, shares)
        sensitivities = []
        for group, block in zip(self.groups, self.blocks, strict=True):
            local = [column[i] for i in group]
            sensitivities.append(
                math.fsum(
                    a
                    * math.fsum(value * b for value, b in zip(row, local, strict=True))
                    for a, row in zip(local, block, strict=True)
                )
            )
        return column[0], sensitivities

    def shares(self, point: Sequence[float]) -> list[float]:
        """Return ``theta_k = (1 - eps) z_k^2 / sum_j z_j^2 + eps / K``.

        References:
            [EQ-MC-MF-010], [REF-NOCEDAL-WRIGHT-2006, Chapter 2].

        """
        total = math.fsum(value * value for value in point)
        if total <= 0.0:
            raise ValueError("the share parameters must not all vanish")
        base = self.floor / len(point)
        scale = (1.0 - self.floor) / total
        return [scale * value * value + base for value in point]

    def _at(self, point: Vector) -> tuple[float, list[float]]:
        """Return :meth:`evaluate` at ``theta(z)``, remembering the last point.

        The objective and its gradient are requested at the same point by
        the line search, so one solve serves both.

        References:
            [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020].

        """
        key = tuple(point)
        if self.last is None or self.last[0] != key:
            self.last = (key, self.evaluate(self.shares(key)))
        return self.last[1]

    def objective(self, point: Vector) -> float:
        """Return ``log g(theta(z))``.

        References:
            [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020].

        """
        value = self._at(point)[0]
        if value <= 0.0:
            raise ValueError("the normalized variance is not positive")
        return math.log(value)

    def gradient(self, point: Vector) -> list[float]:
        """Return the gradient of ``log g`` with respect to ``z``.

        ``d log g / d z_j = 2 (1 - eps) z_j (dbar - d_j) / (g |z|^2)`` with
        ``dbar = sum_k z_k^2 d_k / |z|^2``.

        References:
            [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020].

        """
        value, sensitivities = self._at(point)
        total = math.fsum(z * z for z in point)
        average = (
            math.fsum(z * z * d for z, d in zip(point, sensitivities, strict=True))
            / total
        )
        factor = 2.0 * (1.0 - self.floor) / (value * total)
        return [
            factor * z * (average - d)
            for z, d in zip(point, sensitivities, strict=True)
        ]


def _solve_shares(
    problem: _ShareProblem, tolerance: float, max_restarts: int
) -> tuple[list[float], float, float, int, bool, str]:
    """Minimize the normalized variance over the budget simplex.

    BFGS runs in ``z`` from the equal-share point. After each run the bound
    ``g* >= g^2 / max_k d_k`` is evaluated; if the relative gap exceeds
    ``tolerance`` the run is restarted from ``0.5 theta + 0.5 theta_boost``,
    where ``theta_boost`` puts its mass on the groups with ``d_k > g`` (the
    directions in which a coordinate stuck near zero should grow). Finally
    the shares of at most ``eps`` are set to zero and the rest
    renormalized if that does not increase ``g``.

    Returns:
        ``(shares, g, lower bound on g*, iterations, converged, message)``.

    References:
        [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020],
        [REF-CROCI-WILLCOX-WRIGHT-2023].

    """
    count = len(problem.groups)
    start = [1.0] * count
    best_shares = problem.shares(start)
    best_value, sensitivities = problem.evaluate(best_shares)
    best_bound = best_value * best_value / max(sensitivities)
    iterations = 0
    message = "equal shares"
    for attempt in range(max_restarts + 1):
        if 1.0 - best_bound / best_value <= tolerance:
            break
        try:
            run = minimize_bfgs(
                problem.objective,
                start,
                gradient=problem.gradient,
                gradient_tolerance=1e-10,
                max_iterations=500,
            )
            candidate = problem.shares(run.x)
            value, sensitivities = problem.evaluate(candidate)
        except ValueError as error:
            message = f"BFGS run {attempt + 1} failed: {error}"
            break
        iterations += run.iterations
        message = f"BFGS run {attempt + 1}: {run.message}"
        if value < best_value:
            best_shares, best_value = candidate, value
        best_bound = max(best_bound, value * value / max(sensitivities))
        boost = [max(d - value, 0.0) for d in sensitivities]
        total = math.fsum(boost)
        if total <= 0.0:
            continue
        start = [
            math.sqrt(0.5 * share + 0.5 * extra / total)
            for share, extra in zip(candidate, boost, strict=True)
        ]
    cleaned = [0.0 if share <= problem.floor else share for share in best_shares]
    total = math.fsum(cleaned)
    if total > 0.0:
        cleaned = [share / total for share in cleaned]
        try:
            value = problem.evaluate(cleaned)[0]
        except ValueError:
            value = math.inf
        if value <= best_value:
            best_shares, best_value = cleaned, value
    best_bound = min(best_bound, best_value)
    converged = 1.0 - best_bound / best_value <= tolerance
    if not converged:
        message += "; the optimality gap exceeds the tolerance"
    return best_shares, best_value, best_bound, iterations, converged, message


def _round_allocation(
    matrix: Matrix,
    groups: Groups,
    group_costs: Vector,
    budget: float,
    continuous: Sequence[float],
) -> tuple[list[int], float]:
    """Round real group sample counts to integers within the budget.

    The counts are floored; the budget that the floors release is spent
    greedily: at each step every affordable group is tried with one more
    sample, the group with the largest variance reduction per unit cost
    wins, and it receives ``max(1, floor(remaining / (K_a W_k)))`` samples
    (``K_a`` affordable groups), so the loop ends after a number of steps
    that grows only logarithmically with the remaining budget. This is a
    heuristic; the integer problem is not solved to optimality.

    Returns:
        Integer counts and their variance ``[Psi^-1]_00``.

    Raises:
        ValueError: If no affordable integer allocation samples model 0.

    References:
        [EQ-MC-MF-010], [REF-SCHADEN-ULLMANN-2020].

    """
    size = len(matrix)
    blocks = _group_inverses(matrix, groups)

    def variance_of(counts: Sequence[float]) -> float:
        """Return the BLUE variance, infinite when model 0 is not sampled.

        References:
            [EQ-MC-MF-009], [REF-SCHADEN-ULLMANN-2020].

        """
        try:
            return _blue_solve(size, groups, blocks, counts)[0]
        except ValueError:
            return math.inf

    counts = [math.floor(value * (1.0 + 1e-12)) for value in continuous]
    remaining = budget - math.fsum(
        n * w for n, w in zip(counts, group_costs, strict=True)
    )
    current = variance_of(counts)
    for _ in range(50 * len(groups) + 200):
        affordable = [
            k for k, w in enumerate(group_costs) if w <= remaining * (1.0 + 1e-12)
        ]
        if not affordable:
            break
        best_index = -1
        best_score = 0.0
        best_variance = current
        for k in affordable:
            counts[k] += 1
            trial = variance_of(counts)
            counts[k] -= 1
            if math.isinf(current):
                score = -trial
                better = best_index < 0 or score > best_score
                better = better and math.isfinite(trial)
            else:
                score = (current - trial) / group_costs[k]
                better = score > best_score
            if better:
                best_index, best_score, best_variance = k, score, trial
        if best_index < 0:
            break
        step = max(
            1, math.floor(remaining / (len(affordable) * group_costs[best_index]))
        )
        counts[best_index] += step
        remaining -= step * group_costs[best_index]
        current = variance_of(counts) if step > 1 else best_variance
    if math.isinf(current):
        raise ValueError(
            "the budget does not pay for one sample of a group that contains "
            "the high-fidelity model"
        )
    return counts, current


def mlblue_allocation(
    covariance: Iterable[Iterable[object]],
    costs: Sequence[float],
    budget: float,
    *,
    groups: Iterable[Iterable[int]] | None = None,
    max_group_size: int | None = None,
    tolerance: float = 1e-6,
    max_restarts: int = 5,
) -> MLBLUEAllocation:
    """Allocate group samples for an MLBLUE under a cost budget.

    The problem is ``min_n [Psi(n)^-1]_00`` subject to
    ``sum_k n_k W_k <= p`` and ``n >= 0`` with
    ``W_k = sum_(i in S_k) w_i``. The objective is convex and homogeneous of
    degree ``-1`` in ``n``, so the budget is active and with the budget
    shares ``theta_k = n_k W_k / p`` the problem becomes
    ``min g(theta)`` over the unit simplex,
    ``g = [(sum_k theta_k A_k)^-1]_00``,
    ``A_k = (w_0 C_00 / W_k) P_k' C_k^-1 P_k`` and
    ``Var = (C_00 w_0 / p) g``.

    The reference formulation of Croci, Willcox and Wright solves this as a
    semidefinite program. Here the simplex is parametrized by
    ``theta_k = z_k^2 / |z|^2`` and ``log g`` is minimized by BFGS with the
    analytic gradient ``2 z_k (1 - d_k / g) / |z|^2``,
    ``d_k = v' A_k v``, ``v = Phi^-1 e_0``. Because
    ``g(theta') >= (v' e_0)^2 / (v' Phi(theta') v)`` for every ``theta'``
    (Cauchy-Schwarz), ``g^2 / max_k d_k`` is a lower bound on the optimal
    value; the result is flagged ``converged`` only if the relative gap to
    this bound is at most ``tolerance``. The continuous optimum is then
    rounded: floors, followed by a greedy use of the released budget. The
    rounding is a heuristic and ``variance`` (integers) is never below the
    continuous optimum.

    Args:
        covariance: Model covariance matrix (high-fidelity model first),
            independent of the estimation samples.
        costs: Positive cost of one evaluation of every model.
        budget: Positive cost budget ``p``.
        groups: Model groups to allocate; default all groups up to
            ``max_group_size`` models.
        max_group_size: Size limit of the default groups (ignored when
            ``groups`` is given).
        tolerance: Relative optimality gap for the convergence flag.
        max_restarts: Restarts of BFGS when the gap is not closed.

    Returns:
        Continuous optimum with its lower bound, the rounded allocation, and
        the convergence flag.

    Raises:
        ValueError: If no group contains model 0, the budget does not pay
            for one sample of such a group, or an input is invalid.

    Equation:
        [EQ-MC-MF-009], [EQ-MC-MF-010].

    References:
        [REF-SCHADEN-ULLMANN-2020], [REF-CROCI-WILLCOX-WRIGHT-2023] (problem
        statement only; the semidefinite program is replaced by a smooth
        nonlinear program).

    """
    matrix = _as_covariance(covariance)
    size = len(matrix)
    weights = _as_costs(costs, size)
    spend = _as_budget(budget)
    gap = require_finite_real(tolerance, name="tolerance")
    if gap <= 0.0:
        raise ValueError("tolerance must be positive")
    restarts = _as_count(max_restarts, name="max_restarts", minimum=0)
    if groups is None:
        sets = mlblue_groups(size, max_size=max_group_size)
    else:
        sets = _as_groups(groups, size)
    if not any(0 in group for group in sets):
        raise ValueError("no group contains the high-fidelity model 0")
    group_costs = tuple(math.fsum(weights[i] for i in group) for group in sets)
    problem = _ShareProblem(matrix, sets, group_costs, 0.25 * gap)
    shares, value, bound, iterations, converged, message = _solve_shares(
        problem, gap, restarts
    )
    mc_variance = matrix[0][0] * weights[0] / spend
    continuous = tuple(
        spend * share / cost for share, cost in zip(shares, group_costs, strict=True)
    )
    counts, variance = _round_allocation(matrix, sets, group_costs, spend, continuous)
    evaluations = [0] * size
    for group, count in zip(sets, counts, strict=True):
        for index in group:
            evaluations[index] += count
    return MLBLUEAllocation(
        groups=sets,
        group_costs=group_costs,
        shares=tuple(shares),
        continuous_samples=continuous,
        continuous_variance=mc_variance * value,
        lower_bound=mc_variance * bound,
        samples=tuple(counts),
        variance=variance,
        mc_variance=mc_variance,
        cost=math.fsum(n * w for n, w in zip(counts, group_costs, strict=True)),
        budget=spend,
        evaluations=tuple(evaluations),
        converged=converged,
        iterations=iterations,
        message=message,
    )


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def compare_multifidelity_estimators(
    covariance: Iterable[Iterable[object]],
    costs: Sequence[float],
    budget: float,
    *,
    max_group_size: int | None = None,
    acv_starts: int = 4,
    seed: int | RandomStream | None = 0,
) -> MultifidelityComparison:
    """Compare predicted variances of MC, MFMC, ACV-MF, and MLBLUE.

    For one covariance, cost vector, and budget the table holds

    * ``MC``: ``C_00 / floor(p / w_0)`` (continuous: ``C_00 w_0 / p``);
    * ``MFMC``: the best admissible subset from :func:`select_mfmc_models`;
    * ``ACV-MF``: the heuristic allocation of :func:`acv_allocation`;
    * ``MLBLUE``: :func:`mlblue_allocation` over all groups of at most
      ``max_group_size`` models.

    All numbers are predictions from ``covariance``: they are exact for the
    true covariance and estimates for a pilot covariance (whose cost is not
    part of ``budget``). ``variance`` refers to integer allocations and
    ``continuous_variance`` to the real-valued optima; among the continuous
    values MLBLUE with all groups is never above MFMC, because the nested
    MFMC samples are one feasible group allocation of the same cost.

    Args:
        covariance: Model covariance matrix (high-fidelity model first).
        costs: Positive cost of one evaluation of every model.
        budget: Positive cost budget ``p`` (at least ``w_0``).
        max_group_size: Largest MLBLUE group; default all groups.
        acv_starts: Start points per subset of the ACV-MF allocation.
        seed: Seed of the ACV-MF random starts.

    Returns:
        Table with one row per method.

    Raises:
        ValueError: If the budget is below ``w_0`` or an input is invalid.

    Equation:
        [EQ-MC-MF-011].

    References:
        [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016],
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020],
        [REF-SCHADEN-ULLMANN-2020].

    """
    matrix = _as_covariance(covariance)
    size = len(matrix)
    weights = _as_costs(costs, size)
    spend = _as_budget(budget)
    if spend < weights[0]:
        raise ValueError("budget is below the cost of one high-fidelity evaluation")
    plain_count = math.floor(spend / weights[0] * (1.0 + 1e-12))
    plain = matrix[0][0] / plain_count
    mfmc = select_mfmc_models(matrix, weights, spend)
    acv = acv_allocation(
        matrix, weights, spend, kind="MF", starts=acv_starts, seed=seed
    )
    blue = mlblue_allocation(matrix, weights, spend, max_group_size=max_group_size)
    rows = (
        MultifidelityComparisonRow(
            method="MC",
            variance=plain,
            continuous_variance=matrix[0][0] * weights[0] / spend,
            variance_ratio=1.0,
            cost=plain_count * weights[0],
            evaluations=(plain_count, *([0] * (size - 1))),
            converged=True,
        ),
        MultifidelityComparisonRow(
            method="MFMC",
            variance=mfmc.variance,
            continuous_variance=mfmc.continuous_variance,
            variance_ratio=mfmc.variance / plain,
            cost=mfmc.cost,
            evaluations=mfmc.evaluations,
            converged=True,
        ),
        MultifidelityComparisonRow(
            method="ACV-MF",
            variance=acv.design.variance,
            continuous_variance=acv.continuous_variance,
            variance_ratio=acv.design.variance / plain,
            cost=acv.design.cost,
            evaluations=acv.design.samples,
            converged=acv.converged,
        ),
        MultifidelityComparisonRow(
            method="MLBLUE",
            variance=blue.variance,
            continuous_variance=blue.continuous_variance,
            variance_ratio=blue.variance / plain,
            cost=blue.cost,
            evaluations=blue.evaluations,
            converged=blue.converged,
        ),
    )
    return MultifidelityComparison(
        budget=spend,
        rows=rows,
        mfmc_models=mfmc.models,
        mlblue_groups=tuple(
            group
            for group, count in zip(blue.groups, blue.samples, strict=True)
            if count
        ),
    )
