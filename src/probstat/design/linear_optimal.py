"""Alphabetic criteria and budgeted sensor selection for linear Gaussian problems.

Experimental (added in 0.28).

For observations ``y = A x + e`` of an unknown state ``x ~ N(m, P)`` with noise
``e ~ N(0, R)``, every design is a finite set of rows ``a_j'`` of a candidate
matrix. With ``P = L L'`` and the whitened observation matrix
``B = R^{-1/2} A L``, the posterior precision in whitened coordinates is
``I + H``, ``H = B'B``, and the posterior covariance in state coordinates is
``C = L (I + H)^{-1} L'`` (EQ-DES-LIN-001). The criteria are

* A-optimal: ``trace(C)`` (minimize; depends on the state metric),
* D-optimal: ``log det(I + H)``, whose half is the expected information gain
  (EQ-DES-LIN-002; invariant under invertible reparameterization),
* E-optimal: smallest eigenvalue of ``I + H`` (whitened) and largest
  eigenvalue of ``C`` (state coordinates; EQ-DES-LIN-003),
* goal-oriented: ``trace(Q C Q')`` for linear quantities of interest ``Q x``
  (EQ-DES-LIN-004).

Without a prior the classical Fisher matrix ``F = A' R^{-1} A`` is used; its
log-determinant is ``-inf`` when ``F`` is singular and the null space is
reported instead of silently adding a prior (EQ-DES-LIN-005).

:func:`greedy_linear_design` selects independent-noise sensors one at a time
with the determinant lemma and the Sherman-Morrison covariance downdate
(EQ-DES-LIN-006); :func:`exhaustive_linear_design` enumerates all subsets of a
given size as an oracle (EQ-DES-LIN-008). Correlated noise across the selected
sensors is handled by the joint formula (EQ-DES-LIN-007); information is then
not additive over sensors.

Assumptions:
    Linear observations, Gaussian prior and noise, known covariances.

Common pitfalls:
    - A-optimal and E-optimal values depend on the units of the state; D-optimal
      and goal values transform consistently under ``x -> D x``.
    - Greedy selection is not guaranteed to be globally optimal; compare with
      :func:`exhaustive_linear_design` on small problems.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    inverse_spd,
    log_det_spd,
    matmul,
    matvec,
    solve_lower,
    symmetric_eigen,
    transpose,
)

_CRITERIA = ("d", "a", "goal")
_SYMMETRY_TOLERANCE = 1e-10
_RANK_FACTOR = 1e-12
_ESTIMABLE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_symmetric(matrix: Matrix, *, name: str) -> None:
    """Raise if a square matrix is not symmetric relative to its scale.

    Raises:
        ValueError: If ``|A_ij - A_ji| > 1e-10 max |A|`` for some pair.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    for i in range(len(matrix)):
        for j in range(i):
            if abs(matrix[i][j] - matrix[j][i]) > _SYMMETRY_TOLERANCE * scale:
                raise ValueError(f"{name} must be symmetric")


def _check_prior(prior: Sequence[Sequence[float]], dimension: int) -> Matrix:
    """Validate a prior covariance for ``dimension`` unknowns.

    Raises:
        ValueError: If it is not a symmetric ``dimension`` square matrix.

    References:
        [EQ-DES-LIN-001]. No equation is evaluated.

    """
    sigma = as_matrix(prior, name="prior_covariance")
    if len(sigma) != dimension or any(len(row) != dimension for row in sigma):
        raise ValueError("prior_covariance must be n by n for n candidate columns")
    _require_symmetric(sigma, name="prior_covariance")
    return sigma


def _check_candidates(
    candidate_rows: Sequence[Sequence[float]], noise_variances: Sequence[float] | None
) -> tuple[Matrix, Vector | None]:
    """Validate candidate rows and, if given, their noise variances.

    Raises:
        ValueError: If rows are empty or ragged, or a variance is not positive.

    References:
        [EQ-DES-LIN-001]. No equation is evaluated.

    """
    rows = as_matrix(candidate_rows, name="candidate_rows")
    if not rows or not rows[0]:
        raise ValueError("candidate_rows must be a non-empty matrix")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("candidate_rows must be rectangular")
    if noise_variances is None:
        return rows, None
    variances = as_vector(noise_variances, name="noise_variances")
    if len(variances) != len(rows) or any(v <= 0.0 for v in variances):
        raise ValueError("noise_variances must be positive, one per candidate row")
    return rows, variances


def _check_indices(selected: Sequence[int], count: int) -> tuple[int, ...]:
    """Validate a sequence of candidate indices (repetitions allowed).

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If an index is out of range.

    References:
        [EQ-DES-LIN-001]. No equation is evaluated.

    """
    indices = tuple(selected)
    for position, index in enumerate(indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"selected[{position}] must be an integer")
        if not 0 <= index < count:
            raise ValueError(f"selected[{position}] is out of range")
    return indices


def _goal_matrix(goal: Iterable[object] | None, dimension: int) -> Matrix | None:
    """Normalize a goal (vector ``c`` or matrix ``Q``) to a matrix ``Q``.

    Raises:
        ValueError: If the goal is empty or has the wrong width.

    References:
        [EQ-DES-LIN-004]. No equation is evaluated.

    """
    if goal is None:
        return None
    items = tuple(goal)
    if not items:
        raise ValueError("goal must not be empty")
    first = items[0]
    if isinstance(first, Sequence):
        q = as_matrix(cast("Sequence[Sequence[object]]", items), name="goal")
    else:
        q = (as_vector(items, name="goal"),)
    if any(len(row) != dimension for row in q):
        raise ValueError("goal must have one column per state variable")
    return q


def _trace_sandwich(q: Matrix, c: Matrix) -> float:
    """Return ``trace(Q C Q')``.

    References:
        [EQ-DES-LIN-004]. Evaluates the goal-oriented variance.

    """
    return math.fsum(
        math.fsum(qi * wi for qi, wi in zip(row, matvec(c, row), strict=True))
        for row in q
    )


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearDesignCriteria:
    """Deterministic information criteria of one linear design.

    Attributes:
        mode: ``"bayesian"`` (a prior was given) or ``"classical"``.
        posterior_covariance: ``C = L (I + H)^-1 L'`` in state coordinates
            (``None`` in classical mode).
        information_matrix: Fisher matrix ``F = A' R^-1 A`` of the selection
            (state coordinates; zero for an empty selection).
        rank: Numerical rank of ``F`` (eigenvalues above ``1e-12`` times the
            largest, scaled by the dimension).
        rank_deficient: ``rank < n``. In Bayesian mode the criteria stay
            finite (the prior supplies the missing directions); in classical
            mode they are degenerate.
        nullspace: Orthonormal eigenvectors of ``F`` for the numerically zero
            eigenvalues (the state directions the design cannot see).
        a_optimal: Bayesian ``trace(C)``; classical ``trace(F^-1)`` or ``inf``.
        d_optimal: Bayesian ``log det(I + H)``; classical ``log det F`` or
            ``-inf`` when rank deficient.
        expected_information_gain: ``0.5 log det(I + H)`` in nats (Bayesian
            mode only, else ``None``).
        e_information_eigenvalue: Smallest eigenvalue of ``I + H`` (Bayesian,
            whitened coordinates) or of ``F`` (classical), as computed.
        max_covariance_eigenvalue: Largest eigenvalue of ``C`` (Bayesian,
            state coordinates) or ``1 / lambda_min(F)`` (classical; ``inf``
            when rank deficient).
        goal_variance: ``trace(Q C Q')`` (classical: ``trace(Q F^+ Q')`` if
            every row of ``Q`` lies in the range of ``F``, else ``inf``);
            ``None`` without a goal.

    References:
        [REF-ALEXANDERIAN-2021], [REF-CHALONER-VERDINELLI-1995].

    """

    mode: str
    posterior_covariance: Matrix | None
    information_matrix: Matrix
    rank: int
    rank_deficient: bool
    nullspace: tuple[Vector, ...]
    a_optimal: float
    d_optimal: float
    expected_information_gain: float | None
    e_information_eigenvalue: float
    max_covariance_eigenvalue: float
    goal_variance: float | None


def _whiten_rows(
    rows: Matrix, variances: Vector | None, covariance: Matrix | None
) -> Matrix:
    """Return ``R^{-1/2} A`` with ``R^{-1/2} = L_R^{-1}`` for the selection.

    For independent noise this divides row ``j`` by ``s_j``; for a full
    covariance it solves ``L_R W = A`` column by column (EQ-DES-LIN-007).

    References:
        [EQ-DES-LIN-007], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    if covariance is None:
        assert variances is not None
        return tuple(
            tuple(value / math.sqrt(v) for value in row)
            for row, v in zip(rows, variances, strict=True)
        )
    lower = cholesky(covariance)
    columns = tuple(
        solve_lower(lower, tuple(row[j] for row in rows)) for j in range(len(rows[0]))
    )
    return transpose(columns)


def _noise_of_selection(
    variances: Sequence[float] | None,
    covariance: Sequence[Sequence[float]] | None,
    indices: tuple[int, ...],
    count: int,
) -> tuple[Vector | None, Matrix | None]:
    """Resolve the noise model of a selection.

    Raises:
        ValueError: If not exactly one noise argument is given or shapes
            disagree.

    References:
        [EQ-DES-LIN-007]. No equation is evaluated.

    """
    if (variances is None) == (covariance is None):
        raise ValueError("give exactly one of noise_variances or noise_covariance")
    if covariance is None:
        assert variances is not None
        per_candidate = as_vector(variances, name="noise_variances")
        if len(per_candidate) != count or any(v <= 0.0 for v in per_candidate):
            raise ValueError("noise_variances must be positive, one per candidate row")
        return tuple(per_candidate[i] for i in indices), None
    full = as_matrix(covariance, name="noise_covariance")
    if len(full) != len(indices) or any(len(row) != len(indices) for row in full):
        raise ValueError("noise_covariance must be k by k for k selected sensors")
    _require_symmetric(full, name="noise_covariance")
    return None, full


def linear_design_criteria(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float] | None,
    selected: Sequence[int],
    *,
    prior_covariance: Sequence[Sequence[float]] | None = None,
    goal: Sequence[float] | Sequence[Sequence[float]] | None = None,
    noise_covariance: Sequence[Sequence[float]] | None = None,
) -> LinearDesignCriteria:
    """Return the A, D, E and goal-oriented criteria of a linear design.

    The design consists of rows ``selected`` of ``candidate_rows`` (an index
    may repeat: a replicate). With a prior ``P = L L'``, ``B = R^-1/2 A L``,
    ``H = B'B`` the posterior covariance is ``C = L (I + H)^-1 L'``,
    ``D = log det(I + H)``, the expected information gain is ``D / 2``
    (EQ-DES-LIN-001, EQ-DES-LIN-002). Without a prior the classical Fisher
    matrix ``F = A' R^-1 A`` is analysed instead and degenerate criteria are
    reported as ``-inf`` / ``inf`` with ``rank_deficient`` set, never repaired
    (EQ-DES-LIN-005). An explicit ``noise_covariance`` of the selected sensors
    (in ``selected`` order) replaces the independent variances; information is
    then not additive over sensors but the joint formula still applies
    (EQ-DES-LIN-007).

    Args:
        candidate_rows: ``N`` by ``n`` matrix; row ``j`` is the observation
            functional ``a_j'``.
        noise_variances: Positive noise variance per candidate, or ``None``
            when ``noise_covariance`` is given.
        selected: Candidate indices of the design (may be empty or repeat).
        prior_covariance: SPD ``n`` by ``n`` prior covariance; omitted means a
            classical analysis.
        goal: Linear quantity of interest: a vector ``c`` or a matrix ``Q``.
        noise_covariance: Full SPD ``k`` by ``k`` noise covariance of the
            selected sensors.

    Returns:
        The criteria and diagnostics.

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If shapes disagree, a covariance is not symmetric or not
            positive definite, an index is out of range, or the noise
            arguments are not exactly one of the two forms.

    Equation:
        [EQ-DES-LIN-001], [EQ-DES-LIN-002], [EQ-DES-LIN-003],
        [EQ-DES-LIN-004], [EQ-DES-LIN-005], [EQ-DES-LIN-007].

    References:
        [REF-ALEXANDERIAN-2021], [REF-CHALONER-VERDINELLI-1995].

    """
    rows, _ = _check_candidates(candidate_rows, None)
    n = len(rows[0])
    indices = _check_indices(selected, len(rows))
    variances, covariance = _noise_of_selection(
        noise_variances, noise_covariance, indices, len(rows)
    )
    q = _goal_matrix(goal, n)
    sigma = None if prior_covariance is None else _check_prior(prior_covariance, n)
    chosen = tuple(rows[i] for i in indices)
    if chosen:
        whitened = _whiten_rows(chosen, variances, covariance)
        fisher = matmul(transpose(whitened), whitened)
    else:
        whitened = ()
        fisher = tuple((0.0,) * n for _ in range(n))
    eigenvalues, vectors = symmetric_eigen(fisher)
    top = max(eigenvalues[-1], 0.0)
    threshold = _RANK_FACTOR * n * top
    null_columns = [j for j, value in enumerate(eigenvalues) if value <= threshold]
    rank = n - len(null_columns)
    nullspace = tuple(tuple(vectors[i][j] for i in range(n)) for j in null_columns)
    if sigma is None:
        return _classical(fisher, eigenvalues, vectors, rank, nullspace, q, threshold)
    lower = cholesky(sigma)
    if chosen:
        mapped = matmul(whitened, lower)
        gram = matmul(transpose(mapped), mapped)
    else:
        gram = tuple((0.0,) * n for _ in range(n))
    shifted = tuple(
        tuple(gram[i][j] + (1.0 if i == j else 0.0) for j in range(n)) for i in range(n)
    )
    inverse = inverse_spd(shifted)
    covariance_post = matmul(matmul(lower, inverse), transpose(lower))
    covariance_post = tuple(
        tuple(0.5 * (covariance_post[i][j] + covariance_post[j][i]) for j in range(n))
        for i in range(n)
    )
    log_det = log_det_spd(shifted)
    info_eigenvalues, _ = symmetric_eigen(shifted)
    cov_eigenvalues, _ = symmetric_eigen(covariance_post)
    return LinearDesignCriteria(
        mode="bayesian",
        posterior_covariance=covariance_post,
        information_matrix=fisher,
        rank=rank,
        rank_deficient=rank < n,
        nullspace=nullspace,
        a_optimal=math.fsum(covariance_post[i][i] for i in range(n)),
        d_optimal=log_det,
        expected_information_gain=0.5 * log_det,
        e_information_eigenvalue=info_eigenvalues[0],
        max_covariance_eigenvalue=cov_eigenvalues[-1],
        goal_variance=None if q is None else _trace_sandwich(q, covariance_post),
    )


def _classical(
    fisher: Matrix,
    eigenvalues: Vector,
    vectors: Matrix,
    rank: int,
    nullspace: tuple[Vector, ...],
    q: Matrix | None,
    threshold: float,
) -> LinearDesignCriteria:
    """Return the classical (prior-free) criteria from the spectrum of ``F``.

    References:
        [EQ-DES-LIN-005], [REF-CHALONER-VERDINELLI-1995].

    """
    n = len(fisher)
    deficient = rank < n
    positive = [j for j, value in enumerate(eigenvalues) if value > threshold]
    if deficient:
        a_value = math.inf
        d_value = -math.inf
        max_cov = math.inf
    else:
        a_value = math.fsum(1.0 / eigenvalues[j] for j in positive)
        d_value = math.fsum(math.log(eigenvalues[j]) for j in positive)
        max_cov = 1.0 / eigenvalues[0]
    goal_variance: float | None = None
    if q is not None:
        goal_variance = _classical_goal(q, eigenvalues, vectors, positive, deficient)
    return LinearDesignCriteria(
        mode="classical",
        posterior_covariance=None,
        information_matrix=fisher,
        rank=rank,
        rank_deficient=deficient,
        nullspace=nullspace,
        a_optimal=a_value,
        d_optimal=d_value,
        expected_information_gain=None,
        e_information_eigenvalue=eigenvalues[0],
        max_covariance_eigenvalue=max_cov,
        goal_variance=goal_variance,
    )


def _classical_goal(
    q: Matrix,
    eigenvalues: Vector,
    vectors: Matrix,
    positive: Sequence[int],
    deficient: bool,
) -> float:
    """Return ``trace(Q F^+ Q')`` if ``Q`` is estimable, else ``inf``.

    A row ``c`` is estimable when its component in the null space of ``F`` is
    negligible (``1e-9 ||c||``).

    References:
        [EQ-DES-LIN-005], [EQ-DES-LIN-004].

    """
    n = len(vectors)
    total = 0.0
    for row in q:
        norm = math.sqrt(math.fsum(v * v for v in row))
        projections = [
            math.fsum(vectors[i][j] * row[i] for i in range(n)) for j in range(n)
        ]
        if deficient:
            kept = set(positive)
            lost = math.sqrt(
                math.fsum(p * p for j, p in enumerate(projections) if j not in kept)
            )
            if lost > _ESTIMABLE_TOLERANCE * norm:
                return math.inf
        total += math.fsum(projections[j] ** 2 / eigenvalues[j] for j in positive)
    return total


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearGreedyResult:
    """Sensors chosen one at a time by a rank-one greedy rule.

    Attributes:
        indices: Selected candidate indices in selection order.
        gains: Criterion improvement of each step (log-determinant increase
            for ``"d"``, trace decrease for ``"a"`` and ``"goal"``).
        log_det_gains: ``log(1 + a'Ca / s^2)`` of each step (``2 *`` the EIG
            increment), for every criterion.
        criterion: ``"d"``, ``"a"`` or ``"goal"``.
        initial_value: Criterion value of the empty design (``0`` for D,
            ``trace(P)`` for A, ``trace(Q P Q')`` for goal).
        values: Criterion value after each step (cumulative log-determinant
            for D, remaining trace otherwise).
        total_cost: Cost of the selection (its count when no costs are given).
        budget: The budget used.
        stop_reason: ``"budget_exhausted"``, ``"no_affordable_candidate"`` or
            ``"no_positive_gain"``.
        posterior_covariance: Posterior covariance of the final design.

    References:
        [REF-ALEXANDERIAN-2021], [REF-CHALONER-VERDINELLI-1995].

    """

    indices: tuple[int, ...]
    gains: Vector
    log_det_gains: Vector
    criterion: str
    initial_value: float
    values: Vector
    total_cost: float
    budget: float
    stop_reason: str
    posterior_covariance: Matrix


def _greedy_inputs(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    prior_covariance: Sequence[Sequence[float]],
    criterion: str,
    goal: Iterable[object] | None,
) -> tuple[Matrix, Vector, Matrix, Matrix | None]:
    """Validate the arguments shared by the greedy and exhaustive routines.

    Raises:
        ValueError: For an unknown criterion or a goal mismatch.

    References:
        [EQ-DES-LIN-006]. No equation is evaluated.

    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}")
    rows, variances = _check_candidates(candidate_rows, noise_variances)
    assert variances is not None
    sigma = _check_prior(prior_covariance, len(rows[0]))
    cholesky(sigma)  # raises ValueError unless positive definite
    q = _goal_matrix(goal, len(rows[0]))
    if criterion == "goal" and q is None:
        raise ValueError("criterion='goal' requires a goal")
    if criterion != "goal" and q is not None:
        raise ValueError("goal is only used with criterion='goal'")
    return rows, variances, sigma, q


def greedy_linear_design(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    *,
    prior_covariance: Sequence[Sequence[float]],
    budget: float,
    costs: Sequence[float] | None = None,
    criterion: str = "d",
    goal: Sequence[float] | Sequence[Sequence[float]] | None = None,
    allow_repeats: bool = False,
    gain_per_cost: bool = False,
) -> LinearGreedyResult:
    """Select independent-noise sensors greedily with rank-one updates.

    With the current posterior covariance ``C``, candidate ``a`` of noise
    variance ``s^2`` changes the criteria by

    * D: ``log det`` gain ``log1p(a'Ca / s^2)`` (matrix determinant lemma),
    * A: trace drop ``||C a||^2 / (s^2 + a'Ca)``,
    * goal: ``trace(Q C Q')`` drop ``||Q C a||^2 / (s^2 + a'Ca)``,

    and after the choice ``C <- C - C a a' C / (s^2 + a'Ca)`` (Sherman-Morrison;
    EQ-DES-LIN-006). Ties are resolved to the lowest index (exact floating
    point ties only). Selection stops when the budget cannot afford any
    remaining candidate or when no candidate has a positive gain. Without
    ``costs`` the budget is the number of sensors; with ``costs`` it is the
    total cost, compared in floating point as ``spent + cost <= budget``. The
    D gains ``log1p(a'Ca / s^2)`` are non-increasing along the selection for
    independent noise (diminishing returns); no optimality guarantee is
    claimed here for any criterion, and none for ``gain_per_cost``.

    Args:
        candidate_rows: ``N`` by ``n`` observation rows.
        noise_variances: Positive noise variance per candidate (independent).
        prior_covariance: SPD ``n`` by ``n`` prior covariance.
        budget: Positive integer count (no ``costs``) or positive total cost.
        costs: Optional positive cost per candidate.
        criterion: ``"d"``, ``"a"`` or ``"goal"``.
        goal: Required vector ``c`` or matrix ``Q`` for ``"goal"``.
        allow_repeats: Whether a candidate may be chosen again (replicates).
        gain_per_cost: Rank candidates by gain divided by cost.

    Returns:
        The selection, per-step gains, total cost and final covariance.

    Raises:
        TypeError: If the count budget is not an integer.
        ValueError: For invalid shapes, covariances, costs, budget, criterion
            or goal.

    Equation:
        [EQ-DES-LIN-006], [EQ-DES-LIN-002], [EQ-DES-LIN-004].

    References:
        [REF-ALEXANDERIAN-2021], [REF-CHALONER-VERDINELLI-1995].

    """
    rows, variances, sigma, q = _greedy_inputs(
        candidate_rows, noise_variances, prior_covariance, criterion, goal
    )
    count = len(rows)
    if costs is None:
        steps = require_non_negative_integer(budget, name="budget")
        if steps == 0:
            raise ValueError("budget must be positive")
        price: Vector = (1.0,) * count
        limit = float(steps)
    else:
        price = as_vector(costs, name="costs")
        if len(price) != count or any(c <= 0.0 for c in price):
            raise ValueError("costs must be positive, one per candidate")
        limit = require_finite_real(budget, name="budget")
        if limit <= 0.0:
            raise ValueError("budget must be positive")
    cov = [list(row) for row in sigma]
    initial = {
        "d": 0.0,
        "a": math.fsum(sigma[i][i] for i in range(len(sigma))),
        "goal": 0.0 if q is None else _trace_sandwich(q, sigma),
    }[criterion]
    chosen: list[int] = []
    gains: list[float] = []
    log_gains: list[float] = []
    values: list[float] = []
    spent = 0.0
    current = initial
    stop = "budget_exhausted"
    while True:
        best = -1
        best_rank = -math.inf
        best_data: tuple[float, float, list[float]] | None = None
        affordable = False
        for j in range(count):
            if (not allow_repeats and j in chosen) or spent + price[j] > limit:
                continue
            affordable = True
            a = rows[j]
            v = [
                math.fsum(ci * ai for ci, ai in zip(crow, a, strict=True))
                for crow in cov
            ]
            quad = math.fsum(ai * vi for ai, vi in zip(a, v, strict=True))
            denom = variances[j] + quad
            log_gain = math.log1p(quad / variances[j])
            if criterion == "d":
                gain = log_gain
            elif criterion == "a":
                gain = math.fsum(vi * vi for vi in v) / denom
            else:
                assert q is not None
                gain = (
                    math.fsum(
                        math.fsum(qi * vi for qi, vi in zip(row, v, strict=True)) ** 2
                        for row in q
                    )
                    / denom
                )
            rank = gain / price[j] if gain_per_cost else gain
            if rank > best_rank:
                best, best_rank, best_data = j, rank, (gain, log_gain, v)
        if not affordable:
            stop = "no_affordable_candidate"
            break
        assert best_data is not None
        gain, log_gain, v = best_data
        if gain <= 0.0:
            stop = "no_positive_gain"
            break
        denom = variances[best] + math.fsum(
            ai * vi for ai, vi in zip(rows[best], v, strict=True)
        )
        for i, vi in enumerate(v):
            for k, vk in enumerate(v):
                cov[i][k] -= vi * vk / denom
        chosen.append(best)
        gains.append(gain)
        log_gains.append(log_gain)
        spent += price[best]
        current = current + gain if criterion == "d" else current - gain
        values.append(current)
        if costs is None and len(chosen) == steps:
            break
    return LinearGreedyResult(
        indices=tuple(chosen),
        gains=tuple(gains),
        log_det_gains=tuple(log_gains),
        criterion=criterion,
        initial_value=initial,
        values=tuple(values),
        total_cost=spent,
        budget=limit,
        stop_reason=stop,
        posterior_covariance=tuple(tuple(row) for row in cov),
    )


# ---------------------------------------------------------------------------
# Exhaustive oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearExhaustiveResult:
    """Optimal subset of a given size and the full ranking of all subsets.

    Attributes:
        indices: Optimal subset (ascending candidate indices).
        value: Its criterion value (``log det`` for D, trace or goal
            variance otherwise).
        ranking: ``(indices, value)`` of every subset, best first; ties are
            ordered lexicographically by indices.
        criterion: ``"d"``, ``"a"`` or ``"goal"``.
        combinations: Number of subsets evaluated.

    References:
        [REF-ALEXANDERIAN-2021], [REF-CHALONER-VERDINELLI-1995].

    """

    indices: tuple[int, ...]
    value: float
    ranking: tuple[tuple[tuple[int, ...], float], ...]
    criterion: str
    combinations: int


def exhaustive_linear_design(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    *,
    prior_covariance: Sequence[Sequence[float]],
    size: int,
    criterion: str = "d",
    goal: Sequence[float] | Sequence[Sequence[float]] | None = None,
    max_combinations: int = 20000,
) -> LinearExhaustiveResult:
    """Evaluate every subset of ``size`` candidates (without repeats).

    Each subset is scored with :func:`linear_design_criteria` (full
    recomputation, independent of the rank-one updates), so the result is an
    oracle for :func:`greedy_linear_design` on tiny problems (EQ-DES-LIN-008).
    D is maximized; A and goal variances are minimized.

    Args:
        candidate_rows: ``N`` by ``n`` observation rows.
        noise_variances: Positive noise variance per candidate (independent).
        prior_covariance: SPD ``n`` by ``n`` prior covariance.
        size: Number ``k`` of sensors, ``1 <= k <= N``.
        criterion: ``"d"``, ``"a"`` or ``"goal"``.
        goal: Required vector ``c`` or matrix ``Q`` for ``"goal"``.
        max_combinations: Cap on ``C(N, k)``; larger problems are refused.

    Returns:
        The optimum and the ranking of all subsets.

    Raises:
        ValueError: For invalid inputs or if ``C(N, k)`` exceeds the cap.

    Equation:
        [EQ-DES-LIN-008], [EQ-DES-LIN-002], [EQ-DES-LIN-003].

    References:
        [REF-ALEXANDERIAN-2021], [REF-CHALONER-VERDINELLI-1995].

    """
    rows, variances, sigma, _ = _greedy_inputs(
        candidate_rows, noise_variances, prior_covariance, criterion, goal
    )
    k = require_non_negative_integer(size, name="size")
    cap = require_non_negative_integer(max_combinations, name="max_combinations")
    if not 1 <= k <= len(rows):
        raise ValueError("size must satisfy 1 <= size <= number of candidates")
    total = math.comb(len(rows), k)
    if total > cap:
        raise ValueError(f"{total} combinations exceed max_combinations={cap}")
    scored: list[tuple[tuple[int, ...], float]] = []
    for subset in itertools.combinations(range(len(rows)), k):
        result = linear_design_criteria(
            rows,
            variances,
            subset,
            prior_covariance=sigma,
            goal=goal,
        )
        if criterion == "d":
            value = result.d_optimal
        elif criterion == "a":
            value = result.a_optimal
        else:
            assert result.goal_variance is not None
            value = result.goal_variance
        scored.append((subset, value))
    sign = -1.0 if criterion == "d" else 1.0
    scored.sort(key=lambda item: (sign * item[1], item[0]))
    best_indices, best_value = scored[0]
    return LinearExhaustiveResult(
        best_indices, best_value, tuple(scored), criterion, total
    )
