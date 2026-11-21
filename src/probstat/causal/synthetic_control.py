"""Synthetic control for a single treated unit with placebo inference.

A synthetic control is a convex combination of untreated *donor* units
chosen to reproduce the treated unit's pre-treatment characteristics:

``w* = argmin (x1 - X0 w)' V (x1 - X0 w)`` subject to ``w >= 0``,
``sum w = 1`` (EQ-CAUSAL-SC-001),

with ``x1`` the treated unit's ``k`` predictors, ``X0`` the ``k x J``
donor predictors and ``V`` a diagonal matrix of predictor importances.
The effect path is ``tau_t = Y_1t - sum_j w_j Y_jt`` (EQ-CAUSAL-SC-003).

The simplex-constrained quadratic program is solved by projected gradient
with the exact Euclidean projection onto the simplex (EQ-CAUSAL-SC-002),
accelerated by an exact solve on the current support whenever that
improves the certificate. Convergence is certified by the duality gap
``g'w - min_j g_j >= f(w) - f(w*)`` (``g`` the gradient), which is zero
exactly when the Karush-Kuhn-Tucker conditions hold.

Assumptions: the untreated outcome of the treated unit is well
approximated by a fixed convex combination of donors both before and
after treatment (a factor-model argument, not checked here), donors are
unaffected by the intervention, and no anticipation. ``V`` is supplied by
the user (identity by default); the data-driven choice of ``V`` by nested
optimization is not implemented. The weights need not be unique when
there are more donors than predictors; the fitted combination is.

Inference is by in-space placebos: the method is re-run pretending each
donor was treated, and the treated unit's post/pre RMSPE ratio is ranked
among all units (EQ-CAUSAL-SC-004). The resulting p-value is a placebo
rank, not a sampling-based p-value: it is a randomization p-value only if
the treated unit were chosen at random.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    lu_solve,
    symmetric_eigen,
)

_SUPPORT_THRESHOLD = 1e-10
_POLISH_EVERY = 20


def project_to_simplex(values: Iterable[object]) -> Vector:
    """Project a vector onto the probability simplex in Euclidean norm.

    The projection of ``v`` is ``w_j = max(v_j - theta, 0)`` where
    ``theta`` makes the entries sum to one (the KKT conditions of
    ``min |w - v|^2`` on the simplex). With ``u`` the entries sorted in
    decreasing order, ``rho = max{j : u_j - (sum_{i<=j} u_i - 1) / j > 0}``
    and ``theta = (sum_{i<=rho} u_i - 1) / rho``.

    Args:
        values: Finite vector of length at least one.

    Returns:
        Non-negative vector summing to one, the closest such vector to
        ``values``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the vector is empty or not finite.

    Equation:
        [EQ-CAUSAL-SC-002].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 16] (KKT conditions of a convex
        quadratic program; the closed form is derived from them).

    """
    v = as_vector(values, name="values")
    ordered = sorted(v, reverse=True)
    running = 0.0
    theta = 0.0
    for index, value in enumerate(ordered, start=1):
        running += value
        candidate = (running - 1.0) / index
        if value - candidate > 0.0:
            theta = candidate
    return tuple(max(value - theta, 0.0) for value in v)


@dataclass(frozen=True, slots=True)
class SimplexWeights:
    """Solution of the simplex-constrained weighted least-squares problem.

    Attributes:
        weights: Donor weights (non-negative, summing to one).
        objective: ``(x1 - X0 w)' V (x1 - X0 w)`` at the weights.
        duality_gap: ``g'w - min_j g_j``, an upper bound on
            ``objective - optimum``.
        iterations: Projected-gradient iterations used.
        converged: Whether the gap fell below the tolerance; ``False``
            means the weights are approximate.

    References:
        [REF-ABADIE-DIAMOND-HAINMUELLER-2010].

    """

    weights: Vector
    objective: float
    duality_gap: float
    iterations: int
    converged: bool


def _support_solve(
    gram: Matrix, linear: Vector, support: list[int], size: int
) -> Vector | None:
    """Minimize the quadratic on the face spanned by ``support``.

    Solves the equality-constrained problem ``min w'Qw - 2 c'w`` with
    ``sum w = 1`` and ``w_j = 0`` off the support through its KKT system
    ``[[2 Q_SS, 1], [1', 0]] [w_S, lambda] = [2 c_S, 1]``.

    Args:
        gram: ``Q = X0' V X0``.
        linear: ``c = X0' V x1``.
        support: Indices allowed to be positive.
        size: Number of donors.

    Returns:
        Feasible weights, or ``None`` if the system is singular or its
        solution has a negative entry.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 16] (equality-constrained
        quadratic programs). See [EQ-CAUSAL-SC-001].

    """
    m = len(support)
    system = [[2.0 * gram[a][b] for b in support] + [1.0] for a in support] + [
        [1.0] * m + [0.0]
    ]
    right = [2.0 * linear[a] for a in support] + [1.0]
    try:
        solution = lu_solve(tuple(tuple(row) for row in system), tuple(right))
    except (ValueError, ZeroDivisionError):
        return None
    part = solution[:m]
    if not all(math.isfinite(value) for value in part):
        return None
    if min(part) < -1e-12:
        return None
    clipped = [max(value, 0.0) for value in part]
    total = math.fsum(clipped)
    if total <= 0.0:
        return None
    weights = [0.0] * size
    for index, value in zip(support, clipped, strict=True):
        weights[index] = value / total
    return tuple(weights)


def synthetic_control_weights(
    treated_predictors: Iterable[object],
    donor_predictors: Iterable[Iterable[object]],
    *,
    importance: Iterable[object] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> SimplexWeights:
    """Find the convex donor combination closest to the treated unit.

    Minimizes ``f(w) = sum_m v_m (x1_m - sum_j w_j X0_jm)^2`` over the
    simplex by projected gradient ``w <- P(w - g / L)`` with
    ``g = 2 (Q w - c)``, ``Q = X0' V X0``, ``c = X0' V x1`` and step
    ``1 / L``, ``L = 2 lambda_max(Q)``. Every 20 iterations (and at the
    start) the quadratic is minimized exactly on the current support and
    the result is kept if it is feasible and has a smaller duality gap.
    Iteration stops when ``gap <= tolerance * max(x1' V x1, tiny)``.

    Args:
        treated_predictors: ``k`` predictor values of the treated unit.
        donor_predictors: ``J x k`` matrix, one row per donor.
        importance: ``k`` non-negative predictor importances (the diagonal
            of ``V``), not all zero; default all ones.
        tolerance: Relative duality-gap tolerance (positive).
        max_iterations: Iteration limit (positive integer).

    Returns:
        :class:`SimplexWeights`; ``converged`` is ``False`` when the
        iteration limit was reached first.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree or an argument is out of range.

    Equation:
        [EQ-CAUSAL-SC-001], [EQ-CAUSAL-SC-002].

    References:
        [REF-ABADIE-DIAMOND-HAINMUELLER-2010],
        [REF-NOCEDAL-WRIGHT-2006, Chapter 16].

    """
    x1 = as_vector(treated_predictors, name="treated_predictors")
    donors = as_matrix(donor_predictors, name="donor_predictors")
    k, size = len(x1), len(donors)
    if len(donors[0]) != k:
        raise ValueError(
            "donor_predictors needs one column per treated predictor "
            f"({k}), got {len(donors[0])}"
        )
    if importance is None:
        v: Vector = (1.0,) * k
    else:
        v = as_vector(importance, name="importance")
        if len(v) != k:
            raise ValueError("importance must have one entry per predictor")
        if any(value < 0.0 for value in v) or not any(value > 0.0 for value in v):
            raise ValueError("importance must be non-negative and not all zero")
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    gram = tuple(
        tuple(
            math.fsum(v[m] * donors[a][m] * donors[b][m] for m in range(k))
            for b in range(size)
        )
        for a in range(size)
    )
    linear = tuple(
        math.fsum(v[m] * donors[a][m] * x1[m] for m in range(k)) for a in range(size)
    )
    constant = math.fsum(v[m] * x1[m] * x1[m] for m in range(k))
    threshold = tol * max(constant, 1e-300)

    def gradient(w: Vector) -> list[float]:
        """Return ``g = 2 (Q w - c)``.

        References:
            [EQ-CAUSAL-SC-001].

        """
        return [
            2.0 * (math.fsum(q * x for q, x in zip(row, w, strict=True)) - c)
            for row, c in zip(gram, linear, strict=True)
        ]

    def gap_of(w: Vector, g: list[float]) -> float:
        """Return the duality gap ``g'w - min_j g_j``.

        References:
            [EQ-CAUSAL-SC-001].

        """
        return math.fsum(a * b for a, b in zip(g, w, strict=True)) - min(g)

    def polish(w: Vector, gap: float) -> tuple[Vector, list[float], float] | None:
        """Try the exact minimizer on the support of ``w``.

        References:
            [EQ-CAUSAL-SC-001].

        """
        support = [j for j in range(size) if w[j] > _SUPPORT_THRESHOLD]
        candidate = _support_solve(gram, linear, support, size)
        if candidate is None:
            return None
        g_new = gradient(candidate)
        gap_new = gap_of(candidate, g_new)
        if gap_new < gap:
            return candidate, g_new, gap_new
        return None

    lipschitz = 2.0 * max(symmetric_eigen(gram)[0][-1], 0.0) if size > 1 else 0.0
    w: Vector = (1.0 / size,) * size
    g = gradient(w)
    gap = gap_of(w, g)
    iterations = 0
    while gap > threshold and lipschitz > 0.0 and iterations < max_iterations:
        if iterations % _POLISH_EVERY == 0:
            improved = polish(w, gap)
            if improved is not None:
                w, g, gap = improved
                if gap <= threshold:
                    break
        w = project_to_simplex(
            [wj - gj / lipschitz for wj, gj in zip(w, g, strict=True)]
        )
        g = gradient(w)
        gap = gap_of(w, g)
        iterations += 1
    if gap > threshold:
        improved = polish(w, gap)
        if improved is not None:
            w, g, gap = improved
    residual = [
        x1[m] - math.fsum(w[j] * donors[j][m] for j in range(size)) for m in range(k)
    ]
    return SimplexWeights(
        weights=w,
        objective=math.fsum(v[m] * residual[m] ** 2 for m in range(k)),
        duality_gap=max(gap, 0.0),
        iterations=iterations,
        converged=gap <= threshold,
    )


def _rmspe(values: Vector) -> float:
    """Return the root mean squared value of a gap path.

    Args:
        values: Gaps ``Y_1t - synthetic_t``.

    Returns:
        ``sqrt(mean(values^2))``.

    References:
        [EQ-CAUSAL-SC-003].

    """
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


@dataclass(frozen=True, slots=True)
class SyntheticControlResult:
    """Synthetic control fit and effect path of one treated unit.

    Attributes:
        weights: Donor weights.
        synthetic: Synthetic outcome ``sum_j w_j Y_jt`` for every period.
        effects: Gap ``tau_t = Y_1t - synthetic_t`` for every period; the
            first ``n_pre`` entries measure pre-treatment fit, the rest
            are the estimated effects.
        n_pre: Number of pre-treatment periods.
        pre_rmspe: Root mean squared pre-treatment gap.
        post_rmspe: Root mean squared post-treatment gap.
        rmspe_ratio: ``post_rmspe / pre_rmspe`` (``inf`` for a perfect
            pre-treatment fit with a non-zero post gap, ``nan`` if both
            are zero).
        average_effect: Mean post-treatment gap.
        objective: Minimized predictor discrepancy.
        duality_gap: Certificate of the weight optimization.
        iterations: Projected-gradient iterations.
        converged: Whether the weight optimization converged.

    References:
        [REF-ABADIE-DIAMOND-HAINMUELLER-2010].

    """

    weights: Vector
    synthetic: Vector
    effects: Vector
    n_pre: int
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    average_effect: float
    objective: float
    duality_gap: float
    iterations: int
    converged: bool


def _validated_panel(
    treated_outcome: Iterable[object],
    donor_outcomes: Iterable[Iterable[object]],
    n_pre: int,
) -> tuple[Vector, Matrix]:
    """Validate the outcome paths and the number of pre-periods.

    Args:
        treated_outcome: ``T`` outcomes of the treated unit.
        donor_outcomes: ``J x T`` donor outcomes.
        n_pre: Number of pre-treatment periods.

    Returns:
        Treated path and donor matrix.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree or ``n_pre`` is not in
            ``[1, T - 1]``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    y1 = as_vector(treated_outcome, name="treated_outcome")
    y0 = as_matrix(donor_outcomes, name="donor_outcomes")
    if len(y0[0]) != len(y1):
        raise ValueError("donor_outcomes needs one column per period")
    if isinstance(n_pre, bool) or not isinstance(n_pre, int):
        raise ValueError("n_pre must be an integer")
    if not 1 <= n_pre < len(y1):
        raise ValueError("n_pre must leave at least one pre and one post period")
    return y1, y0


def synthetic_control(
    treated_outcome: Iterable[object],
    donor_outcomes: Iterable[Iterable[object]],
    n_pre: int,
    *,
    treated_predictors: Iterable[object] | None = None,
    donor_predictors: Iterable[Iterable[object]] | None = None,
    importance: Iterable[object] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> SyntheticControlResult:
    """Fit a synthetic control and return the effect path.

    Args:
        treated_outcome: ``T`` outcomes of the treated unit in period
            order.
        donor_outcomes: ``J x T`` outcomes of the donors (one row per
            donor).
        n_pre: Number of pre-treatment periods ``T0``; the intervention
            starts in period ``T0 + 1``.
        treated_predictors: Optional ``k`` predictors of the treated unit;
            by default the ``T0`` pre-treatment outcomes.
        donor_predictors: ``J x k`` donor predictors (required exactly
            when ``treated_predictors`` is given).
        importance: Predictor importances (diagonal of ``V``).
        tolerance: Relative duality-gap tolerance.
        max_iterations: Iteration limit of the weight optimization.

    Returns:
        :class:`SyntheticControlResult`. ``effects[n_pre:]`` estimate the
        treatment effects only if the pre-treatment fit is good
        (``pre_rmspe`` small relative to the outcome scale); check
        ``converged``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, only one of the predictor
            arguments is given, or an argument is out of range.

    Equation:
        [EQ-CAUSAL-SC-001], [EQ-CAUSAL-SC-003].

    References:
        [REF-ABADIE-DIAMOND-HAINMUELLER-2010].

    """
    y1, y0 = _validated_panel(treated_outcome, donor_outcomes, n_pre)
    if (treated_predictors is None) != (donor_predictors is None):
        raise ValueError(
            "treated_predictors and donor_predictors must be given together"
        )
    if treated_predictors is None or donor_predictors is None:
        x1: Vector = y1[:n_pre]
        x0: Matrix = tuple(row[:n_pre] for row in y0)
    else:
        x1 = as_vector(treated_predictors, name="treated_predictors")
        x0 = as_matrix(donor_predictors, name="donor_predictors")
        if len(x0) != len(y0):
            raise ValueError("donor_predictors needs one row per donor")
    fit = synthetic_control_weights(
        x1,
        x0,
        importance=importance,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    synthetic = tuple(
        math.fsum(w * row[t] for w, row in zip(fit.weights, y0, strict=True))
        for t in range(len(y1))
    )
    effects = tuple(a - b for a, b in zip(y1, synthetic, strict=True))
    pre, post = _rmspe(effects[:n_pre]), _rmspe(effects[n_pre:])
    if pre > 0.0:
        ratio = post / pre
    else:
        ratio = math.inf if post > 0.0 else math.nan
    return SyntheticControlResult(
        weights=fit.weights,
        synthetic=synthetic,
        effects=effects,
        n_pre=n_pre,
        pre_rmspe=pre,
        post_rmspe=post,
        rmspe_ratio=ratio,
        average_effect=math.fsum(effects[n_pre:]) / (len(y1) - n_pre),
        objective=fit.objective,
        duality_gap=fit.duality_gap,
        iterations=fit.iterations,
        converged=fit.converged,
    )


@dataclass(frozen=True, slots=True)
class PlaceboInference:
    """In-space placebo comparison for a synthetic control.

    Attributes:
        treated: Fit of the actually treated unit.
        placebo_ratios: Post/pre RMSPE ratio of each donor when it is
            treated as if it had received the intervention.
        rank: Number of units (donors and the treated unit) whose ratio is
            at least the treated unit's; ``1`` means the most extreme.
        p_value: ``rank / (J + 1)``. A placebo rank, not a sampling-based
            p-value; its smallest possible value is ``1 / (J + 1)``.
        n_units: ``J + 1``.
        all_converged: Whether every weight optimization converged.

    References:
        [REF-ABADIE-DIAMOND-HAINMUELLER-2010].

    """

    treated: SyntheticControlResult
    placebo_ratios: Vector
    rank: int
    p_value: float
    n_units: int
    all_converged: bool


def synthetic_control_placebo(
    treated_outcome: Iterable[object],
    donor_outcomes: Iterable[Iterable[object]],
    n_pre: int,
    *,
    treated_in_placebo_pools: bool = True,
    importance: Iterable[object] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> PlaceboInference:
    """Rank the treated unit's RMSPE ratio among in-space placebos.

    Each donor in turn is treated as if it had received the intervention
    and a synthetic control is built for it from the remaining donors
    (plus the truly treated unit when ``treated_in_placebo_pools``). The
    statistic is the ratio of post- to pre-treatment RMSPE, which
    discounts units that were never fitted well. With ratios ``r_1`` (the
    treated unit) and ``r_2, ..., r_{J+1}``,
    ``p = #{j : r_j >= r_1} / (J + 1)``. A donor with a ``nan`` ratio (zero
    gaps before and after) counts as not exceeding the treated unit; if
    the treated unit's own ratio is ``nan`` the rank is ``J + 1``.

    Predictors are the pre-treatment outcomes.

    Args:
        treated_outcome: ``T`` outcomes of the treated unit.
        donor_outcomes: ``J x T`` donor outcomes, ``J >= 2``.
        n_pre: Number of pre-treatment periods.
        treated_in_placebo_pools: Move the treated unit into the donor
            pool of every placebo run (default) or leave it out.
        importance: Importances of the ``n_pre`` pre-treatment outcomes.
        tolerance: Relative duality-gap tolerance.
        max_iterations: Iteration limit of each weight optimization.

    Returns:
        :class:`PlaceboInference`. The p-value is a placebo rank: it is a
        valid randomization p-value only under random assignment of the
        intervention among the ``J + 1`` units, and says nothing about
        sampling uncertainty.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree or fewer than two donors are given.

    Equation:
        [EQ-CAUSAL-SC-004], [EQ-CAUSAL-SC-003].

    References:
        [REF-ABADIE-DIAMOND-HAINMUELLER-2010].

    """
    y1, y0 = _validated_panel(treated_outcome, donor_outcomes, n_pre)
    if len(y0) < 2:
        raise ValueError("placebo inference needs at least two donors")
    treated = synthetic_control(
        y1,
        y0,
        n_pre,
        importance=importance,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    ratios: list[float] = []
    converged = treated.converged
    for j, path in enumerate(y0):
        pool = [row for i, row in enumerate(y0) if i != j]
        if treated_in_placebo_pools:
            pool.append(y1)
        run = synthetic_control(
            path,
            pool,
            n_pre,
            importance=importance,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        ratios.append(run.rmspe_ratio)
        converged = converged and run.converged
    if math.isnan(treated.rmspe_ratio):
        rank = len(y0) + 1
    else:
        rank = 1 + sum(1 for ratio in ratios if ratio >= treated.rmspe_ratio)
    return PlaceboInference(
        treated=treated,
        placebo_ratios=tuple(ratios),
        rank=rank,
        p_value=rank / (len(y0) + 1),
        n_units=len(y0) + 1,
        all_converged=converged,
    )
