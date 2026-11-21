"""Approximate control variates (ACV-IS and ACV-MF) for multifidelity means.

A classical control variate needs the exact mean of the control. For
low-fidelity models that mean is unknown, so the approximate control variate
of Gorodetsky, Geraci, Eldred and Jakeman replaces it by a second, larger
sample mean of the same model:

``Q = mean_N f_0 + sum_i alpha_i (mean_N f_i - mu_hat_i)``.

``mu_hat_i`` uses ``r_i N`` inputs, either nested in one shared sequence
(ACV-MF) or the ``N`` shared inputs plus extra inputs drawn independently for
every model (ACV-IS). The differences have expectation zero, so the estimator
is unbiased for any fixed weights; the optimal weights and the variance are
functions of the model covariance ``C`` and the ratios ``r_i``. ``C`` must be
independent of the estimation samples (see
:func:`probstat.monte_carlo.multifidelity.pilot_statistics`). Model 0 is the
high-fidelity model.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Matrix, Vector
from probstat.optimization import multi_start

from .multifidelity import (
    _MAX_EXHAUSTIVE_MODELS,
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
)

ACVKind = Literal["MF", "IS"]


@dataclass(frozen=True, slots=True)
class ACVDesign:
    """Sample design, optimal weights, and variance of an ACV estimator.

    Args:
        kind: ``"MF"`` (nested low-fidelity samples) or ``"IS"`` (independent
            extra samples per model).
        samples: Evaluations ``n_i`` of every model: ``n_0 = N`` for the
            high-fidelity model, ``n_i = r_i N > N`` for a control, and 0 for
            a model that is not used.
        weights: Control-variate weights ``alpha_i`` for models ``1..k-1``
            (0 for unused models); the estimator adds
            ``alpha_i (mean_N f_i - mu_hat_i)``.
        variance: Predicted estimator variance.
        mc_variance: Variance ``C_00 / N`` of the high-fidelity sample mean
            that the controls correct.
        cost: ``sum_i w_i n_i``.

    References:
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """

    kind: ACVKind
    samples: tuple[int, ...]
    weights: Vector
    variance: float
    mc_variance: float
    cost: float

    @property
    def ratios(self) -> Vector:
        """Return ``r_i = n_i / N`` for models ``1..k-1`` (0 when unused).

        Equation:
            [EQ-MC-MF-005].

        References:
            [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

        """
        base = self.samples[0]
        return tuple(count / base for count in self.samples[1:])

    @property
    def variance_ratio(self) -> float:
        """Return ``variance / mc_variance`` at equal high-fidelity samples.

        Equation:
            [EQ-MC-MF-005].

        References:
            [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

        """
        return self.variance / self.mc_variance


@dataclass(frozen=True, slots=True)
class ACVAllocation:
    """Result of the numerical ACV budget allocation.

    Args:
        design: Integer design that respects the budget.
        continuous_samples: Real-valued sample sizes of the best local
            optimum found for the selected control subset (0 for unused
            models).
        continuous_variance: Predicted variance at ``continuous_samples``.
        budget: Cost budget ``p``.
        converged: Whether the local optimizer reported convergence for the
            selected subset. The problem is not convex, so this is a local
            statement, never a certificate of global optimality.
        message: Optimizer message of the selected run.

    References:
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """

    design: ACVDesign
    continuous_samples: Vector
    continuous_variance: float
    budget: float
    converged: bool
    message: str


def _as_kind(kind: str) -> ACVKind:
    """Return a validated ACV sampling scheme name.

    Raises:
        ValueError: If ``kind`` is neither ``"MF"`` nor ``"IS"``.

    References:
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020]. No equation is evaluated.

    """
    if kind == "MF":
        return "MF"
    if kind == "IS":
        return "IS"
    raise ValueError("kind must be 'MF' or 'IS'")


def _acv_moments(
    matrix: Matrix,
    base: float,
    active: Sequence[int],
    ratios: Sequence[float],
    kind: ACVKind,
) -> tuple[Vector, float]:
    """Return the optimal ACV weights of the active controls and the variance.

    With ``f_i = (r_i - 1) / r_i``, ``N Cov(D_i, Q_0) = f_i C_0i`` and
    ``N Cov(D_i, D_j) = F_ij C_ij`` where ``F_ii = f_i`` and, off the
    diagonal, ``F_ij = f_i f_j`` (IS) or
    ``F_ij = (min(r_i, r_j) - 1) / min(r_i, r_j)`` (MF). The weights are
    ``alpha = -Cov(D, D)^-1 Cov(D, Q_0)`` and the variance is
    ``(C_00 + alpha' N Cov(D, Q_0)) / N``.

    Raises:
        ValueError: If the control covariance is numerically singular.

    References:
        [EQ-MC-MF-005], [EQ-MC-MF-006], [EQ-MC-MF-007],
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """
    if not active:
        return (), matrix[0][0] / base
    factors = [(r - 1.0) / r for r in ratios]
    size = len(active)
    control = [[0.0] * size for _ in range(size)]
    for a in range(size):
        for b in range(size):
            if a == b:
                factor = factors[a]
            elif kind == "IS":
                factor = factors[a] * factors[b]
            else:
                smaller = min(ratios[a], ratios[b])
                factor = (smaller - 1.0) / smaller
            control[a][b] = factor * matrix[active[a]][active[b]]
    cross = [factors[a] * matrix[0][active[a]] for a in range(size)]
    solution = _solve_spd(control, cross)
    weights = tuple(-value for value in solution)
    reduced = matrix[0][0] + math.fsum(
        w * c for w, c in zip(weights, cross, strict=True)
    )
    return weights, reduced / base


def acv_design(
    covariance: Iterable[Iterable[object]],
    costs: Sequence[float],
    samples: Sequence[int],
    *,
    kind: str = "MF",
) -> ACVDesign:
    """Return the optimal weights and variance of an ACV sample design.

    The estimator is
    ``Q = mean_N f_0 + sum_i alpha_i (mean_N f_i - mu_hat_i)`` where all
    ``mean_N`` use the same ``N = n_0`` inputs and ``mu_hat_i`` is the mean
    of ``f_i`` over ``n_i = r_i N`` inputs:

    * ``kind="MF"``: the first ``n_i`` inputs of one shared sequence that
      starts with the ``N`` high-fidelity inputs (nested samples);
    * ``kind="IS"``: the ``N`` high-fidelity inputs plus ``n_i - N`` extra
      inputs drawn independently for every model.

    Writing ``D_i = mean_N f_i - mu_hat_i``, the optimal weights and the
    variance are ``alpha = -Cov(D, D)^-1 Cov(D, Q_0)`` and
    ``Var = C_00 / N - Cov(D, Q_0)' Cov(D, D)^-1 Cov(D, Q_0)`` with the
    covariances of [EQ-MC-MF-006] (IS) or [EQ-MC-MF-007] (MF). As every
    ``r_i`` grows the variance tends to the classical control-variate limit
    ``(C_00 - c' C_LL^-1 c) / N`` with all low-fidelity means known.

    Args:
        covariance: Model covariance matrix (high-fidelity model first),
            independent of the estimation samples.
        costs: Positive cost of one evaluation of every model.
        samples: Evaluations ``n_i`` of every model: ``n_0 >= 1``; for
            ``i >= 1`` either ``n_i > n_0`` or ``n_i = 0`` (model unused).
        kind: ``"MF"`` or ``"IS"``.

    Returns:
        Design with weights, predicted variance, and cost.

    Raises:
        ValueError: If a sample count is out of domain, ``kind`` is unknown,
            or an input is invalid.

    Equation:
        [EQ-MC-MF-005], [EQ-MC-MF-006], [EQ-MC-MF-007].

    References:
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """
    matrix = _as_covariance(covariance)
    size = len(matrix)
    weights = _as_costs(costs, size)
    scheme = _as_kind(kind)
    counts = tuple(samples)
    if len(counts) != size:
        raise ValueError("samples must have one entry per model")
    base = _as_count(counts[0], name="samples[0]", minimum=1)
    for index in range(1, size):
        count = _as_count(counts[index], name=f"samples[{index}]", minimum=0)
        if 0 < count <= base:
            raise ValueError(
                f"samples[{index}] must be 0 (unused) or exceed the "
                f"high-fidelity sample size {base}"
            )
    active = [i for i in range(1, size) if counts[i] > 0]
    ratios = [counts[i] / base for i in active]
    active_weights, variance = _acv_moments(matrix, float(base), active, ratios, scheme)
    alpha = [0.0] * (size - 1)
    for index, value in zip(active, active_weights, strict=True):
        alpha[index - 1] = value
    return ACVDesign(
        kind=scheme,
        samples=counts,
        weights=tuple(alpha),
        variance=variance,
        mc_variance=matrix[0][0] / base,
        cost=math.fsum(w * n for w, n in zip(weights, counts, strict=True)),
    )


def _softmax(logits: Sequence[float]) -> list[float]:
    """Return the softmax of ``logits`` (a point in the open simplex).

    References:
        [EQ-MC-MF-008], [REF-NOCEDAL-WRIGHT-2006, Chapter 2].

    """
    top = max(logits)
    powers = [math.exp(value - top) for value in logits]
    total = math.fsum(powers)
    return [value / total for value in powers]


def _acv_continuous(
    weights: Vector,
    budget: float,
    active: Sequence[int],
    logits: Sequence[float],
) -> tuple[float, list[float]]:
    """Map budget-share logits to ``(N, [r_i])`` for the active controls.

    The share ``theta_0`` pays ``N (w_0 + sum_active w_i)`` (every active
    model on the ``N`` shared inputs) and ``theta_i`` pays the
    ``(r_i - 1) N w_i`` additional evaluations of model ``i``.

    References:
        [EQ-MC-MF-008], [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """
    shares = _softmax(logits)
    shared_cost = weights[0] + math.fsum(weights[i] for i in active)
    base = budget * shares[0] / shared_cost
    ratios = [
        1.0 + budget * share / (base * weights[i])
        for i, share in zip(active, shares[1:], strict=True)
    ]
    return base, ratios


def _acv_round(
    matrix: Matrix,
    weights: Vector,
    budget: float,
    active: Sequence[int],
    base: float,
    ratios: Sequence[float],
    kind: ACVKind,
) -> ACVDesign | None:
    """Round a continuous ACV design down to integers within the budget.

    ``N = floor(N_c)`` and ``n_i = max(N + 1, floor(r_i N_c))``. Returns
    ``None`` when ``N < 1`` or the rounded design exceeds the budget.

    References:
        [EQ-MC-MF-008], [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """
    whole = math.floor(base * (1.0 + 1e-12))
    if whole < 1:
        return None
    counts = [0] * len(matrix)
    counts[0] = whole
    for index, ratio in zip(active, ratios, strict=True):
        counts[index] = max(whole + 1, math.floor(ratio * base * (1.0 + 1e-12)))
    cost = math.fsum(w * n for w, n in zip(weights, counts, strict=True))
    if cost > budget * (1.0 + 1e-12):
        return None
    return acv_design(matrix, weights, counts, kind=kind)


def acv_allocation(
    covariance: Iterable[Iterable[object]],
    costs: Sequence[float],
    budget: float,
    *,
    kind: str = "MF",
    starts: int = 4,
    seed: int | RandomStream | None = 0,
) -> ACVAllocation:
    """Choose ACV sample sizes for a cost budget by numerical optimization.

    For every subset ``S`` of the low-fidelity models the predicted variance
    [EQ-MC-MF-005] is minimized over the high-fidelity sample size ``N`` and
    the ratios ``r_i > 1`` subject to ``N (w_0 + sum_S w_i r_i) = p``. The
    constraint is removed by the budget shares
    ``theta = softmax(z)``: ``N = p theta_0 / (w_0 + sum_S w_i)`` and
    ``r_i = 1 + p theta_i / (N w_i)``, and ``log Var`` is minimized by the
    Nelder-Mead simplex method from ``starts`` start points (the equal
    shares and seeded random logits in ``[-3, 3]``). A derivative-free
    method is used because the ACV-MF variance has kinks where two ratios
    cross (``min(r_i, r_j)``), and optima often lie on such a kink. Each
    local optimum is rounded down to integers, and the subset whose integer
    design has the smallest predicted variance is returned.

    This is a heuristic: the problem is not convex in ``(N, r)``, the search
    is local, and no optimality certificate is produced. The subset search is
    exhaustive (``2^(k-1)`` subsets, at most 12 models). Plain Monte Carlo
    (no control) is always a candidate, so the result is never worse than
    ``C_00 / floor(p / w_0)``.

    Args:
        covariance: Model covariance matrix (high-fidelity model first),
            independent of the estimation samples.
        costs: Positive cost of one evaluation of every model.
        budget: Positive cost budget ``p`` (at least ``w_0``).
        kind: ``"MF"`` or ``"IS"``.
        starts: Start points per subset (at least 1).
        seed: Seed of the random starts; the default 0 makes the allocation
            deterministic.

    Returns:
        Integer design, the continuous optimum it was rounded from, and the
        convergence flag of the selected local run.

    Raises:
        ValueError: If the budget is below ``w_0``, there are more than 12
            models, or an input is invalid.

    Equation:
        [EQ-MC-MF-005], [EQ-MC-MF-008].

    References:
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """
    matrix = _as_covariance(covariance)
    size = len(matrix)
    weights = _as_costs(costs, size)
    spend = _as_budget(budget)
    scheme = _as_kind(kind)
    count = _as_count(starts, name="starts", minimum=1)
    if size > _MAX_EXHAUSTIVE_MODELS:
        raise ValueError(
            f"exhaustive selection supports at most {_MAX_EXHAUSTIVE_MODELS} models"
        )
    if spend < weights[0]:
        raise ValueError("budget is below the cost of one high-fidelity evaluation")
    root = resolve_stream(seed)
    plain = acv_design(
        matrix,
        weights,
        [math.floor(spend / weights[0] * (1.0 + 1e-12))] + [0] * (size - 1),
        kind=scheme,
    )
    plain_continuous = (spend / weights[0], *([0.0] * (size - 1)))
    best = ACVAllocation(
        design=plain,
        continuous_samples=plain_continuous,
        continuous_variance=matrix[0][0] * weights[0] / spend,
        budget=spend,
        converged=True,
        message="plain Monte Carlo (no control selected)",
    )
    for extra in range(1, size):
        for chosen in itertools.combinations(range(1, size), extra):
            candidate = _acv_subset(matrix, weights, spend, chosen, scheme, count, root)
            if candidate is not None and (
                candidate.design.variance < best.design.variance * (1.0 - 1e-12)
            ):
                best = candidate
    return best


def _acv_subset(
    matrix: Matrix,
    weights: Vector,
    budget: float,
    active: tuple[int, ...],
    kind: ACVKind,
    starts: int,
    root: RandomStream,
) -> ACVAllocation | None:
    """Optimize the ACV budget shares for one subset of controls.

    Returns ``None`` when no local optimum rounds to an affordable integer
    design.

    References:
        [EQ-MC-MF-008], [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """

    def objective(logits: Vector) -> float:
        """Return the log of the predicted variance at budget-share logits.

        References:
            [EQ-MC-MF-005], [EQ-MC-MF-008].

        """
        base, ratios = _acv_continuous(weights, budget, active, logits)
        if any(ratio <= 1.0 for ratio in ratios):
            raise ValueError("a budget share underflowed")
        variance = _acv_moments(matrix, base, active, ratios, kind)[1]
        if variance <= 0.0:
            raise ValueError("predicted variance is not positive")
        return math.log(variance)

    dimension = len(active) + 1
    try:
        search = multi_start(
            objective,
            [(-3.0, 3.0)] * dimension,
            starts=starts,
            seed=root.child("acv-allocation", *active),
            method="nelder-mead",
            x0=[0.0] * dimension,
            bounded=False,
            initial_step=0.5,
            tolerance=1e-12,
            max_evaluations=400 * dimension + 2000,
        )
    except ValueError:
        return None
    best: ACVAllocation | None = None
    for run in search.runs:
        if not math.isfinite(run.value):
            continue
        base, ratios = _acv_continuous(weights, budget, active, run.x)
        design = _acv_round(matrix, weights, budget, active, base, ratios, kind)
        if design is None:
            continue
        if best is None or design.variance < best.design.variance:
            continuous = [0.0] * len(matrix)
            continuous[0] = base
            for index, ratio in zip(active, ratios, strict=True):
                continuous[index] = ratio * base
            best = ACVAllocation(
                design=design,
                continuous_samples=tuple(continuous),
                continuous_variance=math.exp(run.value),
                budget=budget,
                converged=run.converged,
                message=run.message,
            )
    return best


def acv_estimate(
    models: Sequence[ModelFunction],
    input_sampler: InputSampler,
    design: ACVDesign,
    *,
    seed: int | RandomStream | None = None,
) -> MultifidelityEstimate:
    """Estimate the high-fidelity mean with an approximate control variate.

    ``Q = mean_N f_0 + sum_i alpha_i (mean_N f_i - mu_hat_i)`` with the
    weights and sample sizes of ``design``.

    * ``kind="MF"``: one input sequence of length ``max_i n_i`` is drawn
      from ``root.child("acv-mf")``; all models use its first ``N`` entries
      for ``mean_N`` and model ``i`` its first ``n_i`` entries for
      ``mu_hat_i``.
    * ``kind="IS"``: the ``N`` shared inputs come from
      ``root.child("acv-is", "shared")`` and the ``n_i - N`` extra inputs of
      model ``i`` from ``root.child("acv-is", "extra", i)``; ``mu_hat_i`` is
      the mean over both sets.

    The estimator is unbiased for any fixed weights; weights from a pilot
    need a pilot that is independent of the inputs drawn here.

    Args:
        models: All model callables, indexed as in the covariance matrix
            that produced ``design``.
        input_sampler: ``input_sampler(n, stream)`` returning ``n``
            independent inputs.
        design: Result of :func:`acv_design` or the ``design`` of
            :func:`acv_allocation`.
        seed: Root seed or stream.

    Returns:
        Estimate with the design's predicted variance and cost.

    Raises:
        ValueError: If the design does not match the number of models, the
            sampler returns the wrong number of inputs, or an output is not
            finite.

    Equation:
        [EQ-MC-MF-005].

    References:
        [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020].

    """
    functions = tuple(models)
    if not isinstance(design, ACVDesign):
        raise ValueError("design must be an ACVDesign")
    if len(design.samples) != len(functions):
        raise ValueError("design and models disagree on the number of models")
    root = resolve_stream(seed)
    base = design.samples[0]
    active = [i for i in range(1, len(functions)) if design.samples[i] > 0]
    if design.kind == "MF":
        inputs = _draw_inputs(input_sampler, max(design.samples), root.child("acv-mf"))
        shared = inputs[:base]
    else:
        shared = _draw_inputs(input_sampler, base, root.child("acv-is", "shared"))
        inputs = []
    high_fidelity_mean = _mean(_evaluate(functions[0], shared, 0))
    estimate = high_fidelity_mean
    for index in active:
        count = design.samples[index]
        if design.kind == "MF":
            points = inputs[:count]
        else:
            points = shared + _draw_inputs(
                input_sampler, count - base, root.child("acv-is", "extra", index)
            )
        values = _evaluate(functions[index], points, index)
        estimate += design.weights[index - 1] * (_mean(values[:base]) - _mean(values))
    return MultifidelityEstimate(
        method=f"ACV-{design.kind}",
        estimate=estimate,
        variance=design.variance,
        high_fidelity_mean=high_fidelity_mean,
        high_fidelity_samples=base,
        evaluations=design.samples,
        cost=design.cost,
    )
