"""Operator-path, sequential and robust extensions of linear optimal design.

Experimental (added in 0.28).

:mod:`probstat.design.linear_optimal` forms the dense posterior covariance of
every design. This module adds four things for linear Gaussian problems
``y = A x + e``, ``x ~ N(m, P)``, ``e ~ N(0, R)``:

* **Operator-path criteria** that never form a dense posterior covariance.
  The A criterion ``tr(C)`` with ``C = (P^-1 + B'B)^-1`` (``B = R^-1/2 A_S``)
  is estimated by Hutchinson probes with an inner conjugate-gradient solve of
  the posterior precision per probe, and reports the probe count, the CG
  iterations and stop reasons, and the standard error of the trace estimate
  (EQ-DES-OPX-001). The D criterion ``log det(I + H)`` is
  ``sum_i log(1 + lambda_i)`` over the eigenvalues of the prior-preconditioned
  data-misfit Hessian, obtained from Lanczos Ritz values or a randomized SVD,
  with a stated truncation bound (EQ-DES-OPX-002).
* **Stochastic greedy design** on the operator-path A criterion with common
  random numbers across candidates: every candidate is scored with the same
  probe vectors, so differences between candidates are paired
  (EQ-DES-OPX-003).
* **Sequential design**: choose, simulate the observation from a separate
  stream, update the posterior exactly, choose again, and compare the realised
  posterior against a batch design of the same size (EQ-DES-OPX-004).
* **Robust design**: a Bayesian-average (expected) local design over samples of
  an uncertain nonlinear parameter, whose Jacobian varies with the parameter,
  and a worst-case design over a finite set of prior/noise scenarios, each
  with an exhaustive oracle (EQ-DES-OPX-005, EQ-DES-OPX-006).

Assumptions:
    Linear observations, Gaussian prior and noise, known covariances. The
    prior is any covariance object of :mod:`probstat.linalg.covariance_operators`
    on the operator path and a dense matrix elsewhere.

Common pitfalls:
    - The Hutchinson A estimate has Monte Carlo error. Differences between
      candidates are meaningful only relative to the *paired* standard error
      reported with each greedy step.
    - In a linear Gaussian problem the posterior covariance does not depend on
      the data, so a sequential design with a fixed criterion selects exactly
      the batch design; adaptivity pays only when the criterion depends on the
      data (for example through a posterior-mean-dependent target).
    - The D truncation bound is an upper bound on the *underestimate*; it is
      not a confidence interval.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.design.linear_optimal import (
    _check_candidates,
    _check_indices,
    _check_prior,
    greedy_linear_design,
    linear_design_criteria,
)
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Matrix, Vector, as_vector, cholesky
from probstat.linalg.krylov import (
    conjugate_gradient,
    hutchinson_trace,
    lanczos_eigenpairs,
    randomized_svd,
)
from probstat.linalg.operators import CallbackOperator, ComposedOperator, DenseOperator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_prior_object(prior: Any, dimension: int) -> None:
    """Require a covariance object of the right dimension.

    Raises:
        TypeError: If ``prior`` lacks ``solve``, ``apply`` or ``dimension``.
        ValueError: If its dimension differs from the state dimension.

    References:
        [EQ-DES-OPX-001]. No equation is evaluated.

    """
    for name in ("solve", "apply", "factor_operator", "dimension"):
        if not hasattr(prior, name):
            raise TypeError(
                "prior must be a covariance object from "
                f"probstat.linalg.covariance_operators (missing {name!r})"
            )
    if prior.dimension != dimension:
        raise ValueError("prior dimension must equal the number of candidate columns")


def _whitened_selection(
    rows: Matrix, variances: Vector, indices: tuple[int, ...]
) -> Matrix:
    """Return ``R^{-1/2} A_S`` for independent noise (row ``j`` over ``s_j``).

    References:
        [EQ-DES-OPX-001]. No equation is evaluated.

    """
    return tuple(
        tuple(value / math.sqrt(variances[i]) for value in rows[i]) for i in indices
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the inner product with compensated summation.

    References:
        [REF-GOLUB-VANLOAN-2013]. No equation is evaluated.

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _independent_inputs(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    selected: Sequence[int],
    prior: Any,
) -> tuple[Matrix, Vector, tuple[int, ...]]:
    """Validate rows, variances, selection and prior of the operator path.

    References:
        [EQ-DES-OPX-001]. No equation is evaluated.

    """
    rows, variances = _check_candidates(candidate_rows, noise_variances)
    assert variances is not None
    indices = _check_indices(selected, len(rows))
    _check_prior_object(prior, len(rows[0]))
    return rows, variances, indices


# ---------------------------------------------------------------------------
# Operator-path A criterion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatorACriterion:
    """Hutchinson estimate of ``tr(C)`` with inner CG solves.

    Attributes:
        trace_estimate: Mean of ``z_k' C z_k`` over the probes.
        standard_error: Standard error of the mean over the probes (``nan``
            for one probe). It covers probe variance only, not CG error.
        probes: Number of Rademacher probes.
        samples: The individual ``z_k' C z_k`` (usable for paired
            differences between designs that share ``seed``).
        cg_iterations: CG iterations of each probe.
        cg_stop_reasons: CG stop reason of each probe.
        all_converged: Whether every inner CG solve converged.
        max_relative_residual: Largest true relative residual
            ``||z - Prec x|| / ||z||`` over the probes.
        seed: Seed that fixed the probes (common random numbers across
            designs scored with the same seed).
        selected: The design (candidate indices).

    References:
        [EQ-DES-OPX-001], [REF-HUTCHINSON-1989], [REF-ALEXANDERIAN-2021].

    """

    trace_estimate: float
    standard_error: float
    probes: int
    samples: Vector
    cg_iterations: tuple[int, ...]
    cg_stop_reasons: tuple[str, ...]
    all_converged: bool
    max_relative_residual: float
    seed: int | RandomStream | None
    selected: tuple[int, ...]


def operator_a_criterion(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    selected: Sequence[int],
    *,
    prior: Any,
    probes: int = 30,
    seed: int | RandomStream | None = 0,
    cg_tolerance: float = 1e-10,
    cg_max_iterations: int | None = None,
) -> OperatorACriterion:
    """Estimate the A criterion ``tr(C)`` without a dense posterior covariance.

    The posterior precision ``Q = P^-1 + B'B``, ``B = R^-1/2 A_S``, is applied
    as an operator (prior solve plus two thin products), and
    ``tr(C) = tr(Q^-1) = E[z' Q^-1 z]`` for Rademacher ``z`` is estimated by
    :func:`probstat.linalg.krylov.hutchinson_trace` whose operator is "solve
    ``Q x = z`` by conjugate gradients". The estimate is unbiased up to the CG
    residual and has the standard error reported (EQ-DES-OPX-001). Nothing of
    size ``n`` by ``n`` is formed.

    Args:
        candidate_rows: ``N`` by ``n`` observation rows.
        noise_variances: Positive noise variance per candidate (independent).
        selected: Candidate indices of the design (may be empty or repeat).
        prior: Covariance object (``solve`` and ``apply``) of dimension ``n``.
        probes: Number of Hutchinson probes (positive).
        seed: Seed or stream fixing the probes.
        cg_tolerance: Relative CG residual tolerance.
        cg_max_iterations: CG iteration cap (default ``2 n``).

    Returns:
        :class:`OperatorACriterion`; non-converged inner solves are flagged in
        ``all_converged`` and ``cg_stop_reasons``, never hidden.

    Raises:
        TypeError: If ``prior`` is not a covariance object or an index is not
            an integer.
        ValueError: For invalid shapes, indices, probes or tolerances.

    Equation:
        [EQ-DES-OPX-001].

    References:
        [REF-HUTCHINSON-1989], [REF-HESTENES-STIEFEL-1952],
        [REF-ALEXANDERIAN-2021].

    """
    rows, variances, indices = _independent_inputs(
        candidate_rows, noise_variances, selected, prior
    )
    n = len(rows[0])
    thin = _whitened_selection(rows, variances, indices)
    results: list[Any] = []

    def precision(v: Vector) -> Vector:
        """Apply ``P^-1 v + B'B v`` without forming a matrix.

        References:
            [EQ-DES-OPX-001].

        """
        out = list(prior.solve(v))
        for row in thin:
            weight = _dot(row, v)
            if weight != 0.0:
                for i in range(n):
                    out[i] += row[i] * weight
        return tuple(out)

    def solve(z: Vector) -> Vector:
        """Return ``Q^-1 z`` by CG and record the solver result.

        References:
            [EQ-DES-OPX-001], [REF-HESTENES-STIEFEL-1952].

        """
        outcome = conjugate_gradient(
            CallbackOperator((n, n), precision, precision, name="posterior precision"),
            z,
            tolerance=cg_tolerance,
            max_iterations=cg_max_iterations,
        )
        results.append((outcome, math.sqrt(_dot(z, z))))
        return outcome.solution

    estimate = hutchinson_trace(
        CallbackOperator((n, n), solve, solve, name="posterior covariance"),
        probes=probes,
        seed=seed,
    )
    relative = max((r.true_residual_norm / scale for r, scale in results), default=0.0)
    return OperatorACriterion(
        trace_estimate=estimate.estimate,
        standard_error=estimate.standard_error,
        probes=estimate.probes,
        samples=estimate.samples,
        cg_iterations=tuple(r.iterations for r, _ in results),
        cg_stop_reasons=tuple(r.stop_reason for r, _ in results),
        all_converged=all(r.converged for r, _ in results),
        max_relative_residual=relative,
        seed=seed,
        selected=indices,
    )


# ---------------------------------------------------------------------------
# Operator-path D criterion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatorDCriterion:
    """Spectral estimate of ``log det(I + H)``.

    Attributes:
        log_det: ``sum_i log(1 + theta_i)`` over the retained approximate
            eigenvalues of the prior-preconditioned data-misfit Hessian.
        expected_information_gain: ``log_det / 2`` (nats).
        eigenvalues: The retained approximate eigenvalues ``theta_i``
            (descending).
        rank: Number of retained eigenvalues (the truncation rank).
        data_dimension: Number ``k`` of selected observations (the number of
            nonzero eigenvalues is at most ``min(k, n)``).
        trace_misfit: Exact ``tr(H) = sum_j a_j' P a_j / s_j^2``.
        truncation_bound: ``max(min(tr(H) - sum theta_i, m log(1 + tr(H) / m)
            - estimate), 0)`` with ``m = min(k, n)``: the first term uses
            ``log(1 + x) <= x``, the second Jensen's inequality for the
            concave ``log(1 + x)`` over ``m`` eigenvalues of sum ``tr(H)``. It
            bounds ``true - estimate`` from above in exact arithmetic (the
            estimate never exceeds the truth because Ritz and sketched values
            are lower bounds of the eigenvalues).
        exact: Whether ``rank >= min(k, n)`` so the whole spectrum is
            retained (the bound is then only rounding).
        method: ``"lanczos"`` or ``"randomized_svd"``.
        selected: The design.

    References:
        [EQ-DES-OPX-002], [REF-LANCZOS-1950], [REF-HALKO-MARTINSSON-TROPP-2011].

    """

    log_det: float
    expected_information_gain: float
    eigenvalues: Vector
    rank: int
    data_dimension: int
    trace_misfit: float
    truncation_bound: float
    exact: bool
    method: str
    selected: tuple[int, ...]


def operator_d_criterion(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    selected: Sequence[int],
    *,
    prior: Any,
    rank: int | None = None,
    method: str = "lanczos",
    seed: int | RandomStream | None = 0,
    oversampling: int = 5,
    power_iterations: int = 2,
) -> OperatorDCriterion:
    """Estimate ``log det(I + H)`` from a truncated spectrum of ``H``.

    ``H = P^1/2 A_S' R^-1 A_S P^1/2`` has the same nonzero eigenvalues as the
    ``k`` by ``k`` data-space operator ``M = W W'`` with
    ``W = R^-1/2 A_S P^1/2``, and ``log det(I + H) = sum_i log(1 + lambda_i)``
    (EQ-DES-OPX-002). ``method="lanczos"`` takes ``rank`` Lanczos steps on
    ``M`` (prior applications only); ``method="randomized_svd"`` takes the
    squared singular values of a randomized SVD of ``W`` (prior factor
    applications). Both return lower bounds ``theta_i <= lambda_i`` of the
    leading eigenvalues, so the estimate is a lower bound; its deficiency is at
    most the smaller of ``tr(H) - sum theta_i`` and the Jensen bound
    ``m log(1 + tr(H)/m) - estimate``, with ``tr(H)`` computed exactly.

    Args:
        candidate_rows: ``N`` by ``n`` observation rows.
        noise_variances: Positive noise variance per candidate (independent).
        selected: Candidate indices of the design.
        prior: Covariance object of dimension ``n``.
        rank: Truncation rank (default ``min(k, n)``, the whole spectrum).
        method: ``"lanczos"`` or ``"randomized_svd"``.
        seed: Seed or stream for the start vector / test matrix.
        oversampling: Extra samples of the randomized SVD.
        power_iterations: Power passes of the randomized SVD.

    Returns:
        :class:`OperatorDCriterion`; an empty design gives ``0``.

    Raises:
        TypeError: If ``prior`` is not a covariance object.
        ValueError: For an unknown method, a rank outside ``1..min(k, n)`` or
            invalid inputs.

    Equation:
        [EQ-DES-OPX-002].

    References:
        [REF-LANCZOS-1950], [REF-HALKO-MARTINSSON-TROPP-2011],
        [REF-ALEXANDERIAN-2021].

    """
    if method not in ("lanczos", "randomized_svd"):
        raise ValueError("method must be 'lanczos' or 'randomized_svd'")
    rows, variances, indices = _independent_inputs(
        candidate_rows, noise_variances, selected, prior
    )
    n = len(rows[0])
    k = len(indices)
    if k == 0:
        return OperatorDCriterion(0.0, 0.0, (), 0, 0, 0.0, 0.0, True, method, indices)
    full = min(k, n)
    chosen_rank = (
        full if rank is None else require_non_negative_integer(rank, name="rank")
    )
    if not 1 <= chosen_rank <= full:
        raise ValueError(f"rank must satisfy 1 <= rank <= min(k, n) = {full}")
    thin = _whitened_selection(rows, variances, indices)
    trace_misfit = math.fsum(_dot(row, prior.apply(row)) for row in thin)
    if method == "lanczos":

        def misfit(v: Vector) -> Vector:
            """Apply ``M v = W W' v = B P B' v`` in data space.

            References:
                [EQ-DES-OPX-002].

            """
            back = [math.fsum(thin[r][i] * v[r] for r in range(k)) for i in range(n)]
            pushed = prior.apply(tuple(back))
            return tuple(_dot(row, pushed) for row in thin)

        ritz = lanczos_eigenpairs(
            CallbackOperator((k, k), misfit, misfit, name="data-space Hessian"),
            chosen_rank,
            seed=seed,
        )
        theta = tuple(ritz.eigenvalues[:chosen_rank])
    else:
        sketch = randomized_svd(
            ComposedOperator([DenseOperator(thin), prior.factor_operator()]),
            chosen_rank,
            oversampling=oversampling,
            power_iterations=power_iterations,
            seed=seed,
        )
        theta = tuple(s * s for s in sketch.singular_values[:chosen_rank])
    log_det = math.fsum(math.log1p(t) for t in theta)
    jensen = full * math.log1p(trace_misfit / full)
    bound = max(min(trace_misfit - math.fsum(theta), jensen - log_det), 0.0)
    return OperatorDCriterion(
        log_det=log_det,
        expected_information_gain=0.5 * log_det,
        eigenvalues=theta,
        rank=len(theta),
        data_dimension=k,
        trace_misfit=trace_misfit,
        truncation_bound=bound,
        exact=chosen_rank >= full,
        method=method,
        selected=indices,
    )


# ---------------------------------------------------------------------------
# Stochastic greedy design on the operator-path A criterion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatorGreedyResult:
    """Sensors chosen greedily on the operator-path A criterion.

    Attributes:
        indices: Chosen candidate indices in selection order.
        trace_estimates: Hutchinson ``tr(C)`` of the design after each step.
        standard_errors: Their (unpaired) standard errors.
        candidate_estimates: Per step, the estimate of every candidate
            (``nan`` for already chosen candidates).
        margins: Per step, runner-up estimate minus winner estimate on the
            shared probes (``nan`` when there is no runner-up).
        margin_standard_errors: Standard error of each paired margin.
        probes: Probes per evaluation.
        seed: Seed shared by every evaluation.
        common_random_numbers: Always ``True``: every candidate of every step
            is scored with the same probe vectors.
        all_converged: Whether every inner CG solve of the run converged.
        cg_solves: Total inner CG solves.
        stop_reason: ``"size_reached"`` or ``"no_candidates_left"``.

    References:
        [EQ-DES-OPX-003], [REF-HUTCHINSON-1989].

    """

    indices: tuple[int, ...]
    trace_estimates: Vector
    standard_errors: Vector
    candidate_estimates: tuple[Vector, ...]
    margins: Vector
    margin_standard_errors: Vector
    probes: int
    seed: int | RandomStream | None
    common_random_numbers: bool
    all_converged: bool
    cg_solves: int
    stop_reason: str


def _paired_se(first: Vector, second: Vector) -> float:
    """Return the standard error of the mean of ``first - second``.

    References:
        [EQ-DES-OPX-003]. Probe differences are i.i.d.; no equation of the
        registry is evaluated beyond the sample standard error.

    """
    count = len(first)
    if count < 2:
        return math.nan
    diffs = [a - b for a, b in zip(first, second, strict=True)]
    mean = math.fsum(diffs) / count
    var = math.fsum((d - mean) ** 2 for d in diffs) / (count - 1)
    return math.sqrt(var / count)


def greedy_operator_a_design(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    *,
    prior: Any,
    size: int,
    probes: int = 30,
    seed: int | RandomStream | None = 0,
    cg_tolerance: float = 1e-10,
    allow_repeats: bool = False,
) -> OperatorGreedyResult:
    """Select ``size`` sensors greedily by the stochastic operator-path A value.

    At every step each remaining candidate ``j`` is scored by
    :func:`operator_a_criterion` of ``S + {j}`` and the smallest estimate wins
    (ties to the lowest index). All scores use the same ``seed`` and therefore
    the same Rademacher probes (common random numbers): the candidates are
    compared on identical probes, so the paired margin standard error, not the
    unpaired one, measures whether the winner is resolved (EQ-DES-OPX-003).
    The result is stochastic; with few probes a near-tie can be resolved
    wrongly, which the margin diagnostics expose.

    Args:
        candidate_rows: ``N`` by ``n`` observation rows.
        noise_variances: Positive noise variance per candidate (independent).
        prior: Covariance object of dimension ``n``.
        size: Number of sensors (``>= 1``).
        probes: Hutchinson probes per evaluation.
        seed: Seed or stream fixing the probes.
        cg_tolerance: Relative CG tolerance.
        allow_repeats: Whether a candidate may be chosen again.

    Returns:
        :class:`OperatorGreedyResult`.

    Raises:
        TypeError: If ``prior`` is not a covariance object.
        ValueError: For invalid inputs or ``size < 1``.

    Equation:
        [EQ-DES-OPX-003].

    References:
        [REF-HUTCHINSON-1989], [REF-ALEXANDERIAN-2021].

    """
    rows, variances = _check_candidates(candidate_rows, noise_variances)
    assert variances is not None
    _check_prior_object(prior, len(rows[0]))
    steps = require_non_negative_integer(size, name="size")
    if steps == 0:
        raise ValueError("size must be positive")
    count = len(rows)
    chosen: list[int] = []
    traces: list[float] = []
    errors: list[float] = []
    table: list[Vector] = []
    margins: list[float] = []
    margin_errors: list[float] = []
    converged = True
    solves = 0
    stop = "size_reached"
    for _ in range(steps):
        scored: dict[int, OperatorACriterion] = {}
        for j in range(count):
            if j in chosen and not allow_repeats:
                continue
            scored[j] = operator_a_criterion(
                rows,
                variances,
                (*chosen, j),
                prior=prior,
                probes=probes,
                seed=seed,
                cg_tolerance=cg_tolerance,
            )
            converged = converged and scored[j].all_converged
            solves += probes
        if not scored:
            stop = "no_candidates_left"
            break
        order = sorted(scored, key=lambda j: (scored[j].trace_estimate, j))
        best = scored[order[0]]
        if len(order) > 1:
            runner = scored[order[1]]
            margins.append(runner.trace_estimate - best.trace_estimate)
            margin_errors.append(_paired_se(runner.samples, best.samples))
        else:
            margins.append(math.nan)
            margin_errors.append(math.nan)
        table.append(
            tuple(
                scored[j].trace_estimate if j in scored else math.nan
                for j in range(count)
            )
        )
        chosen.append(order[0])
        traces.append(best.trace_estimate)
        errors.append(best.standard_error)
    return OperatorGreedyResult(
        indices=tuple(chosen),
        trace_estimates=tuple(traces),
        standard_errors=tuple(errors),
        candidate_estimates=tuple(table),
        margins=tuple(margins),
        margin_standard_errors=tuple(margin_errors),
        probes=probes,
        seed=seed,
        common_random_numbers=True,
        all_converged=converged,
        cg_solves=solves,
        stop_reason=stop,
    )


# ---------------------------------------------------------------------------
# Sequential design
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SequentialDesignResult:
    """Outcome of a simulated sequential design and its batch comparison.

    Attributes:
        indices: Sensors chosen sequentially, in order.
        observations: Simulated observations ``y_t``.
        posterior_means: Posterior mean after each observation.
        trace_history: ``tr(C_t)`` after each observation.
        initial_trace: ``tr(P)``.
        final_mean: Posterior mean after the last observation.
        final_covariance: Posterior covariance after the last observation.
        truth: The state that generated the data.
        squared_errors: ``||m_t - x_true||^2`` after each observation.
        batch_indices: Greedy batch design of the same size (criterion
            ``"a"`` when ``goal_policy`` is given, else ``criterion``).
        batch_trace: ``tr(C)`` of the batch design.
        batch_squared_error: ``||m - x_true||^2`` of the batch posterior using
            the same noise draws for the same sensors.
        criterion: Criterion of the sequential choices.
        adaptive: Whether a ``goal_policy`` made the criterion data-dependent.
        stop_reason: ``"size_reached"`` or the stop reason of the greedy step
            that ended early.

    References:
        [EQ-DES-OPX-004], [REF-ALEXANDERIAN-2021].

    """

    indices: tuple[int, ...]
    observations: Vector
    posterior_means: tuple[Vector, ...]
    trace_history: Vector
    initial_trace: float
    final_mean: Vector
    final_covariance: Matrix
    truth: Vector
    squared_errors: Vector
    batch_indices: tuple[int, ...]
    batch_trace: float
    batch_squared_error: float
    criterion: str
    adaptive: bool
    stop_reason: str


def _kalman_update(
    mean: list[float],
    cov: list[list[float]],
    row: Sequence[float],
    variance: float,
    y: float,
) -> None:
    """Update ``(mean, cov)`` in place with one scalar observation.

    ``S = a'Ca + s^2``, ``m += C a (y - a'm) / S``, ``C -= C a a' C / S``.

    References:
        [EQ-DES-OPX-004], [REF-KAIPIO-SOMERSALO-2007].

    """
    v = [_dot(crow, row) for crow in cov]
    s = _dot(row, v) + variance
    innovation = y - _dot(row, mean)
    for i, vi in enumerate(v):
        mean[i] += vi * innovation / s
        for k, vk in enumerate(v):
            cov[i][k] -= vi * vk / s


def _noise_draw(stream: RandomStream, candidate: int, repeat: int) -> float:
    """Return a standard normal draw keyed by sensor and repetition.

    Keying by ``(candidate, repeat)`` gives the same sensor the same noise in
    the sequential and in the batch run.

    References:
        [EQ-DES-OPX-004]. No equation is evaluated.

    """
    return stream.child("observation", candidate, repeat).generator().gauss(0.0, 1.0)


def sequential_linear_design(
    candidate_rows: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    *,
    prior_mean: Sequence[float],
    prior_covariance: Sequence[Sequence[float]],
    size: int,
    criterion: str = "a",
    goal: Sequence[float] | Sequence[Sequence[float]] | None = None,
    goal_policy: Callable[[Vector], Sequence[float] | Sequence[Sequence[float]]]
    | None = None,
    truth: Sequence[float] | None = None,
    seed: int | RandomStream | None = 0,
    allow_repeats: bool = False,
) -> SequentialDesignResult:
    """Choose, observe, update exactly, choose again; compare with batch.

    Step ``t`` picks the next sensor greedily on the *current* posterior (the
    rank-one rules of :func:`greedy_linear_design`), simulates
    ``y = a'x + s z`` for the true state from a stream separate from the
    truth stream, and updates the posterior mean and covariance by the exact
    Kalman formulas (EQ-DES-OPX-004). The choice at step ``t`` is a function
    of observations ``y_1..y_(t-1)`` only: no look-ahead. The true state is
    ``truth`` or a draw from the prior on stream ``("truth",)``; noise is
    drawn on ``("observation", sensor, repeat)``.

    For a fixed criterion the posterior covariance does not depend on the
    data, so the sequential and the greedy batch design coincide exactly
    (checked in the tests). With ``goal_policy`` (a function of the current
    posterior mean returning the goal ``c`` or ``Q``) the criterion becomes
    data-dependent and the sequential design can differ from the batch design,
    which uses criterion ``"a"``; the realised comparison is reported, not
    assumed.

    Args:
        candidate_rows: ``N`` by ``n`` observation rows.
        noise_variances: Positive noise variance per candidate (independent).
        prior_mean: Prior mean ``m`` (length ``n``).
        prior_covariance: SPD ``n`` by ``n`` prior covariance.
        size: Number of observations (``>= 1``).
        criterion: ``"d"``, ``"a"`` or ``"goal"`` (goal needs ``goal``).
        goal: Fixed goal for ``criterion="goal"``.
        goal_policy: Optional ``mean -> goal`` rule evaluated before every
            step; overrides ``criterion`` and ``goal``.
        truth: The state generating the data (default: prior draw).
        seed: Seed or stream; truth and noise use separate children.
        allow_repeats: Whether a sensor may be used again.

    Returns:
        :class:`SequentialDesignResult`.

    Raises:
        ValueError: For invalid inputs, a non-positive size or a size larger
            than the number of candidates without repeats.

    Equation:
        [EQ-DES-OPX-004].

    References:
        [REF-ALEXANDERIAN-2021], [REF-KAIPIO-SOMERSALO-2007].

    """
    rows, variances = _check_candidates(candidate_rows, noise_variances)
    assert variances is not None
    n = len(rows[0])
    sigma = _check_prior(prior_covariance, n)
    m0 = as_vector(prior_mean, name="prior_mean")
    if len(m0) != n:
        raise ValueError("prior_mean must have one entry per candidate column")
    steps = require_non_negative_integer(size, name="size")
    if steps == 0:
        raise ValueError("size must be positive")
    if not allow_repeats and steps > len(rows):
        raise ValueError("size exceeds the number of candidates without repeats")
    stream = resolve_stream(seed)
    if truth is None:
        lower = cholesky(sigma)
        gen = stream.child("truth").generator()
        z = [gen.gauss(0.0, 1.0) for _ in range(n)]
        x_true: Vector = tuple(
            m0[i] + math.fsum(lower[i][j] * z[j] for j in range(i + 1))
            for i in range(n)
        )
    else:
        x_true = as_vector(truth, name="truth")
        if len(x_true) != n:
            raise ValueError("truth must have one entry per candidate column")
    adaptive = goal_policy is not None
    step_criterion = "goal" if adaptive else criterion
    mean = list(m0)
    cov = [list(row) for row in sigma]
    initial_trace = math.fsum(cov[i][i] for i in range(n))
    chosen: list[int] = []
    uses: dict[int, int] = {}
    ys: list[float] = []
    means: list[Vector] = []
    traces: list[float] = []
    errors: list[float] = []
    stop = "size_reached"
    for _ in range(steps):
        pool = [j for j in range(len(rows)) if allow_repeats or j not in chosen]
        if not pool:
            stop = "no_candidates_left"
            break
        step_goal = goal_policy(tuple(mean)) if goal_policy is not None else goal
        pick = greedy_linear_design(
            tuple(rows[j] for j in pool),
            tuple(variances[j] for j in pool),
            prior_covariance=tuple(tuple(r) for r in cov),
            budget=1,
            criterion=step_criterion,
            goal=step_goal if step_criterion == "goal" else None,
        )
        if not pick.indices:
            stop = pick.stop_reason
            break
        j = pool[pick.indices[0]]
        repeat = uses.get(j, 0)
        uses[j] = repeat + 1
        y = _dot(rows[j], x_true) + math.sqrt(variances[j]) * _noise_draw(
            stream, j, repeat
        )
        _kalman_update(mean, cov, rows[j], variances[j], y)
        chosen.append(j)
        ys.append(y)
        means.append(tuple(mean))
        traces.append(math.fsum(cov[i][i] for i in range(n)))
        errors.append(math.fsum((mean[i] - x_true[i]) ** 2 for i in range(n)))
    batch = greedy_linear_design(
        rows,
        variances,
        prior_covariance=sigma,
        budget=steps,
        criterion="a" if adaptive else criterion,
        goal=None if adaptive else goal,
        allow_repeats=allow_repeats,
    )
    b_mean = list(m0)
    b_cov = [list(row) for row in sigma]
    b_uses: dict[int, int] = {}
    for j in batch.indices:
        repeat = b_uses.get(j, 0)
        b_uses[j] = repeat + 1
        y = _dot(rows[j], x_true) + math.sqrt(variances[j]) * _noise_draw(
            stream, j, repeat
        )
        _kalman_update(b_mean, b_cov, rows[j], variances[j], y)
    return SequentialDesignResult(
        indices=tuple(chosen),
        observations=tuple(ys),
        posterior_means=tuple(means),
        trace_history=tuple(traces),
        initial_trace=initial_trace,
        final_mean=tuple(mean),
        final_covariance=tuple(tuple(r) for r in cov),
        truth=x_true,
        squared_errors=tuple(errors),
        batch_indices=batch.indices,
        batch_trace=math.fsum(b_cov[i][i] for i in range(n)),
        batch_squared_error=math.fsum((b_mean[i] - x_true[i]) ** 2 for i in range(n)),
        criterion=step_criterion,
        adaptive=adaptive,
        stop_reason=stop,
    )


# ---------------------------------------------------------------------------
# Robust design
# ---------------------------------------------------------------------------

_ROBUST_CRITERIA = ("a", "d")
_AGGREGATES = ("mean", "max")


@dataclass(frozen=True, slots=True)
class DesignScenario:
    """One scenario of a robust design problem.

    Attributes:
        candidate_rows: ``N`` by ``n`` observation rows under this scenario
            (for a nonlinear model, the Jacobian at a parameter sample).
        noise_variances: Positive noise variance per candidate.
        prior_covariance: SPD ``n`` by ``n`` prior covariance.
        weight: Non-negative weight in the average (ignored by ``"max"``).
        label: Free-text name used in reports.

    References:
        [EQ-DES-OPX-005], [REF-CHALONER-VERDINELLI-1995].

    """

    candidate_rows: Matrix
    noise_variances: Vector
    prior_covariance: Matrix
    weight: float = 1.0
    label: str = ""


def jacobian_scenarios(
    jacobian: Callable[[Vector], Sequence[Sequence[float]]],
    theta_samples: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    prior_covariance: Sequence[Sequence[float]],
) -> tuple[DesignScenario, ...]:
    """Build one local-design scenario per parameter sample.

    The observation model is nonlinear in an uncertain parameter ``theta``;
    locally the design problem is linear with rows ``J(theta)``, so each
    sample ``theta_s`` gives the scenario ``(J(theta_s), R, P)``. Averaging
    the local criterion over the samples is the Bayesian-average (expected)
    local design (EQ-DES-OPX-005).

    Args:
        jacobian: ``theta -> N by n`` matrix of observation rows.
        theta_samples: Parameter samples (equal weights).
        noise_variances: Positive noise variance per candidate.
        prior_covariance: SPD prior covariance of the linearized unknowns.

    Returns:
        One :class:`DesignScenario` per sample, labelled ``theta[s]``.

    Raises:
        TypeError: If ``jacobian`` is not callable.
        ValueError: For no samples or inconsistent Jacobian shapes.

    Equation:
        [EQ-DES-OPX-005].

    References:
        [REF-CHALONER-VERDINELLI-1995].

    """
    if not callable(jacobian):
        raise TypeError("jacobian must be callable")
    if not theta_samples:
        raise ValueError("theta_samples must be non-empty")
    out: list[DesignScenario] = []
    for s, theta in enumerate(theta_samples):
        rows, variances = _check_candidates(
            jacobian(as_vector(theta, name="theta")), noise_variances
        )
        assert variances is not None
        sigma = _check_prior(prior_covariance, len(rows[0]))
        out.append(DesignScenario(rows, variances, sigma, 1.0, f"theta[{s}]"))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class RobustDesignResult:
    """A design chosen against a set of scenarios.

    Attributes:
        indices: Selected candidate indices (greedy order, or ascending for the
            exhaustive method).
        value: Aggregate objective of the design (mean or max over scenarios
            of the scenario value; ``"a"``: trace, ``"d"``: ``log det``).
        scenario_values: The criterion of the design in every scenario.
        worst_scenario: Index of the worst scenario (largest trace, or smallest
            ``log det``).
        criterion: ``"a"`` or ``"d"``.
        aggregate: ``"mean"`` (Bayesian average) or ``"max"`` (worst case).
        method: ``"greedy"`` or ``"exhaustive"``.
        combinations: Subsets evaluated (``0`` for greedy).

    References:
        [EQ-DES-OPX-005], [EQ-DES-OPX-006], [REF-CHALONER-VERDINELLI-1995].

    """

    indices: tuple[int, ...]
    value: float
    scenario_values: Vector
    worst_scenario: int
    criterion: str
    aggregate: str
    method: str
    combinations: int


def _check_scenarios(
    scenarios: Sequence[DesignScenario],
) -> tuple[tuple[DesignScenario, ...], Vector, int]:
    """Validate scenarios and return normalised weights and ``N``.

    Raises:
        TypeError: If an item is not a :class:`DesignScenario`.
        ValueError: If shapes disagree, weights are invalid or all zero.

    References:
        [EQ-DES-OPX-005]. No equation is evaluated.

    """
    items = tuple(scenarios)
    if not items:
        raise ValueError("scenarios must be non-empty")
    checked: list[DesignScenario] = []
    for sc in items:
        if not isinstance(sc, DesignScenario):
            raise TypeError("scenarios must be DesignScenario instances")
        rows, variances = _check_candidates(sc.candidate_rows, sc.noise_variances)
        assert variances is not None
        sigma = _check_prior(sc.prior_covariance, len(rows[0]))
        if not math.isfinite(sc.weight) or sc.weight < 0.0:
            raise ValueError("scenario weights must be finite and non-negative")
        checked.append(DesignScenario(rows, variances, sigma, sc.weight, sc.label))
    first = checked[0]
    for sc in checked[1:]:
        if len(sc.candidate_rows) != len(first.candidate_rows) or len(
            sc.candidate_rows[0]
        ) != len(first.candidate_rows[0]):
            raise ValueError("all scenarios must share the candidate set shape")
    total = math.fsum(sc.weight for sc in checked)
    if total <= 0.0:
        raise ValueError("scenario weights must not all be zero")
    return (
        tuple(checked),
        tuple(sc.weight / total for sc in checked),
        len(first.candidate_rows),
    )


def _loss(criterion: str, scenario_value: float) -> float:
    """Return the quantity to minimise for a scenario value.

    References:
        [EQ-DES-OPX-005]. No equation is evaluated.

    """
    return scenario_value if criterion == "a" else -scenario_value


def _aggregate(
    aggregate: str, weights: Vector, criterion: str, values: Sequence[float]
) -> float:
    """Aggregate scenario values: weighted mean or the worst scenario.

    References:
        [EQ-DES-OPX-005], [EQ-DES-OPX-006].

    """
    if aggregate == "mean":
        return math.fsum(w * v for w, v in zip(weights, values, strict=True))
    losses = [_loss(criterion, v) for v in values]
    worst = max(range(len(losses)), key=lambda s: (losses[s], -s))
    return values[worst]


def robust_linear_design(
    scenarios: Sequence[DesignScenario],
    *,
    size: int,
    criterion: str = "a",
    aggregate: str = "mean",
    method: str = "greedy",
    max_combinations: int = 20000,
) -> RobustDesignResult:
    """Choose ``size`` sensors against a finite set of scenarios.

    Scenario criteria are the A trace ``tr(C_s)`` (minimise) or the D value
    ``log det(I + H_s)`` (maximise). ``aggregate="mean"`` optimises the
    weighted average over scenarios (Bayesian-average design,
    EQ-DES-OPX-005); ``aggregate="max"`` optimises the worst scenario: the
    largest trace, or the smallest ``log det`` (EQ-DES-OPX-006). The greedy
    method adds the sensor with the best aggregate after a rank-one update of
    every scenario covariance (ties to the lowest index, no repeats);
    ``method="exhaustive"`` scores every subset by
    :func:`linear_design_criteria` and is the oracle on small problems. Greedy
    is not guaranteed optimal, and the worst-case objective is not submodular
    in general.

    Args:
        scenarios: Scenarios sharing the candidate set shape.
        size: Number of sensors, ``1 <= size <= N``.
        criterion: ``"a"`` or ``"d"``.
        aggregate: ``"mean"`` or ``"max"``.
        method: ``"greedy"`` or ``"exhaustive"``.
        max_combinations: Cap on ``C(N, size)`` for the exhaustive method.

    Returns:
        :class:`RobustDesignResult`.

    Raises:
        TypeError: If a scenario has the wrong type.
        ValueError: For unknown options, invalid scenarios, a size outside
            ``1..N`` or too many combinations.

    Equation:
        [EQ-DES-OPX-005], [EQ-DES-OPX-006].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-ALEXANDERIAN-2021].

    """
    if criterion not in _ROBUST_CRITERIA:
        raise ValueError(f"criterion must be one of {_ROBUST_CRITERIA}")
    if aggregate not in _AGGREGATES:
        raise ValueError(f"aggregate must be one of {_AGGREGATES}")
    if method not in ("greedy", "exhaustive"):
        raise ValueError("method must be 'greedy' or 'exhaustive'")
    items, weights, count = _check_scenarios(scenarios)
    k = require_non_negative_integer(size, name="size")
    if not 1 <= k <= count:
        raise ValueError("size must satisfy 1 <= size <= number of candidates")

    def values_of(subset: tuple[int, ...]) -> Vector:
        """Return the criterion of ``subset`` in every scenario (dense).

        References:
            [EQ-DES-LIN-001], [EQ-DES-OPX-005].

        """
        out = []
        for sc in items:
            crit = linear_design_criteria(
                sc.candidate_rows,
                sc.noise_variances,
                subset,
                prior_covariance=sc.prior_covariance,
            )
            out.append(crit.a_optimal if criterion == "a" else crit.d_optimal)
        return tuple(out)

    def score(values: Sequence[float]) -> float:
        """Return the minimisation key of an aggregate.

        References:
            [EQ-DES-OPX-005], [EQ-DES-OPX-006].

        """
        return _loss(criterion, _aggregate(aggregate, weights, criterion, values))

    combos = 0
    if method == "exhaustive":
        total = math.comb(count, k)
        if total > require_non_negative_integer(
            max_combinations, name="max_combinations"
        ):
            raise ValueError(f"{total} combinations exceed max_combinations")
        best_subset: tuple[int, ...] = ()
        best_key = math.inf
        for subset in itertools.combinations(range(count), k):
            key = score(values_of(subset))
            if key < best_key:
                best_subset, best_key = subset, key
        combos = total
        chosen: tuple[int, ...] = best_subset
    else:
        covs = [[list(r) for r in sc.prior_covariance] for sc in items]
        current: list[float] = [
            math.fsum(c[i][i] for i in range(len(c))) if criterion == "a" else 0.0
            for c in covs
        ]
        picked: list[int] = []
        for _ in range(k):
            best_j = -1
            best_key = math.inf
            for j in range(count):
                if j in picked:
                    continue
                trial: list[float] = []
                for s, sc in enumerate(items):
                    a = sc.candidate_rows[j]
                    v = [_dot(crow, a) for crow in covs[s]]
                    quad = _dot(a, v)
                    var = sc.noise_variances[j]
                    if criterion == "a":
                        trial.append(current[s] - _dot(v, v) / (var + quad))
                    else:
                        trial.append(current[s] + math.log1p(quad / var))
                key = score(trial)
                if key < best_key:
                    best_j, best_key = j, key
            for s, sc in enumerate(items):
                a = sc.candidate_rows[best_j]
                v = [_dot(crow, a) for crow in covs[s]]
                quad = _dot(a, v)
                var = sc.noise_variances[best_j]
                denom = var + quad
                if criterion == "a":
                    current[s] -= _dot(v, v) / denom
                else:
                    current[s] += math.log1p(quad / var)
                for i, vi in enumerate(v):
                    for t, vt in enumerate(v):
                        covs[s][i][t] -= vi * vt / denom
            picked.append(best_j)
        chosen = tuple(picked)
    final = values_of(chosen)
    losses = [_loss(criterion, v) for v in final]
    worst = max(range(len(losses)), key=lambda s: (losses[s], -s))
    return RobustDesignResult(
        indices=chosen,
        value=_aggregate(aggregate, weights, criterion, final),
        scenario_values=final,
        worst_scenario=worst,
        criterion=criterion,
        aggregate=aggregate,
        method=method,
        combinations=combos,
    )
